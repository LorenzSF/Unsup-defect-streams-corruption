"""Per-frame renderer plus optional live dashboard.

`StreamVisualizer` keeps producing the same headless-friendly file/window
output as before (controlled by `cfg.mode`). When `cfg.dashboard_enabled`
is true it also runs a FastAPI + WebSocket server in a daemon thread that
pushes live metrics, the heatmap-overlaid frame, and the per-frame
embedding 2D projection to a browser dashboard. Single file by design
(BEST_PRACTICES rules 6, 7).
"""
from __future__ import annotations

import asyncio
import base64
import io
import math
import threading
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .schemas import Frame, MetricSnapshot, Prediction, VizConfig

# FastAPI / uvicorn are imported at module level so type annotations on the
# WebSocket route resolve under `from __future__ import annotations`. When the
# dashboard is disabled in config, the imports are still cheap and harmless;
# `_DashboardServer` only fires up when `cfg.dashboard_enabled is True`.
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    import uvicorn
    _FASTAPI_IMPORT_ERROR: Optional[BaseException] = None
except ModuleNotFoundError as _exc:
    FastAPI = WebSocket = WebSocketDisconnect = HTMLResponse = uvicorn = None  # type: ignore[assignment]
    _FASTAPI_IMPORT_ERROR = _exc


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Streaming Defect Detection</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root {
  --bg: #0b0f15;
  --panel: #131922;
  --ink: #e6edf3;
  --muted: #8b95a4;
  --line: #232c39;
  --good: #10b981;
  --bad: #ef4444;
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--ink);
  font-size: 14px;
}
header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 18px 24px 14px;
  border-bottom: 1px solid var(--line);
}
header h1 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  letter-spacing: 0.01em;
}
header .subtitle { color: var(--muted); font-size: 12px; }
.dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-right: 6px;
  vertical-align: 1px;
}
.dot.live { background: var(--good); box-shadow: 0 0 8px rgba(16,185,129,0.6); }
.dot.dead { background: var(--bad); }
main {
  padding: 16px 24px 24px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.metrics {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 12px;
}
.tile {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 14px 16px;
}
.tile .label {
  color: var(--muted);
  font-size: 10.5px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 600;
}
.tile .value {
  font-size: 24px;
  font-weight: 700;
  margin-top: 6px;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.01em;
}
.tile .unit { font-size: 13px; color: var(--muted); font-weight: 500; margin-left: 2px; }
.panels { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 16px;
  min-height: 520px;
  display: flex;
  flex-direction: column;
}
.panel-title {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 12px;
}
.panel-title h2 {
  margin: 0;
  font-size: 12px;
  font-weight: 600;
  color: var(--muted);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.panel-title .meta { font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums; }
.frame-shell {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #050709;
  border-radius: 8px;
  overflow: hidden;
}
.frame-shell img { display: block; max-width: 100%; max-height: 100%; object-fit: contain; }
#embedding-plot { flex: 1; min-height: 460px; }
.empty {
  color: var(--muted);
  font-size: 13px;
  padding: 24px;
  text-align: center;
  line-height: 1.6;
}
@media (max-width: 1200px) {
  .metrics { grid-template-columns: repeat(3, 1fr); }
  .panels { grid-template-columns: 1fr; }
}
@media (max-width: 640px) {
  .metrics { grid-template-columns: repeat(2, 1fr); }
}
</style>
</head>
<body>
<header>
  <div>
    <h1>Streaming Defect Detection</h1>
    <div class="subtitle">Live inference monitoring</div>
  </div>
  <div class="subtitle" id="conn-status"><span class="dot dead"></span>connecting...</div>
</header>
<main>
  <section class="metrics">
    <div class="tile"><div class="label">FPS</div><div class="value" id="m-fps">&mdash;</div></div>
    <div class="tile"><div class="label">Latency p95</div><div class="value" id="m-lat">&mdash;</div></div>
    <div class="tile"><div class="label">AUROC</div><div class="value" id="m-auroc">&mdash;</div></div>
    <div class="tile"><div class="label">F1</div><div class="value" id="m-f1">&mdash;</div></div>
    <div class="tile"><div class="label">Anomaly rate</div><div class="value" id="m-anom">&mdash;</div></div>
    <div class="tile"><div class="label">Frames</div><div class="value" id="m-frames">&mdash;</div></div>
  </section>
  <section class="panels">
    <div class="panel">
      <div class="panel-title">
        <h2>Current frame &middot; heatmap</h2>
        <div class="meta" id="frame-meta"></div>
      </div>
      <div class="frame-shell"><img id="frame-img" alt=""></div>
    </div>
    <div class="panel">
      <div class="panel-title">
        <h2>Embedding &middot; PCA(2) projection</h2>
        <div class="meta" id="embedding-meta">waiting for data...</div>
      </div>
      <div id="embedding-plot"></div>
    </div>
  </section>
</main>
<script>
const COLOR_REF = '#5b6573';
const COLOR_LATEST_BORDER = '#fef3c7';
const PLOT_LAYOUT = {
  paper_bgcolor: '#131922',
  plot_bgcolor: '#0b0f15',
  font: { color: '#e6edf3', family: 'Segoe UI, sans-serif', size: 11 },
  margin: { l: 36, r: 16, t: 8, b: 32 },
  xaxis: { gridcolor: '#232c39', zerolinecolor: '#232c39' },
  yaxis: { gridcolor: '#232c39', zerolinecolor: '#232c39' },
  showlegend: false,
  hovermode: 'closest',
};
const PLOT_CONFIG = { displayModeBar: false, responsive: true };

let bootstrapped = false;
let embeddingEnabled = false;
let liveHistory = [];
let maxLive = 200;
let threshold = 1.0;

function fmtNum(v, digits) {
  digits = digits == null ? 2 : digits;
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Number(v).toFixed(digits);
}
function fmtInt(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Math.round(Number(v)).toLocaleString();
}
function fmtPct(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return (Number(v) * 100).toFixed(1) + '%';
}

function pointColor(scoreRatio) {
  const r = Math.max(0, Math.min(1, Number(scoreRatio) || 0));
  // green -> amber -> red
  if (r < 0.5) {
    const t = r / 0.5;
    return 'rgb(' + Math.round(16 + t * (245 - 16)) + ','
      + Math.round(185 - t * (185 - 158)) + ','
      + Math.round(129 - t * (129 - 11)) + ')';
  } else {
    const t = (r - 0.5) / 0.5;
    return 'rgb(' + Math.round(245 + t * (239 - 245)) + ','
      + Math.round(158 - t * (158 - 68)) + ','
      + Math.round(11 + t * (68 - 11)) + ')';
  }
}

function setConn(connected) {
  const el = document.getElementById('conn-status');
  el.innerHTML = connected
    ? '<span class="dot live"></span>live'
    : '<span class="dot dead"></span>disconnected &mdash; retrying...';
}

function initEmbeddingPlot(refX, refY, axis) {
  const xRange = [axis[0], axis[1]];
  const yRange = [axis[2], axis[3]];
  const traces = [
    {
      x: refX, y: refY,
      mode: 'markers', type: 'scattergl',
      marker: { size: 4, color: COLOR_REF, opacity: 0.5 },
      hoverinfo: 'skip',
    },
    {
      x: [], y: [],
      mode: 'markers', type: 'scattergl',
      marker: { size: 8, color: [], opacity: 0.85, line: { width: 0.5, color: '#0b0f15' } },
      hovertemplate: 'score %{customdata:.4f}<extra></extra>',
      customdata: [],
    },
    {
      x: [], y: [],
      mode: 'markers', type: 'scattergl',
      marker: { size: 14, color: '#ef4444', opacity: 1,
                line: { width: 2, color: COLOR_LATEST_BORDER } },
      hoverinfo: 'skip',
    },
  ];
  const layout = Object.assign({}, PLOT_LAYOUT, {
    xaxis: Object.assign({}, PLOT_LAYOUT.xaxis, { range: xRange }),
    yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, { range: yRange }),
  });
  Plotly.newPlot('embedding-plot', traces, layout, PLOT_CONFIG);
  document.getElementById('embedding-meta').textContent =
    refX.length + ' warmup pts · 0 live';
}

function showEmbeddingDisabled() {
  document.getElementById('embedding-plot').innerHTML =
    '<div class="empty">Embedding unavailable for this run.<br>'
    + '(detector emitted no warmup embeddings or warmup_embeddings &lt; 2)</div>';
  document.getElementById('embedding-meta').textContent = 'disabled';
}

function applyEmbedding() {
  if (!embeddingEnabled || liveHistory.length === 0) return;
  const x = liveHistory.map(function (p) { return p.x; });
  const y = liveHistory.map(function (p) { return p.y; });
  const colors = liveHistory.map(function (p) { return pointColor(p.score_ratio); });
  const scores = liveHistory.map(function (p) { return p.score; });
  Plotly.restyle('embedding-plot', {
    x: [x], y: [y],
    'marker.color': [colors],
    customdata: [scores],
  }, [1]);
  const last = liveHistory[liveHistory.length - 1];
  Plotly.restyle('embedding-plot', {
    x: [[last.x]], y: [[last.y]],
    'marker.color': [[pointColor(last.score_ratio)]],
  }, [2]);
  document.getElementById('embedding-meta').textContent =
    liveHistory.length + ' live pts · latest score ' + fmtNum(last.score, 4);
}

function applyMetrics(m) {
  document.getElementById('m-fps').textContent = fmtNum(m.fps, 1);
  document.getElementById('m-lat').innerHTML =
    fmtNum(m.p95_latency_ms, 1) + '<span class="unit"> ms</span>';
  document.getElementById('m-auroc').textContent = fmtNum(m.auroc, 3);
  document.getElementById('m-f1').textContent = fmtNum(m.f1, 3);
  document.getElementById('m-anom').textContent = fmtPct(m.anomaly_rate);
  document.getElementById('m-frames').textContent = fmtInt(m.frames_seen);
}

function applyFrame(b64, score) {
  if (b64) {
    document.getElementById('frame-img').src = 'data:image/png;base64,' + b64;
  }
  document.getElementById('frame-meta').textContent =
    (score !== undefined && score !== null)
      ? 'score ' + fmtNum(score, 4) + ' · threshold ' + fmtNum(threshold, 4)
      : '';
}

function onBootstrap(data) {
  threshold = Number(data.threshold) || 1.0;
  maxLive = Number(data.max_live_points) || 200;
  embeddingEnabled = !!data.embedding_available;
  if (embeddingEnabled) {
    initEmbeddingPlot(data.reference_cloud_x || [],
                      data.reference_cloud_y || [],
                      data.axis_bounds || [-1, 1, -1, 1]);
    liveHistory = (data.live_history || []).slice(-maxLive);
    if (liveHistory.length > 0) applyEmbedding();
  } else {
    showEmbeddingDisabled();
  }
  applyMetrics(data.metrics || {});
  applyFrame(data.frame_b64, data.score);
  bootstrapped = true;
}

function onLive(data) {
  applyMetrics(data.metrics || {});
  applyFrame(data.frame_b64, data.score);
  if (data.new_point) {
    liveHistory.push(data.new_point);
    if (liveHistory.length > maxLive) liveHistory.shift();
    applyEmbedding();
  }
}

let ws = null;
let reconnectTimer = null;
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const url = proto + '://' + location.host + '/ws';
  ws = new WebSocket(url);
  ws.onopen = function () { setConn(true); };
  ws.onclose = function () {
    setConn(false);
    bootstrapped = false;
    if (reconnectTimer) clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(connect, 1500);
  };
  ws.onerror = function () { /* onclose will fire */ };
  ws.onmessage = function (event) {
    let data;
    try { data = JSON.parse(event.data); } catch (e) { return; }
    if (data.type === 'bootstrap') onBootstrap(data);
    else if (data.type === 'live' && bootstrapped) onLive(data);
  };
}
connect();
</script>
</body>
</html>
"""


class _DashboardState:
    """Thread-safe holder for the dashboard's bootstrap + live state.

    `set_reference_cloud` is called once after warmup; `update_live` is
    called from the main thread on every render(); the WebSocket handler
    reads through `bootstrap_payload` / `live_payload`.
    """

    def __init__(self, threshold: float, max_live_points: int) -> None:
        self._lock = threading.Lock()
        self._threshold = float(threshold)
        self._max_live_points = int(max_live_points)
        self._reference_x: List[float] = []
        self._reference_y: List[float] = []
        self._axis_bounds: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0)
        self._embedding_available = False
        self._frame_b64: str = ""
        self._score: float = 0.0
        self._fps: float = 0.0
        self._p95_lat: float = 0.0
        self._auroc: Optional[float] = None
        self._f1: Optional[float] = None
        self._anomaly_rate: float = 0.0
        self._frames_seen: int = 0
        self._live_history: List[dict] = []

    def set_reference_cloud(self, ref_2d: np.ndarray) -> None:
        if ref_2d.size == 0:
            return
        with self._lock:
            self._reference_x = [float(v) for v in ref_2d[:, 0]]
            self._reference_y = [float(v) for v in ref_2d[:, 1]]
            min_x, max_x = float(ref_2d[:, 0].min()), float(ref_2d[:, 0].max())
            min_y, max_y = float(ref_2d[:, 1].min()), float(ref_2d[:, 1].max())
            pad_x = max((max_x - min_x) * 0.15, 0.01)
            pad_y = max((max_y - min_y) * 0.15, 0.01)
            self._axis_bounds = (
                min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y,
            )
            self._embedding_available = True

    def update_live(
        self,
        frame_b64: str,
        score: float,
        snapshot: MetricSnapshot,
        embedding_2d: Optional[Tuple[float, float]],
    ) -> None:
        anom_rate = (
            (snapshot.n_anomalies / snapshot.n_seen) if snapshot.n_seen > 0 else 0.0
        )
        with self._lock:
            self._frame_b64 = frame_b64
            self._score = float(score)
            self._fps = float(snapshot.throughput_fps)
            self._p95_lat = float(snapshot.p95_latency_ms)
            self._auroc = None if math.isnan(snapshot.auroc) else float(snapshot.auroc)
            self._f1 = None if math.isnan(snapshot.f1) else float(snapshot.f1)
            self._anomaly_rate = float(anom_rate)
            self._frames_seen = int(snapshot.n_seen)
            if embedding_2d is not None:
                self._live_history.append({
                    "x": float(embedding_2d[0]),
                    "y": float(embedding_2d[1]),
                    "score": float(score),
                    "score_ratio": _score_ratio(score, self._threshold),
                })
                if len(self._live_history) > self._max_live_points:
                    del self._live_history[: len(self._live_history) - self._max_live_points]

    def _metrics_locked(self) -> dict:
        return {
            "fps": self._fps,
            "p95_latency_ms": self._p95_lat,
            "auroc": self._auroc,
            "f1": self._f1,
            "anomaly_rate": self._anomaly_rate,
            "frames_seen": self._frames_seen,
        }

    def bootstrap_payload(self) -> dict:
        with self._lock:
            return {
                "type": "bootstrap",
                "embedding_available": self._embedding_available,
                "reference_cloud_x": list(self._reference_x),
                "reference_cloud_y": list(self._reference_y),
                "axis_bounds": list(self._axis_bounds),
                "threshold": self._threshold,
                "max_live_points": self._max_live_points,
                "live_history": list(self._live_history),
                "metrics": self._metrics_locked(),
                "frame_b64": self._frame_b64,
                "score": self._score,
            }

    def live_payload(self) -> dict:
        with self._lock:
            new_point = self._live_history[-1] if self._live_history else None
            return {
                "type": "live",
                "metrics": self._metrics_locked(),
                "frame_b64": self._frame_b64,
                "score": self._score,
                "new_point": new_point,
            }


def _score_ratio(score: float, threshold: float) -> float:
    denom = max(float(threshold), 1e-9)
    return float(max(0.0, min(1.0, float(score) / denom)))


class _DashboardServer:
    """FastAPI + uvicorn + WebSocket broadcaster running in a daemon thread.

    `broadcast_live()` is callable from the main thread; it enqueues the
    current `live_payload` into every connected client's asyncio queue via
    `loop.call_soon_threadsafe`.
    """

    def __init__(self, host: str, port: int, state: _DashboardState) -> None:
        self._host = host
        self._port = int(port)
        self._state = state
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._clients: List[asyncio.Queue] = []
        self._clients_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._server = None
        self._app = self._build_app()

    def _build_app(self):
        if _FASTAPI_IMPORT_ERROR is not None:
            raise RuntimeError(
                "dashboard requires 'fastapi' and 'uvicorn'. Install from "
                "requirements.txt."
            ) from _FASTAPI_IMPORT_ERROR

        app = FastAPI()

        @app.get("/", response_class=HTMLResponse)
        async def index() -> str:
            return HTML_PAGE

        @app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket) -> None:
            try:
                await websocket.accept()
            except Exception as exc:
                print(f"[viz] WebSocket accept failed: {type(exc).__name__}: {exc}")
                return
            print("[viz] WebSocket client connected")
            queue: asyncio.Queue = asyncio.Queue(maxsize=8)
            with self._clients_lock:
                self._clients.append(queue)
            try:
                await websocket.send_json(self._state.bootstrap_payload())
                while True:
                    payload = await queue.get()
                    await websocket.send_json(payload)
            except WebSocketDisconnect:
                print("[viz] WebSocket client disconnected (clean)")
            except Exception as exc:
                print(
                    f"[viz] WebSocket loop error: {type(exc).__name__}: {exc}"
                )
            finally:
                with self._clients_lock:
                    if queue in self._clients:
                        self._clients.remove(queue)

        return app

    def broadcast_live(self) -> None:
        if self._loop is None:
            return
        payload = self._state.live_payload()
        with self._clients_lock:
            queues = list(self._clients)
        for q in queues:
            try:
                self._loop.call_soon_threadsafe(self._enqueue, q, payload)
            except RuntimeError:
                pass

    @staticmethod
    def _enqueue(queue: "asyncio.Queue", payload: dict) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        if _FASTAPI_IMPORT_ERROR is not None:
            raise RuntimeError(
                "dashboard requires 'uvicorn'. Install from requirements.txt."
            ) from _FASTAPI_IMPORT_ERROR
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        config = uvicorn.Config(
            app=self._app,
            host=self._host,
            port=self._port,
            log_level="info",
            loop="asyncio",
            ws="wsproto",
        )
        self._server = uvicorn.Server(config)
        try:
            self._loop.run_until_complete(self._server.serve())
        except Exception as exc:
            print(f"[viz] uvicorn server error: {type(exc).__name__}: {exc}")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class StreamVisualizer:
    def __init__(
        self,
        cfg: VizConfig,
        out_dir: Path,
        threshold: float,
        warmup_embeddings: Optional[np.ndarray] = None,
    ) -> None:
        if cfg.mode not in {"file", "window", "none"}:
            raise ValueError(
                f"viz mode must be 'file' | 'window' | 'none', got {cfg.mode!r}"
            )
        self.cfg = cfg
        self._out_dir = out_dir
        self._window_open = False
        self._counter = 0
        self._threshold = float(threshold)
        if self.cfg.mode == "file":
            self._out_dir.mkdir(parents=True, exist_ok=True)

        self._embedding_pca = None
        self._state: Optional[_DashboardState] = None
        self._server: Optional[_DashboardServer] = None

        if cfg.dashboard_enabled:
            self._init_dashboard(warmup_embeddings)

    def _init_dashboard(self, warmup_embeddings: Optional[np.ndarray]) -> None:
        self._state = _DashboardState(
            threshold=self._threshold,
            max_live_points=self.cfg.dashboard_max_live_points,
        )
        embedding_ok = False
        if warmup_embeddings is not None and warmup_embeddings.ndim == 2 and warmup_embeddings.shape[0] >= 2:
            try:
                from sklearn.decomposition import PCA

                pca = PCA(n_components=2)
                ref_2d = pca.fit_transform(warmup_embeddings)
                self._embedding_pca = pca
                self._state.set_reference_cloud(ref_2d)
                embedding_ok = True
            except Exception as exc:
                print(f"[viz] embedding PCA fit failed: {exc}")

        self._server = _DashboardServer(
            host=self.cfg.dashboard_host,
            port=self.cfg.dashboard_port,
            state=self._state,
        )
        self._server.start()
        print(
            f"[viz] dashboard at http://{self.cfg.dashboard_host}:{self.cfg.dashboard_port}"
            f" (embedding {'on' if embedding_ok else 'OFF'})"
        )

    def render(
        self, frame: Frame, pred: Prediction, snapshot: MetricSnapshot
    ) -> None:
        if self.cfg.mode == "none" and self._server is None:
            return
        self._counter += 1
        if self._counter % max(1, self.cfg.every_n_frames) != 0:
            return

        composite = _compose(frame, pred, snapshot, self.cfg.overlay_alpha)

        if self.cfg.mode == "file":
            out = self._out_dir / f"frame_{frame.index:06d}.png"
            Image.fromarray(composite).save(out)
        elif self.cfg.mode == "window":
            try:
                import cv2  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "viz mode 'window' requires opencv-python"
                ) from e
            cv2.imshow("stream", composite[:, :, ::-1])  # RGB->BGR
            cv2.waitKey(1)
            self._window_open = True

        if self._server is not None and self._state is not None:
            self._push_to_dashboard(composite, pred, snapshot)

    def _push_to_dashboard(
        self,
        composite: np.ndarray,
        pred: Prediction,
        snapshot: MetricSnapshot,
    ) -> None:
        # downsample for WebSocket bandwidth
        img = Image.fromarray(composite)
        max_side = 720
        if max(img.size) > max_side:
            scale = max_side / max(img.size)
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        frame_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        embedding_2d: Optional[Tuple[float, float]] = None
        if (
            self._embedding_pca is not None
            and pred.embedding is not None
            and pred.embedding.size > 0
        ):
            try:
                exp_dim = int(self._embedding_pca.n_features_in_)
                emb = np.asarray(pred.embedding, dtype=np.float32).reshape(-1)
                if emb.size == exp_dim:
                    proj = self._embedding_pca.transform(emb.reshape(1, -1))[0]
                    embedding_2d = (float(proj[0]), float(proj[1]))
            except Exception:
                embedding_2d = None

        self._state.update_live(
            frame_b64=frame_b64,
            score=float(pred.score),
            snapshot=snapshot,
            embedding_2d=embedding_2d,
        )
        self._server.broadcast_live()

    def close(self) -> None:
        if self._window_open:
            try:
                import cv2  # type: ignore

                cv2.destroyAllWindows()
            except ImportError:
                pass
        if self._server is not None:
            self._server.stop()


def _compose(
    frame: Frame, pred: Prediction, snap: MetricSnapshot, alpha: float
) -> np.ndarray:
    base = frame.image
    if base.ndim == 2:
        base = np.repeat(base[..., None], 3, axis=-1)

    overlaid = base.copy()
    if pred.anomaly_map is not None:
        heat = _heatmap(pred.anomaly_map, base.shape[:2])
        overlaid = (
            (1.0 - alpha) * base.astype(np.float32) + alpha * heat.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(overlaid)
    draw = ImageDraw.Draw(img)
    text = (
        f"idx={frame.index} score={pred.score:.3f} "
        f"auroc={snap.auroc:.3f} f1={snap.f1:.3f} "
        f"p95={snap.p95_latency_ms:.1f}ms fps={snap.throughput_fps:.1f}"
    )
    draw.rectangle((0, 0, img.width, 18), fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255))
    return np.array(img)


def _heatmap(amap: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Min-max normalize then map to a red-on-dark gradient."""
    h, w = target_hw
    if amap.shape != (h, w):
        ys = np.linspace(0, amap.shape[0] - 1, h).astype(int)
        xs = np.linspace(0, amap.shape[1] - 1, w).astype(int)
        amap = amap[np.ix_(ys, xs)]
    lo, hi = float(amap.min()), float(amap.max())
    if hi - lo < 1e-9:
        norm = np.zeros_like(amap, dtype=np.float32)
    else:
        norm = (amap - lo) / (hi - lo)
    out = np.zeros((h, w, 3), dtype=np.float32)
    out[..., 0] = norm * 255.0
    out[..., 1] = (norm ** 2) * 80.0
    return out
