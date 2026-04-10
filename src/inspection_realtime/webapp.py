from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Callable
from urllib.parse import unquote, urlparse

from .report_generator import RuntimeOutputWriter


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Inspection Realtime</title>
  <style>
    :root {
      --bg: #f3f4ef;
      --card: #fffef9;
      --ink: #1d2b28;
      --muted: #6a7772;
      --accent: #0f766e;
      --line: #d7ddd8;
    }
    body {
      margin: 0;
      font-family: "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #eef3ef 0%, #f8f8f2 100%);
      color: var(--ink);
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 32px;
    }
    .subtitle {
      color: var(--muted);
      margin-bottom: 24px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 8px 24px rgba(20, 33, 28, 0.05);
    }
    .label {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 8px;
    }
    .value {
      font-size: 28px;
      font-weight: 700;
    }
    .value.state-CALIBRATION { color: #a16207; }
    .value.state-PRODUCTION { color: var(--accent); }
    .section {
      display: grid;
      grid-template-columns: 1.3fr 1fr;
      gap: 16px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-weight: 600;
    }
    .mono {
      font-family: Consolas, monospace;
      font-size: 12px;
      word-break: break-word;
    }
    img {
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #f2f4f1;
    }
    @media (max-width: 960px) {
      .section {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main>
    <h1>Inspection Realtime</h1>
    <div class="subtitle">Live monitoring for calibration and production states.</div>
    <div class="grid" id="cards"></div>
    <div class="section">
      <div class="card">
        <div class="label">Recent Decisions</div>
        <table>
          <thead>
            <tr>
              <th>Score</th>
              <th>Decision</th>
              <th>Path</th>
            </tr>
          </thead>
          <tbody id="recent-decisions"></tbody>
        </table>
      </div>
      <div class="card">
        <div class="label">Latest Fail Heatmap</div>
        <div id="heatmap-container" class="value small">No fail heatmap available.</div>
      </div>
    </div>
  </main>
  <script>
    const cardOrder = [
      ["state", "State"],
      ["active_model", "Active Model"],
      ["frames_seen", "Frames Seen"],
      ["decisions_emitted", "Decisions"],
      ["fail_count", "Fail Count"],
      ["no_decision_count", "No Decision"],
      ["object_change_count", "Object Changes"],
      ["mean_latency_ms", "Mean Latency (ms)"],
      ["p95_latency_ms", "P95 Latency (ms)"],
      ["processed_fps", "Processed FPS"],
      ["classifier_confidence", "Classifier Confidence"],
      ["threshold", "Threshold"]
    ];

    function formatValue(key, value) {
      if (typeof value === "number") {
        return value.toFixed(2);
      }
      return value ?? "-";
    }

    function fileUrl(relativePath) {
      return "/session/" + relativePath.replaceAll("\\\\", "/");
    }

    function renderCards(data) {
      const root = document.getElementById("cards");
      root.innerHTML = "";
      for (const [key, label] of cardOrder) {
        const card = document.createElement("div");
        const valueClass = key === "state" ? `value state-${data[key]}` : "value";
        card.className = "card";
        card.innerHTML = `
          <div class="label">${label}</div>
          <div class="${valueClass}">${formatValue(key, data[key])}</div>
        `;
        root.appendChild(card);
      }
    }

    function renderDecisions(data) {
      const body = document.getElementById("recent-decisions");
      body.innerHTML = "";
      for (const row of data.recent_decisions || []) {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${Number(row.score || 0).toFixed(4)}</td>
          <td>${row.pred_is_anomaly === 1 ? "FAIL" : "PASS"}</td>
          <td class="mono">${row.path || ""}</td>
        `;
        body.appendChild(tr);
      }
    }

    function renderHeatmap(data) {
      const container = document.getElementById("heatmap-container");
      const recentFail = (data.recent_fails || [])[0];
      if (!recentFail || !recentFail.heatmap_path) {
        container.textContent = "No fail heatmap available.";
        return;
      }
      container.innerHTML = `<img src="${fileUrl(recentFail.heatmap_path)}" alt="Latest fail heatmap">`;
    }

    async function refresh() {
      const response = await fetch("/api/status", { cache: "no-store" });
      const data = await response.json();
      renderCards(data);
      renderDecisions(data);
      renderHeatmap(data);
    }

    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


class _RuntimeDashboardHandler(BaseHTTPRequestHandler):
    server_version = "InspectionRuntime/0.1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._write_bytes(HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/status":
            payload = json.dumps(self.server.status_provider(), indent=2).encode("utf-8")
            self._write_bytes(payload, "application/json; charset=utf-8")
            return
        if parsed.path.startswith("/session/"):
            relative = unquote(parsed.path[len("/session/"):])
            self._serve_session_file(relative)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _serve_session_file(self, relative_path: str) -> None:
        session_dir = self.server.session_dir.resolve()
        target = (session_dir / relative_path).resolve()
        try:
            target.relative_to(session_dir)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN, "Invalid path")
            return
        if not target.exists() or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        content = target.read_bytes()
        self._write_bytes(content, RuntimeOutputWriter.guess_content_type(target))

    def _write_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class _RuntimeDashboardServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        session_dir: Path,
        status_provider: Callable[[], dict],
    ) -> None:
        super().__init__(address, _RuntimeDashboardHandler)
        self.session_dir = Path(session_dir)
        self.status_provider = status_provider


class LiveDashboardServer:
    def __init__(
        self,
        host: str,
        port: int,
        session_dir: Path,
        status_provider: Callable[[], dict],
    ) -> None:
        self.host = host
        self.port = int(port)
        self._server = _RuntimeDashboardServer(
            (self.host, self.port),
            session_dir=Path(session_dir),
            status_provider=status_provider,
        )
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
