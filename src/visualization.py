from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .schemas import Frame, MetricSnapshot, Prediction, VizConfig


class StreamVisualizer:
    def __init__(self, cfg: VizConfig) -> None:
        if cfg.mode not in {"file", "window", "none"}:
            raise ValueError(
                f"viz mode must be 'file' | 'window' | 'none', got {cfg.mode!r}"
            )
        self.cfg = cfg
        self._out_dir = Path("outputs")
        self._window_open = False
        self._counter = 0
        if self.cfg.mode == "file":
            self._out_dir.mkdir(parents=True, exist_ok=True)

    def render(
        self, frame: Frame, pred: Prediction, snapshot: MetricSnapshot
    ) -> None:
        if self.cfg.mode == "none":
            return
        self._counter += 1
        if self._counter % max(1, self.cfg.every_n_frames) != 0:
            return

        composite = _compose(frame, pred, snapshot, self.cfg.overlay_alpha)

        if self.cfg.mode == "file":
            out = self._out_dir / f"frame_{frame.index:06d}.png"
            Image.fromarray(composite).save(out)
        else:  # window
            try:
                import cv2  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "viz mode 'window' requires opencv-python"
                ) from e
            cv2.imshow("stream", composite[:, :, ::-1])  # RGB→BGR
            cv2.waitKey(1)
            self._window_open = True

    def close(self) -> None:
        if self._window_open:
            try:
                import cv2  # type: ignore

                cv2.destroyAllWindows()
            except ImportError:
                pass


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
    out[..., 0] = norm * 255.0  # R
    out[..., 1] = (norm ** 2) * 80.0  # mild G
    return out
