from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import cv2
import numpy as np

from .contracts import DecisionRecord, FramePacket, ModelPrediction


class RuntimeOutputWriter:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.heatmaps_dir = self.session_dir / "heatmaps"
        self.heatmaps_dir.mkdir(exist_ok=True)

        self.predictions_path = self.session_dir / "predictions.json"
        self.status_path = self.session_dir / "live_status.json"
        self._rows: list[dict[str, Any]] = []
        self._lock = Lock()
        self.predictions_path.write_text("[]", encoding="utf-8")

    def append_decision(self, decision: DecisionRecord) -> None:
        with self._lock:
            self._rows.append(decision.to_dict())
            self.predictions_path.write_text(
                json.dumps(self._rows, indent=2),
                encoding="utf-8",
            )

    def save_heatmap(
        self,
        frame: FramePacket,
        prediction: ModelPrediction,
    ) -> Optional[str]:
        if prediction.pred_is_anomaly != 1 or prediction.heatmap is None:
            return None

        heatmap = np.asarray(prediction.heatmap, dtype=np.float32)
        if heatmap.ndim != 2:
            return None

        resized = cv2.resize(
            heatmap,
            (frame.raw_image_bgr.shape[1], frame.raw_image_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        clipped = np.clip(resized, 0.0, 1.0)
        colored = cv2.applyColorMap((clipped * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame.raw_image_bgr, 0.55, colored, 0.45, 0.0)

        relative_path = Path("heatmaps") / f"frame_{frame.frame_id:06d}.png"
        output_path = self.session_dir / relative_path
        cv2.imwrite(str(output_path), overlay)
        return relative_path.as_posix()

    def write_status(self, status: dict[str, Any]) -> None:
        with self._lock:
            self.status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    @staticmethod
    def guess_content_type(path: Path) -> str:
        return mimetypes.guess_type(str(path))[0] or "application/octet-stream"
