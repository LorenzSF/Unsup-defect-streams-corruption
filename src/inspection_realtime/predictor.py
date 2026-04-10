from __future__ import annotations

import time

from benchmark_AD.models import BaseModel

from .contracts import FramePacket, ModelPrediction


class Predictor:
    def __init__(self, model_name: str, model: BaseModel) -> None:
        self.model_name = model_name
        self.model = model

    def predict(self, frame: FramePacket) -> tuple[ModelPrediction, float]:
        started = time.perf_counter()
        output = self.model.predict(frame.model_input)
        latency_ms = (time.perf_counter() - started) * 1000.0
        prediction = ModelPrediction(
            model_name=self.model_name,
            score=float(output.score),
            pred_is_anomaly=int(output.is_anomaly),
            heatmap=output.heatmap,
        )
        return prediction, latency_ms
