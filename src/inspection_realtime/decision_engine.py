from __future__ import annotations

from .contracts import DecisionRecord, FramePacket, ModelPrediction


class DecisionEngine:
    def decide(self, frame: FramePacket, prediction: ModelPrediction) -> DecisionRecord:
        return DecisionRecord(
            model=prediction.model_name,
            path=str(frame.path),
            label=frame.label,
            defect_type=frame.defect_type,
            score=float(prediction.score),
            pred_is_anomaly=int(prediction.pred_is_anomaly),
            heatmap_path=None,
        )
