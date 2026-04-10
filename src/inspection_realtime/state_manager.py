from __future__ import annotations

from .contracts import RuntimeState


class RuntimeStateManager:
    def __init__(
        self,
        min_calibration_frames: int,
        baseline_sample_count: int,
        require_baseline: bool = True,
    ) -> None:
        self.min_calibration_frames = max(1, int(min_calibration_frames))
        self.baseline_sample_count = int(baseline_sample_count)
        self.require_baseline = bool(require_baseline)

        self.state = RuntimeState.CALIBRATION
        self.calibration_frames = 0
        self.object_index = 0
        self.object_change_count = 0
        self._online_score_sum = 0.0

    def reset_for_new_object(self) -> None:
        self.object_index += 1
        self.object_change_count += 1
        self.state = RuntimeState.CALIBRATION
        self.calibration_frames = 0
        self._online_score_sum = 0.0

    def observe_calibration_score(self, score: float) -> RuntimeState:
        if self.state != RuntimeState.CALIBRATION:
            return self.state

        self.calibration_frames += 1
        self._online_score_sum += float(score)

        has_required_baseline = (self.baseline_sample_count > 0) or (not self.require_baseline)
        if self.calibration_frames >= self.min_calibration_frames and has_required_baseline:
            self.state = RuntimeState.PRODUCTION
        return self.state

    def calibration_score_mean(self) -> float:
        if self.calibration_frames == 0:
            return 0.0
        return self._online_score_sum / float(self.calibration_frames)
