from __future__ import annotations

from .contracts import ObjectMatch


class TransitionDetector:
    def __init__(
        self,
        consecutive_different_frames: int = 3,
        min_confidence: float = 0.15,
    ) -> None:
        self.consecutive_different_frames = max(1, int(consecutive_different_frames))
        self.min_confidence = float(min_confidence)
        self._different_count = 0

    def reset(self) -> None:
        self._different_count = 0

    def update(self, match: ObjectMatch) -> bool:
        if match.is_same_reference:
            self._different_count = 0
            return False

        if match.confidence < self.min_confidence:
            return False

        self._different_count += 1
        if self._different_count >= self.consecutive_different_frames:
            self._different_count = 0
            return True
        return False
