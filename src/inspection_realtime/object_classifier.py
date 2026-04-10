from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .contracts import ObjectMatch


class RelativeObjectClassifier:
    def __init__(
        self,
        embedding_size: int = 24,
        distance_threshold: float = 0.12,
        reference_bank_size: int = 24,
    ) -> None:
        self.embedding_size = max(8, int(embedding_size))
        self.distance_threshold = float(distance_threshold)
        self._references: deque[np.ndarray] = deque(maxlen=max(1, int(reference_bank_size)))

    @property
    def has_reference(self) -> bool:
        return len(self._references) > 0

    def reset_with_frame(self, image_bgr: np.ndarray) -> None:
        self._references.clear()
        self.update_reference(image_bgr)

    def update_reference(self, image_bgr: np.ndarray) -> None:
        self._references.append(self._signature(image_bgr))

    def classify(self, image_bgr: np.ndarray) -> ObjectMatch:
        if not self.has_reference:
            return ObjectMatch(is_same_reference=True, confidence=1.0, distance=0.0)

        signature = self._signature(image_bgr)
        reference = self._reference_signature()
        distance = float(max(0.0, 1.0 - float(np.dot(signature, reference))))
        is_same_reference = distance <= self.distance_threshold
        confidence = self._confidence(distance)
        return ObjectMatch(
            is_same_reference=is_same_reference,
            confidence=confidence,
            distance=distance,
        )

    def _reference_signature(self) -> np.ndarray:
        stacked = np.stack(list(self._references), axis=0)
        averaged = stacked.mean(axis=0).astype(np.float32)
        norm = float(np.linalg.norm(averaged))
        if norm > 0.0:
            averaged /= norm
        return averaged

    def _signature(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(
            gray,
            (self.embedding_size, self.embedding_size),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32) / 255.0

        hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).reshape(-1).astype(np.float32)
        hist_sum = float(hist.sum())
        if hist_sum > 0.0:
            hist /= hist_sum

        signature = np.concatenate([thumb.reshape(-1), hist], axis=0).astype(np.float32)
        norm = float(np.linalg.norm(signature))
        if norm > 0.0:
            signature /= norm
        return signature

    def _confidence(self, distance: float) -> float:
        scale = max(self.distance_threshold, 1e-6)
        margin = abs(distance - self.distance_threshold)
        return float(np.clip(margin / scale, 0.0, 1.0))
