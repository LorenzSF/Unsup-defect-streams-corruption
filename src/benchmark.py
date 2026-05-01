import time
from collections import deque
from typing import Deque

import numpy as np

from .schemas import (
    BenchmarkConfig,
    Frame,
    MetricsConfig,
    MetricSnapshot,
    Prediction,
)
from .metrics import OnlineMetrics


class OnlineBaseline:
    """Per-frame incremental PCA-style detector.
    """

    GRID = 64
    K = 32
    RESERVOIR = 200

    def __init__(self, cfg: BenchmarkConfig) -> None:
        self.cfg = cfg
        if cfg.baseline != "online_autoencoder":
            raise ValueError(
                f"unknown baseline '{cfg.baseline}' "
                f"(supported: 'online_autoencoder')"
            )
        self.lr = cfg.learning_rate
        self._reservoir: Deque[np.ndarray] = deque(maxlen=self.RESERVOIR)
        self._mean: np.ndarray | None = None
        self._components: np.ndarray | None = None
        self._metrics = OnlineMetrics(MetricsConfig(window_size=500))

    def update(self, frame: Frame, pred: Prediction) -> None:
        """Take one online step on this frame and record baseline metrics.
        `pred` is the SOTA model's prediction for the same frame and is not
        used here — it is part of the signature so main.py treats baseline
        and SOTA model symmetrically.
        """
        del pred  
        t0 = time.perf_counter()
        x = self._flatten(frame.image)
        score = self._score(x)

        # Score first, then update the online state with the current frame.
        # This avoids leaking the frame into its own anomaly estimate.
        if self._mean is None:
            self._mean = x.copy()
        else:
            self._mean = (1.0 - self.lr) * self._mean + self.lr * x
        self._reservoir.append(x - self._mean)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        self._metrics.update(
            frame,
            Prediction(score=score, anomaly_map=None, latency_ms=latency_ms),
        )

    def snapshot(self) -> MetricSnapshot:
        return self._metrics.snapshot()

    # -- internals --

    def _score(self, x: np.ndarray) -> float:
        if len(self._reservoir) < 2 or self._mean is None:
            return 0.0
        m = np.stack(list(self._reservoir))  # (N, D)
        k = min(self.K, m.shape[0], m.shape[1])
        _, _, vt = np.linalg.svd(m, full_matrices=False)
        comps = vt[:k]
        xc = x - self._mean
        proj = comps @ xc
        recon = comps.T @ proj
        residual = xc - recon
        return float(np.linalg.norm(residual))

    def _flatten(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        ys = np.linspace(0, h - 1, self.GRID).astype(int)
        xs = np.linspace(0, w - 1, self.GRID).astype(int)
        small = image[np.ix_(ys, xs)]
        if small.ndim == 2:
            small = np.repeat(small[..., None], 3, axis=-1)
        return small.astype(np.float32).flatten() / 255.0
