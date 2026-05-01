import math
import time
from collections import deque
from typing import Deque

import numpy as np

from .schemas import Frame, MetricsConfig, MetricSnapshot, Prediction


class OnlineMetrics:
    def __init__(self, cfg: MetricsConfig) -> None:
        self.cfg = cfg
        self._scores: Deque[float] = deque(maxlen=cfg.window_size)
        self._labels: Deque[int] = deque(maxlen=cfg.window_size)
        self._global_auroc = _HistogramAUROC()

        # global running counts
        self._n_seen = 0
        self._n_anomalies = 0

        # latency stats: running mean (Welford) + P² for p95
        self._lat_sum = 0.0
        self._lat_n = 0
        self._p95 = _P2Quantile(0.95)

        # throughput: based on wall-clock since the first update
        self._t_first: float | None = None
        self._t_last: float | None = None


    def update(self, frame: Frame, pred: Prediction) -> None:
        now = time.perf_counter()
        if self._t_first is None:
            self._t_first = now
        self._t_last = now

        self._n_seen += 1
        if frame.label == 1:
            self._n_anomalies += 1

        # only labelled frames contribute to AUROC / F1
        if frame.label in (0, 1):
            self._scores.append(pred.score)
            self._labels.append(frame.label)
            self._global_auroc.add(pred.score, frame.label)

        self._lat_sum += pred.latency_ms
        self._lat_n += 1
        self._p95.add(pred.latency_ms)


    def snapshot(self) -> MetricSnapshot:
        window_auroc = _auroc(self._scores, self._labels)
        global_auroc = self._global_auroc.value()
        f1 = _best_f1(self._scores, self._labels)
        mean_lat = self._lat_sum / self._lat_n if self._lat_n else 0.0
        p95 = self._p95.value()

        if self._t_first is None or self._t_last is None or self._t_last == self._t_first:
            fps = 0.0
        else:
            fps = self._n_seen / (self._t_last - self._t_first)

        return MetricSnapshot(
            n_seen=self._n_seen,
            n_anomalies=self._n_anomalies,
            auroc=global_auroc if not math.isnan(global_auroc) else window_auroc,
            f1=f1,
            mean_latency_ms=mean_lat,
            p95_latency_ms=p95,
            throughput_fps=fps,
        )


    def finalize(self) -> dict:
        snap = self.snapshot()
        return {
            "n_seen": snap.n_seen,
            "n_anomalies": snap.n_anomalies,
            "auroc": snap.auroc,
            "window_auroc": _auroc(self._scores, self._labels),
            "global_auroc": self._global_auroc.value(),
            "f1": snap.f1,
            "mean_latency_ms": snap.mean_latency_ms,
            "p95_latency_ms": snap.p95_latency_ms,
            "throughput_fps": snap.throughput_fps,
            "window_size": self.cfg.window_size,
        }


def _auroc(scores, labels) -> float:
    """Mann–Whitney U formulation of AUROC, with tie correction."""
    if len(scores) == 0:
        return float("nan")
    arr = np.asarray(scores, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.int64)
    n_pos = int((lab == 1).sum())
    n_neg = int((lab == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(arr, kind="mergesort")
    arr_s = arr[order]
    lab_s = lab[order]
    # average ranks (1-indexed) with ties
    ranks = np.empty(len(arr_s), dtype=np.float64)
    i = 0
    while i < len(arr_s):
        j = i
        while j + 1 < len(arr_s) and arr_s[j + 1] == arr_s[i]:
            j += 1
        avg = 0.5 * (i + 1 + j + 1)
        ranks[i : j + 1] = avg
        i = j + 1
    sum_ranks_pos = ranks[lab_s == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _best_f1(scores, labels) -> float:
    """F1 at the threshold that maximizes it on the sliding window."""
    if len(scores) == 0:
        return float("nan")
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n_pos = int((y == 1).sum())
    if n_pos == 0 or n_pos == len(y):
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_s = y[order]
    tp = np.cumsum(y_s == 1)
    fp = np.cumsum(y_s == 0)
    fn = n_pos - tp
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(f1.max())


class _HistogramAUROC:
    """Bounded-memory global AUROC estimate from fixed score histograms."""

    BINS = 512

    def __init__(self) -> None:
        self._pos = np.zeros(self.BINS, dtype=np.int64)
        self._neg = np.zeros(self.BINS, dtype=np.int64)
        self._n_pos = 0
        self._n_neg = 0

    def add(self, score: float, label: int) -> None:
        idx = self._bin_index(score)
        if label == 1:
            self._pos[idx] += 1
            self._n_pos += 1
        elif label == 0:
            self._neg[idx] += 1
            self._n_neg += 1

    def value(self) -> float:
        if self._n_pos == 0 or self._n_neg == 0:
            return float("nan")

        wins = 0.0
        neg_before = 0
        for pos_count, neg_count in zip(self._pos, self._neg):
            wins += float(pos_count) * (float(neg_before) + 0.5 * float(neg_count))
            neg_before += int(neg_count)
        return wins / float(self._n_pos * self._n_neg)

    def _bin_index(self, score: float) -> int:
        unit = math.atan(float(score)) / math.pi + 0.5
        idx = int(unit * self.BINS)
        return max(0, min(self.BINS - 1, idx))


class _P2Quantile:
    def __init__(self, p: float) -> None:
        self.p = p
        self._n = 0
        self._q: list[float] = []
        self._np: list[float] = []
        self._npd: list[float] = [
            1.0,
            1.0 + 2.0 * p,
            1.0 + 4.0 * p,
            3.0 + 2.0 * p,
            5.0,
        ]
        self._dn: list[float] = [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0]

    def add(self, x: float) -> None:
        self._n += 1
        if self._n <= 5:
            self._q.append(x)
            if self._n == 5:
                self._q.sort()
                self._np = [1.0, 2.0, 3.0, 4.0, 5.0]
            return

        if x < self._q[0]:
            self._q[0] = x
            k = 0
        elif x >= self._q[4]:
            self._q[4] = x
            k = 3
        else:
            k = 0
            for i in range(4):
                if self._q[i] <= x < self._q[i + 1]:
                    k = i
                    break

        for i in range(k + 1, 5):
            self._np[i] += 1.0
        for i in range(5):
            self._npd[i] += self._dn[i]

        for i in range(1, 4):
            d = self._npd[i] - self._np[i]
            gap_up = self._np[i + 1] - self._np[i]
            gap_dn = self._np[i - 1] - self._np[i]
            if (d >= 1.0 and gap_up > 1.0) or (d <= -1.0 and gap_dn < -1.0):
                d_sign = 1.0 if d >= 0 else -1.0
                qp = self._parabolic(i, d_sign)
                if not (self._q[i - 1] < qp < self._q[i + 1]):
                    qp = self._linear(i, d_sign)
                self._q[i] = qp
                self._np[i] += d_sign

    def value(self) -> float:
        if self._n == 0:
            return 0.0
        if self._n < 5:
            s = sorted(self._q)
            idx = max(0, min(len(s) - 1, int(round((len(s) - 1) * self.p))))
            return s[idx]
        return self._q[2]

    def _parabolic(self, i: int, d: float) -> float:
        return self._q[i] + d / (self._np[i + 1] - self._np[i - 1]) * (
            (self._np[i] - self._np[i - 1] + d)
            * (self._q[i + 1] - self._q[i])
            / (self._np[i + 1] - self._np[i])
            + (self._np[i + 1] - self._np[i] - d)
            * (self._q[i] - self._q[i - 1])
            / (self._np[i] - self._np[i - 1])
        )

    def _linear(self, i: int, d: float) -> float:
        j = i + int(d)
        return self._q[i] + d * (self._q[j] - self._q[i]) / (self._np[j] - self._np[i])
