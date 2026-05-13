import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Deque

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .schemas import Frame, MetricsConfig, MetricSnapshot, Prediction


class OnlineMetrics:
    def __init__(self, cfg: MetricsConfig) -> None:
        self.cfg = cfg
        self._threshold = (
            cfg.threshold_value
            if cfg.threshold_value is not None
            else cfg.manual_threshold
        )
        if self._threshold is None:
            raise ValueError(
                "OnlineMetrics requires MetricsConfig.threshold_value or "
                "MetricsConfig.manual_threshold to be set"
            )
        self._scores: Deque[float] = deque(maxlen=cfg.window_size)
        self._labels: Deque[int] = deque(maxlen=cfg.window_size)
        self._global_auroc = _HistogramAUROC()
        self._all_scores: list[float] = []
        self._all_labels: list[int] = []
        self._n_nonfinite_scores = 0

        # global running counts
        self._n_seen = 0
        self._n_anomalies = 0
        self._n_label_normals = 0
        self._n_label_anomalies = 0
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._tn = 0

        # latency stats: running mean (Welford) + P² for p95
        self._lat_sum = 0.0
        self._lat_n = 0
        self._p95 = _P2Quantile(0.95)

        # throughput: based on wall-clock since the first update
        self._t_first: float | None = None
        self._t_last: float | None = None


    def set_threshold(self, threshold: float) -> None:
        if not math.isfinite(float(threshold)):
            raise ValueError(f"threshold must be finite, got {threshold}")
        self._threshold = float(threshold)


    def update(
        self, frame: Frame, pred: Prediction, threshold: float | None = None
    ) -> None:
        now = time.perf_counter()
        if self._t_first is None:
            self._t_first = now
        self._t_last = now

        threshold_used = self._threshold if threshold is None else float(threshold)
        pred_label = _pred_label(pred.score, threshold_used)
        self._n_seen += 1
        if pred_label == 1:
            self._n_anomalies += 1

        # only labelled frames contribute to AUROC / F1
        if frame.label in (0, 1):
            if frame.label == 1:
                self._n_label_anomalies += 1
            else:
                self._n_label_normals += 1
            self._update_binary_counts(pred_label, frame.label)
            if np.isfinite(pred.score):
                self._scores.append(float(pred.score))
                self._labels.append(frame.label)
                self._global_auroc.add(pred.score, frame.label)
                self._all_scores.append(float(pred.score))
                self._all_labels.append(frame.label)
            else:
                self._n_nonfinite_scores += 1

        self._lat_sum += pred.latency_ms
        self._lat_n += 1
        self._p95.add(pred.latency_ms)


    def snapshot(self) -> MetricSnapshot:
        window_auroc = _auroc(self._scores, self._labels)
        global_auroc = self._global_auroc.value()
        _, _, f1, _ = _binary_metrics(self._scores, self._labels, self._threshold)
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
        scores = np.asarray(self._all_scores, dtype=np.float64)
        labels = np.asarray(self._all_labels, dtype=np.int64)
        precision, recall, f1, accuracy = _binary_metrics_from_counts(
            self._tp,
            self._fp,
            self._fn,
            self._tn,
            self._n_label_anomalies,
            self._n_label_normals,
        )
        auroc = _exact_auroc(scores, labels)
        aupr = _exact_aupr(scores, labels)
        return {
            "n_seen": snap.n_seen,
            "n_anomalies": snap.n_anomalies,
            "n_predicted_anomalies": snap.n_anomalies,
            "n_label_anomalies": self._n_label_anomalies,
            "n_scored": int(labels.size),
            "n_nonfinite_scores": self._n_nonfinite_scores,
            "labels_available": bool(
                self._n_label_anomalies + self._n_label_normals > 0
            ),
            "auroc": auroc,
            "aupr": aupr,
            "window_auroc": _auroc(self._scores, self._labels),
            "global_auroc": self._global_auroc.value(),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "mean_latency_ms": snap.mean_latency_ms,
            "p95_latency_ms": snap.p95_latency_ms,
            "throughput_fps": snap.throughput_fps,
            "window_size": self.cfg.window_size,
            "threshold_mode": self.cfg.threshold_mode,
            "threshold_used": self._threshold,
        }


    def _update_binary_counts(self, pred_label: int, true_label: int) -> None:
        if pred_label == 1 and true_label == 1:
            self._tp += 1
        elif pred_label == 1 and true_label == 0:
            self._fp += 1
        elif pred_label == 0 and true_label == 1:
            self._fn += 1
        elif pred_label == 0 and true_label == 0:
            self._tn += 1


class FrameLogger:
    """Per-frame JSONL trace writer; line-buffered so a crash leaves a valid prefix."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh = None

    def __enter__(self) -> "FrameLogger":
        self._fh = self._path.open("w", buffering=1, encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def write(self, frame: Frame, pred: Prediction, threshold: float) -> None:
        score = float(pred.score)
        pred_label = _frame_pred_label(score, threshold)
        image_id = frame.image_id or Path(frame.source_id).stem
        self._fh.write(
            json.dumps(
                {
                    "idx": int(frame.index),
                    "image_id": image_id,
                    "score": score if math.isfinite(score) else None,
                    "pred_label": pred_label,
                    "threshold_used": float(threshold),
                    "true_label": int(frame.label),
                    "latency_ms": float(pred.latency_ms),
                }
            )
            + "\n"
        )


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


def _binary_metrics(scores, labels, threshold: float) -> tuple[float, float, float, float]:
    if len(scores) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    pred = (s >= float(threshold)).astype(np.int64)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return float(precision), float(recall), float(f1), float(accuracy)


def _binary_metrics_from_counts(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    n_label_anomalies: int,
    n_label_normals: int,
) -> tuple[float, float, float, float]:
    if n_label_anomalies == 0 or n_label_normals == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return float(precision), float(recall), float(f1), float(accuracy)


def _pred_label(score: float, threshold: float) -> int:
    if not math.isfinite(float(score)):
        return 1
    return int(float(score) >= float(threshold))


def _frame_pred_label(score: float, threshold: float) -> int:
    if not math.isfinite(float(score)):
        return -1
    return int(float(score) >= float(threshold))


def _exact_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    if scores.size == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _exact_aupr(scores: np.ndarray, labels: np.ndarray) -> float:
    if scores.size == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


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
