from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class MetricsResult:
    precision: float
    recall: float
    f1: float
    accuracy: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MetricsResult:
    return MetricsResult(
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
    )


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    """Compute image-level binary metrics, including threshold-free scores."""
    if y_true.size == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "auroc": 0.0,
            "aupr": 0.0,
        }

    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if len(np.unique(y_true)) >= 2:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        metrics["aupr"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["auroc"] = 0.0
        metrics["aupr"] = 0.0
    return metrics
