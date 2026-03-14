"""Visualization package for the real-time visual defect detection pipeline.

Modules
-------
plots
    Static chart generation: score distributions, ROC curves, confusion
    matrices, and per-defect-type breakdowns.
dashboard
    Interactive Plotly Dash application for exploring run outputs in a
    browser-based dashboard.
"""

from .plots import (
    plot_score_distribution,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_scores_by_defect_type,
    plot_embedding_umap,
)
from .dashboard import build_app, run_dashboard

__all__ = [
    "plot_score_distribution",
    "plot_roc_curve",
    "plot_confusion_matrix",
    "plot_scores_by_defect_type",
    "plot_embedding_umap",
    "build_app",
    "run_dashboard",
]
