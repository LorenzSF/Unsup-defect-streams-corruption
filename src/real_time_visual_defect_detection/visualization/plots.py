"""Static chart generation for defect detection run results.

This module provides functions that consume the ``predictions.json`` output
produced by :func:`~real_time_visual_defect_detection.pipelines.run_pipeline.run_pipeline`
and generate publication-ready figures using Plotly.

Typical usage
-------------
::

    from real_time_visual_defect_detection.visualization.plots import (
        plot_score_distribution,
        plot_roc_curve,
        plot_confusion_matrix,
        plot_scores_by_defect_type,
        plot_embedding_umap,
    )

    records = json.loads(Path("data/runs/baseline_.../predictions.json").read_text())

    fig = plot_score_distribution(records)
    fig.write_html("score_dist.html")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def plot_score_distribution(
    records: List[Dict[str, Any]],
    threshold: float = 0.5,
    title: str = "Anomaly Score Distribution",
):
    """Render overlapping histograms of anomaly scores for good vs. defective images.

    Produces a Plotly histogram figure with two overlapping traces — one for
    label=0 (good) samples and one for label=1 (defective) samples — plus a
    vertical dashed line at ``threshold`` indicating the decision boundary.
    Useful for visually assessing score separability and choosing an optimal
    threshold.

    Parameters
    ----------
    records:
        List of prediction dicts as written to ``predictions.json``.  Each
        dict must contain at least ``"label"`` (int) and ``"score"`` (float).
    threshold:
        Decision threshold to overlay as a vertical line.
    title:
        Chart title string.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure; call ``.show()`` or ``.write_html()``.
    """
    raise NotImplementedError


def plot_roc_curve(
    records: List[Dict[str, Any]],
    title: str = "ROC Curve",
):
    """Render the Receiver Operating Characteristic (ROC) curve for a run.

    Computes the false-positive rate and true-positive rate across all score
    thresholds and plots the resulting curve together with the AUROC score
    annotated in the legend.  Samples with ``label == -1`` (unlabeled) are
    excluded from the computation.

    Parameters
    ----------
    records:
        List of prediction dicts from ``predictions.json``.  Each dict must
        contain ``"label"`` (int) and ``"score"`` (float).
    title:
        Chart title string.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure.
    """
    raise NotImplementedError


def plot_confusion_matrix(
    records: List[Dict[str, Any]],
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
):
    """Render a color-coded confusion matrix heatmap.

    Builds a 2×2 confusion matrix (normal vs. anomalous) from the binary
    ``pred_is_anomaly`` predictions and the ground-truth ``label`` field.
    Annotates each cell with its count and percentage.  Samples with
    ``label == -1`` are excluded.

    Parameters
    ----------
    records:
        List of prediction dicts from ``predictions.json``.
    labels:
        Display names for the two classes.  Defaults to
        ``["Normal", "Anomalous"]``.
    title:
        Chart title string.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure (Plotly heatmap).
    """
    raise NotImplementedError


def plot_scores_by_defect_type(
    records: List[Dict[str, Any]],
    title: str = "Anomaly Scores by Defect Type",
):
    """Render a box-plot of anomaly scores broken down by defect category.

    Groups defective samples (``label == 1``) by their ``defect_type`` field
    and plots per-group box-plots side by side.  A separate box for good
    samples (``label == 0``) is prepended for reference.  Helps identify which
    defect categories are harder or easier to detect.

    Parameters
    ----------
    records:
        List of prediction dicts from ``predictions.json``.  Each defective
        record should contain a ``"defect_type"`` key (or ``null`` if the
        category is unknown).
    title:
        Chart title string.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure.
    """
    raise NotImplementedError


def plot_embedding_umap(
    embeddings: np.ndarray,
    labels: List[int],
    paths: List[str],
    scores: Optional[List[float]] = None,
    defect_types: Optional[List[Optional[str]]] = None,
    title: str = "Feature Embeddings (UMAP)",
    umap_kwargs: Optional[Dict[str, Any]] = None,
):
    """Reduce embeddings to 2-D with UMAP and render an interactive scatter plot.

    Each point represents one test-set image.  Points are coloured by class
    label (good = blue, anomalous = red, unlabeled = grey) and annotated with
    hover text showing the file path, anomaly score, and defect type.

    Parameters
    ----------
    embeddings:
        2-D float32 array of shape ``(N, D)`` where *N* is the number of
        samples and *D* is the embedding dimensionality.
    labels:
        Ground-truth class label for each sample: 0 = good, 1 = anomalous,
        -1 = unlabeled.
    paths:
        File path string for each sample (used in hover text).
    scores:
        Anomaly score for each sample (used in hover text).  Pass ``None``
        to omit scores from hover text.
    defect_types:
        Defect category string for each sample (``None`` for good/unlabeled
        images).  Used in hover text.
    title:
        Chart title string.
    umap_kwargs:
        Extra keyword arguments forwarded to ``umap.UMAP()``.  Defaults
        are ``n_neighbors=15``, ``min_dist=0.1``, ``random_state=42``.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly scatter figure.  Save with ``.write_html()``.
    """
    import umap
    import plotly.graph_objects as go

    kw = {"n_neighbors": 15, "min_dist": 0.1, "random_state": 42}
    if umap_kwargs:
        kw.update(umap_kwargs)

    coords = umap.UMAP(n_components=2, **kw).fit_transform(embeddings)

    label_meta = {
        0:  {"name": "Good",       "color": "#2196F3"},
        1:  {"name": "Anomalous",  "color": "#F44336"},
        -1: {"name": "Unlabeled",  "color": "#9E9E9E"},
    }

    fig = go.Figure()

    for lbl, meta in label_meta.items():
        idx = [i for i, l in enumerate(labels) if l == lbl]
        if not idx:
            continue

        hover_parts = [
            f"<b>{meta['name']}</b><br>"
            f"path: {paths[i].split('/')[-1]}<br>"
            + (f"score: {scores[i]:.4f}<br>" if scores is not None else "")
            + (f"type: {defect_types[i]}<br>" if defect_types is not None and defect_types[i] else "")
            for i in idx
        ]

        fig.add_trace(go.Scatter(
            x=coords[idx, 0],
            y=coords[idx, 1],
            mode="markers",
            name=meta["name"],
            marker=dict(color=meta["color"], size=6, opacity=0.8),
            text=hover_parts,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        legend_title="Class",
        template="plotly_white",
        hoverlabel=dict(bgcolor="white"),
    )

    return fig
