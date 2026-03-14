"""Interactive Plotly Dash dashboard for exploring pipeline run results.

This module exposes two public callables:

* :func:`build_app` — constructs and returns a configured ``Dash`` application
  instance without starting a server, useful for testing or embedding.
* :func:`run_dashboard` — convenience wrapper that calls :func:`build_app` and
  launches the development server on a configurable host / port.

Dashboard layout
----------------
The dashboard is organised into three sections:

1. **Run selector** — dropdown populated from the ``data/runs/`` directory that
   lets the user switch between multiple saved pipeline runs without restarting
   the server.  Selecting a run reloads all charts from the corresponding
   ``predictions.json`` and ``config_snapshot.json``.

2. **Summary panel** — key-metric cards showing total samples, precision,
   recall, F1, accuracy, and AUROC for the selected run.

3. **Chart tabs** — tabbed area containing the four plots from
   :mod:`~real_time_visual_defect_detection.visualization.plots`:

   * Score Distribution
   * ROC Curve
   * Confusion Matrix
   * Scores by Defect Type

Typical usage
-------------
::

    from real_time_visual_defect_detection.visualization.dashboard import run_dashboard

    run_dashboard(runs_dir="data/runs", host="127.0.0.1", port=8050, debug=True)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def build_app(
    runs_dir: str | Path = "data/runs",
    default_run: Optional[str] = None,
):
    """Construct and return the Dash application instance.

    Reads available runs from *runs_dir*, sets up the layout (run selector,
    summary cards, chart tabs), and registers all Dash callbacks for
    interactivity.  The server is **not** started; call ``app.run(...)`` or
    use :func:`run_dashboard` to launch it.

    Parameters
    ----------
    runs_dir:
        Directory that contains timestamped run subdirectories produced by
        :func:`~real_time_visual_defect_detection.pipelines.run_pipeline.run_pipeline`.
        Each subdirectory must contain ``predictions.json`` and
        ``config_snapshot.json``.
    default_run:
        Name of the run subdirectory to load on startup.  When ``None`` the
        most recently modified run is selected automatically.

    Returns
    -------
    dash.Dash
        Fully configured Dash application instance ready to serve.
    """
    raise NotImplementedError


def run_dashboard(
    runs_dir: str | Path = "data/runs",
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    default_run: Optional[str] = None,
) -> None:
    """Build the Dash app and start the development server.

    Convenience wrapper around :func:`build_app` that immediately launches the
    browser-accessible server.  Blocks until the server process is terminated
    (e.g. with Ctrl-C).

    Parameters
    ----------
    runs_dir:
        Directory containing pipeline run outputs (see :func:`build_app`).
    host:
        Network interface to bind.  Use ``"0.0.0.0"`` to expose the dashboard
        on all interfaces (e.g. inside Docker).
    port:
        TCP port to listen on.
    debug:
        When ``True`` enables Dash hot-reloading and verbose error messages.
        Keep ``False`` in production.
    default_run:
        Run subdirectory name to pre-select on load.  ``None`` picks the
        latest run automatically.
    """
    raise NotImplementedError
