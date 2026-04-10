from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .contracts import CalibrationBaseline, RuntimeArtifact


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _resolve_model_cfg(cfg: dict[str, Any], model_name: str) -> dict[str, Any]:
    bench_cfg = cfg.get("benchmark", {})
    models = bench_cfg.get("models", [])
    if isinstance(models, list):
        for entry in models:
            if isinstance(entry, dict) and entry.get("name") == model_name:
                return dict(entry)

    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict):
        resolved = dict(model_cfg)
        resolved["name"] = model_name
        return resolved
    raise ValueError(f"Unable to resolve model configuration for: {model_name}")


def _select_summary_row(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("model")) == model_name:
            return row
    raise ValueError(f"Model '{model_name}' not found in benchmark summary.")


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_runtime_artifact(
    run_dir: str | Path,
    model_name: Optional[str] = None,
) -> RuntimeArtifact:
    run_dir = Path(run_dir).resolve()
    config_path = run_dir / "config_snapshot.json"
    summary_path = run_dir / "benchmark_summary.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config snapshot: {config_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing benchmark summary: {summary_path}")

    cfg = _load_json(config_path)
    summary_rows = _load_json(summary_path, default=[])
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config snapshot: {config_path}")
    if not isinstance(summary_rows, list) or len(summary_rows) == 0:
        raise ValueError(f"Invalid benchmark summary: {summary_path}")

    if model_name is None:
        model_name = str(summary_rows[0].get("model", ""))
    if not model_name:
        raise ValueError("Unable to resolve runtime model name from benchmark summary.")

    summary_row = _select_summary_row(summary_rows, model_name=model_name)
    validation_predictions_path = run_dir / f"validation_predictions_{_safe_name(model_name)}.json"
    validation_rows = _load_json(validation_predictions_path, default=[])

    normal_scores: list[float] = []
    if isinstance(validation_rows, list):
        for row in validation_rows:
            if not isinstance(row, dict):
                continue
            label = int(row.get("label", -1))
            if label == 0:
                normal_scores.append(float(row.get("score", 0.0)))

    if len(normal_scores) == 0 and isinstance(validation_rows, list):
        normal_scores = [
            float(row.get("score", 0.0))
            for row in validation_rows
            if isinstance(row, dict) and "score" in row
        ]

    baseline = CalibrationBaseline(
        threshold=float(summary_row.get("threshold_used", cfg.get("model", {}).get("threshold", 0.5))),
        score_mean=float(np.mean(normal_scores)) if normal_scores else 0.0,
        score_std=float(np.std(normal_scores)) if normal_scores else 0.0,
        sample_count=len(normal_scores),
    )

    return RuntimeArtifact(
        run_dir=run_dir,
        model_name=model_name,
        model_cfg=_resolve_model_cfg(cfg, model_name=model_name),
        runtime_cfg=dict(cfg.get("runtime", {})),
        dataset_cfg=dict(cfg.get("dataset", {})),
        preprocessing_cfg=dict(cfg.get("preprocessing", {})),
        threshold=baseline.threshold,
        baseline=baseline,
        validation_predictions_path=validation_predictions_path if validation_predictions_path.exists() else None,
    )
