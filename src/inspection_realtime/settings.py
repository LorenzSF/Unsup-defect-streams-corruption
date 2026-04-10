from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_SETTINGS_FILE = Path("src") / "inspection_realtime" / "settings.yaml"


def load_settings(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Runtime config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            cfg = json.load(handle)
        else:
            cfg = yaml.safe_load(handle)

    if not isinstance(cfg, dict):
        raise ValueError(f"Runtime config must be a dictionary-like object: {path}")
    return cfg


def _read_summary_models(run_dir: Path) -> list[str]:
    summary_path = run_dir / "benchmark_summary.json"
    if not summary_path.exists():
        return []
    try:
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(rows, list):
        return []
    return [str(row.get("model", "")) for row in rows if isinstance(row, dict)]


def find_latest_run(
    runs_root: str | Path,
    model_name: Optional[str] = None,
) -> Path:
    runs_root = Path(runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")

    candidates: list[Path] = []
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        if model_name is None:
            candidates.append(child)
            continue
        if model_name in _read_summary_models(child):
            candidates.append(child)

    if not candidates:
        detail = f" for model '{model_name}'" if model_name else ""
        raise FileNotFoundError(f"No benchmark runs found{detail} in: {runs_root}")
    return candidates[-1].resolve()


def resolve_runtime_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    resolved = dict(cfg)

    run_cfg = dict(resolved.get("run", {}))
    run_cfg.setdefault("output_dir", "data/inspection realtime")
    run_cfg.setdefault("session_name", "inspection realtime")
    run_cfg.setdefault("target_fps", 10.0)
    run_cfg.setdefault("latency_sla_ms", 100.0)
    run_cfg.setdefault("max_frames", None)
    resolved["run"] = run_cfg

    artifact_cfg = dict(resolved.get("artifact", {}))
    artifact_cfg.setdefault("runs_root", "data/runs")
    artifact_cfg.setdefault("run_dir", None)
    artifact_cfg.setdefault("model_name", "rd4ad")
    artifact_cfg.setdefault("fit_policy", "auto")

    run_dir = artifact_cfg.get("run_dir")
    if run_dir in (None, "", "latest"):
        model_name = artifact_cfg.get("model_name")
        artifact_cfg["resolved_run_dir"] = str(
            find_latest_run(
                runs_root=artifact_cfg["runs_root"],
                model_name=None if model_name in (None, "") else str(model_name),
            )
        )
    else:
        artifact_cfg["resolved_run_dir"] = str(Path(str(run_dir)).resolve())
    resolved["artifact"] = artifact_cfg

    input_cfg = dict(resolved.get("input", {}))
    input_cfg.setdefault("root_dir", "data/raw/extracted")
    input_cfg.setdefault("loop", False)
    input_cfg.setdefault("sequence_mode", "interleaved_labels")
    resolved["input"] = input_cfg

    pre_cfg = dict(resolved.get("preprocessing", {}))
    resize_cfg = dict(pre_cfg.get("resize", {}))
    resize_cfg.setdefault("enabled", True)
    resize_cfg.setdefault("width", 256)
    resize_cfg.setdefault("height", 256)
    pre_cfg["resize"] = resize_cfg

    normalize_cfg = dict(pre_cfg.get("normalize", {}))
    normalize_cfg.setdefault("enabled", True)
    normalize_cfg.setdefault("mode", "0_1")
    pre_cfg["normalize"] = normalize_cfg
    resolved["preprocessing"] = pre_cfg

    change_cfg = dict(resolved.get("object_change", {}))
    change_cfg.setdefault("embedding_size", 24)
    change_cfg.setdefault("distance_threshold", 0.12)
    change_cfg.setdefault("reference_bank_size", 24)
    change_cfg.setdefault("consecutive_different_frames", 3)
    change_cfg.setdefault("min_confidence", 0.15)
    resolved["object_change"] = change_cfg

    calibration_cfg = dict(resolved.get("calibration", {}))
    calibration_cfg.setdefault("min_frames", 20)
    calibration_cfg.setdefault("require_baseline", True)
    resolved["calibration"] = calibration_cfg

    web_cfg = dict(resolved.get("web", {}))
    web_cfg.setdefault("enabled", True)
    web_cfg.setdefault("host", "127.0.0.1")
    web_cfg.setdefault("port", 8765)
    web_cfg.setdefault("refresh_ms", 1000)
    resolved["web"] = web_cfg

    return resolved
