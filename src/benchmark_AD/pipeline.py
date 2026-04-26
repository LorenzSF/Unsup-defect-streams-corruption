from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def _read_cfg_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f) if suffix == ".json" else yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a dictionary-like object: {path}")
    return cfg


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *overlay* onto *base*, returning a new dict.

    Dict values are merged key-by-key; any other type in *overlay* replaces
    the corresponding entry in *base*. This lets small per-dataset configs
    override only the fields they care about (via a top-level ``_extends``).
    """
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a JSON or YAML config, resolving an optional ``_extends`` chain.

    When a config contains a top-level ``_extends: <path>`` key, that base
    file is loaded first (recursively) and the current file is merged over
    it. The ``_extends`` value is interpreted relative to the current file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg = _read_cfg_file(path)
    parent_ref = cfg.pop("_extends", None)
    if parent_ref is None:
        return cfg

    parent_path = (path.parent / str(parent_ref)).resolve()
    parent_cfg = load_config(parent_path)
    return _deep_merge(parent_cfg, cfg)


import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from tqdm.auto import tqdm

from benchmark_AD.evaluation import compute_binary_metrics
from benchmark_AD.data import (
    apply_dataset_split,
    resolve_dataset_labeled,
)
from benchmark_AD.models import available_models, build_model
from benchmark_AD.data import (
    normalize_0_1,
    read_image_bgr,
    resize,
)
from corruptions.corruption_registry import get_corruption


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _progress_enabled() -> bool:
    return bool(sys.stdout.isatty() or sys.stderr.isatty())


def _stage_started(
    model_name: str,
    stage_idx: int,
    total_stages: int,
    stage_name: str,
    details: str = "",
) -> None:
    suffix = f" | {details}" if details else ""
    print(f"[{model_name}] Stage {stage_idx}/{total_stages} {stage_name}: started{suffix}")


def _stage_done(
    model_name: str,
    stage_idx: int,
    total_stages: int,
    stage_name: str,
    details: str = "",
) -> None:
    remaining = max(0, total_stages - stage_idx)
    suffix = f" | {details}" if details else ""
    print(
        f"[{model_name}] Stage {stage_idx}/{total_stages} {stage_name}: done"
        f" | remaining={remaining}{suffix}"
    )


def _resolve_runtime(runtime_cfg: Dict[str, Any]) -> Dict[str, Any]:
    runtime = dict(runtime_cfg or {})
    requested = str(runtime.get("device", "auto")).lower()
    if requested == "auto":
        resolved = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif requested in ("cpu",):
        resolved = "cpu"
    elif requested in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("runtime.device='cuda' requested but CUDA is not available.")
        resolved = "cuda:0"
    elif requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"runtime.device='{requested}' requested but CUDA is not available.")
        resolved = requested
    else:
        raise ValueError("runtime.device must be one of: auto, cpu, cuda, cuda:<index>.")

    runtime["resolved_device"] = resolved
    runtime["precision"] = str(runtime.get("precision", "fp32")).lower()
    if runtime["precision"] not in {"fp32", "fp16", "bf16"}:
        raise ValueError("runtime.precision must be one of: fp32, fp16, bf16.")
    runtime["num_workers"] = int(runtime.get("num_workers", 0))
    runtime["pin_memory"] = bool(runtime.get("pin_memory", resolved.startswith("cuda")))
    runtime["cudnn_benchmark"] = bool(runtime.get("cudnn_benchmark", True))

    # Keep backend tuning explicit and reproducible.
    if resolved.startswith("cuda"):
        torch.backends.cudnn.benchmark = runtime["cudnn_benchmark"]
    return runtime


def _runtime_info(runtime_cfg: Dict[str, Any]) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "resolved_device": runtime_cfg["resolved_device"],
        "precision": runtime_cfg["precision"],
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if runtime_cfg["resolved_device"].startswith("cuda"):
        idx = int(runtime_cfg["resolved_device"].split(":")[1])
        info["gpu_name"] = torch.cuda.get_device_name(idx)
    return info


def _save_umap(
    out_dir: Path,
    model_name: str,
    embeddings: List[np.ndarray],
    labels: List[int],
    paths: List[str],
    scores: List[float],
    defect_types: List[Optional[str]],
) -> None:
    from benchmark_AD.evaluation import plot_embedding_umap

    matrix = np.stack(embeddings, axis=0)
    fig = plot_embedding_umap(
        embeddings=matrix,
        labels=labels,
        paths=paths,
        scores=scores,
        defect_types=defect_types,
        title=f"Feature Embeddings (UMAP) - {model_name}",
    )
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    fig.write_html(str(plots_dir / f"embedding_umap_{_safe_name(model_name)}.html"))


def _model_cfgs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    bench_cfg = cfg.get("benchmark", {})
    models = bench_cfg.get("models")
    if isinstance(models, list) and len(models) > 0:
        return models
    return [cfg["model"]]


def _best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return None
    p, r, t = precision_recall_curve(y_true, y_score)
    if t.size == 0:
        return None
    f1 = (2 * p[:-1] * r[:-1]) / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.nanargmax(f1))])


def _quantile_threshold_from_negatives(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fpr: float,
) -> Optional[float]:
    negatives = y_score[y_true == 0]
    if negatives.size == 0:
        return None
    q = float(np.clip(1.0 - target_fpr, 0.0, 1.0))
    return float(np.quantile(negatives, q))


def _recall_at_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fpr: float,
) -> Optional[float]:
    """Highest TPR achievable while FPR stays at or below ``target_fpr``.

    Prevalence-invariant operating-point summary — answers "if the line
    only tolerates ``target_fpr`` false alarms, what fraction of defects
    do we catch?". Returns ``None`` if y_true lacks both classes; ``0.0``
    when no threshold satisfies the FPR cap.
    """
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return None
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= float(target_fpr)
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))


def _per_defect_recall(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-``defect_type`` recall plus simple-mean and support-weighted aggregates.

    The macro mean treats every defect class as equally important — useful
    when one class dominates the dataset (e.g. Deceuninck scratches at 80%
    of all bads) and the global recall would otherwise hide failures on
    minority classes. The weighted mean reproduces the global recall on
    bads and is reported for completeness.
    """
    by_type: Dict[str, Dict[str, int]] = {}
    for r in rows:
        if r.get("label") != 1:
            continue
        dtype = r.get("defect_type") or "_unknown"
        bucket = by_type.setdefault(dtype, {"tp": 0, "fn": 0})
        if r.get("pred_is_anomaly") == 1:
            bucket["tp"] += 1
        else:
            bucket["fn"] += 1

    per_recall: Dict[str, float] = {}
    per_support: Dict[str, int] = {}
    for dtype, bucket in by_type.items():
        n = bucket["tp"] + bucket["fn"]
        per_support[dtype] = n
        per_recall[dtype] = (bucket["tp"] / n) if n > 0 else 0.0

    if per_recall:
        macro = sum(per_recall.values()) / len(per_recall)
        total = sum(per_support.values())
        weighted = (
            sum(r * per_support[d] for d, r in per_recall.items()) / total
            if total > 0 else 0.0
        )
    else:
        macro = 0.0
        weighted = 0.0

    return {
        "per_defect_recall": {k: round(v, 6) for k, v in per_recall.items()},
        "per_defect_support": per_support,
        "macro_recall": round(macro, 6),
        "weighted_recall": round(weighted, 6),
    }


def _apply_threshold(rows: List[Dict[str, Any]], threshold: float) -> None:
    for row in rows:
        row["pred_is_anomaly"] = int(float(row["score"]) >= threshold)


def _collect_metrics_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    y_true = np.asarray([r["label"] for r in rows if r["label"] in (0, 1)], dtype=np.int32)
    y_pred = np.asarray([r["pred_is_anomaly"] for r in rows if r["label"] in (0, 1)], dtype=np.int32)
    y_score = np.asarray([r["score"] for r in rows if r["label"] in (0, 1)], dtype=np.float64)
    return compute_binary_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score)


def _maybe_calibrate_threshold(
    model: Any,
    model_cfg: Dict[str, Any],
    val_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    threshold_cfg = model_cfg.get("thresholding", {})
    mode = str(threshold_cfg.get("mode", "fixed")).lower()
    current = float(getattr(model, "threshold", model_cfg.get("threshold", 0.5)))
    result = {"mode": mode, "threshold": current, "calibrated": False}

    if mode == "fixed" or not hasattr(model, "threshold"):
        return result

    y_true = np.asarray([r["label"] for r in val_rows if r["label"] in (0, 1)], dtype=np.int32)
    y_score = np.asarray([r["score"] for r in val_rows if r["label"] in (0, 1)], dtype=np.float64)
    if y_true.size == 0:
        return result

    target_fpr = float(threshold_cfg.get("target_fpr", 0.01))
    if mode == "val_f1":
        threshold = _best_f1_threshold(y_true=y_true, y_score=y_score)
        if threshold is None:
            threshold = _quantile_threshold_from_negatives(
                y_true=y_true,
                y_score=y_score,
                target_fpr=target_fpr,
            )
    elif mode == "val_quantile":
        threshold = _quantile_threshold_from_negatives(
            y_true=y_true,
            y_score=y_score,
            target_fpr=target_fpr,
        )
    else:
        raise ValueError("thresholding.mode must be one of: fixed, val_f1, val_quantile.")

    if threshold is None:
        return result

    model.threshold = float(threshold)
    result["threshold"] = float(threshold)
    result["calibrated"] = True
    return result


def _build_corruption_fn(
    corruption_cfg: Dict[str, Any],
) -> Optional[Any]:
    """Return a per-image callable when corruption is enabled, else ``None``.

    Validation happens here (not lazily per frame) so a misconfigured run
    aborts before any model is loaded.
    """
    if not corruption_cfg or not bool(corruption_cfg.get("enabled", False)):
        return None
    name = str(corruption_cfg["type"])
    severity = int(corruption_cfg["severity"])
    return get_corruption(name, severity)


def _confusion_case(label: int, pred_is_anomaly: int) -> Optional[str]:
    if label == 1 and pred_is_anomaly == 1:
        return "TP"
    if label == 0 and pred_is_anomaly == 1:
        return "FP"
    if label == 1 and pred_is_anomaly == 0:
        return "FN"
    if label == 0 and pred_is_anomaly == 0:
        return "TN"
    return None


def _update_confusion_sample(
    candidates: Dict[str, Dict[str, Any]],
    case: Optional[str],
    image: np.ndarray,
    row: Dict[str, Any],
    threshold: float,
) -> None:
    if case is None:
        return

    distance = abs(float(row["score"]) - threshold)
    current = candidates.get(case)
    if current is not None and distance >= float(current["distance_to_threshold"]):
        return

    candidates[case] = {
        "case": case,
        "image": image.copy(),
        "row": dict(row),
        "threshold": threshold,
        "distance_to_threshold": distance,
    }


def _export_corrupted_confusion_samples(
    out_dir: Path,
    model_name: str,
    candidates: Dict[str, Dict[str, Any]],
    corruption_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Save one corrupted input image per available confusion class."""
    samples_dir = out_dir / "corrupted_confusion_samples" / _safe_name(model_name)
    samples_dir.mkdir(parents=True, exist_ok=True)

    exported: List[Dict[str, Any]] = []
    for case in ("TP", "FP", "FN", "TN"):
        candidate = candidates.get(case)
        if candidate is None:
            continue

        row = candidate["row"]
        original_path = Path(str(row["path"]))
        file_name = f"{original_path.stem}_{case}.png"
        output_path = samples_dir / file_name
        if not cv2.imwrite(str(output_path), candidate["image"]):
            raise OSError(f"Failed to write corrupted confusion sample: {output_path}")

        exported.append(
            {
                "case": case,
                "original_path": str(original_path),
                "saved_path": str(output_path.relative_to(out_dir)),
                "score": float(row["score"]),
                "threshold": float(candidate["threshold"]),
                "distance_to_threshold": float(candidate["distance_to_threshold"]),
                "label": int(row["label"]),
                "pred_is_anomaly": int(row["pred_is_anomaly"]),
                "defect_type": row.get("defect_type"),
            }
        )

    metadata = {
        "model": model_name,
        "corruption": {
            "type": str(corruption_cfg.get("type", "")),
            "severity": int(corruption_cfg.get("severity", 0)),
        },
        "selection": "closest_to_threshold",
        "available_cases": [item["case"] for item in exported],
        "missing_cases": [
            case for case in ("TP", "FP", "FN", "TN") if case not in candidates
        ],
        "samples": exported,
    }
    (samples_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def _run_inference(
    model: Any,
    model_name: str,
    samples: List[Any],
    pre_cfg: Dict[str, Any],
    collect_embeddings: bool,
    stage_name: str,
    corruption_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    do_resize = bool(pre_cfg.get("resize", {}).get("enabled", False))
    w = int(pre_cfg.get("resize", {}).get("width", 256))
    h = int(pre_cfg.get("resize", {}).get("height", 256))

    rows: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []
    embeddings: List[np.ndarray] = []
    emb_labels: List[int] = []
    emb_paths: List[str] = []
    emb_scores: List[float] = []
    emb_defect_types: List[Optional[str]] = []
    allow_embeddings = collect_embeddings
    predict_seconds = 0.0
    confusion_samples: Dict[str, Dict[str, Any]] = {}
    threshold = float(getattr(model, "threshold", 0.5))

    for sample in tqdm(
        samples,
        total=len(samples),
        desc=f"[{model_name}] {stage_name}",
        unit="img",
        dynamic_ncols=True,
        disable=not _progress_enabled(),
    ):
        img = read_image_bgr(str(sample.path))
        if do_resize:
            img = resize(img, (w, h))
        # Corruption acts on the resized BGR uint8 frame; normalization is
        # applied afterwards so the model still sees its expected dtype/range.
        if corruption_fn is not None:
            img = corruption_fn(img)
        x = normalize_0_1(img)

        t_pred0 = time.perf_counter()
        out = model.predict(x)
        sample_latency_s = time.perf_counter() - t_pred0
        predict_seconds += sample_latency_s
        latencies_ms.append(sample_latency_s * 1000.0)
        pred_is_anomaly = int(out.is_anomaly)

        row = {
            "model": model_name,
            "path": str(sample.path),
            "label": sample.label,
            "defect_type": sample.defect_type,
            "score": float(out.score),
            "pred_is_anomaly": pred_is_anomaly,
        }
        rows.append(row)

        if corruption_fn is not None and stage_name == "test":
            _update_confusion_sample(
                candidates=confusion_samples,
                case=_confusion_case(int(sample.label), pred_is_anomaly),
                image=img,
                row=row,
                threshold=threshold,
            )

        if allow_embeddings:
            emb = model.get_embedding(x)
            if emb is None:
                allow_embeddings = False
            else:
                embeddings.append(emb)
                emb_labels.append(sample.label)
                emb_paths.append(str(sample.path))
                emb_scores.append(float(out.score))
                emb_defect_types.append(sample.defect_type)

    return {
        "rows": rows,
        "predict_seconds": predict_seconds,
        "latencies_ms": latencies_ms,
        "embeddings": embeddings,
        "emb_labels": emb_labels,
        "emb_paths": emb_paths,
        "emb_scores": emb_scores,
        "emb_defect_types": emb_defect_types,
        "confusion_samples": confusion_samples,
    }


def _row_extras(
    corruption_cfg: Optional[Dict[str, Any]],
    corruption_fn: Optional[Any],
    dataset_name: str,
) -> Dict[str, Any]:
    """Per-frame metadata that the streaming-shape JSON needs in every row.

    Kept identical for clean and corrupted runs (empty/0 when disabled) so
    downstream loaders can rely on a stable schema.
    """
    active = corruption_fn is not None and bool((corruption_cfg or {}).get("enabled", False))
    return {
        "corruption_type": str((corruption_cfg or {}).get("type", "")) if active else "",
        "severity": int((corruption_cfg or {}).get("severity", 0)) if active else 0,
        "dataset": dataset_name,
    }


def _percentile_or_zero(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def _build_live_status(
    out_dir: Path,
    model_name: str,
    rows: List[Dict[str, Any]],
    latencies_ms: List[float],
    threshold: float,
    metrics: Dict[str, float],
    extras: Dict[str, Any],
    recent_window: int = 12,
    fail_window: int = 8,
) -> Dict[str, Any]:
    """Project the per-model batch run into the streaming-session snapshot shape.

    Keeps the same field names produced by the live runtime so the same loaders
    (notebooks, dashboards) work over both batch and streaming outputs.
    """
    decisions_emitted = len(rows)
    fail_count = sum(1 for row in rows if int(row.get("pred_is_anomaly", 0)) == 1)
    total_latency_s = sum(latencies_ms) / 1000.0
    processed_fps = float(decisions_emitted / total_latency_s) if total_latency_s > 0 else 0.0
    recent_decisions = list(reversed(rows[-recent_window:])) if rows else []
    recent_fails = [row for row in reversed(rows) if int(row.get("pred_is_anomaly", 0)) == 1][:fail_window]

    status: Dict[str, Any] = {
        "session_dir": str(out_dir),
        "active_model": model_name,
        "frames_seen": decisions_emitted,
        "decisions_emitted": decisions_emitted,
        "fail_count": fail_count,
        "input_fps": 0.0,
        "processed_fps": processed_fps,
        "decision_fps": processed_fps,
        "mean_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "p95_latency_ms": _percentile_or_zero(latencies_ms, 95),
        "latency_sla_ms": 0.0,
        "threshold": float(threshold),
        "recent_decisions": recent_decisions,
        "recent_fails": recent_fails,
        "corruption_type": extras["corruption_type"],
        "severity": extras["severity"],
        "dataset": extras["dataset"],
        "auroc": float(metrics.get("auroc", 0.0)),
        "f1": float(metrics.get("f1", 0.0)),
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "aupr": float(metrics.get("aupr", 0.0)),
    }
    return status


def _run_single_model(
    out_dir: Path,
    model_cfg: Dict[str, Any],
    runtime_cfg: Dict[str, Any],
    train_samples: List[Any],
    val_samples: List[Any],
    test_samples: List[Any],
    pre_cfg: Dict[str, Any],
    save_umap: bool,
    dataset_name: str,
    corruption_cfg: Optional[Dict[str, Any]] = None,
    corruption_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    model_name = str(model_cfg.get("name", "unknown"))
    model = build_model(model_cfg, runtime_cfg)
    total_stages = 5

    fit_context = {
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
    }
    train_paths = [s.path for s in train_samples]

    if runtime_cfg["resolved_device"].startswith("cuda"):
        idx = int(runtime_cfg["resolved_device"].split(":")[1])
        torch.cuda.reset_peak_memory_stats(idx)

    _stage_started(
        model_name=model_name,
        stage_idx=1,
        total_stages=total_stages,
        stage_name="fit",
        details=f"train_samples={len(train_paths)}",
    )
    t0 = time.perf_counter()
    model.fit(train_paths, fit_context=fit_context)
    fit_seconds = time.perf_counter() - t0
    _stage_done(
        model_name=model_name,
        stage_idx=1,
        total_stages=total_stages,
        stage_name="fit",
        details=f"time={fit_seconds:.2f}s",
    )

    _stage_started(
        model_name=model_name,
        stage_idx=2,
        total_stages=total_stages,
        stage_name="validation",
        details=f"val_samples={len(val_samples)}",
    )
    val_run = _run_inference(
        model=model,
        model_name=model_name,
        samples=val_samples,
        pre_cfg=pre_cfg,
        collect_embeddings=False,
        stage_name="validation",
    )
    _stage_done(
        model_name=model_name,
        stage_idx=2,
        total_stages=total_stages,
        stage_name="validation",
        details=f"time={float(val_run['predict_seconds']):.2f}s",
    )

    _stage_started(
        model_name=model_name,
        stage_idx=3,
        total_stages=total_stages,
        stage_name="threshold",
    )
    val_rows = val_run["rows"]
    threshold_state = _maybe_calibrate_threshold(model=model, model_cfg=model_cfg, val_rows=val_rows)
    _apply_threshold(val_rows, float(threshold_state["threshold"]))
    val_metrics = _collect_metrics_from_rows(val_rows)
    _stage_done(
        model_name=model_name,
        stage_idx=3,
        total_stages=total_stages,
        stage_name="threshold",
        details=f"mode={threshold_state['mode']}, value={float(threshold_state['threshold']):.6f}",
    )

    _stage_started(
        model_name=model_name,
        stage_idx=4,
        total_stages=total_stages,
        stage_name="test",
        details=f"test_samples={len(test_samples)}",
    )
    test_run = _run_inference(
        model=model,
        model_name=model_name,
        samples=test_samples,
        pre_cfg=pre_cfg,
        collect_embeddings=True,
        stage_name="test",
        corruption_fn=corruption_fn,
    )
    rows = test_run["rows"]
    predict_seconds = float(test_run["predict_seconds"])
    test_latencies_ms = list(test_run["latencies_ms"])
    embeddings = test_run["embeddings"]
    emb_labels = test_run["emb_labels"]
    emb_paths = test_run["emb_paths"]
    emb_scores = test_run["emb_scores"]
    emb_defect_types = test_run["emb_defect_types"]
    confusion_samples = test_run["confusion_samples"]
    _stage_done(
        model_name=model_name,
        stage_idx=4,
        total_stages=total_stages,
        stage_name="test",
        details=f"time={predict_seconds:.2f}s",
    )

    _stage_started(
        model_name=model_name,
        stage_idx=5,
        total_stages=total_stages,
        stage_name="artifacts",
    )
    # Stamp every row with the corruption + dataset identity so the JSON is
    # self-describing; matches the streaming-session predictions.json schema.
    extras = _row_extras(corruption_cfg, corruption_fn, dataset_name)
    for row in rows:
        row.update(extras)
        row.setdefault("heatmap_path", None)
    for row in val_rows:
        row.update(extras)
        row.setdefault("heatmap_path", None)

    pred_path = out_dir / f"predictions_{_safe_name(model_name)}.json"
    pred_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    # Persist the validation rows with the same enriched schema; harmless
    # for clean runs and necessary so the streaming runtime can reuse them.
    (out_dir / f"validation_predictions_{_safe_name(model_name)}.json").write_text(
        json.dumps(val_rows, indent=2),
        encoding="utf-8",
    )

    confusion_sample_metadata: Optional[Dict[str, Any]] = None
    if corruption_fn is not None and bool((corruption_cfg or {}).get("enabled", False)):
        confusion_sample_metadata = _export_corrupted_confusion_samples(
            out_dir=out_dir,
            model_name=model_name,
            candidates=confusion_samples,
            corruption_cfg=corruption_cfg or {},
        )

    if save_umap and len(embeddings) > 0:
        _save_umap(
            out_dir=out_dir,
            model_name=model_name,
            embeddings=embeddings,
            labels=emb_labels,
            paths=emb_paths,
            scores=emb_scores,
            defect_types=emb_defect_types,
        )

    metrics = _collect_metrics_from_rows(rows)

    # Streaming-shape JSON snapshot for this (model, corruption, severity,
    # dataset) tuple. Mirrors the fields that the live runtime publishes in
    # data/streaming_input/<session>/live_status.json so tooling is shared.
    live_status = _build_live_status(
        out_dir=out_dir,
        model_name=model_name,
        rows=rows,
        latencies_ms=test_latencies_ms,
        threshold=float(threshold_state["threshold"]),
        metrics=metrics,
        extras=extras,
    )
    (out_dir / f"live_status_{_safe_name(model_name)}.json").write_text(
        json.dumps(live_status, indent=2), encoding="utf-8"
    )

    peak_vram_mb = 0.0
    if runtime_cfg["resolved_device"].startswith("cuda"):
        idx = int(runtime_cfg["resolved_device"].split(":")[1])
        peak_vram_mb = float(torch.cuda.max_memory_allocated(idx) / (1024 * 1024))

    # Industrial-relevance metrics: prevalence-invariant operating points
    # plus per-defect breakdown so a dominant class can't hide failures
    # on minority classes (Deceuninck scratches are 80% of all bads).
    y_true_test = np.asarray(
        [r["label"] for r in rows if r["label"] in (0, 1)], dtype=np.int32,
    )
    y_score_test = np.asarray(
        [r["score"] for r in rows if r["label"] in (0, 1)], dtype=np.float64,
    )
    recall_at_1 = _recall_at_fpr(y_true_test, y_score_test, 0.01)
    recall_at_5 = _recall_at_fpr(y_true_test, y_score_test, 0.05)
    per_defect = _per_defect_recall(rows)

    n_test = max(1, len(rows))
    summary = {
        "model": model_name,
        "train_samples": len(train_paths),
        "val_samples": len(val_rows),
        "test_samples": len(rows),
        "fit_seconds": round(fit_seconds, 6),
        "val_predict_seconds": round(float(val_run["predict_seconds"]), 6),
        "predict_seconds": round(predict_seconds, 6),
        "ms_per_image": round((predict_seconds / n_test) * 1000.0, 6),
        "fps": round(n_test / predict_seconds, 6) if predict_seconds > 0 else 0.0,
        "peak_vram_mb": round(peak_vram_mb, 3),
        "threshold_mode": threshold_state["mode"],
        "threshold_used": round(float(threshold_state["threshold"]), 6),
        "threshold_calibrated": bool(threshold_state["calibrated"]),
    }
    for key, value in val_metrics.items():
        summary[f"val_{key}"] = value
    summary.update(metrics)
    summary["recall_at_fpr_1pct"] = (
        round(recall_at_1, 6) if recall_at_1 is not None else None
    )
    summary["recall_at_fpr_5pct"] = (
        round(recall_at_5, 6) if recall_at_5 is not None else None
    )
    summary["macro_recall"] = per_defect["macro_recall"]
    summary["weighted_recall"] = per_defect["weighted_recall"]
    summary["per_defect_recall"] = per_defect["per_defect_recall"]
    summary["per_defect_support"] = per_defect["per_defect_support"]
    summary["model_cfg"] = dict(model_cfg)

    # Persist corruption identity per row so block 1.3 can build robustness
    # curves directly from benchmark_summary.json without cross-referencing.
    corr_active = corruption_fn is not None and bool((corruption_cfg or {}).get("enabled", False))
    summary["corruption_type"] = str((corruption_cfg or {}).get("type", "")) if corr_active else ""
    summary["corruption_severity"] = (
        int((corruption_cfg or {}).get("severity", 0)) if corr_active else 0
    )
    if confusion_sample_metadata is not None:
        summary["corrupted_confusion_samples"] = {
            "directory": str(
                Path("corrupted_confusion_samples") / _safe_name(model_name)
            ),
            "available_cases": confusion_sample_metadata["available_cases"],
            "missing_cases": confusion_sample_metadata["missing_cases"],
        }
    _stage_done(
        model_name=model_name,
        stage_idx=5,
        total_stages=total_stages,
        stage_name="artifacts",
    )
    return summary


def run_pipeline(cfg: Dict[str, Any]) -> Path:
    run_cfg = cfg["run"]
    out_root = Path(run_cfg["output_dir"])
    run_id = f'{run_cfg.get("run_name", "run")}_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}'
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(run_cfg.get("seed", 42))
    np.random.seed(seed)

    runtime_cfg = _resolve_runtime(cfg.get("runtime", {}))
    (out_dir / "runtime_info.json").write_text(
        json.dumps(_runtime_info(runtime_cfg), indent=2), encoding="utf-8"
    )

    ds = cfg["dataset"]
    samples = resolve_dataset_labeled(
        ds["source_type"],
        ds["path"],
        ds["extract_dir"],
        dataset_format=ds.get("format"),
        cameras=ds.get("cameras"),
    )
    if len(samples) == 0:
        raise ValueError(
            f"No image files found for dataset source_type='{ds['source_type']}'"
            f" at path='{ds['path']}'."
        )

    split_result = apply_dataset_split(samples, ds.get("split", {}), fallback_seed=seed)
    train_samples = split_result.train
    val_samples = split_result.val
    test_samples = split_result.test

    model_cfgs = _model_cfgs(cfg)
    supported_models = set(available_models())
    invalid = [str(m.get("name", "<missing>")) for m in model_cfgs if m.get("name") not in supported_models]
    if invalid:
        supported = ", ".join(available_models())
        raise ValueError(
            f"Unknown model(s): {invalid}. Supported model names are: {supported}."
        )

    pre_cfg = cfg["preprocessing"]
    corruption_cfg = dict(cfg.get("corruption", {}))
    corruption_fn = _build_corruption_fn(corruption_cfg)
    if corruption_fn is not None:
        print(
            f"[Benchmark] Corruption active on test set: "
            f"type={corruption_cfg['type']}, severity={corruption_cfg['severity']}"
        )

    save_umap = bool(cfg.get("benchmark", {}).get("save_umap", True))
    # Identify the dataset by its config path (basename of the dataset path
    # already encodes Real-IAD vs. Deceuninck for our two configs); falls back
    # to the run_name when no path is available.
    dataset_name = str(Path(str(ds.get("path", ""))).name or run_cfg.get("run_name", "dataset"))
    summary_rows: List[Dict[str, Any]] = []

    for model_cfg in model_cfgs:
        print(f"[Benchmark] Running model: {model_cfg.get('name')}")
        summary = _run_single_model(
            out_dir=out_dir,
            model_cfg=model_cfg,
            runtime_cfg=runtime_cfg,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            pre_cfg=pre_cfg,
            save_umap=save_umap,
            dataset_name=dataset_name,
            corruption_cfg=corruption_cfg,
            corruption_fn=corruption_fn,
        )
        summary_rows.append(summary)

    summary_payload = {
        "run": {
            "seed": seed,
            "output_dir": str(out_root),
            "run_name": str(run_cfg.get("run_name", "run")),
            "run_id": run_id,
        },
        "runtime": dict(runtime_cfg),
        "dataset": dict(ds),
        "preprocessing": dict(pre_cfg),
        "corruption": dict(corruption_cfg),
        "models": summary_rows,
    }
    (out_dir / "benchmark_summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )

    return out_dir
