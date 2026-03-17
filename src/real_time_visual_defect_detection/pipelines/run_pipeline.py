from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from tqdm.auto import tqdm

from real_time_visual_defect_detection.evaluation.metrics import compute_binary_metrics
from real_time_visual_defect_detection.io.dataset_loader import (
    apply_dataset_split,
    resolve_dataset_labeled,
)
from real_time_visual_defect_detection.models.registry import available_models, build_model
from real_time_visual_defect_detection.preprocessing.corruption import apply_corruption
from real_time_visual_defect_detection.preprocessing.standard import (
    normalize_0_1,
    read_image_bgr,
    resize,
)


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
    from real_time_visual_defect_detection.visualization.plots import plot_embedding_umap

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


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def _run_inference(
    model: Any,
    model_name: str,
    samples: List[Any],
    pre_cfg: Dict[str, Any],
    corr_cfg: Dict[str, Any],
    collect_embeddings: bool,
    stage_name: str,
) -> Dict[str, Any]:
    do_resize = bool(pre_cfg.get("resize", {}).get("enabled", False))
    w = int(pre_cfg.get("resize", {}).get("width", 256))
    h = int(pre_cfg.get("resize", {}).get("height", 256))
    do_corr = bool(corr_cfg.get("enabled", False))
    corr_type = str(corr_cfg.get("type", ""))
    corr_params = corr_cfg.get("params", {})

    rows: List[Dict[str, Any]] = []
    embeddings: List[np.ndarray] = []
    emb_labels: List[int] = []
    emb_paths: List[str] = []
    emb_scores: List[float] = []
    emb_defect_types: List[Optional[str]] = []
    allow_embeddings = collect_embeddings
    predict_seconds = 0.0

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
        if do_corr:
            img = apply_corruption(img, corr_type, corr_params)
        x = normalize_0_1(img)

        t_pred0 = time.perf_counter()
        out = model.predict(x)
        predict_seconds += time.perf_counter() - t_pred0

        rows.append(
            {
                "model": model_name,
                "path": str(sample.path),
                "label": sample.label,
                "defect_type": sample.defect_type,
                "score": float(out.score),
                "pred_is_anomaly": int(out.is_anomaly),
            }
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
        "embeddings": embeddings,
        "emb_labels": emb_labels,
        "emb_paths": emb_paths,
        "emb_scores": emb_scores,
        "emb_defect_types": emb_defect_types,
    }


def _run_single_model(
    out_dir: Path,
    model_cfg: Dict[str, Any],
    runtime_cfg: Dict[str, Any],
    train_samples: List[Any],
    val_samples: List[Any],
    test_samples: List[Any],
    pre_cfg: Dict[str, Any],
    corr_cfg: Dict[str, Any],
    save_umap: bool,
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
        corr_cfg=corr_cfg,
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
    (out_dir / f"validation_predictions_{_safe_name(model_name)}.json").write_text(
        json.dumps(val_rows, indent=2),
        encoding="utf-8",
    )
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
        corr_cfg=corr_cfg,
        collect_embeddings=True,
        stage_name="test",
    )
    rows = test_run["rows"]
    predict_seconds = float(test_run["predict_seconds"])
    embeddings = test_run["embeddings"]
    emb_labels = test_run["emb_labels"]
    emb_paths = test_run["emb_paths"]
    emb_scores = test_run["emb_scores"]
    emb_defect_types = test_run["emb_defect_types"]
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
    pred_path = out_dir / f"predictions_{_safe_name(model_name)}.json"
    pred_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

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

    peak_vram_mb = 0.0
    if runtime_cfg["resolved_device"].startswith("cuda"):
        idx = int(runtime_cfg["resolved_device"].split(":")[1])
        peak_vram_mb = float(torch.cuda.max_memory_allocated(idx) / (1024 * 1024))

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

    # Persist the exact runtime config with resolved device for reproducibility.
    cfg_snapshot = dict(cfg)
    cfg_snapshot["runtime"] = dict(runtime_cfg)
    (out_dir / "config_snapshot.json").write_text(
        json.dumps(cfg_snapshot, indent=2), encoding="utf-8"
    )

    ds = cfg["dataset"]
    samples = resolve_dataset_labeled(ds["source_type"], ds["path"], ds["extract_dir"])
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
    corr_cfg = cfg["corruption"]
    save_umap = bool(cfg.get("benchmark", {}).get("save_umap", True))
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
            corr_cfg=corr_cfg,
            save_umap=save_umap,
        )
        summary_rows.append(summary)

    if len(model_cfgs) == 1:
        single_name = _safe_name(str(model_cfgs[0]["name"]))
        src = out_dir / f"predictions_{single_name}.json"
        (out_dir / "predictions.json").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    (out_dir / "benchmark_summary.json").write_text(
        json.dumps(summary_rows, indent=2), encoding="utf-8"
    )
    _write_csv(out_dir / "benchmark_summary.csv", summary_rows)

    return out_dir
