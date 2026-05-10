from __future__ import annotations

import dataclasses
import json
import math
import random
import time
from pathlib import Path
from typing import Any, List

import numpy as np

from src.benchmark import OnlineBaseline
from src.corruption import apply_corruption
from src.metrics import OnlineMetrics
from src.models import build_model
from src.schemas import Frame, RunConfig
from src.stream import (
    build_calibration_stream,
    build_stream,
    build_warmup_stream,
    warmup,
)
from src.visualization import StreamVisualizer


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_run_dir(output_dir: str, experiment_name: str) -> Path:
    run_dir = Path(output_dir) / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_report(report: dict[str, Any], run_dir: Path) -> Path:
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(_jsonify(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report_path


def _derive_experiment_name(cfg: RunConfig) -> str:
    parts = [cfg.model.name, cfg.stream.dataset]
    if cfg.stream.category:
        parts.append(cfg.stream.category)
    if cfg.corruption.enabled and cfg.corruption.specs:
        for spec in cfg.corruption.specs:
            parts.append(f"{spec.kind}_s{spec.severity}")
    parts.append(time.strftime("%Y%m%d-%H%M%S"))
    return "_".join(parts)


def _jsonify(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {k: _jsonify(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _calibrate_threshold(
    cfg: RunConfig,
    model,
    warmup_frames: List[Frame],
    calibration_frames: List[Frame] | None = None,
) -> tuple[float, dict[str, Any]]:
    mode = cfg.metrics.threshold_mode
    if mode == "manual":
        assert cfg.metrics.manual_threshold is not None
        threshold = float(cfg.metrics.manual_threshold)
        return threshold, {
            "mode": mode,
            "threshold": threshold,
            "n_calibration_frames": 0,
        }

    if mode == "f1_optimal":
        if not calibration_frames:
            raise RuntimeError(
                "f1_optimal threshold calibration requires a non-empty "
                "calibration_frames list (held-out OK + NG split)"
            )
        scores_list: list[float] = []
        labels_list: list[int] = []
        for frame in calibration_frames:
            pred = model.predict(frame)
            score = float(pred.score)
            if not math.isfinite(score):
                continue
            scores_list.append(score)
            labels_list.append(int(frame.label))
        scores_arr = np.asarray(scores_list, dtype=np.float64)
        labels_arr = np.asarray(labels_list, dtype=np.int64)
        n_ok = int((labels_arr == 0).sum())
        n_ng = int((labels_arr == 1).sum())
        if n_ok == 0 or n_ng == 0:
            raise RuntimeError(
                "f1_optimal threshold calibration requires at least one OK "
                f"and one NG frame in the calibration set, got n_ok={n_ok} "
                f"n_ng={n_ng} (n_total={scores_arr.size})"
            )
        threshold, best_f1 = _f1_optimal_threshold(scores_arr, labels_arr)
        return threshold, {
            "mode": mode,
            "threshold": threshold,
            "n_calibration_frames": int(scores_arr.size),
            "n_calibration_ok": n_ok,
            "n_calibration_ng": n_ng,
            "calibration_f1": best_f1,
        }

    raise ValueError(f"unknown threshold_mode {mode!r}")


def _f1_optimal_threshold(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    """Pick the threshold that maximizes binary F1 on (scores, labels).

    Sweeps every score value as a candidate cut. Ties are broken by
    picking the smallest threshold (most permissive). Used by
    `threshold_mode='f1_optimal'`.
    """
    candidates = np.unique(scores)
    best_f1 = -1.0
    best_thr = float(np.median(scores))
    for thr in candidates:
        pred = (scores >= float(thr)).astype(np.int64)
        tp = int(np.sum((pred == 1) & (labels == 1)))
        fp = int(np.sum((pred == 1) & (labels == 0)))
        fn = int(np.sum((pred == 0) & (labels == 1)))
        if tp + fp == 0 or tp + fn == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        denom = precision + recall
        f1 = 0.0 if denom == 0.0 else 2.0 * precision * recall / denom
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, float(best_f1)


def main() -> None:
    cfg = RunConfig.from_yaml("config.yaml")
    experiment_name = _derive_experiment_name(cfg)
    run_dir = build_run_dir(cfg.output_dir, experiment_name)
    print(f"[main] experiment={experiment_name} seed={cfg.seed}")

    set_seeds(cfg.seed)
    model = build_model(cfg.model)

    print("[main] warming up model...")
    peak_vram_mb = 0.0
    cold_start_t0 = time.perf_counter()
    try:
        import torch

        if torch.cuda.is_available() and cfg.model.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None
    set_seeds(cfg.seed)
    warmup_stream = build_warmup_stream(cfg.stream, cfg.warmup.warmup_steps)
    if not cfg.warmup.use_clean_frames:
        warmup_stream = apply_corruption(warmup_stream, cfg.corruption)
    warmup_frames = warmup(model, warmup_stream, cfg.warmup)

    calibration_frames: List[Frame] | None = None
    if cfg.metrics.threshold_mode == "f1_optimal":
        print(
            f"[main] collecting calibration set: "
            f"{cfg.metrics.calibration_ok} OK + {cfg.metrics.calibration_ng} NG..."
        )
        set_seeds(cfg.seed)
        calibration_stream = build_calibration_stream(
            cfg.stream,
            cfg.warmup.warmup_steps,
            cfg.metrics.calibration_ok,
            cfg.metrics.calibration_ng,
        )
        if not cfg.warmup.use_clean_frames:
            calibration_stream = apply_corruption(calibration_stream, cfg.corruption)
        calibration_frames = list(calibration_stream)

    print(f"[main] calibrating threshold mode={cfg.metrics.threshold_mode}...")
    threshold, threshold_report = _calibrate_threshold(
        cfg, model, warmup_frames, calibration_frames
    )
    cold_start_s = time.perf_counter() - cold_start_t0
    print(
        "[main] threshold ready: "
        f"mode={threshold_report['mode']} value={threshold:.6f}"
    )

    resolved_metrics_cfg = dataclasses.replace(cfg.metrics, threshold_value=threshold)
    set_seeds(cfg.seed)
    stream = build_stream(
        cfg.stream,
        cfg.warmup.warmup_steps,
        cfg.metrics.calibration_ok,
        cfg.metrics.calibration_ng,
    )

    metrics = OnlineMetrics(resolved_metrics_cfg)
    viz = StreamVisualizer(cfg.visualization, run_dir)
    baseline = OnlineBaseline(cfg.benchmark) if cfg.benchmark.enabled else None

    corrupted = apply_corruption(stream, cfg.corruption)

    print("[main] starting streaming inference loop")
    for frame in corrupted:
        pred = model.predict(frame)
        metrics.update(frame, pred)
        viz.render(frame, pred, metrics.snapshot())
        if baseline is not None:
            baseline.update(frame, pred)
        if frame.index % cfg.log_every == 0:
            print(f"[step {frame.index}] {metrics.snapshot()}")

    report = metrics.finalize()
    if torch is not None and torch.cuda.is_available() and cfg.model.device.startswith("cuda"):
        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    report["runtime"] = {
        "cold_start_s": cold_start_s,
        "peak_vram_mb": peak_vram_mb,
    }
    report["threshold"] = threshold_report
    report["run"] = {
        "experiment_name": experiment_name,
        "seed": cfg.seed,
    }
    report["stream"] = dataclasses.asdict(cfg.stream)
    report["warmup"] = dataclasses.asdict(cfg.warmup)
    report["model"] = dataclasses.asdict(cfg.model)
    report["corruption"] = dataclasses.asdict(cfg.corruption)
    if baseline is not None:
        report["baseline"] = baseline.snapshot()
    report_path = save_report(report, run_dir)
    viz.close()
    print(f"[main] done: {report_path}")


if __name__ == "__main__":
    main()
