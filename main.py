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
from src.stream import build_stream, build_warmup_stream, warmup
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
    cfg: RunConfig, model, warmup_frames: List[Frame]
) -> tuple[float, dict[str, Any]]:
    mode = cfg.metrics.threshold_mode
    if mode == "manual":
        assert cfg.metrics.manual_threshold is not None
        threshold = float(cfg.metrics.manual_threshold)
        return threshold, {
            "mode": mode,
            "threshold": threshold,
            "n_calibration_frames": 0,
            "quantile": None,
        }

    if mode == "quantile":
        scores: list[float] = []
        for frame in warmup_frames:
            pred = model.predict(frame)
            score = float(pred.score)
            if math.isfinite(score):
                scores.append(score)
        if not scores:
            raise RuntimeError(
                "quantile threshold calibration: no finite scores from warmup frames"
            )
        q = cfg.metrics.threshold_quantile
        threshold = float(np.quantile(np.asarray(scores, dtype=np.float64), q))
        return threshold, {
            "mode": mode,
            "threshold": threshold,
            "n_calibration_frames": len(scores),
            "quantile": q,
        }

    raise ValueError(f"unknown threshold_mode {mode!r}")


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

    print(f"[main] calibrating threshold mode={cfg.metrics.threshold_mode}...")
    threshold, threshold_report = _calibrate_threshold(cfg, model, warmup_frames)
    cold_start_s = time.perf_counter() - cold_start_t0
    print(
        "[main] threshold ready: "
        f"mode={threshold_report['mode']} value={threshold:.6f}"
    )

    resolved_metrics_cfg = dataclasses.replace(cfg.metrics, threshold_value=threshold)
    set_seeds(cfg.seed)
    stream = build_stream(cfg.stream, cfg.warmup.warmup_steps)

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
