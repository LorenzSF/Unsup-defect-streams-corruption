from __future__ import annotations

import dataclasses
import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, List

import numpy as np

from scipy.stats import genpareto

from src.corruption import apply_corruption
from src.metrics import FrameLogger, OnlineMetrics
from src.models import build_model
from src.schemas import Frame, RunConfig
from src.stream import build_stream, build_warmup_stream, warmup
from src.visualization import StreamVisualizer, prediction_projection_vector


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
    input_name = Path(cfg.stream.input_path).name
    if input_name:
        parts.append(input_name)
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


def _collect_hardware_info(model_device: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cpu": {
            "logical_cores": os.cpu_count(),
        },
        "accelerator": {
            "requested_device": model_device,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_devices": [],
        },
    }

    try:
        import torch
    except ImportError:
        info["accelerator"]["torch_available"] = False
        return info

    info["accelerator"]["torch_available"] = True
    info["accelerator"]["torch_version"] = torch.__version__

    try:
        cuda_available = bool(torch.cuda.is_available())
        cuda_devices: list[dict[str, Any]] = []
        if cuda_available:
            for device_index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(device_index)
                cuda_devices.append(
                    {
                        "index": device_index,
                        "name": props.name,
                        "total_memory_mb": float(
                            props.total_memory / (1024.0 * 1024.0)
                        ),
                        "compute_capability": [
                            int(props.major),
                            int(props.minor),
                        ],
                        "multiprocessor_count": int(props.multi_processor_count),
                    }
                )
        info["accelerator"].update(
            {
                "cuda_available": cuda_available,
                "cuda_device_count": len(cuda_devices),
                "cuda_devices": cuda_devices,
            }
        )
    except (RuntimeError, AttributeError) as exc:
        info["accelerator"]["probe_error"] = f"{type(exc).__name__}: {exc}"

    return info


def _calibrate_threshold(
    cfg: RunConfig,
    scores: List[float],
) -> tuple[float, dict[str, Any]]:
    mode = cfg.metrics.threshold_mode
    if mode == "manual":
        assert cfg.metrics.manual_threshold is not None
        threshold = float(cfg.metrics.manual_threshold)
        return threshold, {"mode": mode, "threshold": threshold}

    if mode == "pot":
        scores_arr = np.asarray(scores, dtype=np.float64)
        if scores_arr.size == 0:
            raise RuntimeError(
                "pot calibration requires at least one finite calibration score"
            )
        threshold, report = _pot_threshold(scores_arr, cfg.metrics.pot_risk)
        report.update({"mode": mode, "threshold": threshold})
        return threshold, report

    raise ValueError(f"unknown threshold_mode {mode!r}")


def _pot_threshold(
    scores: np.ndarray, pot_risk: float
) -> tuple[float, dict[str, Any]]:
    """Siffer et al. 2017, KDD, §3.2. Fit a Generalized Pareto to the upper 
    tail of the calibration scores and derive the threshold at target risk
    `pot_risk` (false positive rate)."""
    init_q = 0.98
    u = float(np.quantile(scores, init_q))
    tail = scores[scores > u] - u
    if tail.size < 10:
        raise RuntimeError(
            f"pot calibration: only {tail.size} exceedances above q={init_q} "
            f"(need >= 10); increase metrics.calibration_steps"
        )
    ksi, _, sigma = genpareto.fit(tail, floc=0.0)
    n, n_u = int(scores.size), int(tail.size)
    ratio = (n / n_u) * pot_risk
    if abs(ksi) < 1e-9:
        threshold = u - float(sigma) * math.log(ratio)
    else:
        threshold = u + (float(sigma) / float(ksi)) * (ratio ** (-float(ksi)) - 1.0)
    return float(threshold), {
        "pot_risk": float(pot_risk),
        "pot_init_quantile": init_q,
        "pot_u": u,
        "pot_ksi": float(ksi),
        "pot_sigma": float(sigma),
        "pot_n_tail": n_u,
        "n_calibration_scores": n,
    }


def _collect_warmup_projection_vectors(
    model, warmup_frames: List[Frame], enabled: bool
) -> "np.ndarray | None":
    """Re-score warmup frames to harvest vectors for dashboard PCA.

    The dashboard's reference cloud is the StandardScaler + PCA(2)
    projection of these vectors. The model is already fitted at this point;
    this second pass exists because vectors depend on `predict` outputs.
    Returns None when the dashboard is disabled or vector dimensions are
    inconsistent across frames.
    """
    if not enabled:
        return None
    vecs: list[np.ndarray] = []
    for frame in warmup_frames:
        pred = model.predict(frame)
        vec = prediction_projection_vector(pred)
        if vec is None:
            continue
        vecs.append(vec)
    if len(vecs) < 2:
        return None
    sizes = {v.size for v in vecs}
    if len(sizes) != 1:
        return None
    return np.stack(vecs, axis=0)


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
    warmup_frames = warmup(model, warmup_stream, cfg.warmup)
    cold_start_s = time.perf_counter() - cold_start_t0

    if cfg.metrics.threshold_mode == "manual":
        assert cfg.metrics.manual_threshold is not None
        active_threshold = float(cfg.metrics.manual_threshold)
        threshold_report: dict[str, Any] = {
            "mode": "manual",
            "threshold": active_threshold,
        }
        print(f"[main] threshold ready: mode=manual value={active_threshold:.6f}")
    else:
        active_threshold = float(cfg.metrics.initial_threshold)
        threshold_report = {
            "mode": "pot",
            "initial_threshold": active_threshold,
            "status": "calibrating",
            "calibration_steps": cfg.metrics.calibration_steps,
            "switch_frame": None,
        }
        print(
            "[main] threshold initializing: "
            f"mode=pot initial={active_threshold:.6f} "
            f"calibration_steps={cfg.metrics.calibration_steps}"
        )

    resolved_metrics_cfg = dataclasses.replace(
        cfg.metrics, threshold_value=active_threshold
    )
    set_seeds(cfg.seed)
    stream = build_stream(cfg.stream, cfg.warmup.warmup_steps)

    metrics = OnlineMetrics(resolved_metrics_cfg)
    warmup_projection_vectors = _collect_warmup_projection_vectors(
        model, warmup_frames, cfg.visualization.dashboard_enabled
    )
    viz = StreamVisualizer(
        cfg.visualization,
        run_dir,
        active_threshold,
        cfg.model.name,
        warmup_projection_vectors,
    )

    corrupted = apply_corruption(stream, cfg.corruption)
    calibration_scores: list[float] = []
    calibration_seen = 0
    pot_ready = cfg.metrics.threshold_mode != "pot"

    print("[main] starting streaming inference loop")
    with FrameLogger(run_dir / "frames.jsonl") as frames_log:
        for frame in corrupted:
            pred = model.predict(frame)
            threshold_used = active_threshold
            metrics.update(frame, pred, threshold_used)
            viz.render(frame, pred, metrics.snapshot())
            frames_log.write(frame, pred, threshold_used)
            if frame.index % cfg.log_every == 0:
                print(f"[step {frame.index}] {metrics.snapshot()}")

            if not pot_ready:
                calibration_seen += 1
                score = float(pred.score)
                if math.isfinite(score):
                    calibration_scores.append(score)
                if calibration_seen >= cfg.metrics.calibration_steps:
                    active_threshold, threshold_report = _calibrate_threshold(
                        cfg, calibration_scores
                    )
                    threshold_report.update(
                        {
                            "status": "calibrated",
                            "initial_threshold": float(cfg.metrics.initial_threshold),
                            "calibration_steps": cfg.metrics.calibration_steps,
                            "calibration_seen": calibration_seen,
                            "switch_frame": int(frame.index) + 1,
                        }
                    )
                    metrics.set_threshold(active_threshold)
                    viz.set_threshold(active_threshold)
                    pot_ready = True
                    print(
                        "[main] threshold switched: "
                        f"mode=pot value={active_threshold:.6f} "
                        f"after_frame={frame.index}"
                    )

    if not pot_ready:
        threshold_report.update(
            {
                "status": "incomplete",
                "calibration_seen": calibration_seen,
                "n_calibration_scores": len(calibration_scores),
                "threshold": active_threshold,
            }
        )

    report = metrics.finalize()
    if (
        torch is not None
        and torch.cuda.is_available()
        and cfg.model.device.startswith("cuda")
    ):
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
    report["hardware"] = _collect_hardware_info(cfg.model.device)
    report["stream"] = dataclasses.asdict(cfg.stream)
    report["warmup"] = dataclasses.asdict(cfg.warmup)
    report["model"] = dataclasses.asdict(cfg.model)
    report["corruption"] = dataclasses.asdict(cfg.corruption)
    report_path = save_report(report, run_dir)
    viz.close()
    print(f"[main] done: {report_path}")


if __name__ == "__main__":
    main()
