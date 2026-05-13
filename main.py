from __future__ import annotations

import dataclasses
import json
import math
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
) -> tuple[float, dict[str, Any]]:
    mode = cfg.metrics.threshold_mode
    if mode == "manual":
        assert cfg.metrics.manual_threshold is not None
        threshold = float(cfg.metrics.manual_threshold)
        return threshold, {"mode": mode, "threshold": threshold}

    if mode == "pot":
        scores_list: list[float] = []
        for frame in warmup_frames:
            pred = model.predict(frame)
            score = float(pred.score)
            if math.isfinite(score):
                scores_list.append(score)
        scores_arr = np.asarray(scores_list, dtype=np.float64)
        if scores_arr.size == 0:
            raise RuntimeError(
                "pot calibration requires at least one finite warmup score"
            )
        threshold, report = _pot_threshold(scores_arr, cfg.metrics.pot_risk)
        report.update({"mode": mode, "threshold": threshold})
        return threshold, report

    raise ValueError(f"unknown threshold_mode {mode!r}")


def _pot_threshold(
    scores: np.ndarray, pot_risk: float
) -> tuple[float, dict[str, Any]]:
    """Siffer et al. 2017, KDD, §3.2. Fit a Generalized Pareto to the upper
    tail of the warm-up scores and derive the threshold at target risk
    `pot_risk` (false positive rate)."""
    init_q = 0.98
    u = float(np.quantile(scores, init_q))
    tail = scores[scores > u] - u
    if tail.size < 10:
        raise RuntimeError(
            f"pot calibration: only {tail.size} exceedances above q={init_q} "
            f"(need >= 10); increase warmup.warmup_steps"
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
        "n_warmup_scores": n,
    }


def _collect_warmup_embeddings(
    model, warmup_frames: List[Frame], enabled: bool
) -> "np.ndarray | None":
    """Re-score the warmup frames to harvest their per-frame embeddings.

    The dashboard's reference cloud (gray dots in the 2D scatter) is the
    PCA(2) projection of these vectors. The model is already fitted at
    this point — this second pass exists because embeddings are emitted
    by `predict`, not by `fit_warmup`. Returns None when the dashboard is
    disabled, when no detector emits an embedding, or when embeddings
    have inconsistent dimensions across frames.
    """
    if not enabled:
        return None
    vecs: list[np.ndarray] = []
    for frame in warmup_frames:
        pred = model.predict(frame)
        if pred.embedding is None:
            continue
        emb = np.asarray(pred.embedding, dtype=np.float32).reshape(-1)
        if emb.size == 0:
            continue
        vecs.append(emb)
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
    warmup_embeddings = _collect_warmup_embeddings(
        model, warmup_frames, cfg.visualization.dashboard_enabled
    )
    viz = StreamVisualizer(
        cfg.visualization, run_dir, threshold, warmup_embeddings
    )

    corrupted = apply_corruption(stream, cfg.corruption)

    print("[main] starting streaming inference loop")
    with FrameLogger(run_dir / "frames.jsonl") as frames_log:
        for frame in corrupted:
            pred = model.predict(frame)
            metrics.update(frame, pred)
            viz.render(frame, pred, metrics.snapshot())
            frames_log.write(frame, pred)
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
    report_path = save_report(report, run_dir)
    viz.close()
    print(f"[main] done: {report_path}")


if __name__ == "__main__":
    main()
