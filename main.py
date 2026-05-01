from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from src.benchmark import OnlineBaseline
from src.corruption import apply_corruption
from src.metrics import OnlineMetrics
from src.models import build_model, warmup
from src.schemas import RunConfig
from src.stream import build_stream
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


def save_report(report: dict[str, Any], output_dir: str, experiment_name: str) -> Path:
    run_dir = Path(output_dir) / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(_jsonify(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report_path


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


def main() -> None:
    cfg = RunConfig.from_yaml("config.yaml")
    print(f"[main] experiment={cfg.experiment_name} seed={cfg.seed}")

    set_seeds(cfg.seed)

    stream = build_stream(cfg.stream)
    model = build_model(cfg.model)

    print("[main] warming up model...")
    warmup_stream = build_stream(cfg.stream)
    if not cfg.warmup.use_clean_frames:
        warmup_stream = apply_corruption(warmup_stream, cfg.corruption)
    warmup(model, warmup_stream, cfg.warmup)

    metrics = OnlineMetrics(cfg.metrics)
    viz = StreamVisualizer(cfg.visualization)
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
    if baseline is not None:
        report["baseline"] = baseline.snapshot()
    report_path = save_report(report, cfg.output_dir, cfg.experiment_name)
    viz.close()
    print(f"[main] done: {report_path}")


if __name__ == "__main__":
    main()
