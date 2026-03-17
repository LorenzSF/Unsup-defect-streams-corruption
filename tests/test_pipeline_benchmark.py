import json
from pathlib import Path

import numpy as np
from PIL import Image

from real_time_visual_defect_detection.pipelines.run_pipeline import run_pipeline


def _write_img(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_pipeline_writes_benchmark_outputs(tmp_path: Path):
    dataset = tmp_path / "dataset"
    good_dir = dataset / "good"
    bad_dir = dataset / "bad"
    good_dir.mkdir(parents=True)
    bad_dir.mkdir(parents=True)

    for idx in range(4):
        _write_img(good_dir / f"good_{idx}.png", idx)
    for idx in range(2):
        _write_img(bad_dir / f"bad_{idx}.png", idx + 100)

    cfg = {
        "run": {"seed": 7, "output_dir": str(tmp_path / "runs"), "run_name": "ut"},
        "runtime": {"device": "cpu", "num_workers": 0, "pin_memory": False},
        "dataset": {
            "source_type": "folder",
            "path": str(dataset),
            "extract_dir": str(dataset),
            "split": {"test_ratio": 0.5, "train_on_good_only": True, "seed": 7},
        },
        "preprocessing": {"resize": {"enabled": False}, "normalize": {"enabled": True, "mode": "0_1"}},
        "corruption": {"enabled": False, "type": "gaussian_noise", "params": {}},
        "model": {"name": "dummy_distance", "threshold": 0.5},
        "benchmark": {"save_umap": False, "models": [{"name": "dummy_distance", "threshold": 0.5}]},
        "evaluation": {"metrics": ["precision", "recall", "f1", "accuracy", "auroc", "aupr"]},
    }

    out_dir = run_pipeline(cfg)
    assert out_dir.exists()
    assert (out_dir / "runtime_info.json").exists()
    assert (out_dir / "benchmark_summary.json").exists()
    assert (out_dir / "benchmark_summary.csv").exists()
    assert (out_dir / "validation_predictions_dummy_distance.json").exists()
    assert (out_dir / "predictions_dummy_distance.json").exists()
    assert (out_dir / "predictions.json").exists()

    summary = json.loads((out_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert len(summary) == 1
    assert summary[0]["model"] == "dummy_distance"
    assert "val_samples" in summary[0]
    assert "val_accuracy" in summary[0]
