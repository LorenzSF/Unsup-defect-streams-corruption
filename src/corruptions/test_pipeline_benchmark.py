import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from benchmark_AD.data import LabeledSample, SplitResult
from benchmark_AD.models import ModelOutput
from benchmark_AD.pipeline import run_pipeline


def _write_img(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _write_constant_img(path: Path, value: int) -> None:
    arr = np.full((32, 32, 3), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


class _StubModel:
    def __init__(self, threshold: float):
        self.threshold = float(threshold)

    def fit(self, train_paths, fit_context=None) -> None:
        del train_paths, fit_context

    def predict(self, x: np.ndarray) -> ModelOutput:
        score = float(np.mean(x))
        return ModelOutput(score=score, is_anomaly=score >= self.threshold)

    def get_embedding(self, x: np.ndarray):
        del x
        return None


def test_pipeline_writes_benchmark_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import benchmark_AD.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "available_models", lambda: ["stub_model"])
    monkeypatch.setattr(
        pipeline_module,
        "build_model",
        lambda model_cfg, runtime_cfg: _StubModel(threshold=float(model_cfg.get("threshold", 0.5))),
    )

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
        "model": {"name": "stub_model", "threshold": 0.5},
        "benchmark": {"save_umap": False, "models": [{"name": "stub_model", "threshold": 0.5}]},
        "evaluation": {"metrics": ["precision", "recall", "f1", "accuracy", "auroc", "aupr"]},
    }

    out_dir = run_pipeline(cfg)
    assert out_dir.exists()
    assert (out_dir / "runtime_info.json").exists()
    assert (out_dir / "benchmark_summary.json").exists()
    assert (out_dir / "validation_predictions_stub_model.json").exists()
    assert (out_dir / "predictions_stub_model.json").exists()

    summary = json.loads((out_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary["run"]["run_name"] == "ut"
    assert len(summary["models"]) == 1
    assert summary["models"][0]["model"] == "stub_model"
    assert "val_samples" in summary["models"][0]
    assert "val_accuracy" in summary["models"][0]


def test_pipeline_saves_corrupted_confusion_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import benchmark_AD.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "available_models", lambda: ["stub_model"])
    monkeypatch.setattr(
        pipeline_module,
        "build_model",
        lambda model_cfg, runtime_cfg: _StubModel(
            threshold=float(model_cfg.get("threshold", 0.5))
        ),
    )

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    train_path = dataset / "train_normal.png"
    val_path = dataset / "val_normal.png"
    tp_path = dataset / "part_tp.png"
    fp_path = dataset / "part_fp.png"
    fn_path = dataset / "part_fn.png"
    tn_path = dataset / "part_tn.png"

    _write_constant_img(train_path, 80)
    _write_constant_img(val_path, 80)
    _write_constant_img(tp_path, 140)
    _write_constant_img(fp_path, 130)
    _write_constant_img(fn_path, 124)
    _write_constant_img(tn_path, 100)

    train_sample = LabeledSample(path=train_path, label=0)
    val_sample = LabeledSample(path=val_path, label=0)
    test_samples = [
        LabeledSample(path=tp_path, label=1, defect_type="scratch"),
        LabeledSample(path=fp_path, label=0),
        LabeledSample(path=fn_path, label=1, defect_type="spot"),
        LabeledSample(path=tn_path, label=0),
    ]

    monkeypatch.setattr(
        pipeline_module,
        "resolve_dataset_labeled",
        lambda source_type, path, extract_dir, **kwargs: [train_sample, val_sample, *test_samples],
    )
    monkeypatch.setattr(
        pipeline_module,
        "apply_dataset_split",
        lambda samples, split_cfg, fallback_seed: SplitResult(
            train=[train_sample],
            val=[val_sample],
            test=test_samples,
        ),
    )

    cfg = {
        "run": {"seed": 7, "output_dir": str(tmp_path / "runs"), "run_name": "ut"},
        "runtime": {"device": "cpu", "num_workers": 0, "pin_memory": False},
        "dataset": {
            "source_type": "folder",
            "path": str(dataset),
            "extract_dir": str(dataset),
            "split": {"test_ratio": 0.5, "train_on_good_only": True, "seed": 7},
        },
        "preprocessing": {
            "resize": {"enabled": False},
            "normalize": {"enabled": True, "mode": "0_1"},
        },
        "corruption": {"enabled": True, "type": "gaussian_blur", "severity": 3},
        "model": {
            "name": "stub_model",
            "threshold": 0.5,
            "thresholding": {"mode": "fixed"},
        },
        "benchmark": {
            "save_umap": False,
            "models": [
                {
                    "name": "stub_model",
                    "threshold": 0.5,
                    "thresholding": {"mode": "fixed"},
                }
            ],
        },
        "evaluation": {"metrics": ["precision", "recall", "f1", "accuracy", "auroc", "aupr"]},
    }

    out_dir = run_pipeline(cfg)
    samples_dir = out_dir / "corrupted_confusion_samples" / "stub_model"
    metadata_path = samples_dir / "metadata.json"
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert set(metadata["available_cases"]) == {"TP", "FP", "FN", "TN"}
    assert metadata["missing_cases"] == []
    assert (samples_dir / "part_tp_TP.png").exists()
    assert (samples_dir / "part_fp_FP.png").exists()
    assert (samples_dir / "part_fn_FN.png").exists()
    assert (samples_dir / "part_tn_TN.png").exists()

    summary = json.loads((out_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary["models"][0]["corrupted_confusion_samples"]["available_cases"] == [
        "TP",
        "FP",
        "FN",
        "TN",
    ]
