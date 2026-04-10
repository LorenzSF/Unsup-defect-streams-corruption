from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmark_AD.data import LabeledSample
from inspection_realtime.contracts import FramePacket, ModelPrediction
from inspection_realtime.decision_engine import DecisionEngine
from inspection_realtime.input_handler import _interleave_by_label
from inspection_realtime.object_classifier import RelativeObjectClassifier
from inspection_realtime.report_generator import RuntimeOutputWriter
from inspection_realtime.state_manager import RuntimeStateManager
from inspection_realtime.transition_detector import TransitionDetector


def _frame(image: np.ndarray, frame_id: int = 0) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        timestamp_utc="2026-01-01T00:00:00+00:00",
        path=Path("sample.png"),
        raw_image_bgr=image,
        model_input=image.astype(np.float32) / 255.0,
        label=0,
        defect_type=None,
    )


def test_transition_detector_confirms_change_after_consecutive_frames():
    classifier = RelativeObjectClassifier(distance_threshold=0.05, reference_bank_size=4)
    detector = TransitionDetector(consecutive_different_frames=2, min_confidence=0.1)

    base = np.zeros((32, 32, 3), dtype=np.uint8)
    changed = np.full((32, 32, 3), 255, dtype=np.uint8)

    classifier.reset_with_frame(base)
    match_same = classifier.classify(base)
    assert detector.update(match_same) is False

    match_diff = classifier.classify(changed)
    assert detector.update(match_diff) is False
    assert detector.update(match_diff) is True


def test_state_manager_enters_production_after_min_frames():
    manager = RuntimeStateManager(
        min_calibration_frames=3,
        baseline_sample_count=10,
        require_baseline=True,
    )
    assert manager.state.value == "CALIBRATION"

    manager.observe_calibration_score(0.1)
    manager.observe_calibration_score(0.2)
    assert manager.state.value == "CALIBRATION"

    manager.observe_calibration_score(0.3)
    assert manager.state.value == "PRODUCTION"


def test_decision_engine_keeps_baseline_prediction_shape():
    engine = DecisionEngine()
    frame = _frame(np.zeros((16, 16, 3), dtype=np.uint8))
    prediction = ModelPrediction(
        model_name="stub_model",
        score=0.75,
        pred_is_anomaly=1,
        heatmap=np.ones((16, 16), dtype=np.float32),
    )

    record = engine.decide(frame=frame, prediction=prediction)
    row = record.to_dict()
    assert set(row) == {
        "model",
        "path",
        "label",
        "defect_type",
        "score",
        "pred_is_anomaly",
        "heatmap_path",
    }
    assert row["pred_is_anomaly"] == 1


def test_output_writer_saves_predictions_and_heatmap(tmp_path: Path):
    writer = RuntimeOutputWriter(session_dir=tmp_path)
    image = np.full((24, 24, 3), 100, dtype=np.uint8)
    frame = _frame(image=image, frame_id=7)
    prediction = ModelPrediction(
        model_name="stub_model",
        score=0.9,
        pred_is_anomaly=1,
        heatmap=np.ones((24, 24), dtype=np.float32),
    )

    heatmap_path = writer.save_heatmap(frame=frame, prediction=prediction)
    record = DecisionEngine().decide(frame=frame, prediction=prediction)
    record.heatmap_path = heatmap_path
    writer.append_decision(record)

    assert heatmap_path is not None
    assert (tmp_path / heatmap_path).exists()
    assert writer.predictions_path.exists()


def test_interleave_by_label_mixes_good_bad_and_unlabeled():
    samples = [
        LabeledSample(path=Path("good_1.png"), label=0),
        LabeledSample(path=Path("good_2.png"), label=0),
        LabeledSample(path=Path("bad_1.png"), label=1),
        LabeledSample(path=Path("bad_2.png"), label=1),
        LabeledSample(path=Path("unknown_1.png"), label=-1),
    ]

    ordered = _interleave_by_label(samples)
    ordered_paths = [sample.path.name for sample in ordered]
    assert ordered_paths == [
        "good_1.png",
        "bad_1.png",
        "unknown_1.png",
        "good_2.png",
        "bad_2.png",
    ]
