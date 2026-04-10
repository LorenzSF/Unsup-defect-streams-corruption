from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import load_runtime_artifact
from .settings import resolve_runtime_settings
from .decision_engine import DecisionEngine
from .input_handler import FolderInputHandler
from .live_metrics import LiveMetrics
from .model_registry import SingleModelRegistry
from .object_classifier import RelativeObjectClassifier
from .predictor import Predictor
from .report_generator import RuntimeOutputWriter
from .state_manager import RuntimeStateManager
from .transition_detector import TransitionDetector
from .webapp import LiveDashboardServer


class InspectionRuntimeApp:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = resolve_runtime_settings(cfg)

    def run(self) -> Path:
        artifact = load_runtime_artifact(
            run_dir=self.cfg["artifact"]["resolved_run_dir"],
            model_name=str(self.cfg["artifact"]["model_name"]),
        )

        session_dir = self._create_session_dir()
        writer = RuntimeOutputWriter(session_dir=session_dir)
        metrics = LiveMetrics(
            session_dir=session_dir,
            input_fps=float(self.cfg["run"]["target_fps"]),
            latency_sla_ms=float(self.cfg["run"]["latency_sla_ms"]),
        )
        state_manager = RuntimeStateManager(
            min_calibration_frames=int(self.cfg["calibration"]["min_frames"]),
            baseline_sample_count=int(artifact.baseline.sample_count),
            require_baseline=bool(self.cfg["calibration"]["require_baseline"]),
        )
        object_classifier = RelativeObjectClassifier(
            embedding_size=int(self.cfg["object_change"]["embedding_size"]),
            distance_threshold=float(self.cfg["object_change"]["distance_threshold"]),
            reference_bank_size=int(self.cfg["object_change"]["reference_bank_size"]),
        )
        transition_detector = TransitionDetector(
            consecutive_different_frames=int(
                self.cfg["object_change"]["consecutive_different_frames"]
            ),
            min_confidence=float(self.cfg["object_change"]["min_confidence"]),
        )
        decision_engine = DecisionEngine()

        registry = SingleModelRegistry(
            artifact=artifact,
            fit_policy=str(self.cfg["artifact"]["fit_policy"]),
        )
        model = registry.load()
        predictor = Predictor(model_name=artifact.model_name, model=model)

        resize_wh = None
        if bool(self.cfg["preprocessing"]["resize"]["enabled"]):
            resize_wh = (
                int(self.cfg["preprocessing"]["resize"]["width"]),
                int(self.cfg["preprocessing"]["resize"]["height"]),
            )

        input_handler = FolderInputHandler(
            root_dir=Path(self.cfg["input"]["root_dir"]),
            resize_wh=resize_wh,
            normalize=bool(self.cfg["preprocessing"]["normalize"]["enabled"]),
            loop=bool(self.cfg["input"]["loop"]),
            max_frames=self.cfg["run"]["max_frames"],
            sequence_mode=str(self.cfg["input"]["sequence_mode"]),
        )

        status_provider = lambda: metrics.snapshot(state_manager=state_manager, artifact=artifact)
        web_server = self._start_web_server(session_dir=session_dir, status_provider=status_provider)
        writer.write_status(status_provider())

        target_interval = 0.0
        target_fps = float(self.cfg["run"]["target_fps"])
        if target_fps > 0.0:
            target_interval = 1.0 / target_fps

        try:
            for frame in input_handler.iter_frames():
                frame_started = time.perf_counter()
                metrics.record_frame_seen()

                if not object_classifier.has_reference:
                    object_classifier.reset_with_frame(frame.raw_image_bgr)

                match = object_classifier.classify(frame.raw_image_bgr)
                if transition_detector.update(match):
                    object_classifier.reset_with_frame(frame.raw_image_bgr)
                    transition_detector.reset()
                    state_manager.reset_for_new_object()
                    metrics.record_transition()
                    match = object_classifier.classify(frame.raw_image_bgr)

                metrics.record_match(match)

                prediction, latency_ms = predictor.predict(frame)
                metrics.record_latency(latency_ms)

                if state_manager.state.value == "CALIBRATION":
                    object_classifier.update_reference(frame.raw_image_bgr)
                    state_manager.observe_calibration_score(prediction.score)
                    metrics.record_no_decision()
                elif not match.is_same_reference:
                    metrics.record_no_decision()
                else:
                    object_classifier.update_reference(frame.raw_image_bgr)
                    decision = decision_engine.decide(frame=frame, prediction=prediction)
                    decision.heatmap_path = writer.save_heatmap(frame=frame, prediction=prediction)
                    writer.append_decision(decision)
                    metrics.record_decision(decision)

                writer.write_status(status_provider())
                self._sleep_to_target(started=frame_started, target_interval=target_interval)
        finally:
            if web_server is not None:
                web_server.stop()

        return session_dir

    def _create_session_dir(self) -> Path:
        output_dir = Path(self.cfg["run"]["output_dir"])
        session_name = str(self.cfg["run"]["session_name"])
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        session_dir = output_dir / f"{session_name}_{timestamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _start_web_server(self, session_dir: Path, status_provider: Any) -> LiveDashboardServer | None:
        if not bool(self.cfg["web"]["enabled"]):
            return None
        server = LiveDashboardServer(
            host=str(self.cfg["web"]["host"]),
            port=int(self.cfg["web"]["port"]),
            session_dir=session_dir,
            status_provider=status_provider,
        )
        server.start()
        return server

    @staticmethod
    def _sleep_to_target(started: float, target_interval: float) -> None:
        if target_interval <= 0.0:
            return
        elapsed = time.perf_counter() - started
        remaining = target_interval - elapsed
        if remaining > 0.0:
            time.sleep(remaining)
