from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from threading import Lock

import numpy as np

from .contracts import DecisionRecord, LiveSnapshot, ObjectMatch, RuntimeArtifact
from .state_manager import RuntimeStateManager


class LiveMetrics:
    def __init__(
        self,
        session_dir: Path,
        input_fps: float,
        latency_sla_ms: float,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.input_fps = float(input_fps)
        self.latency_sla_ms = float(latency_sla_ms)

        self.started_at = time.perf_counter()
        self.frames_seen = 0
        self.decisions_emitted = 0
        self.fail_count = 0
        self.no_decision_count = 0
        self.object_change_count = 0
        self.latencies_ms: deque[float] = deque(maxlen=2048)
        self.recent_decisions: deque[dict] = deque(maxlen=12)
        self.recent_fails: deque[dict] = deque(maxlen=8)
        self.last_classifier_confidence = 0.0
        self.last_classifier_distance = 0.0
        self._lock = Lock()

    def record_frame_seen(self) -> None:
        with self._lock:
            self.frames_seen += 1

    def record_latency(self, latency_ms: float) -> None:
        with self._lock:
            self.latencies_ms.append(float(latency_ms))

    def record_match(self, match: ObjectMatch) -> None:
        with self._lock:
            self.last_classifier_confidence = float(match.confidence)
            self.last_classifier_distance = float(match.distance)

    def record_transition(self) -> None:
        with self._lock:
            self.object_change_count += 1

    def record_no_decision(self) -> None:
        with self._lock:
            self.no_decision_count += 1

    def record_decision(self, decision: DecisionRecord) -> None:
        row = decision.to_dict()
        with self._lock:
            self.decisions_emitted += 1
            self.recent_decisions.appendleft(row)
            if int(decision.pred_is_anomaly) == 1:
                self.fail_count += 1
                self.recent_fails.appendleft(row)

    def snapshot(
        self,
        state_manager: RuntimeStateManager,
        artifact: RuntimeArtifact,
    ) -> dict:
        with self._lock:
            uptime_s = max(time.perf_counter() - self.started_at, 1e-6)
            latencies = np.asarray(list(self.latencies_ms), dtype=np.float32)
            mean_latency_ms = float(np.mean(latencies)) if latencies.size > 0 else 0.0
            p95_latency_ms = float(np.percentile(latencies, 95)) if latencies.size > 0 else 0.0
            snapshot = LiveSnapshot(
                state=state_manager.state.value,
                session_dir=str(self.session_dir),
                active_model=artifact.model_name,
                frames_seen=self.frames_seen,
                decisions_emitted=self.decisions_emitted,
                fail_count=self.fail_count,
                no_decision_count=self.no_decision_count,
                object_change_count=self.object_change_count,
                input_fps=self.input_fps,
                processed_fps=float(self.frames_seen / uptime_s),
                decision_fps=float(self.decisions_emitted / uptime_s),
                mean_latency_ms=mean_latency_ms,
                p95_latency_ms=p95_latency_ms,
                latency_sla_ms=self.latency_sla_ms,
                classifier_confidence=self.last_classifier_confidence,
                classifier_distance=self.last_classifier_distance,
                calibration_frames=state_manager.calibration_frames,
                calibration_score_mean=state_manager.calibration_score_mean(),
                baseline_score_mean=artifact.baseline.score_mean,
                baseline_score_std=artifact.baseline.score_std,
                threshold=artifact.threshold,
                recent_decisions=list(self.recent_decisions),
                recent_fails=list(self.recent_fails),
            )
            return snapshot.to_dict()
