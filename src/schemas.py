import dataclasses
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, List, Optional, Union, get_args, get_origin, get_type_hints

import numpy as np
import yaml

@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    label: int
    timestamp: float
    source_id: str
    index: int


@dataclass(frozen=True)
class Prediction:
    score: float
    anomaly_map: Optional[np.ndarray]
    latency_ms: float


@dataclass
class MetricSnapshot:
    n_seen: int
    n_anomalies: int
    auroc: float
    f1: float
    mean_latency_ms: float
    p95_latency_ms: float
    throughput_fps: float


@dataclass(frozen=True)
class CorruptionSpec:
    kind: str
    severity: int
    probability: float

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("CorruptionSpec.kind must be non-empty")
        if not 1 <= self.severity <= 5:
            raise ValueError(
                f"CorruptionSpec.severity must be in 1..5, got {self.severity}"
            )
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                "CorruptionSpec.probability must be in 0..1, "
                f"got {self.probability}"
            )


@dataclass(frozen=True)
class StreamConfig:
    dataset: str
    category: str
    data_root: str
    extensions: List[str]
    shuffle: bool
    max_frames: Optional[int]

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError("StreamConfig.dataset must be non-empty")
        if not self.category:
            raise ValueError("StreamConfig.category must be non-empty")
        if not self.data_root:
            raise ValueError("StreamConfig.data_root must be non-empty")
        if not self.extensions:
            raise ValueError("StreamConfig.extensions must be non-empty")
        for ext in self.extensions:
            if not ext.startswith("."):
                raise ValueError(
                    "StreamConfig.extensions: each entry must start with '.', "
                    f"got {ext!r}"
                )
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError(
                f"StreamConfig.max_frames must be > 0 when set, got {self.max_frames}"
            )


@dataclass(frozen=True)
class WarmupConfig:
    warmup_steps: int
    use_clean_frames: bool

    def __post_init__(self) -> None:
        if self.warmup_steps <= 0:
            raise ValueError(
                f"WarmupConfig.warmup_steps must be > 0, got {self.warmup_steps}"
            )


@dataclass(frozen=True)
class ModelConfig:
    name: str
    backbone: str
    device: str
    checkpoint: Optional[str]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ModelConfig.name must be non-empty")
        if not self.device:
            raise ValueError("ModelConfig.device must be non-empty")


@dataclass(frozen=True)
class CorruptionConfig:
    enabled: bool
    specs: List[CorruptionSpec]

    def __post_init__(self) -> None:
        if self.enabled and not self.specs:
            raise ValueError("CorruptionConfig.specs must be non-empty when enabled")


@dataclass(frozen=True)
class MetricsConfig:
    window_size: int
    threshold_mode: str = "manual"
    manual_threshold: Optional[float] = 0.5
    threshold_quantile: float = 1.0
    threshold_value: Optional[float] = 0.5

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError(
                f"MetricsConfig.window_size must be > 0, got {self.window_size}"
            )
        if self.threshold_mode not in {"manual", "quantile"}:
            raise ValueError(
                "MetricsConfig.threshold_mode must be one of "
                f"'manual', 'quantile', got {self.threshold_mode!r}"
            )
        if self.threshold_mode == "manual" and self.manual_threshold is None:
            raise ValueError(
                "MetricsConfig.manual_threshold must be set when "
                "threshold_mode == 'manual'"
            )
        if not 0.0 <= self.threshold_quantile <= 1.0:
            raise ValueError(
                "MetricsConfig.threshold_quantile must be in 0..1, "
                f"got {self.threshold_quantile}"
            )
        if self.manual_threshold is not None and not np.isfinite(self.manual_threshold):
            raise ValueError(
                "MetricsConfig.manual_threshold must be finite when set, "
                f"got {self.manual_threshold}"
            )
        if self.threshold_value is not None and not np.isfinite(self.threshold_value):
            raise ValueError(
                "MetricsConfig.threshold_value must be finite when set, "
                f"got {self.threshold_value}"
            )


@dataclass(frozen=True)
class VizConfig:
    mode: str
    every_n_frames: int
    overlay_alpha: float

    def __post_init__(self) -> None:
        if self.mode not in {"window", "file", "none"}:
            raise ValueError(
                "VizConfig.mode must be one of 'window', 'file', 'none', "
                f"got {self.mode!r}"
            )
        if self.every_n_frames <= 0:
            raise ValueError(
                "VizConfig.every_n_frames must be > 0, "
                f"got {self.every_n_frames}"
            )
        if not 0.0 <= self.overlay_alpha <= 1.0:
            raise ValueError(
                "VizConfig.overlay_alpha must be in 0..1, "
                f"got {self.overlay_alpha}"
            )


@dataclass(frozen=True)
class BenchmarkConfig:
    enabled: bool
    baseline: str
    learning_rate: float

    def __post_init__(self) -> None:
        if not self.baseline:
            raise ValueError("BenchmarkConfig.baseline must be non-empty")
        if self.learning_rate <= 0.0:
            raise ValueError(
                "BenchmarkConfig.learning_rate must be > 0, "
                f"got {self.learning_rate}"
            )


@dataclass
class RunConfig:
    seed: int
    output_dir: str
    log_every: int
    stream: StreamConfig
    warmup: WarmupConfig
    model: ModelConfig
    corruption: CorruptionConfig
    metrics: MetricsConfig
    visualization: VizConfig
    benchmark: BenchmarkConfig

    def __post_init__(self) -> None:
        if not self.output_dir:
            raise ValueError("RunConfig.output_dir must be non-empty")
        if self.log_every <= 0:
            raise ValueError(f"RunConfig.log_every must be > 0, got {self.log_every}")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise TypeError(
                f"config root must be a mapping, got {type(raw).__name__}"
            )
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        return _build_dataclass(cls, data, path="$")
    

def _build_dataclass(dc_type: type, data: Any, path: str) -> Any:
    if not isinstance(data, dict):
        raise TypeError(
            f"{path}: expected mapping for {dc_type.__name__}, "
            f"got {type(data).__name__}"
        )

    hints = get_type_hints(dc_type)
    expected = {f.name: f for f in fields(dc_type)}

    unknown = set(data.keys()) - set(expected.keys())
    if unknown:
        raise ValueError(
            f"{path}: unknown keys for {dc_type.__name__}: {sorted(unknown)}"
        )

    kwargs: dict = {}
    for name, f in expected.items():
        if name not in data:
            has_default = (
                f.default is not dataclasses.MISSING
                or f.default_factory is not dataclasses.MISSING  # type: ignore[misc]
            )
            if has_default:
                continue
            raise ValueError(
                f"{path}: missing required key '{name}' for {dc_type.__name__}"
            )
        kwargs[name] = _coerce(hints[name], data[name], f"{path}.{name}")
    return dc_type(**kwargs)


def _coerce(annotation: Any, value: Any, path: str) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union:
        if value is None and type(None) in args:
            return None
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _coerce(non_none[0], value, path)
        errors = []
        for cand in non_none:
            try:
                return _coerce(cand, value, path)
            except (TypeError, ValueError) as e:
                errors.append(f"{cand}: {e}")
        raise TypeError(
            f"{path}: value {value!r} matches none of {non_none} ({errors})"
        )

    if origin in (list, List):
        if not isinstance(value, list):
            raise TypeError(
                f"{path}: expected list, got {type(value).__name__}"
            )
        inner = args[0] if args else Any
        return [_coerce(inner, v, f"{path}[{i}]") for i, v in enumerate(value)]

    if is_dataclass(annotation):
        return _build_dataclass(annotation, value, path)

    if annotation is bool:
        if not isinstance(value, bool):
            raise TypeError(
                f"{path}: expected bool, got {type(value).__name__}"
            )
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{path}: expected int, got {type(value).__name__}"
            )
        return value
    if annotation is float:
        if isinstance(value, bool):
            raise TypeError(f"{path}: expected float, got bool")
        if isinstance(value, (int, float)):
            return float(value)
        raise TypeError(
            f"{path}: expected float, got {type(value).__name__}"
        )
    if annotation is str:
        if not isinstance(value, str):
            raise TypeError(
                f"{path}: expected str, got {type(value).__name__}"
            )
        return value
    if annotation is Any:
        return value

    if isinstance(annotation, type) and isinstance(value, annotation):
        return value

    raise TypeError(f"{path}: unsupported annotation {annotation!r}")
