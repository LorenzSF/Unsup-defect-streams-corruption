from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, Iterator, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from real_time_visual_defect_detection.models.base import BaseModel, ModelOutput


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    if torch.is_tensor(value):
        return float(value.detach().reshape(-1)[0].item())
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return float(arr.reshape(-1)[0])


class AnomalibAdapter(BaseModel):
    """Common utilities shared by anomalib model wrappers."""

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        threshold: float,
        device: str,
        image_size: int,
        batch_size: int,
        imagenet_normalize: bool = True,
    ) -> None:
        self.threshold = float(threshold)
        self.device = device
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        self.imagenet_normalize = bool(imagenet_normalize)

        transform_steps = [Resize((self.image_size, self.image_size)), ToTensor()]
        if self.imagenet_normalize:
            transform_steps.append(Normalize(self._IMAGENET_MEAN, self._IMAGENET_STD))
        self._transform = Compose(transform_steps)
        self._model = None
        self._is_fitted = False

    def _progress_enabled(self) -> bool:
        return bool(sys.stdout.isatty() or sys.stderr.isatty())

    @staticmethod
    def _raise_missing_dependency(exc: ModuleNotFoundError, model_label: str) -> None:
        missing = getattr(exc, "name", None) or "unknown"
        hint_map = {
            "lightning": "lightning",
            "FrEIA": "FrEIA",
            "kornia": "kornia",
            "anomalib": "anomalib",
        }
        hint = hint_map.get(missing, missing)
        raise RuntimeError(
            f"{model_label} adapter is unavailable: missing dependency '{missing}'. "
            f"Install it with: pip install {hint}"
        ) from exc

    def _to_tensor_batch_from_paths(self, paths: Sequence[Path]) -> torch.Tensor:
        tensors = []
        for path in paths:
            with Image.open(str(path)) as img:
                tensors.append(self._transform(img.convert("RGB")))
        batch = torch.stack(tensors, dim=0)
        return batch.to(self.device)

    def _to_tensor_single(self, x: np.ndarray) -> torch.Tensor:
        img_rgb = (x[..., ::-1] * 255.0).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(img_rgb)
        return self._transform(pil).unsqueeze(0).to(self.device)

    def _fit_paths(
        self,
        train_paths: list[Path],
        fit_context: Optional[Dict[str, Any]],
    ) -> list[Path]:
        train_samples = (fit_context or {}).get("train_samples")
        if train_samples:
            fit_paths = [Path(s.path) for s in train_samples if getattr(s, "label", -1) != 1]
        else:
            fit_paths = [Path(p) for p in train_paths]
        if len(fit_paths) == 0:
            raise ValueError("No normal samples available to fit the selected anomalib model.")
        return fit_paths

    def _iter_training_batches(
        self,
        fit_paths: list[Path],
        desc: str,
    ) -> Iterator[torch.Tensor]:
        total_batches = (len(fit_paths) + self.batch_size - 1) // self.batch_size
        for start in tqdm(
            range(0, len(fit_paths), self.batch_size),
            total=total_batches,
            desc=desc,
            unit="batch",
            dynamic_ncols=True,
            disable=not self._progress_enabled(),
        ):
            batch_paths = fit_paths[start : start + self.batch_size]
            yield self._to_tensor_batch_from_paths(batch_paths)

    def fit(
        self,
        train_paths: list[Path],
        fit_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError("Use a concrete anomalib model wrapper.")

    def predict(self, x: np.ndarray) -> ModelOutput:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("AnomalibAdapter is not ready. Call fit() first.")

        tensor = self._to_tensor_single(x)
        with torch.no_grad():
            pred = self._model(tensor)

        score = _as_float(getattr(pred, "pred_score", None), default=0.0)
        is_anomaly = score >= self.threshold
        return ModelOutput(score=score, is_anomaly=is_anomaly)
