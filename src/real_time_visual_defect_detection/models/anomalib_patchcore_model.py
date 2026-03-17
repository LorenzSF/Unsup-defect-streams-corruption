from __future__ import annotations

from typing import Sequence

import torch

from real_time_visual_defect_detection.models.anomalib_adapter import AnomalibAdapter


class AnomalibPatchcoreModel(AnomalibAdapter):
    """Pipeline wrapper for anomalib PatchCore."""

    def __init__(
        self,
        threshold: float,
        device: str,
        image_size: int,
        batch_size: int,
        pre_trained: bool,
        backbone: str,
        layers: Sequence[str],
        coreset_sampling_ratio: float,
        num_neighbors: int,
    ) -> None:
        super().__init__(
            threshold=threshold,
            device=device,
            image_size=image_size,
            batch_size=batch_size,
            imagenet_normalize=True,
        )
        self.pre_trained = bool(pre_trained)
        self.backbone = backbone
        self.layers = list(layers)
        self.coreset_sampling_ratio = float(coreset_sampling_ratio)
        self.num_neighbors = int(num_neighbors)

    def _import_patchcore_model(self):
        try:
            from anomalib.models.image.patchcore.torch_model import PatchcoreModel
        except ModuleNotFoundError as exc:  # pragma: no cover
            self._raise_missing_dependency(exc, "PatchCore")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "PatchCore adapter requires anomalib>=2.2 with patchcore model module available."
            ) from exc
        return PatchcoreModel

    def fit(self, train_paths, fit_context=None) -> None:
        if len(train_paths) == 0:
            raise ValueError("AnomalibPatchcoreModel requires at least one training image.")
        fit_paths = self._fit_paths(train_paths=train_paths, fit_context=fit_context)

        PatchcoreModel = self._import_patchcore_model()
        model = PatchcoreModel(
            backbone=self.backbone,
            layers=self.layers,
            pre_trained=self.pre_trained,
            num_neighbors=self.num_neighbors,
        ).to(self.device)
        model.train()

        for batch in self._iter_training_batches(fit_paths, "[anomalib_patchcore] fit"):
            with torch.no_grad():
                _ = model(batch)

        model.subsample_embedding(self.coreset_sampling_ratio)
        model.eval()
        self._model = model
        self._is_fitted = True
