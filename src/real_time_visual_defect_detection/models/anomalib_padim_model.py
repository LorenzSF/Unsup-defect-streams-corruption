from __future__ import annotations

from typing import Sequence

import torch

from real_time_visual_defect_detection.models.anomalib_adapter import AnomalibAdapter


class AnomalibPadimModel(AnomalibAdapter):
    """Pipeline wrapper for anomalib PaDiM."""

    def __init__(
        self,
        threshold: float,
        device: str,
        image_size: int,
        batch_size: int,
        pre_trained: bool,
        backbone: str,
        layers: Sequence[str],
        n_features: int | None,
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
        self.n_features = int(n_features) if n_features is not None else None

    def _import_padim_model(self):
        try:
            from anomalib.models.image.padim.torch_model import PadimModel
        except ModuleNotFoundError as exc:  # pragma: no cover
            self._raise_missing_dependency(exc, "PaDiM")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "PaDiM adapter requires anomalib>=2.2 with padim model module available."
            ) from exc
        return PadimModel

    def fit(self, train_paths, fit_context=None) -> None:
        if len(train_paths) == 0:
            raise ValueError("AnomalibPadimModel requires at least one training image.")
        fit_paths = self._fit_paths(train_paths=train_paths, fit_context=fit_context)

        PadimModel = self._import_padim_model()
        model = PadimModel(
            backbone=self.backbone,
            layers=self.layers,
            pre_trained=self.pre_trained,
            n_features=self.n_features,
        ).to(self.device)
        model.train()

        for batch in self._iter_training_batches(fit_paths, "[anomalib_padim] fit"):
            with torch.no_grad():
                _ = model(batch)

        model.fit()
        model.eval()
        self._model = model
        self._is_fitted = True
