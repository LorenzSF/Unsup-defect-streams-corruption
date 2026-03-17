from __future__ import annotations

from typing import Tuple

import torch

from real_time_visual_defect_detection.models.anomalib_adapter import AnomalibAdapter


class AnomalibDraemModel(AnomalibAdapter):
    """Pipeline wrapper for anomalib DRAEM."""

    def __init__(
        self,
        threshold: float,
        device: str,
        image_size: int,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        beta: Tuple[float, float],
    ) -> None:
        super().__init__(
            threshold=threshold,
            device=device,
            image_size=image_size,
            batch_size=batch_size,
            imagenet_normalize=False,
        )
        self.epochs = max(1, int(epochs))
        self.learning_rate = float(learning_rate)
        self.beta = (float(beta[0]), float(beta[1]))

    def _import_draem_components(self):
        try:
            from anomalib.models.image.draem.torch_model import DraemModel
            from anomalib.models.image.draem.loss import DraemLoss
            from anomalib.data.utils.generators.perlin import PerlinAnomalyGenerator
        except ModuleNotFoundError as exc:  # pragma: no cover
            self._raise_missing_dependency(exc, "DRAEM")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "DRAEM adapter requires anomalib>=2.2 with draem modules available."
            ) from exc
        return DraemModel, DraemLoss, PerlinAnomalyGenerator

    def fit(self, train_paths, fit_context=None) -> None:
        if len(train_paths) == 0:
            raise ValueError("AnomalibDraemModel requires at least one training image.")
        fit_paths = self._fit_paths(train_paths=train_paths, fit_context=fit_context)

        DraemModel, DraemLoss, PerlinAnomalyGenerator = self._import_draem_components()
        model = DraemModel(sspcab=False).to(self.device)
        loss_fn = DraemLoss()
        augmenter = PerlinAnomalyGenerator(
            anomaly_source_path=None,
            blend_factor=self.beta,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        model.train()
        for epoch in range(self.epochs):
            desc = f"[anomalib_draem] fit {epoch + 1}/{self.epochs}"
            for batch in self._iter_training_batches(fit_paths, desc):
                optimizer.zero_grad(set_to_none=True)
                augmented_image, anomaly_mask = augmenter(batch)
                reconstruction, prediction = model(augmented_image)
                loss = loss_fn(batch, reconstruction, anomaly_mask, prediction)
                loss.backward()
                optimizer.step()

        model.eval()
        self._model = model
        self._is_fitted = True
