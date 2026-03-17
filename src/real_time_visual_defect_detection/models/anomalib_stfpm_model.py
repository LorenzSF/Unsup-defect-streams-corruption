from __future__ import annotations

from typing import Sequence

import torch

from real_time_visual_defect_detection.models.anomalib_adapter import AnomalibAdapter


class AnomalibStfpmModel(AnomalibAdapter):
    """Pipeline wrapper for anomalib STFPM."""

    def __init__(
        self,
        threshold: float,
        device: str,
        image_size: int,
        batch_size: int,
        backbone: str,
        layers: Sequence[str],
        epochs: int,
        learning_rate: float,
    ) -> None:
        super().__init__(
            threshold=threshold,
            device=device,
            image_size=image_size,
            batch_size=batch_size,
            imagenet_normalize=True,
        )
        self.backbone = backbone
        self.layers = list(layers)
        self.epochs = max(1, int(epochs))
        self.learning_rate = float(learning_rate)

    def _import_stfpm_components(self):
        try:
            from anomalib.models.image.stfpm.torch_model import STFPMModel
            from anomalib.models.image.stfpm.loss import STFPMLoss
        except ModuleNotFoundError as exc:  # pragma: no cover
            self._raise_missing_dependency(exc, "STFPM")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "STFPM adapter requires anomalib>=2.2 with stfpm modules available."
            ) from exc
        return STFPMModel, STFPMLoss

    def fit(self, train_paths, fit_context=None) -> None:
        if len(train_paths) == 0:
            raise ValueError("AnomalibStfpmModel requires at least one training image.")
        fit_paths = self._fit_paths(train_paths=train_paths, fit_context=fit_context)

        STFPMModel, STFPMLoss = self._import_stfpm_components()
        model = STFPMModel(backbone=self.backbone, layers=self.layers).to(self.device)
        loss_fn = STFPMLoss()
        optimizer = torch.optim.SGD(
            params=model.student_model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            dampening=0.0,
            weight_decay=0.001,
        )

        model.train()
        for epoch in range(self.epochs):
            desc = f"[anomalib_stfpm] fit {epoch + 1}/{self.epochs}"
            for batch in self._iter_training_batches(fit_paths, desc):
                optimizer.zero_grad(set_to_none=True)
                teacher_features, student_features = model(batch)
                loss = loss_fn(teacher_features, student_features)
                loss.backward()
                optimizer.step()

        model.eval()
        self._model = model
        self._is_fitted = True
