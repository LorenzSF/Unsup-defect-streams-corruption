from __future__ import annotations

import torch

from real_time_visual_defect_detection.models.anomalib_adapter import AnomalibAdapter


class AnomalibCsflowModel(AnomalibAdapter):
    """Pipeline wrapper for anomalib CS-Flow."""

    def __init__(
        self,
        threshold: float,
        device: str,
        image_size: int,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        cross_conv_hidden_channels: int,
        n_coupling_blocks: int,
        clamp: int,
    ) -> None:
        super().__init__(
            threshold=threshold,
            device=device,
            image_size=image_size,
            batch_size=batch_size,
            imagenet_normalize=True,
        )
        self.epochs = max(1, int(epochs))
        self.learning_rate = float(learning_rate)
        self.cross_conv_hidden_channels = int(cross_conv_hidden_channels)
        self.n_coupling_blocks = int(n_coupling_blocks)
        self.clamp = int(clamp)

    def _import_csflow_components(self):
        try:
            from anomalib.models.image.csflow.torch_model import CsFlowModel
            from anomalib.models.image.csflow.loss import CsFlowLoss
        except ModuleNotFoundError as exc:  # pragma: no cover
            self._raise_missing_dependency(exc, "CS-Flow")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "CS-Flow adapter requires anomalib>=2.2 with csflow modules available."
            ) from exc
        return CsFlowModel, CsFlowLoss

    def fit(self, train_paths, fit_context=None) -> None:
        if len(train_paths) == 0:
            raise ValueError("AnomalibCsflowModel requires at least one training image.")
        fit_paths = self._fit_paths(train_paths=train_paths, fit_context=fit_context)

        CsFlowModel, CsFlowLoss = self._import_csflow_components()
        model = CsFlowModel(
            input_size=(self.image_size, self.image_size),
            cross_conv_hidden_channels=self.cross_conv_hidden_channels,
            n_coupling_blocks=self.n_coupling_blocks,
            clamp=self.clamp,
            num_channels=3,
        ).to(self.device)
        loss_fn = CsFlowLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            eps=1e-4,
            weight_decay=1e-5,
            betas=(0.5, 0.9),
        )

        model.train()
        for epoch in range(self.epochs):
            desc = f"[anomalib_csflow] fit {epoch + 1}/{self.epochs}"
            for batch in self._iter_training_batches(fit_paths, desc):
                optimizer.zero_grad(set_to_none=True)
                z_dist, jacobians = model(batch)
                loss = loss_fn(z_dist, jacobians)
                loss.backward()
                optimizer.step()

        model.eval()
        self._model = model
        self._is_fitted = True
