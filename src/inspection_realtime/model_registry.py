from __future__ import annotations

from typing import Optional

from benchmark_AD.data import apply_dataset_split, resolve_dataset_labeled
from benchmark_AD.models import BaseModel, build_model

from .contracts import RuntimeArtifact


class SingleModelRegistry:
    def __init__(self, artifact: RuntimeArtifact, fit_policy: str = "auto") -> None:
        self.artifact = artifact
        self.fit_policy = str(fit_policy).lower()
        self._model: Optional[BaseModel] = None

    def load(self) -> BaseModel:
        if self._model is not None:
            return self._model

        model = build_model(dict(self.artifact.model_cfg), dict(self.artifact.runtime_cfg))
        if hasattr(model, "threshold"):
            model.threshold = float(self.artifact.threshold)

        if self._should_fit(model):
            samples = resolve_dataset_labeled(
                self.artifact.dataset_cfg["source_type"],
                self.artifact.dataset_cfg["path"],
                self.artifact.dataset_cfg["extract_dir"],
            )
            split_cfg = self.artifact.dataset_cfg.get("split", {})
            split_result = apply_dataset_split(
                samples=samples,
                split_cfg=split_cfg,
                fallback_seed=42,
            )
            fit_context = {
                "train_samples": split_result.train,
                "val_samples": split_result.val,
                "test_samples": split_result.test,
            }
            train_paths = [sample.path for sample in split_result.train]
            model.fit(train_paths, fit_context=fit_context)
            if hasattr(model, "threshold"):
                model.threshold = float(self.artifact.threshold)

        if not bool(getattr(model, "_is_fitted", True)):
            raise RuntimeError(
                "Runtime model is not ready. Use fit_policy='historical_fit' or select a warm-start artifact."
            )

        self._model = model
        return model

    def _should_fit(self, model: BaseModel) -> bool:
        if self.fit_policy == "historical_fit":
            return True
        if self.fit_policy == "skip_fit":
            return False
        return not bool(getattr(model, "_is_fitted", False))
