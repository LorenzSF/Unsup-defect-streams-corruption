from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ModelOutput:
    score: float
    is_anomaly: bool
    heatmap: Optional[np.ndarray] = None


class BaseModel:
    def fit(
        self,
        train_paths: List[Path],
        fit_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Train the model on *train_paths*.
        Default implementation is a no-op for models that require no
        training (e.g. reference-free heuristics).  Override in
        subclasses that need a fitting step."""

    def predict(self, x: np.ndarray) -> ModelOutput:
        raise NotImplementedError

    def get_embedding(self, x: np.ndarray) -> Optional[np.ndarray]:
        """Return a 1-D feature embedding for image *x*, or ``None``.
        Models that expose an internal representation (e.g. a bottleneck
        vector) should override this method and return a 1-D float32
        numpy array.  Models that do not support embeddings keep the
        default ``None`` return value; the pipeline will skip UMAP
        generation in that case.

        Parameters
        ----------
        x:
            Preprocessed image array as produced by the pipeline
            (BGR float32/64 in [0, 1]).

        Returns
        -------
        numpy.ndarray or None
            1-D float32 embedding vector, or ``None`` if unsupported.
        """
        return None



from pathlib import Path
import sys
from typing import Any, Dict, Iterator, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from torchvision.transforms import Compose, Normalize, Resize, ToTensor



def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    if torch.is_tensor(value):
        return float(value.detach().reshape(-1)[0].item())
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return float(arr.reshape(-1)[0])


def _as_heatmap(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if torch.is_tensor(value):
        arr = value.detach().float().cpu().numpy()
    else:
        arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return None

    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        return None

    arr = arr.astype(np.float32)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if max_val > min_val:
        arr = (arr - min_val) / (max_val - min_val)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr


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
        heatmap = _as_heatmap(getattr(pred, "anomaly_map", None))
        is_anomaly = score >= self.threshold
        return ModelOutput(score=score, is_anomaly=is_anomaly, heatmap=heatmap)


import torch



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


from typing import Tuple

import torch



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


from typing import Sequence

import torch



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


from typing import Sequence

import torch



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


from typing import Sequence

import torch



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


import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA



def _topk_mean(arr: np.ndarray, frac: float = 0.01) -> float:
    flat = arr.ravel()
    k = max(1, int(len(flat) * frac))
    idx = np.argpartition(flat, -k)[-k:]
    return float(np.mean(flat[idx]))


class AnomalibRd4adModel(AnomalibAdapter):
    """Pipeline wrapper for anomalib Reverse Distillation (RD4AD)."""

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
        anomaly_map_mode: str = "multiply",
        pre_trained: bool = True,
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
        self.anomaly_map_mode = str(anomaly_map_mode)
        self.pre_trained = bool(pre_trained)

    def _import_rd4ad_components(self):
        try:
            from anomalib.models.image.reverse_distillation.torch_model import (
                ReverseDistillationModel,
            )
            from anomalib.models.image.reverse_distillation.loss import (
                ReverseDistillationLoss,
            )
            from anomalib.models.image.reverse_distillation.anomaly_map import (
                AnomalyMapGenerationMode,
            )
        except ModuleNotFoundError as exc:  # pragma: no cover
            self._raise_missing_dependency(exc, "RD4AD")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "RD4AD adapter requires anomalib>=2.2 with reverse_distillation modules available."
            ) from exc
        return ReverseDistillationModel, ReverseDistillationLoss, AnomalyMapGenerationMode

    def fit(self, train_paths, fit_context=None) -> None:
        if len(train_paths) == 0:
            raise ValueError("AnomalibRd4adModel requires at least one training image.")
        fit_paths = self._fit_paths(train_paths=train_paths, fit_context=fit_context)

        Rd4adModel, Rd4adLoss, AnomalyMapMode = self._import_rd4ad_components()
        try:
            mode = AnomalyMapMode(self.anomaly_map_mode)
        except ValueError as exc:
            valid = ", ".join(m.value for m in AnomalyMapMode)
            raise ValueError(
                f"Invalid anomaly_map_mode '{self.anomaly_map_mode}'. Expected one of: {valid}."
            ) from exc

        model = Rd4adModel(
            backbone=self.backbone,
            input_size=(self.image_size, self.image_size),
            layers=self.layers,
            anomaly_map_mode=mode,
            pre_trained=self.pre_trained,
        ).to(self.device)
        loss_fn = Rd4adLoss()

        # Encoder is the frozen teacher (forward() pins it to eval). Train only
        # the bottleneck embedding + decoder, per Deng & Li 2022.
        trainable = list(model.bottleneck.parameters()) + list(model.decoder.parameters())
        optimizer = torch.optim.Adam(
            trainable,
            lr=self.learning_rate,
            betas=(0.5, 0.999),
        )

        model.train()
        for epoch in range(self.epochs):
            desc = f"[anomalib_rd4ad] fit {epoch + 1}/{self.epochs}"
            for batch in self._iter_training_batches(fit_paths, desc):
                optimizer.zero_grad(set_to_none=True)
                encoder_features, decoder_features = model(batch)
                loss = loss_fn(encoder_features, decoder_features)
                loss.backward()
                optimizer.step()

        model.eval()
        self._model = model
        self._is_fitted = True


class SubspaceADModel(BaseModel):
    """Minimal SubspaceAD-style model: frozen DINOv2 features + PCA residual scoring."""

    def __init__(
        self,
        threshold: float = 0.5,
        device: str = "cpu",
        model_ckpt: str = "facebook/dinov2-with-registers-large",
        image_size: int = 256,
        batch_size: int = 4,
        pca_ev: float = 0.99,
        pca_dim: Optional[int] = None,
        img_score_agg: str = "mtop1p",
        layers: Optional[Sequence[int]] = None,
    ) -> None:
        self.threshold = float(threshold)
        self.device = device
        self.model_ckpt = model_ckpt
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        self.pca_ev = float(pca_ev)
        self.pca_dim = pca_dim
        self.img_score_agg = str(img_score_agg)
        self.layers = list(layers) if layers else [-12, -13, -14, -15, -16, -17, -18]

        self._processor = None
        self._model = None
        self._pca: Optional[PCA] = None
        self._is_fitted = False

    def _load_extractor(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            from transformers import AutoImageProcessor, AutoModel
        except Exception as exc:  # pragma: no cover - dependency/runtime specific
            raise RuntimeError(
                "SubspaceAD requires 'transformers'. Install project dependencies first."
            ) from exc

        self._processor = AutoImageProcessor.from_pretrained(self.model_ckpt)
        self._model = AutoModel.from_pretrained(self.model_ckpt).eval().to(self.device)

    def _images_to_pil(self, paths: List[Path]) -> List[Image.Image]:
        out: List[Image.Image] = []
        for path in paths:
            img = Image.open(str(path)).convert("RGB")
            out.append(img)
        return out

    @torch.no_grad()
    def _extract_tokens(
        self, pil_images: List[Image.Image]
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._processor is not None and self._model is not None

        inputs = self._processor(
            images=pil_images,
            return_tensors="pt",
            do_resize=True,
            size={"height": self.image_size, "width": self.image_size},
            do_center_crop=False,
        ).to(self.device)
        outputs = self._model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        layer_feats = [hidden_states[idx][:, 1:, :] for idx in self.layers]
        fused = torch.stack(layer_feats, dim=0).mean(dim=0)  # [B, N, C]
        b, n, c = fused.shape
        side = int(math.sqrt(n))
        n_valid = side * side
        fused = fused[:, :n_valid, :]
        spatial = fused.reshape(b, side, side, c).cpu().numpy().astype(np.float32)

        cls_tokens = hidden_states[self.layers[0]][:, 0, :].cpu().numpy().astype(np.float32)
        return spatial, cls_tokens

    def _aggregate_score(self, anomaly_map: np.ndarray) -> float:
        if self.img_score_agg == "max":
            return float(np.max(anomaly_map))
        if self.img_score_agg == "mean":
            return float(np.mean(anomaly_map))
        if self.img_score_agg == "p99":
            return float(np.percentile(anomaly_map, 99))
        if self.img_score_agg == "mtop5":
            return float(np.mean(np.sort(anomaly_map.reshape(-1))[-5:]))
        return _topk_mean(anomaly_map, frac=0.01)

    def fit(
        self,
        train_paths: List[Path],
        fit_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(train_paths) == 0:
            raise ValueError("SubspaceADModel requires at least one training image.")

        self._load_extractor()
        all_tokens: List[np.ndarray] = []

        for start in range(0, len(train_paths), self.batch_size):
            batch_paths = train_paths[start : start + self.batch_size]
            pil_images = self._images_to_pil(batch_paths)
            spatial, _ = self._extract_tokens(pil_images)
            flat = spatial.reshape(-1, spatial.shape[-1])
            all_tokens.append(flat)

        features = np.concatenate(all_tokens, axis=0)
        n_components: int | float = self.pca_dim if self.pca_dim is not None else self.pca_ev
        self._pca = PCA(n_components=n_components, svd_solver="full", whiten=False)
        self._pca.fit(features)
        self._is_fitted = True

    def predict(self, x: np.ndarray) -> ModelOutput:
        if not self._is_fitted or self._pca is None:
            raise RuntimeError("SubspaceADModel is not ready. Call fit() first.")

        img_rgb = (x[..., ::-1] * 255.0).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(img_rgb)
        spatial, _ = self._extract_tokens([pil])

        flat = spatial.reshape(-1, spatial.shape[-1])
        recon = self._pca.inverse_transform(self._pca.transform(flat))
        scores = np.sum((flat - recon) ** 2, axis=1)
        score_map = scores.reshape(spatial.shape[1], spatial.shape[2]).astype(np.float32)
        score_map = cv2.resize(
            score_map, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )

        score = self._aggregate_score(score_map)
        return ModelOutput(
            score=score,
            is_anomaly=score >= self.threshold,
            heatmap=_as_heatmap(score_map),
        )

    def get_embedding(self, x: np.ndarray) -> Optional[np.ndarray]:
        if self._model is None or self._processor is None:
            return None
        img_rgb = (x[..., ::-1] * 255.0).clip(0, 255).astype(np.uint8)
        _, cls = self._extract_tokens([Image.fromarray(img_rgb)])
        return cls.reshape(-1).astype(np.float32)


from typing import Any, Dict


def _build_rd4ad(model_cfg: Dict[str, Any], runtime_cfg: Dict[str, Any]) -> BaseModel:

    an_cfg = model_cfg.get("anomalib", {})
    rd_cfg = model_cfg.get("rd4ad", {})
    return AnomalibRd4adModel(
        threshold=float(model_cfg.get("threshold", 0.5)),
        device=str(runtime_cfg.get("resolved_device", "cpu")),
        image_size=int(rd_cfg.get("image_size", an_cfg.get("image_size", 256))),
        batch_size=int(rd_cfg.get("batch_size", an_cfg.get("batch_size", 16))),
        backbone=str(rd_cfg.get("backbone", an_cfg.get("backbone", "wide_resnet50_2"))),
        layers=rd_cfg.get("layers", an_cfg.get("layers", ["layer1", "layer2", "layer3"])),
        epochs=int(rd_cfg.get("epochs", 200)),
        learning_rate=float(rd_cfg.get("learning_rate", 0.005)),
        anomaly_map_mode=str(rd_cfg.get("anomaly_map_mode", "multiply")),
        pre_trained=bool(rd_cfg.get("pre_trained", an_cfg.get("pre_trained", True))),
    )


def _build_anomalib_patchcore(
    model_cfg: Dict[str, Any], runtime_cfg: Dict[str, Any]
) -> BaseModel:

    an_cfg = model_cfg.get("anomalib", {})
    return AnomalibPatchcoreModel(
        threshold=float(model_cfg.get("threshold", 0.5)),
        device=str(runtime_cfg.get("resolved_device", "cpu")),
        image_size=int(an_cfg.get("image_size", 256)),
        batch_size=int(an_cfg.get("batch_size", 8)),
        pre_trained=bool(an_cfg.get("pre_trained", True)),
        backbone=str(an_cfg.get("backbone", "wide_resnet50_2")),
        layers=an_cfg.get("layers", ["layer2", "layer3"]),
        coreset_sampling_ratio=float(an_cfg.get("coreset_sampling_ratio", 0.1)),
        num_neighbors=int(an_cfg.get("num_neighbors", 9)),
    )


def _build_anomalib_padim(
    model_cfg: Dict[str, Any], runtime_cfg: Dict[str, Any]
) -> BaseModel:

    an_cfg = model_cfg.get("anomalib", {})
    return AnomalibPadimModel(
        threshold=float(model_cfg.get("threshold", 0.5)),
        device=str(runtime_cfg.get("resolved_device", "cpu")),
        image_size=int(an_cfg.get("image_size", 256)),
        batch_size=int(an_cfg.get("batch_size", 8)),
        pre_trained=bool(an_cfg.get("pre_trained", True)),
        backbone=str(an_cfg.get("backbone", "resnet18")),
        layers=an_cfg.get("layers", ["layer1", "layer2", "layer3"]),
        n_features=int(an_cfg["n_features"]) if an_cfg.get("n_features") is not None else None,
    )


def _build_anomalib_stfpm(
    model_cfg: Dict[str, Any], runtime_cfg: Dict[str, Any]
) -> BaseModel:

    an_cfg = model_cfg.get("anomalib", {})
    st_cfg = model_cfg.get("stfpm", {})
    return AnomalibStfpmModel(
        threshold=float(model_cfg.get("threshold", 0.5)),
        device=str(runtime_cfg.get("resolved_device", "cpu")),
        image_size=int(an_cfg.get("image_size", 256)),
        batch_size=int(an_cfg.get("batch_size", 8)),
        backbone=str(st_cfg.get("backbone", an_cfg.get("backbone", "resnet18"))),
        layers=st_cfg.get("layers", an_cfg.get("layers", ["layer1", "layer2", "layer3"])),
        epochs=int(st_cfg.get("epochs", 1)),
        learning_rate=float(st_cfg.get("learning_rate", 0.4)),
    )


def _build_anomalib_csflow(
    model_cfg: Dict[str, Any], runtime_cfg: Dict[str, Any]
) -> BaseModel:

    an_cfg = model_cfg.get("anomalib", {})
    cs_cfg = model_cfg.get("csflow", {})
    return AnomalibCsflowModel(
        threshold=float(model_cfg.get("threshold", 0.5)),
        device=str(runtime_cfg.get("resolved_device", "cpu")),
        image_size=int(an_cfg.get("image_size", 256)),
        batch_size=int(an_cfg.get("batch_size", 8)),
        epochs=int(cs_cfg.get("epochs", 1)),
        learning_rate=float(cs_cfg.get("learning_rate", 2e-4)),
        cross_conv_hidden_channels=int(cs_cfg.get("cross_conv_hidden_channels", 1024)),
        n_coupling_blocks=int(cs_cfg.get("n_coupling_blocks", 4)),
        clamp=int(cs_cfg.get("clamp", 3)),
    )


def _parse_beta(beta_value: Any) -> tuple[float, float]:
    if isinstance(beta_value, (list, tuple)) and len(beta_value) == 2:
        return float(beta_value[0]), float(beta_value[1])
    if beta_value is None:
        return 0.2, 1.0
    scalar = float(beta_value)
    return scalar, scalar


def _build_anomalib_draem(
    model_cfg: Dict[str, Any], runtime_cfg: Dict[str, Any]
) -> BaseModel:

    an_cfg = model_cfg.get("anomalib", {})
    dr_cfg = model_cfg.get("draem", {})
    return AnomalibDraemModel(
        threshold=float(model_cfg.get("threshold", 0.5)),
        device=str(runtime_cfg.get("resolved_device", "cpu")),
        image_size=int(an_cfg.get("image_size", 256)),
        batch_size=int(an_cfg.get("batch_size", 8)),
        epochs=int(dr_cfg.get("epochs", 1)),
        learning_rate=float(dr_cfg.get("learning_rate", 1e-4)),
        beta=_parse_beta(dr_cfg.get("beta", (0.2, 1.0))),
    )


def _build_subspacead(model_cfg: Dict[str, Any], runtime_cfg: Dict[str, Any]) -> BaseModel:

    sub_cfg = model_cfg.get("subspacead", {})
    return SubspaceADModel(
        threshold=float(model_cfg.get("threshold", 0.5)),
        device=str(runtime_cfg.get("resolved_device", "cpu")),
        model_ckpt=str(
            sub_cfg.get("model_ckpt", "facebook/dinov2-with-registers-large")
        ),
        image_size=int(sub_cfg.get("image_size", 256)),
        batch_size=int(sub_cfg.get("batch_size", 4)),
        pca_ev=float(sub_cfg.get("pca_ev", 0.99)),
        pca_dim=int(sub_cfg["pca_dim"]) if sub_cfg.get("pca_dim") is not None else None,
        img_score_agg=str(sub_cfg.get("img_score_agg", "mtop1p")),
        layers=sub_cfg.get("layers", [-12, -13, -14, -15, -16, -17, -18]),
    )


# Central model registry used by the pipeline and benchmark runner.
_MODEL_BUILDERS = {
    "rd4ad": _build_rd4ad,
    "anomalib_patchcore": _build_anomalib_patchcore,
    "anomalib_padim": _build_anomalib_padim,
    "anomalib_stfpm": _build_anomalib_stfpm,
    "anomalib_csflow": _build_anomalib_csflow,
    "anomalib_draem": _build_anomalib_draem,
    "subspacead": _build_subspacead,
}


def build_model(model_cfg: Dict[str, Any], runtime_cfg: Dict[str, Any]) -> BaseModel:
    name = str(model_cfg.get("name", "dummy_distance"))
    builder = _MODEL_BUILDERS.get(name)
    if builder is None:
        supported = ", ".join(sorted(_MODEL_BUILDERS))
        raise ValueError(f"Unknown model name: '{name}'. Supported: {supported}.")
    return builder(model_cfg, runtime_cfg)


def available_models() -> list[str]:
    return sorted(_MODEL_BUILDERS)


# Model dependency mapping - specifies which Python packages each model requires
_MODEL_DEPENDENCIES = {
    "rd4ad": ("anomalib", "torch", "numpy"),
    "anomalib_patchcore": ("anomalib", "torch", "numpy"),
    "anomalib_padim": ("anomalib", "torch", "numpy"),
    "anomalib_stfpm": ("anomalib", "torch", "numpy"),
    "anomalib_csflow": ("anomalib", "FrEIA", "torch", "numpy"),
    "anomalib_draem": ("anomalib", "torch", "numpy"),
    "subspacead": ("transformers", "torch", "numpy", "sklearn"),
}


def model_dependencies(model_name: str) -> tuple[str, ...]:
    """Return a tuple of required module names for the given model.
    
    Parameters
    ----------
    model_name : str
        The name of the model.
        
    Returns
    -------
    tuple[str, ...]
        A tuple of module names required by the model.
        
    Raises
    ------
    ValueError
        If the model name is not recognized.
    """
    if model_name not in _MODEL_DEPENDENCIES:
        supported = ", ".join(sorted(_MODEL_DEPENDENCIES))
        raise ValueError(
            f"Unknown model name: '{model_name}'. Supported: {supported}."
        )
    return _MODEL_DEPENDENCIES[model_name]
