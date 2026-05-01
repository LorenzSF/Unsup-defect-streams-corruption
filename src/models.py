from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
from PIL import Image

from .schemas import Frame, ModelConfig, Prediction, WarmupConfig


@runtime_checkable
class Model(Protocol):
    def predict(self, frame: Frame) -> Prediction: ...


def build_model(cfg: ModelConfig) -> Model:
    name = cfg.name.lower()
    if name == "pca":
        return PCADetector(cfg)
    if name in {"patchcore", "anomalib_patchcore"}:
        return PatchcoreDetector(cfg)
    if name in {"padim", "anomalib_padim"}:
        return PadimDetector(cfg)
    if name == "subspacead":
        return SubspaceADDetector(cfg)
    if name in {"stfpm", "anomalib_stfpm"}:
        return StfpmDetector(cfg)
    if name in {"csflow", "anomalib_csflow"}:
        return CsflowDetector(cfg)
    if name in {"draem", "anomalib_draem"}:
        return DraemDetector(cfg)
    if name in {"rd4ad", "reverse_distillation"}:
        return Rd4adDetector(cfg)
    if name == "efficientad":
        return EfficientAdDetector(cfg)
    supported = (
        "pca, patchcore, padim, subspacead, stfpm, csflow, draem, rd4ad, "
        "reverse_distillation, efficientad"
    )
    raise ValueError(f"unknown model '{cfg.name}' (supported: {supported})")


def warmup(model: Model, stream: Iterator[Frame], cfg: WarmupConfig) -> None:
    if not hasattr(model, "fit_warmup"):
        raise TypeError(f"model {type(model).__name__} has no fit_warmup() method")

    frames: List[Frame] = []
    for i, frame in enumerate(stream):
        if i >= cfg.warmup_steps:
            break
        frames.append(frame)
    if not frames:
        raise RuntimeError("no frames available for warmup")
    if len(frames) < cfg.warmup_steps:
        raise RuntimeError(
            f"warmup requires {cfg.warmup_steps} frames, got only {len(frames)}"
        )
    model.fit_warmup(frames)


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This model requires 'torch'. Install dependencies from requirements.txt."
        ) from exc
    return torch


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This model requires 'opencv-python'. Install dependencies from requirements.txt."
        ) from exc
    return cv2


def _require_torchvision_transforms():
    try:
        from torchvision.transforms import Compose, Normalize, Resize, ToTensor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This model requires 'torchvision'. Install dependencies from requirements.txt."
        ) from exc
    return Compose, Normalize, Resize, ToTensor


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        torch = _require_torch()
        if torch.is_tensor(value):
            return float(value.detach().reshape(-1)[0].item())
    except RuntimeError:
        pass
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return float(arr.reshape(-1)[0])


def _as_heatmap(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        torch = _require_torch()
        if torch.is_tensor(value):
            arr = value.detach().float().cpu().numpy()
        else:
            arr = np.asarray(value, dtype=np.float32)
    except RuntimeError:
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
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr


def _prediction_from_anomalib_output(output: Any, latency_ms: float) -> Prediction:
    score = _extract_score(output)
    heatmap = _extract_heatmap(output)
    if heatmap is None and hasattr(output, "anomaly_map"):
        heatmap = _as_heatmap(getattr(output, "anomaly_map"))
    if heatmap is None and hasattr(output, "pred_mask"):
        heatmap = _as_heatmap(getattr(output, "pred_mask"))
    if heatmap is None and score == 0.0:
        heatmap = _extract_heatmap(output)
        if heatmap is not None:
            score = float(np.max(heatmap))
    return Prediction(score=score, anomaly_map=heatmap, latency_ms=latency_ms)


def _extract_score(output: Any) -> float:
    for attr in ("pred_score", "score", "anomaly_score"):
        if hasattr(output, attr):
            return _as_float(getattr(output, attr), 0.0)
        if isinstance(output, dict) and attr in output:
            return _as_float(output[attr], 0.0)
    heatmap = _extract_heatmap(output)
    if heatmap is not None:
        return float(np.max(heatmap))
    return 0.0


def _extract_heatmap(output: Any) -> Optional[np.ndarray]:
    for attr in ("anomaly_map", "heatmap", "pred_mask"):
        if hasattr(output, attr):
            return _as_heatmap(getattr(output, attr))
        if isinstance(output, dict) and attr in output:
            return _as_heatmap(output[attr])
    return None


def _load_state_dict(model: Any, checkpoint: str) -> None:
    torch = _require_torch()
    payload = torch.load(checkpoint, map_location="cpu")
    if isinstance(payload, dict):
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            payload = payload["state_dict"]
        elif "model" in payload and isinstance(payload["model"], dict):
            payload = payload["model"]
    if not isinstance(payload, dict):
        raise ValueError(f"unsupported checkpoint payload in {checkpoint}")
    cleaned = {}
    for key, value in payload.items():
        new_key = key[6:] if key.startswith("model.") else key
        cleaned[new_key] = value
    model.load_state_dict(cleaned, strict=False)


class _TorchWarmupModel:
    def __init__(
        self,
        cfg: ModelConfig,
        *,
        image_size: int,
        batch_size: int,
        imagenet_normalize: bool,
    ) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.image_size = image_size
        self.batch_size = batch_size
        self.imagenet_normalize = imagenet_normalize
        self._model = None
        self._transform = None
        self._ready = False

    def _ensure_transform(self):
        if self._transform is not None:
            return self._transform
        Compose, Normalize, Resize, ToTensor = _require_torchvision_transforms()
        steps: list[Any] = [Resize((self.image_size, self.image_size)), ToTensor()]
        if self.imagenet_normalize:
            steps.append(Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        self._transform = Compose(steps)
        return self._transform

    def _image_to_tensor(self, image: np.ndarray):
        torch = _require_torch()
        pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        tensor = self._ensure_transform()(pil)
        return tensor.to(self.device)

    def _batch_from_frames(self, frames: Sequence[Frame]):
        torch = _require_torch()
        tensors = [self._image_to_tensor(frame.image) for frame in frames]
        return torch.stack(tensors, dim=0)

    def _iter_batches(self, frames: Sequence[Frame]):
        for start in range(0, len(frames), self.batch_size):
            yield self._batch_from_frames(frames[start : start + self.batch_size])

    def predict(self, frame: Frame) -> Prediction:
        if not self._ready or self._model is None:
            raise RuntimeError("call fit_warmup() before predict()")
        torch = _require_torch()
        x = self._image_to_tensor(frame.image).unsqueeze(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            output = self._model(x)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return _prediction_from_anomalib_output(output, latency_ms)


class PatchcoreDetector(_TorchWarmupModel):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg, image_size=512, batch_size=8, imagenet_normalize=True)
        if cfg.checkpoint is not None:
            raise ValueError("patchcore does not support checkpoint loading in this pipeline")

    def fit_warmup(self, frames: Sequence[Frame]) -> None:
        torch = _require_torch()
        try:
            from anomalib.models.image.patchcore.torch_model import PatchcoreModel
        except ModuleNotFoundError as exc:
            raise RuntimeError("patchcore requires anomalib and torch dependencies") from exc

        model = PatchcoreModel(
            backbone=self.cfg.backbone or "wide_resnet50_2",
            layers=["layer2", "layer3"],
            pre_trained=True,
            num_neighbors=9,
        ).to(self.device)
        model.train()
        embeddings = []
        for batch in self._iter_batches(frames):
            with torch.no_grad():
                embeddings.append(model(batch).detach())
        if not embeddings:
            raise RuntimeError("patchcore warmup produced no embeddings")
        model.subsample_embedding(torch.cat(embeddings, dim=0), 0.1)
        model.eval()
        self._model = model
        self._ready = True


class PadimDetector(_TorchWarmupModel):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg, image_size=512, batch_size=8, imagenet_normalize=True)
        if cfg.checkpoint is not None:
            raise ValueError("padim does not support checkpoint loading in this pipeline")

    def fit_warmup(self, frames: Sequence[Frame]) -> None:
        torch = _require_torch()
        try:
            from anomalib.models.image.padim.torch_model import PadimModel
        except ModuleNotFoundError as exc:
            raise RuntimeError("padim requires anomalib and torch dependencies") from exc

        model = PadimModel(
            backbone=self.cfg.backbone or "resnet18",
            layers=["layer1", "layer2", "layer3"],
            pre_trained=True,
            n_features=None,
        ).to(self.device)
        model.train()
        for batch in self._iter_batches(frames):
            with torch.no_grad():
                _ = model(batch)
        model.fit()
        model.eval()
        self._model = model
        self._ready = True


class StfpmDetector(_TorchWarmupModel):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg, image_size=512, batch_size=8, imagenet_normalize=True)
        if cfg.checkpoint is not None:
            raise ValueError("stfpm does not support checkpoint loading in this pipeline")

    def fit_warmup(self, frames: Sequence[Frame]) -> None:
        try:
            from anomalib.models.image.stfpm.loss import STFPMLoss
            from anomalib.models.image.stfpm.torch_model import STFPMModel
        except ModuleNotFoundError as exc:
            raise RuntimeError("stfpm requires anomalib and torch dependencies") from exc
        torch = _require_torch()

        model = STFPMModel(
            backbone=self.cfg.backbone or "resnet18",
            layers=["layer1", "layer2", "layer3"],
        ).to(self.device)
        loss_fn = STFPMLoss()
        optimizer = torch.optim.SGD(
            params=model.student_model.parameters(),
            lr=0.4,
            momentum=0.9,
            dampening=0.0,
            weight_decay=0.001,
        )

        model.train()
        for batch in self._iter_batches(frames):
            optimizer.zero_grad(set_to_none=True)
            teacher_features, student_features = model(batch)
            loss = loss_fn(teacher_features, student_features)
            loss.backward()
            optimizer.step()

        model.eval()
        self._model = model
        self._ready = True


class CsflowDetector(_TorchWarmupModel):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg, image_size=512, batch_size=8, imagenet_normalize=True)
        if cfg.checkpoint is not None:
            raise ValueError("csflow does not support checkpoint loading in this pipeline")

    def fit_warmup(self, frames: Sequence[Frame]) -> None:
        try:
            from anomalib.models.image.csflow.loss import CsFlowLoss
            from anomalib.models.image.csflow.torch_model import CsFlowModel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "csflow requires anomalib, torch, and FrEIA dependencies"
            ) from exc
        torch = _require_torch()

        model = CsFlowModel(
            input_size=(self.image_size, self.image_size),
            cross_conv_hidden_channels=1024,
            n_coupling_blocks=4,
            clamp=3,
            num_channels=3,
        ).to(self.device)
        loss_fn = CsFlowLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=2e-4,
            eps=1e-4,
            weight_decay=1e-5,
            betas=(0.5, 0.9),
        )

        model.train()
        for batch in self._iter_batches(frames):
            optimizer.zero_grad(set_to_none=True)
            z_dist, jacobians = model(batch)
            loss = loss_fn(z_dist, jacobians)
            loss.backward()
            optimizer.step()

        model.eval()
        self._model = model
        self._ready = True


class DraemDetector(_TorchWarmupModel):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg, image_size=512, batch_size=8, imagenet_normalize=False)
        if cfg.checkpoint is not None:
            raise ValueError("draem does not support checkpoint loading in this pipeline")

    def fit_warmup(self, frames: Sequence[Frame]) -> None:
        try:
            from anomalib.data.utils.generators.perlin import PerlinAnomalyGenerator
            from anomalib.models.image.draem.loss import DraemLoss
            from anomalib.models.image.draem.torch_model import DraemModel
        except ModuleNotFoundError as exc:
            raise RuntimeError("draem requires anomalib and torch dependencies") from exc
        torch = _require_torch()

        model = DraemModel(sspcab=False).to(self.device)
        loss_fn = DraemLoss()
        augmenter = PerlinAnomalyGenerator(anomaly_source_path=None, blend_factor=(0.2, 1.0))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        model.train()
        for batch in self._iter_batches(frames):
            optimizer.zero_grad(set_to_none=True)
            augmented_image, anomaly_mask = augmenter(batch)
            reconstruction, prediction = model(augmented_image)
            loss = loss_fn(batch, reconstruction, anomaly_mask, prediction)
            loss.backward()
            optimizer.step()

        model.eval()
        self._model = model
        self._ready = True


class Rd4adDetector(_TorchWarmupModel):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg, image_size=512, batch_size=8, imagenet_normalize=True)
        if cfg.checkpoint is not None:
            raise ValueError("rd4ad does not support checkpoint loading in this pipeline")

    def fit_warmup(self, frames: Sequence[Frame]) -> None:
        try:
            from anomalib.models.image.reverse_distillation.anomaly_map import (
                AnomalyMapGenerationMode,
            )
            from anomalib.models.image.reverse_distillation.loss import (
                ReverseDistillationLoss,
            )
            from anomalib.models.image.reverse_distillation.torch_model import (
                ReverseDistillationModel,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError("rd4ad requires anomalib and torch dependencies") from exc
        torch = _require_torch()

        model = ReverseDistillationModel(
            backbone=self.cfg.backbone or "wide_resnet50_2",
            input_size=(self.image_size, self.image_size),
            layers=["layer1", "layer2", "layer3"],
            anomaly_map_mode=AnomalyMapGenerationMode("multiply"),
            pre_trained=True,
        ).to(self.device)
        loss_fn = ReverseDistillationLoss()
        optimizer = torch.optim.Adam(
            list(model.bottleneck.parameters()) + list(model.decoder.parameters()),
            lr=0.005,
            betas=(0.5, 0.999),
        )

        model.train()
        for batch in self._iter_batches(frames):
            optimizer.zero_grad(set_to_none=True)
            encoder_features, decoder_features = model(batch)
            loss = loss_fn(encoder_features, decoder_features)
            loss.backward()
            optimizer.step()

        model.eval()
        self._model = model
        self._ready = True


class SubspaceADDetector:
    GRID = 512

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.model_ckpt = cfg.backbone or "facebook/dinov2-with-registers-large"
        self._processor = None
        self._model = None
        self._pca = None
        self._ready = False

    def _load_extractor(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "subspacead requires transformers, torch, and scikit-learn dependencies"
            ) from exc
        self._processor = AutoImageProcessor.from_pretrained(self.model_ckpt)
        self._model = AutoModel.from_pretrained(self.model_ckpt).eval().to(self.device)

    def fit_warmup(self, frames: Sequence[Frame]) -> None:
        self._load_extractor()
        try:
            from sklearn.decomposition import PCA
        except ModuleNotFoundError as exc:
            raise RuntimeError("subspacead requires scikit-learn") from exc

        all_tokens: list[np.ndarray] = []
        for frame in frames:
            spatial, _ = self._extract_tokens([frame.image])
            all_tokens.append(spatial.reshape(-1, spatial.shape[-1]))

        features = np.concatenate(all_tokens, axis=0)
        self._pca = PCA(n_components=0.99, svd_solver="full", whiten=False)
        self._pca.fit(features)
        self._ready = True

    def predict(self, frame: Frame) -> Prediction:
        if not self._ready or self._pca is None:
            raise RuntimeError("call fit_warmup() before predict()")
        cv2 = _require_cv2()
        t0 = time.perf_counter()
        spatial, _ = self._extract_tokens([frame.image])
        flat = spatial.reshape(-1, spatial.shape[-1])
        recon = self._pca.inverse_transform(self._pca.transform(flat))
        scores = np.sum((flat - recon) ** 2, axis=1)
        side = spatial.shape[1]
        score_map = scores.reshape(side, side).astype(np.float32)
        score_map = cv2.resize(
            score_map,
            (frame.image.shape[1], frame.image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return Prediction(
            score=float(np.mean(np.sort(score_map.reshape(-1))[-max(1, score_map.size // 100) :])),
            anomaly_map=_as_heatmap(score_map),
            latency_ms=latency_ms,
        )

    def _extract_tokens(self, images: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        torch = _require_torch()
        assert self._processor is not None and self._model is not None
        pil_images = [Image.fromarray(img.astype(np.uint8)).convert("RGB") for img in images]
        inputs = self._processor(
            images=pil_images,
            return_tensors="pt",
            do_resize=True,
            size={"height": self.GRID, "width": self.GRID},
            do_center_crop=False,
        ).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        layers = [-12, -13, -14, -15, -16, -17, -18]
        layer_feats = [hidden_states[idx][:, 1:, :] for idx in layers]
        fused = torch.stack(layer_feats, dim=0).mean(dim=0)
        b, n, c = fused.shape
        side = int(math.sqrt(n))
        n_valid = side * side
        fused = fused[:, :n_valid, :]
        spatial = fused.reshape(b, side, side, c).cpu().numpy().astype(np.float32)
        cls_tokens = hidden_states[layers[0]][:, 0, :].cpu().numpy().astype(np.float32)
        return spatial, cls_tokens


class EfficientAdDetector(_TorchWarmupModel):
    """Checkpoint-backed EfficientAD wrapper.

    With the current flat schema we keep this model inference-only: it expects
    `ModelConfig.checkpoint` to point to a trained state dict.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg, image_size=256, batch_size=1, imagenet_normalize=False)
        if cfg.checkpoint is None:
            raise ValueError(
                "efficientad requires ModelConfig.checkpoint in the current flat pipeline"
            )

    def fit_warmup(self, frames: Sequence[Frame]) -> None:
        del frames
        try:
            from anomalib.models.image.efficient_ad.torch_model import (
                EfficientAdModel,
                EfficientAdModelSize,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError("efficientad requires anomalib and torch dependencies") from exc

        model = EfficientAdModel(
            teacher_out_channels=384,
            model_size=EfficientAdModelSize.S,
            padding=False,
            pad_maps=True,
        ).to(self.device)
        _load_state_dict(model, str(self.cfg.checkpoint))
        model.eval()
        self._model = model
        self._ready = True


class PCADetector:
    GRID = 64
    K = 32

    def __init__(self, cfg: ModelConfig) -> None:
        if cfg.checkpoint is not None:
            raise ValueError("PCADetector does not support checkpoint loading")
        self.cfg = cfg
        self._mean: np.ndarray | None = None
        self._components: np.ndarray | None = None
        self._frozen = False

    def fit_warmup(self, frames: Sequence[Frame]) -> None:
        if self._frozen:
            raise RuntimeError("model already warmed up")
        x = np.stack([self._flatten(f.image) for f in frames])
        self._mean = x.mean(axis=0)
        xc = x - self._mean
        k = min(self.K, xc.shape[0], xc.shape[1])
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
        self._components = vt[:k]
        self._frozen = True

    def predict(self, frame: Frame) -> Prediction:
        if not self._frozen:
            raise RuntimeError("call fit_warmup() before predict()")
        assert self._mean is not None and self._components is not None

        t0 = time.perf_counter()
        x = self._flatten(frame.image)
        xc = x - self._mean
        proj = self._components @ xc
        recon = self._components.T @ proj
        residual = xc - recon
        score = float(np.linalg.norm(residual))
        per_pixel = (residual**2).reshape(self.GRID, self.GRID, 3).mean(axis=2)
        amap = self._upsample(per_pixel, frame.image.shape[:2])
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return Prediction(score=score, anomaly_map=amap, latency_ms=latency_ms)

    def _flatten(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        ys = np.linspace(0, h - 1, self.GRID).astype(int)
        xs = np.linspace(0, w - 1, self.GRID).astype(int)
        small = image[np.ix_(ys, xs)]
        if small.ndim == 2:
            small = np.repeat(small[..., None], 3, axis=-1)
        return small.astype(np.float32).flatten() / 255.0

    def _upsample(self, small: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        h, w = target_hw
        ys = np.linspace(0, self.GRID - 1, h).astype(int)
        xs = np.linspace(0, self.GRID - 1, w).astype(int)
        return small[np.ix_(ys, xs)].astype(np.float32)
