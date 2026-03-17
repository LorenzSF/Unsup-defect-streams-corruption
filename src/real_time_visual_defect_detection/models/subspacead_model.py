from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA

from real_time_visual_defect_detection.models.base import BaseModel, ModelOutput


def _topk_mean(arr: np.ndarray, frac: float = 0.01) -> float:
    flat = arr.ravel()
    k = max(1, int(len(flat) * frac))
    idx = np.argpartition(flat, -k)[-k:]
    return float(np.mean(flat[idx]))


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
        return ModelOutput(score=score, is_anomaly=score >= self.threshold)

    def get_embedding(self, x: np.ndarray) -> Optional[np.ndarray]:
        if self._model is None or self._processor is None:
            return None
        img_rgb = (x[..., ::-1] * 255.0).clip(0, 255).astype(np.uint8)
        _, cls = self._extract_tokens([Image.fromarray(img_rgb)])
        return cls.reshape(-1).astype(np.float32)
