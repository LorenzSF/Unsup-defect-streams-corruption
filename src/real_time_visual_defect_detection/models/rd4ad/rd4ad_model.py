from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset

from ..base import BaseModel, ModelOutput
from .de_resnet import de_wide_resnet50_2
from .resnet import wide_resnet50_2


# ---------------------------------------------------------------------------
# Internal dataset for training
# ---------------------------------------------------------------------------

class _PathDataset(Dataset):
    def __init__(self, paths: List[Path], transform: T.Compose) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(str(self.paths[idx])).convert("RGB")
        return self.transform(img)


# ---------------------------------------------------------------------------
# RD4AD model wrapper
# ---------------------------------------------------------------------------

class RD4ADModel(BaseModel):
    """Reverse Distillation for Anomaly Detection (CVPR 2022).

    Wraps the original rd4ad implementation so it conforms to the
    pipeline's BaseModel interface (fit / predict).

    Parameters
    ----------
    image_size:
        Spatial size used for both training and inference (default 256).
    epochs:
        Number of training epochs (default 200).
    learning_rate:
        Adam learning rate (default 0.005).
    batch_size:
        Mini-batch size for training (default 16).
    threshold:
        Anomaly score threshold for the binary is_anomaly decision.
    checkpoint_path:
        Where to save/load the trained bn+decoder weights.
    """

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        image_size: int = 256,
        epochs: int = 200,
        learning_rate: float = 0.005,
        batch_size: int = 16,
        threshold: float = 0.5,
        checkpoint_path: str = "data/checkpoints/rd4ad.pth",
    ) -> None:
        self.image_size = image_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.threshold = threshold
        self.checkpoint_path = Path(checkpoint_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=self._IMAGENET_MEAN, std=self._IMAGENET_STD),
        ])

        self.encoder = None
        self.bn = None
        self.decoder = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self, pretrained_encoder: bool = True) -> None:
        encoder, bn = wide_resnet50_2(pretrained=pretrained_encoder)
        self.encoder = encoder.to(self.device)
        self.bn = bn.to(self.device)
        self.encoder.eval()
        self.decoder = de_wide_resnet50_2(pretrained=False).to(self.device)

    def _cosine_loss(self, a: list, b: list) -> torch.Tensor:
        cos_loss = torch.nn.CosineSimilarity()
        loss = torch.tensor(0.0, device=self.device)
        for item in range(len(a)):
            loss += torch.mean(
                1 - cos_loss(
                    a[item].view(a[item].shape[0], -1),
                    b[item].view(b[item].shape[0], -1),
                )
            )
        return loss

    def _anomaly_map(self, inputs: list, outputs: list) -> np.ndarray:
        amap = np.zeros([self.image_size, self.image_size], dtype=np.float32)
        for fs, ft in zip(inputs, outputs):
            a = 1 - F.cosine_similarity(fs, ft)
            a = torch.unsqueeze(a, dim=1)
            a = F.interpolate(a, size=self.image_size, mode="bilinear", align_corners=True)
            amap += a[0, 0, :, :].cpu().detach().numpy()
        return amap

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        """Convert a pipeline BGR float [0,1] image to a model-ready tensor."""
        img_rgb = (x[..., ::-1] * 255).clip(0, 255).astype(np.uint8)
        return self._transform(Image.fromarray(img_rgb)).unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, train_paths: List[Path]) -> None:
        """Train bn + decoder on *train_paths* (normal images).

        The encoder (wide_resnet50_2, ImageNet pretrained) stays frozen.
        Saves a checkpoint to ``self.checkpoint_path`` when done.
        """
        self._build_model(pretrained_encoder=True)

        dataset = _PathDataset(train_paths, self._transform)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        optimizer = torch.optim.Adam(
            list(self.decoder.parameters()) + list(self.bn.parameters()),
            lr=self.learning_rate,
            betas=(0.5, 0.999),
        )

        for epoch in range(self.epochs):
            self.bn.train()
            self.decoder.train()
            losses = []
            for batch in loader:
                batch = batch.to(self.device)
                with torch.no_grad():
                    inputs = self.encoder(batch)
                outputs = self.decoder(self.bn(inputs))
                loss = self._cosine_loss(inputs, outputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            print(
                f"RD4AD epoch [{epoch + 1}/{self.epochs}] loss: {np.mean(losses):.4f}"
            )

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"bn": self.bn.state_dict(), "decoder": self.decoder.state_dict()},
            self.checkpoint_path,
        )
        self._is_fitted = True

    def load_checkpoint(self) -> None:
        """Load bn + decoder weights from ``self.checkpoint_path``."""
        if self.encoder is None:
            self._build_model(pretrained_encoder=True)
        ckp = torch.load(self.checkpoint_path, map_location=self.device)
        self.decoder.load_state_dict(ckp["decoder"])
        self.bn.load_state_dict(ckp["bn"])
        self._is_fitted = True

    def predict(self, x: np.ndarray) -> ModelOutput:
        """Predict anomaly score for a single image.

        Parameters
        ----------
        x:
            Float64 numpy array in BGR channel order, values in [0, 1],
            as produced by the pipeline's ``normalize_0_1``.

        Returns
        -------
        ModelOutput
            ``score`` is the max value of the Gaussian-smoothed anomaly map.
            ``is_anomaly`` is ``True`` when ``score > threshold``.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "RD4ADModel is not ready. Call fit() or load_checkpoint() first."
            )

        self.encoder.eval()
        self.bn.eval()
        self.decoder.eval()

        tensor = self._to_tensor(x)

        with torch.no_grad():
            inputs = self.encoder(tensor)
            outputs = self.decoder(self.bn(inputs))

        amap = self._anomaly_map(inputs, outputs)
        amap = gaussian_filter(amap, sigma=4)
        score = float(np.max(amap))

        return ModelOutput(score=score, is_anomaly=score > self.threshold)

    def get_embedding(self, x: np.ndarray) -> Optional[np.ndarray]:
        """Return a 1-D feature embedding derived from the BN bottleneck.

        Runs the encoder and BN layer, then applies global average pooling
        to produce a compact, fixed-length vector that summarises the image
        representation learned by the model.

        Parameters
        ----------
        x:
            Float64 numpy array in BGR channel order, values in [0, 1].

        Returns
        -------
        numpy.ndarray
            1-D float32 array of shape ``(C,)`` where *C* is the number of
            channels in the BN output feature map.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "RD4ADModel is not ready. Call fit() or load_checkpoint() first."
            )

        self.encoder.eval()
        self.bn.eval()

        tensor = self._to_tensor(x)

        with torch.no_grad():
            inputs = self.encoder(tensor)
            bn_out = self.bn(inputs)          # (1, C, H, W)

        # Global average pool → (C,)
        embedding = bn_out.mean(dim=[2, 3]).squeeze(0).cpu().numpy().astype(np.float32)
        return embedding
