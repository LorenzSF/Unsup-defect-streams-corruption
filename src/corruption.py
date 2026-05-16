import dataclasses
import io
import random
from typing import Callable, Iterator

import numpy as np
from PIL import Image, ImageFilter

from .schemas import CorruptionConfig, Frame


def apply_corruption(
    stream: Iterator[Frame], cfg: CorruptionConfig
) -> Iterator[Frame]:
    """Wrap a `Frame` stream and yield corrupted frames.

    If `cfg.enabled` is false the stream passes through unchanged. Otherwise
    each spec is evaluated independently per frame: with probability
    `spec.probability` the corresponding kernel is applied to the running
    image, then the next spec is considered. This lets multiple corruptions
    compose on the same frame (matching the ImageNet-C convention).
    """
    if not cfg.enabled or not cfg.specs:
        yield from stream
        return

    for spec in cfg.specs:
        if spec.kind not in _CORRUPTIONS:
            raise ValueError(
                f"unknown corruption kind '{spec.kind}' "
                f"(supported: {sorted(_CORRUPTIONS)})"
            )

    for frame in stream:
        img = frame.image
        for spec in cfg.specs:
            if random.random() < spec.probability:
                img = _CORRUPTIONS[spec.kind](img, spec.severity)
        if img is frame.image:
            yield frame
        else:
            yield dataclasses.replace(frame, image=img)


def _gaussian_noise(img: np.ndarray, severity: int) -> np.ndarray:
    sigma = [0.04, 0.06, 0.08][severity - 1] * 255.0
    noise = np.random.normal(0.0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _shot_noise(img: np.ndarray, severity: int) -> np.ndarray:
    lam = [60, 25, 12][severity - 1]
    x = img.astype(np.float32) / 255.0
    return np.clip(np.random.poisson(x * lam) / lam * 255.0, 0, 255).astype(np.uint8)


def _motion_blur(img: np.ndarray, severity: int) -> np.ndarray:
    radius = [3, 5, 7][severity - 1]
    n = radius * 2 + 1
    flat = [0.0] * (n * n)
    mid = n // 2
    for j in range(n):
        flat[mid * n + j] = 1.0
    kernel = ImageFilter.Kernel(size=(n, n), kernel=flat, scale=float(n))
    return np.array(Image.fromarray(img).filter(kernel))


def _defocus_blur(img: np.ndarray, severity: int) -> np.ndarray:
    radius = [1.0, 2.0, 3.0][severity - 1]
    pil = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(pil)


def _brightness(img: np.ndarray, severity: int) -> np.ndarray:
    delta = [0.1, 0.2, 0.3][severity - 1]
    x = img.astype(np.float32) / 255.0
    return np.clip((x + delta) * 255.0, 0, 255).astype(np.uint8)


def _contrast(img: np.ndarray, severity: int) -> np.ndarray:
    factor = [0.75, 0.5, 0.4][severity - 1]
    x = img.astype(np.float32) / 255.0
    mean = x.mean(axis=(0, 1), keepdims=True)
    return (np.clip((x - mean) * factor + mean, 0, 1) * 255.0).astype(np.uint8)


def _jpeg_compression(img: np.ndarray, severity: int) -> np.ndarray:
    quality = [80, 65, 58][severity - 1]
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def _pixelate(img: np.ndarray, severity: int) -> np.ndarray:
    factor = [0.6, 0.5, 0.4][severity - 1]
    h, w = img.shape[:2]
    nh, nw = max(1, int(h * factor)), max(1, int(w * factor))
    pil = Image.fromarray(img)
    small = pil.resize((nw, nh), Image.BOX)
    return np.array(small.resize((w, h), Image.NEAREST))


_CORRUPTIONS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "gaussian_noise": _gaussian_noise,
    "shot_noise": _shot_noise,
    "motion_blur": _motion_blur,
    "defocus_blur": _defocus_blur,
    "brightness": _brightness,
    "contrast": _contrast,
    "jpeg_compression": _jpeg_compression,
    "pixelate": _pixelate,
}
