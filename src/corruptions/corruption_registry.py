"""Synthetic image corruptions for robustness benchmarking.

Each corruption is a standalone function with the contract
``(image: np.ndarray, severity: int) -> np.ndarray`` where
``severity ∈ {1, 2, 3, 4, 5}``. The output preserves the input shape and
dtype so the pipeline can swap a clean image for a corrupted one without
any branching downstream.

The registry is intentionally restricted to the three corruption types
named in PLAN.md §1.2 — ``gaussian_blur``, ``motion_blur``,
``jpeg_compression`` — chosen as the minimal set that covers the
visual-degradation modes a factory line is most likely to encounter
(focus loss, camera shake, transmission/storage compression).

Stochastic corruptions seed their RNG from a hash of the image bytes so
the same image always receives the same perturbation across reruns;
this keeps benchmark numbers reproducible without a global seed.

The public entry point is :func:`get_corruption`.
"""

from __future__ import annotations

import hashlib
import io
from typing import Callable, Dict, Tuple

import cv2
import numpy as np
from PIL import Image


SEVERITY_LEVELS: Tuple[int, ...] = (1, 2, 3, 4, 5)


def _validate_severity(severity: int) -> int:
    if severity not in SEVERITY_LEVELS:
        raise ValueError(
            f"severity must be one of {SEVERITY_LEVELS}, got {severity!r}."
        )
    return severity


def _image_rng(image: np.ndarray) -> np.random.Generator:
    # Hash the raw bytes so repeated runs over the same image produce the
    # same noise/position; avoids leaking a global seed into the pipeline.
    digest = hashlib.blake2b(image.tobytes(), digest_size=8).digest()
    seed = int.from_bytes(digest, "little", signed=False)
    return np.random.default_rng(seed)


def gaussian_blur(image: np.ndarray, severity: int) -> np.ndarray:
    sigmas = {1: 1.0, 2: 1.8, 3: 2.6, 4: 3.6, 5: 5.0}
    sigma = sigmas[_validate_severity(severity)]
    return cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)


def motion_blur(image: np.ndarray, severity: int) -> np.ndarray:
    sizes = {1: 5, 2: 9, 3: 13, 4: 17, 5: 23}
    ksize = sizes[_validate_severity(severity)]

    # Build a horizontal-line kernel and rotate it to a deterministic angle
    # derived from the image, so the direction is reproducible per sample.
    rng = _image_rng(image)
    angle_deg = float(rng.uniform(0.0, 180.0))

    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (ksize, ksize))
    kernel /= max(float(kernel.sum()), 1e-6)

    return cv2.filter2D(image, ddepth=-1, kernel=kernel)


def jpeg_compression(image: np.ndarray, severity: int) -> np.ndarray:
    qualities = {1: 70, 2: 50, 3: 35, 4: 20, 5: 10}
    quality = qualities[_validate_severity(severity)]

    # Use Pillow so the function is dtype-agnostic and works for both BGR
    # and RGB callers; pipeline reads BGR via cv2 so we round-trip in BGR.
    if image.ndim == 3 and image.shape[2] == 3:
        pil = Image.fromarray(image[..., ::-1])  # BGR -> RGB
        with io.BytesIO() as buf:
            pil.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            decoded = np.array(Image.open(buf).convert("RGB"))
        return decoded[..., ::-1].astype(image.dtype)  # RGB -> BGR

    pil = Image.fromarray(image)
    with io.BytesIO() as buf:
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        decoded = np.array(Image.open(buf))
    return decoded.astype(image.dtype)


_CORRUPTIONS: Dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "jpeg_compression": jpeg_compression,
}


def available_corruptions() -> Tuple[str, ...]:
    return tuple(sorted(_CORRUPTIONS))


def get_corruption(name: str, severity: int) -> Callable[[np.ndarray], np.ndarray]:
    """Return a per-image callable that applies *name* at the given *severity*.

    Validating both arguments at factory time means the pipeline fails fast
    on a misconfigured run instead of crashing mid-inference on the first
    test image.
    """
    if name not in _CORRUPTIONS:
        supported = ", ".join(available_corruptions())
        raise ValueError(f"Unknown corruption '{name}'. Supported: {supported}.")
    _validate_severity(severity)
    func = _CORRUPTIONS[name]
    return lambda image: func(image, severity)
