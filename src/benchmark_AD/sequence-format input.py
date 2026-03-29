from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import cv2
import json
import math
import random
import zipfile


# ---------------------------------------------------------------------------
# Legacy simulator - yields plain Path objects (backward compatible)
# ---------------------------------------------------------------------------

@dataclass
class StreamSimulator:
    """Iterate over a list of paths one by one, with an optional hard cap.

    Yields plain :class:`~pathlib.Path` objects.  Kept for backward
    compatibility with code that does not need image loading.
    """

    paths: List[Path]
    limit: Optional[int] = None

    def __iter__(self) -> Iterator[Path]:
        count = 0
        for p in self.paths:
            yield p
            count += 1
            if self.limit is not None and count >= self.limit:
                break


# ---------------------------------------------------------------------------
# Rich frame container
# ---------------------------------------------------------------------------

@dataclass
class StreamFrame:
    """One frame emitted by :class:`ImageStreamSimulator`.

    Attributes
    ----------
    image:
        Pixel data as a ``float32`` NumPy array in ``[0, 1]`` (H Ã— W Ã— C,
        BGR channel order).  Shape depends on whether a resize was
        requested.
    label:
        Ground-truth class from the source :class:`~benchmark_AD.data.LabeledSample`:
        ``0`` = good, ``1`` = anomalous, ``-1`` = unknown / unlabeled.
    path:
        Original file path on disk.
    defect_type:
        Defect sub-category string (e.g. ``"scratch"``) when available,
        otherwise ``None``.
    """

    image: np.ndarray
    label: int
    path: Path
    defect_type: Optional[str] = field(default=None)


@dataclass
class ImageStreamSimulator:
    """Stream images from a list of labeled samples, loading lazily at
    iteration time.  Suitable for a real-time display loop or online
    evaluation where the full dataset does not need to fit in memory.

    Parameters
    ----------
    samples:
        Ordered list of :class:`~benchmark_AD.data.LabeledSample`
        objects (from :func:`~benchmark_AD.data.list_labeled_images`
        or :func:`~benchmark_AD.data.resolve_dataset_labeled`).
    resize_wh:
        ``(width, height)`` tuple.  When provided each frame is resized
        before being yielded.  ``None`` skips resizing.
    normalize:
        When ``True`` (default) pixel values are scaled to ``[0, 1]``
        float32.  Set to ``False`` to keep raw ``uint8`` BGR values.
    corruption_cfg:
        Optional dict with ``type`` and ``params`` keys (matching the
        ``corruption`` section of the pipeline config).  When provided
        the corruption is applied **after** resize and **before**
        normalization - exactly the order used in :func:`~benchmark_AD.pipeline.run_pipeline`.
    limit:
        Stop after emitting this many frames.  ``None`` iterates the
        full ``samples`` list.
    skip_errors:
        When ``True``, images that fail to load are silently skipped
        instead of raising an exception.  Useful for large real-world
        datasets that may contain corrupt files.
    """

    samples: List[LabeledSample]
    resize_wh: Optional[Tuple[int, int]] = field(default=None)
    normalize: bool = field(default=True)
    corruption_cfg: Optional[Dict[str, Any]] = field(default=None)
    limit: Optional[int] = field(default=None)
    skip_errors: bool = field(default=False)

    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[StreamFrame]:
        count = 0
        for sample in self.samples:
            if self.limit is not None and count >= self.limit:
                break

            try:
                img = read_image_bgr(str(sample.path))
            except (ValueError, OSError) as exc:
                if self.skip_errors:
                    continue
                raise RuntimeError(
                    f"ImageStreamSimulator: failed to load '{sample.path}': {exc}"
                ) from exc

            if self.resize_wh is not None:
                img = resize(img, self.resize_wh)

            if self.corruption_cfg is not None:
                img = apply_corruption(
                    img,
                    self.corruption_cfg["type"],
                    self.corruption_cfg.get("params", {}),
                )

            pixel_data: np.ndarray = normalize_0_1(img) if self.normalize else img

            yield StreamFrame(
                image=pixel_data,
                label=sample.label,
                path=sample.path,
                defect_type=sample.defect_type,
            )
            count += 1

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pipeline_cfg(
        cls,
        samples: List[LabeledSample],
        cfg: Dict[str, Any],
        limit: Optional[int] = None,
        skip_errors: bool = False,
    ) -> "ImageStreamSimulator":
        """Build an :class:`ImageStreamSimulator` directly from a pipeline
        config dictionary (compatible with ``default.yaml``).

        Parameters
        ----------
        samples:
            Labeled samples to stream.
        cfg:
            Full pipeline config dict as returned by
            :func:`~benchmark_AD.pipeline.load_config`.
        limit:
            Optional frame cap (overrides any limit in the config).
        skip_errors:
            Silently skip unreadable images.
        """
        pre = cfg.get("preprocessing", {})
        resize_cfg = pre.get("resize", {})
        resize_wh: Optional[Tuple[int, int]] = None
        if resize_cfg.get("enabled", False):
            resize_wh = (int(resize_cfg["width"]), int(resize_cfg["height"]))

        corr_cfg = cfg.get("corruption", {})
        corruption_cfg: Optional[Dict[str, Any]] = None
        if corr_cfg.get("enabled", False):
            corruption_cfg = {
                "type": corr_cfg["type"],
                "params": corr_cfg.get("params", {}),
            }

        return cls(
            samples=samples,
            resize_wh=resize_wh,
            normalize=pre.get("normalize", {}).get("enabled", True),
            corruption_cfg=corruption_cfg,
            limit=limit,
            skip_errors=skip_errors,
        )

