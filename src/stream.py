import json
import random
import time
from pathlib import Path
from typing import Iterator, List, Set, Tuple

import numpy as np
from PIL import Image

from .models import Model
from .schemas import Frame, StreamConfig, WarmupConfig


Entry = Tuple[Path, int, str]


def build_warmup_stream(cfg: StreamConfig, warmup_steps: int) -> Iterator[Frame]:
    """Yield the first `warmup_steps` images from the configured input folder."""
    entries = _discover_input_images(cfg)
    selected = entries[:warmup_steps]
    yield from _yield_frames(selected)


def build_stream(cfg: StreamConfig, warmup_steps: int) -> Iterator[Frame]:
    """Yield the post-warmup inference stream from the configured input folder."""
    entries = _discover_input_images(cfg)
    inference_entries = entries[warmup_steps:]
    if cfg.max_frames is not None:
        inference_entries = inference_entries[: cfg.max_frames]
    if not inference_entries:
        raise FileNotFoundError(
            f"no inference frames left under {Path(cfg.input_path).resolve()} "
            f"after reserving {warmup_steps} frames for warmup"
        )
    yield from _yield_frames(inference_entries)


def warmup(model: Model, stream: Iterator[Frame], cfg: WarmupConfig) -> List[Frame]:
    """Consume `cfg.warmup_steps` frames and call `model.fit_warmup`."""
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
    return frames


def _yield_frames(entries: List[Entry]) -> Iterator[Frame]:
    for index, (img_path, label, image_id) in enumerate(entries):
        yield Frame(
            image=_load_image(img_path),
            label=label,
            timestamp=time.time(),
            source_id=str(img_path.as_posix()),
            image_id=image_id,
            index=index,
        )


def _discover_input_images(cfg: StreamConfig) -> List[Entry]:
    root = Path(cfg.input_path)
    if not root.is_dir():
        raise FileNotFoundError(f"stream.input_path is not a directory: {root}")

    subdirs = sorted(p.name for p in root.iterdir() if p.is_dir())
    if subdirs:
        raise ValueError(
            "stream.input_path must be a flat folder of images. "
            "Subfolders are not supported; see README.md for the required input format. "
            f"Found subfolders: {subdirs}"
        )

    extensions = {ext.lower() for ext in cfg.extensions}
    labels = _load_labels(root)
    paths = list(_iter_images(root, extensions))
    if not paths:
        raise FileNotFoundError(
            f"no images with extensions {sorted(extensions)} found under {root.resolve()}"
        )

    seen_ids: Set[str] = set()
    entries: List[Entry] = []
    for img_path in paths:
        image_id = img_path.stem
        if image_id in seen_ids:
            raise ValueError(
                f"duplicate image id {image_id!r} under {root}; "
                "filenames without extension must be unique"
            )
        seen_ids.add(image_id)
        entries.append((img_path, labels.get(image_id, -1), image_id))

    unknown_label_ids = sorted(set(labels) - seen_ids)
    if unknown_label_ids:
        raise ValueError(
            "labels.json contains ids that do not match any input image: "
            f"{unknown_label_ids}"
        )

    if cfg.shuffle:
        random.shuffle(entries)
    return entries


def _load_labels(root: Path) -> dict[str, int]:
    path = root / "labels.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise TypeError("labels.json must be an object mapping image_id to 'OK' or 'NG'")

    labels: dict[str, int] = {}
    for image_id, label in raw.items():
        if not isinstance(image_id, str) or not image_id:
            raise TypeError("labels.json keys must be non-empty image_id strings")
        if label == "OK":
            labels[image_id] = 0
        elif label == "NG":
            labels[image_id] = 1
        else:
            raise ValueError(
                "labels.json values must be exactly 'OK' or 'NG', "
                f"got {label!r} for image_id {image_id!r}"
            )
    return labels


def _iter_images(root: Path, extensions: Set[str]) -> Iterator[Path]:
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        yield path


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))
