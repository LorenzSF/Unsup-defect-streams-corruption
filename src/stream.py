import random
import time
from pathlib import Path
from typing import Iterator, List, Set, Tuple

import numpy as np
from PIL import Image

from .models import Model
from .schemas import Frame, StreamConfig, WarmupConfig


def build_warmup_stream(cfg: StreamConfig, warmup_steps: int) -> Iterator[Frame]:
    """Yield up to `warmup_steps` OK frames, randomly chosen.

    Used to bootstrap the model's normal-data state. Only OK frames are
    returned; NG frames are reserved for the inference stream so warmup
    fits on uncontaminated data.
    """
    ok_entries, _ = _discover_ok_ng(cfg)
    if not ok_entries:
        raise FileNotFoundError(
            f"no OK images found for {cfg.dataset}/{cfg.category} under "
            f"{Path(cfg.data_root).resolve()}"
        )
    if cfg.shuffle:
        random.shuffle(ok_entries)
    selected = ok_entries[:warmup_steps]
    yield from _yield_frames(selected)


def build_stream(cfg: StreamConfig, warmup_steps: int) -> Iterator[Frame]:
    """Yield the inference stream: remaining OK + all NG, mixed.

    The first `warmup_steps` random OK frames are excluded (they go to
    `build_warmup_stream`). The rest is shuffled together so OK and NG
    frames are interleaved. `cfg.max_frames` truncates the final list.
    """
    ok_entries, ng_entries = _discover_ok_ng(cfg)
    if cfg.shuffle:
        random.shuffle(ok_entries)
    rest_ok = ok_entries[warmup_steps:]
    inference_entries = rest_ok + ng_entries
    if not inference_entries:
        raise FileNotFoundError(
            f"no inference frames left for {cfg.dataset}/{cfg.category} "
            f"after reserving {warmup_steps} OK frames for warmup"
        )
    if cfg.shuffle:
        random.shuffle(inference_entries)
    if cfg.max_frames is not None:
        inference_entries = inference_entries[: cfg.max_frames]
    yield from _yield_frames(inference_entries)


def warmup(model: Model, stream: Iterator[Frame], cfg: WarmupConfig) -> List[Frame]:
    """Consume `cfg.warmup_steps` frames and call `model.fit_warmup`.

    Returns the consumed frames so downstream code (e.g. threshold
    calibration) can re-score them after fit without re-iterating the
    stream.
    """
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


def _yield_frames(entries: List[Tuple[Path, int]]) -> Iterator[Frame]:
    for index, (img_path, label) in enumerate(entries):
        yield Frame(
            image=_load_image(img_path),
            label=label,
            timestamp=time.time(),
            source_id=str(img_path.as_posix()),
            index=index,
        )


def _discover_ok_ng(
    cfg: StreamConfig,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """Return (ok_entries, ng_entries) under `cfg.data_root / cfg.category`.

    Convention shared by Real-IAD, MVTec-AD, VisA and similar industrial
    datasets:
        <data_root>/<category>/OK/<specimen>/*.{ext}
        <data_root>/<category>/NG/<defect>/<specimen>/*.{ext}

    or with one extra nested `<category>/` level:
        <data_root>/<category>/<category>/OK/...
        <data_root>/<category>/<category>/NG/...
    """
    root = _resolve_category_root(cfg)
    extensions = {ext.lower() for ext in cfg.extensions}

    ok_entries: List[Tuple[Path, int]] = []
    ok_dir = root / "OK"
    if ok_dir.is_dir():
        for img_path in _iter_images(ok_dir, extensions):
            ok_entries.append((img_path, 0))

    ng_entries: List[Tuple[Path, int]] = []
    ng_dir = root / "NG"
    if ng_dir.is_dir():
        for defect_dir in sorted(p for p in ng_dir.iterdir() if p.is_dir()):
            for img_path in _iter_images(defect_dir, extensions):
                ng_entries.append((img_path, 1))

    return ok_entries, ng_entries


def _resolve_category_root(cfg: StreamConfig) -> Path:
    base = Path(cfg.data_root)
    direct = base / cfg.category
    nested = direct / cfg.category
    for candidate in (direct, nested):
        if (candidate / "OK").is_dir() or (candidate / "NG").is_dir():
            return candidate
    raise FileNotFoundError(
        f"missing category folder for '{cfg.category}'. "
        f"Expected '{direct}' or '{nested}' with OK/ and/or NG/ subfolders."
    )


def _iter_images(root: Path, extensions: Set[str]) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        yield path


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))
