import random
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

from .schemas import Frame, StreamConfig


_REAL_IAD_EXTS = {".jpg", ".jpeg"}
_DATA_ROOT = Path("data") / "Real-IAD_dataset" / "realiad_1024"


def build_stream(cfg: StreamConfig) -> Iterator[Frame]:
    """Yield `Frame`s from the configured dataset, one image at a time.

    The flat migration uses the simplest possible Real-IAD contract:
    one extracted category folder under `data/Real-IAD_dataset/realiad_1024/`,
    streamed in full. `cfg.split` is accepted for schema compatibility but
    ignored for Real-IAD because the whole category is the stream.
    """
    if cfg.dataset != "real_iad":
        raise ValueError(
            f"unsupported dataset '{cfg.dataset}' (only 'real_iad' is implemented)"
        )

    entries = _discover_real_iad(cfg.category)
    if not entries:
        raise FileNotFoundError(
            f"no images found for real_iad/{cfg.category} "
            f"under {_DATA_ROOT.resolve()}"
        )

    if cfg.shuffle:
        random.shuffle(entries)
    if cfg.max_frames is not None:
        entries = entries[: cfg.max_frames]

    for index, (img_path, mask_path, label) in enumerate(entries):
        yield Frame(
            image=_load_image(img_path),
            label=label,
            mask=_load_mask(mask_path) if mask_path is not None else None,
            timestamp=time.time(),
            source_id=str(img_path.as_posix()),
            index=index,
        )


def _discover_real_iad(category: str) -> List[Tuple[Path, Optional[Path], int]]:
    """Return a sorted list of (image_path, None, label) tuples.

    Accepted on-disk layouts:
        data/Real-IAD_dataset/realiad_1024/<category>/OK/<specimen>/*.jpg
        data/Real-IAD_dataset/realiad_1024/<category>/NG/<defect>/<specimen>/*.jpg

    or the same with one extra nested `<category>/` level:
        .../<category>/<category>/OK/...
        .../<category>/<category>/NG/...
    """
    root = _resolve_category_root(category)
    entries: List[Tuple[Path, Optional[Path], int]] = []
    ok_dir = root / "OK"
    if ok_dir.is_dir():
        for img_path in _iter_images(ok_dir):
            entries.append((img_path, None, 0))

    ng_dir = root / "NG"
    if ng_dir.is_dir():
        for defect_dir in sorted(p for p in ng_dir.iterdir() if p.is_dir()):
            for img_path in _iter_images(defect_dir):
                entries.append((img_path, None, 1))

    return entries


def _resolve_category_root(category: str) -> Path:
    direct = _DATA_ROOT / category
    nested = direct / category
    for candidate in (direct, nested):
        if (candidate / "OK").is_dir() or (candidate / "NG").is_dir():
            return candidate
    raise FileNotFoundError(
        f"missing Real-IAD category folder for '{category}'. "
        f"Expected '{direct}' or '{nested}' with OK/ and/or NG/ subfolders."
    )


def _iter_images(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _REAL_IAD_EXTS:
            continue
        yield path


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))
