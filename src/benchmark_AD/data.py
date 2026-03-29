from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import json
import math
import random
import zipfile

import cv2
import numpy as np


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Well-known names for JSON label files that may ship alongside a dataset.
_LABEL_JSON_NAMES = {
    "labels.json",
    "annotations.json",
    "ground_truth.json",
    "metadata.json",
}

_BAD_DIR_NAMES = ("bad", "defects", "defective", "anomaly", "anomalous")
_JSON_PATH_KEYS = ("path", "file", "filename", "image", "name")
_JSON_LABEL_KEYS = ("label", "class", "category", "is_anomaly", "split")


# ---------------------------------------------------------------------------
# Labeled sample container
# ---------------------------------------------------------------------------

@dataclass
class LabeledSample:
    """One image entry with an optional class label and defect category.

    Attributes
    ----------
    path:
        Absolute (or relative-to-cwd) path of the image file.
    label:
        0 = good/normal, 1 = anomalous/bad, -1 = unknown (unlabeled).
    defect_type:
        Subdirectory name inside ``bad/`` when defect categories are
        organised into sub-folders (e.g. ``bad/scratch/``).  ``None``
        for good images, unlabeled images, or bad images that are not
        further sub-categorised.
    """

    path: Path
    label: int = -1
    defect_type: Optional[str] = field(default=None)


# ---------------------------------------------------------------------------
# Existing flat-list helpers (unchanged — backward compatible)
# ---------------------------------------------------------------------------

def _iter_image_paths(root_dir: Path, recursive: bool = True) -> Iterator[Path]:
    """Yield sorted image files under *root_dir*."""
    walker = root_dir.rglob("*") if recursive else root_dir.iterdir()
    # Scan entries in deterministic order and keep supported image files only.
    for p in sorted(walker):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def _resolve_source_root(source_type: str, path: str, extract_dir: str) -> Path:
    if source_type == "zip":
        return extract_zip(path, extract_dir)
    if source_type == "folder":
        return Path(path)
    raise ValueError(f"Unknown source_type: {source_type}. Use 'zip' or 'folder'.")

def list_images(root_dir: str | Path) -> List[Path]:
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

    return list(_iter_image_paths(root_dir))


def resolve_dataset(source_type: str, path: str, extract_dir: str) -> List[Path]:
    """
    Returns a list of image paths.
    - source_type="zip": extracts zip to extract_dir then lists images
    - source_type="folder": lists images under path
    """
    return list_images(_resolve_source_root(source_type, path, extract_dir))


# ---------------------------------------------------------------------------
# JSON label-file helpers
# ---------------------------------------------------------------------------

def _parse_label_value(val: Any) -> int:
    """Normalise a raw label value to 0, 1, or -1 (unknown)."""
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, int):
        return val if val in (0, 1) else -1
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("0", "good", "normal", "ok"):
            return 0
        if v in ("1", "bad", "anomaly", "anomalous", "defect", "defective"):
            return 1
    return -1


def _load_label_json(json_path: Path) -> Dict[str, int]:
    """Parse a label JSON file into a ``{filename: label}`` mapping.

    Supported formats
    -----------------
    * Dict  ``{"img.png": 0, "img2.png": "bad",}``
    * List  ``[{"path": "img.png", "label": 1},]``

    Keys are matched by *filename only* (basename), so callers need not
    worry about directory prefixes inside the JSON.
    """
    data: Any = json.loads(json_path.read_text(encoding="utf-8"))
    result: Dict[str, int] = {}

    if isinstance(data, dict):
        # Convert mapping entries to a basename -> normalized-label table.
        for k, v in data.items():
            result[Path(k).name] = _parse_label_value(v)

    elif isinstance(data, list):
        # Read record-style entries and extract path/label fields when available.
        for item in data:
            if not isinstance(item, dict):
                continue
            path_key = next((k for k in _JSON_PATH_KEYS if k in item), None)
            label_key = next((k for k in _JSON_LABEL_KEYS if k in item), None)
            if path_key and label_key:
                result[Path(item[path_key]).name] = _parse_label_value(item[label_key])

    return result


def _find_label_json(root_dir: Path) -> Optional[Path]:
    """Return the label JSON file inside *root_dir*, or ``None``.

    Search order:
    1. Any of the well-known names in ``_LABEL_JSON_NAMES``.
    2. The single ``.json`` file present at root level (if exactly one
       exists), as a last-resort heuristic.
    """
    # Check known label-file names first.
    for name in _LABEL_JSON_NAMES:
        candidate = root_dir / name
        if candidate.is_file():
            return candidate

    # Heuristic: exactly one JSON at root level -> treat it as labels file.
    json_files = [
        p for p in root_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".json"
    ]
    if len(json_files) == 1:
        return json_files[0]

    return None


# ---------------------------------------------------------------------------
# Labeled dataset helpers
# ---------------------------------------------------------------------------

def list_labeled_images(root_dir: str | Path) -> List[LabeledSample]:
    """Return image samples with labels, using the best available source.

    Detection priority
    ------------------
    1. **JSON label file** if a recognised label file exists at
       *root_dir* (see :func:`_find_label_json`), every discovered image
       is matched by filename; unmatched images get label ``-1``.
    2. **good / bad subdirectories** if ``good/`` or ``bad/``
       subdirectories are present, images are labelled accordingly.
       Sub-folders inside ``bad/`` (e.g. ``bad/scratch/``) are captured
       as ``defect_type``.
    3. **Flat fallback** all images are returned with label ``-1``
       (unlabeled), reusing :func:`list_images`.
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

    samples: List[LabeledSample] = []

    # ------------------------------------------------------------------
    # Priority 1: JSON label file
    # ------------------------------------------------------------------
    label_json = _find_label_json(root_dir)
    if label_json is not None:
        label_map = _load_label_json(label_json)
        # Match each discovered image with its JSON label by filename.
        for p in _iter_image_paths(root_dir):
            samples.append(LabeledSample(path=p, label=label_map.get(p.name, -1)))
        if samples:
            return samples

    # ------------------------------------------------------------------
    # Priority 2: good/ bad/ subdirectory convention
    # Recognised names for the defect folder (checked in order).
    # ------------------------------------------------------------------
    good_dir = root_dir / "good"
    bad_dir = next(
        (root_dir / name for name in _BAD_DIR_NAMES if (root_dir / name).is_dir()),
        root_dir / "bad",  # default (will evaluate has_bad=False if none exist)
    )
    has_good = good_dir.is_dir()
    has_bad = bad_dir.is_dir()

    if has_good or has_bad:
        if has_good:
            # Collect all normal images from the good directory.
            for p in _iter_image_paths(good_dir):
                samples.append(LabeledSample(path=p, label=0, defect_type=None))

        if has_bad:
            # Images directly in bad/ (no defect sub-category)
            for p in _iter_image_paths(bad_dir, recursive=False):
                samples.append(LabeledSample(path=p, label=1, defect_type=None))

            # Sub-folders inside bad/ represent defect categories
            for subdir in sorted(d for d in bad_dir.iterdir() if d.is_dir()):
                defect_type = subdir.name
                # Collect all defect images for the current category folder.
                for p in _iter_image_paths(subdir):
                    samples.append(LabeledSample(path=p, label=1, defect_type=defect_type))

        return samples

    # ------------------------------------------------------------------
    # Priority 3 : flat fallback (unlabeled)
    # ------------------------------------------------------------------
    # Return every image as unlabeled when no labeling source is available.
    for p in _iter_image_paths(root_dir):
        samples.append(LabeledSample(path=p, label=-1, defect_type=None))

    return samples


def resolve_dataset_labeled(
    source_type: str,
    path: str,
    extract_dir: str,
) -> List[LabeledSample]:
    """Labeled counterpart of :func:`resolve_dataset`.

    Returns a list of :class:`LabeledSample` objects instead of plain
    paths.  The extraction logic is identical to the original function.
    """
    return list_labeled_images(_resolve_source_root(source_type, path, extract_dir))


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    """Outcome of :func:`apply_dataset_split`.

    Attributes
    ----------
    train:
        Samples intended for model training.  When ``train_on_good_only``
        is ``True`` this contains only label-0 and label-(-1) entries.
    test:
        Samples intended for evaluation / validation.  Always contains
        the labeled samples (label in {0, 1}) so that metrics can be
        computed.  May also contain a portion of unlabeled samples when
        ``train_on_good_only`` is ``False``.
    """

    train: List[LabeledSample]
    val: List[LabeledSample]
    test: List[LabeledSample]


def _cap_and_shuffle(
    samples: List[LabeledSample],
    max_count: Optional[int],
    rng: random.Random,
) -> List[LabeledSample]:
    """Shuffle *samples* in-place (copy) and apply an optional cap."""
    out = list(samples)
    rng.shuffle(out)
    if max_count is not None:
        out = out[:max_count]
    return out


def _split_group(
    samples: List[LabeledSample],
    test_ratio: float,
    rng: random.Random,
) -> Tuple[List[LabeledSample], List[LabeledSample]]:
    """Split a single group into (train, test) honoring *test_ratio*."""
    out = list(samples)
    rng.shuffle(out)
    n_test = math.ceil(len(out) * test_ratio)
    return out[n_test:], out[:n_test]


def _partition_by_label(
    samples: List[LabeledSample],
) -> Tuple[List[LabeledSample], List[LabeledSample], List[LabeledSample]]:
    """Split samples into (good, bad, unlabeled) lists."""
    good = [s for s in samples if s.label == 0]
    bad = [s for s in samples if s.label == 1]
    unlabeled = [s for s in samples if s.label == -1]
    return good, bad, unlabeled


def _split_stratified_groups(
    good: List[LabeledSample],
    bad: List[LabeledSample],
    unlabeled: List[LabeledSample],
    ratio: float,
    rng: random.Random,
) -> Tuple[List[LabeledSample], List[LabeledSample]]:
    """Split each label group independently and merge back."""
    train_good, test_good = _split_group(good, ratio, rng)
    train_bad, test_bad = _split_group(bad, ratio, rng)
    train_unlabeled, test_unlabeled = _split_group(unlabeled, ratio, rng)
    train = train_good + train_bad + train_unlabeled
    test = test_good + test_bad + test_unlabeled
    return train, test


def apply_dataset_split(
    samples: List[LabeledSample],
    split_cfg: Dict[str, Any],
    fallback_seed: int = 42,
) -> SplitResult:
    """Apply composition limits and a train/test partition to *samples*.

    Parameters
    ----------
    samples:
        Full list of :class:`LabeledSample` objects as returned by
        :func:`list_labeled_images` or :func:`resolve_dataset_labeled`.
    split_cfg:
        The ``dataset.split`` configuration dictionary.  All keys are
        optional and fall back to sensible defaults when absent.

        * ``max_good`` *(int | null)* cap on label-0 samples.
        * ``max_bad`` *(int | null)* cap on label-1 samples.
        * ``max_unlabeled`` *(int | null)* cap on label-(-1) samples.
        * ``bad_fraction`` *(float | null)* if set, keeps
          ``bad = ceil(n_good * bad_fraction)`` samples, overriding
          ``max_bad``.
        * ``test_ratio`` *(float)*  fraction of each group held out
          for evaluation (default ``0.2``).
        * ``stratify`` *(bool)*  split each label group independently
          so class proportions are maintained (default ``True``).
        * ``train_on_good_only`` *(bool)*  when ``True``, all bad
          samples are routed to the test set (suitable for one-class /
          unsupervised anomaly detection models, default ``True``).
        * ``seed`` *(int | null)*  RNG seed; falls back to
          *fallback_seed* when absent or ``None``.
    fallback_seed:
        Seed used when ``split_cfg`` does not specify one.

    Returns
    -------
    SplitResult
        Named trio of ``(train, val, test)`` sample lists.
    """
    seed_val = split_cfg.get("seed")
    seed = int(seed_val) if seed_val is not None else fallback_seed
    rng = random.Random(seed)

    test_ratio = float(split_cfg.get("test_ratio", 0.2))
    val_ratio = float(split_cfg.get("val_ratio", 0.1))
    stratify = bool(split_cfg.get("stratify", True))
    train_on_good_only = bool(split_cfg.get("train_on_good_only", True))

    # --- Separate by group ------------------------------------------------
    good, bad, unlabeled = _partition_by_label(samples)

    # --- Apply composition caps -------------------------------------------
    max_good = split_cfg.get("max_good")
    max_bad = split_cfg.get("max_bad")
    max_unlabeled = split_cfg.get("max_unlabeled")
    bad_fraction = split_cfg.get("bad_fraction")

    good = _cap_and_shuffle(good, int(max_good) if max_good is not None else None, rng)
    unlabeled = _cap_and_shuffle(
        unlabeled, int(max_unlabeled) if max_unlabeled is not None else None, rng
    )

    # bad_fraction overrides max_bad
    if bad_fraction is not None:
        derived_max_bad = math.ceil(len(good) * float(bad_fraction))
        bad = _cap_and_shuffle(bad, derived_max_bad, rng)
    else:
        bad = _cap_and_shuffle(bad, int(max_bad) if max_bad is not None else None, rng)

    # --- Train / test partition -------------------------------------------
    if train_on_good_only:
        # Bad samples go entirely into test no train leakage of anomalies.
        train_good, test_good = _split_group(good, test_ratio, rng)
        train_unlabeled, test_unlabeled = _split_group(unlabeled, test_ratio, rng)
        train = train_good + train_unlabeled
        test = test_good + test_unlabeled + bad
    elif stratify:
        # Stratified: split each class group independently.
        train, test = _split_stratified_groups(good, bad, unlabeled, test_ratio, rng)
    else:
        # Non-stratified: pool everything and split once.
        all_samples = good + bad + unlabeled
        train, test = _split_group(all_samples, test_ratio, rng)

    # Optional validation holdout taken from the train split.
    val: List[LabeledSample] = []
    if val_ratio > 0.0 and len(train) > 0:
        if stratify:
            tr_good, tr_bad, tr_unl = _partition_by_label(train)
            train, val = _split_stratified_groups(tr_good, tr_bad, tr_unl, val_ratio, rng)
        else:
            train, val = _split_group(train, val_ratio, rng)

    # Final shuffle so splits are not class-ordered.
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return SplitResult(train=train, val=val, test=test)


def _validate_zip_member(member_name: str, extract_dir: Path) -> None:
    member_path = Path(member_name)
    if member_path.is_absolute():
        raise ValueError(f"Unsafe ZIP member (absolute path): {member_name}")

    resolved_target = (extract_dir / member_path).resolve()
    try:
        resolved_target.relative_to(extract_dir)
    except ValueError as exc:
        raise ValueError(f"Unsafe ZIP member (path traversal): {member_name}") from exc


def extract_zip(zip_path: str | Path, extract_dir: str | Path) -> Path:
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    extract_dir = Path(extract_dir).resolve()
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Validate each archive member before extraction.
        for info in zf.infolist():
            _validate_zip_member(info.filename, extract_dir)
        zf.extractall(extract_dir)

    return extract_dir


def apply_corruption(img: np.ndarray, corruption_type: str, params: Dict[str, Any]) -> np.ndarray:
    if corruption_type == "gaussian_noise":
        sigma = float(params.get("sigma", 10.0))
        noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
        out = img.astype(np.float32) + noise
        return np.clip(out, 0, 255).astype(img.dtype)

    if corruption_type == "gaussian_blur":
        ksize = int(params.get("ksize", 5))
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(img, (ksize, ksize), 0)

    if corruption_type == "resolution_reduction":
        scale = float(params.get("scale", 0.5))
        h, w = img.shape[:2]
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    raise ValueError(f"Unknown corruption_type: {corruption_type}")


def read_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def resize(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def normalize_0_1(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0
