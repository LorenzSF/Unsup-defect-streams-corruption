from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from real_time_visual_defect_detection.io.dataset_loader import (
    resolve_dataset_labeled,
    apply_dataset_split,
)
from real_time_visual_defect_detection.preprocessing.standard import (
    read_image_bgr,
    resize,
    normalize_0_1,
)
from real_time_visual_defect_detection.preprocessing.corruption import apply_corruption
from real_time_visual_defect_detection.models.base import BaseModel
from real_time_visual_defect_detection.models.embedding_distance import DummyDistanceModel


def _build_model(model_cfg: Dict[str, Any]) -> BaseModel:
    name = model_cfg.get("name", "dummy_distance")

    if name == "dummy_distance":
        return DummyDistanceModel(threshold=float(model_cfg["threshold"]))

    if name == "rd4ad":
        from real_time_visual_defect_detection.models.rd4ad import RD4ADModel

        rd_cfg = model_cfg.get("rd4ad", {})
        return RD4ADModel(
            image_size=int(rd_cfg.get("image_size", 256)),
            epochs=int(rd_cfg.get("epochs", 200)),
            learning_rate=float(rd_cfg.get("learning_rate", 0.005)),
            batch_size=int(rd_cfg.get("batch_size", 16)),
            threshold=float(model_cfg.get("threshold", 0.5)),
            checkpoint_path=str(rd_cfg.get("checkpoint_path", "data/checkpoints/rd4ad.pth")),
        )

    raise ValueError(
        f"Unknown model name: '{name}'. Supported: 'dummy_distance', 'rd4ad'."
    )


def _save_umap(
    out_dir: Path,
    embeddings: List[np.ndarray],
    labels: List[int],
    paths: List[str],
    scores: List[float],
    defect_types: List[Optional[str]],
) -> None:
    """Run UMAP on *embeddings* and save an interactive HTML scatter plot."""
    from real_time_visual_defect_detection.visualization.plots import plot_embedding_umap

    matrix = np.stack(embeddings, axis=0)          # (N, D)
    fig = plot_embedding_umap(
        embeddings=matrix,
        labels=labels,
        paths=paths,
        scores=scores,
        defect_types=defect_types,
    )
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    fig.write_html(str(plots_dir / "embedding_umap.html"))
    print(f"UMAP plot saved to: {plots_dir / 'embedding_umap.html'}")


def run_pipeline(cfg: Dict[str, Any]) -> Path:
    run_cfg = cfg["run"]
    out_root = Path(run_cfg["output_dir"])
    run_id = f'{run_cfg.get("run_name", "run")}_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}'
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(run_cfg.get("seed", 42))
    np.random.seed(seed)

    # Save config snapshot
    (out_dir / "config_snapshot.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )

    ds = cfg["dataset"]
    samples = resolve_dataset_labeled(ds["source_type"], ds["path"], ds["extract_dir"])
    if len(samples) == 0:
        raise ValueError(
            f"No image files found for dataset source_type='{ds['source_type']}'"
            f" at path='{ds['path']}'."
        )

    split_result = apply_dataset_split(samples, ds.get("split", {}), fallback_seed=seed)
    train_samples = split_result.train
    test_samples = split_result.test

    pre = cfg["preprocessing"]
    do_resize = pre["resize"]["enabled"]
    w, h = int(pre["resize"]["width"]), int(pre["resize"]["height"])

    corr = cfg["corruption"]
    do_corr = bool(corr["enabled"])
    corr_type = corr.get("type", "")
    corr_params = corr.get("params", {})

    model_cfg = cfg["model"]
    model = _build_model(model_cfg)

    # --- Training phase ---------------------------------------------------
    train_paths = [s.path for s in train_samples]
    model.fit(train_paths)

    # --- Inference phase --------------------------------------------------
    rows: List[Dict[str, Any]] = []
    embeddings: List[np.ndarray] = []
    emb_labels: List[int] = []
    emb_paths: List[str] = []
    emb_scores: List[float] = []
    emb_defect_types: List[Optional[str]] = []
    collect_embeddings = True  # will be set False if model returns None

    for sample in test_samples:
        img = read_image_bgr(str(sample.path))
        if do_resize:
            img = resize(img, (w, h))
        if do_corr:
            img = apply_corruption(img, corr_type, corr_params)
        x = normalize_0_1(img)
        out = model.predict(x)

        rows.append(
            {
                "path": str(sample.path),
                "label": sample.label,
                "defect_type": sample.defect_type,
                "score": out.score,
                "pred_is_anomaly": int(out.is_anomaly),
            }
        )

        if collect_embeddings:
            emb = model.get_embedding(x)
            if emb is None:
                collect_embeddings = False
            else:
                embeddings.append(emb)
                emb_labels.append(sample.label)
                emb_paths.append(str(sample.path))
                emb_scores.append(out.score)
                emb_defect_types.append(sample.defect_type)

    (out_dir / "predictions.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    # --- UMAP visualisation -----------------------------------------------
    if embeddings:
        print(f"Generating UMAP for {len(embeddings)} embeddings…")
        _save_umap(out_dir, embeddings, emb_labels, emb_paths, emb_scores, emb_defect_types)

    return out_dir
