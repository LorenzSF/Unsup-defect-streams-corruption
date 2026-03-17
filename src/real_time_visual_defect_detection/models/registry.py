from __future__ import annotations

from typing import Any, Dict

from real_time_visual_defect_detection.models.base import BaseModel
from real_time_visual_defect_detection.models.embedding_distance import DummyDistanceModel


def _build_dummy(model_cfg: Dict[str, Any], runtime_cfg: Dict[str, Any]) -> BaseModel:
    return DummyDistanceModel(threshold=float(model_cfg.get("threshold", 0.5)))


def _build_rd4ad(model_cfg: Dict[str, Any], runtime_cfg: Dict[str, Any]) -> BaseModel:
    from real_time_visual_defect_detection.models.rd4ad import RD4ADModel

    rd_cfg = model_cfg.get("rd4ad", {})
    return RD4ADModel(
        image_size=int(rd_cfg.get("image_size", 256)),
        epochs=int(rd_cfg.get("epochs", 200)),
        learning_rate=float(rd_cfg.get("learning_rate", 0.005)),
        batch_size=int(rd_cfg.get("batch_size", 16)),
        threshold=float(model_cfg.get("threshold", 0.5)),
        device=str(runtime_cfg.get("resolved_device", "cpu")),
        checkpoint_path=str(rd_cfg.get("checkpoint_path", "data/checkpoints/rd4ad.pth")),
    )


def _build_anomalib_patchcore(
    model_cfg: Dict[str, Any], runtime_cfg: Dict[str, Any]
) -> BaseModel:
    from real_time_visual_defect_detection.models.anomalib_patchcore_model import (
        AnomalibPatchcoreModel,
    )

    an_cfg = model_cfg.get("anomalib", {})
    return AnomalibPatchcoreModel(
        threshold=float(model_cfg.get("threshold", 0.5)),
        device=str(runtime_cfg.get("resolved_device", "cpu")),
        image_size=int(an_cfg.get("image_size", 256)),
        batch_size=int(an_cfg.get("batch_size", 8)),
        pre_trained=bool(an_cfg.get("pre_trained", False)),
        backbone=str(an_cfg.get("backbone", "wide_resnet50_2")),
        layers=an_cfg.get("layers", ["layer2", "layer3"]),
        coreset_sampling_ratio=float(an_cfg.get("coreset_sampling_ratio", 0.1)),
        num_neighbors=int(an_cfg.get("num_neighbors", 9)),
    )


def _build_anomalib_padim(
    model_cfg: Dict[str, Any], runtime_cfg: Dict[str, Any]
) -> BaseModel:
    from real_time_visual_defect_detection.models.anomalib_padim_model import (
        AnomalibPadimModel,
    )

    an_cfg = model_cfg.get("anomalib", {})
    return AnomalibPadimModel(
        threshold=float(model_cfg.get("threshold", 0.5)),
        device=str(runtime_cfg.get("resolved_device", "cpu")),
        image_size=int(an_cfg.get("image_size", 256)),
        batch_size=int(an_cfg.get("batch_size", 8)),
        pre_trained=bool(an_cfg.get("pre_trained", True)),
        backbone=str(an_cfg.get("backbone", "resnet18")),
        layers=an_cfg.get("layers", ["layer1", "layer2", "layer3"]),
        n_features=int(an_cfg["n_features"]) if an_cfg.get("n_features") is not None else None,
    )


def _build_anomalib_stfpm(
    model_cfg: Dict[str, Any], runtime_cfg: Dict[str, Any]
) -> BaseModel:
    from real_time_visual_defect_detection.models.anomalib_stfpm_model import (
        AnomalibStfpmModel,
    )

    an_cfg = model_cfg.get("anomalib", {})
    st_cfg = model_cfg.get("stfpm", {})
    return AnomalibStfpmModel(
        threshold=float(model_cfg.get("threshold", 0.5)),
        device=str(runtime_cfg.get("resolved_device", "cpu")),
        image_size=int(an_cfg.get("image_size", 256)),
        batch_size=int(an_cfg.get("batch_size", 8)),
        backbone=str(st_cfg.get("backbone", an_cfg.get("backbone", "resnet18"))),
        layers=st_cfg.get("layers", an_cfg.get("layers", ["layer1", "layer2", "layer3"])),
        epochs=int(st_cfg.get("epochs", 1)),
        learning_rate=float(st_cfg.get("learning_rate", 0.4)),
    )


def _build_anomalib_csflow(
    model_cfg: Dict[str, Any], runtime_cfg: Dict[str, Any]
) -> BaseModel:
    from real_time_visual_defect_detection.models.anomalib_csflow_model import (
        AnomalibCsflowModel,
    )

    an_cfg = model_cfg.get("anomalib", {})
    cs_cfg = model_cfg.get("csflow", {})
    return AnomalibCsflowModel(
        threshold=float(model_cfg.get("threshold", 0.5)),
        device=str(runtime_cfg.get("resolved_device", "cpu")),
        image_size=int(an_cfg.get("image_size", 256)),
        batch_size=int(an_cfg.get("batch_size", 8)),
        epochs=int(cs_cfg.get("epochs", 1)),
        learning_rate=float(cs_cfg.get("learning_rate", 2e-4)),
        cross_conv_hidden_channels=int(cs_cfg.get("cross_conv_hidden_channels", 1024)),
        n_coupling_blocks=int(cs_cfg.get("n_coupling_blocks", 4)),
        clamp=int(cs_cfg.get("clamp", 3)),
    )


def _parse_beta(beta_value: Any) -> tuple[float, float]:
    if isinstance(beta_value, (list, tuple)) and len(beta_value) == 2:
        return float(beta_value[0]), float(beta_value[1])
    if beta_value is None:
        return 0.2, 1.0
    scalar = float(beta_value)
    return scalar, scalar


def _build_anomalib_draem(
    model_cfg: Dict[str, Any], runtime_cfg: Dict[str, Any]
) -> BaseModel:
    from real_time_visual_defect_detection.models.anomalib_draem_model import (
        AnomalibDraemModel,
    )

    an_cfg = model_cfg.get("anomalib", {})
    dr_cfg = model_cfg.get("draem", {})
    return AnomalibDraemModel(
        threshold=float(model_cfg.get("threshold", 0.5)),
        device=str(runtime_cfg.get("resolved_device", "cpu")),
        image_size=int(an_cfg.get("image_size", 256)),
        batch_size=int(an_cfg.get("batch_size", 8)),
        epochs=int(dr_cfg.get("epochs", 1)),
        learning_rate=float(dr_cfg.get("learning_rate", 1e-4)),
        beta=_parse_beta(dr_cfg.get("beta", (0.2, 1.0))),
    )


def _build_subspacead(model_cfg: Dict[str, Any], runtime_cfg: Dict[str, Any]) -> BaseModel:
    from real_time_visual_defect_detection.models.subspacead_model import SubspaceADModel

    sub_cfg = model_cfg.get("subspacead", {})
    return SubspaceADModel(
        threshold=float(model_cfg.get("threshold", 0.5)),
        device=str(runtime_cfg.get("resolved_device", "cpu")),
        model_ckpt=str(
            sub_cfg.get("model_ckpt", "facebook/dinov2-with-registers-large")
        ),
        image_size=int(sub_cfg.get("image_size", 256)),
        batch_size=int(sub_cfg.get("batch_size", 4)),
        pca_ev=float(sub_cfg.get("pca_ev", 0.99)),
        pca_dim=int(sub_cfg["pca_dim"]) if sub_cfg.get("pca_dim") is not None else None,
        img_score_agg=str(sub_cfg.get("img_score_agg", "mtop1p")),
        layers=sub_cfg.get("layers", [-12, -13, -14, -15, -16, -17, -18]),
    )


# Central model registry used by the pipeline and benchmark runner.
_MODEL_BUILDERS = {
    "dummy_distance": _build_dummy,
    "rd4ad": _build_rd4ad,
    "anomalib_patchcore": _build_anomalib_patchcore,
    "anomalib_padim": _build_anomalib_padim,
    "anomalib_stfpm": _build_anomalib_stfpm,
    "anomalib_csflow": _build_anomalib_csflow,
    "anomalib_draem": _build_anomalib_draem,
    "subspacead": _build_subspacead,
}


def build_model(model_cfg: Dict[str, Any], runtime_cfg: Dict[str, Any]) -> BaseModel:
    name = str(model_cfg.get("name", "dummy_distance"))
    builder = _MODEL_BUILDERS.get(name)
    if builder is None:
        supported = ", ".join(sorted(_MODEL_BUILDERS))
        raise ValueError(f"Unknown model name: '{name}'. Supported: {supported}.")
    return builder(model_cfg, runtime_cfg)


def available_models() -> list[str]:
    return sorted(_MODEL_BUILDERS)
