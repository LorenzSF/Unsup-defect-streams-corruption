from real_time_visual_defect_detection.models.registry import available_models, build_model


def test_registry_has_expected_models():
    names = set(available_models())
    assert "dummy_distance" in names
    assert "rd4ad" in names
    assert "anomalib_patchcore" in names
    assert "anomalib_padim" in names
    assert "anomalib_stfpm" in names
    assert "anomalib_csflow" in names
    assert "anomalib_draem" in names
    assert "subspacead" in names


def test_build_dummy_model():
    model = build_model(
        {"name": "dummy_distance", "threshold": 0.5},
        {"resolved_device": "cpu", "num_workers": 0},
    )
    assert model is not None
    assert hasattr(model, "predict")
