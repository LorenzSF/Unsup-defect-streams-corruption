import pytest

from benchmark_AD.models import available_models, build_model, model_dependencies


def test_registry_has_expected_models():
    names = set(available_models())
    assert "rd4ad" in names
    assert "anomalib_patchcore" in names
    assert "anomalib_padim" in names
    assert "anomalib_stfpm" in names
    assert "anomalib_csflow" in names
    assert "anomalib_draem" in names
    assert "subspacead" in names


def test_build_rd4ad_model():
    model = build_model(
        {"name": "rd4ad", "threshold": 0.5},
        {"resolved_device": "cpu", "num_workers": 0},
    )
    assert model is not None
    assert hasattr(model, "predict")


@pytest.mark.parametrize("model_name", available_models())
def test_build_all_registered_models(model_name: str):
    model = build_model(
        {"name": model_name, "threshold": 0.5},
        {"resolved_device": "cpu", "num_workers": 0},
    )
    assert model is not None
    assert hasattr(model, "predict")


def test_model_dependencies_for_known_models():
    for model_name in available_models():
        deps = model_dependencies(model_name)
        assert isinstance(deps, tuple)
        assert len(deps) >= 1
        assert all(isinstance(dep, str) and dep for dep in deps)


def test_model_dependencies_for_unknown_model_raises():
    with pytest.raises(ValueError):
        _ = model_dependencies("does_not_exist")
