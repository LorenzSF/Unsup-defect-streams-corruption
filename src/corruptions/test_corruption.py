import numpy as np
import pytest

from corruptions.corruption_registry import (
    SEVERITY_LEVELS,
    available_corruptions,
    get_corruption,
)


def _sample_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)


def test_registry_lists_expected_names():
    assert set(available_corruptions()) == {
        "gaussian_blur",
        "motion_blur",
        "jpeg_compression",
    }


@pytest.mark.parametrize("name", available_corruptions())
@pytest.mark.parametrize("severity", SEVERITY_LEVELS)
def test_corruption_preserves_shape_and_dtype(name: str, severity: int):
    image = _sample_image()
    apply = get_corruption(name, severity)
    out = apply(image)
    assert out.shape == image.shape
    assert out.dtype == image.dtype


@pytest.mark.parametrize("name", available_corruptions())
def test_corruption_actually_modifies_image(name: str):
    # Severity 5 should always perturb the image; if a function is a no-op
    # the integration with the pipeline would silently break the benchmark.
    image = _sample_image(seed=1)
    out = get_corruption(name, 5)(image)
    assert not np.array_equal(out, image), f"{name} at severity=5 left the image unchanged"


def test_get_corruption_rejects_unknown_name():
    with pytest.raises(ValueError):
        get_corruption("does_not_exist", 1)


@pytest.mark.parametrize("severity", [0, 6, -1, 10])
def test_get_corruption_rejects_invalid_severity(severity: int):
    with pytest.raises(ValueError):
        get_corruption("gaussian_blur", severity)


def test_corruption_is_deterministic_per_image():
    # motion_blur picks its kernel angle from the per-image RNG, so two
    # independent calls on the same array must produce identical outputs.
    # If this regresses, the registry has reintroduced a global-seed leak.
    image = _sample_image(seed=2)
    apply = get_corruption("motion_blur", 3)
    assert np.array_equal(apply(image), apply(image))
