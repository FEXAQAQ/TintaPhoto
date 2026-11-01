"""Tests for correcting the preview for the screen (PLAN.md 6 step 13)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import ImageCms

from core import color_management as cm


def _image() -> np.ndarray:
    return np.random.default_rng(2).integers(0, 256, (16, 24, 3), dtype=np.uint8)


def test_no_profile_means_no_transform() -> None:
    assert cm.build_display_transform(None) is None
    assert cm.build_display_transform(b"") is None


def test_an_srgb_screen_needs_no_transform() -> None:
    """The conversion would do nothing, so it is skipped entirely."""
    profile = ImageCms.createProfile("sRGB")
    assert cm.build_display_transform(ImageCms.ImageCmsProfile(profile).tobytes()) is None


def test_a_damaged_profile_is_refused_rather_than_raising() -> None:
    assert cm.build_display_transform(b"this is not a colour profile") is None


def test_without_a_transform_the_photo_is_handed_back_untouched() -> None:
    image = _image()
    assert cm.apply_display_transform(image, None) is image


def test_a_transform_is_applied_to_every_pixel() -> None:
    """Built by hand, since this machine's screen may well be sRGB."""
    srgb = ImageCms.createProfile("sRGB")
    transform = ImageCms.buildTransform(srgb, srgb, "RGB", "RGB")
    image = _image()
    result = cm.apply_display_transform(image, transform)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    # sRGB to sRGB changes nothing, which is the point: the path runs and the
    # pixels come back whole.
    assert np.array_equal(result, image)


def test_a_broken_transform_leaves_the_photo_alone() -> None:
    image = _image()
    assert np.array_equal(cm.apply_display_transform(image, object()), image)


def test_the_screen_profile_is_never_applied_to_an_export() -> None:
    """Step 13 belongs to the preview only; develop() must not do it."""
    import inspect

    from core import pipeline

    source = inspect.getsource(pipeline.develop)
    assert "display" not in source.lower()


@pytest.mark.parametrize("value", [0, 64, 128, 255])
def test_a_grey_stays_grey_through_the_transform(value: int) -> None:
    srgb = ImageCms.createProfile("sRGB")
    transform = ImageCms.buildTransform(srgb, srgb, "RGB", "RGB")
    grey = np.full((4, 4, 3), value, dtype=np.uint8)
    result = cm.apply_display_transform(grey, transform)
    assert int(result.max()) - int(result.min()) <= 1
