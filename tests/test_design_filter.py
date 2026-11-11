"""The designed looks (``tools/design_filter.py``) and the teal and orange one.

A designed look is not measured, so what is tested is that the recipe says
what it is meant to and the file in ``resources/filters`` is what the recipe
writes: run the tool again and nothing changes.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from core import color_management as cm
from core.filters import Look, load

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import design_filter  # noqa: E402

SHIPPED = PROJECT_ROOT / "resources" / "filters"


@pytest.fixture(scope="module")
def teal_orange() -> Look:
    return load(SHIPPED / "teal-orange.json")


def _through(look: Look, srgb: tuple[float, float, float], strength: float = 1.0) -> np.ndarray:
    """Return one encoded sRGB colour after *look*, as Oklab."""
    encoded = np.asarray([[srgb]], dtype=np.float32)
    working = cm.apply_matrix(cm.srgb_to_linear(encoded), cm.SRGB_TO_PROPHOTO)
    return cm.prophoto_to_oklab(look.apply(working, strength))[0, 0]


def _oklab(srgb: tuple[float, float, float]) -> np.ndarray:
    encoded = np.asarray([[srgb]], dtype=np.float32)
    return cm.prophoto_to_oklab(
        cm.apply_matrix(cm.srgb_to_linear(encoded), cm.SRGB_TO_PROPHOTO)
    )[0, 0]


def _hue(lab: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(lab[2], lab[1])) % 360.0)


def _chroma(lab: np.ndarray) -> float:
    return float(np.hypot(lab[1], lab[2]))


def _hue_gap(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def test_the_shipped_file_is_what_the_recipe_writes(teal_orange: Look) -> None:
    built = design_filter.build(design_filter.TEAL_ORANGE)
    assert built.category == teal_orange.category
    assert built.source == teal_orange.source
    for mine, theirs in zip(built.curves, teal_orange.curves):
        np.testing.assert_allclose(mine, theirs, atol=1e-12)
    for name in ("chroma", "hue", "lightness"):
        np.testing.assert_allclose(getattr(built, name), getattr(teal_orange, name), atol=1e-12)


def test_teal_orange_sits_with_the_cinematic_looks(teal_orange: Look) -> None:
    assert teal_orange.category == "cinematic"
    assert teal_orange.source is not None
    assert teal_orange.source["method"] == "designed"


def test_a_mid_grey_stays_neutral(teal_orange: Look) -> None:
    assert _chroma(_through(teal_orange, (0.5, 0.5, 0.5))) < 0.004


def test_the_shadows_lean_teal_and_the_highlights_warm(teal_orange: Look) -> None:
    shadow = _through(teal_orange, (0.2, 0.2, 0.2))
    highlight = _through(teal_orange, (0.8, 0.8, 0.8))
    assert _chroma(shadow) > 0.005 and _chroma(highlight) > 0.005
    assert 180.0 < _hue(shadow) < 260.0
    assert 30.0 < _hue(highlight) < 110.0


def test_black_and_white_stay_put(teal_orange: Look) -> None:
    for value in (0.0, 1.0):
        lab = _through(teal_orange, (value, value, value))
        assert _chroma(lab) < 0.002
        assert lab[0] == pytest.approx(value, abs=0.01)


def test_skin_turns_warmer_without_shouting(teal_orange: Look) -> None:
    skin = (205 / 255, 150 / 255, 120 / 255)
    before, after = _oklab(skin), _through(teal_orange, skin)
    assert _hue_gap(_hue(after), 60.0) < _hue_gap(_hue(before), 60.0)
    assert 1.0 < _chroma(after) / _chroma(before) < 1.35


def test_greens_step_back_towards_teal(teal_orange: Look) -> None:
    grass = (90 / 255, 160 / 255, 60 / 255)
    before, after = _oklab(grass), _through(teal_orange, grass)
    assert _hue(after) > _hue(before) + 5.0
    assert _chroma(after) < _chroma(before) * 0.9


def test_skies_gather_into_teal(teal_orange: Look) -> None:
    sky = (110 / 255, 160 / 255, 220 / 255)
    before, after = _oklab(sky), _through(teal_orange, sky)
    assert _hue_gap(_hue(after), 215.0) < _hue_gap(_hue(before), 215.0)


def test_half_strength_lands_between(teal_orange: Look) -> None:
    grass = (90 / 255, 160 / 255, 60 / 255)
    before = _hue(_oklab(grass))
    half = _hue(_through(teal_orange, grass, 0.5))
    full = _hue(_through(teal_orange, grass, 1.0))
    assert before < half < full


def test_a_recipe_that_turns_too_far_is_refused() -> None:
    greedy = replace(
        design_filter.TEAL_ORANGE,
        bumps=(design_filter.Bump(hue=215.0, width=90.0, pull=1.0),),
    )
    with pytest.raises(ValueError, match="hue"):
        design_filter.build(greedy)


def test_a_recipe_whose_curve_falls_is_refused() -> None:
    steep = replace(design_filter.TEAL_ORANGE, tone=design_filter.Tone(contrast=1.5))
    with pytest.raises(ValueError, match="falls"):
        design_filter.build(steep)


def test_a_recipe_in_an_unknown_category_is_refused() -> None:
    with pytest.raises(ValueError, match="category"):
        design_filter.build(replace(design_filter.TEAL_ORANGE, category="nowhere"))
