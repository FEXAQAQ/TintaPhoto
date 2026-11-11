"""The film looks of PLAN.md's filter tab (:mod:`core.filters`).

What matters about a look is less what it does to one colour than what it
refuses to do: leave a photograph alone when it is not asked for, keep a grey
grey when nothing asks it to turn, and stay inside its limits whatever is in
the file it was read from.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from core import color_management as cm
from core.filters import (
    CATEGORIES,
    DEFAULT_CATEGORY,
    MAX_HUE_TURN,
    MAX_LOG_SCALE,
    Look,
    curve,
    fourier,
    load,
    load_catalogue,
    save,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHIPPED = PROJECT_ROOT / "resources" / "filters"
LENS_DATABASE = PROJECT_ROOT / "resources" / "lensfun"


@pytest.fixture
def photo() -> np.ndarray:
    """A small spread of colours in the working space."""
    generator = np.random.default_rng(20261023)
    return generator.random((12, 16, 3), dtype=np.float32)


def a_look(identifier: str = "test") -> Look:
    """Return a look that does something to every part of a photo."""
    knots = np.linspace(0.0, 1.0, 17)
    return Look(
        identifier=identifier,
        curves=(
            tuple(np.clip(knots * 1.1, 0.0, 1.0).tolist()),
            tuple(knots.tolist()),
            tuple((knots**1.2).tolist()),
        ),
        chroma=(-0.1, 0.05, 0.02),
        hue=(0.02, -0.03, 0.01),
        lightness=(0.01, 0.02, -0.01),
    )


# -- doing nothing ---------------------------------------------------------


def test_an_identity_look_returns_the_photo_untouched(photo: np.ndarray) -> None:
    assert np.array_equal(Look.identity().apply(photo), photo)


def test_identity_knows_itself() -> None:
    assert Look.identity().is_identity
    assert not a_look().is_identity


def test_no_strength_returns_the_photo_untouched(photo: np.ndarray) -> None:
    assert np.array_equal(a_look().apply(photo, 0.0), photo)


def test_strength_moves_the_look_the_whole_way(photo: np.ndarray) -> None:
    """Half of a look sits between doing nothing and doing all of it."""
    full = a_look().apply(photo, 1.0)
    half = a_look().apply(photo, 0.5)
    distance = np.abs(full - photo).mean()
    assert np.abs(half - photo).mean() < distance
    assert np.abs(half - full).mean() < distance


def test_strength_past_one_is_the_same_as_one(photo: np.ndarray) -> None:
    assert np.array_equal(a_look().apply(photo, 2.0), a_look().apply(photo, 1.0))


# -- what a look may not do ------------------------------------------------


def test_a_grey_stays_grey_when_the_three_curves_agree() -> None:
    """A neutral has no hue, so none of the three hue functions may move it."""
    knots = np.linspace(0.0, 1.0, 17)
    bent = tuple((knots**1.4).tolist())
    look = Look(
        identifier="grey",
        curves=(bent, bent, bent),
        chroma=(0.3, 0.2, -0.1),
        hue=(0.2, 0.1, 0.05),
        lightness=(0.2, -0.1, 0.05),
    )
    greys = np.repeat(
        np.linspace(0.02, 0.95, 24, dtype=np.float32).reshape(-1, 1, 1), 3, axis=2
    )
    result = look.apply(greys)
    spread = result.max(axis=2) - result.min(axis=2)
    # Not exactly equal: the look goes out to sRGB and through Oklab's cube
    # root and back, and float32 does not come home to the last bit. PLAN.md
    # section 6 sets the bar for a round trip at a thousandth, and what is
    # left here is a fifth of that.
    assert float(spread.max()) < 1e-3


def test_a_highlight_above_one_is_not_flattened() -> None:
    """The vignette can push a pixel past white; a look must not clip it."""
    bright = np.full((2, 2, 3), 2.5, dtype=np.float32)
    result = a_look().apply(bright)
    assert float(result.min()) > 1.0


def test_a_look_stays_inside_its_limits(photo: np.ndarray) -> None:
    """However wild the numbers in the file, the result is still a photo."""
    wild = Look(
        identifier="wild",
        curves=Look.identity().curves,
        chroma=(50.0, -40.0),
        hue=(30.0, 20.0),
        lightness=(25.0, -25.0),
    )
    result = wild.apply(photo)
    assert np.isfinite(result).all()
    lab_before = cm.prophoto_to_oklab(photo)
    lab_after = cm.prophoto_to_oklab(result)
    chroma_before = np.hypot(lab_before[..., 1], lab_before[..., 2])
    chroma_after = np.hypot(lab_after[..., 1], lab_after[..., 2])
    coloured = chroma_before > 0.01
    ratio = chroma_after[coloured] / chroma_before[coloured]
    assert float(ratio.max()) <= np.exp(MAX_LOG_SCALE) * 1.01


# -- reading and writing ---------------------------------------------------


def test_a_look_survives_a_round_trip_through_its_file(tmp_path: Path) -> None:
    look = a_look("round-trip")
    path = tmp_path / "round-trip.json"
    save(look, path)
    read = load(path)
    assert read.identifier == look.identifier
    assert read.curves == look.curves
    assert read.chroma == look.chroma
    assert read.hue == look.hue
    assert read.lightness == look.lightness
    assert read.category == look.category


def test_a_category_survives_a_round_trip_through_its_file(tmp_path: Path) -> None:
    path = tmp_path / "medium.json"
    save(replace(a_look("medium"), category="medium-format"), path)
    assert load(path).category == "medium-format"


@pytest.mark.parametrize("written", [None, "", "black-and-white", 3])
def test_a_category_this_copy_does_not_know_reads_as_the_first(written: object) -> None:
    """A look from a newer version keeps working; it just sits in the
    first group until this copy learns where it belongs."""
    data: dict[str, object] = {
        "id": "x",
        "curves": {"r": [0.0, 1.0], "g": [0.0, 1.0], "b": [0.0, 1.0]},
    }
    if written is not None:
        data["category"] = written
    assert Look.from_dict(data).category == DEFAULT_CATEGORY


def test_saving_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    save(a_look(), tmp_path / "look.json")
    assert [path.name for path in tmp_path.iterdir()] == ["look.json"]


@pytest.mark.parametrize(
    "broken",
    [
        {"id": "x"},
        {"id": "x", "curves": {"r": [0.0, 1.0], "g": [0.0, 1.0]}},
        {"id": "x", "curves": {"r": [0.0], "g": [0.0], "b": [0.0]}},
        {"id": "x", "curves": {"r": [0.0, 1.0], "g": [0.0, 1.0], "b": [0.0, 0.5, 1.0]}},
    ],
)
def test_a_half_written_look_is_refused(broken: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Look.from_dict(broken)


def test_a_missing_series_reads_as_no_change() -> None:
    look = Look.from_dict(
        {"id": "x", "curves": {"r": [0.0, 1.0], "g": [0.0, 1.0], "b": [0.0, 1.0]}}
    )
    assert look.is_identity


def test_a_catalogue_skips_what_it_cannot_read(tmp_path: Path) -> None:
    save(a_look("good"), tmp_path / "good.json")
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "empty.json").write_text(json.dumps({"id": "empty"}), encoding="utf-8")
    assert [look.identifier for look in load_catalogue(tmp_path)] == ["good"]


def test_a_catalogue_of_nowhere_is_empty(tmp_path: Path) -> None:
    assert load_catalogue(tmp_path / "not-here") == ()


# -- the looks that ship ---------------------------------------------------


def test_every_shipped_look_reads() -> None:
    looks = load_catalogue(SHIPPED)
    assert looks, f"no looks found in {SHIPPED}"
    for look in looks:
        assert look.identifier
        assert not look.is_identity
        assert len(look.curves[0]) == len(look.curves[1]) == len(look.curves[2])


def test_every_shipped_look_was_fitted_inside_the_limits() -> None:
    """A look that met its clamp was cut off, and is not what was measured."""
    hue = np.linspace(-np.pi, np.pi, 361, dtype=np.float32)
    for look in load_catalogue(SHIPPED):
        for name, series, limit in (
            ("chroma", look.chroma, MAX_LOG_SCALE),
            ("hue", look.hue, MAX_HUE_TURN),
            ("lightness", look.lightness, MAX_LOG_SCALE),
        ):
            reach = float(np.abs(fourier(np.asarray(series, dtype=np.float32), hue)).max())
            assert reach <= limit, f"{look.identifier}: {name} reaches {reach:.3f}"


def test_every_shipped_look_has_monotone_curves() -> None:
    """A curve that turned back on itself would posterise a gradient."""
    for look in load_catalogue(SHIPPED):
        for channel, curve in zip("rgb", look.curves):
            rise = np.diff(np.asarray(curve, dtype=np.float64))
            assert rise.min() >= -1e-9, f"{look.identifier}: the {channel} curve falls"


def test_every_shipped_look_says_where_it_came_from() -> None:
    for look in load_catalogue(SHIPPED):
        assert look.source is not None
        assert "method" in look.source
        # A measured look says how well it fits; a designed one keeps the
        # recipe it was written from, so it can be read and written again.
        if look.source["method"] == "designed":
            assert "recipe" in look.source
        else:
            assert "delta_e_oklab" in look.source
        # No make, model or file name: a look is this project's own.
        assert "camera" not in look.source


def test_every_shipped_look_sits_in_a_known_category() -> None:
    for look in load_catalogue(SHIPPED):
        assert look.category in CATEGORIES, look.identifier


def _camera_makers() -> set[str]:
    """Every maker the bundled lens database names, in lower case.

    The names are read from the database rather than listed here, so that
    the test itself names no maker and still covers every one it knows.
    """
    makers: set[str] = set()
    for path in LENS_DATABASE.glob("*.xml"):
        for maker in re.findall(r"<maker>([^<]+)</maker>", path.read_text(encoding="utf-8")):
            first_word = maker.split()[0].strip(" ,.").lower()
            # Short names would turn up inside ordinary words.
            if len(first_word) >= 4:
                makers.add(first_word)
    return makers


def test_no_shipped_look_names_a_camera_maker() -> None:
    """PLAN.md 13.3: the looks are this project's own approximations, so
    nothing that ships with them may name the body they were measured from."""
    makers = _camera_makers()
    assert len(makers) > 20, "the lens database should name many makers"
    for path in SHIPPED.glob("*.json"):
        text = path.read_text(encoding="utf-8").lower()
        for maker in makers:
            assert maker not in text, f"{path.name} mentions {maker}"


# -- the tone curve --------------------------------------------------------


def test_a_curve_reads_its_own_knots_back() -> None:
    points = np.array([0.0, 0.2, 0.5, 0.9, 1.0], dtype=np.float32)
    at_knots = np.linspace(0.0, 1.0, len(points), dtype=np.float32)

    assert np.allclose(curve(points, at_knots), points, atol=1e-6)


def test_a_curve_runs_straight_on_past_both_ends() -> None:
    """Nothing is clamped: a highlight past white keeps going up."""
    points = np.array([0.1, 0.4, 0.7, 1.0], dtype=np.float32)
    slope = (points[-1] - points[-2]) * (len(points) - 1)

    beyond = curve(points, np.array([2.0], dtype=np.float32))[0]
    below = curve(points, np.array([-1.0], dtype=np.float32))[0]

    assert beyond == pytest.approx(points[-1] + slope, rel=1e-5)
    assert below < points[0]


def test_a_curve_agrees_with_interpolating_it_the_slow_way() -> None:
    points = np.array([0.0, 0.3, 0.45, 0.8, 1.0], dtype=np.float32)
    values = np.linspace(0.0, 1.0, 97, dtype=np.float32)
    knots = np.linspace(0.0, 1.0, len(points), dtype=np.float32)

    assert np.allclose(curve(points, values), np.interp(values, knots, points), atol=1e-6)


# -- the hue functions -----------------------------------------------------


def test_the_hue_functions_join_up_round_the_circle() -> None:
    """A Fourier series is periodic, so a hue near pi meets one near minus pi."""
    coefficients = np.array([0.1, 0.3, -0.2, 0.05, 0.1], dtype=np.float32)
    ends = fourier(coefficients, np.array([-np.pi, np.pi], dtype=np.float32))

    assert ends[0] == pytest.approx(float(ends[1]), abs=1e-5)


def test_a_look_moves_smoothly_from_one_hue_to_the_next(photo: np.ndarray) -> None:
    """The hue table must not show as a step in a gradient."""
    look = a_look()
    angles = np.linspace(-np.pi, np.pi, 2000, dtype=np.float32)
    lab = np.stack(
        (
            np.full_like(angles, 0.6),
            np.cos(angles) * 0.12,
            np.sin(angles) * 0.12,
        ),
        axis=-1,
    ).astype(np.float32)
    result = cm.prophoto_to_oklab(look.apply(cm.oklab_to_prophoto(lab[None, :, :])))[0]

    chroma = np.hypot(result[:, 1], result[:, 2])
    steps = np.abs(np.diff(chroma))
    assert float(steps.max()) < 0.002
