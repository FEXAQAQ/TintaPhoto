"""Tests for the watermark laid over exported photos (PLAN.md 2.9)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from core import watermark
from core.watermark import Fonts, WatermarkSpec

FRAME = (1000, 600)


@pytest.fixture(scope="module")
def fonts() -> Fonts:
    return watermark.load_fonts()


def _cjk_fonts() -> Fonts:
    found = watermark.load_fonts()
    if not found.has_cjk:
        pytest.skip("this computer has no Chinese system font")
    return found


# -- the settings -----------------------------------------------------------


def test_sane_keeps_every_value_in_its_range() -> None:
    spec = WatermarkSpec(
        kind="logo",
        text="a\nb" + "x" * 200,
        colour="red",
        opacity=0,
        size=99.0,
        anchor="middle",
        offset=(5.0, -5.0),
        margin=-3,
    ).sane()
    assert spec.kind == watermark.KIND_TEXT
    assert "\n" not in spec.text and len(spec.text) == watermark.TEXT_MAX_LENGTH
    assert spec.colour == watermark.COLOUR_WHITE
    assert spec.opacity == watermark.OPACITY_MIN
    assert spec.size == watermark.SIZE_MAX
    assert spec.anchor == watermark.ANCHOR_DEFAULT
    assert spec.offset == (watermark.OFFSET_LIMIT, -watermark.OFFSET_LIMIT)
    assert spec.margin == watermark.MARGIN_MIN


def test_stored_settings_come_back_the_same() -> None:
    spec = WatermarkSpec(
        text="© 2026 王小明", bold=True, italic=True, colour="black", opacity=55,
        size=4.5, anchor="top", offset=(0.1, -0.05), margin=7,
    )
    assert WatermarkSpec.from_dict(spec.to_dict()) == spec


def test_unusable_stored_settings_fall_back_to_the_defaults() -> None:
    assert WatermarkSpec.from_dict(None) == WatermarkSpec()
    spec = WatermarkSpec.from_dict(
        {"text": 12, "bold": "yes", "size": "big", "offset": [1], "opacity": float("nan")}
    )
    assert spec == WatermarkSpec()


def test_a_kind_from_a_later_version_draws_nothing() -> None:
    spec = WatermarkSpec.from_dict({"kind": "image", "text": "x"})
    assert spec.is_empty


def test_blank_text_is_no_watermark() -> None:
    assert WatermarkSpec(text="   ").is_empty
    assert not WatermarkSpec(text="© me").is_empty


def test_resetting_the_style_keeps_only_the_text() -> None:
    spec = WatermarkSpec(text="me", bold=True, size=9.0, anchor="top", offset=(0.2, 0.0))
    assert spec.with_style_reset() == WatermarkSpec(text="me")


# -- fonts (W-04) -----------------------------------------------------------


def test_latin_text_is_set_in_inter(fonts: Fonts) -> None:
    assert fonts.regular is not None and fonts.regular.name == "Inter-Regular.ttf"
    choice = watermark.choose_font(WatermarkSpec(text="© 2026"), fonts)
    assert choice.path == fonts.regular and not choice.slant
    bold = watermark.choose_font(WatermarkSpec(text="© 2026", bold=True), fonts)
    assert bold.path == fonts.bold


def test_italic_uses_the_italic_face_when_there_is_one(tmp_path: Path) -> None:
    for name in ("Inter-Regular.ttf", "Inter-SemiBold.ttf"):
        shutil.copy(watermark.FONTS_DIR / name, tmp_path / name)
    shutil.copy(watermark.FONTS_DIR / "Inter-Regular.ttf", tmp_path / "Inter-Italic.ttf")
    with_italic = watermark.load_fonts(tmp_path, tmp_path)
    choice = watermark.choose_font(WatermarkSpec(text="a", italic=True), with_italic)
    assert choice.path == tmp_path / "Inter-Italic.ttf" and not choice.slant
    # No bold italic face: the bold one is slanted instead.
    choice = watermark.choose_font(WatermarkSpec(text="a", bold=True, italic=True), with_italic)
    assert choice.path == tmp_path / "Inter-SemiBold.ttf" and choice.slant


def test_chinese_text_is_set_in_the_system_font_and_slanted_for_italic() -> None:
    fonts = _cjk_fonts()
    choice = watermark.choose_font(WatermarkSpec(text="© 王小明", italic=True), fonts)
    assert choice.path == fonts.cjk_regular and choice.slant and not choice.missing_glyphs
    bold = watermark.choose_font(WatermarkSpec(text="王", bold=True), fonts)
    assert bold.path == fonts.cjk_bold


def test_chinese_text_without_a_chinese_font_says_so(tmp_path: Path) -> None:
    shutil.copy(watermark.FONTS_DIR / "Inter-Regular.ttf", tmp_path / "Inter-Regular.ttf")
    fonts = watermark.load_fonts(tmp_path, tmp_path / "none")
    choice = watermark.choose_font(WatermarkSpec(text="王小明"), fonts)
    assert choice.missing_glyphs and choice.path == tmp_path / "Inter-Regular.ttf"
    sprite = watermark.rasterise(WatermarkSpec(text="王小明"), FRAME, fonts)
    assert sprite is not None and sprite.missing_glyphs


# -- where it goes (W-07, W-08) ---------------------------------------------


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        ("top-left", (18.0, 18.0)),
        ("top", (450.0, 18.0)),
        ("top-right", (882.0, 18.0)),
        ("left", (18.0, 290.0)),
        ("centre", (450.0, 290.0)),
        ("right", (882.0, 290.0)),
        ("bottom-left", (18.0, 562.0)),
        ("bottom", (450.0, 562.0)),
        ("bottom-right", (882.0, 562.0)),
    ],
)
def test_each_anchor_sits_a_margin_in_from_its_edges(
    anchor: str, expected: tuple[float, float]
) -> None:
    # The margin is 3 % of the short edge of 600: 18 px.
    spec = WatermarkSpec(text="x", anchor=anchor, margin=3)
    assert watermark.layout(spec, FRAME, (100, 20)) == pytest.approx(expected)


def test_the_offset_is_measured_in_short_edges() -> None:
    spec = WatermarkSpec(text="x", anchor="top-left", margin=0, offset=(0.1, 0.05))
    assert watermark.layout(spec, FRAME, (100, 20)) == pytest.approx((60.0, 30.0))


def test_the_watermark_never_leaves_the_photo() -> None:
    spec = WatermarkSpec(text="x", anchor="bottom-right", margin=0, offset=(0.5, 0.5))
    assert watermark.layout(spec, FRAME, (100, 20)) == pytest.approx((900.0, 580.0))


def test_the_same_spec_lands_in_the_same_corner_on_any_shape() -> None:
    spec = WatermarkSpec(text="x", anchor="bottom-right", margin=3)
    for frame in ((6000, 4000), (4000, 6000), (4000, 4000)):
        x, y = watermark.layout(spec, frame, (100, 20))
        margin = 0.03 * min(frame)
        assert frame[0] - (x + 100) == pytest.approx(margin)
        assert frame[1] - (y + 20) == pytest.approx(margin)


def test_a_drop_puts_the_watermark_where_it_was_left() -> None:
    spec = WatermarkSpec(text="x")
    # Nearest the middle column (450) and the middle row (290).
    drop = watermark.from_drop(spec, FRAME, (100, 20), (300.0, 200.0), snap=0.0)
    assert drop.spec.anchor == "centre"
    assert watermark.layout(drop.spec, FRAME, (100, 20)) == pytest.approx((300.0, 200.0))
    assert not drop.snapped_across and not drop.snapped_down


def test_a_drop_near_an_anchor_line_is_pulled_onto_it() -> None:
    spec = WatermarkSpec(text="x", margin=3)
    drop = watermark.from_drop(spec, FRAME, (100, 20), (455.0, 300.0), snap=8.0)
    assert drop.spec.anchor == "bottom" or drop.spec.anchor == "centre"
    assert drop.snapped_across and drop.spec.offset[0] == 0.0
    near_bottom = watermark.from_drop(spec, FRAME, (100, 20), (700.0, 556.0), snap=8.0)
    assert near_bottom.spec.anchor == "bottom-right"
    assert near_bottom.snapped_down and not near_bottom.snapped_across
    assert watermark.layout(near_bottom.spec, FRAME, (100, 20))[1] == pytest.approx(562.0)


def test_a_drop_outside_the_photo_stays_inside() -> None:
    drop = watermark.from_drop(WatermarkSpec(text="x"), FRAME, (100, 20), (-50.0, 900.0), 0.0)
    assert watermark.layout(drop.spec, FRAME, (100, 20)) == pytest.approx((0.0, 580.0))


def test_guides_follow_the_lines_the_watermark_sits_on() -> None:
    spec = WatermarkSpec(text="x", anchor="bottom", margin=3)
    assert watermark.guides(spec, FRAME, (100, 20)) == pytest.approx((500.0, 582.0))
    off = WatermarkSpec(text="x", anchor="bottom", margin=3, offset=(0.1, 0.0))
    vertical, horizontal = watermark.guides(off, FRAME, (100, 20))
    assert vertical is None and horizontal == pytest.approx(582.0)


def test_the_picker_marks_an_anchor_only_while_on_it() -> None:
    assert watermark.anchor_of(WatermarkSpec(anchor="top")) == "top"
    assert watermark.anchor_of(WatermarkSpec(anchor="top", offset=(0.0, 0.1))) is None
    moved = WatermarkSpec(anchor="top", offset=(0.0, 0.1)).with_anchor("left")
    assert moved.anchor == "left" and moved.offset == (0.0, 0.0)


# -- drawing ----------------------------------------------------------------


def test_nothing_to_draw_draws_nothing(fonts: Fonts) -> None:
    assert watermark.rasterise(WatermarkSpec(text=""), FRAME, fonts) is None


def test_the_line_is_the_asked_share_of_the_short_edge(fonts: Fonts) -> None:
    for size in (2.0, 5.0, 10.0):
        sprite = watermark.rasterise(WatermarkSpec(text="Hg", size=size), FRAME, fonts)
        assert sprite is not None
        expected = 600 * size / 100
        # A pixel of room either side for the antialiasing, and rounding.
        assert abs(sprite.size[1] - expected) <= 2 * watermark._PAD + 2


def test_landscape_and_portrait_get_the_same_watermark(fonts: Fonts) -> None:
    spec = WatermarkSpec(text="© 2026")
    wide = watermark.rasterise(spec, (6000, 4000), fonts)
    tall = watermark.rasterise(spec, (4000, 6000), fonts)
    assert wide is not None and tall is not None
    assert wide.size == tall.size


def test_the_watermark_scales_with_the_photo(fonts: Fonts) -> None:
    spec = WatermarkSpec(text="© 2026 Wang")
    small = watermark.rasterise(spec, (1500, 1000), fonts)
    large = watermark.rasterise(spec, (6000, 4000), fonts)
    assert small is not None and large is not None
    assert large.size[0] / small.size[0] == pytest.approx(4.0, rel=0.08)


def test_a_line_too_wide_is_set_smaller(fonts: Fonts) -> None:
    spec = WatermarkSpec(text="A very long watermark line indeed", size=20.0)
    sprite = watermark.rasterise(spec, (1000, 1000), fonts)
    assert sprite is not None
    assert sprite.size[0] <= 1000 * watermark.WIDTH_LIMIT


def test_bold_covers_more_than_regular(fonts: Fonts) -> None:
    regular = watermark.rasterise(WatermarkSpec(text="Watermark", opacity=100), FRAME, fonts)
    bold = watermark.rasterise(WatermarkSpec(text="Watermark", bold=True, opacity=100), FRAME, fonts)
    assert regular is not None and bold is not None
    assert bold.coverage.sum() > regular.coverage.sum() * 1.15


def test_a_slanted_italic_leans_right(fonts: Fonts) -> None:
    upright = watermark._draw("I", watermark._font(fonts.regular, 100), False)
    slanted = watermark._draw("I", watermark._font(fonts.regular, 100), True)
    assert slanted.shape[0] == upright.shape[0]
    assert slanted.shape[1] > upright.shape[1]
    rows = np.nonzero(slanted.max(axis=1) > 0.5)[0]
    top_ink = np.nonzero(slanted[rows[0]] > 0.5)[0].mean()
    bottom_ink = np.nonzero(slanted[rows[-1]] > 0.5)[0].mean()
    assert top_ink > bottom_ink


def test_chinese_italic_is_wider_than_upright() -> None:
    fonts = _cjk_fonts()
    upright = watermark.rasterise(WatermarkSpec(text="王小明"), FRAME, fonts)
    italic = watermark.rasterise(WatermarkSpec(text="王小明", italic=True), FRAME, fonts)
    assert upright is not None and italic is not None
    assert italic.size[0] > upright.size[0]
    assert italic.size[1] == upright.size[1]


def test_opacity_scales_the_coverage(fonts: Fonts) -> None:
    full = watermark.rasterise(WatermarkSpec(text="Hg", opacity=100), FRAME, fonts)
    half = watermark.rasterise(WatermarkSpec(text="Hg", opacity=50), FRAME, fonts)
    assert full is not None and half is not None
    assert float(full.coverage.max()) == pytest.approx(1.0)
    assert float(half.coverage.max()) == pytest.approx(0.5)


# -- laying it over a photo -------------------------------------------------


def _photo(dtype: type) -> np.ndarray:
    rng = np.random.default_rng(7)
    top = 255 if dtype == np.uint8 else 65535
    return rng.integers(0, top, size=(600, 1000, 3)).astype(dtype)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_only_the_pixels_under_the_watermark_change(fonts: Fonts, dtype: type) -> None:
    image = _photo(dtype)
    before = image.copy()
    spec = WatermarkSpec(text="© 2026", size=5.0)
    sprite = watermark.rasterise(spec, FRAME, fonts)
    assert sprite is not None
    result = watermark.compose(image, spec, sprite)
    np.testing.assert_array_equal(image, before)
    x, y = (round(v) for v in watermark.layout(spec, FRAME, sprite.size))
    outside = np.ones(image.shape[:2], dtype=bool)
    outside[y : y + sprite.size[1], x : x + sprite.size[0]] = False
    np.testing.assert_array_equal(result[outside], before[outside])
    assert not np.array_equal(result[~outside], before[~outside])
    assert result.dtype == image.dtype


def test_solid_white_text_comes_out_white(fonts: Fonts) -> None:
    image = np.zeros((600, 1000, 3), dtype=np.uint8)
    spec = WatermarkSpec(text="I", size=20.0, opacity=100, anchor="centre")
    result = watermark.compose(image, spec, watermark.rasterise(spec, FRAME, fonts))
    assert int(result.max()) == 255


def test_black_text_darkens(fonts: Fonts) -> None:
    image = np.ones((600, 1000, 3), dtype=np.float32)
    spec = WatermarkSpec(text="I", size=20.0, opacity=50, colour="black", anchor="centre")
    result = watermark.compose(image, spec, watermark.rasterise(spec, FRAME, fonts))
    assert result.dtype == np.float32
    assert float(result.min()) == pytest.approx(0.5, abs=0.01)


def test_no_watermark_returns_the_photo_itself(fonts: Fonts) -> None:
    image = _photo(np.uint8)
    assert watermark.apply(image, None, fonts) is image
    assert watermark.apply(image, WatermarkSpec(text=""), fonts) is image


def test_the_screen_copy_carries_the_colour_and_the_coverage(fonts: Fonts) -> None:
    spec = WatermarkSpec(text="I", size=20.0, opacity=50, colour="black")
    sprite = watermark.rasterise(spec, FRAME, fonts)
    assert sprite is not None
    rgba = watermark.to_rgba(sprite)
    assert rgba.shape == (sprite.size[1], sprite.size[0], 4)
    assert int(rgba[:, :, :3].max()) == 0
    assert int(rgba[:, :, 3].max()) == 128


def test_the_bundled_inter_has_its_italics(fonts: Fonts) -> None:
    assert fonts.italic is not None and fonts.italic.name == "Inter-Italic.ttf"
    assert fonts.bold_italic is not None and fonts.bold_italic.name == "Inter-SemiBoldItalic.ttf"
    choice = watermark.choose_font(WatermarkSpec(text="© 2026", bold=True, italic=True), fonts)
    assert choice.path == fonts.bold_italic and not choice.slant
