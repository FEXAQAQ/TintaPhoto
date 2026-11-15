"""Tests for the depth of field (PLAN.md 2.13 and section 6 step 6.4)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from core import color_management as cm
from core import depth_of_field as dof
from core import geometry, pipeline
from core.decoder import ColorSpace, DecodedPhoto
from core.depth_of_field import DEFAULT_DEPTH, DepthOfField
from core.edit_state import Edit, EditState
from core.exporter import ExportOptions, can_copy_original
from core.geometry import Geometry, Rect, Window
from core.skin_range import SkinStroke

JPEG = EditState(is_raw=False)
BLURRED = DepthOfField(amount=100.0)


def _photo(image: np.ndarray, encoded: bool = False) -> DecodedPhoto:
    """Return linear sRGB *image* as a decoded sRGB JPEG."""
    working = cm.apply_matrix(image.astype(np.float32), cm.SRGB_TO_PROPHOTO)
    shape = image.shape[:2]
    return DecodedPhoto(
        image=np.ascontiguousarray(working),
        is_raw=False,
        clipped=np.zeros(shape, dtype=bool),
        color_space=ColorSpace.SRGB,
        encoded=np.zeros((*shape, 3), np.uint8) if encoded else None,
    )


def _noise(shape: tuple[int, int] = (240, 360), seed: int = 0) -> np.ndarray:
    """Return a frame of grey speckles, which a blur visibly smooths."""
    rng = np.random.default_rng(seed)
    grey = rng.uniform(0.05, 0.4, shape).astype(np.float32)
    return np.repeat(grey[..., None], 3, axis=2)


def _develop(photo: DecodedPhoto, depth: DepthOfField | None, **kwargs: object) -> np.ndarray:
    """Return *photo* with *depth* developed to 16 bit, as floats from 0 to 1."""
    framed = geometry.apply(photo, Geometry(depth=depth or DEFAULT_DEPTH))
    out = pipeline.develop(framed, JPEG, bits=pipeline.SIXTEEN_BIT, **kwargs)  # type: ignore[arg-type]
    return out.astype(np.float64) / 65535.0


# -- what is held ------------------------------------------------------------


def test_nothing_is_blurred_until_the_amount_is_raised() -> None:
    assert DEFAULT_DEPTH.is_default
    assert not DEFAULT_DEPTH.blurs
    assert BLURRED.blurs
    assert not BLURRED.is_default


def test_the_sliders_are_held_within_their_ranges() -> None:
    depth = DEFAULT_DEPTH.with_values(amount=500.0, transition=-3.0, highlights=101.0, ratio=0.0)

    assert depth.amount == dof.AMOUNT_MAX
    assert depth.transition == dof.TRANSITION_MIN
    assert depth.highlights == dof.HIGHLIGHTS_MAX
    assert depth.ratio == dof.RATIO_MIN


def test_the_depth_of_field_is_stored_and_read_back() -> None:
    stroke = SkinStroke(points=((0.2, 0.3), (0.25, 0.35)), radius=0.02, adding=False)
    depth = replace(
        BLURRED,
        shape=dof.SHAPE_BAND,
        centre=(0.4, 0.6),
        handle=(0.45, 0.5),
        ratio=0.5,
        transition=70.0,
        bokeh=dof.BOKEH_HEXAGON,
        highlights=20.0,
        strokes=(replace(stroke, snap=False),),
    )

    assert DepthOfField.from_dict(depth.to_dict()) == depth
    stored = Geometry(depth=depth)
    assert Geometry.from_dict(stored.to_dict()).depth == depth


def test_odd_stored_data_falls_back() -> None:
    assert DepthOfField.from_dict(None) == DEFAULT_DEPTH
    assert DepthOfField.from_dict({"amount": "lots", "shape": "star", "bokeh": 3}) == DEFAULT_DEPTH
    read = DepthOfField.from_dict({"amount": 40, "centre": [0.1], "strokes": [{"radius": 1}]})
    assert read.amount == 40.0
    assert read.centre == dof.DEFAULT_CENTRE
    assert read.strokes == ()
    assert Geometry.from_dict({}).depth == DEFAULT_DEPTH


def test_switched_off_nothing_is_blurred_and_everything_is_kept() -> None:
    off = replace(BLURRED, enabled=False, centre=(0.3, 0.3))

    assert not off.blurs
    assert not off.is_default
    assert DepthOfField.from_dict(off.to_dict()) == off
    assert DepthOfField.from_dict({"amount": 40}).enabled
    photo = _photo(_noise(), encoded=True)
    framed = geometry.apply(photo, Geometry(depth=off))
    assert framed.depth is None
    assert pipeline.is_passthrough(framed, JPEG)
    assert can_copy_original(Path("photo.jpg"), JPEG, Geometry(depth=off), ExportOptions())


def test_changing_the_shape_puts_it_where_that_shape_starts() -> None:
    moved = replace(BLURRED, centre=(0.2, 0.2), handle=(0.3, 0.2))
    band = moved.with_shape(dof.SHAPE_BAND)

    assert band.shape == dof.SHAPE_BAND
    assert band.centre == dof.DEFAULT_CENTRE
    assert band.handle == dof.DEFAULT_BAND_HANDLE
    assert band.amount == 100.0
    assert band.with_shape(dof.SHAPE_BAND) is band


def test_a_depth_of_field_set_in_any_way_counts_as_an_edit() -> None:
    reference = Edit(JPEG)

    assert Edit(JPEG, Geometry(depth=BLURRED)).is_untouched(reference) is False
    moved = replace(DEFAULT_DEPTH, centre=(0.3, 0.3))
    assert Edit(JPEG, Geometry(depth=moved)).is_untouched(reference) is False
    assert Edit(JPEG, Geometry()).is_untouched(reference)


# -- where it is sharp -----------------------------------------------------------


def test_the_shape_is_sharp_and_the_blur_grows_beyond_it() -> None:
    size = (400, 300)
    depth = replace(BLURRED, centre=(0.5, 0.5), handle=(0.5, 0.25), ratio=1.0, transition=10.0)
    blurred = dof.blurriness(depth, size)

    assert blurred.shape == (300, 400)
    assert blurred[150, 200] == 0.0
    # 75 pixels up is the edge; just inside is sharp, the corner all blurred.
    assert blurred[80, 200] == 0.0
    assert blurred[0, 0] == 1.0
    along = blurred[150, 280:400]
    assert np.all(np.diff(along) >= 0.0)
    assert 0.0 < blurred[150, 285] < 1.0


def test_an_ellipse_is_longer_along_its_handle() -> None:
    size = (400, 400)
    depth = replace(BLURRED, handle=(0.5, 0.2), ratio=0.5, transition=0.0)
    blurred = dof.blurriness(depth, size)

    # 120 pixels up is inside the long axis, 120 across is outside the short one.
    assert blurred[200 - 110, 200] == 0.0
    assert blurred[200, 200 + 110] > 0.5


def test_a_band_is_sharp_along_its_whole_length() -> None:
    size = (400, 300)
    depth = replace(BLURRED, shape=dof.SHAPE_BAND, handle=(0.5, 0.4), transition=0.0)
    blurred = dof.blurriness(depth, size)

    assert np.all(blurred[150, :] == 0.0)
    # Thirty pixels either side of the middle line.
    assert np.all(blurred[122:178, 0] == 0.0)
    assert blurred[0, 200] == 1.0 and blurred[299, 200] == 1.0


def test_strokes_keep_sharp_or_blur_and_the_later_one_wins() -> None:
    size = (400, 300)
    corner = ((0.1, 0.1),)
    keep = SkinStroke(points=corner, radius=0.05, adding=True, snap=False)
    blur = SkinStroke(points=((0.5, 0.5),), radius=0.05, adding=False, snap=False)

    kept = dof.blurriness(replace(BLURRED, strokes=(keep, blur)), size)
    assert kept[30, 40] == 0.0
    assert kept[150, 200] == 1.0

    blurred_again = replace(keep, adding=False)
    both = dof.blurriness(replace(BLURRED, strokes=(keep, blurred_again)), size)
    assert both[30, 40] == 1.0
    back = dof.blurriness(replace(BLURRED, strokes=(blurred_again, keep)), size)
    assert back[30, 40] == 0.0


def test_the_shape_follows_a_turn_and_a_crop() -> None:
    size = (400, 300)
    depth = replace(BLURRED, centre=(0.25, 0.25), handle=(0.25, 0.1), ratio=1.0, transition=0.0)
    for shape in (
        Geometry(depth=depth, quarter_turns=1),
        Geometry(depth=depth, mirrored=True),
        Geometry(depth=depth, crop=Rect(0.0, 0.0, 0.5, 0.5)),
    ):
        framed = geometry.framed_depth(shape, size)
        assert framed is not None
        centre = geometry.shown_point(depth.centre, size, shape)
        handle = geometry.shown_point(depth.handle, size, shape)
        assert np.allclose(framed.centre, centre)
        assert np.allclose(framed.handle, handle)


def test_the_framed_photo_carries_the_depth_of_field_only_when_it_blurs() -> None:
    photo = _photo(_noise())

    assert geometry.apply(photo, Geometry(depth=DEFAULT_DEPTH)).depth is None
    turned = geometry.apply(photo, Geometry(depth=BLURRED, quarter_turns=1))
    assert turned.depth == geometry.framed_depth(
        Geometry(depth=BLURRED, quarter_turns=1), photo.size
    )
    assert geometry.frame_key(Geometry(depth=BLURRED)) == Geometry()


# -- the blur ---------------------------------------------------------------


def test_with_no_blur_the_photo_is_exactly_as_before() -> None:
    photo = _photo(_noise())
    before = pipeline.develop(photo, JPEG, bits=pipeline.SIXTEEN_BIT)
    nothing = geometry.apply(photo, Geometry(depth=replace(DEFAULT_DEPTH, centre=(0.1, 0.1))))

    assert nothing.depth is None
    assert np.array_equal(pipeline.develop(nothing, JPEG, bits=pipeline.SIXTEEN_BIT), before)


def test_the_sharp_shape_keeps_every_pixel_and_the_rest_is_smoothed() -> None:
    photo = _photo(_noise())
    depth = replace(BLURRED, transition=0.0)
    before = _develop(photo, None)
    after = _develop(photo, depth)

    assert np.array_equal(after[110:130, 170:190], before[110:130, 170:190])
    assert after[:30, :30].std() < 0.3 * before[:30, :30].std()
    # The blur moves light around; it does not brighten or darken the photo.
    assert abs(after[:30, :30].mean() - before[:30, :30].mean()) < 0.02


def test_more_blur_smooths_more() -> None:
    photo = _photo(_noise())
    some = _develop(photo, replace(BLURRED, amount=20.0))
    all_ = _develop(photo, BLURRED)

    assert all_[:40, :40].std() < some[:40, :40].std()


def test_a_jpeg_with_blur_is_no_longer_passed_straight_through() -> None:
    photo = _photo(_noise(), encoded=True)

    assert pipeline.is_passthrough(photo, JPEG)
    blurred = geometry.apply(photo, Geometry(depth=BLURRED))
    assert not pipeline.is_passthrough(blurred, JPEG)
    assert not pipeline.is_passthrough(blurred)
    options = ExportOptions()
    source = Path("photo.jpg")
    assert can_copy_original(source, JPEG, Geometry(), options)
    assert not can_copy_original(source, JPEG, Geometry(depth=BLURRED), options)


def test_the_sharp_subject_does_not_bleed_into_the_blur_around_it() -> None:
    image = np.full((240, 360, 3), 0.05, dtype=np.float32)
    image[90:150, 150:210] = 0.9
    photo = _photo(image)
    # Sharp exactly over the bright square, blurred straight beyond it.
    depth = replace(
        BLURRED, centre=(0.5, 0.5), handle=(0.5, 90 / 240), ratio=1.0, transition=0.0
    )
    depth = replace(
        depth,
        shape=dof.SHAPE_BAND,
        handle=(0.5, 0.5 - 31 / 240),
    )
    after = _develop(photo, depth)
    dark = _develop(_photo(np.full((240, 360, 3), 0.05, dtype=np.float32)), None)

    # Just above and below the band the background stays as dark as it was.
    assert abs(after[50, 180].mean() - dark[50, 180].mean()) < 0.03


def test_bright_points_become_bright_discs() -> None:
    image = np.full((240, 360, 3), 0.02, dtype=np.float32)
    image[30:33, 30:33] = 1.0
    photo = _photo(image)
    plain = _develop(photo, replace(BLURRED, highlights=0.0))
    lit = _develop(photo, replace(BLURRED, highlights=100.0))

    area = (slice(15, 50), slice(15, 50))
    assert lit[area].max() > plain[area].max() + 0.05
    # A wide flat area stays as bright whatever the highlights slider says.
    flat = _photo(np.full((240, 360, 3), 0.8, dtype=np.float32))
    assert np.allclose(
        _develop(flat, replace(BLURRED, highlights=100.0))[:20, :20],
        _develop(flat, replace(BLURRED, highlights=0.0))[:20, :20],
        atol=2e-3,
    )


def test_the_bokeh_shapes() -> None:
    disc = pipeline.bokeh_kernel(8.0, dof.BOKEH_CIRCLE)
    hexagon = pipeline.bokeh_kernel(8.0, dof.BOKEH_HEXAGON)

    assert disc.shape == hexagon.shape == (17, 17)
    assert disc.max() == hexagon.max() == 1.0
    assert disc[8, 0] > 0.3 and disc[8, 16] > 0.3
    # The hexagon reaches its corners left and right and only its flat sides
    # up and down, short of a full radius.
    assert hexagon[8, 1] > 0.5
    assert hexagon[0, 8] < 0.1 < disc[0, 8]
    assert disc.sum() > hexagon.sum()


def test_the_exposure_scales_the_blur_with_the_photo() -> None:
    photo = geometry.apply(_photo(_noise()), Geometry(depth=BLURRED))
    field = pipeline.depth_field(photo, JPEG)
    assert field is not None
    rows, columns = pipeline.depth_axes(Window.whole(photo.size))
    blurred = pipeline.apply_depth(photo.image, field, rows, columns, 0.0)
    brighter = pipeline.apply_depth(photo.image * 2.0, field, rows, columns, 1.0)

    assert np.allclose(brighter, blurred * 2.0, rtol=1e-4, atol=1e-6)
    assert pipeline.depth_field(photo, JPEG.with_value("exposure", 2.0)) is not None
    assert pipeline.depth_key(JPEG, photo.depth) == pipeline.depth_key(
        JPEG.with_value("exposure", 2.0), photo.depth
    )


def test_a_window_comes_out_as_the_whole_photo_would() -> None:
    photo = geometry.apply(_photo(_noise((240, 360))), Geometry(depth=BLURRED))
    field = pipeline.depth_field(photo, JPEG)
    whole = pipeline.develop(photo, JPEG, depth=field)
    pane = Window(left=100, top=40, right=260, bottom=200, frame=(360, 240))
    part = geometry.window_of(photo, pane)
    window = pipeline.develop(part, JPEG, window=pane, depth=field)

    assert np.array_equal(window, whole[40:200, 100:260])


def test_the_proxy_and_the_full_size_photo_agree() -> None:
    small = _noise((240, 360))
    large = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
    proxy = geometry.apply(_photo(small), Geometry(depth=BLURRED))
    full = geometry.apply(_photo(large), Geometry(depth=BLURRED))
    shown = pipeline.develop(proxy, JPEG).astype(np.int16)
    exported = pipeline.develop(full, JPEG)
    shrunk = exported.reshape(240, 2, 360, 2, 3).mean(axis=(1, 3))

    # Away from the sharp middle, where the proxy's own detail differs.
    assert np.abs(shown[:60, :60] - shrunk[:60, :60]).max() <= 4
