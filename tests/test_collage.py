"""Tests for the two photo collage (PLAN.md 2.8, 3.13)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from core import collage
from core.collage import Cell, Layout, Placement

LANDSCAPE = collage.ORIENTATION_LANDSCAPE
PORTRAIT = collage.ORIENTATION_PORTRAIT
SIDE = collage.SPLIT_SIDE_BY_SIDE
STACKED = collage.SPLIT_STACKED


def _solid(colour: tuple[int, int, int], size: tuple[int, int] = (300, 200)) -> np.ndarray:
    width, height = size
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[...] = colour
    return image


def _gradient(size: tuple[int, int]) -> np.ndarray:
    """A smooth 8 bit picture whose every region is different."""
    width, height = size
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    image = np.stack(
        [xs + 0 * ys, ys + 0 * xs, 0.5 * (xs + ys)], axis=-1
    )
    return np.round(image * 255).astype(np.uint8)


# -- the canvas and its halves (requirement section 4) -----------------------


@pytest.mark.parametrize(
    ("orientation", "split", "half"),
    [
        (LANDSCAPE, SIDE, (960, 1080)),
        (LANDSCAPE, STACKED, (1920, 540)),
        (PORTRAIT, SIDE, (540, 1350)),
        (PORTRAIT, STACKED, (1080, 675)),
    ],
)
def test_without_a_gap_each_half_is_the_size_asked_for(
    orientation: str, split: str, half: tuple[int, int]
) -> None:
    layout = Layout(orientation=orientation, split=split)
    first, second = collage.cells(layout)
    assert (first.width, first.height) == half
    assert (second.width, second.height) == half
    assert layout.canvas_size == collage.CANVAS_SIZES[orientation]


@pytest.mark.parametrize("gap", range(collage.GAP_MIN, collage.GAP_MAX + 1))
@pytest.mark.parametrize("split", [SIDE, STACKED])
@pytest.mark.parametrize("orientation", [LANDSCAPE, PORTRAIT])
def test_the_gap_comes_out_of_the_halves_and_never_grows_the_canvas(
    orientation: str, split: str, gap: int
) -> None:
    layout = Layout(orientation=orientation, split=split, gap=gap)
    width, height = layout.canvas_size
    first, second = collage.cells(layout)
    if split == SIDE:
        assert first.width + gap + second.width == width
        assert second.x == first.x + first.width + gap
        assert first.height == second.height == height
    else:
        assert first.height + gap + second.height == height
        assert second.y == first.y + first.height + gap
        assert first.width == second.width == width
    # An odd gap leaves one pixel over, which goes to the second half.
    assert 0 <= (second.width + second.height) - (first.width + first.height) <= 1


def test_values_out_of_range_are_brought_back() -> None:
    layout = Layout(orientation="sideways", split="diagonal", fit="stretch", gap=99, background="red")
    sane = layout.sane()
    assert sane == Layout(gap=collage.GAP_MAX)
    assert Layout(gap=-5).sane().gap == collage.GAP_MIN


# -- placing a photo -----------------------------------------------------------

CELL = Cell(0, 0, 960, 1080)


@pytest.mark.parametrize("photo", [(6000, 4000), (4000, 6000), (1000, 1000)])
def test_cover_fills_the_half_and_contain_shows_all_of_the_photo(
    photo: tuple[int, int],
) -> None:
    x, y, width, height = collage.photo_rect(Placement(), photo, CELL, collage.FIT_COVER)
    assert x <= 1e-6 and y <= 1e-6
    assert x + width >= CELL.width - 1e-6 and y + height >= CELL.height - 1e-6
    assert min(width - CELL.width, height - CELL.height) == pytest.approx(0.0, abs=1e-6)

    x, y, width, height = collage.photo_rect(Placement(), photo, CELL, collage.FIT_CONTAIN)
    assert x >= -1e-6 and y >= -1e-6
    assert x + width <= CELL.width + 1e-6 and y + height <= CELL.height + 1e-6
    assert width / height == pytest.approx(photo[0] / photo[1])


def test_a_covering_photo_never_slides_far_enough_to_uncover_the_background() -> None:
    photo = (6000, 4000)
    placement = collage.panned(Placement(), (10_000.0, -10_000.0), photo, CELL, collage.FIT_COVER)
    x, y, width, height = collage.photo_rect(placement, photo, CELL, collage.FIT_COVER)
    assert x == pytest.approx(0.0, abs=1e-6)
    assert y <= 1e-6 and y + height >= CELL.height - 1e-6


def test_a_photo_smaller_than_its_half_stays_centred() -> None:
    photo = (6000, 4000)
    placement = collage.panned(Placement(), (300.0, 300.0), photo, CELL, collage.FIT_CONTAIN)
    x, y, width, height = collage.photo_rect(placement, photo, CELL, collage.FIT_CONTAIN)
    assert x == pytest.approx(0.0, abs=1e-6)
    assert y + height / 2 == pytest.approx(CELL.height / 2)


def test_panning_moves_the_photo_by_as_much_as_the_pointer() -> None:
    photo = (6000, 4000)
    before = collage.photo_rect(Placement(), photo, CELL, collage.FIT_COVER)
    placement = collage.panned(Placement(), (100.0, 0.0), photo, CELL, collage.FIT_COVER)
    after = collage.photo_rect(placement, photo, CELL, collage.FIT_COVER)
    assert after[0] - before[0] == pytest.approx(100.0)


def test_zooming_keeps_the_point_under_the_pointer_where_it_was() -> None:
    photo = (6000, 4000)
    anchor = (700.0, 300.0)
    before = collage.photo_rect(Placement(), photo, CELL, collage.FIT_COVER)
    placement = collage.zoomed(Placement(), 2.0, anchor, photo, CELL, collage.FIT_COVER)
    after = collage.photo_rect(placement, photo, CELL, collage.FIT_COVER)
    for axis in (0, 1):
        fraction_before = (anchor[axis] - before[axis]) / before[axis + 2]
        fraction_after = (anchor[axis] - after[axis]) / after[axis + 2]
        assert fraction_after == pytest.approx(fraction_before)
    assert placement.zoom == 2.0


def test_the_zoom_stays_in_range() -> None:
    photo = (6000, 4000)
    assert collage.zoomed(Placement(), 99.0, (0, 0), photo, CELL, collage.FIT_COVER).zoom == (
        collage.ZOOM_MAX
    )
    assert collage.zoomed(Placement(), 0.1, (0, 0), photo, CELL, collage.FIT_COVER).zoom == (
        collage.ZOOM_MIN
    )


# -- rendering -----------------------------------------------------------------


@pytest.mark.parametrize("background", [collage.BACKGROUND_WHITE, collage.BACKGROUND_BLACK])
def test_each_photo_lands_in_its_own_half_and_the_gap_is_the_background(
    background: str,
) -> None:
    layout = Layout(gap=10, background=background)
    red, blue = _solid((200, 30, 30)), _solid((30, 30, 200))
    result = collage.render(layout, [red, blue], [Placement(), Placement()])
    assert result.shape == (1080, 1920, 3) and result.dtype == np.uint8
    first, second = collage.cells(layout)
    assert np.all(result[:, first.x + 2 : first.x + first.width - 2] == (200, 30, 30))
    assert np.all(result[:, second.x + 2 : second.x + second.width - 2] == (30, 30, 200))
    gap = result[:, first.x + first.width : second.x]
    assert gap.shape[1] == 10
    assert np.all(gap == collage.BACKGROUND_SRGB8[background])


def test_an_empty_half_is_left_the_background_colour() -> None:
    layout = Layout(split=STACKED, background=collage.BACKGROUND_BLACK)
    result = collage.render(layout, [_solid((90, 90, 90)), None], [Placement(), Placement()])
    first, second = collage.cells(layout)
    assert np.all(result[second.y + 1 :] == 0)
    assert np.all(result[: first.height - 1] == 90)


def test_contain_leaves_bars_of_the_background_beside_the_photo() -> None:
    layout = Layout(fit=collage.FIT_CONTAIN, background=collage.BACKGROUND_WHITE)
    landscape_photo = _solid((10, 120, 10), (600, 400))
    result = collage.render(layout, [landscape_photo, None], [Placement(), Placement()])
    first = collage.cells(layout)[0]
    # 960 wide makes the photo 640 tall; the rest of 1080 is white.
    bar = (first.height - 640) // 2
    assert np.all(result[: bar - 1, : first.width] == 255)
    assert np.all(result[first.height - bar + 1 :, : first.width] == 255)
    assert np.all(result[bar + 2 : first.height - bar - 2, 2 : first.width - 2] == (10, 120, 10))


def test_the_same_placement_gives_the_same_collage_from_any_size_of_photo() -> None:
    """The preview works on a small copy and the export on the full photo."""
    layout = Layout(orientation=PORTRAIT, split=SIDE)
    placement = Placement(zoom=1.7, focus_x=0.3, focus_y=0.6)
    small = _gradient((900, 600))
    large = _gradient((3600, 2400))
    from_small = collage.render(layout, [small, None], [placement, Placement()])
    from_large = collage.render(layout, [large, None], [placement, Placement()])
    difference = np.abs(from_small.astype(int) - from_large.astype(int))
    assert difference.mean() < 0.6
    assert difference.max() <= 3


def test_sixteen_bit_output_keeps_its_depth() -> None:
    layout = Layout()
    deep = (_gradient((800, 600)).astype(np.uint16) * 257)
    result = collage.render(layout, [deep, deep], [Placement(), Placement()], bits=16)
    assert result.dtype == np.uint16 and result.shape == (1080, 1920, 3)


def test_scaling_happens_in_linear_light() -> None:
    """A fine black and white pattern shrinks to the grey of half the light."""
    pattern = np.zeros((2160, 1920, 3), dtype=np.uint8)
    pattern[::2] = 255
    layout = Layout(split=SIDE)
    result = collage.render(layout, [pattern, None], [Placement(), Placement()])
    first = collage.cells(layout)[0]
    middle = result[first.height // 2, first.width // 2]
    # Half the light is sRGB 188, where averaging the encoded values gives 128.
    assert abs(int(middle[0]) - 188) <= 2


def test_compose_refuses_a_half_of_the_wrong_size() -> None:
    with pytest.raises(ValueError):
        collage.compose(Layout(), [np.zeros((10, 10, 3), dtype=np.float32), None])


def test_rendering_does_not_change_the_photos_given_to_it() -> None:
    photo = _gradient((400, 300))
    kept = photo.copy()
    collage.render(Layout(), [photo, photo], [Placement(zoom=2.0), Placement()])
    assert np.array_equal(photo, kept)


@pytest.mark.parametrize(
    ("format_", "bits", "suffix"),
    [("jpg", 8, ".jpg"), ("png", 8, ".png"), ("png", 16, ".png"), ("tiff", 16, ".tif")],
)
def test_a_finished_collage_is_written_in_every_format(
    tmp_path, format_: str, bits: int, suffix: str
) -> None:
    from core import exporter
    from core.exporter import ExportOptions

    options = ExportOptions(format=format_, bits=bits)
    layout = Layout(orientation=PORTRAIT)
    canvas = collage.render(
        layout, [_gradient((300, 200)), None], [Placement(), Placement()], bits=options.depth
    )
    result = exporter.export_pixels(canvas, tmp_path / "collage.jpg", options)
    assert result.path.suffix == options.extension and result.path.exists()
    assert not result.copied
    if format_ == "tiff":
        import tifffile

        assert tifffile.imread(result.path).shape == (1350, 1080, 3)
    else:
        import cv2

        decoded = cv2.imdecode(np.fromfile(result.path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert decoded.shape == (1350, 1080, 3)
        assert decoded.dtype == (np.uint16 if bits == 16 and format_ == "png" else np.uint8)


# -- the seam (PLAN.md 2.8 K-10) ----------------------------------------------


@pytest.mark.parametrize("seam", [0.2, 0.37, 0.5, 0.63, 0.8])
@pytest.mark.parametrize("gap", [0, 1, 7, 20])
def test_the_seam_moves_the_halves_but_never_the_canvas(seam: float, gap: int) -> None:
    layout = Layout(seam=seam, gap=gap)
    first, second = collage.cells(layout)
    assert first.width + gap + second.width == 1920
    middle_of_gap = first.width + gap / 2
    assert abs(middle_of_gap - 1920 * seam) <= 1.0


def test_the_seam_stays_in_range() -> None:
    assert Layout(seam=0.0).sane().seam == collage.SEAM_MIN
    assert Layout(seam=5.0).sane().seam == collage.SEAM_MAX
    assert Layout(seam=float("nan")).sane().seam == collage.SEAM_DEFAULT


# -- linked: one scene shot twice ------------------------------------------------


def test_linked_both_photos_are_laid_over_the_whole_canvas() -> None:
    layout = Layout(linked=True, fit=collage.FIT_CONTAIN)
    photo = (2000, 1333)
    placement = Placement(zoom=1.3, focus_x=0.45, focus_y=0.55)
    first = collage.placed_rect(layout, 0, photo, placement)
    second = collage.placed_rect(layout, 1, photo, placement)
    assert first == second
    # Linked always covers, and covers the canvas rather than a half.
    x, y, width, height = collage.placed_rect(layout, 0, photo, Placement())
    assert x <= 0 and y <= 0 and x + width >= 1920 - 1e-6 and y + height >= 1080 - 1e-6


def test_linked_each_half_shows_its_own_part_of_the_scene() -> None:
    """The left half of photo A beside the right half of photo B."""
    a = _gradient((1600, 900))
    b = (255 - _gradient((1600, 900))).astype(np.uint8)
    unlinked_a = collage.render(Layout(), [a, None], [Placement(), Placement()])
    linked = collage.render(Layout(linked=True), [a, b], [Placement(), Placement()])
    whole_a = collage.render(
        Layout(linked=True, seam=collage.SEAM_MAX), [a, None], [Placement(), Placement()]
    )
    whole_b = collage.render(
        Layout(linked=True, seam=collage.SEAM_MIN), [None, b], [Placement(), Placement()]
    )
    left, right = collage.cells(Layout(linked=True))
    assert np.abs(linked[:, 10:900].astype(int) - whole_a[:, 10:900].astype(int)).max() <= 1
    assert np.abs(linked[:, 1000:1900].astype(int) - whole_b[:, 1000:1900].astype(int)).max() <= 1
    # Not what the unlinked collage shows, which centres A in its half.
    assert np.abs(linked[:, 10:900].astype(int) - unlinked_a[:, 10:900].astype(int)).mean() > 10


def test_alignment_only_counts_when_linked() -> None:
    photo = (2000, 1333)
    shifted = collage.Alignment(shift_x=0.1, scale=1.1)
    plain = collage.placed_rect(Layout(), 1, photo, Placement(), shifted)
    assert plain == collage.placed_rect(Layout(), 1, photo, Placement())
    linked = collage.placed_rect(Layout(linked=True), 1, photo, Placement(), shifted)
    base = collage.placed_rect(Layout(linked=True), 1, photo, Placement())
    assert linked[2] == pytest.approx(base[2] * 1.1)
    centre = linked[0] + linked[2] / 2
    assert centre == pytest.approx(base[0] + base[2] / 2 + 0.1 * base[2])


def test_nudging_moves_the_aligned_photo_by_as_much_as_asked() -> None:
    rect = (0.0, 0.0, 2000.0, 1200.0)
    moved = collage.nudged(collage.UNALIGNED, (1.0, -3.0), rect)
    after = collage.aligned(rect, moved)
    assert after[0] == pytest.approx(1.0)
    assert after[1] == pytest.approx(-3.0)


def test_rescaling_keeps_the_point_under_the_pointer_still() -> None:
    rect = (-40.0, -60.0, 2000.0, 1200.0)
    start = collage.Alignment(shift_x=0.02, shift_y=-0.01, scale=1.0)
    anchor = (700.0, 300.0)
    before = collage.aligned(rect, start)
    result = collage.rescaled(start, 1.05, anchor, rect)
    after = collage.aligned(rect, result)
    for axis in (0, 1):
        assert (anchor[axis] - after[axis]) / after[axis + 2] == pytest.approx(
            (anchor[axis] - before[axis]) / before[axis + 2]
        )
    assert result.scale == pytest.approx(1.05)


def test_alignment_stays_in_range() -> None:
    wild = collage.Alignment(shift_x=3.0, shift_y=-3.0, scale=9.0).sane()
    assert wild == collage.Alignment(
        shift_x=collage.ALIGN_SHIFT_MAX, shift_y=-collage.ALIGN_SHIFT_MAX, scale=collage.ALIGN_SCALE_MAX
    )


def test_aligning_never_leaves_a_bare_strip() -> None:
    """Shrinking or moving the second photo zooms the pair in just enough."""
    layout = Layout(linked=True)
    size = (2000, 1333)
    alignment = collage.Alignment(shift_x=-0.03, shift_y=0.01, scale=0.95)
    placement = collage.linked_placement(layout, Placement(), [size, size], alignment)
    assert placement.zoom > 1.0
    left, top, width, height = collage.placed_rect(layout, 1, size, placement, alignment)
    half = collage.cells(layout)[1]
    assert left <= half.x + 1e-6 and top <= 1e-6
    assert left + width >= half.x + half.width - 1e-6 and top + height >= 1080 - 1e-6
    first = collage.placed_rect(layout, 0, size, placement)
    assert first[0] <= 1e-6 and first[0] + first[2] >= 1920 - 1e-6
    # Just enough: a hair less zoom would show the background again.
    less = replace(placement, zoom=placement.zoom * 0.99)
    assert not _covers(collage.placed_rect(layout, 1, size, less, alignment), half)

    black = np.zeros((1333, 2000, 3), dtype=np.uint8)
    result = collage.render(
        Layout(linked=True, background=collage.BACKGROUND_WHITE),
        [black, black],
        [placement, placement],
        [collage.UNALIGNED, alignment],
    )
    assert result.max() == 0


def test_a_placement_that_already_covers_is_left_alone() -> None:
    layout = Layout(linked=True)
    size = (2000, 1333)
    asked = Placement(zoom=1.5, focus_x=0.48, focus_y=0.52)
    assert collage.linked_placement(layout, asked, [size, size], collage.Alignment(shift_x=0.01)) == asked


def _covers(rect, cell) -> bool:
    left, top, width, height = rect
    return (
        left <= cell.x + 1e-6
        and top <= cell.y + 1e-6
        and left + width >= cell.x + cell.width - 1e-6
        and top + height >= cell.y + cell.height - 1e-6
    )


# -- the single photos of a carousel -----------------------------------------


@pytest.mark.parametrize("orientation", [LANDSCAPE, PORTRAIT])
@pytest.mark.parametrize("linked", [False, True])
def test_a_single_photo_takes_the_collage_size(orientation: str, linked: bool) -> None:
    layout = Layout(orientation=orientation, linked=linked)
    single = collage.render_single(layout, _gradient((600, 400)), Placement())
    width, height = layout.canvas_size
    assert single.shape == (height, width, 3)
    assert single.dtype == np.float32


def test_a_single_photo_keeps_the_point_its_half_was_centred_on() -> None:
    layout = Layout()
    placement = Placement(zoom=1.5, focus_x=0.4, focus_y=0.6)
    size = (6000, 4000)
    left, top, width, height = collage.single_rect(layout, size, placement)
    centre_x, centre_y = collage.canvas_cell(layout).centre
    assert (centre_x - left) / width == pytest.approx(0.4)
    assert (centre_y - top) / height == pytest.approx(0.6)
    # Zoomed as far past the canvas's fit as it was past its half's.
    cover = collage.fitted_scale(size, collage.canvas_cell(layout), collage.FIT_COVER)
    assert width / size[0] == pytest.approx(cover * 1.5)


def test_a_single_photo_shows_more_than_its_half() -> None:
    layout = Layout()
    size = (6000, 4000)
    half = collage.placed_rect(layout, 0, size, Placement())
    single = collage.single_rect(layout, size, Placement())
    first = collage.cells(layout)[0]
    canvas = collage.canvas_cell(layout)
    assert canvas.width / single[2] > first.width / half[2]


def test_a_filled_single_photo_leaves_no_background() -> None:
    layout = Layout(background=collage.BACKGROUND_BLACK)
    single = collage.render_single(layout, _solid((255, 255, 255), (300, 900)), Placement())
    assert single.min() > 0.99


def test_a_contained_single_photo_shows_all_of_itself() -> None:
    layout = Layout(fit=collage.FIT_CONTAIN, background=collage.BACKGROUND_BLACK)
    single = collage.render_single(layout, _solid((255, 255, 255), (300, 900)), Placement())
    height, width = single.shape[:2]
    assert single[height // 2, width // 2].min() > 0.99
    assert single[height // 2, 0].max() == 0.0
    assert single[0, width // 2].min() > 0.99


def test_a_linked_first_photo_is_the_collage_left_whole() -> None:
    layout = Layout(linked=True)
    size = (900, 600)
    placement = collage.linked_placement(layout, Placement(zoom=1.3, focus_x=0.4), [size, size])
    assert collage.single_rect(layout, size, placement) == pytest.approx(
        collage.placed_rect(layout, 0, size, placement)
    )


def test_a_linked_second_photo_leaves_its_alignment_out() -> None:
    layout = Layout(linked=True, background=collage.BACKGROUND_BLACK)
    size = (900, 600)
    alignment = collage.Alignment(shift_x=0.1, scale=0.85)
    placement = collage.linked_placement(layout, Placement(), [size, size], alignment)
    single = collage.render_single(layout, _solid((255, 255, 255), size), placement)
    assert single.min() > 0.99


def test_the_single_photos_are_not_named_like_a_taken_name() -> None:
    assert len(collage.SINGLE_SUFFIXES) == collage.SLOTS
    for suffix in collage.SINGLE_SUFFIXES:
        assert not suffix.lstrip("_").isdigit()
