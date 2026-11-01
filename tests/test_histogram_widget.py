"""Tests for the histogram's two ways of drawing (A-12) and its triangles (A-13)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from app.settings import HISTOGRAM_LUMA, HISTOGRAM_RGB

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from ui import theme  # noqa: E402
from ui.histogram_widget import TRIANGLE_SIZE, HistogramWidget, luma_values  # noqa: E402

WIDTH = 300


@pytest.fixture
def histogram(qt_app: object) -> Iterator[HistogramWidget]:
    """Return a shown histogram the width of the side panel."""
    widget = HistogramWidget()
    widget.resize(WIDTH, theme.HISTOGRAM_HEIGHT)
    widget.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    yield widget
    widget.hide()
    widget.deleteLater()


def _colourful() -> np.ndarray:
    """A photo with strong, separate red, green and blue areas."""
    image = np.zeros((60, 90, 3), dtype=np.uint8)
    image[:, :30, 0] = 220
    image[:, 30:60, 1] = 180
    image[:, 60:, 2] = 140
    return image


def test_luma_weighs_the_channels_by_their_brightness() -> None:
    pixels = np.array([[[0, 0, 0], [255, 255, 255], [128, 128, 128], [0, 255, 0]]], np.uint8)
    assert luma_values(pixels).tolist() == [[0, 255, 128, round(0.7152 * 255)]]


def test_a_grey_photo_has_one_brightness(histogram: HistogramWidget) -> None:
    histogram.set_photo(np.full((40, 40, 3), 90, dtype=np.uint8))
    counts = histogram._luma_counts
    assert counts is not None
    assert counts.sum() == 20 * 20
    assert counts[90] == counts.sum()


def test_the_plot_starts_in_three_channels(histogram: HistogramWidget) -> None:
    assert histogram.mode == HISTOGRAM_RGB


def test_toggling_swaps_the_mode_and_says_so(histogram: HistogramWidget) -> None:
    heard: list[str] = []
    histogram.mode_changed.connect(heard.append)
    histogram.toggle_mode()
    histogram.toggle_mode()
    assert heard == [HISTOGRAM_LUMA, HISTOGRAM_RGB]


def test_an_unknown_mode_means_three_channels(histogram: HistogramWidget) -> None:
    histogram.set_mode(HISTOGRAM_LUMA)
    histogram.set_mode("waveform")
    assert histogram.mode == HISTOGRAM_RGB


def test_either_arrow_swaps_the_mode(histogram: HistogramWidget) -> None:
    histogram.next_button.click()
    assert histogram.mode == HISTOGRAM_LUMA
    histogram.next_button.click()
    assert histogram.mode == HISTOGRAM_RGB
    histogram.previous_button.click()
    assert histogram.mode == HISTOGRAM_LUMA


def test_the_title_names_the_mode(histogram: HistogramWidget) -> None:
    rgb = histogram.title()
    histogram.toggle_mode()
    assert histogram.title() != rgb
    assert histogram.title()


def test_the_arrows_sit_either_side_of_a_narrower_plot(histogram: HistogramWidget) -> None:
    plot = histogram.plot_rect()
    left = histogram.previous_button.geometry()
    right = histogram.next_button.geometry()
    assert left.right() < plot.left() and plot.right() < right.left()
    assert left.center().y() == right.center().y() == plot.center().y()
    assert plot.top() >= theme.HISTOGRAM_TITLE_HEIGHT
    for size in (left.width(), left.height(), right.width(), right.height()):
        assert size == theme.SMALL_ICON_BUTTON_SIZE


def test_the_arrows_name_their_shortcut(histogram: HistogramWidget) -> None:
    from app.i18n import tr

    assert tr("shortcut.histogram_mode") in histogram.next_button.toolTip()
    assert tr("shortcut.histogram_mode") in histogram.previous_button.toolTip()


def test_clicking_a_triangle_marks_shadows_and_leaves_the_mode(
    histogram: HistogramWidget,
) -> None:
    histogram.set_photo(_colourful())
    corner = QPoint(
        theme.HISTOGRAM_EDGE + TRIANGLE_SIZE // 2, theme.HISTOGRAM_TITLE_HEIGHT // 2
    )
    QTest.mouseClick(histogram, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, corner)
    assert histogram.mode == HISTOGRAM_RGB
    assert histogram.showing_clipping == (False, True)


def test_luminance_is_drawn_without_hues(histogram: HistogramWidget) -> None:
    histogram.set_photo(_colourful())
    histogram.set_mode(HISTOGRAM_LUMA)
    pixels = histogram.grab().toImage()
    plot = histogram.plot_rect()
    tinted = 0
    for y in range(plot.top(), plot.bottom()):
        for x in range(plot.left(), plot.right()):
            colour = pixels.pixelColor(x, y)
            if max(colour.red(), colour.green(), colour.blue()) - min(
                colour.red(), colour.green(), colour.blue()
            ) > 2:
                tinted += 1
    assert tinted == 0


def test_three_channels_are_drawn_in_their_hues(histogram: HistogramWidget) -> None:
    histogram.set_photo(_colourful())
    pixels = histogram.grab().toImage()
    reds = sum(
        1
        for y in range(pixels.height())
        for x in range(pixels.width())
        if pixels.pixelColor(x, y).red() > pixels.pixelColor(x, y).green() + 30
    )
    assert reds > 0
