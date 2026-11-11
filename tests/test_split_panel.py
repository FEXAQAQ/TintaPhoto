"""Tests for the panel and the cuts shown while a photo is cut (PLAN.md 3.17)."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QRectF  # noqa: E402

from app.i18n import tr  # noqa: E402
from core import split  # noqa: E402
from core.split import SplitSpec  # noqa: E402
from ui import theme  # noqa: E402
from ui.split_overlay import SplitOverlay  # noqa: E402
from ui.split_panel import SplitPanel  # noqa: E402


@pytest.fixture
def panel(qt_app: object) -> SplitPanel:
    return SplitPanel()


def _changes(panel: SplitPanel) -> list[SplitSpec]:
    seen: list[SplitSpec] = []
    panel.spec_changed.connect(seen.append)
    return seen


def test_it_is_as_wide_as_the_inspector(panel: SplitPanel) -> None:
    assert panel.width() == theme.PANEL_WIDTH


def test_it_shows_the_cut_it_is_given_without_announcing_it(panel: SplitPanel) -> None:
    seen = _changes(panel)
    panel.set_spec(SplitSpec(split.SPLIT_STACKED, 3))
    assert seen == []
    assert panel.split_choice.value == split.SPLIT_STACKED
    assert panel.count_choice.value == "3"
    assert panel.count_choice.isEnabled()


def test_choosing_announces_the_new_cut(panel: SplitPanel) -> None:
    seen = _changes(panel)
    panel.split_choice.buttons()[split.SPLIT_STACKED].click()
    panel.count_choice.buttons()["4"].click()
    assert seen[-1] == SplitSpec(split.SPLIT_STACKED, 4)


def test_a_grid_holds_the_count_at_four_and_gives_the_old_one_back(panel: SplitPanel) -> None:
    seen = _changes(panel)
    panel.count_choice.buttons()["3"].click()
    panel.split_choice.buttons()[split.SPLIT_GRID].click()
    assert seen[-1] == SplitSpec(split.SPLIT_GRID, 4)
    assert panel.count_choice.value == "4"
    assert not panel.count_choice.isEnabled()
    panel.split_choice.buttons()[split.SPLIT_SIDE_BY_SIDE].click()
    assert seen[-1] == SplitSpec(split.SPLIT_SIDE_BY_SIDE, 3)
    assert panel.count_choice.isEnabled()


def test_it_says_how_large_each_piece_comes_out(panel: SplitPanel) -> None:
    panel.set_spec(SplitSpec(split.SPLIT_STACKED, 2))
    panel.set_photo_size((40, 100))
    assert panel.size_label.text() == tr(
        "split.size", size=tr("export.pixels", width=40, height=50), count=2
    )
    panel.set_photo_size(None)
    assert panel.size_label.text() == ""


def test_the_cuts_are_where_the_export_cuts(qt_app: object) -> None:
    overlay = SplitOverlay()
    overlay.set_frame((400, 300))
    overlay.set_spec(SplitSpec(split.SPLIT_GRID))
    pieces = overlay.pieces_on_screen(QRectF(10, 20, 200, 150))
    assert pieces == [
        QRectF(10, 20, 100, 75),
        QRectF(110, 20, 100, 75),
        QRectF(10, 95, 100, 75),
        QRectF(110, 95, 100, 75),
    ]


def test_the_cuts_are_drawn_only_while_shown(qt_app: object) -> None:
    from PySide6.QtGui import QColor, QImage, QPainter

    overlay = SplitOverlay()
    overlay.set_frame((100, 100))
    overlay.set_spec(SplitSpec(split.SPLIT_SIDE_BY_SIDE, 2))
    image = QImage(100, 100, QImage.Format.Format_RGB32)

    def middle_after_painting() -> str:
        image.fill(QColor(theme.SCRIM))
        painter = QPainter(image)
        overlay.paint(painter, QRectF(0, 0, 100, 100))
        painter.end()
        return image.pixelColor(50, 80).name().upper()

    assert middle_after_painting() == theme.SCRIM
    overlay.set_shown(True)
    assert middle_after_painting() != theme.SCRIM
