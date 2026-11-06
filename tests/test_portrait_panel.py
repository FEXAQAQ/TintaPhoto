"""Tests for the portrait's panel (PLAN.md 2.10, 3.15)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from core import skin_range
from core.edit_state import EditState
from core.skin_range import SkinRange, SkinStroke

pytest.importorskip("PySide6.QtWidgets")

from app.i18n import tr  # noqa: E402
from ui import theme  # noqa: E402
from ui.portrait_panel import BRUSH_ADD, BRUSH_REMOVE, PortraitPanel  # noqa: E402

STROKE = SkinStroke(points=((0.5, 0.5),), radius=0.03)


@pytest.fixture
def panel(qt_app: object) -> Iterator[PortraitPanel]:
    widget = PortraitPanel()
    widget.resize(theme.PANEL_WIDTH, 700)
    widget.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    yield widget
    widget.hide()
    widget.deleteLater()


def test_it_is_as_wide_as_the_inspector(panel: PortraitPanel) -> None:
    assert panel.width() == theme.PANEL_WIDTH


def test_the_effect_comes_first_then_the_area(panel: PortraitPanel) -> None:
    assert [group.header.text() for group in panel.groups()] == [
        tr("portrait.group.effect"),
        tr("portrait.group.range"),
    ]
    rows = panel.rows()
    assert list(rows) == ["skin", "whiten"]
    assert rows["whiten"].y() > rows["skin"].y()


def test_a_slider_says_what_changed(panel: PortraitPanel) -> None:
    seen: list[tuple[str, float, bool]] = []
    panel.value_set.connect(lambda name, value, dragging: seen.append((name, value, dragging)))
    panel.rows()["skin"].set_value(30.0, announce=True)
    assert seen[-1] == ("skin", pytest.approx(30.0), False)


def test_undo_shows_the_sliders_without_announcing_them(panel: PortraitPanel) -> None:
    seen: list[object] = []
    panel.value_set.connect(lambda *args: seen.append(args))
    panel.set_state(EditState.for_photo(False).with_value("whiten", 25.0))
    assert panel.rows()["whiten"].value == pytest.approx(25.0)
    assert not seen


def test_it_says_how_many_faces_it_found(panel: PortraitPanel) -> None:
    panel.set_faces(None)
    assert panel.faces_caption.text() == tr("adjust.faces.looking")
    panel.set_faces(2)
    assert panel.faces_caption.text() == tr("adjust.faces.many", count=2)
    panel.set_faces(0)
    assert panel.faces_caption.text() == tr("portrait.faces.none")
    # Skin can still be painted in where no face was found.
    assert panel.rows()["whiten"].isEnabled()


def test_the_brush_is_down_until_a_job_is_chosen(panel: PortraitPanel) -> None:
    changes: list[bool] = []
    panel.brush_changed.connect(lambda: changes.append(True))
    assert panel.brush is None

    panel.add_button.click()
    assert panel.brush == BRUSH_ADD
    panel.remove_button.click()
    assert panel.brush == BRUSH_REMOVE
    assert not panel.add_button.isChecked()
    panel.remove_button.click()  # pressed again: put down
    assert panel.brush is None
    assert len(changes) == 3
    assert "B" in panel.add_button.toolTip() and "E" in panel.remove_button.toolTip()


def test_the_brush_size_counts_thousandths_of_the_long_edge(panel: PortraitPanel) -> None:
    seen: list[tuple[float, bool]] = []
    panel.brush_size_set.connect(lambda radius, dragging: seen.append((radius, dragging)))
    assert panel.brush_size == skin_range.RADIUS_DEFAULT
    panel.brush_size_row.set_value(50.0, announce=True)
    assert seen[-1] == (pytest.approx(0.05), False)
    panel.set_brush_size(5.0)
    assert panel.brush_size == skin_range.RADIUS_MAX


def test_the_area_shows_the_range_and_says_how_it_was_changed(panel: PortraitPanel) -> None:
    autos: list[bool] = []
    panel.auto_set.connect(autos.append)
    panel.set_skin(SkinRange())
    assert panel.auto_switch.isChecked()
    assert panel.strokes_caption.text() == tr("portrait.hint")
    assert not panel.reset_button.isEnabled()

    panel.set_skin(SkinRange(auto=False, strokes=(STROKE, STROKE)))
    assert not panel.auto_switch.isChecked()
    assert panel.strokes_caption.text() == tr("portrait.strokes.many", count=2)
    assert panel.reset_button.isEnabled()
    assert not autos  # shown, not announced

    panel.auto_switch.click()
    assert autos == [True]


def test_sticking_to_skin_starts_on_and_showing_starts_off(panel: PortraitPanel) -> None:
    assert panel.snap
    assert not panel.showing
    shown: list[bool] = []
    panel.showing_changed.connect(shown.append)
    panel.set_showing(True)
    assert panel.showing and shown == [True]
