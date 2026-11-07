"""Tests for the liquify panel (PLAN.md 2.11, 3.16)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from core import liquify

pytest.importorskip("PySide6.QtWidgets")

from app.i18n import tr  # noqa: E402
from ui import theme  # noqa: E402
from ui.liquify_panel import TOOL_ROWS, LiquifyPanel  # noqa: E402


@pytest.fixture
def panel(qt_app: object) -> Iterator[LiquifyPanel]:
    widget = LiquifyPanel()
    widget.resize(theme.PANEL_WIDTH, 700)
    widget.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    yield widget
    widget.hide()
    widget.deleteLater()


def test_it_is_as_wide_as_the_inspector(panel: LiquifyPanel) -> None:
    assert panel.width() == theme.PANEL_WIDTH


def test_the_tools_come_first_then_the_brush_then_the_face(panel: LiquifyPanel) -> None:
    assert [group.header.text() for group in panel.groups()] == [
        tr("liquify.group.tools"),
        tr("liquify.group.brush"),
        tr("liquify.group.face"),
    ]
    assert panel.strength_row.y() > panel.brush_size_row.y()


def test_all_seven_tools_are_laid_out_three_two_two(panel: LiquifyPanel) -> None:
    assert [len(row) for row in TOOL_ROWS] == [3, 2, 2]
    assert sorted(panel.tool_buttons) == sorted(liquify.TOOLS)
    rows = [{panel.tool_buttons[tool].y() for tool in row} for row in TOOL_ROWS]
    assert all(len(row) == 1 for row in rows)
    tops = [next(iter(row)) for row in rows]
    assert tops == sorted(tops)
    # The rows are as far apart as the buttons in a row.
    gap = tops[1] - tops[0] - theme.BUTTON_HEIGHT
    beside = panel.tool_buttons[liquify.TOOL_BLOAT]
    push = panel.tool_buttons[liquify.TOOL_PUSH]
    assert gap == beside.x() - (push.x() + push.width()) == theme.SPACE_SM
    # Every row runs edge to edge alike.
    first = panel.tool_buttons[liquify.TOOL_PUSH]
    last = panel.tool_buttons[liquify.TOOL_PUCKER]
    lower = panel.tool_buttons[liquify.TOOL_TWIRL_CCW]
    assert abs((last.x() + last.width()) - (lower.x() + lower.width())) <= 1
    for row in TOOL_ROWS:
        widths = {panel.tool_buttons[tool].width() for tool in row}
        assert max(widths) - min(widths) <= 1
    for button in panel.tool_buttons.values():
        assert button.height() == theme.BUTTON_HEIGHT
        assert button.toolTip()


def test_one_tool_is_always_pressed_and_push_comes_first(panel: LiquifyPanel) -> None:
    chosen: list[str] = []
    panel.tool_changed.connect(chosen.append)
    assert panel.tool == liquify.TOOL_PUSH

    panel.tool_buttons[liquify.TOOL_BLOAT].click()
    assert panel.tool == liquify.TOOL_BLOAT
    assert chosen == [liquify.TOOL_BLOAT]
    pressed = [tool for tool, button in panel.tool_buttons.items() if button.isChecked()]
    assert pressed == [liquify.TOOL_BLOAT]

    # Pressing the one already pressed keeps it pressed.
    panel.tool_buttons[liquify.TOOL_BLOAT].click()
    assert panel.tool == liquify.TOOL_BLOAT


def test_the_size_counts_thousandths_of_the_long_edge(panel: LiquifyPanel) -> None:
    sizes: list[tuple[float, bool]] = []
    panel.brush_size_set.connect(lambda radius, dragging: sizes.append((radius, dragging)))
    assert panel.brush_size_row.value == pytest.approx(50.0)
    assert panel.brush_size == pytest.approx(liquify.RADIUS_DEFAULT)

    panel.brush_size_row.set_value(120.0, announce=True)
    assert sizes[-1] == (pytest.approx(0.12), False)
    panel.set_brush_size(9.0)
    assert panel.brush_size == liquify.RADIUS_MAX
    assert panel.brush_size_row.value == pytest.approx(250.0)


def test_the_strength_counts_hundredths(panel: LiquifyPanel) -> None:
    assert panel.strength_row.value == pytest.approx(50.0)
    panel.strength_row.set_value(80.0, announce=True)
    assert panel.strength == pytest.approx(0.8)
    panel.set_strength(0.0)
    assert panel.strength == liquify.STRENGTH_MIN


def test_it_says_how_to_paint_or_how_many_strokes(panel: LiquifyPanel) -> None:
    assert panel.strokes_caption.text() == tr("liquify.hint")
    assert not panel.reset_button.isEnabled()
    panel.set_strokes(1)
    assert panel.strokes_caption.text() == tr("liquify.strokes.one")
    panel.set_strokes(4)
    assert panel.strokes_caption.text() == tr("liquify.strokes.many", count=4)
    assert panel.reset_button.isEnabled()


def test_the_buttons_along_the_bottom_ask_for_what_they_say(panel: LiquifyPanel) -> None:
    asked: list[str] = []
    panel.reset_requested.connect(lambda: asked.append("reset"))
    panel.leave_requested.connect(lambda: asked.append("leave"))
    panel.set_strokes(2)
    panel.reset_button.click()
    panel.leave_button.click()
    assert asked == ["reset", "leave"]
    assert tr("shortcut.liquify") in panel.leave_button.toolTip()


def test_a_language_change_keeps_what_was_chosen(panel: LiquifyPanel) -> None:
    panel.tool_buttons[liquify.TOOL_SMOOTH].click()
    panel.set_brush_size(0.1)
    panel.set_strength(0.3)
    panel.retranslate()
    assert panel.tool == liquify.TOOL_SMOOTH
    assert panel.brush_size == pytest.approx(0.1)
    assert panel.strength == pytest.approx(0.3)
    assert panel.tool_buttons[liquify.TOOL_SMOOTH].text() == tr("liquify.tool.smooth")


# -- the face sliders (L-11) -------------------------------------------------


def test_the_face_sliders_run_from_minus_to_plus_a_hundred(panel: LiquifyPanel) -> None:
    assert list(panel.face_rows) == list(liquify.FACE_NAMES)
    tops = [row.y() for row in panel.face_rows.values()]
    assert tops == sorted(tops)
    for name, row in panel.face_rows.items():
        assert row.value == 0.0
        assert row.spec.minimum == -100.0 and row.spec.maximum == 100.0
        assert row.spec.label_key == f"liquify.face.{name}"


def test_the_face_sliders_wait_for_a_face(panel: LiquifyPanel) -> None:
    panel.set_faces(None)
    assert panel.faces_caption.text() == tr("adjust.faces.looking")
    assert not any(row.isEnabled() for row in panel.face_rows.values())
    panel.set_faces(0)
    assert panel.faces_caption.text() == tr("liquify.faces.none")
    assert not any(row.isEnabled() for row in panel.face_rows.values())
    panel.set_faces(2)
    assert panel.faces_caption.text() == tr("adjust.faces.many", count=2)
    assert all(row.isEnabled() for row in panel.face_rows.values())


def test_a_face_slider_says_what_changed(panel: LiquifyPanel) -> None:
    seen: list[tuple[str, float, bool]] = []
    panel.face_value_set.connect(lambda name, value, dragging: seen.append((name, value, dragging)))
    panel.set_faces(1)
    panel.face_rows["eyes"].set_value(40.0, announce=True)
    assert seen[-1] == ("eyes", pytest.approx(40.0), False)
    assert panel.reset_button.isEnabled()  # a slider moved is something to reset


def test_undo_shows_the_face_sliders_without_announcing_them(panel: LiquifyPanel) -> None:
    seen: list[object] = []
    panel.face_value_set.connect(lambda *args: seen.append(args))
    panel.set_face_shape(liquify.FaceShape(chin=-30.0))
    assert panel.face_rows["chin"].value == pytest.approx(-30.0)
    assert panel.reset_button.isEnabled()
    panel.set_face_shape(liquify.DEFAULT_FACE_SHAPE)
    assert not panel.reset_button.isEnabled()
    assert not seen



# -- placing the face's points by hand (L-12) --------------------------------


def test_the_point_buttons_wait_for_a_face_and_for_a_point_moved(panel: LiquifyPanel) -> None:
    panel.set_faces(0)
    assert not panel.edit_faces_button.isEnabled()
    panel.set_faces(1)
    assert panel.edit_faces_button.isEnabled()
    assert not panel.reset_faces_button.isEnabled()
    assert panel.edit_faces_button.y() > panel.faces_caption.y()
    assert panel.edit_faces_button.width() == panel.reset_faces_button.width()
    marks = liquify.FaceMarks(points=((0.4, 0.4),) * 6, edited=True)
    panel.set_face_shape(liquify.FaceShape(faces=(marks,)))
    assert panel.reset_faces_button.isEnabled()


def test_editing_says_how_and_is_announced(panel: LiquifyPanel) -> None:
    seen: list[object] = []
    panel.editing_toggled.connect(seen.append)
    panel.face_marks_reset_requested.connect(lambda: seen.append("reset"))
    panel.set_faces(1)
    panel.edit_faces_button.click()
    assert panel.editing and seen == [True]
    assert panel.faces_caption.text() == tr("liquify.editing_hint")
    panel.set_editing(False)  # told, not announced
    assert not panel.editing and seen == [True]
    assert panel.faces_caption.text() == tr("adjust.faces.one")
    panel.set_faces_edited(True)
    panel.reset_faces_button.click()
    assert seen[-1] == "reset"
