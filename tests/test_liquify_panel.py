"""Tests for the liquify panel (PLAN.md 2.11, 3.16)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from core import liquify

pytest.importorskip("PySide6.QtWidgets")

from app.i18n import tr  # noqa: E402
from ui import theme  # noqa: E402
from ui.liquify_panel import (  # noqa: E402
    FACE_GROUPS,
    PAGE_BRUSH,
    PAGE_FACE,
    TOOL_ROWS,
    LiquifyPanel,
)


@pytest.fixture
def panel(qt_app: object) -> Iterator[LiquifyPanel]:
    widget = LiquifyPanel()
    widget.resize(theme.PANEL_WIDTH, 800)
    widget.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    yield widget
    widget.hide()
    widget.deleteLater()


def _brush(panel: LiquifyPanel, qt_app: object) -> None:
    panel.page_switch.buttons()[PAGE_BRUSH].click()
    qt_app.processEvents()  # type: ignore[attr-defined]


def test_it_is_as_wide_as_the_inspector(panel: LiquifyPanel) -> None:
    assert panel.width() == theme.PANEL_WIDTH


# -- the two pages -----------------------------------------------------------


def test_one_page_at_a_time_and_the_face_comes_first(panel: LiquifyPanel, qt_app: object) -> None:
    pages: list[str] = []
    panel.page_changed.connect(pages.append)
    assert panel.page == PAGE_FACE
    assert panel.face_page.isVisible() and not panel.brush_page.isVisible()
    _brush(panel, qt_app)
    assert pages == [PAGE_BRUSH] and panel.page == PAGE_BRUSH
    assert panel.brush_page.isVisible() and not panel.face_page.isVisible()
    panel.set_page(PAGE_FACE)  # told, not announced
    assert panel.face_page.isVisible() and pages == [PAGE_BRUSH]
    assert [button.text() for button in panel.page_switch.buttons().values()] == [
        tr("liquify.page.face"),
        tr("liquify.page.brush"),
    ]


# -- the face page -----------------------------------------------------------


def test_the_face_sliders_come_in_two_groups_shape_first(panel: LiquifyPanel) -> None:
    panel.set_faces(1)
    titles = [group.header.text() for group in panel.groups()][:2]
    assert titles == [tr("liquify.group.shape"), tr("liquify.group.features")]
    order = [name for _title, names in FACE_GROUPS for name in names]
    assert order == ["face_width", "chin", "eyes", "nose", "mouth"]
    assert sorted(order) == sorted(liquify.FACE_NAMES)
    tops = [panel.face_rows[name].mapTo(panel, panel.face_rows[name].rect().topLeft()).y() for name in order]
    assert tops == sorted(tops)


def test_every_face_slider_says_what_its_ends_do(panel: LiquifyPanel) -> None:
    panel.set_faces(1)
    for name, row in panel.face_rows.items():
        assert row.spec.minimum == -100.0 and row.spec.maximum == 100.0 and row.value == 0.0
        ends = panel.face_ends[name]
        assert ends.isVisible() and ends.texts == (
            tr(f"liquify.face.{name}.low"),
            tr(f"liquify.face.{name}.high"),
        )
        assert ends.y() > row.y()
    assert panel.face_ends["face_width"].texts == (tr("liquify.face.face_width.low"), tr("liquify.face.face_width.high"))


def test_the_face_page_waits_for_a_face_then_offers_the_brush_without_one(
    panel: LiquifyPanel, qt_app: object
) -> None:
    pages: list[str] = []
    panel.page_changed.connect(pages.append)
    panel.set_faces(None)
    assert panel.faces_caption.text() == tr("adjust.faces.looking")
    assert not panel.face_rows["eyes"].isEnabled()
    panel.set_faces(2)
    assert panel.faces_caption.text() == tr("adjust.faces.many", count=2)
    assert panel.face_rows["eyes"].isEnabled() and panel.edit_faces_link.isEnabled()

    panel.set_faces(0)
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert not panel.face_rows["eyes"].isVisible()
    assert panel.no_face_title.isVisible()
    assert panel.no_face_title.text() == tr("liquify.faces.none")
    panel.use_brush_button.click()
    assert pages == [PAGE_BRUSH] and panel.page == PAGE_BRUSH

    # Faces kept with the sliders can still be shaped before a new search.
    marks = liquify.FaceMarks(points=((0.4, 0.4),) * 6)
    panel.set_page(PAGE_FACE)
    panel.set_face_shape(liquify.FaceShape(eyes=10.0, faces=(marks,)))
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert panel.face_rows["eyes"].isVisible() and panel.face_rows["eyes"].isEnabled()


def test_a_face_slider_says_what_changed_and_where_the_pointer_is(
    panel: LiquifyPanel, qt_app: object
) -> None:
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QEnterEvent
    from PySide6.QtWidgets import QApplication

    seen: list[tuple[str, float, bool]] = []
    hovered: list[str] = []
    panel.face_value_set.connect(lambda name, value, dragging: seen.append((name, value, dragging)))
    panel.face_hovered.connect(hovered.append)
    panel.set_faces(1)
    panel.face_rows["chin"].set_value(40.0, announce=True)
    assert seen[-1] == ("chin", pytest.approx(40.0), False)
    assert panel.reset_face_link.isEnabled() and panel.reset_button.isEnabled()

    row = panel.face_rows["chin"]
    QApplication.sendEvent(row, QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
    QApplication.sendEvent(row, QEvent(QEvent.Type.Leave))
    assert hovered == ["chin", ""]


def test_the_face_can_be_reset_without_the_strokes(panel: LiquifyPanel) -> None:
    asked: list[str] = []
    panel.face_reset_requested.connect(lambda: asked.append("face"))
    panel.set_faces(1)
    assert not panel.reset_face_link.isEnabled()
    panel.set_face_shape(liquify.FaceShape(chin=-30.0))
    assert panel.face_rows["chin"].value == pytest.approx(-30.0)
    assert panel.reset_face_link.isEnabled()
    panel.reset_face_link.click()
    assert asked == ["face"]
    assert panel.every_face_caption.text() == tr("liquify.every_face")


def test_placing_the_points_has_a_page_of_its_own(panel: LiquifyPanel, qt_app: object) -> None:
    seen: list[object] = []
    panel.editing_toggled.connect(seen.append)
    panel.face_marks_reset_requested.connect(lambda: seen.append("reset"))
    panel.set_faces(1)
    assert panel.edit_faces_link.text() == tr("liquify.edit_faces.link")
    panel.edit_faces_link.click()
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert panel.editing and seen == [True]
    assert panel.editing_caption.isVisible()
    assert panel.editing_caption.text() == tr("liquify.editing_hint")
    assert not panel.face_rows["eyes"].isVisible()
    assert not panel.reset_faces_button.isEnabled()
    panel.set_faces_edited(True)
    panel.reset_faces_button.click()
    assert seen[-1] == "reset"
    panel.finish_editing_button.click()
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert not panel.editing and seen[-1] is False
    assert panel.face_rows["eyes"].isVisible()
    panel.set_editing(True)  # told, not announced
    assert seen[-1] is False


# -- the brush page ----------------------------------------------------------


def test_six_tools_in_two_rows_of_three(panel: LiquifyPanel, qt_app: object) -> None:
    _brush(panel, qt_app)
    assert [len(row) for row in TOOL_ROWS] == [3, 3]
    assert set(panel.tool_buttons) == set(liquify.TOOLS) - {liquify.TOOL_TWIRL_CCW}
    assert panel.tool_buttons[liquify.TOOL_TWIRL_CW].text() == tr("liquify.tool.twirl")
    for row in TOOL_ROWS:
        tops = {panel.tool_buttons[tool].y() for tool in row}
        widths = {panel.tool_buttons[tool].width() for tool in row}
        assert len(tops) == 1 and max(widths) - min(widths) <= 1
    first, second = TOOL_ROWS
    gap = panel.tool_buttons[second[0]].y() - panel.tool_buttons[first[0]].y() - theme.BUTTON_HEIGHT
    assert gap == theme.SPACE_SM
    for button in panel.tool_buttons.values():
        assert button.height() == theme.BUTTON_HEIGHT and button.toolTip()


def test_the_chosen_tool_says_what_it_does(panel: LiquifyPanel, qt_app: object) -> None:
    _brush(panel, qt_app)
    chosen: list[str] = []
    panel.tool_changed.connect(chosen.append)
    assert panel.tool == liquify.TOOL_PUSH
    assert panel.tool_caption.text() == tr("liquify.tool.push.tip")
    panel.tool_buttons[liquify.TOOL_TWIRL_CW].click()
    assert chosen == [liquify.TOOL_TWIRL_CW]
    assert panel.tool_caption.text() == tr("liquify.tool.twirl.tip")
    assert "Alt" in panel.tool_caption.text()
    panel.tool_buttons[liquify.TOOL_TWIRL_CW].click()  # stays pressed
    assert panel.tool == liquify.TOOL_TWIRL_CW
    assert panel.brush_hint.text() == tr("liquify.hint")


def test_the_size_counts_thousandths_and_the_strength_hundredths(panel: LiquifyPanel) -> None:
    sizes: list[tuple[float, bool]] = []
    panel.brush_size_set.connect(lambda radius, dragging: sizes.append((radius, dragging)))
    assert panel.brush_size_row.value == pytest.approx(50.0)
    panel.brush_size_row.set_value(120.0, announce=True)
    assert sizes[-1] == (pytest.approx(0.12), False)
    panel.set_brush_size(9.0)
    assert panel.brush_size == liquify.RADIUS_MAX
    assert panel.strength_row.value == pytest.approx(50.0)
    panel.strength_row.set_value(80.0, announce=True)
    assert panel.strength == pytest.approx(0.8)
    panel.set_strength(0.0)
    assert panel.strength == liquify.STRENGTH_MIN


def test_it_says_how_many_strokes(panel: LiquifyPanel) -> None:
    assert panel.strokes_caption.text() == tr("liquify.strokes.none")
    assert not panel.reset_button.isEnabled()
    panel.set_strokes(1)
    assert panel.strokes_caption.text() == tr("liquify.strokes.one")
    panel.set_strokes(4)
    assert panel.strokes_caption.text() == tr("liquify.strokes.many", count=4)
    assert panel.reset_button.isEnabled()


# -- along the bottom, and languages ------------------------------------------


def test_the_buttons_along_the_bottom_ask_for_what_they_say(panel: LiquifyPanel) -> None:
    asked: list[str] = []
    panel.reset_requested.connect(lambda: asked.append("reset"))
    panel.leave_requested.connect(lambda: asked.append("leave"))
    panel.set_strokes(2)
    panel.reset_button.click()
    panel.leave_button.click()
    assert asked == ["reset", "leave"]
    assert tr("shortcut.liquify") in panel.leave_button.toolTip()


def test_a_language_change_keeps_what_was_chosen(panel: LiquifyPanel, qt_app: object) -> None:
    _brush(panel, qt_app)
    panel.tool_buttons[liquify.TOOL_SMOOTH].click()
    panel.set_brush_size(0.1)
    panel.set_strength(0.3)
    panel.set_face_shape(liquify.FaceShape(nose=20.0))
    panel.retranslate()
    assert panel.page == PAGE_BRUSH
    assert panel.tool == liquify.TOOL_SMOOTH
    assert panel.brush_size == pytest.approx(0.1)
    assert panel.strength == pytest.approx(0.3)
    assert panel.face_rows["nose"].value == pytest.approx(20.0)
    assert panel.tool_caption.text() == tr("liquify.tool.smooth.tip")
