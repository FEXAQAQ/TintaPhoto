"""Tests for the curves' editor and its group (PLAN.md 2.2 A-34, A-35, 3.1)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from app.i18n import tr  # noqa: E402
from core import tone_curve  # noqa: E402
from core.edit_state import CURVE_NAMES, LUMA_CURVE, RGB_CURVE, EditState  # noqa: E402
from core.tone_curve import STRAIGHT  # noqa: E402
from ui import theme  # noqa: E402
from ui.adjust_tab import AdjustTab  # noqa: E402
from ui.curve_editor import CURVE_CHOICES, CurveEditor, CurvePlot  # noqa: E402

S_CURVE: tone_curve.Curve = ((0.0, 0.0), (0.25, 0.18), (0.75, 0.84), (1.0, 1.0))
LEFT = Qt.MouseButton.LeftButton


@pytest.fixture
def editor(qt_app: object) -> Iterator[CurveEditor]:
    """Return a shown curve editor."""
    widget = CurveEditor()
    widget.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    yield widget
    widget.hide()
    widget.deleteLater()


@pytest.fixture
def tab(qt_app: object) -> Iterator[AdjustTab]:
    """Return a shown adjust tab holding an untouched JPEG."""
    widget = AdjustTab()
    widget.resize(theme.PANEL_WIDTH, 900)
    widget.set_photo(EditState.for_photo(False))
    widget.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    yield widget
    widget.hide()
    widget.deleteLater()


def _at(plot: CurvePlot, x: float, y: float) -> QPoint:
    """Return the pixel of the curve's (*x*, *y*)."""
    return plot.position_of(x, y).toPoint()


def _heard(editor: CurveEditor) -> list[tuple[tone_curve.Curve, bool]]:
    """Return a list that fills with what the editor announces."""
    heard: list[tuple[tone_curve.Curve, bool]] = []
    editor.curve_set.connect(lambda curve, dragging: heard.append((curve, dragging)))
    return heard


# -- the square ------------------------------------------------------------


def test_the_square_is_as_wide_as_a_group_s_controls(editor: CurveEditor) -> None:
    side = theme.PANEL_WIDTH - 2 * theme.PANEL_PADDING

    assert (editor.plot.width(), editor.plot.height()) == (side, side)
    assert side % 4 == 0 and theme.CURVE_READOUT_HEIGHT % 4 == 0


def test_it_starts_straight_and_says_how_to_begin(editor: CurveEditor) -> None:
    assert editor.curve == STRAIGHT
    assert editor.readout.text() == tr("curve.hint")


def test_a_press_on_empty_ground_adds_a_point_and_holds_it(editor: CurveEditor) -> None:
    heard = _heard(editor)

    QTest.mousePress(editor.plot, LEFT, pos=_at(editor.plot, 0.5, 0.7))

    assert len(editor.curve) == 3
    assert editor.curve[1] == pytest.approx((0.5, 0.7), abs=0.01)
    assert editor.plot.selected == 1 and editor.plot.dragging
    assert heard[-1][1] is True

    QTest.mouseRelease(editor.plot, LEFT, pos=_at(editor.plot, 0.5, 0.7))

    assert not editor.plot.dragging
    assert heard[-1] == (editor.curve, False)


def test_dragging_a_point_is_a_draft_until_it_is_let_go(editor: CurveEditor) -> None:
    editor.set_curve(S_CURVE)
    heard = _heard(editor)

    QTest.mousePress(editor.plot, LEFT, pos=_at(editor.plot, 0.25, 0.18))
    assert heard == []  # taking hold of a point changes nothing
    QTest.mouseMove(editor.plot, _at(editor.plot, 0.3, 0.4))
    QTest.mouseMove(editor.plot, _at(editor.plot, 0.35, 0.5))
    QTest.mouseRelease(editor.plot, LEFT, pos=_at(editor.plot, 0.35, 0.5))

    assert [dragging for _, dragging in heard] == [True, True, False]
    assert editor.curve[1] == pytest.approx((0.35, 0.5), abs=0.01)
    assert heard[-1][0] == editor.curve


def test_a_click_that_moves_nothing_announces_nothing(editor: CurveEditor) -> None:
    editor.set_curve(S_CURVE)
    heard = _heard(editor)

    QTest.mouseClick(editor.plot, LEFT, pos=_at(editor.plot, 0.75, 0.84))

    assert heard == []
    assert editor.plot.selected == 2


def test_an_end_only_moves_up_and_down(editor: CurveEditor) -> None:
    QTest.mousePress(editor.plot, LEFT, pos=_at(editor.plot, 0.0, 0.0))
    QTest.mouseMove(editor.plot, _at(editor.plot, 0.4, 0.3))
    QTest.mouseRelease(editor.plot, LEFT, pos=_at(editor.plot, 0.4, 0.3))

    assert len(editor.curve) == 2
    assert editor.curve[0] == (0.0, pytest.approx(0.3, abs=0.01))


def test_a_point_cannot_be_dragged_past_its_neighbour(editor: CurveEditor) -> None:
    editor.set_curve(S_CURVE)

    QTest.mousePress(editor.plot, LEFT, pos=_at(editor.plot, 0.25, 0.18))
    QTest.mouseMove(editor.plot, _at(editor.plot, 0.95, 0.5))
    QTest.mouseRelease(editor.plot, LEFT, pos=_at(editor.plot, 0.95, 0.5))

    assert editor.curve[1][0] == pytest.approx(0.75 - tone_curve.MIN_GAP)
    assert [x for x, _ in editor.curve] == sorted(x for x, _ in editor.curve)


def test_a_double_click_takes_a_point_away_but_not_an_end(editor: CurveEditor) -> None:
    editor.set_curve(S_CURVE)
    heard = _heard(editor)

    QTest.mouseDClick(editor.plot, LEFT, pos=_at(editor.plot, 0.25, 0.18))

    assert editor.curve == (S_CURVE[0], S_CURVE[2], S_CURVE[3])
    assert heard[-1] == (editor.curve, False)

    QTest.mouseDClick(editor.plot, LEFT, pos=_at(editor.plot, 1.0, 1.0))
    assert len(editor.curve) == 3


def test_delete_takes_the_chosen_point_away(editor: CurveEditor) -> None:
    editor.set_curve(S_CURVE)
    heard = _heard(editor)
    QTest.mouseClick(editor.plot, LEFT, pos=_at(editor.plot, 0.75, 0.84))

    QTest.keyClick(editor.plot, Qt.Key.Key_Delete)

    assert editor.curve == (S_CURVE[0], S_CURVE[1], S_CURVE[3])
    assert heard == [(editor.curve, False)]
    assert editor.plot.selected == -1

    QTest.keyClick(editor.plot, Qt.Key.Key_Delete)  # nothing chosen: nothing happens
    assert len(heard) == 1


def test_the_line_below_reads_the_chosen_point_in_levels(editor: CurveEditor) -> None:
    editor.set_curve(((0.0, 0.0), (0.2, 0.4), (1.0, 1.0)))

    QTest.mouseClick(editor.plot, LEFT, pos=_at(editor.plot, 0.2, 0.4))

    assert editor.readout.text() == tr("curve.readout", input=51, output=102)


def test_showing_a_curve_announces_nothing_and_lets_go_of_the_point(
    editor: CurveEditor,
) -> None:
    editor.set_curve(S_CURVE)
    QTest.mouseClick(editor.plot, LEFT, pos=_at(editor.plot, 0.25, 0.18))
    heard = _heard(editor)

    editor.set_curve(STRAIGHT)

    assert heard == []
    assert editor.plot.selected == -1
    assert editor.readout.text() == tr("curve.hint")


def test_the_histogram_behind_the_curve_comes_and_goes(editor: CurveEditor) -> None:
    assert not editor.plot.has_backdrop

    editor.set_backdrop(np.full((8, 8, 3), 90, dtype=np.uint8))
    assert editor.plot.has_backdrop
    editor.plot.grab()  # painting it must not fail

    editor.set_backdrop(None)
    assert not editor.plot.has_backdrop


def test_the_curve_is_drawn_where_it_runs(editor: CurveEditor) -> None:
    curve: tone_curve.Curve = ((0.0, 0.0), (0.5, 0.9), (1.0, 1.0))
    editor.set_curve(curve)
    height = float(tone_curve.evaluate(curve, np.array([0.2]))[0])

    image = editor.plot.grab().toImage()
    on_curve = _at(editor.plot, 0.2, height)
    off_curve = _at(editor.plot, 0.9, 0.1)

    assert image.pixelColor(on_curve).lightness() > image.pixelColor(off_curve).lightness()


# -- the group on the adjust tab -------------------------------------------


def test_the_curve_group_sits_after_light_and_starts_folded(tab: AdjustTab) -> None:
    titles = [group.header.text() for group in tab.groups()]

    assert titles.index(tr("adjust.group.curve")) == titles.index(tr("adjust.group.light")) + 1
    assert not tab.curve_group.is_open
    assert not tab.curve_editor.isVisible()

    tab.curve_group.set_open(True)
    assert tab.curve_editor.isVisible()


def test_drawing_on_the_tab_passes_the_curve_on(tab: AdjustTab) -> None:
    tab.curve_group.set_open(True)
    heard: list[tuple[tone_curve.Curve, bool]] = []
    tab.curve_set.connect(lambda curve, dragging: heard.append((curve, dragging)))
    plot = tab.curve_editor.plot

    QTest.mouseClick(plot, LEFT, pos=_at(plot, 0.5, 0.7))

    assert heard[-1][1] is False
    assert tab.state.luma_curve == heard[-1][0] != STRAIGHT


def test_the_reset_button_straightens_the_curve_and_then_rests(tab: AdjustTab) -> None:
    assert not tab.curve_reset_button.isEnabled()
    assert tab.curve_reset_button.toolTip() == tr("adjust.reset_curve")
    tab.set_state(tab.state.with_curve(S_CURVE))
    assert tab.curve_reset_button.isEnabled()
    heard: list[tuple[tone_curve.Curve, bool]] = []
    tab.curve_set.connect(lambda curve, dragging: heard.append((curve, dragging)))

    tab.curve_reset_button.click()

    assert heard == [(STRAIGHT, False)]
    assert tab.curve_editor.curve == STRAIGHT
    assert not tab.curve_reset_button.isEnabled()


def test_the_tab_shows_the_curve_a_photo_arrives_with(tab: AdjustTab) -> None:
    tab.set_photo(EditState.for_photo(False).with_curve(S_CURVE))

    assert tab.curve_editor.curve == S_CURVE
    assert tab.curve_reset_button.isEnabled()

    tab.set_state(EditState.for_photo(False))
    assert tab.curve_editor.curve == STRAIGHT


# -- the choice of curve (A-35) --------------------------------------------


def test_the_row_offers_the_five_curves_and_starts_on_brightness(editor: CurveEditor) -> None:
    names = [button.text() for button in editor.channels.buttons().values()]

    assert names == [tr(key) for _name, key in CURVE_CHOICES]
    assert list(editor.channels.buttons()) == list(CURVE_NAMES)
    assert editor.channel == LUMA_CURVE
    assert editor.channels.value == LUMA_CURVE


def test_choosing_a_curve_shows_its_points_and_announces_only_the_choice(
    editor: CurveEditor,
) -> None:
    state = EditState.for_photo(False).with_curve(S_CURVE, "red_curve")
    editor.set_curves(state)
    heard = _heard(editor)
    chosen: list[str] = []
    editor.channel_changed.connect(chosen.append)

    editor.channels.buttons()["red_curve"].click()

    assert chosen == ["red_curve"]
    assert editor.channel == "red_curve"
    assert editor.curve == S_CURVE
    assert heard == []
    editor.channels.buttons()[LUMA_CURVE].click()
    assert editor.curve == STRAIGHT


def test_a_point_drawn_is_announced_with_the_name_of_its_curve(editor: CurveEditor) -> None:
    named: list[str] = []
    editor.curve_set.connect(lambda _curve, _dragging, name: named.append(name))
    editor.set_channel("blue_curve")

    QTest.mouseClick(editor.plot, LEFT, pos=_at(editor.plot, 0.5, 0.7))

    assert named and set(named) == {"blue_curve"}
    assert editor.curve_of("blue_curve") != STRAIGHT
    assert editor.curve_of(LUMA_CURVE) == STRAIGHT


def test_a_bent_curve_carries_a_dot_after_its_name(editor: CurveEditor) -> None:
    editor.set_curves(EditState.for_photo(False).with_curve(S_CURVE, "green_curve"))

    marked = [name for name in CURVE_NAMES if editor.channels.is_marked(name)]
    assert marked == ["green_curve"]
    editor.channels.grab()  # painting the dot must not fail

    editor.set_channel(RGB_CURVE)
    QTest.mouseClick(editor.plot, LEFT, pos=_at(editor.plot, 0.5, 0.7))
    assert editor.channels.is_marked(RGB_CURVE)
    editor.set_curves(EditState.for_photo(False))
    assert not any(editor.channels.is_marked(name) for name in CURVE_NAMES)


def test_a_channel_s_curve_is_drawn_in_its_colour(editor: CurveEditor) -> None:
    assert editor.plot.curve_colour() == theme.TEXT_PRIMARY
    editor.set_channel(RGB_CURVE)
    assert editor.plot.curve_colour() == theme.TEXT_PRIMARY
    editor.set_channel("red_curve")
    assert editor.plot.curve_colour() == theme.HISTOGRAM_RED
    editor.set_backdrop(np.full((8, 8, 3), 90, dtype=np.uint8))
    for name in CURVE_NAMES:
        editor.set_channel(name)
        editor.plot.grab()  # each histogram behind the curve must paint


def test_the_reset_button_straightens_only_the_shown_curve(tab: AdjustTab) -> None:
    tab.curve_group.set_open(True)
    tab.set_state(tab.state.with_curve(S_CURVE).with_curve(S_CURVE, "blue_curve"))
    heard: list[tuple[tone_curve.Curve, bool, str]] = []
    tab.curve_set.connect(lambda curve, dragging, name: heard.append((curve, dragging, name)))
    tab.curve_editor.set_channel(RGB_CURVE)
    assert not tab.curve_reset_button.isEnabled()

    tab.curve_editor.set_channel("blue_curve")
    assert tab.curve_reset_button.isEnabled()
    tab.curve_reset_button.click()

    assert heard == [(STRAIGHT, False, "blue_curve")]
    assert tab.state.blue_curve == STRAIGHT
    assert tab.state.luma_curve == S_CURVE
    assert not tab.curve_editor.channels.is_marked("blue_curve")
    assert tab.curve_editor.channels.is_marked(LUMA_CURVE)


def test_the_tab_says_when_another_curve_is_chosen(tab: AdjustTab) -> None:
    chosen: list[str] = []
    tab.curve_channel_changed.connect(chosen.append)

    tab.curve_editor.channels.buttons()["green_curve"].click()

    assert chosen == ["green_curve"]
    assert tab.state.curves_are_straight
