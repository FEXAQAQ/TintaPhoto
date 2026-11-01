"""Tests for the slider row and the adjust tab (PLAN.md 2.2, 3.6)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from core import color_management as cm
from core import edit_state
from core.edit_state import EXPOSURE_STEP_EV, EditState

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from app.i18n import tr  # noqa: E402
from ui import theme  # noqa: E402
from ui.adjust_tab import AdjustTab, raw_specs  # noqa: E402
from ui.widgets.slider_row import FINE_FACTOR, SliderRow, SliderSpec  # noqa: E402

WIDTH = 300
SHOT = (5234.0, 7.6)


def _spec(**overrides) -> SliderSpec:
    base = {
        "name": "contrast",
        "label_key": "adjust.contrast",
        "minimum": -100.0,
        "maximum": 100.0,
        "default": 0.0,
    }
    base.update(overrides)
    return SliderSpec(**base)


@pytest.fixture
def row(qt_app: object) -> Iterator[SliderRow]:
    widget = SliderRow(_spec())
    widget.resize(WIDTH, 28)
    widget.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    yield widget
    widget.hide()
    widget.deleteLater()


def _click(widget: SliderRow, x: int, button=Qt.MouseButton.LeftButton, double: bool = False):
    kind = QMouseEvent.Type.MouseButtonDblClick if double else QMouseEvent.Type.MouseButtonPress
    position = QPointF(x, widget.height() / 2)
    event = QMouseEvent(
        kind, position, widget.mapToGlobal(position), button, button, Qt.KeyboardModifier.NoModifier
    )
    QApplication.sendEvent(widget, event)


# -- the value ------------------------------------------------------------


def test_the_value_stays_inside_the_range(row: SliderRow) -> None:
    row.set_value(500.0)
    assert row.value == 100.0
    row.set_value(-500.0)
    assert row.value == -100.0


def test_setting_a_value_is_silent_unless_asked(row: SliderRow) -> None:
    seen: list[tuple] = []
    row.value_changed.connect(lambda *args: seen.append(args))
    row.set_value(30.0)
    assert seen == []
    row.set_value(40.0, announce=True)
    assert seen == [("contrast", 40.0, False)]


def test_resetting_goes_back_to_where_it_rests(row: SliderRow) -> None:
    row.set_value(70.0)
    row.reset()
    assert row.value == 0.0


# -- how the number reads -------------------------------------------------


def test_a_bipolar_slider_shows_its_sign(row: SliderRow) -> None:
    row.set_value(25.0)
    assert row._value_label.text() == "+25"
    row.set_value(-25.0)
    assert row._value_label.text() == "−25"  # a real minus sign


def test_a_temperature_shows_no_sign_and_its_unit(qt_app: object) -> None:
    widget = SliderRow(raw_specs(*SHOT)["temperature"])
    widget.set_value(5600.0)
    assert widget._value_label.text() == "5600 K"
    widget.deleteLater()


def test_exposure_shows_two_decimals(qt_app: object) -> None:
    widget = SliderRow(_spec(name="exposure", minimum=-5.0, maximum=5.0, decimals=2))
    widget.set_value(0.35)
    assert widget._value_label.text() == "+0.35"
    widget.deleteLater()


# -- where the handle goes ------------------------------------------------


def test_a_straight_track_is_evenly_spaced(row: SliderRow) -> None:
    assert row._to_position(-100.0) == pytest.approx(0.0)
    assert row._to_position(0.0) == pytest.approx(0.5)
    assert row._to_position(100.0) == pytest.approx(1.0)


def test_the_temperature_track_puts_the_warm_end_on_the_right(qt_app: object) -> None:
    """A higher setting warms the photo, so it belongs to the right."""
    widget = SliderRow(raw_specs(*SHOT)["temperature"])
    assert widget._to_position(2000.0) == pytest.approx(0.0, abs=1e-6)
    assert widget._to_position(50000.0) == pytest.approx(1.0, abs=1e-6)
    assert widget._to_position(5000.0) < widget._to_position(9000.0)
    widget.deleteLater()


def test_the_temperature_track_is_spaced_by_reciprocal(qt_app: object) -> None:
    """Equal steps must feel equal, which kelvin itself does not give."""
    widget = SliderRow(raw_specs(*SHOT)["temperature"])
    low = widget._to_position(3000.0) - widget._to_position(2500.0)
    high = widget._to_position(10000.0) - widget._to_position(9500.0)
    assert low > high * 10  # 500 K matters far more down at 2500 K
    widget.deleteLater()


@pytest.mark.parametrize("kelvin", [2000.0, 3200.0, 5600.0, 12000.0, 50000.0])
def test_the_temperature_track_round_trips(qt_app: object, kelvin: float) -> None:
    widget = SliderRow(raw_specs(*SHOT)["temperature"])
    assert widget._from_position(widget._to_position(kelvin)) == pytest.approx(kelvin, rel=1e-6)
    widget.deleteLater()


# -- the mouse and the keyboard -------------------------------------------


def test_clicking_the_track_moves_the_slider(row: SliderRow) -> None:
    seen: list[tuple] = []
    row.value_changed.connect(lambda *args: seen.append(args))
    _click(row, WIDTH - 60)
    assert row.value > 0
    assert seen and seen[-1][2] is True  # still dragging


def test_double_clicking_the_name_resets(row: SliderRow) -> None:
    row.set_value(60.0)
    _click(row, 10, double=True)
    assert row.value == 0.0


def test_the_wheel_leaves_the_value_alone(row: SliderRow) -> None:
    """Scrolling the panel must not change a photo (PLAN.md 3.6)."""
    row.set_value(40.0)
    event = QWheelEvent(
        QPointF(WIDTH / 2, 14),
        QPointF(row.mapToGlobal(QPoint(int(WIDTH / 2), 14))),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(row, event)
    assert row.value == 40.0


def test_shift_makes_a_drag_ten_times_finer(row: SliderRow) -> None:
    row.set_value(0.0)
    row._dragging = True
    row._move_to(WIDTH - 40, fine=False)
    coarse = row.value

    row.set_value(0.0)
    row._move_to(WIDTH - 40, fine=True)
    assert row.value == pytest.approx(coarse * FINE_FACTOR, rel=1e-6)


# -- the exposure slider's step (PLAN.md 2.2, A-03) -----------------------


def _exposure(qt_app: object) -> SliderRow:
    widget = SliderRow(raw_specs(*SHOT)["exposure"])
    widget.resize(WIDTH, 28)
    return widget


def test_the_exposure_slider_moves_in_twentieths_of_a_stop(qt_app: object) -> None:
    """PLAN.md 2.2 gives A-03 a step of 0.05 EV."""
    widget = _exposure(qt_app)
    assert widget.spec.step == pytest.approx(EXPOSURE_STEP_EV)

    for asked, shown in ((0.37, 0.35), (0.38, 0.40), (-1.23, -1.25), (2.0, 2.0)):
        widget.set_value(asked, announce=True)
        assert widget.value == pytest.approx(shown), asked
    widget.deleteLater()


def test_dragging_the_exposure_slider_lands_on_a_step(qt_app: object) -> None:
    widget = _exposure(qt_app)
    widget.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    seen: list[float] = []
    widget.value_changed.connect(lambda _name, value, _dragging: seen.append(value))
    widget._dragging = True
    for x in range(widget.width() // 2, widget.width()):
        widget._move_to(x, fine=False)
    assert seen, "the drag moved the slider"
    for value in seen:
        assert value == pytest.approx(round(value / EXPOSURE_STEP_EV) * EXPOSURE_STEP_EV)
    widget.hide()
    widget.deleteLater()


def test_an_arrow_key_moves_the_exposure_slider_one_step(qt_app: object) -> None:
    """The arrows are section 8's fine adjustment, and one step is as fine as
    this slider gets - Shift has nothing smaller to offer."""
    from PySide6.QtGui import QKeyEvent

    widget = _exposure(qt_app)
    widget.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    for modifier in (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ShiftModifier):
        widget.set_value(0.0, announce=True)
        QApplication.sendEvent(
            widget, QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Right, modifier)
        )
        assert widget.value == pytest.approx(EXPOSURE_STEP_EV), modifier
        QApplication.sendEvent(
            widget, QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Left, modifier)
        )
        assert widget.value == pytest.approx(0.0, abs=1e-9), modifier
    widget.hide()
    widget.deleteLater()


def test_a_stored_exposure_between_two_steps_is_shown_as_it_is(qt_app: object) -> None:
    """A parameter file written before the step existed must not be rounded
    on screen while the photo is still developed with what it really says."""
    widget = _exposure(qt_app)
    widget.set_value(0.37)
    assert widget.value == pytest.approx(0.37)
    assert widget._value_label.text() == "+0.37"
    widget.deleteLater()


def test_the_other_sliders_still_run_free(row: SliderRow) -> None:
    assert row.spec.step == 0.0
    row.set_value(12.5, announce=True)
    assert row.value == pytest.approx(12.5)


def test_typing_a_number_sets_it(row: SliderRow) -> None:
    row._start_editing()
    row._editor.setText("-42")
    row._finish_editing()
    assert row.value == -42.0


def test_typing_nonsense_leaves_the_value_alone(row: SliderRow) -> None:
    row.set_value(15.0)
    row._start_editing()
    row._editor.setText("not a number")
    row._finish_editing()
    assert row.value == 15.0


# -- the adjust tab -------------------------------------------------------


@pytest.fixture
def tab(qt_app: object) -> Iterator[AdjustTab]:
    widget = AdjustTab()
    widget.resize(300, 400)
    widget.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    yield widget
    widget.hide()
    widget.deleteLater()


def test_a_raw_photo_gets_a_temperature_in_kelvin(tab: AdjustTab) -> None:
    tab.set_photo(EditState.for_photo(True, *SHOT), SHOT)
    row = tab.rows()["temperature"]
    assert row.spec.minimum == 2000.0
    assert row.spec.maximum == 50000.0
    assert row.value == pytest.approx(SHOT[0])
    assert "K" in row._value_label.text()


def test_a_jpeg_gets_a_relative_temperature(tab: AdjustTab) -> None:
    tab.set_photo(EditState(is_raw=False))
    row = tab.rows()["temperature"]
    assert (row.spec.minimum, row.spec.maximum) == (-100.0, 100.0)
    assert row.value == 0.0


def test_the_sliders_rest_where_the_camera_was(tab: AdjustTab) -> None:
    """A RAW photo opens on what the camera recorded, not on zero."""
    tab.set_photo(EditState.for_photo(True, *SHOT), SHOT)
    assert tab.rows()["temperature"].spec.default == pytest.approx(SHOT[0])
    assert tab.rows()["tint"].spec.default == pytest.approx(SHOT[1])


def test_moving_a_slider_says_what_changed(tab: AdjustTab) -> None:
    tab.set_photo(EditState.for_photo(True, *SHOT), SHOT)
    seen: list[tuple] = []
    tab.value_set.connect(lambda name, value, dragging: seen.append((name, value, dragging)))
    tab.rows()["exposure"].set_value(1.5, announce=True)
    assert seen
    name, value, dragging = seen[-1]
    assert name == "exposure"
    assert value == pytest.approx(1.5)
    assert dragging is False


def test_a_relative_temperature_is_sent_as_kelvin(tab: AdjustTab) -> None:
    tab.set_temperature_unit(False)
    tab.set_photo(EditState.for_photo(True, *SHOT), SHOT)
    seen: list[tuple] = []
    tab.value_set.connect(lambda name, value, dragging: seen.append((name, value)))
    tab.rows()["temperature"].set_value(20.0, announce=True)
    assert seen[-1][0] == "temperature"
    assert seen[-1][1] == pytest.approx(
        cm.relative_to_kelvin(20.0, SHOT[0], edit_state.TEMPERATURE_MIN_K, edit_state.TEMPERATURE_MAX_K)
    )


def test_every_adjustment_but_the_portrait_has_a_slider(tab: AdjustTab) -> None:
    """The portrait sliders are on the retouch tab (``tests/test_retouch_tab.py``)."""
    from core.edit_state import ADJUSTMENT_NAMES

    assert set(tab.rows()) == set(ADJUSTMENT_NAMES) - {"skin", "whiten"}


# -- the three collapsible groups (PLAN.md 3.1) ---------------------------


def _settle_fold(group) -> None:
    """Let a group finish folding, the way the eye waits for it.

    A group opens and closes over the 160 ms of PLAN.md 3.6, so what is on
    screen is only settled once the animation has run out.
    """
    import time

    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + 2.0
    while group.folding and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)


def test_every_group_has_a_header_that_names_it(tab: AdjustTab) -> None:
    from ui.adjust_tab import GROUPS

    headers = [group.header.text() for group in tab.groups()]
    assert headers == (
        [tr("adjust.group.presets")]
        + [tr(title_key) for title_key, _names in GROUPS]
    )


def test_closing_a_group_folds_its_sliders_away(tab: AdjustTab) -> None:
    """PLAN.md 3.1: 調整分頁的群組可收合."""
    tab.set_photo(EditState.for_photo(False), SHOT)
    colour = next(
        group
        for group in tab.groups()
        if group.header.text() == tr("adjust.group.colour")
    )
    saturation = tab.rows()["saturation"]
    exposure = tab.rows()["exposure"]
    assert saturation.isVisibleTo(tab)

    colour.header.click()
    _settle_fold(colour)
    assert not colour.is_open
    assert not saturation.isVisibleTo(tab)
    # Only that group folds away; the others stay as they were.
    assert exposure.isVisibleTo(tab)

    colour.header.click()
    _settle_fold(colour)
    assert colour.is_open
    assert saturation.isVisibleTo(tab)


def test_the_arrow_and_the_tooltip_say_which_way_the_group_goes(
    tab: AdjustTab,
) -> None:
    header = tab.groups()[0].header
    assert header.toolTip() == tr("group.collapse")
    opened = header.icon().cacheKey()

    header.click()
    assert header.toolTip() == tr("group.expand")
    assert header.icon().cacheKey() != opened


def test_a_group_title_is_heavier_than_the_names_under_it(tab: AdjustTab) -> None:
    """The complaint that started this: a title fainter than its own
    contents reads as a remark rather than as a heading."""
    qss = theme.build_qss()
    rule = qss.split('QPushButton[role="group"] {')[1].split("}")[0]
    assert f"font-weight: {theme.FONT_WEIGHT_TITLE}" in rule
    assert f"color: {theme.TEXT_PRIMARY}" in rule
    # The names of the sliders are the quieter of the two.
    assert tab.rows()["saturation"].findChildren(QLabel)[0].property("role") == "secondary"


def test_a_line_separates_one_group_from_the_next(tab: AdjustTab) -> None:
    from ui.widgets.group import Divider

    dividers = tab.findChildren(Divider)
    # One between each pair of groups, and none above the first, where the
    # tab bar's own underline is already the edge.
    assert len(dividers) == len(tab.groups()) - 1
    assert all(divider.height() == theme.BORDER_WIDTH for divider in dividers)


def test_dragging_a_slider_fades_the_group_titles(tab: AdjustTab) -> None:
    """PLAN.md 3.6: 拖曳時其他群組標題淡出."""
    tab.set_photo(EditState.for_photo(False), SHOT)
    header = tab.groups()[0].header
    assert header.property("faded") in (None, "false")

    tab._on_row_changed("exposure", 0.5, True)
    assert header.property("faded") == "true"
    tab._on_row_changed("exposure", 0.5, False)
    assert header.property("faded") == "false"


def test_a_group_folds_over_the_documented_time(tab: AdjustTab) -> None:
    """PLAN.md 3.6 gives a panel 160 ms to collapse; a group folds at the
    same pace rather than blinking shut."""
    from ui import native

    if not native.animations_enabled():
        pytest.skip("Windows is set to show no animation")

    colour = tab.groups()[-1]
    colour.header.click()
    assert colour.folding
    _settle_fold(colour)
    assert not colour.folding
    assert not tab.rows()["saturation"].isVisibleTo(tab)


# -- the focus ring (PLAN.md 3.6) -----------------------------------------


def _focus(row: SliderRow, reason: Qt.FocusReason) -> None:
    """Give *row* the focus the way *reason* says it arrived.

    Sent by hand because the offscreen platform never activates the window,
    so Qt delivers no focus events of its own.
    """
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QFocusEvent

    QApplication.sendEvent(row, QFocusEvent(QEvent.Type.FocusIn, reason))


def _ring_shown(row: SliderRow) -> bool:
    """Return whether the accent focus ring is on screen.

    Sampled along the top edge, where nothing but the ring is ever drawn.
    The line is antialiased onto a transparent background, so it arrives at
    half strength rather than exactly the accent colour.
    """
    from PySide6.QtGui import QColor

    from ui import theme

    pixel = row.grab().toImage().pixelColor(row.width() // 2, 0)
    accent = QColor(theme.ACCENT)
    if not pixel.alpha():
        return False
    return all(
        abs(drawn - wanted) <= 8
        for drawn, wanted in zip(
            (pixel.red(), pixel.green(), pixel.blue()),
            (accent.red(), accent.green(), accent.blue()),
        )
    )


def test_dragging_a_slider_does_not_ring_it(row: SliderRow) -> None:
    """PLAN.md 3.6: 鍵盤焦點顯示 1 px accent 色外框，滑鼠操作時不顯示."""
    _click(row, row.width() // 2)
    assert row.hasFocus()
    _focus(row, Qt.FocusReason.MouseFocusReason)
    assert not _ring_shown(row)


def test_reaching_the_slider_by_keyboard_rings_it(row: SliderRow) -> None:
    _focus(row, Qt.FocusReason.TabFocusReason)
    assert _ring_shown(row)


def test_nudging_with_an_arrow_key_rings_a_slider_clicked_with_the_mouse(
    row: SliderRow,
) -> None:
    """Once the keyboard is in use the ring says which slider will answer."""
    from PySide6.QtGui import QKeyEvent

    _click(row, row.width() // 2)
    _focus(row, Qt.FocusReason.MouseFocusReason)
    assert not _ring_shown(row)

    QApplication.sendEvent(
        row,
        QKeyEvent(
            QKeyEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier
        ),
    )
    assert _ring_shown(row)


# -- the detail group (PLAN.md 2.2, 3.1) -----------------------------------


def test_the_detail_group_comes_last_with_its_four_sliders(tab: AdjustTab) -> None:
    titles = [group.header.text() for group in tab.groups()]

    assert titles[-1] == tr("adjust.group.detail")
    names = list(tab.rows())
    assert names[-4:] == ["sharpening", "sharpen_masking", "luminance_noise", "colour_noise"]


def test_a_raw_photo_rests_its_detail_where_the_photo_does(tab: AdjustTab) -> None:
    tab.set_photo(EditState.for_photo(True, *SHOT), SHOT)
    rows = tab.rows()

    assert rows["sharpening"].spec.default == edit_state.RAW_SHARPENING_DEFAULT
    assert rows["sharpening"].value == pytest.approx(edit_state.RAW_SHARPENING_DEFAULT)
    assert rows["colour_noise"].spec.default == edit_state.RAW_COLOUR_NOISE_DEFAULT
    assert rows["sharpening"].spec.maximum == edit_state.SHARPENING_MAX


def test_a_jpeg_rests_its_detail_at_zero(tab: AdjustTab) -> None:
    tab.set_photo(EditState(is_raw=False))
    rows = tab.rows()

    for name in edit_state.DETAIL_NAMES:
        assert rows[name].spec.default == 0.0
        assert rows[name].value == 0.0
        assert not rows[name].spec.signed
