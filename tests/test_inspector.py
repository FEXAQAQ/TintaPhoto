"""Tests for the right hand panel's frame (PLAN.md 3.1 and 3.3).

With every group open the adjust tab is taller than the panel, and before
the tab pages scrolled Qt cut the last rows off: the sliders are a fixed
height, so a page with too little room for them loses them altogether
rather than squeezing them. PLAN.md 3.3 has the wheel scrolling over the
panel, and 3.4 has the slider ignoring it so that scrolling cannot nudge a
photo.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from core.edit_state import EditState
from core.geometry import Geometry

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QScrollBar  # noqa: E402

from app.i18n import tr  # noqa: E402
from ui import theme  # noqa: E402
from ui.inspector import Inspector, InspectorTab  # noqa: E402

#: A panel this short cannot show the adjust tab's four groups at once;
#: it is the height the panel gets in a window at ``WINDOW_MIN_HEIGHT``.
SHORT: int = 400

#: Where to look for the handle: the groove starts at the top of the bar,
#: because the stylesheet gives the arrow buttons no size at all, and the
#: handle is at least ``SPACE_XXL`` long.
HANDLE: int = 16

#: The tabs laid out in groups, which are the ones that scroll.
GROUPED: tuple[InspectorTab, ...] = (
    InspectorTab.ADJUST,
    InspectorTab.RETOUCH,
    InspectorTab.CROP,
)


@pytest.fixture
def panel(qt_app: object) -> Iterator[Inspector]:
    widget = Inspector()
    widget.set_adjustments(EditState(is_raw=True), (5500.0, 0.0))
    widget.set_geometry(Geometry())
    widget.resize(theme.PANEL_WIDTH, 900)
    widget.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    yield widget
    widget.hide()
    widget.deleteLater()


def squeeze(panel: Inspector, qt_app: object, height: int = SHORT) -> None:
    """Give the panel a window too short to show a whole tab."""
    panel.setFixedHeight(height + theme.HISTOGRAM_HEIGHT + theme.TAB_BAR_HEIGHT
                         + theme.PANEL_ACTION_ROW_HEIGHT)
    qt_app.processEvents()  # type: ignore[attr-defined]


# -- nothing gets cut off --------------------------------------------------


def test_every_group_keeps_its_height_in_a_short_panel(
    panel: Inspector, qt_app: object
) -> None:
    """The groups stay whole; it is the page that scrolls, not the rows."""
    squeeze(panel, qt_app)
    for group in panel.adjust_tab.groups():
        assert group.height() == group.sizeHint().height()


def test_every_slider_keeps_its_height_in_a_short_panel(
    panel: Inspector, qt_app: object
) -> None:
    squeeze(panel, qt_app)
    for row in panel.adjust_tab.rows().values():
        assert row.height() == theme.SLIDER_ROW_HEIGHT


def test_a_short_panel_has_somewhere_to_scroll(panel: Inspector, qt_app: object) -> None:
    squeeze(panel, qt_app)
    bar = panel.scroll_area(InspectorTab.ADJUST).verticalScrollBar()
    assert bar.maximum() > 0


def test_the_last_slider_can_be_scrolled_into_view(
    panel: Inspector, qt_app: object
) -> None:
    """The row a short panel used to cut off is reachable by scrolling."""
    squeeze(panel, qt_app)
    area = panel.scroll_area(InspectorTab.ADJUST)
    # The detail group is the last one (PLAN.md 3.1).
    last = panel.adjust_tab.rows()["colour_noise"]
    area.verticalScrollBar().setValue(area.verticalScrollBar().maximum())
    qt_app.processEvents()  # type: ignore[attr-defined]
    top_left = last.mapTo(area.viewport(), last.rect().topLeft())
    assert area.viewport().rect().contains(top_left)
    assert area.viewport().rect().contains(top_left + QPoint(0, last.height() - 1))


def test_the_panel_no_longer_forces_a_tall_window(panel: Inspector) -> None:
    """A scrolling page lets the window shrink to the minimum PLAN.md sets."""
    assert panel.minimumSizeHint().height() <= theme.WINDOW_MIN_HEIGHT


@pytest.mark.parametrize("tab", GROUPED)
def test_every_grouped_tab_scrolls(panel: Inspector, tab: InspectorTab) -> None:
    assert panel.scroll_area(tab).widgetResizable()


def test_there_are_four_tabs_with_looks_beside_adjust(panel: Inspector) -> None:
    """Adjust, filters, retouch, crop; what is known about the photo is a card (3.9)."""
    assert list(panel.tab_buttons()) == [
        InspectorTab.ADJUST,
        InspectorTab.LOOKS,
        InspectorTab.RETOUCH,
        InspectorTab.CROP,
    ]


def test_portrait_and_lens_correction_are_on_the_retouch_tab(panel: Inspector) -> None:
    """2026-11-01: neither is wanted on every photo, so they left the adjust tab."""
    retouch = panel.retouch_tab
    assert set(retouch.rows()) == {"skin", "whiten"}
    assert set(retouch.lens_rows()) == {"distortion", "vignetting"}
    assert [group.header.text() for group in retouch.groups()] == [
        tr("adjust.group.spots"),
        tr("adjust.group.portrait"),
        tr("adjust.group.lens"),
    ]
    adjust = panel.adjust_tab
    assert not {"skin", "whiten"} & set(adjust.rows())
    assert not hasattr(adjust, "lens_switch")


def test_the_retouch_tab_has_no_action_row_buttons(panel: Inspector) -> None:
    """Copy, paste and reset on the adjust tab already cover the whole photo."""
    assert panel.action_buttons(InspectorTab.RETOUCH) == ()


def test_the_looks_tab_scrolls_its_list_but_not_its_strength(panel: Inspector) -> None:
    """However far the list scrolls, the strength stays in reach."""
    area = panel.scroll_area(InspectorTab.LOOKS)
    looks = panel.looks_tab

    assert area is looks.scroll_area()
    assert area.widgetResizable()
    assert not area.widget().isAncestorOf(looks.strength_row())
    assert looks.isAncestorOf(looks.strength_row())


def test_the_filter_tab_has_no_action_row_buttons(panel: Inspector) -> None:
    """Saving a preset moved to the presets group of the adjust tab."""
    assert panel.action_buttons(InspectorTab.LOOKS) == ()


# -- the gutter stays put --------------------------------------------------


def test_folding_a_group_does_not_move_the_controls_sideways(
    panel: Inspector, qt_app: object
) -> None:
    """The scroll bar gutter is reserved whether the page scrolls or not."""
    area = panel.scroll_area(InspectorTab.ADJUST)
    squeeze(panel, qt_app)
    scrolling = area.viewport().width()
    for group in panel.adjust_tab.groups():
        group.header.setChecked(False)
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert area.viewport().width() == scrolling


def test_the_page_is_as_wide_as_the_panel_less_the_gutter(panel: Inspector) -> None:
    area = panel.scroll_area(InspectorTab.ADJUST)
    assert area.viewport().width() == theme.PANEL_WIDTH - theme.SCROLLBAR_WIDTH


def test_the_handle_is_hidden_while_there_is_nothing_to_scroll(
    panel: Inspector, qt_app: object
) -> None:
    """An empty bar would otherwise draw its handle down the whole track."""
    bar = panel.scroll_area(InspectorTab.CROP).verticalScrollBar()
    assert bar.property("empty") == "true"
    squeeze(panel, qt_app)
    assert panel.scroll_area(InspectorTab.ADJUST).verticalScrollBar().property(
        "empty"
    ) == "false"


# -- the wheel -------------------------------------------------------------


def turn_wheel(widget: object, down: bool = True) -> QWheelEvent:
    """Send one notch of the wheel to *widget*, and give the event back."""
    centre = QPointF(widget.rect().center())  # type: ignore[attr-defined]
    event = QWheelEvent(
        centre,
        QPointF(widget.mapToGlobal(centre.toPoint())),  # type: ignore[attr-defined]
        QPoint(0, 0),
        QPoint(0, -120 if down else 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, event)
    return event


def test_a_slider_leaves_the_wheel_to_the_page(panel: Inspector, qt_app: object) -> None:
    """PLAN.md 3.4: the wheel must not nudge a slider while the panel scrolls."""
    squeeze(panel, qt_app)
    row = panel.adjust_tab.rows()["exposure"]
    before = row.value
    event = turn_wheel(row)
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert not event.isAccepted()
    assert row.value == before


def test_the_wheel_scrolls_the_page(panel: Inspector, qt_app: object) -> None:
    """The page takes the wheel the slider above it let go of.

    Qt hands an ignored wheel event up the parents, but only one that came
    from the window system, so this sends it where that walk would end
    rather than to the slider itself.
    """
    squeeze(panel, qt_app)
    area = panel.scroll_area(InspectorTab.ADJUST)
    assert turn_wheel(area.viewport()).isAccepted()
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert area.verticalScrollBar().value() > 0


def test_tab_moves_to_the_controls_and_not_the_scroll_area(panel: Inspector) -> None:
    for tab in GROUPED:
        assert panel.scroll_area(tab).focusPolicy() == Qt.FocusPolicy.NoFocus


# -- the handle's three states ---------------------------------------------


def handle_colour(bar: QScrollBar, qt_app: object) -> str:
    """Return the colour the handle is painted, at the top of the groove."""
    qt_app.processEvents()  # type: ignore[attr-defined]
    image = bar.grab().toImage()
    return image.pixelColor(bar.width() // 2, HANDLE).name().upper()


def test_the_handle_brightens_as_the_pointer_takes_hold(
    panel: Inspector, qt_app: object
) -> None:
    """PLAN.md 3.4: hovering and pressing each have to show."""
    squeeze(panel, qt_app)
    bar = panel.scroll_area(InspectorTab.ADJUST).verticalScrollBar()
    at = QPoint(bar.width() // 2, HANDLE)
    assert handle_colour(bar, qt_app) == theme.TRACK.upper()
    QTest.mouseMove(bar, at)
    assert handle_colour(bar, qt_app) == theme.TRACK_HOVER.upper()
    QTest.mousePress(bar, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at)
    assert handle_colour(bar, qt_app) == theme.TRACK_PRESSED.upper()
    QTest.mouseRelease(bar, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at)


def test_the_pointer_lights_up_no_handle_in_an_empty_gutter(
    panel: Inspector, qt_app: object
) -> None:
    """Running the mouse down the reserved gutter of a tab that fits must
    not paint a bar with nothing behind it."""
    panel.set_current_tab(InspectorTab.CROP)
    bar = panel.scroll_area(InspectorTab.CROP).verticalScrollBar()
    assert bar.maximum() == 0
    at = QPoint(bar.width() // 2, HANDLE)
    QTest.mouseMove(bar, at)
    QTest.mousePress(bar, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at)
    painted = handle_colour(bar, qt_app)
    QTest.mouseRelease(bar, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at)
    for colour in (theme.TRACK, theme.TRACK_HOVER, theme.TRACK_PRESSED):
        assert painted != colour.upper()
