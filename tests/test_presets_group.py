"""Tests for the saved presets at the top of the adjust tab (P-04 to P-06).

A preset is a set of slider values, so its list sits with the sliders: the
first group of the adjust tab, a plain button per preset that is never
marked, and a + on the header that saves a new one (PLAN.md 3.1).
"""

from __future__ import annotations

import pytest

from core.edit_state import EditState
from core.presets import NAME_MAX_LENGTH, Preset

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QEvent, QPointF  # noqa: E402
from PySide6.QtGui import QEnterEvent  # noqa: E402

from app.i18n import tr  # noqa: E402
from ui import theme  # noqa: E402
from PySide6.QtWidgets import QLabel  # noqa: E402

from ui.adjust_tab import AdjustTab  # noqa: E402
from ui.presets import NameDialog  # noqa: E402

WARM = Preset(id="1", name="Warm", adjustments={"temperature": 7000.0}, white_balance="kelvin")
PUNCHY = Preset(id="2", name="Punchy", adjustments={"vibrance": 40.0})
RAW_REST = EditState.for_photo(True, temperature=5600.0, tint=0.0)


@pytest.fixture
def tab(qt_app: object) -> AdjustTab:
    page = AdjustTab()
    page.set_photo(RAW_REST, (5600.0, 0.0))
    page.set_presets([PUNCHY, WARM])
    return page


# -- where the list is -------------------------------------------------------


def test_the_presets_are_the_first_group_of_the_adjust_tab(tab: AdjustTab) -> None:
    first, second = tab.groups()[:2]
    assert first is tab.presets_group
    assert first.header.text() == tr("adjust.group.presets")
    assert second.header.text() == tr("adjust.group.white_balance")


def test_every_preset_gets_a_row_in_the_group(tab: AdjustTab) -> None:
    assert [row.preset for row in tab.preset_rows()] == [PUNCHY, WARM]
    assert all(tab.presets_group.isAncestorOf(row) for row in tab.preset_rows())


def test_a_preset_row_is_an_ordinary_height_row(tab: AdjustTab) -> None:
    """No thumbnail: the sliders right under it show what it changed."""
    for row in tab.preset_rows():
        assert row.height() == theme.BUTTON_HEIGHT
        assert row.icon().isNull()
        assert not row.isCheckable()


def test_the_group_says_how_to_save_one_until_there_is_one(tab: AdjustTab) -> None:
    """Shown even when empty, so the first preset can be found."""
    hints = [
        label
        for label in tab.presets_group.findChildren(QLabel)
        if label.text() == tr("presets.empty")
    ]
    assert hints and all(label.isHidden() for label in hints)

    tab.set_presets([])
    assert not hints[0].isHidden()
    assert not tab.presets_group.isHidden()


def test_the_header_carries_the_button_that_saves_a_preset(tab: AdjustTab) -> None:
    button = tab.save_preset_button
    assert tab.presets_group.header.isAncestorOf(button)
    assert tr("shortcut.new_preset") in button.toolTip()
    assert tr("panel.action.save_preset") in button.toolTip()


# -- what a click does ------------------------------------------------------------


def test_clicking_a_preset_applies_it(tab: AdjustTab) -> None:
    """PLAN.md 3.1: 點擊才套用."""
    applied: list[Preset] = []
    tab.preset_applied.connect(applied.append)

    tab.preset_rows()[0].click()

    assert applied == [PUNCHY]


def test_clicking_a_preset_twice_applies_it_twice_and_marks_nothing(tab: AdjustTab) -> None:
    """A preset is an action, not a state the photo is in."""
    applied: list[Preset] = []
    tab.preset_applied.connect(applied.append)

    tab.preset_rows()[1].click()
    tab.preset_rows()[1].click()

    assert applied == [WARM, WARM]
    assert not any(row.isChecked() for row in tab.preset_rows())


def test_resting_on_a_preset_changes_nothing(tab: AdjustTab) -> None:
    applied: list[Preset] = []
    tab.preset_applied.connect(applied.append)
    row = tab.preset_rows()[1]

    point = QPointF(row.rect().center())
    row.enterEvent(QEnterEvent(point, point, point))
    row.leaveEvent(QEvent(QEvent.Type.Leave))

    assert applied == []


def test_every_row_says_what_it_does(tab: AdjustTab) -> None:
    assert tab.preset_rows()[0].toolTip() == tr("presets.hint")


def test_a_long_name_is_cut_short_rather_than_spilling_out(qt_app: object) -> None:
    long = Preset(id="9", name="A very long preset name " * 3, adjustments={"exposure": 1.0})
    page = AdjustTab()
    page.set_photo(RAW_REST, (5600.0, 0.0))
    page.set_presets([long])
    page.resize(theme.PANEL_WIDTH, 800)
    page.show()
    try:
        row = page.preset_rows()[0]
        assert row.text() != long.name
        assert row.text().endswith("…")
    finally:
        page.hide()


def test_a_folded_group_opens_to_show_a_preset_just_saved(tab: AdjustTab) -> None:
    tab.presets_group.header.setChecked(False)

    row = tab.preset_row(WARM.id)

    assert row is not None and row.preset == WARM
    assert tab.presets_group.header.isChecked()


# -- the name dialog -------------------------------------------------------


def test_the_name_dialog_starts_from_the_name_it_was_given(qt_app: object) -> None:
    dialog = NameDialog("presets.rename", "Warm")

    assert dialog.name == "Warm"
    assert dialog.windowTitle() == tr("presets.rename")


def test_the_name_dialog_has_a_length_limit(qt_app: object) -> None:
    dialog = NameDialog("presets.save_title")
    dialog.field.setText("x" * 500)

    assert len(dialog.name) == NAME_MAX_LENGTH
