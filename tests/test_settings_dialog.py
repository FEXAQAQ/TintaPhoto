"""Tests for the settings dialog and what each setting does (PLAN.md 3.10).

Every option is in the dialog so that it can be changed, so each test here
changes one and checks both that the setting took it on and that something
in the program actually follows it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app import settings as settings_module
from app.settings import Settings
from core import color_management as cm
from core.cache import CACHE_LIMITS_BYTES
from core.edit_state import EditState

pytest.importorskip("PySide6.QtWidgets")

from app.i18n import tr  # noqa: E402
from ui import theme  # noqa: E402
from ui.histogram_widget import clipping_overlay  # noqa: E402
from ui.settings_dialog import SettingsDialog  # noqa: E402
from ui.shortcuts import SHORTCUTS  # noqa: E402


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def dialog(qt_app: object, settings: Settings) -> SettingsDialog:
    return SettingsDialog(settings)


# -- the dialog itself ----------------------------------------------------


def test_it_has_the_five_categories_of_plan(dialog: SettingsDialog) -> None:
    assert dialog.categories.count() == 5
    assert dialog.pages.count() == 5
    assert dialog.categories.item(0).text() == tr("settings.category.general")


def test_it_is_the_size_plan_asks_for(dialog: SettingsDialog) -> None:
    assert dialog.width() == theme.DIALOG_WIDTH_SETTINGS


def test_choosing_a_category_shows_its_page(dialog: SettingsDialog) -> None:
    dialog.categories.setCurrentRow(3)
    assert dialog.pages.currentIndex() == 3


def test_there_is_no_ok_or_cancel(dialog: SettingsDialog) -> None:
    """PLAN.md 3.10: changes take effect at once, so there is nothing to confirm."""
    from PySide6.QtWidgets import QDialogButtonBox

    assert dialog.findChild(QDialogButtonBox) is None


def test_every_change_is_announced(dialog: SettingsDialog) -> None:
    announced: list[int] = []
    dialog.changed.connect(lambda: announced.append(1))

    dialog.warn_switch.setChecked(False)
    dialog.background_choice.buttons()[settings_module.BACKGROUND_MID].click()
    dialog.suffix_field.setText("_web")
    dialog._on_suffix("_web")

    assert len(announced) == 3


# -- general --------------------------------------------------------------


def test_the_language_is_remembered_and_needs_no_restart(
    dialog: SettingsDialog, settings: Settings
) -> None:
    """Changing it takes effect at once, so the row says nothing about it.

    What the interface then does with it is in ``tests/test_language_switch``.
    """
    language_row = dialog.language_box.parent()

    dialog.language_box.setCurrentIndex(dialog.language_box.findData("en"))

    assert settings.language == "en"
    assert not language_row.note.text()


def test_what_happens_at_start_up_can_be_changed(
    dialog: SettingsDialog, settings: Settings
) -> None:
    dialog.startup_box.setCurrentIndex(
        dialog.startup_box.findData(settings_module.STARTUP_LAST_FOLDER)
    )
    assert settings.startup_action == settings_module.STARTUP_LAST_FOLDER


def test_the_read_only_warning_can_be_turned_off(
    dialog: SettingsDialog, settings: Settings
) -> None:
    dialog.warn_switch.setChecked(False)
    assert not settings.warn_read_only


# -- preview and colour ---------------------------------------------------


def test_the_clipping_marks_can_be_colourless(
    dialog: SettingsDialog, settings: Settings
) -> None:
    dialog.clipping_choice.buttons()[settings_module.CLIPPING_MONO].click()
    assert settings.clipping_marks == settings_module.CLIPPING_MONO


def test_colourless_marks_are_the_opposite_of_what_they_sit_on() -> None:
    """A mark has to show on what it marks, which a hue is not needed for."""
    photo = np.zeros((2, 2, 3), dtype=np.uint8)
    photo[0, 0] = 255  # a blown highlight
    photo[1, 1] = 0  # a blocked shadow

    coloured = clipping_overlay(photo, True, True, coloured=True)
    mono = clipping_overlay(photo, True, True, coloured=False)

    assert tuple(coloured[0, 0]) != tuple(mono[0, 0])
    assert tuple(mono[0, 0]) == (0, 0, 0)
    assert tuple(mono[1, 1]) == (255, 255, 255)


def test_the_clipped_shadows_are_marked_blue_whatever_the_interface_wears() -> None:
    """PLAN.md 2.2 asks for blue over a blocked shadow and red over a blown
    highlight. The blue used to be the accent colour, which is now the white
    of the headings, so the mark has a colour of its own."""
    from PySide6.QtGui import QColor

    from ui import theme

    photo = np.zeros((2, 2, 3), dtype=np.uint8)
    photo[0, 0] = 255  # a blown highlight
    photo[1, 1] = 0  # a blocked shadow
    marked = clipping_overlay(photo, True, True, coloured=True)

    blue = QColor(theme.CLIP_SHADOW)
    assert tuple(marked[1, 1]) == (blue.red(), blue.green(), blue.blue())
    red = QColor(theme.DANGER)
    assert tuple(marked[0, 0]) == (red.red(), red.green(), red.blue())
    assert blue.blue() > blue.red()


def test_the_preview_background_can_be_changed(
    dialog: SettingsDialog, settings: Settings
) -> None:
    dialog.background_choice.buttons()[settings_module.BACKGROUND_MID].click()
    assert settings.preview_background == settings_module.BACKGROUND_MID


def test_each_background_has_a_colour_of_its_own_in_the_stylesheet() -> None:
    qss = theme.build_qss()
    for choice, colour in (
        (settings_module.BACKGROUND_BLACK, theme.PREVIEW_BG_BLACK),
        (settings_module.BACKGROUND_DARK, theme.PREVIEW_BG_DARK),
        (settings_module.BACKGROUND_MID, theme.PREVIEW_BG_MID),
    ):
        assert f'[background="{choice}"]' in qss
        assert colour in qss


def test_the_temperature_can_be_shown_as_a_nudge(
    dialog: SettingsDialog, settings: Settings
) -> None:
    dialog.temperature_choice.buttons()[settings_module.TEMPERATURE_RELATIVE].click()
    assert settings.temperature_unit == settings_module.TEMPERATURE_RELATIVE


@pytest.mark.parametrize("shot", [3200.0, 5600.0])
@pytest.mark.parametrize("kelvin", [2000.0, 3200.0, 5600.0, 12000.0, 50000.0])
def test_the_relative_temperature_can_be_turned_back_into_kelvin(
    shot: float, kelvin: float
) -> None:
    """Nothing may become unreachable because of how the slider reads."""
    low, high = 2000.0, 50000.0
    relative = cm.kelvin_to_relative(kelvin, shot, low, high)
    assert -100.0 <= relative <= 100.0
    assert cm.relative_to_kelvin(relative, shot, low, high) == pytest.approx(kelvin, rel=1e-6)


def test_the_camera_s_own_temperature_is_the_middle_of_the_nudge() -> None:
    assert cm.kelvin_to_relative(5600.0, 5600.0, 2000.0, 50000.0) == 0.0


def test_a_warmer_photo_is_a_higher_nudge() -> None:
    """The warm end is on the right, as it is on the kelvin slider."""
    assert cm.kelvin_to_relative(8000.0, 5600.0, 2000.0, 50000.0) > 0.0
    assert cm.kelvin_to_relative(3000.0, 5600.0, 2000.0, 50000.0) < 0.0


def test_the_slider_shows_the_nudge_but_the_photo_keeps_kelvin(qt_app: object) -> None:
    from ui.adjust_tab import AdjustTab

    tab = AdjustTab()
    tab.set_photo(EditState(is_raw=True, temperature=5600.0), (5600.0, 0.0))
    assert tab.rows()["temperature"].value == 5600.0

    tab.set_temperature_unit(False)
    assert tab.rows()["temperature"].value == 0.0

    tab.rows()["temperature"].set_value(50.0, announce=True)
    assert tab.state.temperature > 5600.0
    assert tab.state.temperature < 50000.0


def test_a_jpeg_is_not_affected_by_the_temperature_unit(qt_app: object) -> None:
    """PLAN.md 3.10 says RAW only; a JPEG's slider is relative already."""
    from ui.adjust_tab import AdjustTab

    tab = AdjustTab()
    tab.set_photo(EditState(is_raw=False, temperature=20.0), (0.0, 0.0))
    tab.set_temperature_unit(False)
    assert tab.rows()["temperature"].value == 20.0


# -- export ---------------------------------------------------------------


def test_the_format_and_quality_are_one_choice(
    dialog: SettingsDialog, settings: Settings
) -> None:
    dialog.format_box.setCurrentIndex(dialog.format_box.findData("tiff:0:16"))
    assert settings.export.format == "tiff"
    assert settings.export.bits == 16


def test_a_quality_the_export_dialog_set_is_still_shown(qt_app: object) -> None:
    settings = Settings()
    settings.export.quality = 85
    page = SettingsDialog(settings)
    assert page.format_box.findData("jpg:85:8") >= 0


def test_the_suffix_can_be_changed(dialog: SettingsDialog, settings: Settings) -> None:
    dialog.suffix_field.setText("_web")
    dialog._on_suffix("_web")
    assert settings.export.suffix == "_web"


def test_showing_the_file_afterwards_can_be_turned_on(
    dialog: SettingsDialog, settings: Settings
) -> None:
    dialog.reveal_switch.setChecked(True)
    assert settings.export.reveal_after


def test_the_location_goes_back_to_beside_the_original(
    qt_app: object,
) -> None:
    settings = Settings()
    settings.export.folder = "D:/Exports"
    page = SettingsDialog(settings)
    assert page.location_box.currentData() == "chosen"

    page.location_box.setCurrentIndex(page.location_box.findData("beside"))
    assert settings.export.folder is None


# -- performance and cache ------------------------------------------------


class _FakeController:
    """Just enough of the controller for the cache rows."""

    def __init__(self) -> None:
        self.cleared = 0
        self.usage = 3 * 1024 * 1024

    def thumbnail_cache_usage(self) -> int:
        return self.usage

    def clear_thumbnail_cache(self) -> None:
        self.cleared += 1
        self.usage = 0


def test_the_cache_limit_can_be_changed(
    dialog: SettingsDialog, settings: Settings
) -> None:
    smallest = CACHE_LIMITS_BYTES[0]
    dialog.cache_box.setCurrentIndex(dialog.cache_box.findData(str(smallest)))
    assert settings.cache_limit_bytes == smallest


def test_the_cache_row_says_how_much_is_in_use(qt_app: object) -> None:
    controller = _FakeController()
    page = SettingsDialog(Settings(), controller)
    assert "3" in page.cache_row.note.text()


def test_clearing_the_cache_goes_through_the_controller(qt_app: object) -> None:
    controller = _FakeController()
    page = SettingsDialog(Settings(), controller)

    page.clear_button.click()

    assert controller.cleared == 1
    assert page.cache_row.note.text() == tr("settings.cache_usage", size="0 MB")


def test_preloading_can_be_turned_off(
    dialog: SettingsDialog, settings: Settings
) -> None:
    dialog.preload_switch.setChecked(False)
    assert not settings.preload_neighbours


# -- shortcuts ------------------------------------------------------------


def test_every_shortcut_is_listed(dialog: SettingsDialog) -> None:
    assert len(dialog._shortcut_rows) == len(SHORTCUTS)


def test_searching_narrows_the_list(dialog: SettingsDialog) -> None:
    dialog.search_field.setText("Ctrl+Shift+E")
    shown = [row for row, _ in dialog._shortcut_rows if not row.isHidden()]
    assert len(shown) == 1

    dialog.search_field.setText("")
    assert all(not row.isHidden() for row, _ in dialog._shortcut_rows)


def test_searching_matches_the_name_as_well_as_the_keys(
    dialog: SettingsDialog,
) -> None:
    dialog.search_field.setText(tr("toolbar.export").lower())
    shown = [row for row, _ in dialog._shortcut_rows if not row.isHidden()]
    assert shown


# -- the settings file ----------------------------------------------------


def test_every_new_setting_survives_a_round_trip(tmp_path: Path) -> None:
    settings = Settings(
        warn_read_only=False,
        use_display_profile=False,
        clipping_marks=settings_module.CLIPPING_MONO,
        preview_background=settings_module.BACKGROUND_MID,
        temperature_unit=settings_module.TEMPERATURE_RELATIVE,
        cache_limit_bytes=CACHE_LIMITS_BYTES[-1],
    )
    path = tmp_path / "settings.json"
    assert settings.save(path)

    assert Settings.load(path) == settings


def test_nonsense_in_the_file_falls_back_to_what_works() -> None:
    settings = Settings.from_dict(
        {
            "clipping_marks": "rainbow",
            "preview_background": "plaid",
            "temperature_unit": "furlongs",
            "cache_limit_bytes": 7,
        }
    )
    assert settings.clipping_marks == settings_module.CLIPPING_COLOURED
    assert settings.preview_background == settings_module.BACKGROUND_DARK
    assert settings.temperature_unit == settings_module.TEMPERATURE_KELVIN
    assert settings.cache_limit_bytes in CACHE_LIMITS_BYTES
