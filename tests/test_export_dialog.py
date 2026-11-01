"""Tests for the export dialog and the progress on the toolbar (PLAN.md 3.11).

The dialog only decides: these check that what it shows matches what it will
hand to the exporter, that the rows it offers follow the format, and that
what it was last set to comes back the next time it is opened.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.settings import ExportSettings, Settings
from core import exporter

pytest.importorskip("PySide6.QtWidgets")

from app.i18n import tr  # noqa: E402
from ui import theme  # noqa: E402
from ui.export_dialog import ExportDialog  # noqa: E402
from ui.widgets.buttons import ProgressButton  # noqa: E402
from ui.widgets.switch import Switch  # noqa: E402

PHOTO = Path("D:/Photos/2026-10 Kyoto/_DSC1234.ARW")


@pytest.fixture
def dialog(qt_app: object) -> ExportDialog:
    return ExportDialog(PHOTO, ExportSettings())


# -- what the dialog offers ------------------------------------------------


def test_it_opens_at_the_width_plan_asks_for(dialog: ExportDialog) -> None:
    assert dialog.width() == theme.DIALOG_WIDTH_EXPORT


def test_every_row_label_is_the_same_width(dialog: ExportDialog) -> None:
    """PLAN.md 3.11: the label column is a fixed 72 px so the rows line up."""
    labels = dialog.findChildren(type(dialog.pixels_label))
    columns = {
        label.width()
        for label in labels
        if label.width() == theme.DIALOG_LABEL_WIDTH
    }
    assert columns == {theme.DIALOG_LABEL_WIDTH}


def test_the_three_formats_are_offered(dialog: ExportDialog) -> None:
    assert set(dialog.format_choice.buttons()) == set(exporter.FORMATS)


def test_a_jpeg_offers_a_quality_and_the_others_a_depth(dialog: ExportDialog) -> None:
    dialog.show()
    assert dialog.quality_slider.isVisible()
    assert not dialog.depth_choice.isVisible()

    dialog.format_choice.buttons()[exporter.FORMAT_TIFF].click()
    assert not dialog.quality_slider.isVisible()
    assert dialog.depth_choice.isVisible()
    dialog.hide()


def test_the_long_edge_is_only_asked_for_when_it_is_wanted(dialog: ExportDialog) -> None:
    dialog.show()
    assert not dialog.long_edge_field.isVisible()

    dialog.size_choice.buttons()["custom"].click()
    assert dialog.long_edge_field.isVisible()
    dialog.hide()


def test_the_name_is_the_original_plus_the_suffix(dialog: ExportDialog) -> None:
    assert dialog.name_field.text() == "_DSC1234_edit"
    assert dialog.extension_label.text() == ".jpg"


def test_the_extension_follows_the_format(dialog: ExportDialog) -> None:
    dialog.format_choice.buttons()[exporter.FORMAT_PNG].click()
    assert dialog.extension_label.text() == ".png"
    assert dialog.target.name == "_DSC1234_edit.png"


def test_the_file_goes_beside_the_photo_by_default(dialog: ExportDialog) -> None:
    assert dialog.folder == PHOTO.parent
    assert dialog.target == PHOTO.parent / "_DSC1234_edit.jpg"


def test_the_folder_is_shortened_in_the_middle_when_it_is_long(qt_app: object) -> None:
    deep = Path("D:/Photos/a very long folder name indeed/and another one/inside it")
    page = ExportDialog(deep / "_DSC1234.ARW", ExportSettings())
    assert page.folder_label.text() != str(page.folder)
    assert page.folder_label.toolTip() == str(page.folder)


def test_both_metadata_switches_are_on_to_begin_with(dialog: ExportDialog) -> None:
    """Keeping the details and leaving the location out are both the default."""
    assert isinstance(dialog.keep_metadata_switch, Switch)
    assert dialog.keep_metadata_switch.isChecked()
    assert dialog.remove_gps_switch.isChecked()


# -- what it hands over ----------------------------------------------------


def test_the_options_say_what_was_chosen(dialog: ExportDialog) -> None:
    dialog.format_choice.buttons()[exporter.FORMAT_PNG].click()
    dialog.depth_choice.buttons()["16"].click()
    dialog.size_choice.buttons()["custom"].click()
    dialog.long_edge_field.setText("1600")

    options = dialog.options
    assert options.format == exporter.FORMAT_PNG
    assert options.bits == 16
    assert options.long_edge == 1600


def test_the_original_size_means_no_long_edge(dialog: ExportDialog) -> None:
    assert dialog.options.long_edge is None


def test_a_long_edge_that_is_not_a_number_is_ignored(dialog: ExportDialog) -> None:
    dialog.size_choice.buttons()["custom"].click()
    dialog.long_edge_field.setText("not a number")
    assert dialog.options.long_edge is not None


def test_the_quality_is_a_whole_number(dialog: ExportDialog) -> None:
    dialog.quality_slider.set_value(77.4)
    assert dialog.options.quality == 77


def test_changing_anything_asks_for_the_figures_again(dialog: ExportDialog) -> None:
    asked: list[object] = []
    dialog.options_changed.connect(asked.append)

    dialog.format_choice.buttons()[exporter.FORMAT_TIFF].click()
    dialog.quality_slider.set_value(60.0, announce=True)
    dialog.keep_metadata_switch.setChecked(False)

    assert len(asked) >= 3
    assert all(isinstance(options, exporter.ExportOptions) for options in asked)


# -- what it is told -------------------------------------------------------


def test_it_shows_how_many_pixels_will_come_out(dialog: ExportDialog) -> None:
    dialog.set_output_size(6024, 4024)
    assert "6024" in dialog.pixels_label.text()
    assert "4024" in dialog.pixels_label.text()


def test_it_shows_the_estimate_in_megabytes(dialog: ExportDialog) -> None:
    dialog.set_estimate(8 * 1024 * 1024)
    assert "8.0" in dialog.estimate_label.text()


def test_an_estimate_of_nothing_is_shown_as_unknown(dialog: ExportDialog) -> None:
    dialog.set_estimate(0)
    assert dialog.estimate_label.text() == tr("export.estimate_unknown")


def test_it_says_when_the_original_will_simply_be_copied(dialog: ExportDialog) -> None:
    """X-08: the quality is not used then, so the user is told."""
    dialog.show()
    dialog.set_copying(True)
    assert dialog.note_label.isVisible()
    assert dialog.note_label.text() == tr("export.copy_note")

    dialog.set_copying(False)
    assert not dialog.note_label.isVisible()
    dialog.hide()


# -- remembering (PLAN.md 3.11) -------------------------------------------


def test_it_opens_on_what_was_last_chosen(qt_app: object) -> None:
    settings = ExportSettings(
        format=exporter.FORMAT_TIFF,
        bits=16,
        long_edge=2048,
        suffix="_web",
        keep_metadata=False,
        remove_gps=False,
    )
    page = ExportDialog(PHOTO, settings)

    assert page.format_choice.value == exporter.FORMAT_TIFF
    assert page.depth_choice.value == "16"
    assert page.size_choice.value == "custom"
    assert page.long_edge_field.text() == "2048"
    assert page.name_field.text() == "_DSC1234_web"
    assert not page.keep_metadata_switch.isChecked()
    assert not page.remove_gps_switch.isChecked()


def test_what_was_chosen_comes_back_to_be_remembered(dialog: ExportDialog) -> None:
    dialog.format_choice.buttons()[exporter.FORMAT_PNG].click()
    dialog.depth_choice.buttons()["16"].click()
    dialog.name_field.setText("_DSC1234_web")

    remembered = dialog.chosen_settings
    assert remembered.format == exporter.FORMAT_PNG
    assert remembered.bits == 16
    assert remembered.suffix == "_web"
    assert remembered.folder is None


def test_a_folder_of_its_own_is_remembered(qt_app: object) -> None:
    page = ExportDialog(PHOTO, ExportSettings(folder="D:/Exports"))
    assert page.folder == Path("D:/Exports")
    assert Path(page.chosen_settings.folder or "") == Path("D:/Exports")


def test_the_dialog_does_not_change_the_settings_it_was_given(
    dialog: ExportDialog,
) -> None:
    """Nothing is remembered until the export is actually asked for."""
    given = ExportSettings()
    page = ExportDialog(PHOTO, given)
    page.format_choice.buttons()[exporter.FORMAT_PNG].click()
    assert given.format == exporter.FORMAT_JPEG


# -- the progress on the toolbar button (X-07) ----------------------------


def test_the_button_says_it_is_exporting(qt_app: object) -> None:
    """The fill shows how far it has got; the words only say what it does."""
    button = ProgressButton("toolbar.export", "shortcut.export")
    assert button.text() == tr("toolbar.export")

    button.set_progress(0.42)
    assert button.progress == pytest.approx(0.42)
    assert button.text() == tr("toolbar.exporting")

    button.set_progress(None)
    assert button.progress is None
    assert button.text() == tr("toolbar.export")


def test_the_progress_cannot_leave_its_range(qt_app: object) -> None:
    button = ProgressButton("toolbar.export")
    button.set_progress(-1.0)
    assert button.progress == 0.0
    button.set_progress(2.0)
    assert button.progress == 1.0


def test_a_toast_with_something_to_click_stays_longer(qt_app: object) -> None:
    from ui.widgets.toast import Toast

    toast = Toast()
    toast.show_message("exported")
    plain = toast._timer.interval()
    toast.show_message("exported", "show me", lambda: None)
    with_action = toast._timer.interval()

    assert plain == theme.DURATION_TOAST_MS
    assert with_action == theme.DURATION_TOAST_ACTION_MS
    assert with_action > plain


def test_clicking_the_toast_link_does_what_it_offered(qt_app: object) -> None:
    from ui.widgets.toast import Toast

    toast = Toast()
    done: list[bool] = []
    toast.show_message("exported", "show me", lambda: done.append(True))
    toast._action.click()

    assert done == [True]
    assert toast.isHidden()


# -- the settings file (PLAN.md 7.3) --------------------------------------


def test_the_export_settings_survive_a_round_trip(tmp_path: Path) -> None:
    settings = Settings()
    settings.export = ExportSettings(
        format=exporter.FORMAT_PNG,
        bits=16,
        long_edge=1600,
        folder="D:/Exports",
        suffix="_web",
        keep_metadata=False,
        remove_gps=False,
        reveal_after=True,
    )
    path = tmp_path / "settings.json"
    assert settings.save(path)

    back = Settings.load(path).export
    assert back == settings.export


def test_nonsense_in_the_settings_file_falls_back_to_what_works() -> None:
    export = ExportSettings.from_dict(
        {"format": "bmp", "quality": "lots", "bits": 7, "long_edge": -5}
    )
    assert export.format == exporter.FORMAT_JPEG
    assert export.quality == exporter.QUALITY_DEFAULT
    assert export.bits == 8
    assert export.long_edge == exporter.LONG_EDGE_MIN
