"""Tests for choosing several photos and exporting them (PLAN.md 3.1, X-09).

Right-clicking a thumbnail starts a choosing mode in which every thumbnail
has a tick box; the export button then names how many are ticked and writes
all of them, and pressing it again while they are written stops them.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from app.i18n import tr  # noqa: E402
from app.settings import ExportSettings, Settings, WindowState  # noqa: E402
from ui.export_dialog import ExportDialog  # noqa: E402
from ui.filmstrip import Filmstrip, Thumbnail  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402
from ui.widgets.buttons import ExportLook, ProgressButton  # noqa: E402

PHOTOS = [Path(f"D:/Photos/DSC0000{index}.ARW") for index in range(5)]


@pytest.fixture
def strip(qt_app: object) -> Filmstrip:
    filmstrip = Filmstrip()
    filmstrip.set_photos(PHOTOS)
    filmstrip.set_current(0, len(PHOTOS))
    return filmstrip


def _thumbnail(strip: Filmstrip, path: Path) -> Thumbnail:
    found = strip.thumbnail(path)
    assert found is not None
    return found


def _click(strip: Filmstrip, path: Path, modifier=Qt.KeyboardModifier.NoModifier) -> None:
    thumbnail = _thumbnail(strip, path)
    QTest.mouseClick(thumbnail, Qt.MouseButton.LeftButton, modifier, thumbnail.rect().center())


# -- the filmstrip ----------------------------------------------------------


def test_choosing_starts_with_the_photo_right_clicked(strip: Filmstrip) -> None:
    strip.start_choosing(PHOTOS[2])

    assert strip.choosing
    assert strip.checked_photos() == [PHOTOS[2]]
    assert all(_thumbnail(strip, path).checkable for path in PHOTOS)


def test_photos_chosen_with_control_are_ticked_too(strip: Filmstrip) -> None:
    _click(strip, PHOTOS[3], Qt.KeyboardModifier.ControlModifier)

    strip.start_choosing(PHOTOS[4])

    assert strip.checked_photos() == [PHOTOS[0], PHOTOS[3], PHOTOS[4]]


def test_a_click_ticks_without_changing_the_photo_being_edited(strip: Filmstrip) -> None:
    asked: list[Path] = []
    strip.photo_selected.connect(asked.append)
    strip.start_choosing(PHOTOS[1])

    _click(strip, PHOTOS[3])
    _click(strip, PHOTOS[1])

    assert strip.checked_photos() == [PHOTOS[3]]
    assert asked == []


def test_shift_ticks_everything_from_the_last_photo(strip: Filmstrip) -> None:
    strip.start_choosing(PHOTOS[1])

    _click(strip, PHOTOS[4], Qt.KeyboardModifier.ShiftModifier)

    assert strip.checked_photos() == PHOTOS[1:]


def test_all_and_none(strip: Filmstrip) -> None:
    strip.start_choosing(PHOTOS[1])

    strip.check_all()
    assert strip.checked_photos() == PHOTOS
    strip.check_none()
    assert strip.checked_photos() == []


def test_the_position_becomes_the_count_while_choosing(strip: Filmstrip) -> None:
    from PySide6.QtWidgets import QLabel

    def texts() -> list[str]:
        return [label.text() for label in strip.findChildren(QLabel)]

    strip.start_choosing(PHOTOS[1])
    _click(strip, PHOTOS[2])
    assert tr("filmstrip.checked", count=2) in texts()
    assert not strip.done_button.isHidden()

    strip.stop_choosing()
    assert tr("filmstrip.position", index=1, total=len(PHOTOS)) in texts()
    assert strip.done_button.isHidden()
    assert strip.checked_photos() == []


def test_choosing_ends_with_a_new_folder(strip: Filmstrip) -> None:
    strip.start_choosing(PHOTOS[1])

    strip.set_photos(PHOTOS[:2])

    assert not strip.choosing


def test_choosing_is_not_offered_in_the_collage(strip: Filmstrip) -> None:
    strip.start_choosing(PHOTOS[1])

    strip.set_choosing_allowed(False)
    assert not strip.choosing
    strip.start_choosing(PHOTOS[1])
    assert not strip.choosing


def test_a_ticked_photo_has_a_filled_box_and_the_others_an_empty_one(
    strip: Filmstrip, qt_app: object
) -> None:
    from ui import theme

    strip.resize(800, 112)
    strip.show()
    strip.start_choosing(PHOTOS[0])
    qt_app.processEvents()  # type: ignore[attr-defined]
    # Just inside the box's rounded corner, clear of the tick.
    inside = theme.SPACE_XS + theme.CHECK_BOX_SIZE // 4

    def colour(path: Path) -> str:
        picture = _thumbnail(strip, path).grab().toImage()
        return picture.pixelColor(inside, inside).name().upper()

    assert colour(PHOTOS[0]) == theme.ACCENT.upper()
    assert colour(PHOTOS[1]) == theme.BG_APP.upper()
    strip.hide()


# -- the export dialog ------------------------------------------------------


@pytest.fixture
def dialog(qt_app: object) -> ExportDialog:
    return ExportDialog(PHOTOS[0], ExportSettings(), batch=3)


def test_the_dialog_says_how_many_photos(dialog: ExportDialog) -> None:
    from PySide6.QtWidgets import QLabel

    texts = [label.text() for label in dialog.findChildren(QLabel)]
    assert tr("export.batch_photos", count=3) in texts
    assert tr("export.batch_stem") in texts
    assert dialog.estimate_label.text() == tr("export.batch_total", count=3)


def test_the_name_field_holds_only_the_suffix(dialog: ExportDialog) -> None:
    assert dialog.name_field.text() == ExportSettings().suffix

    dialog.name_field.setText("_web")

    assert dialog.suffix == "_web"
    assert dialog.chosen_settings.suffix == "_web"


def test_each_photo_keeps_its_own_size(dialog: ExportDialog) -> None:
    assert dialog.pixels_label.text() == tr("export.size.each")


# -- the export button -------------------------------------------------------


@pytest.fixture
def button(qt_app: object, monkeypatch: pytest.MonkeyPatch) -> Iterator[ProgressButton]:
    from ui import native

    monkeypatch.setattr(native, "animations_enabled", lambda: False)
    widget = ProgressButton("toolbar.export", "shortcut.export")
    widget.show()
    yield widget
    widget.hide()


def test_the_button_names_how_many_photos(button: ProgressButton) -> None:
    button.set_name("toolbar.export_count", count=4)

    assert button.text() == tr("toolbar.export_count", count=4)
    assert button.toolTip().startswith(tr("toolbar.export"))


def test_the_working_button_counts_and_offers_to_stop(
    button: ProgressButton, monkeypatch: pytest.MonkeyPatch
) -> None:
    button.set_working_words(tr("toolbar.exporting_count", index=2, total=9))
    button.set_stoppable(True)
    button.start()
    assert button.look is ExportLook.WORKING
    assert button.text() == tr("toolbar.exporting_count", index=2, total=9)


    # The offscreen platform has no pointer to put over it.
    monkeypatch.setattr(button, "underMouse", lambda: True)
    button.set_stoppable(True)
    assert button.text() == tr("toolbar.stop_export")


def test_resting_forgets_the_count(button: ProgressButton) -> None:
    button.set_working_words("x")
    button.set_stoppable(True)
    button.start()

    button.rest()

    assert not button.stoppable
    button.start()
    assert button.text() == tr("toolbar.exporting")


# -- the whole of it, in the window ----------------------------------------


@pytest.fixture
def window(qt_app: object) -> Iterator[MainWindow]:
    settings = Settings(window=WindowState(x=0, y=0, width=1440, height=900))
    main_window = MainWindow(settings)
    main_window.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    yield main_window
    main_window.batch.shutdown()
    main_window.hide()
    main_window.deleteLater()


def _wait(qt_app: object, predicate, seconds: float = 20.0) -> bool:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        qt_app.processEvents()  # type: ignore[attr-defined]
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _open_folder(window: MainWindow, tmp_path: Path, qt_app: object) -> list[Path]:
    from PIL import Image

    folder = tmp_path / "照片 2026"
    folder.mkdir()
    paths = []
    for index in range(3):
        path = folder / f"p{index}.jpg"
        Image.new("RGB", (60, 40), (40 * index, 90, 120)).save(path, "JPEG")
        paths.append(path)
    window.controller.open(folder)
    assert _wait(qt_app, lambda: window.controller.current is not None)
    return paths


def test_the_ticked_photos_are_written_and_the_choice_ends(
    window: MainWindow, tmp_path: Path, qt_app: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _open_folder(window, tmp_path, qt_app)
    window.filmstrip.start_choosing(paths[1])
    window.check_all_photos()
    button = window.toolbar.export_button
    assert button.text() == tr("toolbar.export_count", count=3)
    assert button.isEnabled()

    monkeypatch.setattr(window, "_run_dialog", lambda dialog, send_off=False: True)
    messages: list[str] = []
    monkeypatch.setattr(window.toast, "show_message", lambda message, *rest: messages.append(message))
    window.export()

    assert not window.filmstrip.choosing
    assert _wait(qt_app, lambda: not window.batch.exporting)
    written = sorted(path.name for path in paths[0].parent.glob("*_edit.jpg"))
    assert written == ["p0_edit.jpg", "p1_edit.jpg", "p2_edit.jpg"]
    assert messages == [tr("export.batch_done", count=3)]


def test_nothing_ticked_cannot_be_exported(window: MainWindow, tmp_path: Path, qt_app: object) -> None:
    paths = _open_folder(window, tmp_path, qt_app)
    window.filmstrip.start_choosing(paths[0])
    window.check_no_photos()

    assert not window.toolbar.export_button.isEnabled()
    window.on_escape()
    assert not window.filmstrip.choosing
    assert window.toolbar.export_button.text() == tr("toolbar.export")
    assert window.toolbar.export_button.isEnabled()


def test_pressing_the_button_while_writing_stops(
    window: MainWindow, tmp_path: Path, qt_app: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _open_folder(window, tmp_path, qt_app)
    window.filmstrip.start_choosing(paths[0])
    window.check_all_photos()
    monkeypatch.setattr(window, "_run_dialog", lambda dialog, send_off=False: True)
    messages: list[str] = []
    monkeypatch.setattr(window.toast, "show_message", lambda message, *rest: messages.append(message))

    window.export()
    assert window.batch.exporting
    window.export()

    assert window.batch.stopping
    assert not window.toolbar.export_button.isEnabled()
    assert _wait(qt_app, lambda: not window.batch.exporting)
    assert len(messages) == 1
    # Stopping may come after any photo, or after the last one.
    assert messages[0] in {
        tr("export.batch_stopped", written=count, total=3) for count in range(4)
    } | {tr("export.batch_done", count=3)}


def test_the_ticks_survive_moving_between_photos(window: MainWindow, tmp_path: Path, qt_app: object) -> None:
    paths = _open_folder(window, tmp_path, qt_app)
    window.filmstrip.start_choosing(paths[2])

    window.controller.select_next()
    qt_app.processEvents()  # type: ignore[attr-defined]

    assert window.filmstrip.checked_photos() == [paths[2]]
