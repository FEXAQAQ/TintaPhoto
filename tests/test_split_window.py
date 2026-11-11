"""Tests for cutting the photo into pieces in the window (PLAN.md 2.12, 3.17)."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6.QtWidgets")

from app.i18n import tr  # noqa: E402
from core import split  # noqa: E402
from core.split import SplitSpec  # noqa: E402
from tests.test_main_window import (  # noqa: E402,F401 - the fixture is used by name
    _accept_export,
    _open_photos,
    _wait_for_photo,
    window,
)
from ui.main_window import MainWindow  # noqa: E402


def _cutting(window: MainWindow, tmp_path: object, qt_app: object) -> MainWindow:
    _open_photos(window, tmp_path, qt_app)
    _wait_for_photo(window, qt_app)
    window.enter_split()
    qt_app.processEvents()  # type: ignore[attr-defined]
    return window


def test_cutting_needs_a_photo(window: MainWindow) -> None:
    window.toggle_split()
    assert not window.in_split
    assert not window.toolbar.split_button.isChecked()


def test_the_cutting_panel_takes_the_inspectors_place(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _cutting(window, tmp_path, qt_app)
    assert window.in_split
    assert window.toolbar.split_button.isChecked()
    assert window.split_panel.isVisible() and not window.inspector.isVisible()
    assert window.toolbar.title_label.text() == tr("split.title")
    assert not window.toolbar.undo_button.isEnabled()
    assert window.preview.canvas.cuts.shown
    # The photo is 200 x 150, cut side by side in two by default.
    assert window.split_panel.size_label.text() == tr(
        "split.size", size=tr("export.pixels", width=100, height=150), count=2
    )

    window.on_escape()
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert not window.in_split
    assert window.inspector.isVisible() and not window.split_panel.isVisible()
    assert window.toolbar.title_label.text().endswith("a.jpg")
    assert not window.preview.canvas.cuts.shown


def test_the_cut_chosen_is_shown_and_remembered(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _cutting(window, tmp_path, qt_app)
    window.split_panel.split_choice.buttons()[split.SPLIT_GRID].click()
    assert window._settings.split.spec == SplitSpec(split.SPLIT_GRID).sane()
    assert window.preview.canvas.cuts.spec == SplitSpec(split.SPLIT_GRID).sane()
    assert window._settings.to_dict()["split"]["split"] == split.SPLIT_GRID


def test_the_modes_switch_straight_to_and_from_cutting(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _cutting(window, tmp_path, qt_app)
    handlers = window.shortcut_handlers()
    window._on_shortcut("shortcut.watermark", handlers["shortcut.watermark"])
    assert window.in_watermark and not window.in_split
    window._on_shortcut("shortcut.split", handlers["shortcut.split"])
    assert window.in_split and not window.in_watermark
    window._on_shortcut("shortcut.collage", handlers["shortcut.collage"])
    assert window.in_collage and not window.in_split
    window._on_shortcut("shortcut.split", handlers["shortcut.split"])
    assert window.in_split and not window.in_collage
    window._on_shortcut("shortcut.portrait", handlers["shortcut.portrait"])
    assert window.in_portrait and not window.in_split
    assert not window.split_panel.isVisible()


def test_editing_keys_do_nothing_while_cutting(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _cutting(window, tmp_path, qt_app)
    handlers = window.shortcut_handlers()
    window._on_shortcut("shortcut.crop", handlers["shortcut.crop"])
    assert not window.controller.cropping


def test_the_shortcut_is_s(window: MainWindow) -> None:
    assert window.shortcut_handlers()["shortcut.split"] == window.toggle_split
    assert tr("shortcut.split") == "S"
    assert tr("shortcut.split") in window.toolbar.split_button.toolTip()


def test_exporting_while_cutting_writes_the_pieces(
    window: MainWindow, tmp_path: object, qt_app: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cutting(window, tmp_path, qt_app)
    window.split_panel.split_choice.buttons()[split.SPLIT_STACKED].click()
    started = _accept_export(window, monkeypatch, True)
    window.export()
    assert started and started[-1][2] == SplitSpec(split.SPLIT_STACKED, 2)

    window.leave_split()
    window.export()
    # Left, the photo is written whole again.
    assert len(started[-1]) == 2


def test_the_export_dialog_says_how_the_photo_is_cut(
    window: MainWindow, tmp_path: object, qt_app: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ui.export_dialog import ExportDialog

    _cutting(window, tmp_path, qt_app)
    shown: list[tuple[bool, str]] = []

    def answer(dialog: ExportDialog) -> int:
        shown.append((dialog.cut_label.isVisibleTo(dialog), dialog.cut_label.text()))
        dialog.reject()
        return dialog.result()

    monkeypatch.setattr(ExportDialog, "exec", answer)
    window.export()
    size = tr("export.pixels", width=100, height=150)
    assert shown == [(True, tr("export.split_note", count=2, size=size, last="b"))]


def test_a_finished_cut_says_how_many_pieces(window: MainWindow, tmp_path: object) -> None:
    from core.exporter import ExportResult

    first = tmp_path / "a_edit_a.jpg"  # type: ignore[operator]
    second = tmp_path / "a_edit_b.jpg"  # type: ignore[operator]
    window.controller.export_finished.emit(
        ExportResult(path=first, copied=False, pieces=(first, second))
    )
    shown = window.toast.findChild(type(window.toolbar.title_label)).text()
    assert shown == tr("export.done_split", name="a_edit_a.jpg", count=2)


def test_choosing_photos_ends_when_cutting_begins(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)
    _wait_for_photo(window, qt_app)
    window.filmstrip.start_choosing()
    window.enter_split()
    assert not window.filmstrip.choosing
    window.leave_split()
    window.filmstrip.start_choosing()
    assert window.filmstrip.choosing
