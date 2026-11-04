"""Tests for changing the interface language while the program runs (U-02).

The language is changed in the settings dialog and takes effect at once: no
widget is rebuilt, so what is checked here is that every kind of text on
screen is read again, that nothing is left in the language just left behind,
and that the photo on screen survives the change.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from app import i18n
from app.i18n import LANGUAGE_EN, LANGUAGE_ZH_TW, Translator
from app.paths import resources_dir
from app.settings import Settings, WindowState

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QObject  # noqa: E402
from PySide6.QtGui import QAction  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton,
    QComboBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QWidget,
)

from ui.inspector import InspectorTab  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402
from ui.retranslate import retranslate_tree  # noqa: E402
from ui.settings_dialog import SettingsDialog  # noqa: E402

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800


def _load(language: str) -> dict[str, str]:
    path = resources_dir() / "i18n" / f"{language}.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture
def window(qt_app: object) -> Iterator[MainWindow]:
    """Return a shown window, with the language put back afterwards."""
    settings = Settings(
        language=LANGUAGE_ZH_TW,
        window=WindowState(x=0, y=0, width=WINDOW_WIDTH, height=WINDOW_HEIGHT),
    )
    i18n.init_translator(LANGUAGE_ZH_TW)
    main_window = MainWindow(settings)
    main_window.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    yield main_window
    main_window.hide()
    main_window.deleteLater()
    i18n.init_translator(LANGUAGE_ZH_TW)


def _switch(window: MainWindow, language: str) -> None:
    """Change the language the way the settings dialog does."""
    window._settings.language = language
    window.apply_settings()


def _open_photo(window: MainWindow, folder: Path, qt_app: object) -> None:
    """Open a folder holding one photo, so the panels have something to show."""
    from PIL import Image

    folder.mkdir(exist_ok=True)
    Image.new("RGB", (240, 180), (90, 120, 150)).save(folder / "a.jpg", "JPEG")
    window.controller.open(folder)
    deadline = time.perf_counter() + 10.0
    while not window.controller.photos and time.perf_counter() < deadline:
        qt_app.processEvents()  # type: ignore[attr-defined]
    qt_app.processEvents()  # type: ignore[attr-defined]


# -- the walk itself ------------------------------------------------------


class _Counter(QWidget):
    """A widget that counts how often it is asked to read its text again."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.answered = 0

    def retranslate(self) -> None:
        self.answered += 1


def test_the_walk_reaches_every_depth(qt_app: object) -> None:
    root = _Counter()
    middle = QWidget(root)
    leaf = _Counter(middle)

    assert retranslate_tree(root) == 2
    assert root.answered == 1
    assert leaf.answered == 1


def test_a_widget_may_replace_its_children_while_being_read(qt_app: object) -> None:
    """A widget that rebuilds its children must not break the walk.

    The application menu does exactly this: its recent folder submenu is
    built again from scratch, so the actions the walk would have gone on to
    are gone by then.
    """

    class Rebuilder(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.child = _Counter(self)

        def retranslate(self) -> None:
            self.child.setParent(None)
            self.child.deleteLater()
            self.child = _Counter(self)

    root = Rebuilder()
    # Reached the new child, not the one that was thrown away.
    assert retranslate_tree(root) == 2
    assert root.child.answered == 1


# -- the window -----------------------------------------------------------


def test_the_toolbar_and_the_tabs_change_at_once(window: MainWindow) -> None:
    english = Translator(LANGUAGE_EN)
    chinese = Translator(LANGUAGE_ZH_TW)
    assert window.toolbar.export_button.text() == chinese.tr("toolbar.export")

    _switch(window, LANGUAGE_EN)

    assert window.toolbar.export_button.text() == english.tr("toolbar.export")
    assert window.toolbar.title_label.text() == english.tr("toolbar.no_photo")
    assert window.toolbar.undo_button.toolTip().startswith(english.tr("toolbar.undo"))
    tab_button = window.inspector.tab_buttons()[InspectorTab.ADJUST]
    assert tab_button.text() == english.tr("panel.tab.adjust")


def test_the_menu_the_welcome_screen_and_the_filmstrip_change(
    window: MainWindow,
) -> None:
    english = Translator(LANGUAGE_EN)

    _switch(window, LANGUAGE_EN)

    assert window.toolbar.menu.open_file_action.text() == english.tr("menu.open_file")
    assert window.toolbar.menu.recent_menu.title() == english.tr("menu.recent")
    assert window.preview.welcome.canvas.tagline == english.tr("welcome.tagline")
    assert window.preview.welcome.open_file_button.label == english.tr("welcome.open_file")
    assert window.preview.welcome.open_folder_button.label == english.tr("welcome.open_folder")
    assert window.preview.before_label.text() == english.tr("preview.before")
    assert window.filmstrip._empty.text() == english.tr("filmstrip.empty")


def test_the_sliders_and_the_crop_ratios_change(window: MainWindow) -> None:
    english = Translator(LANGUAGE_EN)

    _switch(window, LANGUAGE_EN)

    exposure = window.inspector.adjust_tab.rows()["exposure"]
    assert exposure.findChildren(QLabel)[0].text() == english.tr("adjust.exposure")
    crop_tab = window.inspector.crop_tab
    assert crop_tab.straighten_button.text() == english.tr("crop.straighten")
    assert crop_tab.straighten_button.toolTip() == english.tr("crop.straighten.hint")


def test_the_values_of_the_info_tab_are_written_again(window: MainWindow) -> None:
    english = Translator(LANGUAGE_EN)
    info_tab = window.info_card
    info_tab.set_edited(True)

    _switch(window, LANGUAGE_EN)

    assert info_tab._rows["info.edit_state"].value.text() == english.tr("info.edited")


def test_the_photo_stays_open_through_the_change(
    window: MainWindow, qt_app: object, tmp_path: Path
) -> None:
    _open_photo(window, tmp_path / "photos", qt_app)
    before = window.controller.current
    assert before is not None

    _switch(window, LANGUAGE_EN)

    assert window.controller.current == before
    assert window.controller.photos
    assert len(window.filmstrip._thumbnails) == 1


def test_switching_back_puts_the_chinese_back(window: MainWindow) -> None:
    chinese = Translator(LANGUAGE_ZH_TW)

    _switch(window, LANGUAGE_EN)
    _switch(window, LANGUAGE_ZH_TW)

    assert window.toolbar.export_button.text() == chinese.tr("toolbar.export")
    assert window.inspector.tab_buttons()[InspectorTab.ADJUST].text() == chinese.tr(
        "panel.tab.adjust"
    )


def test_nothing_of_the_old_language_is_left_on_screen(
    window: MainWindow, qt_app: object, tmp_path: Path
) -> None:
    """Sweep the whole window for text the language just left behind.

    The settings dialog is open as it would be in use, and a photo is open
    as well, so the sweep covers the panels with something to show. This is
    what catches a widget added later that holds text but offers no way of
    reading it again.
    """
    _open_photo(window, tmp_path / "photos", qt_app)
    window.info_card.set_edited(False)
    chinese = _load(LANGUAGE_ZH_TW)
    english = _load(LANGUAGE_EN)
    only_chinese = {
        text for key, text in chinese.items() if text != english.get(key)
    }
    dialog = SettingsDialog(window._settings, window.controller, window)
    dialog.changed.connect(window.apply_settings)

    dialog.language_box.setCurrentIndex(dialog.language_box.findData(LANGUAGE_EN))
    qt_app.processEvents()  # type: ignore[attr-defined]

    left_behind = sorted(set(_texts_on_screen(window)) & only_chinese)
    assert not left_behind, left_behind
    dialog.deleteLater()


def _texts_on_screen(root: QObject) -> list[str]:
    """Return every piece of text the window is showing, in no order."""
    found: list[str] = []
    for widget in (root, *root.findChildren(QObject)):
        if isinstance(widget, QLabel | QAbstractButton | QAction):
            found.append(widget.text())
        if isinstance(widget, QWidget):
            found.append(widget.toolTip())
            found.append(widget.windowTitle())
        if isinstance(widget, QMenu):
            found.append(widget.title())
        if isinstance(widget, QLineEdit):
            found.append(widget.placeholderText())
        if isinstance(widget, QComboBox):
            found.extend(widget.itemText(index) for index in range(widget.count()))
        if isinstance(widget, QListWidget):
            found.extend(widget.item(index).text() for index in range(widget.count()))
    return found


# -- the settings dialog, where the language is changed -------------------


def test_the_settings_dialog_follows_while_it_is_open(window: MainWindow) -> None:
    """The dialog the language was changed in has to change with it."""
    english = Translator(LANGUAGE_EN)
    dialog = SettingsDialog(window._settings, window.controller, window)
    dialog.changed.connect(window.apply_settings)

    dialog.language_box.setCurrentIndex(dialog.language_box.findData(LANGUAGE_EN))

    assert window._settings.language == LANGUAGE_EN
    assert dialog.windowTitle() == english.tr("settings.title")
    assert dialog.categories.item(0).text() == english.tr("settings.category.general")
    assert dialog.language_box.currentData() == LANGUAGE_EN
    assert dialog.language_box.itemText(
        dialog.language_box.findData(LANGUAGE_EN)
    ) == english.tr("settings.language.en")
    assert dialog.clipping_choice.buttons()[
        "coloured"
    ].text() == english.tr("settings.clipping.coloured")
    dialog.deleteLater()


def test_the_shortcut_list_can_still_be_searched_after_a_change(
    window: MainWindow,
) -> None:
    english = Translator(LANGUAGE_EN)
    dialog = SettingsDialog(window._settings, window.controller, window)
    dialog.changed.connect(window.apply_settings)

    dialog.language_box.setCurrentIndex(dialog.language_box.findData(LANGUAGE_EN))
    dialog.search_field.setText(english.tr("toolbar.export").lower())

    shown = [row for row, _ in dialog._shortcut_rows if not row.isHidden()]
    assert shown
    dialog.deleteLater()
