"""Tests for the window layout and the collapsible panels (PLAN.md 3.1).

The window is rendered on Qt's offscreen platform and then measured, so the
region sizes and the colour of each region are checked against the tokens in
``ui/theme.py`` rather than by eye.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from app.i18n import LANGUAGE_EN, LANGUAGE_ZH_TW, Translator, tr
from app.settings import Settings, WindowState

pytest.importorskip("PySide6.QtWidgets")

from ui import theme  # noqa: E402
from core.geometry import Rect  # noqa: E402
from ui.filmstrip import Thumbnail  # noqa: E402
from ui.inspector import InspectorTab  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402

WINDOW_WIDTH = 1440
WINDOW_HEIGHT = 900


@pytest.fixture
def window(qt_app: object) -> Iterator[MainWindow]:
    """Return a shown main window sized for the measurements below."""
    settings = Settings(
        window=WindowState(x=0, y=0, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    )
    main_window = MainWindow(settings)
    main_window.show()
    main_window.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
    qt_app.processEvents()  # type: ignore[attr-defined]
    yield main_window
    main_window.hide()
    main_window.deleteLater()


def _open_photos(window: MainWindow, tmp_path: object, qt_app: object) -> object:
    """Open a folder with one photo, which is when the panels appear.

    Before a photo is open the side panel and the filmstrip stay out of the
    way (PLAN.md 3.7: the welcome screen is the whole window), so a test
    about them has to put the window into its working state first.
    """
    from PIL import Image

    folder = tmp_path / "photos"  # type: ignore[operator]
    if not folder.is_dir():
        folder.mkdir()
        Image.new("RGB", (200, 150), (80, 120, 160)).save(folder / "a.jpg", "JPEG")
    _open_and_wait(window, folder, qt_app)
    # The panels slide back in over the documented 160 ms, so the test has
    # to let that finish before measuring anything.
    _settle(qt_app, theme.DURATION_PANEL_MS + 120)
    return folder


def _open_and_wait(window: MainWindow, path: Path, qt_app: object) -> None:
    """Open *path* and wait for its folder to be listed, which happens in the background."""
    import time

    window.controller.open(path)
    deadline = time.perf_counter() + 10.0
    while not window.controller.photos and time.perf_counter() < deadline:
        qt_app.processEvents()  # type: ignore[attr-defined]
    qt_app.processEvents()  # type: ignore[attr-defined]


def _settle(qt_app: object, milliseconds: int) -> None:
    """Let animations and queued work finish."""
    from PySide6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()
    qt_app.processEvents()  # type: ignore[attr-defined]


def _pixel(window: MainWindow, x: int, y: int) -> str:
    return window.grab().toImage().pixelColor(x, y).name().upper()


#: Warnings produced by the test platform itself rather than by this code.
_PLATFORM_WARNINGS = (
    "This plugin does not support",
    "QFontDatabase: Cannot find font directory",
)


@contextmanager
def collect_qt_warnings() -> Iterator[list[str]]:
    """Collect the Qt warnings raised inside the block."""
    from PySide6.QtCore import QtMsgType, qInstallMessageHandler

    warnings: list[str] = []

    def handler(mode, context, message) -> None:  # noqa: ANN001 - Qt callback
        if mode is QtMsgType.QtWarningMsg and not message.startswith(_PLATFORM_WARNINGS):
            warnings.append(message)

    previous = qInstallMessageHandler(handler)
    try:
        yield warnings
    finally:
        qInstallMessageHandler(previous)


# -- layout ---------------------------------------------------------------


def test_the_four_regions_have_the_documented_sizes(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)
    assert window.toolbar.height() == theme.TOOLBAR_HEIGHT
    assert window.inspector.width() == theme.PANEL_WIDTH
    assert window.filmstrip.height() == theme.FILMSTRIP_HEIGHT
    assert window.preview.width() == WINDOW_WIDTH - theme.PANEL_WIDTH


def test_each_region_paints_its_own_background_colour(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    # Before a photo there are only two regions: the toolbar and the canvas
    # the welcome screen sits on. The toolbar is read a quarter of the way
    # along, between the buttons on the left and the title in the middle:
    # its very centre lands on whichever letter of the title is there.
    assert _pixel(window, WINDOW_WIDTH // 4, theme.TOOLBAR_HEIGHT // 2) == theme.BG_APP
    assert _pixel(window, 300, 600) == theme.BG_CANVAS

    _open_photos(window, tmp_path, qt_app)
    # Down the panel's own margin, where no control can ever reach: a point
    # out in the middle of it lands on whichever slider the layout has put
    # there, and says nothing about the background.
    panel_margin = WINDOW_WIDTH - theme.PANEL_WIDTH + theme.PANEL_PADDING // 2
    assert _pixel(window, panel_margin, 600) == theme.BG_PANEL
    assert _pixel(window, 600, WINDOW_HEIGHT - 50) == theme.BG_PANEL


def test_the_welcome_screen_is_shown_before_a_photo_is_open(window: MainWindow) -> None:
    assert window.preview.currentWidget() is window.preview.welcome
    # The welcome screen must not paint a lighter block over the canvas.
    assert _pixel(window, 600, 300) == theme.BG_CANVAS


def test_nothing_that_needs_a_photo_is_enabled(window: MainWindow) -> None:
    assert not window.toolbar.undo_button.isEnabled()
    assert not window.toolbar.redo_button.isEnabled()
    assert not window.toolbar.export_button.isEnabled()
    for tab in InspectorTab:
        for button in window.inspector.action_buttons(tab):
            assert not button.isEnabled()


def test_building_and_using_the_window_raises_no_qt_warning(qt_app: object) -> None:
    """Qt warnings mean the interface is misusing an API, so there must be none."""
    with collect_qt_warnings() as warnings:
        window = MainWindow(Settings())
        window.show()
        qt_app.processEvents()  # type: ignore[attr-defined]
        window.toolbar.menu_button.click()
        qt_app.processEvents()  # type: ignore[attr-defined]
        window.toolbar.menu.close()
        window.hide()
        window.deleteLater()
    assert warnings == []


def test_the_application_menu_opens_below_its_button(
    window: MainWindow, qt_app: object
) -> None:
    button = window.toolbar.menu_button
    window.toolbar.menu_button.click()
    qt_app.processEvents()  # type: ignore[attr-defined]
    try:
        assert window.toolbar.menu.isVisible()
        button_bottom = button.mapToGlobal(button.rect().bottomLeft()).y()
        assert window.toolbar.menu.geometry().top() >= button_bottom
    finally:
        window.toolbar.menu.close()


def test_every_icon_button_has_a_tooltip(window: MainWindow) -> None:
    for button in (
        window.toolbar.menu_button,
        window.toolbar.undo_button,
        window.toolbar.redo_button,
        window.toolbar.export_button,
    ):
        assert button.toolTip()


def test_icon_button_tooltips_name_their_shortcut(window: MainWindow) -> None:
    translator = Translator(LANGUAGE_ZH_TW)
    assert translator.tr("shortcut.undo") in window.toolbar.undo_button.toolTip()
    assert translator.tr("shortcut.export") in window.toolbar.export_button.toolTip()


# -- collapsing -----------------------------------------------------------


def test_collapsing_the_inspector_gives_its_width_to_the_preview(
    window: MainWindow, qt_app: object
) -> None:
    window.set_inspector_collapsed(True, animate=False)
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert window.is_inspector_collapsed()
    assert window.inspector.isHidden()
    assert window.preview.width() == WINDOW_WIDTH

    window.set_inspector_collapsed(False, animate=False)
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert not window.is_inspector_collapsed()
    assert window.inspector.width() == theme.PANEL_WIDTH


def test_collapsing_the_filmstrip_restores_its_height(
    window: MainWindow, qt_app: object
) -> None:
    window.set_filmstrip_collapsed(True, animate=False)
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert window.is_filmstrip_collapsed()
    assert window.filmstrip.isHidden()

    window.set_filmstrip_collapsed(False, animate=False)
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert window.filmstrip.height() == theme.FILMSTRIP_HEIGHT


def _tab_bottom(window: MainWindow) -> int:
    """Return the y just below the tab, in the window's central area."""
    tab = window.filmstrip_tab
    return tab.y() + tab.height()


def test_the_filmstrip_tab_stands_on_the_strip_at_its_left_end(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    """PLAN.md 3.1: a bookmark above the first thumbnail, joined to the strip."""
    _open_photos(window, tmp_path, qt_app)
    tab = window.filmstrip_tab
    assert tab.isVisible()
    assert tab.size().width() == theme.FILMSTRIP_TAB_WIDTH
    assert tab.size().height() == theme.FILMSTRIP_TAB_HEIGHT
    assert tab.x() == window.filmstrip.x() + theme.SPACE_MD
    # It covers the strip's top line, which is what joins the two.
    assert _tab_bottom(window) == window.filmstrip.y() + theme.BORDER_WIDTH
    assert not tab.collapsed


def test_the_filmstrip_tab_keeps_clear_of_the_view_bar(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)
    window.resize(theme.WINDOW_MIN_WIDTH, theme.WINDOW_MIN_HEIGHT)
    _settle(qt_app, 50)
    central = window.centralWidget()
    bar = window.preview.view_bar
    bar_left = bar.mapTo(central, bar.rect().topLeft()).x()
    tab = window.filmstrip_tab
    assert tab.x() + tab.width() + theme.SPACE_LG <= bar_left


def test_the_filmstrip_tab_folds_the_strip_away_and_back(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)
    tab = window.filmstrip_tab

    tab.click()
    _settle(qt_app, theme.DURATION_PANEL_MS + 120)
    assert window.is_filmstrip_collapsed()
    assert window.filmstrip.isHidden()
    # The tab waits at the bottom of the window, ready to be pressed.
    assert tab.isVisible()
    assert _tab_bottom(window) == window.centralWidget().height()
    assert tab.collapsed
    assert tr("filmstrip.expand") in tab.toolTip()

    tab.click()
    _settle(qt_app, theme.DURATION_PANEL_MS + 120)
    assert not window.is_filmstrip_collapsed()
    assert window.filmstrip.height() == theme.FILMSTRIP_HEIGHT
    assert _tab_bottom(window) == window.filmstrip.y() + theme.BORDER_WIDTH
    assert tr("filmstrip.collapse") in tab.toolTip()
    # Remembered between runs like F6 (U-04).
    window._store_state()
    assert not window._settings.filmstrip_collapsed


def test_the_filmstrip_tab_rides_on_the_strip_while_it_folds(
    window: MainWindow, tmp_path: object, qt_app: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Halfway through the fold the tab is still on the strip's top edge."""
    from ui import native

    monkeypatch.setattr(native, "animations_enabled", lambda: True)
    _open_photos(window, tmp_path, qt_app)
    window.toggle_filmstrip()
    _settle(qt_app, theme.DURATION_PANEL_MS // 3)
    strip = window.filmstrip
    assert 0 < strip.height() < theme.FILMSTRIP_HEIGHT
    assert _tab_bottom(window) == strip.y() + theme.BORDER_WIDTH


def test_the_filmstrip_tab_names_its_shortcut(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    """PLAN.md 3.8: a control that has a key says which in its tooltip."""
    _open_photos(window, tmp_path, qt_app)
    assert tr("shortcut.toggle_filmstrip") in window.filmstrip_tab.toolTip()


def test_the_filmstrip_tab_is_not_there_before_a_folder_is_open(
    window: MainWindow, qt_app: object
) -> None:
    """PLAN.md 3.7: the welcome screen is the whole window, tab and all."""
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert window.filmstrip_tab.isHidden()


def test_focus_mode_hides_and_restores_both_panels(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)
    # Tab collapses both panels; the widgets stay visible while the 160 ms
    # animation runs, so the collapsed state is what is checked here.
    window.toggle_focus_mode()
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert window.is_inspector_collapsed() and window.is_filmstrip_collapsed()

    window.toggle_focus_mode()
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert not window.is_inspector_collapsed()
    assert not window.is_filmstrip_collapsed()
    assert not window.inspector.isHidden() and not window.filmstrip.isHidden()


def test_a_tab_shortcut_opens_the_panel_and_remembers_it(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    """Ctrl+1 to Ctrl+5 open the panel the tab is in (PLAN.md section 8).

    What the user wants has to move with it: left behind, the panel would
    fold itself away again at the next folder opened, and the key that
    collapses it would do nothing the first time it was pressed.
    """
    _open_photos(window, tmp_path, qt_app)
    window.toggle_inspector()
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert window.is_inspector_collapsed()

    window.shortcut_handlers()["shortcut.tab_looks"]()
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert not window.is_inspector_collapsed()
    assert window.inspector.current_tab is InspectorTab.LOOKS

    window.toggle_inspector()
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert window.is_inspector_collapsed()


def test_switching_tab_with_the_focus_in_the_panel_keeps_the_panels(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    """Switching tabs is not Tab, however Qt gets there (PLAN.md section 8).

    A click on a control in the panel leaves the focus on the page the next
    tab switch hides, and Qt hands that focus on through
    ``focusNextPrevChild`` - the call Tab also arrives through. Read as Tab,
    every tab switch collapsed both panels and let them straight back, so
    the whole window appeared to zoom once per switch.
    """
    from ui.widgets.group import GroupHeader

    _open_photos(window, tmp_path, qt_app)
    header = window.inspector.adjust_tab.findChild(GroupHeader)
    assert header is not None, "the adjust tab has no group to click on"
    header.setFocus()
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert window.inspector.isAncestorOf(qt_app.focusWidget())  # type: ignore[attr-defined]

    window.inspector.set_current_tab(InspectorTab.CROP)
    qt_app.processEvents()  # type: ignore[attr-defined]

    assert not window.is_inspector_collapsed()
    assert not window.is_filmstrip_collapsed()
    assert window.inspector.width() == theme.PANEL_WIDTH


def test_a_tab_shortcut_leaves_the_welcome_screen_whole(
    window: MainWindow, qt_app: object
) -> None:
    """PLAN.md 3.7: with nothing open the welcome screen is the whole window."""
    assert window.is_inspector_collapsed()

    window.shortcut_handlers()["shortcut.tab_crop"]()
    qt_app.processEvents()  # type: ignore[attr-defined]

    assert window.is_inspector_collapsed()


def test_the_window_geometry_survives_repeated_restarts(
    qt_app: object, tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closing and reopening must land on exactly the same spot.

    pos()/move() work in frame coordinates and size()/resize() in client
    coordinates; mixing the two makes a real window creep down the screen by
    the height of its title bar on every restart.
    """
    path = tmp_path / "settings.json"  # type: ignore[operator]
    monkeypatch.setattr("app.settings.settings_file", lambda: path)

    first = MainWindow(Settings())
    first.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    first.move(300, 200)
    first.resize(1100, 700)
    qt_app.processEvents()  # type: ignore[attr-defined]
    expected_position = (first.pos().x(), first.pos().y())
    expected_size = (first.width(), first.height())
    first.close()
    qt_app.processEvents()  # type: ignore[attr-defined]

    for restart in (1, 2):
        again = MainWindow(Settings.load(path))
        again.show()
        qt_app.processEvents()  # type: ignore[attr-defined]
        assert (again.pos().x(), again.pos().y()) == expected_position, restart
        assert (again.width(), again.height()) == expected_size, restart
        again.close()
        qt_app.processEvents()  # type: ignore[attr-defined]


def test_the_collapsed_state_is_remembered(qt_app: object, tmp_path: object) -> None:
    settings = Settings(inspector_collapsed=True, filmstrip_collapsed=False)
    window = MainWindow(settings)
    window.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    _open_photos(window, tmp_path, qt_app)
    try:
        assert window.is_inspector_collapsed()
        assert window.inspector.isHidden()
        assert not window.is_filmstrip_collapsed()
        assert not window.filmstrip.isHidden()
    finally:
        window.hide()
        window.deleteLater()


# -- panel tabs -----------------------------------------------------------


def test_the_adjust_tab_is_selected_at_start(window: MainWindow) -> None:
    assert window.inspector.current_tab is InspectorTab.ADJUST


def test_the_current_tab_is_marked_by_a_white_underline(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    """The underline must be visible, not clipped by the tab bar (PLAN.md 3.1)."""
    _open_photos(window, tmp_path, qt_app)
    inspector = window.inspector
    left = inspector.mapTo(window, inspector.rect().topLeft())
    underline_y = left.y() + theme.HISTOGRAM_HEIGHT + theme.TAB_BAR_HEIGHT - 1
    tab_width = theme.PANEL_WIDTH // len(InspectorTab)
    centres = [left.x() + tab_width * index + tab_width // 2 for index in InspectorTab]

    for selected in InspectorTab:
        inspector.set_current_tab(selected)
        qt_app.processEvents()  # type: ignore[attr-defined]
        for tab, x in zip(InspectorTab, centres):
            expected = theme.TEXT_PRIMARY if tab is selected else theme.BORDER
            assert _pixel(window, x, underline_y) == expected, (selected, tab)


def test_switching_tab_changes_the_page_and_the_action_row(window: MainWindow) -> None:
    inspector = window.inspector
    inspector.set_current_tab(InspectorTab.CROP)
    assert inspector.current_tab is InspectorTab.CROP
    assert len(inspector.action_buttons(InspectorTab.CROP)) == 3
    inspector.set_current_tab(InspectorTab.LOOKS)
    assert inspector.current_tab is InspectorTab.LOOKS
    assert inspector.action_buttons(InspectorTab.LOOKS) == ()


# -- opening photos -------------------------------------------------------


def _drop_event(paths: list[Path]):
    """Build a drag carrying *paths*, the way Explorer does."""
    from PySide6.QtCore import QMimeData, QUrl
    from PySide6.QtGui import QDropEvent

    data = QMimeData()
    data.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    return data


def test_dropping_photos_and_folders_is_accepted(window: MainWindow, tmp_path: object) -> None:
    """PLAN.md 3.7: photos and folders dropped on the window are opened."""
    from ui.main_window import _dropped_paths

    photo = tmp_path / "a.jpg"  # type: ignore[operator]
    photo.write_bytes(b"")
    folder = tmp_path / "folder"  # type: ignore[operator]
    folder.mkdir()

    class _Event:
        def __init__(self, data):
            self._data = data

        def mimeData(self):  # noqa: N802 - matches Qt
            return self._data

    accepted = _dropped_paths(_Event(_drop_event([photo, folder])))
    assert set(accepted) == {photo, folder}


def test_dropping_something_that_is_not_a_photo_is_ignored(
    window: MainWindow, tmp_path: object
) -> None:
    from ui.main_window import _dropped_paths

    other = tmp_path / "notes.txt"  # type: ignore[operator]
    other.write_text("hello", encoding="utf-8")

    class _Event:
        def __init__(self, data):
            self._data = data

        def mimeData(self):  # noqa: N802 - matches Qt
            return self._data

    assert _dropped_paths(_Event(_drop_event([other]))) == []


def test_dropping_several_photos_opens_all_of_them(
    window: MainWindow, tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-01: every photo dropped is opened, not just the first."""
    first = tmp_path / "a.jpg"  # type: ignore[operator]
    second = tmp_path / "b.jpg"  # type: ignore[operator]
    first.write_bytes(b"")
    second.write_bytes(b"")
    opened: list[list[Path]] = []
    monkeypatch.setattr(window.controller, "open_paths", lambda paths: opened.append(list(paths)))

    class _Event:
        def __init__(self, data):
            self._data = data

        def mimeData(self):  # noqa: N802 - matches Qt
            return self._data

        def acceptProposedAction(self) -> None:  # noqa: N802 - matches Qt
            pass

    window.dropEvent(_Event(_drop_event([first, second])))
    assert opened == [[first, second]]


def test_the_open_dialog_takes_several_photos(
    window: MainWindow, tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dialog offers more than one photo and starts where the last one was."""
    from PySide6.QtWidgets import QFileDialog

    window._settings.remember_browse_folder(str(tmp_path))
    asked: list[str] = []
    chosen = [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")]  # type: ignore[operator]

    def get_open_file_names(_parent, _title, start, _filter):
        asked.append(start)
        return chosen, ""

    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(get_open_file_names))
    opened: list[list[Path]] = []
    monkeypatch.setattr(window.controller, "open_paths", lambda paths: opened.append(list(paths)))

    window.open_file_dialog()

    assert asked == [str(tmp_path)]
    assert opened == [[Path(path) for path in chosen]]


def test_opening_a_folder_fills_the_window(window: MainWindow, tmp_path: object, qt_app: object) -> None:
    """The toolbar, the filmstrip and the preview all follow the controller."""
    from PIL import Image

    folder = tmp_path / "photos"  # type: ignore[operator]
    folder.mkdir()
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (200, 150), (80, 120, 160)).save(folder / name, "JPEG")

    _open_and_wait(window, folder, qt_app)

    assert "a.jpg" in window.toolbar.title_label.text()
    assert folder.name in window.toolbar.title_label.text()
    assert len(window.controller.photos) == 2


def test_the_welcome_screen_can_be_used_once_the_window_is_wired(
    window: MainWindow, qt_app: object
) -> None:
    welcome = window.preview.welcome
    # The menu is reachable from the moment the window is up; the screen's
    # own button waits for the opening to put it there (PLAN.md 3.7), so
    # that a click in the first 1320 ms cannot land on a button nobody can
    # see yet.
    assert window.toolbar.menu.open_file_action.isEnabled()
    assert window.toolbar.menu.open_folder_action.isEnabled()
    assert not any(button.isEnabled() for button in welcome.buttons())

    from ui.welcome_view import opening_length

    _settle(qt_app, opening_length() + theme.DURATION_PHOTO_FADE_MS)
    assert all(button.isEnabled() for button in welcome.buttons())


# -- language -------------------------------------------------------------


def test_the_window_is_built_in_the_selected_language(qt_app: object) -> None:
    from app import i18n

    i18n.init_translator(LANGUAGE_EN)
    try:
        window = MainWindow(Settings(language=LANGUAGE_EN))
        expected = Translator(LANGUAGE_EN).tr("toolbar.export")
        assert window.toolbar.export_button.text() == expected
        window.deleteLater()
    finally:
        i18n.init_translator(LANGUAGE_ZH_TW)


# -- editing, history and before / after (PLAN.md 3.1, 3.3, E-04 to E-06) --


def test_undo_and_redo_follow_the_history(window: MainWindow) -> None:
    assert not window.toolbar.undo_button.isEnabled()
    assert not window.toolbar.redo_button.isEnabled()

    window.controller.history_changed.emit(True, False)
    assert window.toolbar.undo_button.isEnabled()
    assert not window.toolbar.redo_button.isEnabled()

    window.controller.history_changed.emit(False, True)
    assert not window.toolbar.undo_button.isEnabled()
    assert window.toolbar.redo_button.isEnabled()


def test_the_undo_and_redo_buttons_ask_the_controller(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[str] = []
    monkeypatch.setattr(window.controller, "undo", lambda: called.append("undo"))
    monkeypatch.setattr(window.controller, "redo", lambda: called.append("redo"))
    window.toolbar.undo_button.setEnabled(True)
    window.toolbar.redo_button.setEnabled(True)

    window.toolbar.undo_button.click()
    window.toolbar.redo_button.click()

    assert called == ["undo", "redo"]


def test_the_reset_button_goes_through_the_controller(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E-05: the controller owns the state the photo was found in."""
    called: list[str] = []
    monkeypatch.setattr(window.controller, "reset", lambda: called.append("reset"))
    button = window.inspector.action_buttons(InspectorTab.ADJUST)[-1]
    button.setEnabled(True)

    button.click()

    assert called == ["reset"]


def test_holding_the_before_and_after_key_asks_for_the_original(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E-06: held down, not toggled (PLAN.md section 8)."""
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    asked: list[bool] = []
    monkeypatch.setattr(window.controller, "set_comparing", asked.append)

    for event_type in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
        window.event(
            QKeyEvent(event_type, Qt.Key.Key_Backslash, Qt.KeyboardModifier.NoModifier)
        )

    assert asked == [True, False]


def test_an_auto_repeated_key_is_ignored(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Holding the key sends repeats; they must not toggle the view."""
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    asked: list[bool] = []
    monkeypatch.setattr(window.controller, "set_comparing", asked.append)

    window.event(
        QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Backslash,
            Qt.KeyboardModifier.NoModifier,
            "\\",
            True,
        )
    )

    assert asked == []


def test_the_compare_button_is_held_down_rather_than_clicked(window: MainWindow) -> None:
    """PLAN.md 3.3: the view bar button works like the keyboard shortcut."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    asked: list[bool] = []
    window.preview.view_bar.comparing.connect(asked.append)
    button = window.preview.view_bar.compare_button

    QTest.mousePress(button, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(button.width() // 2, button.height() // 2))
    assert asked == [True]
    QTest.mouseRelease(button, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                       QPoint(button.width() // 2, button.height() // 2))
    assert asked == [True, False]


def test_the_before_label_is_shown_only_while_the_original_is(
    window: MainWindow,
) -> None:
    """PLAN.md 3.3: 顯示原圖時預覽區左上角出現「修圖前」標籤."""
    import numpy as np

    from app.controller import Preview

    photo = Path("D:/Photos/DSC01234.ARW")
    image = np.full((60, 90, 3), 100, dtype=np.uint8)
    assert not window.preview.before_label.isVisible()

    window._on_preview_ready(Preview(path=photo, image=image, provisional=False, before=True))
    assert window.preview.before_label.isVisible()
    assert window.preview.before_label.text() == tr("preview.before")

    window._on_preview_ready(Preview(path=photo, image=image, provisional=False))
    assert not window.preview.before_label.isVisible()


def test_the_histogram_keeps_describing_the_edit_while_comparing(
    window: MainWindow,
) -> None:
    """The histogram is a reference for the edit, so it stays put (E-06)."""
    import numpy as np

    from app.controller import Preview

    photo = Path("D:/Photos/DSC01234.ARW")
    edited = np.full((60, 90, 3), 200, dtype=np.uint8)
    window._on_preview_ready(Preview(path=photo, image=edited, provisional=False))
    counts = window.inspector.histogram._counts

    window._on_preview_ready(
        Preview(
            path=photo,
            image=np.full((60, 90, 3), 20, dtype=np.uint8),
            provisional=False,
            before=True,
        )
    )

    assert window.inspector.histogram._counts is counts


def test_the_histogram_mode_is_remembered_between_runs(qt_app: object) -> None:
    """The way the histogram is drawn comes back the way it was left (A-12)."""
    from app.settings import HISTOGRAM_LUMA, HISTOGRAM_RGB

    settings = Settings(histogram_mode=HISTOGRAM_LUMA)
    main_window = MainWindow(settings)
    try:
        assert main_window.inspector.histogram.mode == HISTOGRAM_LUMA
        main_window.inspector.histogram.toggle_mode()
        main_window._store_state()
        assert settings.histogram_mode == HISTOGRAM_RGB
    finally:
        main_window.deleteLater()


def test_the_info_card_says_whether_the_photo_has_been_edited(
    window: MainWindow,
) -> None:
    """PLAN.md 3.9: 檔案 / 編輯狀態 - 已編輯（有參數檔）或未編輯."""
    row = window.info_card._rows["info.edit_state"]

    window.info_card.set_edited(True)
    assert row.value.text() == tr("info.edited")

    window.info_card.set_edited(False)
    assert row.value.text() == tr("info.not_edited")


def test_a_photo_being_marked_as_edited_reaches_both_places(
    window: MainWindow,
) -> None:
    photo = Path("D:/Photos/DSC01234.ARW")
    window.filmstrip.set_photos([photo])

    window.controller.edited_changed.emit(photo, True)

    thumbnails = window.filmstrip.findChildren(Thumbnail)
    assert thumbnails and thumbnails[0].edited


# -- cropping in the window (PLAN.md 3.2, C-01 to C-07) -------------------


def _window_with_photo(window: MainWindow, tmp_path, qt_app) -> MainWindow:
    """Open a folder in *window* and wait until its photo can be cropped."""
    import time

    from PIL import Image

    folder = tmp_path / "crop_photos"
    folder.mkdir()
    Image.new("RGB", (240, 160), (90, 130, 170)).save(folder / "a.jpg", "JPEG")

    _open_and_wait(window, folder, qt_app)
    deadline = time.perf_counter() + 20.0
    while time.perf_counter() < deadline:
        qt_app.processEvents()
        if window.controller.photo_size != (1, 1):
            break
        time.sleep(0.01)
    assert window.controller.photo_size != (1, 1), "the photo never arrived"
    return window


def _wait_for_photo_size(window: MainWindow, qt_app: object, size: tuple[int, int]) -> None:
    """Wait until the photo the controller has decoded measures *size*."""
    import time

    deadline = time.perf_counter() + 20.0
    while time.perf_counter() < deadline:
        qt_app.processEvents()  # type: ignore[attr-defined]
        if window.controller.photo_size == size:
            return
        time.sleep(0.01)
    raise AssertionError(f"the photo {size} never arrived")


def test_the_crop_box_is_only_shown_in_the_crop_tab(
    window: MainWindow, tmp_path, qt_app: object
) -> None:
    _window_with_photo(window, tmp_path, qt_app)
    assert not window.preview.canvas.crop.active

    window.inspector.set_current_tab(InspectorTab.CROP)
    assert window.preview.canvas.crop.active
    assert window.controller.cropping

    window.inspector.set_current_tab(InspectorTab.ADJUST)
    assert not window.preview.canvas.crop.active
    assert not window.controller.cropping


def test_the_crop_shortcut_goes_in_and_out(
    window: MainWindow, tmp_path, qt_app: object
) -> None:
    """PLAN.md section 8: R enters and leaves the crop tab."""
    _window_with_photo(window, tmp_path, qt_app)

    window.toggle_crop()
    assert window.inspector.current_tab is InspectorTab.CROP

    window.toggle_crop()
    assert window.inspector.current_tab is InspectorTab.ADJUST


def test_cancelling_leaves_the_tab_and_puts_the_crop_back(
    window: MainWindow, tmp_path, qt_app: object
) -> None:
    _window_with_photo(window, tmp_path, qt_app)
    window.toggle_crop()
    kept = window.controller.geometry

    window._on_crop_dragged(Rect(0.2, 0.2, 0.4, 0.4), False)
    assert window.controller.geometry != kept

    window.cancel_crop()

    assert window.inspector.current_tab is InspectorTab.ADJUST
    assert window.controller.geometry == kept


def test_cancelling_does_not_go_back_into_crop_mode_first(
    window: MainWindow, tmp_path, qt_app: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Esc asks for the photo once, not three times (PLAN.md 3.2).

    Putting the crop back announces the geometry while the crop tab is still
    the tab on screen. Read as a photo arriving in that tab, the
    announcement put the controller straight back into crop mode, so the
    whole photo was developed twice over before leaving the tab developed
    the cropped one.
    """
    _window_with_photo(window, tmp_path, qt_app)
    window.toggle_crop()
    window._on_crop_dragged(Rect(0.2, 0.2, 0.4, 0.4), False)

    asked: list[bool] = []
    original = window.controller.set_cropping

    def record(cropping: bool) -> None:
        asked.append(cropping)
        original(cropping)

    monkeypatch.setattr(window.controller, "set_cropping", record)
    window.cancel_crop()

    assert asked == [False]
    assert not window.controller.cropping


def test_a_photo_that_arrives_in_the_crop_tab_is_cropped_again(
    window: MainWindow, tmp_path, qt_app: object
) -> None:
    """Moving to a photo drops crop mode; the tab on screen puts it back.

    The controller leaves crop mode for the photo being left behind, and the
    new one may arrive a second later; the crop tab is still open, so the
    whole photo is what belongs on screen (PLAN.md 3.2).
    """
    from PIL import Image

    folder = tmp_path / "two_crop_photos"
    folder.mkdir()
    Image.new("RGB", (240, 160), (90, 130, 170)).save(folder / "a.jpg", "JPEG")
    Image.new("RGB", (200, 300), (170, 120, 90)).save(folder / "b.jpg", "JPEG")
    _open_and_wait(window, folder, qt_app)
    _wait_for_photo_size(window, qt_app, (240, 160))

    window.inspector.set_current_tab(InspectorTab.CROP)
    assert window.controller.cropping

    window.controller.select_next()
    _wait_for_photo_size(window, qt_app, (200, 300))

    assert window.controller.cropping
    assert window.preview.canvas.crop.active


def test_finishing_keeps_the_crop(window: MainWindow, tmp_path, qt_app: object) -> None:
    _window_with_photo(window, tmp_path, qt_app)
    window.toggle_crop()

    window._on_crop_dragged(Rect(0.2, 0.2, 0.4, 0.4), False)
    window.finish_crop()

    assert window.inspector.current_tab is InspectorTab.ADJUST
    assert window.controller.geometry.crop.w == pytest.approx(0.4, abs=0.01)


def test_choosing_a_ratio_reshapes_the_crop(
    window: MainWindow, tmp_path, qt_app: object
) -> None:
    """C-02, through the whole chain: button, geometry, controller, box."""
    _window_with_photo(window, tmp_path, qt_app)
    window.toggle_crop()

    window.inspector.crop_tab.aspect_buttons()["1:1"].click()

    crop = window.controller.geometry.crop
    source = window.controller.photo_size
    assert window.controller.geometry.aspect == "1:1"
    assert crop.w * source[0] == pytest.approx(crop.h * source[1], rel=0.01)
    assert window.preview.canvas.crop.crop == crop


def test_turning_the_photo_from_the_keyboard(
    window: MainWindow, tmp_path, qt_app: object
) -> None:
    """C-06: Ctrl+[ and Ctrl+] turn the photo either way."""
    _window_with_photo(window, tmp_path, qt_app)

    window.turn_right()
    assert window.controller.geometry.quarter_turns == 1

    window.turn_left()
    assert window.controller.geometry.quarter_turns == 0


def test_flipping_the_photo_from_the_keyboard(
    window: MainWindow, tmp_path, qt_app: object
) -> None:
    """C-08: Shift+H flips left to right, Shift+V top to bottom, and back."""
    _window_with_photo(window, tmp_path, qt_app)

    window.mirror_horizontally()
    assert window.controller.geometry.mirrored
    assert window.controller.geometry.quarter_turns == 0

    window.mirror_vertically()
    assert not window.controller.geometry.mirrored
    assert window.controller.geometry.quarter_turns == 2

    window.mirror_vertically()
    window.mirror_horizontally()
    assert window.controller.geometry.is_default


def test_the_grid_shortcut_only_works_in_the_crop_tab(
    window: MainWindow, tmp_path, qt_app: object
) -> None:
    _window_with_photo(window, tmp_path, qt_app)
    window.toggle_crop_grid()
    assert not window.preview.canvas.crop.grid_visible

    window.toggle_crop()
    window.toggle_crop_grid()
    assert window.preview.canvas.crop.grid_visible


def test_the_straighten_tool_is_disarmed_on_the_way_out(
    window: MainWindow, tmp_path, qt_app: object
) -> None:
    _window_with_photo(window, tmp_path, qt_app)
    window.toggle_crop()
    window.inspector.crop_tab.straighten_button.click()
    assert window.preview.canvas.crop.straightening

    window.finish_crop()

    assert not window.preview.canvas.crop.straightening
    assert not window.inspector.crop_tab.straighten_button.isChecked()


def test_the_crop_tab_shows_what_the_controller_settled_on(
    window: MainWindow, tmp_path, qt_app: object
) -> None:
    _window_with_photo(window, tmp_path, qt_app)
    window.toggle_crop()

    window._on_angle_changed(7.5, False)

    assert window.inspector.crop_tab.angle_row.value == pytest.approx(7.5)
    assert window.preview.canvas.crop.crop == window.controller.geometry.crop


# -- exporting (PLAN.md 2.6, 3.11) ---------------------------------------


def _folder_with_a_photo(tmp_path: object) -> object:
    """Write one small JPEG and return the folder it is in."""
    from PIL import Image

    folder = tmp_path / "photos"  # type: ignore[operator]
    folder.mkdir()
    Image.new("RGB", (200, 150), (80, 120, 160)).save(folder / "a.jpg", "JPEG")
    return folder


def test_the_export_button_waits_for_a_photo(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    assert not window.toolbar.export_button.isEnabled()

    _open_and_wait(window, _folder_with_a_photo(tmp_path), qt_app)

    assert window.toolbar.export_button.isEnabled()


def test_the_progress_appears_on_the_export_button(window: MainWindow) -> None:
    """X-07: no window over the photo, the button says how far it has got."""
    window.controller.export_progress.emit(0.5)

    assert window.toolbar.export_button.progress == pytest.approx(0.5)
    assert window.toolbar.export_button.text() == tr("toolbar.exporting")


def test_a_finished_export_offers_to_show_the_file(
    window: MainWindow, tmp_path: object
) -> None:
    from core.exporter import ExportResult

    written = tmp_path / "a_edit.jpg"  # type: ignore[operator]
    window.controller.export_finished.emit(ExportResult(path=written, copied=False))

    assert window.toast.isVisible()
    assert "a_edit.jpg" in window.toast.findChild(type(window.toolbar.title_label)).text()
    assert window.toolbar.export_button.progress is None


def test_a_copied_export_says_so(window: MainWindow, tmp_path: object) -> None:
    """X-08: the quality was not used, so the message says what happened."""
    from core.exporter import ExportResult

    written = tmp_path / "a_edit.jpg"  # type: ignore[operator]
    window.controller.export_finished.emit(ExportResult(path=written, copied=True))

    shown = window.toast.findChild(type(window.toolbar.title_label)).text()
    assert shown == tr("export.done_copied", name="a_edit.jpg")


def test_an_export_that_failed_leaves_the_button_as_it_was(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_and_wait(window, _folder_with_a_photo(tmp_path), qt_app)
    window.controller.export_progress.emit(0.4)

    window.controller.export_failed.emit("it did not work")

    assert window.toolbar.export_button.progress is None
    assert window.toolbar.export_button.isEnabled()


def _accept_export(window: MainWindow, monkeypatch: pytest.MonkeyPatch, accept: bool) -> list:
    """Answer the export dialog at once, and keep the export from writing."""
    from ui.export_dialog import ExportDialog

    def answer(dialog: ExportDialog) -> int:
        if accept:
            dialog.accept()
        else:
            dialog.reject()
        return dialog.result()

    started: list = []
    monkeypatch.setattr(ExportDialog, "exec", answer)
    monkeypatch.setattr(
        window.controller, "export", lambda *args: started.append(args) or True
    )
    return started


def test_accepting_the_export_sends_the_photo_into_the_button(
    window: MainWindow, tmp_path: object, qt_app: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PLAN.md 3.6: the photo flies into the button, which then fills."""
    from ui import native
    from ui.widgets.buttons import ExportLook

    monkeypatch.setattr(native, "animations_enabled", lambda: True)
    _open_photos(window, tmp_path, qt_app)
    _settle(qt_app, 600)
    assert not window.preview.canvas.photo_rect().isEmpty()
    started = _accept_export(window, monkeypatch, accept=True)

    window.export()

    assert started, "the export itself does not wait for the flight"
    assert window.flight.flying
    button = window.toolbar.export_button
    assert button.look is ExportLook.LANDING
    assert not button.isEnabled()
    _settle(qt_app, theme.EXPORT_FLY_MS + 120)
    assert not window.flight.flying
    assert button.look is ExportLook.WORKING
    assert window.scrim.isHidden()
    button.rest()


def test_a_cancelled_export_sends_nothing_anywhere(
    window: MainWindow, tmp_path: object, qt_app: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ui import native
    from ui.widgets.buttons import ExportLook

    monkeypatch.setattr(native, "animations_enabled", lambda: True)
    _open_photos(window, tmp_path, qt_app)
    started = _accept_export(window, monkeypatch, accept=False)

    window.export()

    assert not started
    assert not window.flight.flying
    assert window.toolbar.export_button.look is ExportLook.REST
    assert window.toolbar.export_button.isEnabled()


def test_the_room_for_the_export_button_does_not_change_while_it_works(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    """The button widens into room kept for it, so the title never shifts."""
    _open_photos(window, tmp_path, qt_app)
    title = window.toolbar.title_label.geometry()
    window.toolbar.export_button.start()
    window.toolbar.export_button.advance(theme.DURATION_PANEL_MS * 2)
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert window.toolbar.title_label.geometry() == title
    window.toolbar.export_button.rest()


def test_the_window_behind_a_dialog_can_be_dimmed(window: MainWindow) -> None:
    """PLAN.md 3.11: the background dims while the dialog is open."""
    assert window.scrim.isHidden()

    window.scrim.cover(window)
    assert window.scrim.isVisible()
    assert window.scrim.size() == window.size()

    window.scrim.hide()
    assert window.scrim.isHidden()


# -- the rest of section 8 ------------------------------------------------


def _press(window: MainWindow, key_name: str) -> None:
    """Press the combination *key_name* stands for, as the user would."""
    from PySide6.QtGui import QKeySequence
    from PySide6.QtTest import QTest

    sequence = QKeySequence(tr(key_name))
    combination = sequence[0]
    QTest.keyClick(window, combination.key(), combination.keyboardModifiers())


def test_every_shortcut_of_section_eight_is_listed(qt_app: object) -> None:
    """The list in the settings dialog is the whole of PLAN.md section 8."""
    from ui.shortcuts import SHORTCUTS

    assert len(SHORTCUTS) >= 28
    for entry in SHORTCUTS:
        assert tr(entry.label_key) != entry.label_key, entry.label_key
        assert tr(entry.keys_key) != entry.keys_key, entry.keys_key


def test_every_registered_combination_is_one_qt_understands(qt_app: object) -> None:
    from PySide6.QtGui import QKeySequence

    from ui.shortcuts import REGISTERED

    for key in REGISTERED:
        sequence = QKeySequence(tr(key))
        assert not sequence.isEmpty(), key


def test_the_window_listens_for_everything_it_said_it_would(window: MainWindow) -> None:
    from PySide6.QtGui import QKeySequence, QShortcut

    from ui.shortcuts import REGISTERED

    listening = {
        shortcut.key().toString() for shortcut in window.findChildren(QShortcut)
    }
    for key in REGISTERED:
        if key == "shortcut.settings":  # the dialog it opens comes later
            continue
        assert QKeySequence(tr(key)).toString() in listening, key


def test_the_window_has_nothing_to_do_that_the_table_does_not_list(
    window: MainWindow,
) -> None:
    """The table and the window must say the same thing both ways round.

    The window used to carry its own copy of the list; the one that was
    checked was that everything listed is registered, so a combination the
    window knew about but the table had never heard of would have been
    registered and never shown in the settings dialog.
    """
    from ui.shortcuts import REGISTERED

    assert set(window.shortcut_handlers()) == set(REGISTERED)


def test_tab_is_intercepted_rather_than_registered(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    """Qt's own focus travel would eat Tab before a shortcut saw it.

    So the key is caught on its way to the focus widget, by the application
    wide filter, and the panels fold away (PLAN.md section 8).
    """
    from ui.shortcuts import REGISTERED

    assert "shortcut.toggle_panels" not in REGISTERED

    _open_photos(window, tmp_path, qt_app)
    assert not window.is_inspector_collapsed()

    _press(window, "shortcut.toggle_panels")
    qt_app.processEvents()  # type: ignore[attr-defined]

    assert window.is_inspector_collapsed() and window.is_filmstrip_collapsed()


def test_the_number_keys_switch_tabs(window: MainWindow) -> None:
    _press(window, "shortcut.tab_looks")
    assert window.inspector.current_tab == InspectorTab.LOOKS

    _press(window, "shortcut.tab_retouch")
    assert window.inspector.current_tab == InspectorTab.RETOUCH

    _press(window, "shortcut.tab_crop")
    assert window.inspector.current_tab == InspectorTab.CROP

    _press(window, "shortcut.tab_adjust")
    assert window.inspector.current_tab == InspectorTab.ADJUST


def test_i_brings_the_info_card_up_and_takes_it_away(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    """PLAN.md 3.9: the card is over the photo, not a tab of the panel."""
    _open_photos(window, tmp_path, qt_app)
    tab = window.inspector.current_tab

    _press(window, "shortcut.info")
    assert window.info_shown
    assert window.toolbar.info_button.isChecked()
    assert window.inspector.current_tab is tab, "the card is not a tab"
    _settle(qt_app, theme.INFO_CARD_FADE_MS + 200)
    assert not window.info_card.moving
    corner = window.info_card.pos()
    assert (corner.x(), corner.y()) == (theme.CANVAS_MARGIN, theme.CANVAS_MARGIN)

    _press(window, "shortcut.info")
    assert not window.info_shown
    assert not window.toolbar.info_button.isChecked()


def test_the_info_button_and_esc_work_the_card(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)

    window.toolbar.info_button.click()
    assert window.info_shown

    window.on_escape()
    assert not window.info_shown
    assert not window.toolbar.info_button.isChecked()


def test_the_info_card_is_not_up_over_the_welcome_screen_or_the_collage(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    window.toggle_info()
    assert not window.info_shown, "there is no photo to say anything about"

    _open_photos(window, tmp_path, qt_app)
    window.toggle_info()
    window.enter_collage()
    assert not window.info_shown
    assert not window.toolbar.info_button.isEnabled()
    window.leave_collage()
    assert window.toolbar.info_button.isEnabled()


def test_the_before_label_keeps_out_of_the_info_cards_corner(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)
    window.preview.set_showing_before(True)
    label = window.preview.before_label
    canvas = window.preview.canvas
    assert label.geometry().right() == canvas.width() - theme.CANVAS_MARGIN - 1
    assert label.y() == theme.CANVAS_MARGIN
    window.preview.set_showing_before(False)


def test_switching_tab_opens_the_panel_if_it_was_folded_away(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    """A photo is open here: with nothing open the welcome screen keeps the
    whole window, panel and all (PLAN.md 3.7)."""
    _open_photos(window, tmp_path, qt_app)
    window.set_inspector_collapsed(True, animate=False)
    _press(window, "shortcut.tab_crop")

    assert not window.is_inspector_collapsed()
    assert window.inspector.current_tab == InspectorTab.CROP


def test_the_zoom_keys_reach_the_canvas(window: MainWindow, qt_app: object) -> None:
    import numpy as np

    canvas = window.preview.canvas
    canvas.set_photo(np.zeros((400, 600, 3), dtype=np.uint8))
    canvas.fit_to_window()
    assert canvas.fitting

    _press(window, "shortcut.zoom_in")
    assert not canvas.fitting

    _press(window, "shortcut.zoom_fit")
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert canvas.fitting


def test_full_screen_folds_the_panels_away_and_puts_them_back(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    """PLAN.md section 8: F shows the photo, Esc comes back."""
    _open_photos(window, tmp_path, qt_app)
    assert not window.is_inspector_collapsed()

    window.toggle_full_screen()
    assert window.isFullScreen()
    assert window.is_inspector_collapsed()
    assert window.is_filmstrip_collapsed()
    # The photo and nothing else: not even the filmstrip's tab.
    assert window.filmstrip_tab.isHidden()

    window.on_escape()
    assert not window.isFullScreen()
    assert not window.is_inspector_collapsed()


def test_full_screen_remembers_a_panel_that_was_already_folded_away(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)
    window.set_filmstrip_collapsed(True, animate=False)

    window.toggle_full_screen()
    window.toggle_full_screen()
    _settle(qt_app, theme.DURATION_PANEL_MS + 120)

    assert window.is_filmstrip_collapsed()
    assert not window.is_inspector_collapsed()
    # Back to its tab, which full screen had taken away.
    assert window.filmstrip_tab.isVisible()


def test_escape_cancels_the_crop_before_it_leaves_full_screen(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_and_wait(window, _folder_with_a_photo(tmp_path), qt_app)
    window.toggle_full_screen()
    window.toggle_crop()
    assert window.controller.cropping

    window.on_escape()
    assert not window.controller.cropping
    assert window.isFullScreen()

    window.on_escape()
    assert not window.isFullScreen()


# -- the settings dialog reaching the window (PLAN.md 3.10) ---------------


def test_the_settings_are_applied_when_the_window_opens(qt_app: object) -> None:
    from app import settings as settings_module

    settings = Settings(
        window=WindowState(x=0, y=0, width=WINDOW_WIDTH, height=WINDOW_HEIGHT),
        preview_background=settings_module.BACKGROUND_MID,
    )
    main_window = MainWindow(settings)
    try:
        assert main_window.preview.property("background") == settings_module.BACKGROUND_MID
        assert (
            main_window.preview.canvas.property("background")
            == settings_module.BACKGROUND_MID
        )
    finally:
        main_window.hide()
        main_window.deleteLater()


def test_changing_the_background_repaints_the_preview(window: MainWindow) -> None:
    from app import settings as settings_module

    window._settings.preview_background = settings_module.BACKGROUND_BLACK
    window.apply_settings()

    assert window.preview.canvas.property("background") == settings_module.BACKGROUND_BLACK


def test_turning_the_screen_profile_off_reaches_the_controller(
    window: MainWindow,
) -> None:
    window._settings.use_display_profile = False
    window.apply_settings()
    assert window.controller._display_transform is None


def test_the_temperature_unit_reaches_the_sliders(window: MainWindow) -> None:
    from app import settings as settings_module
    from core.edit_state import EditState

    window.inspector.adjust_tab.set_photo(
        EditState(is_raw=True, temperature=5600.0), (5600.0, 0.0)
    )
    window._settings.temperature_unit = settings_module.TEMPERATURE_RELATIVE
    window.apply_settings()

    assert window.inspector.adjust_tab.rows()["temperature"].value == 0.0


def test_the_settings_menu_entry_and_shortcut_are_live(window: MainWindow) -> None:
    from PySide6.QtGui import QKeySequence, QShortcut

    assert window.toolbar.menu.settings_action.isEnabled()
    listening = {
        shortcut.key().toString() for shortcut in window.findChildren(QShortcut)
    }
    assert QKeySequence(tr("shortcut.settings")).toString() in listening


# -- opening the folder from last time (PLAN.md 3.10) --------------------


def test_the_last_folder_is_opened_at_start_when_that_is_asked_for(
    qt_app: object, tmp_path: object
) -> None:
    import time

    from app import settings as settings_module

    folder = _folder_with_a_photo(tmp_path)
    settings = Settings(
        window=WindowState(x=0, y=0, width=WINDOW_WIDTH, height=WINDOW_HEIGHT),
        startup_action=settings_module.STARTUP_LAST_FOLDER,
        last_folder=str(folder),
    )
    main_window = MainWindow(settings)
    try:
        # The folder is listed on a background thread; under load one pass
        # of the event loop is not always enough for it to arrive.
        deadline = time.perf_counter() + 10.0
        while not main_window.controller.photos and time.perf_counter() < deadline:
            qt_app.processEvents()  # type: ignore[attr-defined]
        assert len(main_window.controller.photos) == 1
    finally:
        main_window.controller.shutdown()
        main_window.hide()
        main_window.deleteLater()


def test_the_welcome_screen_stays_when_that_is_asked_for(
    qt_app: object, tmp_path: object
) -> None:
    folder = _folder_with_a_photo(tmp_path)
    settings = Settings(
        window=WindowState(x=0, y=0, width=WINDOW_WIDTH, height=WINDOW_HEIGHT),
        last_folder=str(folder),
    )
    main_window = MainWindow(settings)
    try:
        qt_app.processEvents()  # type: ignore[attr-defined]
        assert main_window.controller.photos == []
    finally:
        main_window.hide()
        main_window.deleteLater()


def test_a_folder_that_has_gone_leaves_the_welcome_screen_up(
    qt_app: object, tmp_path: object
) -> None:
    from app import settings as settings_module

    settings = Settings(
        window=WindowState(x=0, y=0, width=WINDOW_WIDTH, height=WINDOW_HEIGHT),
        startup_action=settings_module.STARTUP_LAST_FOLDER,
        last_folder=str(tmp_path / "not there any more"),  # type: ignore[operator]
    )
    main_window = MainWindow(settings)
    try:
        qt_app.processEvents()  # type: ignore[attr-defined]
        assert main_window.controller.photos == []
    finally:
        main_window.hide()
        main_window.deleteLater()


# -- the welcome screen is the whole window (PLAN.md 3.7) ----------------


def test_the_panels_stay_away_until_there_is_a_photo(window: MainWindow) -> None:
    """An empty panel and an empty filmstrip say nothing, so they wait."""
    assert window.controller.photos == []
    assert window.inspector.isHidden()
    assert window.filmstrip.isHidden()


def test_the_panels_arrive_with_the_first_photo(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)

    assert not window.inspector.isHidden()
    assert not window.filmstrip.isHidden()
    assert window.inspector.width() == theme.PANEL_WIDTH


def test_a_panel_the_user_folded_away_stays_folded_when_a_photo_opens(
    qt_app: object, tmp_path: object
) -> None:
    settings = Settings(
        window=WindowState(x=0, y=0, width=WINDOW_WIDTH, height=WINDOW_HEIGHT),
        inspector_collapsed=True,
    )
    main_window = MainWindow(settings)
    main_window.show()
    try:
        _open_photos(main_window, tmp_path, qt_app)
        assert main_window.inspector.isHidden()
        assert not main_window.filmstrip.isHidden()
    finally:
        main_window.controller.shutdown()
        main_window.hide()
        main_window.deleteLater()


def test_closing_from_the_welcome_screen_keeps_the_panels_as_they_were(
    qt_app: object,
) -> None:
    """The panels being away is the welcome screen's doing, not the user's."""
    settings = Settings(
        window=WindowState(x=0, y=0, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    )
    main_window = MainWindow(settings)
    main_window.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert main_window.inspector.isHidden()

    main_window._store_state()
    try:
        assert not settings.inspector_collapsed
        assert not settings.filmstrip_collapsed
    finally:
        main_window.hide()
        main_window.deleteLater()


def test_the_welcome_screen_shows_the_icon_and_what_the_program_is_for(
    window: MainWindow, qt_app: object
) -> None:
    from ui.welcome_view import opening_length

    welcome = window.preview.welcome
    _settle(qt_app, opening_length() + theme.DURATION_PHOTO_FADE_MS)
    assert not welcome.opening
    assert welcome.canvas.name == tr("app.name")
    assert welcome.canvas.tagline == tr("welcome.tagline")
    # The drop, the name and the tagline are drawn in the room kept for them.
    hero = welcome.hero.mapTo(welcome, welcome.hero.rect().topLeft())
    image = welcome.grab().toImage()
    lit = sum(
        1
        for y in range(hero.y(), hero.y() + welcome.hero.height(), 2)
        for x in range(hero.x(), hero.x() + welcome.hero.width(), 2)
        if image.pixelColor(x, y).lightness() > 80
    )
    assert lit > 200


def test_the_welcome_screen_button_is_outlined_and_says_what_it_opens(
    window: MainWindow,
) -> None:
    """PLAN.md 3.5 keeps the one filled button for exporting, so the button
    on the welcome screen is outlined, with a folder icon beside its name."""
    from ui.widgets.buttons import ROLE_HERO

    button = window.preview.welcome.open_folder_button
    assert button.property("role") == ROLE_HERO
    assert button.label == tr("welcome.open_folder")
    assert tr("shortcut.open_folder") in button.toolTip()
    assert button.height() == theme.WELCOME_BUTTON_HEIGHT


def test_the_welcome_screen_marks_where_photos_can_be_dropped(
    window: MainWindow, qt_app: object
) -> None:
    """PLAN.md 3.7: the drop hint lights up while a drag is over the window."""
    from PySide6.QtWidgets import QWidget

    zone = window.preview.welcome.drop_hint
    window._set_drop_target(True)
    # It takes the documented 80 ms to get there (PLAN.md 3.6).
    _settle(qt_app, theme.DURATION_HOVER_MS * 4)
    assert zone.lit == 1.0
    # The page on top of the preview covers the preview's own edges, so it
    # has to carry the dashed outline too, or nothing is seen at all.
    page = window.preview.findChild(QWidget, "PhotoPage")
    assert page is not None
    assert page.property("dropTarget") == "true"
    window._set_drop_target(False)
    _settle(qt_app, theme.DURATION_HOVER_MS * 4)
    assert zone.lit == 0.0
    assert page.property("dropTarget") == "false"


def test_every_tab_of_the_panel_marks_its_sections_the_same_way(
    window: MainWindow,
) -> None:
    """PLAN.md 3.4: 同類元素外觀完全一致.

    The adjust, crop and info tabs all lay their contents out in groups, so
    all three use the one header; the presets tab is a plain list and has
    none.
    """
    from ui.widgets.group import ControlGroup

    inspector = window.inspector
    for tab in (inspector.adjust_tab, inspector.crop_tab):
        groups = tab.groups()
        assert groups, type(tab).__name__
        for group in groups:
            assert isinstance(group, ControlGroup)
            assert group.header.height() == theme.GROUP_HEADER_HEIGHT
            assert group.is_open


def test_the_info_cards_groups_are_named_but_do_not_fold(window: MainWindow) -> None:
    """Folding half of a card that is only read saves nothing (PLAN.md 3.9)."""
    capture, files = window.info_card.groups()

    capture.header.click()
    assert capture.is_open and files.is_open
    assert not capture.header.isCheckable()
    assert capture.header.icon().isNull() or capture.header.iconSize().isEmpty()


def test_only_a_photo_moved_to_fades_in(window: MainWindow) -> None:
    """PLAN.md 3.6 asks for a fade when the photo changes, and the sliders
    send a stream of previews of the one photo that must not blink."""
    import numpy as np

    from app.controller import Preview

    faded: list[bool] = []
    window.preview.show_photo = (  # type: ignore[method-assign]
        lambda image, fade=False, frame=None, region=None, blend=False: faded.append(fade)
    )

    image = np.full((60, 80, 3), 128, dtype=np.uint8)
    first = Path("D:/Photos/one.ARW")
    window._on_preview_ready(Preview(path=first, image=image, provisional=False))
    window._on_preview_ready(Preview(path=first, image=image, provisional=False))
    window._on_preview_ready(
        Preview(path=Path("D:/Photos/two.ARW"), image=image, provisional=False)
    )

    assert faded == [True, False, True]


def test_the_developed_photo_blends_over_the_cameras_preview_only(window: MainWindow) -> None:
    """The camera's preview and the photo developed here differ in colour, so
    the one gives way to the other; nothing else is blended (PLAN.md 3.7)."""
    import numpy as np

    from app.controller import Preview

    blended: list[bool] = []
    window.preview.show_photo = (  # type: ignore[method-assign]
        lambda image, fade=False, frame=None, region=None, blend=False: blended.append(blend)
    )
    image = np.full((60, 80, 3), 128, dtype=np.uint8)
    one = Path("D:/Photos/one.ARW")
    window._on_preview_ready(Preview(path=one, image=image, provisional=True))
    window._on_preview_ready(Preview(path=one, image=image, provisional=False))
    window._on_preview_ready(Preview(path=one, image=image, provisional=False))
    two = Path("D:/Photos/two.ARW")
    window._on_preview_ready(Preview(path=two, image=image, provisional=True))
    window._on_preview_ready(Preview(path=Path("D:/Photos/three.ARW"), image=image, provisional=False))

    assert blended == [False, True, False, False, False]


def test_every_button_is_the_height_its_token_says(window: MainWindow) -> None:
    """PLAN.md 3.5: every size is a multiple of four.

    A stylesheet height is the height of the contents and Qt adds the border
    to it, so a rule built straight from a token came out two pixels taller
    than the token and off the grid.
    """
    expected = (
        (window.toolbar.export_button, theme.BUTTON_HEIGHT),
        (window.toolbar.undo_button, theme.ICON_BUTTON_SIZE),
        (window.preview.view_bar.fit_button, theme.BUTTON_HEIGHT),
        (window.preview.welcome.open_file_button, theme.WELCOME_BUTTON_HEIGHT),
        (window.preview.welcome.open_folder_button, theme.WELCOME_BUTTON_HEIGHT),
        (window.inspector.adjust_tab.groups()[0].header, theme.GROUP_HEADER_HEIGHT),
        (window.inspector.crop_tab.straighten_button, theme.BUTTON_HEIGHT),
    )
    for button, height in expected:
        assert button.height() == height, type(button).__name__
        assert button.height() % theme.UNIT == 0


def test_the_welcome_screen_closes_up_when_there_are_no_recent_folders(
    window: MainWindow,
) -> None:
    """On a first run the list has nothing in it, and the space it would
    have taken must go with it rather than leave a hole."""
    welcome = window.preview.welcome
    welcome.set_recent_folders(("D:/Photos/2026-10",))
    qt_app_processed = welcome._column.sizeHint().height()

    welcome.set_recent_folders(())
    assert not welcome._recent_block.isVisibleTo(welcome)
    assert welcome._column.sizeHint().height() < qt_app_processed


def _with_recent(window: MainWindow, monkeypatch: pytest.MonkeyPatch, *folders: str) -> list[int]:
    """Give the window *folders* as its recent list; count the saves, not writing them."""
    saves: list[int] = []
    monkeypatch.setattr(window._settings, "save", lambda *_args: saves.append(1) or True)
    for folder in reversed(folders):
        window._settings.remember_folder(folder, 10)
    window._show_recent_folders()
    return saves


def _menu_folders(window: MainWindow) -> list[str]:
    return [action.data() for action in window.toolbar.menu.recent_menu.actions() if action.data()]


def test_removing_a_folder_on_the_welcome_screen_forgets_it_everywhere(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    saves = _with_recent(window, monkeypatch, "D:/A", "D:/B", "D:/C")
    welcome = window.preview.welcome

    welcome.recent_rows()[1].remove_requested.emit()

    assert window._settings.recent_folders == ["D:/A", "D:/C"]
    assert [row.folder for row in welcome.recent_rows()] == ["D:/A", "D:/C"]
    assert _menu_folders(window) == ["D:/A", "D:/C"]
    assert saves, "the list must be saved at once"


def test_the_recent_menu_ends_with_a_command_that_clears_it(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_recent(window, monkeypatch, "D:/A", "D:/B")
    actions = window.toolbar.menu.recent_menu.actions()
    clear = window.toolbar.menu.clear_recent_action
    assert actions[-1] is clear
    assert actions[-2].isSeparator()
    assert clear.text() == tr("menu.recent_clear")
    assert clear.isEnabled()


def test_clearing_the_recent_menu_empties_both_lists(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    saves = _with_recent(window, monkeypatch, "D:/A", "D:/B")

    window.toolbar.menu.clear_recent_action.trigger()

    assert window._settings.recent_folders == []
    assert window._settings.last_folder is None
    assert window.preview.welcome.recent_rows() == []
    assert not window.toolbar.menu.recent_menu.isEnabled()
    assert saves


# -- the white balance eyedropper (A-24) ------------------------------------


def test_w_takes_up_the_eyedropper_and_esc_puts_it_down(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)
    button = window.inspector.adjust_tab.pick_button

    window.toggle_white_balance_pick()
    assert button.isChecked()
    assert window.preview.canvas.picking

    window.on_escape()
    assert not button.isChecked()
    assert not window.preview.canvas.picking


def test_w_brings_up_the_adjust_tab(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)
    window.inspector.set_current_tab(InspectorTab.LOOKS)

    window.toggle_white_balance_pick()

    assert window.inspector.current_tab is InspectorTab.ADJUST
    assert window.preview.canvas.picking


def test_leaving_the_adjust_tab_puts_the_eyedropper_down(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)
    window.toggle_white_balance_pick()

    window.inspector.set_current_tab(InspectorTab.CROP)

    assert not window.preview.canvas.picking


def test_a_pick_puts_the_eyedropper_down_and_moves_the_sliders(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    import time

    _open_photos(window, tmp_path, qt_app)
    window.toggle_white_balance_pick()

    window.preview.canvas.white_balance_picked.emit(0.5, 0.5)
    deadline = time.perf_counter() + 10.0
    while window.controller.state.temperature == 0.0 and time.perf_counter() < deadline:
        qt_app.processEvents()  # type: ignore[attr-defined]

    assert not window.preview.canvas.picking
    # The photo is a blue cast, so the eyedropper warms it.
    assert window.controller.state.temperature > 0.0
    assert window.inspector.adjust_tab.state == window.controller.state


def test_the_eyedropper_does_nothing_before_a_photo_is_open(window: MainWindow) -> None:
    window.toggle_white_balance_pick()
    assert not window.preview.canvas.picking


def test_a_refused_spot_leaves_the_eyedropper_up_for_another_try(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    import time

    from PIL import Image

    folder = tmp_path / "dark"  # type: ignore[operator]
    folder.mkdir()
    Image.new("RGB", (200, 150), (2, 3, 2)).save(folder / "night.jpg", "JPEG")
    _open_and_wait(window, folder, qt_app)
    _settle(qt_app, theme.DURATION_PANEL_MS + 120)
    refused: list[bool] = []
    window.controller.white_balance_refused.connect(lambda: refused.append(True))
    window.toggle_white_balance_pick()

    window.preview.canvas.white_balance_picked.emit(0.5, 0.5)
    deadline = time.perf_counter() + 10.0
    while not refused and time.perf_counter() < deadline:
        qt_app.processEvents()  # type: ignore[attr-defined]

    assert refused
    assert window.preview.canvas.picking


@pytest.mark.parametrize("name, value", [("grain", 50.0), ("exposure", 0.4), ("vignette", -30.0)])
def test_a_filter_chosen_on_its_tab_survives_a_slider_moved_afterwards(
    window: MainWindow, tmp_path: object, qt_app: object, name: str, value: float
) -> None:
    """Moving a slider used to send back the adjust tab's state from before the filter."""
    _open_photos(window, tmp_path, qt_app)
    controller = window.controller
    deadline = 10_000
    while controller.state is None or not window.inspector.adjust_tab.rows():
        _settle(qt_app, 50)
        deadline -= 50
        assert deadline > 0
    _settle(qt_app, 300)
    look = controller.filters[0].identifier

    window.inspector.looks_tab.chosen.emit(look)
    window.inspector.looks_tab.strength_changed.emit(60.0, False)
    window.inspector.adjust_tab.rows()[name].set_value(value, announce=True)

    assert controller.state.filter == look
    assert controller.state.filter_strength == 60.0
    assert getattr(controller.state, name) == pytest.approx(value)


def test_look_thumbnails_are_asked_for_only_while_the_looks_tab_shows(
    window: MainWindow, tmp_path: object, qt_app: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Behind another tab or a folded panel nobody sees them (PLAN.md 3.1)."""
    _open_photos(window, tmp_path, qt_app)
    asked: list[int] = []
    monkeypatch.setattr(window.controller, "set_look_thumbnail_side", asked.append)

    window.inspector.set_current_tab(InspectorTab.LOOKS)
    assert asked[-1] == round(theme.FILTER_THUMBNAIL_SIZE * window.devicePixelRatioF())

    window.toggle_inspector()
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert asked[-1] == 0

    window.toggle_inspector()
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert asked[-1] > 0

    window.inspector.set_current_tab(InspectorTab.ADJUST)
    assert asked[-1] == 0



def test_four_tabs_four_keys(window: MainWindow) -> None:
    """Ctrl+3 is the retouch tab and Ctrl+4 the crop tab; I works the info card (section 8)."""
    from ui.shortcuts import REGISTERED

    assert [tr(key) for key in REGISTERED if key.startswith("shortcut.tab_")] == [
        "Ctrl+1",
        "Ctrl+2",
        "Ctrl+3",
        "Ctrl+4",
    ]
    assert "shortcut.info" in REGISTERED


def test_saving_a_preset_shows_it_at_the_top_of_the_adjust_tab(
    window: MainWindow, tmp_path: object, qt_app: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new preset is in the list the user is taken to (PLAN.md 3.1)."""
    from ui import main_window as window_module

    window.controller._presets_dir = tmp_path / "presets"  # type: ignore[operator]
    _open_photos(window, tmp_path, qt_app)
    window.inspector.set_current_tab(InspectorTab.LOOKS)
    window.controller.set_adjustment("vibrance", 25.0)
    monkeypatch.setattr(
        window_module.NameDialog, "ask", classmethod(lambda *_args, **_kwargs: "Lively")
    )

    window.inspector.adjust_tab.save_preset_button.click()
    _settle(qt_app, 50)

    assert window.inspector.current_tab is InspectorTab.ADJUST
    rows = window.inspector.adjust_tab.preset_rows()
    assert [row.preset.name for row in rows] == ["Lively"]


def test_clicking_a_preset_moves_the_sliders_under_it(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    """The point of the list sitting above the sliders."""
    from core.presets import Preset

    _open_photos(window, tmp_path, qt_app)
    window.inspector.adjust_tab.set_presets(
        [Preset(id="v", name="Vivid", adjustments={"vibrance": 30.0})]
    )

    window.inspector.adjust_tab.preset_rows()[0].click()

    assert window.controller.state.vibrance == 30.0
    assert window.inspector.adjust_tab.rows()["vibrance"].value == 30.0



def test_the_wand_sits_on_the_light_header(window: MainWindow) -> None:
    """Above the sliders it sets (PLAN.md 3.1), with its shortcut in the tooltip."""
    tab = window.inspector.adjust_tab
    light = next(
        group for group in tab.groups() if group.header.text() == tr("adjust.group.light")
    )

    assert light.header.isAncestorOf(tab.auto_button)
    assert tr("shortcut.auto_light") in tab.auto_button.toolTip()


def test_ctrl_u_shows_the_adjust_tab_and_asks_for_the_automatic_light(
    window: MainWindow, tmp_path: object, qt_app: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    _open_photos(window, tmp_path, qt_app)
    asked: list[bool] = []
    monkeypatch.setattr(window.controller, "auto_light", lambda: asked.append(True))
    window.inspector.set_current_tab(InspectorTab.LOOKS)

    window.shortcut_handlers()["shortcut.auto_light"]()

    assert asked == [True]
    assert window.inspector.current_tab is InspectorTab.ADJUST


# -- the toolbar over the welcome screen (PLAN.md 3.7) ----------------------


def _toolbar_shown(window: MainWindow) -> dict[str, bool]:
    bar = window.toolbar
    return {
        "menu": bar.menu_button.isVisibleTo(window),
        "undo": bar.undo_button.isVisibleTo(window),
        "redo": bar.redo_button.isVisibleTo(window),
        "collage": bar.collage_button.isVisibleTo(window),
        "title": bar.title_label.isVisibleTo(window),
        "export": bar.export_button.isVisibleTo(window),
    }


def test_the_welcome_screen_shows_only_the_menu(window: MainWindow) -> None:
    """Buttons that cannot be pressed yet only say that they cannot."""
    shown = _toolbar_shown(window)

    assert shown.pop("menu")
    assert not any(shown.values()), shown


def test_the_toolbar_comes_back_with_the_first_photo(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)

    assert all(_toolbar_shown(window).values())


def test_a_collage_opened_from_the_welcome_screen_can_be_left(window: MainWindow) -> None:
    """Its toolbar button is the way back, so it shows while the collage is up."""
    window.enter_collage()
    assert window.toolbar.collage_button.isVisibleTo(window)

    window.leave_collage()
    assert not window.toolbar.collage_button.isVisibleTo(window)


def test_the_developing_sign_waits_and_goes_when_the_photo_arrives(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    """Shown only once a photo has kept the user waiting (PLAN.md 3.7)."""
    import numpy as np

    from app.controller import Preview

    _open_photos(window, tmp_path, qt_app)
    sign = window.preview.developing
    # A small JPEG is developed well inside the delay: no sign at all.
    assert not sign.active
    assert window._developing_timer.interval() == theme.DEVELOPING_DELAY_MS

    window._on_selection_changed(0, 1)
    assert window._developing_timer.isActive()
    assert not sign.active, "the sign came up before the delay"

    window._developed_path = None
    window._show_developing()
    assert sign.active

    current = window.controller.current
    image = np.full((60, 80, 3), 128, dtype=np.uint8)
    window._on_preview_ready(Preview(path=current, image=image, provisional=True))
    assert sign.active, "the camera's preview is not the developed photo"
    window._on_preview_ready(Preview(path=current, image=image, provisional=False))
    assert not sign.active
    assert not window._developing_timer.isActive()


# -- spot removal (PLAN.md 2.2, A-32) ------------------------------------------


def test_q_takes_up_the_spot_tool_on_the_retouch_tab(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)
    window.inspector.set_current_tab(InspectorTab.LOOKS)

    window.toggle_spot_tool()
    assert window.inspector.current_tab is InspectorTab.RETOUCH
    assert window.inspector.retouch_tab.spot_button.isChecked()
    assert window.preview.canvas.spots.active

    window.toggle_spot_tool()
    assert not window.preview.canvas.spots.active


def test_leaving_the_retouch_tab_or_esc_puts_the_spot_tool_down(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)
    window.toggle_spot_tool()
    window.inspector.set_current_tab(InspectorTab.ADJUST)
    assert not window.spot_tool_on

    window.toggle_spot_tool()
    window.on_escape()
    assert not window.spot_tool_on


def test_the_spot_tool_does_nothing_before_a_photo_is_open(window: MainWindow) -> None:
    window.toggle_spot_tool()
    assert not window.spot_tool_on


def test_painting_with_the_spot_tool_mends_it_without_selecting_it(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    import time

    _open_photos(window, tmp_path, qt_app)
    deadline = time.perf_counter() + 10.0
    while window.controller.photo_size == (1, 1) and time.perf_counter() < deadline:
        qt_app.processEvents()  # type: ignore[attr-defined]
    window.toggle_spot_tool()

    window.preview.canvas.spots.spot_painted.emit([(0.4, 0.5), (0.5, 0.5), (0.6, 0.5)])
    deadline = time.perf_counter() + 10.0
    while not window.controller.geometry.spots and time.perf_counter() < deadline:
        qt_app.processEvents()  # type: ignore[attr-defined]

    assert len(window.controller.geometry.spots) == 1
    assert len(window.controller.geometry.spots[0].points) > 1
    assert window.preview.canvas.spots.selected == -1  # nothing to look after
    assert window.inspector.retouch_tab.spots_caption.text() == tr("adjust.spots.one")

    # Picked by a click, ] makes it larger, Esc lets it go, Delete removes it.
    window.preview.canvas.spots.select(0)
    radius = window.controller.geometry.spots[0].radius
    window.resize_spot_brush(larger=True)
    assert window.controller.geometry.spots[0].radius > radius
    window.on_escape()
    assert window.preview.canvas.spots.selected == -1 and window.spot_tool_on
    window.preview.canvas.spot_deleted.emit(0)
    assert window.controller.geometry.spots == ()


# -- the watermark (PLAN.md 2.9, 3.14) --------------------------------------


def _wait_for_photo(window: MainWindow, qt_app: object) -> None:
    import time

    deadline = time.perf_counter() + 10.0
    while not window.preview.canvas.has_photo and time.perf_counter() < deadline:
        qt_app.processEvents()  # type: ignore[attr-defined]


def test_the_watermark_needs_a_photo(window: MainWindow) -> None:
    window.toggle_watermark()
    assert not window.in_watermark
    assert not window.toolbar.watermark_button.isChecked()


def test_the_watermark_takes_the_inspectors_place(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)
    _wait_for_photo(window, qt_app)
    window.toggle_watermark()
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert window.in_watermark
    assert window.toolbar.watermark_button.isChecked()
    assert window.watermark_panel.isVisible() and not window.inspector.isVisible()
    assert window.toolbar.title_label.text() == tr("watermark.title")
    assert not window.toolbar.undo_button.isEnabled()
    assert window.toolbar.collage_button.isEnabled()
    assert window.preview.canvas.watermark.active

    window.on_escape()
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert not window.in_watermark
    assert window.inspector.isVisible() and not window.watermark_panel.isVisible()
    assert window.toolbar.title_label.text().endswith("a.jpg")
    assert window.toolbar.collage_button.isEnabled()
    assert not window.preview.canvas.watermark.active


def test_typing_the_first_text_turns_the_watermark_on(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    from core.watermark import WatermarkSpec

    _open_photos(window, tmp_path, qt_app)
    _wait_for_photo(window, qt_app)
    window.enter_watermark()
    assert not window._settings.watermark.enabled
    window.watermark_panel.spec_changed.emit(WatermarkSpec(text="© me"))
    assert window._settings.watermark.enabled
    assert window.watermark_panel.enabled
    assert window._settings.watermark.active == WatermarkSpec(text="© me")
    # Drawn over the photo, the size of the photo on screen.
    overlay = window.preview.canvas.watermark
    assert not overlay.screen_rect(window.preview.canvas.photo_rect()).isEmpty()


def test_the_collage_and_the_watermark_switch_straight_to_each_other(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)
    _wait_for_photo(window, qt_app)
    window.enter_collage()
    assert window.toolbar.watermark_button.isEnabled()
    window.toolbar.watermark_button.click()
    assert window.in_watermark and not window.in_collage
    assert window.watermark_panel.isVisible() and not window.collage_panel.isVisible()
    assert window.toolbar.title_label.text() == tr("watermark.title")
    window.toolbar.collage_button.click()
    assert window.in_collage and not window.in_watermark
    assert window.collage_panel.isVisible() and not window.watermark_panel.isVisible()
    assert not window.preview.canvas.watermark.active


def test_the_mode_keys_switch_between_the_modes(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)
    _wait_for_photo(window, qt_app)
    handlers = window.shortcut_handlers()
    window.enter_collage()
    window._on_shortcut("shortcut.portrait", handlers["shortcut.portrait"])
    assert window.in_portrait and not window.in_collage
    window._on_shortcut("shortcut.watermark", handlers["shortcut.watermark"])
    assert window.in_watermark and not window.in_portrait
    window._on_shortcut("shortcut.portrait", handlers["shortcut.portrait"])
    assert window.in_portrait and not window.in_watermark
    window.enter_watermark()
    window._on_shortcut("shortcut.collage", handlers["shortcut.collage"])
    assert window.in_collage and not window.in_watermark
    window._on_shortcut("shortcut.watermark", handlers["shortcut.watermark"])
    assert window.in_watermark and not window.in_collage


def test_a_collage_opened_before_any_photo_keeps_the_photo_modes_off(
    window: MainWindow,
) -> None:
    window.enter_collage()
    assert not window.toolbar.portrait_button.isEnabled()
    assert not window.toolbar.watermark_button.isEnabled()
    window.enter_portrait()
    assert window.in_collage and not window.in_portrait


def test_the_watermark_shortcut_is_t(window: MainWindow) -> None:
    assert window.shortcut_handlers()["shortcut.watermark"] == window.toggle_watermark
    assert tr("shortcut.watermark") == "T"


# -- the portrait (PLAN.md 2.10, 3.15) ---------------------------------------


def test_the_portrait_needs_a_photo(window: MainWindow) -> None:
    window.toggle_portrait()
    assert not window.in_portrait
    assert not window.toolbar.portrait_button.isChecked()


def test_the_portrait_takes_the_inspectors_place_and_keeps_the_history(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)
    _wait_for_photo(window, qt_app)
    window.toggle_portrait()
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert window.in_portrait
    assert window.toolbar.portrait_button.isChecked()
    assert window.portrait_panel.isVisible() and not window.inspector.isVisible()
    assert window.toolbar.title_label.text().endswith("a.jpg")
    assert not window.preview.canvas.skin.active  # the brush waits to be taken up

    window.toggle_portrait_brush("add")
    assert window.preview.canvas.skin.active
    window.on_escape()  # puts the brush down first
    assert window.in_portrait and not window.preview.canvas.skin.active
    window.on_escape()
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert not window.in_portrait
    assert window.inspector.isVisible() and not window.portrait_panel.isVisible()


def test_a_stroke_on_the_photo_shapes_the_skin(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)
    _wait_for_photo(window, qt_app)
    window.enter_portrait()
    window.toggle_portrait_brush("remove")
    window.preview.canvas.skin.stroke_painted.emit([(0.4, 0.5), (0.6, 0.5)])
    skin = window.controller.skin
    assert len(skin.strokes) == 1 and not skin.strokes[0].adding
    assert window.portrait_panel.reset_button.isEnabled()
    window.controller.undo()
    assert window.controller.skin.strokes == ()


def test_the_portrait_and_the_watermark_take_turns(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)
    _wait_for_photo(window, qt_app)
    window.enter_portrait()
    window.enter_watermark()
    assert window.in_watermark and not window.in_portrait
    window.enter_portrait()
    assert window.in_portrait and not window.in_watermark
    window.enter_collage()
    assert not window.in_portrait
    assert window.toolbar.portrait_button.isEnabled()
    window.toolbar.portrait_button.click()
    assert window.in_portrait and not window.in_collage
    assert window.portrait_panel.isVisible() and not window.collage_panel.isVisible()
    assert window.toolbar.title_label.text().endswith("a.jpg")


def test_the_portrait_keys(window: MainWindow) -> None:
    handlers = window.shortcut_handlers()
    assert handlers["shortcut.portrait"] == window.toggle_portrait
    assert handlers["shortcut.portrait_show"] == window.toggle_portrait_showing
    assert [tr(key) for key in (
        "shortcut.portrait", "shortcut.portrait_add", "shortcut.portrait_remove",
        "shortcut.portrait_show",
    )] == ["P", "B", "E", "M"]


# -- liquify (PLAN.md 2.11, 3.16) --------------------------------------------


def test_liquify_needs_a_photo(window: MainWindow) -> None:
    window.toggle_liquify()
    assert not window.in_liquify
    assert not window.toolbar.liquify_button.isChecked()


def test_liquify_takes_the_inspectors_place_with_its_brush_taken_up(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)
    _wait_for_photo(window, qt_app)
    toolbar = window.toolbar
    assert toolbar.liquify_button.x() > toolbar.portrait_button.x()
    assert toolbar.liquify_button.x() < toolbar.collage_button.x()
    window.toggle_liquify()
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert window.in_liquify
    assert toolbar.liquify_button.isChecked()
    assert window.liquify_panel.isVisible() and not window.inspector.isVisible()
    assert toolbar.title_label.text().endswith("a.jpg")

    window.on_escape()
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert not window.in_liquify and not window.preview.canvas.liquify.active
    assert window.inspector.isVisible() and not window.liquify_panel.isVisible()


def test_a_stroke_on_the_photo_reshapes_it_with_the_panels_tool(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    from core import liquify

    _open_photos(window, tmp_path, qt_app)
    _wait_for_photo(window, qt_app)
    window.enter_liquify()
    window.liquify_panel.tool_buttons[liquify.TOOL_BLOAT].click()
    window.liquify_panel.set_strength(0.8)
    window.preview.canvas.liquify.stroke_painted.emit([(0.5, 0.5)] * 6)
    (stroke,) = window.controller.liquify_strokes
    assert stroke.tool == liquify.TOOL_BLOAT and stroke.strength == pytest.approx(0.8)
    assert window.liquify_panel.reset_button.isEnabled()
    assert window.liquify_panel.strokes_caption.text() == tr("liquify.strokes.one")
    window.controller.undo()
    assert window.controller.liquify_strokes == ()
    assert not window.liquify_panel.reset_button.isEnabled()


def test_the_brackets_size_the_liquify_brush(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)
    _wait_for_photo(window, qt_app)
    window.enter_liquify()
    size = window.liquify_panel.brush_size
    window.resize_spot_brush(True)
    assert window.liquify_panel.brush_size == pytest.approx(size * 1.25)


def test_liquify_and_the_other_modes_take_turns(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)
    _wait_for_photo(window, qt_app)
    window.enter_liquify()
    window.enter_portrait()
    assert window.in_portrait and not window.in_liquify
    assert not window.preview.canvas.liquify.active
    window.enter_liquify()
    assert window.in_liquify and not window.in_portrait
    window.enter_watermark()
    assert window.in_watermark and not window.in_liquify
    window.enter_liquify()
    assert window.in_liquify and not window.in_watermark
    window.enter_collage()
    assert not window.in_liquify
    window.toolbar.liquify_button.click()
    assert window.in_liquify and not window.in_collage
    assert window.liquify_panel.isVisible() and not window.collage_panel.isVisible()


def test_the_liquify_key_and_what_it_turns_off(window: MainWindow) -> None:
    handlers = window.shortcut_handlers()
    assert handlers["shortcut.liquify"] == window.toggle_liquify
    assert tr("shortcut.liquify") == "L"
    window._liquify_mode = True
    called: list[str] = []
    window._on_shortcut("shortcut.spot_tool", lambda: called.append("spot"))
    window._on_shortcut("shortcut.crop", lambda: called.append("crop"))
    window._on_shortcut("shortcut.undo", lambda: called.append("undo"))
    window._liquify_mode = False
    assert called == ["undo"]


def test_the_face_sliders_reach_the_controller_and_follow_the_faces(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)
    _wait_for_photo(window, qt_app)
    window.enter_liquify()
    panel = window.liquify_panel
    window._on_faces_ready(0)
    assert not panel.face_rows["eyes"].isEnabled()
    window._on_faces_ready(1)
    assert panel.face_rows["eyes"].isEnabled()
    panel.face_value_set.emit("mouth", 30.0, False)
    assert window.controller.face_shape.mouth == 30.0
    window.controller.undo()
    assert panel.face_rows["mouth"].value == 0.0


def test_placing_the_face_points_puts_the_brush_down_and_esc_ends_it_first(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    _open_photos(window, tmp_path, qt_app)
    _wait_for_photo(window, qt_app)
    window.enter_liquify()
    window._on_faces_ready(1)
    canvas = window.preview.canvas
    window.liquify_panel.edit_faces_link.click()
    assert window.editing_faces and window.controller.editing_faces
    assert canvas.faces.active and not canvas.liquify.active

    window.on_escape()  # ends placing the points first
    assert window.in_liquify and not window.editing_faces
    assert not window.controller.editing_faces and not window.liquify_panel.editing
    window.liquify_panel.edit_faces_link.click()
    window.leave_liquify()
    assert not window.editing_faces and not window.controller.editing_faces
    assert not window.liquify_panel.editing


def test_a_dragged_face_point_reaches_the_controller(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    from core.faces import Face, SkinTone

    _open_photos(window, tmp_path, qt_app)
    _wait_for_photo(window, qt_app)
    controller = window.controller
    face = Face(
        points=((0.45, 0.4), (0.55, 0.4), (0.5, 0.47), (0.47, 0.54), (0.53, 0.54), (0.5, 0.48)),
        tone=SkinTone(red=0.0, blue=0.0, spread=0.1, brightness=-2.0),
    )
    controller._faces[(controller.current, controller.geometry.lens)] = (face,)
    window.enter_liquify()
    window._on_faces_ready(1)
    window.liquify_panel.edit_faces_link.click()
    window.preview.canvas.faces.handle_moved.emit(0, "nose", 0.52, 0.48, False)
    assert controller.face_shape.faces[0].points[2] == pytest.approx((0.52, 0.48), abs=1e-6)
    assert window.liquify_panel.reset_faces_button.isEnabled()


def test_placing_the_points_zooms_onto_the_face_and_puts_the_view_back(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    from core.faces import Face, SkinTone

    _open_photos(window, tmp_path, qt_app)
    _wait_for_photo(window, qt_app)
    controller = window.controller
    face = Face(
        points=((0.45, 0.4), (0.55, 0.4), (0.5, 0.47), (0.47, 0.54), (0.53, 0.54), (0.5, 0.48)),
        tone=SkinTone(red=0.0, blue=0.0, spread=0.1, brightness=-2.0),
    )
    controller._faces[(controller.current, controller.geometry.lens)] = (face,)
    canvas = window.preview.canvas
    assert canvas.fitting
    window.enter_liquify()
    window._on_faces_ready(1)
    window.liquify_panel.edit_faces_link.click()
    # This small photo already fills the window at its largest zoom, so the
    # view only moves its middle onto the face.
    fitting, _scale, centre = canvas.view_state()
    width, height = canvas.frame_size
    (guide,) = controller.face_guides_on_screen()
    xs = [x for x, _ in guide.handles]
    ys = [y for _, y in guide.handles]
    middle = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
    assert not fitting
    assert (centre.x() / width, centre.y() / height) == pytest.approx(middle, abs=0.01)
    window.liquify_panel.finish_editing_button.click()
    assert canvas.fitting


def test_the_brush_paints_only_on_its_page_and_alt_turns_a_twirl_back(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    from core import liquify
    from core.faces import Face, SkinTone
    from ui.liquify_panel import PAGE_BRUSH, PAGE_FACE

    _open_photos(window, tmp_path, qt_app)
    _wait_for_photo(window, qt_app)
    controller = window.controller
    face = Face(
        points=((0.45, 0.4), (0.55, 0.4), (0.5, 0.47), (0.47, 0.54), (0.53, 0.54), (0.5, 0.48)),
        tone=SkinTone(red=0.0, blue=0.0, spread=0.1, brightness=-2.0),
    )
    controller._faces[(controller.current, controller.geometry.lens)] = (face,)
    window.enter_liquify()
    panel = window.liquify_panel
    canvas = window.preview.canvas
    assert panel.page == PAGE_FACE and not canvas.liquify.active
    panel.page_switch.buttons()[PAGE_BRUSH].click()
    assert canvas.liquify.active

    panel.tool_buttons[liquify.TOOL_TWIRL_CW].click()
    canvas.liquify.set_reversed(True)
    canvas.liquify.stroke_painted.emit([(0.5, 0.5)] * 5)
    canvas.liquify.set_reversed(False)
    canvas.liquify.stroke_painted.emit([(0.3, 0.3)] * 5)
    first, second = window.controller.liquify_strokes
    assert first.tool == liquify.TOOL_TWIRL_CCW and second.tool == liquify.TOOL_TWIRL_CW


def test_a_photo_without_a_face_opens_liquify_on_the_brush(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    from ui.liquify_panel import PAGE_BRUSH

    _open_photos(window, tmp_path, qt_app)
    _wait_for_photo(window, qt_app)
    window._on_faces_ready(0)
    controller = window.controller
    controller._faces[(controller.current, controller.geometry.lens)] = ()
    window.enter_liquify()
    assert window.liquify_panel.page == PAGE_BRUSH
    assert window.preview.canvas.liquify.active


def test_a_face_slider_under_the_pointer_is_outlined_on_the_photo(
    window: MainWindow, tmp_path: object, qt_app: object
) -> None:
    from core.faces import Face, SkinTone

    _open_photos(window, tmp_path, qt_app)
    _wait_for_photo(window, qt_app)
    controller = window.controller
    face = Face(
        points=((0.45, 0.4), (0.55, 0.4), (0.5, 0.47), (0.47, 0.54), (0.53, 0.54), (0.5, 0.48)),
        tone=SkinTone(red=0.0, blue=0.0, spread=0.1, brightness=-2.0),
    )
    controller._faces[(controller.current, controller.geometry.lens)] = (face,)
    window._on_faces_ready(1)
    window.enter_liquify()
    faces = window.preview.canvas.faces
    window.liquify_panel.face_hovered.emit("chin")
    assert faces.spotlit == "chin"
    window.liquify_panel.face_value_set.emit("chin", 40.0, True)
    assert faces.spotlit == "chin" and controller.face_shape.chin == 40.0
    window.liquify_panel.face_value_set.emit("chin", 40.0, False)
    window.liquify_panel.face_hovered.emit("")
    assert faces.spotlit == "chin"  # lingers a moment
    window._spotlight_timer.timeout.emit()
    assert faces.spotlit == ""

    window.liquify_panel.face_value_set.emit("chin", 40.0, False)
    controller.reset_face_shape()
    assert controller.face_shape.is_default and controller.face_shape.faces

