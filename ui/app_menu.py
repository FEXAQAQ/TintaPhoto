"""The application menu in the top left corner (PLAN.md 3.1).

Holds the file and application level commands: open file, open folder, recent
folders (with a command at the bottom that clears them), settings, the development log and about. The actions are created here
and exposed as attributes so that the controller can connect them in a later
stage; in stage 0 they are disabled because nothing can be opened yet.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMenu, QWidget

from app.i18n import tr


class AppMenu(QMenu):
    """The menu shown by the toolbar's menu button."""

    #: A folder picked from the recent list.
    folder_chosen = Signal(Path)
    #: The recent list is to be emptied (the records only, not the folders).
    recent_cleared = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        #: The key each command was built from, so that it can be named
        #: again after a language change (U-02).
        self._label_keys: dict[QAction, tuple[str, str | None]] = {}
        #: The folders the recent submenu is showing, for the same reason.
        self._recent_folders: tuple[str, ...] = ()

        self.open_file_action = self._add_action("menu.open_file", "shortcut.open_file")
        self.open_folder_action = self._add_action("menu.open_folder", "shortcut.open_folder")

        self.recent_menu = QMenu(tr("menu.recent"), self)
        self.addMenu(self.recent_menu)
        # Owned by this menu rather than the submenu, so rebuilding the
        # submenu takes it out without deleting it.
        self.clear_recent_action = QAction(tr("menu.recent_clear"), self)
        self.clear_recent_action.triggered.connect(self.recent_cleared)
        self.set_recent_folders(())

        self.addSeparator()
        self.settings_action = self._add_action("menu.settings", "shortcut.settings")
        self.devlog_action = self._add_action("menu.devlog")
        self.about_action = self._add_action("menu.about")

    def _add_action(self, label_key: str, shortcut_key: str | None = None) -> QAction:
        action = QAction(tr(label_key), self)
        if shortcut_key is not None:
            action.setShortcut(QKeySequence(tr(shortcut_key)))
        # Stage 0 has no file handling yet; the controller enables these later.
        action.setEnabled(False)
        self.addAction(action)
        self._label_keys[action] = (label_key, shortcut_key)
        return action

    def retranslate(self) -> None:
        """Name the commands again after a language change (U-02)."""
        for action, (label_key, shortcut_key) in self._label_keys.items():
            action.setText(tr(label_key))
            if shortcut_key is not None:
                action.setShortcut(QKeySequence(tr(shortcut_key)))
        self.recent_menu.setTitle(tr("menu.recent"))
        self.clear_recent_action.setText(tr("menu.recent_clear"))
        self.set_recent_folders(self._recent_folders)

    def set_recent_folders(self, folders: Sequence[str]) -> None:
        """Rebuild the recent folder submenu from *folders*."""
        self._recent_folders = tuple(folders)
        self.recent_menu.clear()
        if not folders:
            placeholder = QAction(tr("menu.recent_empty"), self.recent_menu)
            placeholder.setEnabled(False)
            self.recent_menu.addAction(placeholder)
            self.recent_menu.setEnabled(False)
            return
        self.recent_menu.setEnabled(True)
        for folder in folders:
            action = QAction(Path(folder).name or folder, self.recent_menu)
            action.setData(folder)
            action.setToolTip(folder)
            action.triggered.connect(
                lambda _checked=False, path=folder: self.folder_chosen.emit(Path(path))
            )
            self.recent_menu.addAction(action)
        self.recent_menu.addSeparator()
        self.recent_menu.addAction(self.clear_recent_action)
