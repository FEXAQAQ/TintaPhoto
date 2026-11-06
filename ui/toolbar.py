"""The 44 px toolbar across the top of the window (PLAN.md 3.1).

Left: application menu, undo, redo, the info card, the portrait, the
collage and the watermark. Centre: folder name and file name of the
current photo. Right: the export button, the only filled button in the whole
interface.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from app.i18n import tr
from ui import icons, theme
from ui.app_menu import AppMenu
from ui.widgets.buttons import IconButton, ProgressButton
from ui.widgets.region import styled_background


class Toolbar(QWidget):
    """The top toolbar; file and output commands only."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        styled_background(self, "Toolbar")
        self.setFixedHeight(theme.TOOLBAR_HEIGHT)
        #: What the title says now, so that it can be said again in another
        #: language (U-02).
        self._title: tuple[str | None, str | None] = (None, None)

        self.menu = AppMenu(self)

        self.menu_button = IconButton(icons.ICON_MENU, "toolbar.menu", parent=self)
        # The menu is popped up by hand rather than with setMenu(): Qt derives
        # the menu's font from the button, and a font whose size comes from the
        # stylesheet has no point size, which makes Qt warn. Doing it here also
        # places the menu exactly under the button.
        self.menu_button.clicked.connect(self._show_menu)
        self.undo_button = IconButton(icons.ICON_UNDO, "toolbar.undo", "shortcut.undo", self)
        self.redo_button = IconButton(icons.ICON_REDO, "toolbar.redo", "shortcut.redo", self)
        #: Opens the portrait (PLAN.md 3.15); checked while it is open, like
        #: the collage and the watermark.
        self.portrait_button = IconButton(
            icons.ICON_PORTRAIT, "toolbar.portrait", "shortcut.portrait", self
        )
        self.portrait_button.setCheckable(True)
        # Checked while the collage is open: the one place in the toolbar
        # that shows a current state (PLAN.md 3.13).
        self.collage_button = IconButton(
            icons.ICON_COLLAGE, "toolbar.collage", "shortcut.collage", self
        )
        self.collage_button.setCheckable(True)
        #: Opens the watermark (PLAN.md 3.14); checked while it is open, like
        #: the collage.
        self.watermark_button = IconButton(
            icons.ICON_STAMP, "toolbar.watermark", "shortcut.watermark", self
        )
        self.watermark_button.setCheckable(True)
        #: Brings the info card up over the photo (PLAN.md 3.9); marked while
        #: the card is up, the way the eyedropper is while it is held.
        self.info_button = IconButton(icons.ICON_INFO, "toolbar.info", "shortcut.info", self)
        self.info_button.setCheckable(True)
        self.export_button = ProgressButton("toolbar.export", "shortcut.export", self)
        self.export_button.rested.connect(self.refit_export)

        self.title_label = QLabel(tr("toolbar.no_photo"), self)
        self.title_label.setProperty("role", "secondary")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Nothing can be edited or exported until a photo is open.
        self.undo_button.setEnabled(False)
        self.redo_button.setEnabled(False)
        self.export_button.setEnabled(False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(theme.SPACE_SM, 0, theme.SPACE_SM, 0)
        layout.setSpacing(theme.SPACE_XS)
        layout.addWidget(self.menu_button)
        layout.addSpacing(theme.SPACE_XS)
        layout.addWidget(self.undo_button)
        layout.addWidget(self.redo_button)
        layout.addSpacing(theme.SPACE_XS)
        layout.addWidget(self.info_button)
        layout.addWidget(self.portrait_button)
        layout.addWidget(self.collage_button)
        layout.addWidget(self.watermark_button)
        layout.addStretch(1)
        layout.addWidget(self.title_label)
        layout.addStretch(1)
        # The export button widens while it works and says it is done; the
        # room for its widest look is kept, so nothing beside it shifts.
        self._export_slot = QWidget(self)
        slot = QHBoxLayout(self._export_slot)
        slot.setContentsMargins(0, 0, 0, 0)
        slot.setSpacing(0)
        slot.addStretch(1)
        slot.addWidget(self.export_button)
        self._export_slot.setFixedWidth(self.export_button.widest_width())
        layout.addWidget(self._export_slot)

    def _show_menu(self) -> None:
        """Pop the application menu up directly below the menu button."""
        corner = self.menu_button.mapToGlobal(self.menu_button.rect().bottomLeft())
        self.menu.popup(corner + QPoint(0, theme.SPACE_XS))

    def set_welcome(self, welcome: bool) -> None:
        """Keep only the menu while the welcome screen is up (PLAN.md 3.7).

        Undo, redo, the collage, the title and export have nothing to act on
        or say before a photo is open, and a row of buttons that cannot be
        pressed only says that; they appear with the first photo.
        """
        for widget in (
            self.undo_button,
            self.redo_button,
            self.portrait_button,
            self.collage_button,
            self.watermark_button,
            self.info_button,
            self.title_label,
            self.export_button,
        ):
            widget.setVisible(not welcome)

    def set_export_count(self, count: int | None) -> None:
        """Name the export button by how many photos it writes, or plainly.

        *count* is the number of photos ticked in the filmstrip while they
        are being chosen (X-09), and ``None`` the rest of the time.
        """
        if count is None:
            self.export_button.set_name("toolbar.export")
        else:
            self.export_button.set_name("toolbar.export_count", count=count)
        self.refit_export()

    def refit_export(self) -> None:
        """Keep room for the export button's widest look, after it changed."""
        self._export_slot.setFixedWidth(self.export_button.widest_width())

    def set_title(self, folder: str | None, file_name: str | None) -> None:
        """Show ``folder / file name``, or the empty state when nothing is open."""
        self._title = (folder, file_name)
        if not file_name:
            self.title_label.setText(tr("toolbar.no_photo"))
            return
        self.title_label.setText(f"{folder} / {file_name}" if folder else file_name)

    def retranslate(self) -> None:
        """Say the title again after a language change (U-02).

        Only the empty state is interface text; a photo's own name is shown
        as it is, and the buttons read themselves again.
        """
        self.set_title(*self._title)
        self.refit_export()
