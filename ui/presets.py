"""The pieces of the saved presets that sit in the adjust tab (P-04 to P-06).

A preset is a set of slider values, so its list lives with the sliders: the
top group of the adjust tab, where clicking one moves the sliders right
underneath it. It is an action rather than a state - the photo does not
remember which preset it came from - so a row is a plain button that is
never marked. Reading and writing the files is :mod:`core.presets`, and
putting one on a photo is the controller.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.i18n import tr
from core.presets import NAME_MAX_LENGTH, Preset
from ui import theme
from ui.widgets.buttons import ROLE_CHOICE
from ui.widgets.dialog import Dialog
from ui.widgets.focus import ring_on_keyboard_only


class PresetRow(QPushButton):
    """One saved preset: click to put it on, right click to rename or delete."""

    #: This preset was asked about, at a point on the screen.
    menu_requested = Signal(object, QPoint)

    def __init__(self, preset: Preset, parent: QWidget | None = None) -> None:
        super().__init__(preset.name, parent)
        self.preset = preset
        self.setProperty("role", ROLE_CHOICE)
        # A list of names reads from the left edge, unlike the short choices
        # of the crop tab, which sit centred in a grid.
        self.setProperty("listed", True)
        self.setFixedHeight(theme.BUTTON_HEIGHT)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setToolTip(tr("presets.hint"))
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(
            lambda point: self.menu_requested.emit(self.preset, self.mapToGlobal(point))
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Cut a name too long for the row short with an ellipsis."""
        super().resizeEvent(event)
        room = self.width() - 2 * theme.SPACE_SM
        self.setText(
            self.fontMetrics().elidedText(self.preset.name, Qt.TextElideMode.ElideRight, room)
        )

    def retranslate(self) -> None:
        """Read the hint again after a language change (U-02).

        The name of the row is the user's own name for the preset, so only
        the tooltip changes with the language.
        """
        self.setToolTip(tr("presets.hint"))


class NameDialog(Dialog):
    """Asks what to call a preset (P-04, P-06)."""

    def __init__(
        self, title_key: str, name: str = "", parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr(title_key))
        self.setModal(True)
        self.setMinimumWidth(theme.NAME_DIALOG_MIN_WIDTH)

        self.field = QLineEdit(name, self)
        ring_on_keyboard_only(self.field)
        self.field.setMaxLength(NAME_MAX_LENGTH)
        self.field.setPlaceholderText(tr("presets.name_placeholder"))
        self.field.selectAll()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr("dialog.save"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(
            tr("panel.action.cancel")
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            theme.PANEL_PADDING, theme.PANEL_PADDING, theme.PANEL_PADDING, theme.PANEL_PADDING
        )
        layout.setSpacing(theme.SPACE_MD)
        label = QLabel(tr("presets.name_label"), self)
        label.setProperty("role", "caption")
        layout.addWidget(label)
        layout.addWidget(self.field)
        layout.addWidget(buttons)

    @property
    def name(self) -> str:
        """Return what was typed."""
        return self.field.text()

    @classmethod
    def ask(cls, title_key: str, name: str = "", parent: QWidget | None = None) -> str | None:
        """Show the dialog and return the name, or ``None`` if it was cancelled."""
        dialog = cls(title_key, name, parent)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.name.strip() or None
