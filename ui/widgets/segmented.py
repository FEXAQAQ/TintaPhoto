"""A segmented control: a row of choices where one is taken (PLAN.md 3.11).

The export dialog asks for three of these - the format, the bit depth and
the size - and they are the same control the crop tab uses for its ratios:
one of a set, with the chosen one marked in the accent colour, which is what
PLAN.md 3.4 reserves the accent for.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QWidget

from app.i18n import tr
from ui import theme
from ui.widgets.buttons import ROLE_CHOICE


class Segmented(QWidget):
    """One of several choices, laid out in a row of equal buttons."""

    #: The value that was chosen, as it was given in the options.
    chosen = Signal(str)

    def __init__(
        self, options: Sequence[tuple[str, str]], parent: QWidget | None = None
    ) -> None:
        """Build a control from *options*, each a value and a text key."""
        super().__init__(parent)
        self._buttons: dict[str, QPushButton] = {}
        self._label_keys = {value: label_key for value, label_key in options}
        self._value = options[0][0] if options else ""

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_XS)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for index, (value, label_key) in enumerate(options):
            button = QPushButton(tr(label_key), self)
            button.setProperty("role", ROLE_CHOICE)
            button.setCheckable(True)
            button.setFixedHeight(theme.BUTTON_HEIGHT)
            button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
            button.clicked.connect(lambda _checked, name=value: self._take(name))
            self._group.addButton(button, index)
            self._buttons[value] = button
            layout.addWidget(button, 1)
        self.set_value(self._value)

    @property
    def value(self) -> str:
        """Return the value that is chosen."""
        return self._value

    def set_value(self, value: str) -> None:
        """Show *value* as the chosen one, without saying it was chosen."""
        if value not in self._buttons:
            return
        self._value = value
        for name, button in self._buttons.items():
            button.setChecked(name == value)

    def retranslate(self) -> None:
        """Read every choice's name again after a language change."""
        for value, button in self._buttons.items():
            button.setText(tr(self._label_keys[value]))

    def buttons(self) -> dict[str, QPushButton]:
        """Return the buttons by their value, for the window and the tests."""
        return dict(self._buttons)

    def _take(self, value: str) -> None:
        """Take on the choice that was clicked and say so."""
        if value == self._value:
            # Clicking the one already chosen leaves it chosen; it must not
            # look as though something changed.
            self.set_value(value)
            return
        self.set_value(value)
        self.chosen.emit(value)
