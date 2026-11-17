"""A segmented control: a row of choices where one is taken (PLAN.md 3.11).

The export dialog asks for three of these - the format, the bit depth and
the size - and they are the same control the crop tab uses for its ratios:
one of a set, with the chosen one marked in the accent colour, which is what
PLAN.md 3.4 reserves the accent for.

A choice can also carry a small dot in its top right corner, for "this one
holds something": the curve group marks the curves that have been bent
(PLAN.md 3.1). The dot takes the colour of the name, so it is never the
accent unless the choice is the chosen one. It sits in the corner rather
than after the name because a narrow choice has no room after a long name.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPaintEvent
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QWidget

from app.i18n import tr
from ui import theme
from ui.widgets.buttons import ROLE_CHOICE


class _ChoiceButton(QPushButton):
    """One choice of the row, which can carry a dot in its corner."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._marked = False

    @property
    def marked(self) -> bool:
        """Return whether the dot is shown."""
        return self._marked

    def set_marked(self, marked: bool) -> None:
        """Show or hide the dot in the corner."""
        if marked == self._marked:
            return
        self._marked = marked
        self.update()

    def _mark_colour(self) -> str:
        """Return the colour the name is drawn in, which the dot takes."""
        if not self.isEnabled():
            return theme.TEXT_SECONDARY if self.isChecked() else theme.TEXT_TERTIARY
        if self.isChecked():
            return theme.ACCENT
        if self.underMouse():
            return theme.TEXT_PRIMARY
        return theme.TEXT_SECONDARY

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt naming
        """Draw the button, then the dot inside its top right corner."""
        super().paintEvent(event)
        if not self._marked:
            return
        radius = theme.CHOICE_MARK_RADIUS
        inset = theme.BORDER_WIDTH + theme.CHOICE_MARK_INSET + radius
        centre = QPointF(self.width() - inset, inset)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(self._mark_colour()))
        painter.drawEllipse(centre, radius, radius)


class Segmented(QWidget):
    """One of several choices, laid out in a row of equal buttons."""

    #: The value that was chosen, as it was given in the options.
    chosen = Signal(str)

    def __init__(
        self, options: Sequence[tuple[str, str]], parent: QWidget | None = None
    ) -> None:
        """Build a control from *options*, each a value and a text key."""
        super().__init__(parent)
        self._buttons: dict[str, _ChoiceButton] = {}
        self._label_keys = {value: label_key for value, label_key in options}
        self._value = options[0][0] if options else ""

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_XS)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for index, (value, label_key) in enumerate(options):
            button = _ChoiceButton(tr(label_key), self)
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

    def set_marked(self, value: str, marked: bool) -> None:
        """Show or hide the dot in the corner of the choice *value*."""
        button = self._buttons.get(value)
        if button is not None:
            button.set_marked(marked)

    def is_marked(self, value: str) -> bool:
        """Return whether the choice *value* carries the dot."""
        button = self._buttons.get(value)
        return button is not None and button.marked

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
