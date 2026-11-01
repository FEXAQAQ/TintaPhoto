"""A switch for a setting that is either on or off (PLAN.md 3.10, 3.11).

Drawn rather than built from a checkbox, so that its hover, pressed and
disabled states are exactly the ones PLAN.md 3.8 asks for and every colour
and size comes from :mod:`ui.theme`. Being on is a current state, which is
what the accent colour is for (PLAN.md 3.4).
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QAbstractButton, QWidget

from ui import theme


class Switch(QAbstractButton):
    """A small two state switch; ``toggled`` says when it changes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(theme.SWITCH_WIDTH, theme.SWITCH_HEIGHT)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Draw the track and the knob, in the state the switch is in."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(self._track_colour()))
        radius = self.height() / 2
        painter.drawRoundedRect(self.rect(), radius, radius)

        knob = theme.SWITCH_KNOB_SIZE // 2
        inset = (self.height() - theme.SWITCH_KNOB_SIZE) // 2 + knob
        centre = QPoint(
            self.width() - inset if self.isChecked() else inset, self.height() // 2
        )
        painter.setBrush(QColor(self._knob_colour()))
        painter.drawEllipse(centre, knob, knob)

        if self.hasFocus():
            # The ring goes dark over a switch that is on, which is painted
            # in the accent colour itself.
            ring = theme.BG_APP if self.isChecked() else theme.ACCENT
            painter.setPen(QPen(QColor(ring), theme.FOCUS_RING_WIDTH))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(
                QRect(0, 0, self.width() - 1, self.height() - 1), radius, radius
            )

    def _track_colour(self) -> str:
        """Return the colour of the track in the state the switch is in."""
        if not self.isEnabled():
            return theme.BORDER
        if self.isChecked():
            return theme.ACCENT
        if self.isDown():
            return theme.BG_RAISED
        return theme.BG_RAISED if self.underMouse() else theme.TRACK

    def _knob_colour(self) -> str:
        """Return the colour of the knob in the state the switch is in."""
        if not self.isEnabled():
            return theme.TEXT_TERTIARY
        if self.isChecked():
            return theme.BG_APP
        if self.isDown():
            return theme.TEXT_SECONDARY
        return theme.TEXT_PRIMARY if self.underMouse() else theme.TEXT_SECONDARY

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Brighten the switch while the pointer is on it (PLAN.md 3.6)."""
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Go back to resting when the pointer leaves."""
        super().leaveEvent(event)
        self.update()
