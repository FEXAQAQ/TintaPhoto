"""A dropdown with an arrow that is drawn rather than loaded (PLAN.md 3.10).

Qt's stylesheet can only put a picture in the corner of a combo box, and
this project ships no picture files for the interface: the icons are drawn
(PLAN.md 13.2). So the arrow is drawn here, in the colours of the theme,
and changes with the state of the control like every other element.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import QComboBox, QWidget

from ui import theme
from ui.widgets.focus import ring_on_keyboard_only

#: Half the width of the arrow, and how far it drops, in pixels.
_ARROW_HALF_WIDTH = 4
_ARROW_DEPTH = 2


class ComboBox(QComboBox):
    """A dropdown that draws its own arrow."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        ring_on_keyboard_only(self)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Draw the box from the stylesheet, then the arrow on top of it."""
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        colour = theme.TEXT_SECONDARY if self.isEnabled() else theme.TEXT_TERTIARY
        pen = QPen(colour)
        pen.setWidthF(theme.ICON_STROKE_WIDTH)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)

        middle = QPoint(self.width() - theme.SPACE_MD, self.height() // 2)
        painter.drawPolyline(
            [
                middle + QPoint(-_ARROW_HALF_WIDTH, -_ARROW_DEPTH),
                middle + QPoint(0, _ARROW_DEPTH),
                middle + QPoint(_ARROW_HALF_WIDTH, -_ARROW_DEPTH),
            ]
        )
