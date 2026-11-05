"""The watermark's position picker: three by three squares (PLAN.md 3.14).

One square per anchor of :mod:`core.watermark`, the corners, the middles of
the edges and the centre. The chosen one is filled with the accent colour,
which PLAN.md 3.4 keeps for the current state; once the watermark has been
dragged off every anchor, none is. Clicking a square, or moving with the
arrow keys while the picker has the focus, chooses it.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QToolTip, QWidget

from app.i18n import tr
from core import watermark
from ui import theme

_COLUMNS = 3


class AnchorPicker(QWidget):
    """Nine squares, one per place the watermark can be put."""

    #: The anchor a square was chosen for.
    chosen = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        side = _COLUMNS * theme.ANCHOR_CELL + (_COLUMNS - 1) * theme.ANCHOR_GAP
        self.setFixedSize(side, side)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._anchor: str | None = watermark.ANCHOR_DEFAULT
        self._hovered: int | None = None
        self._pressed: int | None = None

    # -- what it shows -----------------------------------------------------

    @property
    def anchor(self) -> str | None:
        """Return the anchor that is marked, or ``None`` for none."""
        return self._anchor

    def set_anchor(self, anchor: str | None) -> None:
        """Mark *anchor*, or none, without saying it was chosen."""
        self._anchor = anchor if anchor in watermark.ANCHORS else None
        self.update()

    def cell_rect(self, index: int) -> QRectF:
        """Return where square *index* is drawn, counting from the top left."""
        row, column = divmod(index, _COLUMNS)
        step = theme.ANCHOR_CELL + theme.ANCHOR_GAP
        return QRectF(column * step, row * step, theme.ANCHOR_CELL, theme.ANCHOR_CELL)

    def _cell_at(self, point: QPointF) -> int | None:
        for index in range(len(watermark.ANCHORS)):
            if self.cell_rect(index).contains(point):
                return index
        return None

    def _choose(self, index: int) -> None:
        anchor = watermark.ANCHORS[index]
        self.set_anchor(anchor)
        self.chosen.emit(anchor)

    # -- events --------------------------------------------------------------

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt naming
        """Draw the squares in their states: chosen, pressed, hovered or resting."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        enabled = self.isEnabled()
        half = theme.BORDER_WIDTH / 2
        for index, anchor in enumerate(watermark.ANCHORS):
            rect = self.cell_rect(index).adjusted(half, half, -half, -half)
            if anchor == self._anchor:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(theme.ACCENT if enabled else theme.TRACK_HOVER))
            else:
                painter.setPen(
                    QPen(QColor(theme.TRACK_HOVER if enabled else theme.TRACK), theme.BORDER_WIDTH)
                )
                if enabled and index == self._pressed:
                    painter.setBrush(QColor(theme.TRACK_PRESSED))
                elif enabled and index == self._hovered:
                    painter.setBrush(QColor(theme.BG_RAISED))
                else:
                    painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect, theme.RADIUS_CHECK_BOX, theme.RADIUS_CHECK_BOX)
        painter.end()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Light up the square under the pointer."""
        hovered = self._cell_at(event.position())
        if hovered != self._hovered:
            self._hovered = hovered
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Nothing is under the pointer any more."""
        super().leaveEvent(event)
        self._hovered = None
        self._pressed = None
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Press a square down."""
        if event.button() is Qt.MouseButton.LeftButton:
            self._pressed = self._cell_at(event.position())
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Choose the square the press started and ended on."""
        if event.button() is not Qt.MouseButton.LeftButton:
            return
        pressed, self._pressed = self._pressed, None
        if pressed is not None and pressed == self._cell_at(event.position()):
            self._choose(pressed)
        self.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Move the choice with the arrow keys."""
        moves = {
            Qt.Key.Key_Left: (-1, 0),
            Qt.Key.Key_Right: (1, 0),
            Qt.Key.Key_Up: (0, -1),
            Qt.Key.Key_Down: (0, 1),
        }
        move = moves.get(event.key())
        if move is None:
            super().keyPressEvent(event)
            return
        start = self._anchor if self._anchor is not None else watermark.ANCHOR_CENTRE
        row, column = divmod(watermark.ANCHORS.index(start), _COLUMNS)
        column = min(_COLUMNS - 1, max(0, column + move[0]))
        row = min(_COLUMNS - 1, max(0, row + move[1]))
        self._choose(row * _COLUMNS + column)
        event.accept()

    def event(self, event) -> bool:
        """Name the square under the pointer as its tooltip."""
        if event.type() == event.Type.ToolTip:
            index = self._cell_at(QPointF(event.pos()))
            if index is None:
                QToolTip.hideText()
            else:
                QToolTip.showText(
                    event.globalPos(), tr(f"watermark.anchor.{watermark.ANCHORS[index]}"), self
                )
            return True
        return super().event(event)
