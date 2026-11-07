"""The liquify brush, drawn over the photo (PLAN.md 3.16).

While liquify is open the pointer is a circle the size of the brush with a
dot in its middle. Holding the button down paints: a push follows the
pointer, and the other tools also take the pointer's place
:data:`core.liquify.TICKS_PER_SECOND` times a second while it is held still,
which is what makes a swell or a twirl keep working. Every new point hands
the stroke so far over for a draft, and letting go hands over the stroke.

Like the spot brush this is not a widget: it draws into the canvas's painter
and is handed the canvas's mouse events. What it holds is what is on screen
- points in fractions of the frame as shown, the radius as a fraction of
that frame's long edge - and what a stroke is on the unturned photo is the
controller's business.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen

from core import liquify
from ui import theme
from ui.spot_overlay import _at, _fraction_of, _radius

Point = tuple[float, float]

#: How long a held brush waits before it takes the pointer's place again.
TICK_MS = round(1000 / liquify.TICKS_PER_SECOND)


class LiquifyOverlay(QObject):
    """The liquify brush and the stroke it is painting."""

    #: The stroke so far, in fractions of the frame as shown, for a draft.
    stroke_changed = Signal(list)
    #: A stroke painted: its path, in fractions of the frame as shown.
    stroke_painted = Signal(list)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        #: Whether liquify is open, which is when the brush is drawn and paints.
        self._open = False
        self._tool = liquify.TOOL_PUSH
        #: The brush's radius, as a fraction of the shown frame's long edge.
        self._brush = 0.0
        self._pointer: QPointF | None = None
        self._photo = QRectF()
        self._painting: list[Point] | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self._tick)

    # -- what it is showing ----------------------------------------------

    @property
    def active(self) -> bool:
        """Return whether clicks on the photo paint."""
        return self._open

    @property
    def painting(self) -> bool:
        """Return whether the brush is down."""
        return self._painting is not None

    def set_open(self, open_: bool) -> None:
        """Say whether liquify is open; closed draws nothing and drops the stroke."""
        self._open = open_
        if not open_:
            self._stop()

    def set_brush(self, tool: str, radius: float) -> None:
        """Say which tool the brush is and how big it is on screen."""
        self._tool = tool
        self._brush = max(0.0, radius)

    def set_pointer(self, point: QPointF | None) -> None:
        """Say where the pointer is, so the brush can be drawn under it."""
        self._pointer = point

    # -- drawing ---------------------------------------------------------

    def paint(self, painter: QPainter, photo: QRectF) -> None:
        """Draw the brush under the pointer, over the photo at *photo*."""
        if not self._open or photo.isEmpty() or self._pointer is None:
            return
        if not photo.contains(self._pointer) and self._painting is None:
            return
        painter.save()
        painter.setClipRect(photo)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        centre = self._pointer
        radius = _radius(photo, self._brush)
        shape = QPainterPath()
        shape.addEllipse(centre, radius, radius)
        dot = theme.LIQUIFY_BRUSH_DOT / 2
        shape.addEllipse(centre, dot, dot)
        halo = QColor(theme.SPOT_HALO)
        halo.setAlpha(theme.SPOT_HALO_ALPHA)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(halo, theme.SPOT_LINE_WIDTH + 2 * theme.SPOT_HALO_WIDTH))
        painter.drawPath(shape)
        painter.setPen(QPen(QColor(theme.WHITE), theme.SPOT_LINE_WIDTH))
        painter.drawPath(shape)
        painter.restore()

    # -- the mouse -------------------------------------------------------

    def press(self, point: QPointF, photo: QRectF) -> bool:
        """Start painting; return whether the press was used."""
        if not self._open or photo.isEmpty() or not photo.contains(point):
            return False
        self._pointer = point
        self._photo = QRectF(photo)
        self._painting = [_fraction_of(photo, point)]
        if self._tool != liquify.TOOL_PUSH:
            # Only a push needs the pointer to move; the rest work while held.
            self._timer.start()
            self.stroke_changed.emit(list(self._painting))
        return True

    def move(self, point: QPointF, photo: QRectF) -> bool:
        """Follow the pointer, painting a push while the button is down."""
        self._pointer = point
        if self._painting is None or photo.isEmpty():
            return False
        self._photo = QRectF(photo)
        if self._tool != liquify.TOOL_PUSH:
            return True
        last = _at(photo, self._painting[-1])
        step = max(theme.SPOT_PAINT_STEP, _radius(photo, self._brush) * 0.1)
        if math.hypot(point.x() - last.x(), point.y() - last.y()) >= step:
            self._painting.append(_fraction_of(photo, point))
            self.stroke_changed.emit(list(self._painting))
        return True

    def release(self, point: QPointF, photo: QRectF) -> bool:
        """Hand over what was painted; return whether there was any."""
        self._pointer = point
        if self._painting is None:
            return False
        painted = self._painting
        self._stop()
        if not photo.isEmpty() and self._tool == liquify.TOOL_PUSH:
            last = _fraction_of(photo, point)
            if last != painted[-1]:
                painted.append(last)
        self.stroke_painted.emit(painted)
        return True

    def _tick(self) -> None:
        """Take the pointer's place again while a brush other than the push is held."""
        if self._painting is None or self._pointer is None or self._photo.isEmpty():
            return
        if len(self._painting) >= liquify.POINTS_MAX:
            return
        self._painting.append(_fraction_of(self._photo, self._pointer))
        self.stroke_changed.emit(list(self._painting))

    def _stop(self) -> None:
        """Put the brush down without handing anything over."""
        self._timer.stop()
        self._painting = None
