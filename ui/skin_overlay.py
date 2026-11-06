"""The portrait brush and the skin it works on, drawn over the photo (PLAN.md 3.15).

While the portrait is open the skin the sliders work on can be shown as a
rose wash over the photo. With the brush taken up the pointer is a circle
the size of the brush with a + or a − in it; holding the button down paints,
washed white when it adds and dark when it takes away, and letting go hands
the stroke over. The wash of the skin is always shown while painting, so
the brush can be steered by it.

Like the spot brush this is not a widget: it draws into the canvas's painter
and is handed the canvas's mouse events. What it holds is what is on screen
- points in fractions of the frame as shown, the radius as a fraction of
that frame's long edge, and the skin as a picture of that frame - and what a
stroke is on the unturned photo is the controller's business.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen

from ui import theme
from ui.spot_overlay import _at, _fraction_of, _outline, _radius

Point = tuple[float, float]


class SkinOverlay(QObject):
    """The portrait brush, what it is painting, and the skin it works on."""

    #: A stroke painted: its path, in fractions of the frame as shown.
    stroke_painted = Signal(list)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        #: Whether the portrait is open, which is when anything is drawn.
        self._open = False
        #: Whether the brush is taken up, and whether it adds or takes away.
        self._brush_on = False
        self._adding = True
        #: The brush's radius, as a fraction of the shown frame's long edge.
        self._brush = 0.0
        #: Whether the skin is shown when the brush is not down.
        self._showing = False
        #: The skin, as a rose picture of the frame, or ``None``.
        self._skin: QImage | None = None
        self._pointer: QPointF | None = None
        self._painting: list[Point] | None = None

    # -- what it is showing ----------------------------------------------

    @property
    def active(self) -> bool:
        """Return whether clicks on the photo paint."""
        return self._open and self._brush_on

    @property
    def painting(self) -> bool:
        """Return whether the brush is down."""
        return self._painting is not None

    @property
    def showing(self) -> bool:
        """Return whether the skin is shown over the photo."""
        return self._open and self._showing

    @property
    def has_skin(self) -> bool:
        """Return whether there is a picture of the skin to show."""
        return self._skin is not None

    def set_open(self, open_: bool) -> None:
        """Say whether the portrait is open; closed draws nothing and forgets the paint."""
        self._open = open_
        if not open_:
            self._painting = None

    def set_brush(self, on: bool, adding: bool, radius: float) -> None:
        """Take the brush up or put it down, and say what it does and how big it is."""
        self._brush_on = on
        self._adding = adding
        self._brush = max(0.0, radius)
        if not on:
            self._painting = None

    def set_showing(self, showing: bool) -> None:
        """Say whether the skin is shown when the brush is not down."""
        self._showing = showing

    def set_skin(self, skin: NDArray[np.uint8] | None) -> None:
        """Take the skin to show: 0 to 255 for each pixel of the frame, or ``None``."""
        if skin is None or skin.size == 0:
            self._skin = None
            return
        height, width = skin.shape[:2]
        alpha = np.ascontiguousarray(skin, dtype=np.uint8)
        mask = QImage(alpha.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
        picture = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
        picture.fill(QColor(theme.SKIN_RANGE))
        picture.setAlphaChannel(mask)
        self._skin = picture

    def set_pointer(self, point: QPointF | None) -> None:
        """Say where the pointer is, so the brush can be drawn under it."""
        self._pointer = point

    # -- drawing ---------------------------------------------------------

    def paint(self, painter: QPainter, photo: QRectF) -> None:
        """Draw the skin, the paint and the brush over the photo at *photo*."""
        if not self._open or photo.isEmpty():
            return
        painter.save()
        painter.setClipRect(photo)
        if self._skin is not None and (self._showing or self._painting is not None):
            painter.setOpacity(theme.SKIN_RANGE_ALPHA / 255.0)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(photo, self._skin, QRectF(self._skin.rect()))
            painter.setOpacity(1.0)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        brush = _radius(photo, self._brush)
        if self._painting is not None:
            wash = QColor(theme.WHITE if self._adding else theme.SKIN_REMOVE_WASH)
            wash.setAlpha(theme.SPOT_PAINT_ALPHA if self._adding else theme.SKIN_REMOVE_ALPHA)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(wash)
            painter.drawPath(_outline([_at(photo, point) for point in self._painting], brush))
            painter.setBrush(Qt.BrushStyle.NoBrush)
        if self.active and self._pointer is not None and photo.contains(self._pointer):
            self._draw_brush(painter, self._pointer, brush)
        painter.restore()

    def _draw_brush(self, painter: QPainter, centre: QPointF, radius: float) -> None:
        """Draw the brush's circle, with a + when it adds and a − when it takes away."""
        shape = QPainterPath()
        shape.addEllipse(centre, radius, radius)
        mark = float(theme.SKIN_BRUSH_MARK)
        shape.moveTo(centre.x() - mark, centre.y())
        shape.lineTo(centre.x() + mark, centre.y())
        if self._adding:
            shape.moveTo(centre.x(), centre.y() - mark)
            shape.lineTo(centre.x(), centre.y() + mark)
        halo = QColor(theme.SPOT_HALO)
        halo.setAlpha(theme.SPOT_HALO_ALPHA)
        painter.setPen(QPen(halo, theme.SPOT_LINE_WIDTH + 2 * theme.SPOT_HALO_WIDTH))
        painter.drawPath(shape)
        painter.setPen(QPen(QColor(theme.WHITE), theme.SPOT_LINE_WIDTH))
        painter.drawPath(shape)

    # -- the mouse -------------------------------------------------------

    def press(self, point: QPointF, photo: QRectF) -> bool:
        """Start painting; return whether the press was used."""
        if not self.active or photo.isEmpty() or not photo.contains(point):
            return False
        self._painting = [_fraction_of(photo, point)]
        return True

    def move(self, point: QPointF, photo: QRectF) -> bool:
        """Follow the pointer, painting while the button is down; return whether painting."""
        self._pointer = point
        if self._painting is None or photo.isEmpty():
            return False
        last = _at(photo, self._painting[-1])
        step = max(theme.SPOT_PAINT_STEP, _radius(photo, self._brush) * 0.2)
        if math.hypot(point.x() - last.x(), point.y() - last.y()) >= step:
            self._painting.append(_fraction_of(photo, point))
        return True

    def release(self, point: QPointF, photo: QRectF) -> bool:
        """Hand over what was painted; return whether there was any."""
        self._pointer = point
        if self._painting is None:
            return False
        painted = self._painting
        self._painting = None
        if not photo.isEmpty() and photo.contains(point):
            last = _fraction_of(photo, point)
            if last != painted[-1]:
                painted.append(last)
        self.stroke_painted.emit(painted)
        return True
