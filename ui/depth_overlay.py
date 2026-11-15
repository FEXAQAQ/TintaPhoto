"""The depth of field's sharp shape and brush, drawn over the photo (PLAN.md 3.18).

While the depth of field is open its sharp shape is outlined over the photo:
a solid line where the photo stops being sharp, a dashed one where the blur
is all there, and handles to take hold of. The middle moves the shape - so
does a press anywhere inside it - the handle at the end of the long axis
turns it and sizes it, and for an ellipse a third handle on the short axis
makes it rounder or narrower. With the brush taken up the pointer is a circle
the size of the brush with a + (keep sharp) or a − (blur) in it, and holding
the button down paints, as the portrait's brush does.

The range can be shown as a wash over what is blurred, as the portrait shows
its skin, and is always shown while painting.

Like the other overlays this is not a widget: it draws into the canvas's
painter and is handed the canvas's mouse events. What it holds is what is on
screen - points in fractions of the frame as shown - and what that is on
the photo is the controller's business. A press away from the shape is not
taken, so the photo can still be panned.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QTransform

from core import depth_of_field
from core.depth_of_field import DepthOfField
from ui import theme
from ui.spot_overlay import _at, _fraction_of, _outline, _radius

Point = tuple[float, float]

#: The handles that can be taken hold of, and the shape's inside.
HANDLE_CENTRE: Final[str] = "centre"
HANDLE_AXIS: Final[str] = "axis"
HANDLE_RATIO: Final[str] = "ratio"
INSIDE: Final[str] = "inside"


class DepthOverlay(QObject):
    """The sharp shape and its handles, the brush, and the range it all makes."""

    #: The shape was moved: its middle and its handle in fractions of the
    #: frame as shown, the ellipse's ratio, and whether the drag goes on.
    focus_changed = Signal(object, object, float, bool)
    #: A stroke painted: its path, in fractions of the frame as shown.
    stroke_painted = Signal(list)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._open = False
        #: The depth of field as it lies on the frame shown.
        self._depth = depth_of_field.DEFAULT_DEPTH
        self._brush_on = False
        self._sharp = True
        self._brush = 0.0
        self._showing = False
        self._range: QImage | None = None
        self._pointer: QPointF | None = None
        self._painting: list[Point] | None = None
        #: What is being dragged, where the press was and what the shape
        #: was then; and what is under the pointer.
        self._held: str | None = None
        self._held_from: QPointF | None = None
        self._held_depth = self._depth
        self._hovered: str | None = None

    # -- what it is showing ----------------------------------------------

    @property
    def open(self) -> bool:
        """Return whether the depth of field is open, which is when anything is drawn."""
        return self._open

    @property
    def active(self) -> bool:
        """Return whether clicks on the photo paint."""
        return self._open and self._brush_on

    @property
    def painting(self) -> bool:
        """Return whether the brush is down."""
        return self._painting is not None

    @property
    def dragging(self) -> bool:
        """Return whether the shape is being dragged."""
        return self._held is not None

    @property
    def showing(self) -> bool:
        """Return whether the range is shown over the photo."""
        return self._open and self._showing

    def set_open(self, open_: bool) -> None:
        """Say whether the depth of field is open; closed draws nothing and lets go."""
        self._open = open_
        if not open_:
            self._painting = None
            self._held = None
            self._hovered = None

    def set_depth(self, depth: DepthOfField) -> None:
        """Take the depth of field as it lies on the frame shown."""
        if self._held is None:
            self._depth = depth
        else:
            # A drag keeps its own shape; the sliders may still change.
            self._depth = replace(
                depth,
                centre=self._depth.centre,
                handle=self._depth.handle,
                ratio=self._depth.ratio,
            )

    def set_brush(self, on: bool, sharp: bool, radius: float) -> None:
        """Take the brush up or put it down, and say what it does and how big it is."""
        self._brush_on = on
        self._sharp = sharp
        self._brush = max(0.0, radius)
        if not on:
            self._painting = None

    def set_showing(self, showing: bool) -> None:
        """Say whether the range is shown when the brush is not down."""
        self._showing = showing

    def set_range(self, blurred: NDArray[np.uint8] | None) -> None:
        """Take how blurred the frame is: 0 to 255 for each pixel, or ``None``."""
        if blurred is None or blurred.size == 0:
            self._range = None
            return
        height, width = blurred.shape[:2]
        alpha = np.ascontiguousarray(blurred, dtype=np.uint8)
        mask = QImage(alpha.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
        picture = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
        picture.fill(QColor(theme.DEPTH_RANGE))
        picture.setAlphaChannel(mask)
        self._range = picture

    def set_pointer(self, point: QPointF | None) -> None:
        """Say where the pointer is, so the brush can be drawn under it."""
        self._pointer = point

    # -- the shape on screen ---------------------------------------------

    def _axes(self, photo: QRectF) -> tuple[QPointF, QPointF, float, float]:
        """Return the middle, the long axis's unit direction, its length and the short one's."""
        centre = _at(photo, self._depth.centre)
        handle = _at(photo, self._depth.handle)
        dx, dy = handle.x() - centre.x(), handle.y() - centre.y()
        length = math.hypot(dx, dy)
        least = depth_of_field.SIZE_MIN * max(photo.width(), photo.height())
        if length < 1e-6:
            dx, dy, length = 0.0, -1.0, least
        else:
            dx, dy = dx / length, dy / length
        length = max(length, least)
        return centre, QPointF(dx, dy), length, max(length * self._depth.ratio, least)

    def _handles(self, photo: QRectF) -> dict[str, QPointF]:
        """Return where each handle is drawn."""
        centre, along, length, short = self._axes(photo)
        handles = {
            HANDLE_CENTRE: centre,
            HANDLE_AXIS: QPointF(centre.x() + along.x() * length, centre.y() + along.y() * length),
        }
        if self._depth.shape == depth_of_field.SHAPE_ELLIPSE:
            handles[HANDLE_RATIO] = QPointF(
                centre.x() - along.y() * short, centre.y() + along.x() * short
            )
        return handles

    def _reach(self, photo: QRectF) -> float:
        """Return how far beyond the shape the blur is all there, in screen pixels."""
        share = depth_of_field.transition_reach(self._depth.transition)
        return share * max(photo.width(), photo.height())

    def _shape_path(self, photo: QRectF, grown: float) -> QPainterPath:
        """Return the sharp shape's outline, *grown* pixels out from it."""
        centre, along, length, short = self._axes(photo)
        path = QPainterPath()
        if self._depth.shape == depth_of_field.SHAPE_ELLIPSE:
            path.addEllipse(QPointF(0.0, 0.0), length + grown, short + grown)
        else:
            # Two lines across the whole photo, either side of the middle.
            far = 2.0 * math.hypot(photo.width(), photo.height())
            for side in (-1.0, 1.0):
                path.moveTo(side * (length + grown), -far)
                path.lineTo(side * (length + grown), far)
        turn = QTransform()
        turn.translate(centre.x(), centre.y())
        turn.rotateRadians(math.atan2(along.y(), along.x()))
        return turn.map(path)

    def _hit(self, point: QPointF, photo: QRectF) -> str | None:
        """Return what a press at *point* takes hold of, if anything."""
        grab = float(theme.DEPTH_HANDLE_GRAB)
        for name, where in self._handles(photo).items():
            if math.hypot(point.x() - where.x(), point.y() - where.y()) <= grab:
                return name
        centre, along, length, short = self._axes(photo)
        dx, dy = point.x() - centre.x(), point.y() - centre.y()
        u = dx * along.x() + dy * along.y()
        if self._depth.shape == depth_of_field.SHAPE_BAND:
            return INSIDE if abs(u) <= length and photo.contains(point) else None
        v = -dx * along.y() + dy * along.x()
        return INSIDE if (u / length) ** 2 + (v / short) ** 2 <= 1.0 else None

    def over(self, point: QPointF, photo: QRectF) -> bool:
        """Return whether a press at *point* would take hold of the shape."""
        return self._open and not self._brush_on and self._hit(point, photo) is not None

    # -- drawing ---------------------------------------------------------

    def paint(self, painter: QPainter, photo: QRectF) -> None:
        """Draw the range, the shape, the paint and the brush over the photo at *photo*."""
        if not self._open or photo.isEmpty():
            return
        painter.save()
        painter.setClipRect(photo)
        if self._range is not None and (self._showing or self._painting is not None):
            painter.setOpacity(theme.DEPTH_RANGE_ALPHA / 255.0)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(photo, self._range, QRectF(self._range.rect()))
            painter.setOpacity(1.0)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._painting is None:
            self._stroke(painter, self._shape_path(photo, 0.0), Qt.PenStyle.SolidLine)
            self._stroke(painter, self._shape_path(photo, self._reach(photo)), Qt.PenStyle.DashLine)
            if not self._brush_on:
                self._draw_handles(painter, photo)
        brush = _radius(photo, self._brush)
        if self._painting is not None:
            wash = QColor(theme.WHITE if self._sharp else theme.SKIN_REMOVE_WASH)
            wash.setAlpha(theme.SPOT_PAINT_ALPHA if self._sharp else theme.SKIN_REMOVE_ALPHA)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(wash)
            painter.drawPath(_outline([_at(photo, point) for point in self._painting], brush))
            painter.setBrush(Qt.BrushStyle.NoBrush)
        if self.active and self._pointer is not None and photo.contains(self._pointer):
            self._draw_brush(painter, self._pointer, brush)
        painter.restore()

    def _stroke(self, painter: QPainter, path: QPainterPath, style: Qt.PenStyle) -> None:
        """Draw *path* as a white line with a dark edge either side, like the spot brush."""
        halo = QColor(theme.SPOT_HALO)
        halo.setAlpha(theme.SPOT_HALO_ALPHA)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        edge = QPen(halo, theme.SPOT_LINE_WIDTH + 2 * theme.SPOT_HALO_WIDTH)
        edge.setStyle(style)
        painter.setPen(edge)
        painter.drawPath(path)
        line = QPen(QColor(theme.WHITE), theme.SPOT_LINE_WIDTH)
        line.setStyle(style)
        painter.setPen(line)
        painter.drawPath(path)

    def _draw_handles(self, painter: QPainter, photo: QRectF) -> None:
        """Draw each handle as a small ring, filled while under the pointer or held."""
        size = float(theme.DEPTH_HANDLE_SIZE)
        lit = self._held if self._held is not None else self._hovered
        halo = QColor(theme.SPOT_HALO)
        halo.setAlpha(theme.SPOT_HALO_ALPHA)
        for name, where in self._handles(photo).items():
            painter.setPen(QPen(halo, theme.SPOT_LINE_WIDTH + 2 * theme.SPOT_HALO_WIDTH))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(where, size / 2, size / 2)
            painter.setPen(QPen(QColor(theme.WHITE), theme.SPOT_LINE_WIDTH))
            filled = name == lit or (name == HANDLE_CENTRE and lit == INSIDE)
            painter.setBrush(QColor(theme.WHITE) if filled else Qt.BrushStyle.NoBrush)
            painter.drawEllipse(where, size / 2, size / 2)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_brush(self, painter: QPainter, centre: QPointF, radius: float) -> None:
        """Draw the brush's circle, with a + when it keeps sharp and a − when it blurs."""
        shape = QPainterPath()
        shape.addEllipse(centre, radius, radius)
        mark = float(theme.SKIN_BRUSH_MARK)
        shape.moveTo(centre.x() - mark, centre.y())
        shape.lineTo(centre.x() + mark, centre.y())
        if self._sharp:
            shape.moveTo(centre.x(), centre.y() - mark)
            shape.lineTo(centre.x(), centre.y() + mark)
        self._stroke(painter, shape, Qt.PenStyle.SolidLine)

    # -- the mouse -------------------------------------------------------

    def press(self, point: QPointF, photo: QRectF) -> bool:
        """Start painting, or take hold of the shape; return whether the press was used."""
        if not self._open or photo.isEmpty():
            return False
        if self._brush_on:
            if not photo.contains(point):
                return False
            self._painting = [_fraction_of(photo, point)]
            return True
        held = self._hit(point, photo)
        if held is None:
            return False
        self._held = held
        self._held_from = point
        self._held_depth = self._depth
        return True

    def move(self, point: QPointF, photo: QRectF) -> bool:
        """Follow the pointer, painting or dragging; return whether anything changed."""
        self._pointer = point
        if not self._open or photo.isEmpty():
            return False
        if self._painting is not None:
            last = _at(photo, self._painting[-1])
            step = max(theme.SPOT_PAINT_STEP, _radius(photo, self._brush) * 0.2)
            if math.hypot(point.x() - last.x(), point.y() - last.y()) >= step:
                self._painting.append(_fraction_of(photo, point))
            return True
        if self._held is None:
            hovered = None if self._brush_on else self._hit(point, photo)
            changed = hovered != self._hovered
            self._hovered = hovered
            return changed
        self._drag_to(point, photo)
        self._announce(dragging=True)
        return True

    def release(self, point: QPointF, photo: QRectF) -> bool:
        """Hand over what was painted or where the shape was left; return whether either."""
        self._pointer = point
        if self._painting is not None:
            painted = self._painting
            self._painting = None
            if not photo.isEmpty() and photo.contains(point):
                last = _fraction_of(photo, point)
                if last != painted[-1]:
                    painted.append(last)
            self.stroke_painted.emit(painted)
            return True
        if self._held is None:
            return False
        if not photo.isEmpty():
            self._drag_to(point, photo)
        self._announce(dragging=False)
        self._held = None
        self._held_from = None
        return True

    def leave(self) -> None:
        """Forget what was under the pointer once it has left the canvas."""
        self._pointer = None
        self._hovered = None

    def _drag_to(self, point: QPointF, photo: QRectF) -> None:
        """Move the held part of the shape to follow the pointer."""
        assert self._held is not None and self._held_from is not None
        start = self._held_depth
        if self._held in (HANDLE_CENTRE, INSIDE):
            dx = (point.x() - self._held_from.x()) / photo.width()
            dy = (point.y() - self._held_from.y()) / photo.height()
            self._depth = self._replaced(
                centre=(start.centre[0] + dx, start.centre[1] + dy),
                handle=(start.handle[0] + dx, start.handle[1] + dy),
            )
            return
        centre = _at(photo, start.centre)
        if self._held == HANDLE_AXIS:
            self._depth = self._replaced(handle=_fraction_of(photo, point))
            return
        # The ratio handle: how far the pointer is from the long axis.
        held = self._axes_of(start, photo)
        along = held[1]
        dx, dy = point.x() - centre.x(), point.y() - centre.y()
        across = abs(-dx * along.y() + dy * along.x())
        ratio = across / max(held[2], 1e-6)
        ratio = min(max(ratio, depth_of_field.RATIO_MIN), depth_of_field.RATIO_MAX)
        self._depth = self._replaced(ratio=ratio)

    def _axes_of(self, depth: DepthOfField, photo: QRectF) -> tuple[QPointF, QPointF, float, float]:
        """Return :meth:`_axes` for *depth* rather than for the shape held now."""
        kept = self._depth
        self._depth = depth
        try:
            return self._axes(photo)
        finally:
            self._depth = kept

    def _replaced(self, **changes: Any) -> DepthOfField:
        """Return the shape now with *changes*."""
        return replace(self._depth, **changes)

    def _announce(self, dragging: bool) -> None:
        """Say where the shape is now."""
        depth = self._depth
        self.focus_changed.emit(depth.centre, depth.handle, depth.ratio, dragging)
