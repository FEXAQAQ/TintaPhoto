"""The crop box drawn over the photo (PLAN.md 3.2 and 3.6).

A 1 px white outline with L shaped corners, everything outside it darkened,
and the thirds grid only while something is being dragged (C-07). Dragging
inside moves the box, the corners and edges resize it, and dragging outside
turns the photo.

This is not a widget. It draws into the canvas's own painter and is handed
the canvas's mouse events, so that both work in the same coordinates and
neither has to guess where the other put the photo. What it owns is the
interaction; where the crop may be is decided by :mod:`core.geometry`, and
what it becomes is decided by the controller.
"""

from __future__ import annotations

import math
from typing import Final

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QPainter, QPen

from core import geometry
from core.geometry import Geometry, Handle, Rect
from ui import native, theme

#: The thirds grid fades away this long after a drag ends (PLAN.md 3.6).
_GRID_FADE_MS: Final[int] = theme.DURATION_PANEL_MS

#: How far the straighten line must be drawn before it means anything.
_STRAIGHTEN_MINIMUM_PX: Final[float] = 12.0


class CropOverlay(QObject):
    """The crop box, its handles and the straighten line."""

    #: A new crop box, and whether the drag is still going on.
    crop_changed = Signal(object, bool)
    #: A new angle, and whether the drag is still going on.
    angle_changed = Signal(float, bool)
    #: The straighten tool has been used and is no longer armed (C-04).
    straighten_finished = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._active = False
        self._geometry = Geometry()
        self._source = (1, 1)
        self._handle = Handle.NONE
        self._dragging = False
        self._grab_at = QPointF()
        self._crop_at_grab = Rect()
        self._angle_at_grab = 0.0
        self._pointer_angle_at_grab = 0.0
        self._straightening = False
        self._straighten_from: QPointF | None = None
        self._straighten_to: QPointF | None = None
        self._grid_always = False
        self._grid_opacity = 0.0
        self._fade: QVariantAnimation | None = None

    # -- what it is showing ----------------------------------------------

    @property
    def active(self) -> bool:
        """Return whether the crop box is being shown."""
        return self._active

    @property
    def crop(self) -> Rect:
        """Return the crop box now drawn."""
        return self._geometry.crop

    @property
    def straightening(self) -> bool:
        """Return whether the straighten tool is armed (C-04)."""
        return self._straightening

    @property
    def grid_visible(self) -> bool:
        """Return whether the thirds grid is on screen."""
        return self._grid_always or self._grid_opacity > 0.0

    def set_active(self, active: bool) -> None:
        """Show or hide the crop box."""
        if active == self._active:
            return
        self._active = active
        if not active:
            self._handle = Handle.NONE
            self._dragging = False
            self._straightening = False
            self._straighten_from = None
            self._straighten_to = None

    def set_geometry(self, geometry_now: Geometry) -> None:
        """Show the crop box and angle of the photo now open."""
        self._geometry = geometry_now

    def set_source(self, size: tuple[int, int]) -> None:
        """Say how large the photo is, before it was turned or rotated."""
        self._source = size

    def set_grid_always(self, always: bool) -> None:
        """Keep the thirds grid on screen, or let it fade again (O)."""
        self._grid_always = always

    def arm_straighten(self, armed: bool) -> None:
        """Arm or disarm the straighten tool (C-04)."""
        self._straightening = armed
        self._straighten_from = None
        self._straighten_to = None

    # -- drawing ---------------------------------------------------------

    def paint(self, painter: QPainter, photo: QRectF) -> None:
        """Draw the box over the photo, which is drawn in *photo*."""
        if not self._active or photo.isEmpty():
            return
        box = self._box(photo)

        painter.save()
        self._dim_outside(painter, box)
        if self.grid_visible:
            self._draw_grid(painter, box)
        self._draw_outline(painter, box)
        self._draw_corners(painter, box)
        if self._straighten_from is not None and self._straighten_to is not None:
            self._draw_straighten_line(painter)
        painter.restore()

    def _box(self, photo: QRectF) -> QRectF:
        """Return where the crop box is on screen."""
        crop = self._geometry.crop
        return QRectF(
            photo.x() + crop.x * photo.width(),
            photo.y() + crop.y * photo.height(),
            crop.w * photo.width(),
            crop.h * photo.height(),
        )

    def _dim_outside(self, painter: QPainter, box: QRectF) -> None:
        """Darken everything that the crop leaves out (PLAN.md 3.6)."""
        shade = QColor(theme.PREVIEW_BG_BLACK)
        shade.setAlphaF(theme.CROP_OUTSIDE_DIM)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(shade)
        device = painter.device()
        outside = QRectF(0, 0, device.width(), device.height())
        for part in (
            QRectF(outside.x(), outside.y(), outside.width(), box.y() - outside.y()),
            QRectF(outside.x(), box.bottom(), outside.width(), outside.bottom() - box.bottom()),
            QRectF(outside.x(), box.y(), box.x() - outside.x(), box.height()),
            QRectF(box.right(), box.y(), outside.right() - box.right(), box.height()),
        ):
            if part.width() > 0 and part.height() > 0:
                painter.drawRect(part)

    def _draw_outline(self, painter: QPainter, box: QRectF) -> None:
        pen = QPen(QColor(theme.WHITE))
        pen.setWidth(theme.CROP_LINE_WIDTH)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(box)

    def _draw_corners(self, painter: QPainter, box: QRectF) -> None:
        """Draw the four L shaped corner handles (PLAN.md 3.6)."""
        pen = QPen(QColor(theme.WHITE))
        pen.setWidth(theme.CROP_HANDLE_WIDTH)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        length = min(
            theme.CROP_HANDLE_LENGTH, box.width() / 3, box.height() / 3
        )
        inset = theme.CROP_HANDLE_WIDTH / 2
        for x, towards_x in ((box.x() + inset, 1), (box.right() - inset, -1)):
            for y, towards_y in ((box.y() + inset, 1), (box.bottom() - inset, -1)):
                painter.drawLine(QPointF(x, y), QPointF(x + towards_x * length, y))
                painter.drawLine(QPointF(x, y), QPointF(x, y + towards_y * length))

    def _draw_grid(self, painter: QPainter, box: QRectF) -> None:
        """Draw the thirds grid, fading out after a drag (C-07)."""
        colour = QColor(theme.WHITE)
        colour.setAlphaF(1.0 if self._grid_always else self._grid_opacity)
        pen = QPen(colour)
        pen.setWidth(theme.CROP_LINE_WIDTH)
        painter.setPen(pen)
        for step in (1, 2):
            x = box.x() + box.width() * step / 3
            y = box.y() + box.height() * step / 3
            painter.drawLine(QPointF(x, box.y()), QPointF(x, box.bottom()))
            painter.drawLine(QPointF(box.x(), y), QPointF(box.right(), y))

    def _draw_straighten_line(self, painter: QPainter) -> None:
        """Draw the line being drawn along the horizon (C-04)."""
        assert self._straighten_from is not None and self._straighten_to is not None
        pen = QPen(QColor(theme.ACCENT))
        pen.setWidth(theme.CROP_LINE_WIDTH)
        painter.setPen(pen)
        painter.drawLine(self._straighten_from, self._straighten_to)

    # -- the pointer -----------------------------------------------------

    def handle_at(self, point: QPointF, photo: QRectF) -> Handle:
        """Return what the pointer is over."""
        if not self._active or photo.isEmpty():
            return Handle.NONE
        box = self._box(photo)
        grab = theme.CROP_GRAB_RADIUS
        near_left = abs(point.x() - box.x()) <= grab
        near_right = abs(point.x() - box.right()) <= grab
        near_top = abs(point.y() - box.y()) <= grab
        near_bottom = abs(point.y() - box.bottom()) <= grab
        within_x = box.x() - grab <= point.x() <= box.right() + grab
        within_y = box.y() - grab <= point.y() <= box.bottom() + grab

        if within_x and within_y:
            if near_left and near_top:
                return Handle.TOP_LEFT
            if near_right and near_top:
                return Handle.TOP_RIGHT
            if near_left and near_bottom:
                return Handle.BOTTOM_LEFT
            if near_right and near_bottom:
                return Handle.BOTTOM_RIGHT
            if near_left:
                return Handle.LEFT
            if near_right:
                return Handle.RIGHT
            if near_top:
                return Handle.TOP
            if near_bottom:
                return Handle.BOTTOM
        if box.contains(point):
            return Handle.INSIDE
        return Handle.OUTSIDE

    def cursor_for(self, handle: Handle) -> Qt.CursorShape:
        """Return the cursor that says what a handle does."""
        return _CURSORS.get(handle, Qt.CursorShape.ArrowCursor)

    def press(self, point: QPointF, photo: QRectF) -> bool:
        """Take hold of whatever is under the pointer; return whether it did."""
        if not self._active or photo.isEmpty():
            return False
        if self._straightening:
            self._straighten_from = point
            self._straighten_to = point
            return True

        self._handle = self.handle_at(point, photo)
        if self._handle is Handle.NONE:
            return False
        self._dragging = True
        self._grab_at = point
        self._crop_at_grab = self._geometry.crop
        self._angle_at_grab = self._geometry.angle
        if self._handle is Handle.OUTSIDE:
            self._pointer_angle_at_grab = self._angle_around_centre(point, photo)
        self._show_grid()
        return True

    def move(self, point: QPointF, photo: QRectF) -> bool:
        """Carry on the drag; return whether anything was changed."""
        if self._straightening and self._straighten_from is not None:
            self._straighten_to = point
            return True
        if not self._dragging or photo.isEmpty():
            return False

        if self._handle is Handle.OUTSIDE:
            self._rotate_to(point, photo, dragging=True)
            return True

        source = geometry.turned_size(self._source, self._geometry.quarter_turns)
        angle = self._geometry.angle
        if self._handle is Handle.INSIDE:
            delta = (
                (point.x() - self._grab_at.x()) / photo.width(),
                (point.y() - self._grab_at.y()) / photo.height(),
            )
            changed = geometry.move(self._crop_at_grab, delta, source, angle)
        else:
            pointer = (
                (point.x() - photo.x()) / photo.width(),
                (point.y() - photo.y()) / photo.height(),
            )
            ratio = geometry.aspect_ratio(
                self._geometry.aspect, self._geometry.orientation, source
            )
            changed = geometry.resize(
                self._geometry.crop, self._handle, pointer, source, angle, ratio
            )
        if changed == self._geometry.crop:
            return False
        self.crop_changed.emit(changed, True)
        return True

    def release(self, point: QPointF, photo: QRectF) -> bool:
        """Let go; return whether this ended a drag."""
        if self._straightening and self._straighten_from is not None:
            self._finish_straighten()
            return True
        if not self._dragging:
            return False

        self._dragging = False
        handle, self._handle = self._handle, Handle.NONE
        self._hide_grid()
        if handle is Handle.OUTSIDE:
            self._rotate_to(point, photo, dragging=False)
        else:
            # The last value is already shown; this is what makes it one step
            # of history and starts the save (E-02, E-04).
            self.crop_changed.emit(self._geometry.crop, False)
        return True

    def _finish_straighten(self) -> None:
        """Work out the angle from the line that was drawn (C-04)."""
        start, end = self._straighten_from, self._straighten_to
        self._straighten_from = None
        self._straighten_to = None
        self._straightening = False
        self.straighten_finished.emit()
        if start is None or end is None:
            return
        if (end - start).manhattanLength() < _STRAIGHTEN_MINIMUM_PX:
            return
        angle = geometry.straighten_angle(
            (start.x(), start.y()), (end.x(), end.y())
        )
        self.angle_changed.emit(self._geometry.angle + angle, False)

    def _angle_around_centre(self, point: QPointF, photo: QRectF) -> float:
        """Return the angle of the pointer around the middle of the crop box."""
        box = self._box(photo)
        centre = box.center()
        return math.degrees(
            math.atan2(point.y() - centre.y(), point.x() - centre.x())
        )

    def _rotate_to(self, point: QPointF, photo: QRectF, dragging: bool) -> None:
        """Turn the photo by as much as the pointer has gone round."""
        moved = self._angle_around_centre(point, photo) - self._pointer_angle_at_grab
        if moved > 180.0:
            moved -= 360.0
        elif moved < -180.0:
            moved += 360.0
        angle = max(
            geometry.ANGLE_MIN, min(geometry.ANGLE_MAX, self._angle_at_grab + moved)
        )
        self.angle_changed.emit(angle, dragging)

    # -- the grid's fade (PLAN.md 3.6) -----------------------------------

    def _show_grid(self) -> None:
        self._stop_fade()
        self._grid_opacity = 1.0

    def _hide_grid(self) -> None:
        if self._grid_always:
            return
        if not native.animations_enabled():
            self._grid_opacity = 0.0
            return
        self._stop_fade()
        fade = QVariantAnimation(self)
        fade.setDuration(_GRID_FADE_MS)
        fade.setStartValue(1.0)
        fade.setEndValue(0.0)
        fade.valueChanged.connect(self._set_grid_opacity)
        fade.finished.connect(self._stop_fade)
        fade.start()
        self._fade = fade

    def _set_grid_opacity(self, value: float) -> None:
        self._grid_opacity = float(value)
        parent = self.parent()
        if parent is not None and hasattr(parent, "update"):
            parent.update()

    def _stop_fade(self) -> None:
        fade = self._fade
        self._fade = None
        if fade is not None:
            fade.stop()
            fade.deleteLater()


_CURSORS: dict[Handle, Qt.CursorShape] = {
    Handle.INSIDE: Qt.CursorShape.SizeAllCursor,
    Handle.OUTSIDE: Qt.CursorShape.CrossCursor,
    Handle.TOP_LEFT: Qt.CursorShape.SizeFDiagCursor,
    Handle.BOTTOM_RIGHT: Qt.CursorShape.SizeFDiagCursor,
    Handle.TOP_RIGHT: Qt.CursorShape.SizeBDiagCursor,
    Handle.BOTTOM_LEFT: Qt.CursorShape.SizeBDiagCursor,
    Handle.LEFT: Qt.CursorShape.SizeHorCursor,
    Handle.RIGHT: Qt.CursorShape.SizeHorCursor,
    Handle.TOP: Qt.CursorShape.SizeVerCursor,
    Handle.BOTTOM: Qt.CursorShape.SizeVerCursor,
}
