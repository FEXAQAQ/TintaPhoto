"""The spot removal brush drawn over the photo (PLAN.md 2.2, A-32, and 3.6).

While the tool is on, the pointer is a circle the size of the brush. Holding
the button down paints: what has been painted is covered in a light wash, and
letting go mends all of it at once, its source chosen by the program. A click
that does not move is a round dot.

What has been mended is not outlined, so the photo stays clear however many
strokes there are. A click on one still selects it: its outline appears with
the outline of its source, joined to it by a line; then dragging either
moves it, and the window deletes it on Delete. Painting can start anywhere, over an earlier stroke too: only a
press on the selected stroke or its source takes hold of it.

Like the crop box this is not a widget: it draws into the canvas's painter
and is handed the canvas's mouse events, in the same coordinates. What it
holds is only what is on screen - points in fractions of the frame as it is
shown, radii as fractions of that frame's long edge - and what a stroke
really is, on the unturned photo, is the controller's business.
"""

from __future__ import annotations

import math
from typing import Final

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPainterPathStroker, QPen

from ui import theme

Point = tuple[float, float]
#: A stroke as it is shown: its path, its source's path, and the radius as a
#: fraction of the frame's long edge.
ShownSpot = tuple[list[Point], list[Point], float]

_NOTHING: Final[int] = -1


class SpotOverlay(QObject):
    """The brush, what is being painted, and the selected stroke with its source."""

    #: A stroke painted, or a dot clicked: its path, in fractions of the
    #: frame as shown.
    spot_painted = Signal(list)
    #: A stroke dragged: its index, where its first point goes (or its
    #: source's), whether it is the source that moves, and whether the drag
    #: is still going on.
    spot_moved = Signal(int, float, float, bool, bool)
    #: The selected stroke, or -1 for none.
    selection_changed = Signal(int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._active = False
        self._spots: list[ShownSpot] = []
        self._selected = _NOTHING
        #: The brush's radius, as a fraction of the shown frame's long edge.
        self._brush = 0.0
        #: Where the pointer is over the canvas, or ``None`` when it is not.
        self._pointer: QPointF | None = None
        #: What a press took hold of: a stroke's index and whether its source.
        self._held: tuple[int, bool] | None = None
        #: How far the press was from the first point of what it took hold
        #: of, so the stroke does not jump to put that point under it.
        self._grab_offset = QPointF()
        self._moved = False
        #: The path being painted, in fractions of the frame, while the
        #: button is down.
        self._painting: list[Point] | None = None
        #: Where the paint started, and the stroke it started on, if any: a
        #: click that does not move selects that stroke instead of painting.
        self._press_at = QPointF()
        self._press_on: int | None = None

    # -- what it is showing ----------------------------------------------

    @property
    def active(self) -> bool:
        """Return whether the tool is on."""
        return self._active

    @property
    def selected(self) -> int:
        """Return the index of the selected stroke, or -1 for none."""
        return self._selected

    @property
    def dragging(self) -> bool:
        """Return whether a stroke is being dragged."""
        return self._held is not None

    @property
    def painting(self) -> bool:
        """Return whether the brush is down."""
        return self._painting is not None

    def set_active(self, active: bool) -> None:
        """Turn the tool on or off; off forgets the selection and any paint."""
        self._active = active
        if not active:
            self._held = None
            self._painting = None
            self.select(_NOTHING)

    def set_spots(self, spots: list[ShownSpot]) -> None:
        """Show these strokes; a selection past the end is dropped."""
        self._spots = [(list(path), list(source), radius) for path, source, radius in spots]
        if self._selected >= len(self._spots):
            self.select(_NOTHING)

    def set_brush(self, radius: float) -> None:
        """Set the brush's radius, as a fraction of the shown frame's long edge."""
        self._brush = max(0.0, radius)

    def select(self, index: int) -> None:
        """Select the stroke at *index*, or none for -1."""
        if not 0 <= index < len(self._spots):
            index = _NOTHING
        if index == self._selected:
            return
        self._selected = index
        self.selection_changed.emit(index)

    def set_pointer(self, point: QPointF | None) -> None:
        """Say where the pointer is, so the brush can be drawn under it."""
        self._pointer = point

    # -- drawing ---------------------------------------------------------

    def paint(self, painter: QPainter, photo: QRectF) -> None:
        """Draw the strokes, the paint and the brush over the photo at *photo*."""
        if not self._active or photo.isEmpty():
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setClipRect(photo)
        if 0 <= self._selected < len(self._spots):
            path, source, radius = self._spots[self._selected]
            size = _radius(photo, radius)
            outline = _outline([_at(photo, point) for point in path], size)
            source_outline = _outline([_at(photo, point) for point in source], size)
            self._draw_link(painter, _at(photo, source[0]), _at(photo, path[0]), size)
            self._draw_outline(painter, source_outline, theme.SPOT_LINE_WIDTH)
            self._draw_outline(painter, outline, theme.SPOT_SELECTED_WIDTH)
        brush = _radius(photo, self._brush)
        if self._painting is not None:
            wash = QColor(theme.WHITE)
            wash.setAlpha(theme.SPOT_PAINT_ALPHA)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(wash)
            painter.drawPath(_outline([_at(photo, point) for point in self._painting], brush))
            painter.setBrush(Qt.BrushStyle.NoBrush)
        if (
            self._held is None
            and self._pointer is not None
            and photo.contains(self._pointer)
            and not self.over_spot(self._pointer, photo)
        ):
            circle = QPainterPath()
            circle.addEllipse(self._pointer, brush, brush)
            self._draw_outline(painter, circle, theme.SPOT_LINE_WIDTH)
        painter.restore()

    @staticmethod
    def _draw_outline(painter: QPainter, outline: QPainterPath, width: int) -> None:
        """Draw a white line with a dark edge, so it shows on any photo."""
        halo = QColor(theme.SPOT_HALO)
        halo.setAlpha(theme.SPOT_HALO_ALPHA)
        line = QColor(theme.WHITE)
        painter.setPen(QPen(halo, width + 2 * theme.SPOT_HALO_WIDTH))
        painter.drawPath(outline)
        painter.setPen(QPen(line, width))
        painter.drawPath(outline)

    @staticmethod
    def _draw_link(painter: QPainter, source: QPointF, target: QPointF, radius: float) -> None:
        """Join the source's first point to the stroke's, from edge to edge."""
        dx = target.x() - source.x()
        dy = target.y() - source.y()
        length = math.hypot(dx, dy)
        if length <= 2 * radius:
            return
        ux, uy = dx / length, dy / length
        start = QPointF(source.x() + ux * radius, source.y() + uy * radius)
        end = QPointF(target.x() - ux * radius, target.y() - uy * radius)
        line = QPainterPath(start)
        line.lineTo(end)
        SpotOverlay._draw_outline(painter, line, theme.SPOT_LINE_WIDTH)

    # -- the mouse -------------------------------------------------------

    def press(self, point: QPointF, photo: QRectF) -> bool:
        """Take hold of the selected stroke, or start painting; return whether used."""
        if not self._active or photo.isEmpty():
            return False
        held = self._hit_selected(point, photo)
        if held is not None:
            index, source = held
            path, origin, _radius_ = self._spots[index]
            self._held = held
            self._grab_offset = point - _at(photo, (origin if source else path)[0])
            self._moved = False
            return True
        if not photo.contains(point):
            return False
        self._painting = [_fraction_of(photo, point)]
        self._press_at = QPointF(point)
        self._press_on = self._hit_any(point, photo)
        return True

    def move(self, point: QPointF, photo: QRectF) -> bool:
        """Paint, or drag what was taken hold of; return whether either went on."""
        self._pointer = point
        if photo.isEmpty():
            return False
        if self._painting is not None:
            last = _at(photo, self._painting[-1])
            step = max(theme.SPOT_PAINT_STEP, _radius(photo, self._brush) * 0.2)
            if math.hypot(point.x() - last.x(), point.y() - last.y()) >= step:
                self._painting.append(_fraction_of(photo, point))
            return True
        if self._held is None:
            return False
        index, source = self._held
        x, y = _fraction_of(photo, point - self._grab_offset, clamp=True)
        path, origin, radius = self._spots[index]
        anchor = origin[0] if source else path[0]
        dx, dy = x - anchor[0], y - anchor[1]
        moved_origin = [(px + dx, py + dy) for px, py in origin]
        if source:
            self._spots[index] = (path, moved_origin, radius)
        else:
            self._spots[index] = ([(px + dx, py + dy) for px, py in path], moved_origin, radius)
        self._moved = True
        self.spot_moved.emit(index, x, y, source, True)
        return True

    def release(self, point: QPointF, photo: QRectF) -> bool:
        """Mend what was painted, or let go of a stroke; return whether either."""
        self._pointer = point
        if self._painting is not None:
            painted = self._painting
            self._painting = None
            still = (
                math.hypot(point.x() - self._press_at.x(), point.y() - self._press_at.y())
                < theme.SPOT_CLICK_SLOP
            )
            if still and self._press_on is not None:
                # A click on a mended stroke picks it rather than painting a dot.
                self.select(self._press_on)
                return True
            self.select(_NOTHING)
            if not photo.isEmpty() and photo.contains(point) and not still:
                last = _fraction_of(photo, point)
                if last != painted[-1]:
                    painted.append(last)
            self.spot_painted.emit(painted)
            return True
        if self._held is None:
            return False
        index, source = self._held
        self._held = None
        if self._moved and not photo.isEmpty():
            path, origin, _radius_ = self._spots[index]
            where = (origin if source else path)[0]
            self.spot_moved.emit(index, where[0], where[1], source, False)
        return True

    def over_spot(self, point: QPointF, photo: QRectF) -> bool:
        """Return whether *point* would take hold of the selected stroke or its source."""
        return self._active and self._hit_selected(point, photo) is not None

    def _hit_selected(self, point: QPointF, photo: QRectF) -> tuple[int, bool] | None:
        """Return the selected stroke under *point*, its source first, or ``None``."""
        index = self._selected
        if photo.isEmpty() or not 0 <= index < len(self._spots):
            return None
        path, source, radius = self._spots[index]
        for points, is_source in ((source, True), (path, False)):
            if _near(point, [_at(photo, p) for p in points], _radius(photo, radius)):
                return index, is_source
        return None

    def _hit_any(self, point: QPointF, photo: QRectF) -> int | None:
        """Return the last mended stroke under *point*, or ``None``."""
        for index in reversed(range(len(self._spots))):
            path, _source, radius = self._spots[index]
            if _near(point, [_at(photo, p) for p in path], _radius(photo, radius)):
                return index
        return None


def _at(photo: QRectF, point: Point) -> QPointF:
    """Return where a point in fractions of the frame is on screen."""
    return QPointF(photo.left() + point[0] * photo.width(), photo.top() + point[1] * photo.height())


def _fraction_of(photo: QRectF, point: QPointF, clamp: bool = False) -> Point:
    """Return where a point on screen is, in fractions of the frame."""
    x = (point.x() - photo.left()) / photo.width()
    y = (point.y() - photo.top()) / photo.height()
    if clamp:
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
    return x, y


def _radius(photo: QRectF, radius: float) -> float:
    """Return a radius given as a fraction of the long edge in screen pixels."""
    return radius * max(photo.width(), photo.height())


def _outline(points: list[QPointF], radius: float) -> QPainterPath:
    """Return the shape a brush of *radius* covers along *points*."""
    shape = QPainterPath()
    if not points:
        return shape
    if len(points) == 1:
        shape.addEllipse(points[0], radius, radius)
        return shape
    line = QPainterPath(points[0])
    for point in points[1:]:
        line.lineTo(point)
    stroker = QPainterPathStroker()
    stroker.setWidth(2 * radius)
    stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
    stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return stroker.createStroke(line).simplified()


def _near(point: QPointF, path: list[QPointF], radius: float) -> bool:
    """Return whether *point* is within the brush, or a few pixels of a thin one."""
    reach = max(radius, theme.SPOT_GRAB_RADIUS)
    if len(path) == 1:
        return math.hypot(point.x() - path[0].x(), point.y() - path[0].y()) <= reach
    for a, b in zip(path[:-1], path[1:], strict=True):
        dx, dy = b.x() - a.x(), b.y() - a.y()
        length = dx * dx + dy * dy
        t = 0.0 if length == 0 else ((point.x() - a.x()) * dx + (point.y() - a.y()) * dy) / length
        t = min(max(t, 0.0), 1.0)
        if math.hypot(point.x() - (a.x() + t * dx), point.y() - (a.y() + t * dy)) <= reach:
            return True
    return False
