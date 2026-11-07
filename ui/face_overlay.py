"""The face's points, drawn over the photo while they are placed by hand (PLAN.md 3.16, L-12).

Each face found is shown as eight small circles - the two eyes, the tip of
the nose, the corners of the mouth, the chin and the two cheeks - joined by
thin lines across the eyes, the mouth and the cheeks. The point under the
pointer, or the one being dragged, is filled in, and the reach of the slider
it is for is outlined, so it is plain what moving it changes.

Like the spot brush this is not a widget: it draws into the canvas's painter
and is handed the canvas's mouse events. What it holds is what is on screen
- fractions of the frame as shown - and what a point is on the photo is the
controller's business. A press away from every point is not taken, so the
photo can still be panned.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QTransform

from core import liquify
from core.liquify import FaceGuide
from ui import theme
from ui.spot_overlay import _at, _fraction_of

Point = tuple[float, float]

#: The pairs of points joined by a line, as indices into the handles.
_LINES: tuple[tuple[int, int], ...] = (
    (liquify.HANDLES.index("left_eye"), liquify.HANDLES.index("right_eye")),
    (liquify.HANDLES.index("left_mouth"), liquify.HANDLES.index("right_mouth")),
    (liquify.HANDLES.index("left_cheek"), liquify.HANDLES.index("right_cheek")),
)


class FaceOverlay(QObject):
    """The points of every face and the one being dragged."""

    #: A point moved: which face, which point (a name of
    #: :data:`core.liquify.HANDLES`), where to, in fractions of the frame as
    #: shown, and whether the drag is still going on.
    handle_moved = Signal(int, str, float, float, bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        #: Whether the points are being placed, which is when anything is drawn.
        self._open = False
        self._guides: list[FaceGuide] = []
        #: The point under the pointer, as (face, handle index), or ``None``.
        self._over: tuple[int, int] | None = None
        #: The point being dragged and where it is now, or ``None``.
        self._held: tuple[int, int] | None = None
        self._held_at: Point | None = None

    # -- what it is showing ----------------------------------------------

    @property
    def active(self) -> bool:
        """Return whether the points are on the photo."""
        return self._open

    @property
    def dragging(self) -> bool:
        """Return whether a point is being dragged."""
        return self._held is not None

    def set_open(self, open_: bool) -> None:
        """Say whether the points are being placed; closed draws nothing."""
        self._open = open_
        if not open_:
            self._held = None
            self._held_at = None
            self._over = None

    def set_guides(self, guides: list[FaceGuide]) -> None:
        """Take every face's points and reach, in fractions of the frame as shown."""
        self._guides = list(guides)
        if self._held is not None and self._held[0] >= len(self._guides):
            self._held = None
            self._held_at = None
        if self._over is not None and self._over[0] >= len(self._guides):
            self._over = None

    def over_handle(self, point: QPointF, photo: QRectF) -> bool:
        """Return whether *point* is on one of the face's points."""
        return self._hit(point, photo) is not None

    # -- drawing ---------------------------------------------------------

    def paint(self, painter: QPainter, photo: QRectF) -> None:
        """Draw every face's points over the photo at *photo*."""
        if not self._open or photo.isEmpty() or not self._guides:
            return
        painter.save()
        painter.setClipRect(photo)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        marked = self._held if self._held is not None else self._over
        for face, guide in enumerate(self._guides):
            handles = [_at(photo, point) for point in guide.handles]
            if self._held is not None and self._held[0] == face and self._held_at is not None:
                handles[self._held[1]] = _at(photo, self._held_at)
            lines = QPainterPath()
            for start, end in _LINES:
                lines.moveTo(handles[start])
                lines.lineTo(handles[end])
            self._stroke(painter, lines)
            if marked is not None and marked[0] == face:
                feature = liquify.HANDLE_FEATURES[liquify.HANDLES[marked[1]]]
                for centre, across, down in guide.reaches.get(feature, ()):
                    self._draw_reach(painter, photo, centre, across, down)
            for index, centre in enumerate(handles):
                filled = marked == (face, index)
                self._draw_handle(painter, centre, filled)
        painter.restore()

    def _stroke(self, painter: QPainter, path: QPainterPath) -> None:
        """Draw *path* as a white line with a dark edge either side, like the spot brush."""
        halo = QColor(theme.SPOT_HALO)
        halo.setAlpha(theme.SPOT_HALO_ALPHA)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(halo, theme.SPOT_LINE_WIDTH + 2 * theme.SPOT_HALO_WIDTH))
        painter.drawPath(path)
        painter.setPen(QPen(QColor(theme.WHITE), theme.SPOT_LINE_WIDTH))
        painter.drawPath(path)

    def _draw_reach(
        self, painter: QPainter, photo: QRectF, centre: Point, across: Point, down: Point
    ) -> None:
        """Outline one ellipse a slider works within, turned and cut as the frame is."""
        middle = _at(photo, centre)
        first = _at(photo, across) - middle
        second = _at(photo, down) - middle
        # The unit circle taken through the ellipse's two axes: whatever the
        # turn, mirror and rotation, it lands on the ellipse as shown.
        axes = QTransform(first.x(), first.y(), second.x(), second.y(), middle.x(), middle.y())
        circle = QPainterPath()
        circle.addEllipse(QPointF(0.0, 0.0), 1.0, 1.0)
        self._stroke(painter, axes.map(circle))

    def _draw_handle(self, painter: QPainter, centre: QPointF, filled: bool) -> None:
        """Draw one point: a ring, filled in when it is the one in hand."""
        radius = theme.FACE_HANDLE_SIZE / 2
        halo = QColor(theme.SPOT_HALO)
        halo.setAlpha(theme.SPOT_HALO_ALPHA)
        painter.setPen(QPen(halo, theme.SPOT_LINE_WIDTH + 2 * theme.SPOT_HALO_WIDTH))
        painter.setBrush(QColor(theme.ACCENT) if filled else Qt.BrushStyle.NoBrush)
        painter.drawEllipse(centre, radius, radius)
        painter.setPen(QPen(QColor(theme.WHITE), theme.SPOT_LINE_WIDTH))
        painter.drawEllipse(centre, radius, radius)

    # -- the mouse -------------------------------------------------------

    def _hit(self, point: QPointF, photo: QRectF) -> tuple[int, int] | None:
        """Return the nearest point within reach of *point*, as (face, handle index)."""
        if not self._open or photo.isEmpty():
            return None
        best: tuple[int, int] | None = None
        nearest = float(theme.FACE_HANDLE_GRAB)
        for face, guide in enumerate(self._guides):
            for index, handle in enumerate(guide.handles):
                at = _at(photo, handle)
                distance = math.hypot(at.x() - point.x(), at.y() - point.y())
                if distance <= nearest:
                    nearest = distance
                    best = (face, index)
        return best

    def press(self, point: QPointF, photo: QRectF) -> bool:
        """Take hold of the point under the pointer; return whether there was one."""
        hit = self._hit(point, photo)
        if hit is None:
            return False
        self._held = hit
        self._held_at = _fraction_of(photo, point, clamp=True)
        return True

    def move(self, point: QPointF, photo: QRectF) -> bool:
        """Drag the point in hand, or mark the one under the pointer; return whether to redraw."""
        if self._held is None:
            over = self._hit(point, photo)
            changed = over != self._over
            self._over = over
            return changed
        self._held_at = _fraction_of(photo, point, clamp=True)
        face, index = self._held
        x, y = self._held_at
        self.handle_moved.emit(face, liquify.HANDLES[index], x, y, True)
        return True

    def release(self, point: QPointF, photo: QRectF) -> bool:
        """Let go of the point in hand, handing over where it ended; return whether one was held."""
        if self._held is None:
            return False
        face, index = self._held
        x, y = _fraction_of(photo, point, clamp=True)
        self._held = None
        self._held_at = None
        self._over = self._hit(point, photo)
        self.handle_moved.emit(face, liquify.HANDLES[index], x, y, False)
        return True

    def leave(self) -> None:
        """Forget the point under the pointer when it leaves the photo."""
        if self._held is None:
            self._over = None
