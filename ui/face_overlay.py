"""The face's points, drawn over the photo while they are placed by hand (PLAN.md 3.16, L-12).

Each face found is drawn as a small map of itself, so that every point can
be told apart at a glance:

* each feature has its own mark - a ring for the eyes, a triangle for the
  tip of the nose, squares for the corners of the mouth, a ring with a dot
  for the chin and diamonds for the cheeks and the corners of the jaw - and
  every point is named beside it, left and right as they are on screen;
* lines join the eyes and the corners of the mouth, and a dashed curve runs
  through the five points of the lower face - cheek, jaw, chin, jaw, cheek;
* the point under the pointer, or the one being dragged, is filled in, the
  reach of the slider it is for is outlined, and a line on the photo says
  where it belongs.

Apart from placing the points, the same overlay shows where one face slider
works while it is under the pointer or being dragged (PLAN.md 3.16): its
reach outlined with a dashed line, arrows for which way it moves the face,
and a line at the top saying what it does.

Like the spot brush this is not a widget: it draws into the canvas's painter
and is handed the canvas's mouse events. What it holds is what is on screen
- fractions of the frame as shown - and what a point is on the photo is the
controller's business. A press away from every point is not taken, so the
photo can still be panned.
"""

from __future__ import annotations

import math
from typing import Final

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen, QTransform

from app.i18n import tr
from core import liquify
from core.liquify import FaceGuide
from ui import theme
from ui.spot_overlay import _at, _fraction_of

Point = tuple[float, float]

_INDEX: Final[dict[str, int]] = {name: index for index, name in enumerate(liquify.HANDLES)}
#: The pairs of points that stand either side of the face, which are named
#: left and right by where they are on screen.
_PAIRS: Final[tuple[tuple[str, str, str], ...]] = (
    ("left_eye", "right_eye", "eye"),
    ("left_mouth", "right_mouth", "mouth"),
    ("left_cheek", "right_cheek", "cheek"),
    ("left_jaw", "right_jaw", "jaw"),
)
#: The sliders that widen and narrow across the face; the chin moves along
#: it instead.
_ACROSS: Final[frozenset[str]] = frozenset({"eyes", "nose", "mouth", "face_width"})
#: Where along an ellipse's axis across the face the arrows of a slider that
#: widens stand, as a share of the half width.
_ARROW_REACH: Final[float] = 0.6
#: The mark each feature is drawn with.
_MARKS: Final[dict[str, str]] = {
    "eye": "ring",
    "nose": "triangle",
    "mouth": "square",
    "chin": "target",
    "cheek": "diamond",
    "jaw": "diamond",
}


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
        #: The face slider being shown: its name, its value, its ellipses on
        #: screen and the line saying what it does; or ``None``.
        self._spotlight: tuple[str, float, list[tuple[Point, Point, Point]], str] | None = None

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

    @property
    def spotlit(self) -> str:
        """Return the face slider whose reach is shown, or an empty string."""
        return "" if self._spotlight is None else self._spotlight[0]

    def set_spotlight(
        self,
        feature: str,
        value: float,
        reaches: list[tuple[Point, Point, Point]],
        text: str,
    ) -> None:
        """Show where the face slider *feature* works and which way *value* moves it."""
        self._spotlight = (feature, value, list(reaches), text)

    def clear_spotlight(self) -> None:
        """Stop showing a face slider's reach."""
        self._spotlight = None

    def over_handle(self, point: QPointF, photo: QRectF) -> bool:
        """Return whether *point* is on one of the face's points."""
        return self._hit(point, photo) is not None

    def names(self, face: int) -> dict[str, str]:
        """Return the text key naming each point of *face*, left and right as on screen."""
        if not 0 <= face < len(self._guides):
            return {}
        handles = self._guides[face].handles
        named = {"nose": "liquify.handle.nose", "chin": "liquify.handle.chin"}
        for first, second, feature in _PAIRS:
            first_left = handles[_INDEX[first]][0] <= handles[_INDEX[second]][0]
            named[first] = f"liquify.handle.{feature}_{'left' if first_left else 'right'}"
            named[second] = f"liquify.handle.{feature}_{'right' if first_left else 'left'}"
        return named

    # -- drawing ---------------------------------------------------------

    def paint(self, painter: QPainter, photo: QRectF) -> None:
        """Draw the face slider in hand's reach, and every face's points while they are placed."""
        if photo.isEmpty():
            return
        if self._spotlight is not None and not self._open:
            self._draw_spotlight(painter, photo)
        if not self._open or not self._guides:
            return
        painter.save()
        painter.setClipRect(photo)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = QFont(painter.font())
        font.setPixelSize(theme.FONT_SIZE_LABEL)
        painter.setFont(font)
        marked = self._held if self._held is not None else self._over
        for face, guide in enumerate(self._guides):
            handles = [_at(photo, point) for point in guide.handles]
            if self._held is not None and self._held[0] == face and self._held_at is not None:
                handles[self._held[1]] = _at(photo, self._held_at)
            self._draw_outline(painter, handles)
            if marked is not None and marked[0] == face:
                feature = liquify.HANDLE_FEATURES[liquify.HANDLES[marked[1]]]
                for centre, across, down in guide.reaches.get(feature, ()):
                    self._draw_reach(painter, photo, centre, across, down)
            names = self.names(face)
            middle = _middle(handles)
            for index, centre in enumerate(handles):
                handle = liquify.HANDLES[index]
                self._draw_mark(painter, centre, _MARKS[_feature(handle)], marked == (face, index))
                self._draw_name(painter, tr(names[handle]), centre, middle, handle)
        if marked is not None and marked[0] < len(self._guides):
            face, index = marked
            handle = liquify.HANDLES[index]
            hint = tr(
                "liquify.handle.hint",
                name=tr(self.names(face)[handle]),
                where=tr(f"liquify.handle.{_feature(handle)}.where"),
            )
            self._draw_hint(painter, photo, hint)
        painter.restore()

    def _stroke(self, painter: QPainter, path: QPainterPath, style: Qt.PenStyle = Qt.PenStyle.SolidLine) -> None:
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

    def _draw_outline(self, painter: QPainter, handles: list[QPointF]) -> None:
        """Join the eyes and the mouth, and run the lower face's outline through the chin."""
        lines = QPainterPath()
        for first, second in (("left_eye", "right_eye"), ("left_mouth", "right_mouth")):
            lines.moveTo(handles[_INDEX[first]])
            lines.lineTo(handles[_INDEX[second]])
        self._stroke(painter, lines)
        outline = [
            handles[_INDEX[name]]
            for name in ("left_cheek", "left_jaw", "chin", "right_jaw", "right_cheek")
        ]
        self._stroke(painter, _through(outline), Qt.PenStyle.DashLine)

    def _draw_spotlight(self, painter: QPainter, photo: QRectF) -> None:
        """Outline where one face slider works, with arrows for which way, and say what it does."""
        assert self._spotlight is not None
        feature, value, reaches, text = self._spotlight
        painter.save()
        painter.setClipRect(photo)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = QFont(painter.font())
        font.setPixelSize(theme.FONT_SIZE_LABEL)
        painter.setFont(font)
        for index, (centre, across, down) in enumerate(reaches):
            self._draw_reach(painter, photo, centre, across, down, Qt.PenStyle.DashLine)
            # The face width's arrows stand on the cheeks' ellipse only: about
            # the jaw the face moves toward its middle, not out from the corner.
            if value and (feature != "face_width" or index == 0):
                self._draw_arrows(painter, photo, feature, value, centre, across, down)
        self._draw_hint(painter, photo, text)
        painter.restore()

    def _draw_arrows(
        self,
        painter: QPainter,
        photo: QRectF,
        feature: str,
        value: float,
        centre: Point,
        across: Point,
        down: Point,
    ) -> None:
        """Draw which way a face slider moves the face: out or in, down or up."""
        middle = _at(photo, centre)
        side = _at(photo, across) - middle
        below = _at(photo, down) - middle
        length = float(theme.FACE_ARROW_LENGTH)
        arrows = QPainterPath()
        if feature in _ACROSS:
            span = math.hypot(side.x(), side.y())
            if span <= 0.0:
                return
            unit = side / span
            for sign in (-1.0, 1.0):
                foot = middle + side * (sign * _ARROW_REACH)
                pointing = unit * (sign if value > 0 else -sign)
                _add_arrow(arrows, foot - pointing * (length / 2), pointing, length)
        else:
            span = math.hypot(below.x(), below.y())
            if span <= 0.0:
                return
            pointing = below / span * (1.0 if value > 0 else -1.0)
            _add_arrow(arrows, middle - pointing * (length / 2), pointing, length)
        self._stroke(painter, arrows)

    def _draw_reach(
        self,
        painter: QPainter,
        photo: QRectF,
        centre: Point,
        across: Point,
        down: Point,
        style: Qt.PenStyle = Qt.PenStyle.SolidLine,
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
        self._stroke(painter, axes.map(circle), style)

    def _draw_mark(self, painter: QPainter, centre: QPointF, mark: str, filled: bool) -> None:
        """Draw one point with its feature's mark, filled in when it is the one in hand."""
        half = theme.FACE_HANDLE_SIZE / 2
        shape = QPainterPath()
        if mark == "triangle":
            shape.moveTo(centre.x(), centre.y() - half)
            shape.lineTo(centre.x() + half, centre.y() + half)
            shape.lineTo(centre.x() - half, centre.y() + half)
            shape.closeSubpath()
        elif mark == "square":
            shape.addRect(QRectF(centre.x() - half, centre.y() - half, 2 * half, 2 * half))
        elif mark == "diamond":
            shape.moveTo(centre.x(), centre.y() - half)
            shape.lineTo(centre.x() + half, centre.y())
            shape.lineTo(centre.x(), centre.y() + half)
            shape.lineTo(centre.x() - half, centre.y())
            shape.closeSubpath()
        else:
            shape.addEllipse(centre, half, half)
        halo = QColor(theme.SPOT_HALO)
        halo.setAlpha(theme.SPOT_HALO_ALPHA)
        painter.setPen(QPen(halo, theme.SPOT_LINE_WIDTH + 2 * theme.SPOT_HALO_WIDTH))
        painter.setBrush(QColor(theme.ACCENT) if filled else Qt.BrushStyle.NoBrush)
        painter.drawPath(shape)
        painter.setPen(QPen(QColor(theme.WHITE), theme.SPOT_LINE_WIDTH))
        painter.drawPath(shape)
        if mark == "target" and not filled:
            dot = theme.FACE_HANDLE_DOT / 2
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(theme.WHITE))
            painter.drawEllipse(centre, dot, dot)

    def _draw_name(
        self, painter: QPainter, text: str, centre: QPointF, middle: QPointF, handle: str
    ) -> None:
        """Write a point's name beside it, on the side away from the middle of the face."""
        metrics = QFontMetricsF(painter.font())
        width = metrics.horizontalAdvance(text)
        height = metrics.height()
        gap = theme.FACE_HANDLE_SIZE / 2 + theme.SPACE_XS
        if handle in ("left_eye", "right_eye"):
            # Above the eyes, where the brow is, rather than across the face.
            top_left = QPointF(centre.x() - width / 2, centre.y() - gap - height)
        elif handle == "chin":
            top_left = QPointF(centre.x() - width / 2, centre.y() + gap)
        elif centre.x() < middle.x():
            top_left = QPointF(centre.x() - gap - width, centre.y() - height / 2)
        else:
            top_left = QPointF(centre.x() + gap, centre.y() - height / 2)
        path = QPainterPath()
        path.addText(top_left + QPointF(0.0, metrics.ascent()), painter.font(), text)
        halo = QColor(theme.SPOT_HALO)
        halo.setAlpha(theme.SPOT_HALO_ALPHA)
        painter.setPen(QPen(halo, 2 * theme.FACE_LABEL_HALO, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.WHITE))
        painter.drawPath(path)

    def _draw_hint(self, painter: QPainter, photo: QRectF, text: str) -> None:
        """Say where the point in hand belongs, in a small box at the top of the view.

        At the top rather than beside the point, where it would cover the
        names of the points around it.
        """
        metrics = QFontMetricsF(painter.font())
        width = metrics.horizontalAdvance(text) + 2 * theme.SPACE_SM
        height = metrics.height() + 2 * theme.SPACE_XS
        visible = photo.intersected(QRectF(painter.window()))
        if visible.isEmpty():
            return
        box = QRectF(
            visible.center().x() - width / 2, visible.top() + theme.SPACE_MD, width, height
        )
        painter.setPen(QPen(QColor(theme.BORDER), theme.BORDER_WIDTH))
        painter.setBrush(QColor(theme.BG_PANEL))
        painter.drawRoundedRect(box, theme.RADIUS_CONTROL, theme.RADIUS_CONTROL)
        painter.setPen(QColor(theme.TEXT_PRIMARY))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

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


def _feature(handle: str) -> str:
    """Return the part of the face *handle* belongs to: eye, nose, mouth, chin or cheek."""
    return handle.split("_")[-1] if handle not in ("nose", "chin") else handle


def _middle(handles: list[QPointF]) -> QPointF:
    """Return the middle of a face's points on screen."""
    total = QPointF(0.0, 0.0)
    for point in handles:
        total += point
    return total / max(len(handles), 1)


def _add_arrow(path: QPainterPath, start: QPointF, pointing: QPointF, length: float) -> None:
    """Add to *path* an arrow *length* long from *start*, the way *pointing* (of length one) goes."""
    tip = start + pointing * length
    head = length * 0.4
    back = tip - pointing * head
    across = QPointF(-pointing.y(), pointing.x()) * (head * 0.6)
    path.moveTo(start)
    path.lineTo(tip)
    path.moveTo(back + across)
    path.lineTo(tip)
    path.lineTo(back - across)


def _through(points: list[QPointF]) -> QPainterPath:
    """Return a smooth curve through every one of *points*, in order.

    Each stretch is a cubic whose ends lean the way the points on either
    side go, so the curve bends evenly and passes through every point.
    """
    path = QPainterPath()
    if not points:
        return path
    path.moveTo(points[0])
    for index in range(len(points) - 1):
        before = points[max(index - 1, 0)]
        start = points[index]
        end = points[index + 1]
        after = points[min(index + 2, len(points) - 1)]
        path.cubicTo(start + (end - before) / 6.0, end - (after - start) / 6.0, end)
    return path

