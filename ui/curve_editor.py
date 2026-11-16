"""The luminance curve's editor (PLAN.md 2.2 A-34 and 3.1).

A square with the curve drawn across it: how bright a tone was runs left to
right, how bright it is to be runs bottom to top. Behind the curve lies the
brightness histogram of the photo as it is with the curve straight, so a
point can be put under the tones it is meant to move; it does not shift while
the curve is being drawn, since it is what the curve acts on.

A press on empty ground adds a point and takes hold of it, a drag moves a
point, and a double click or Delete takes one away. The two ends stay on
their edges and only move up and down. Under the square one line reads the
chosen point in levels of 255, or says how to begin.

The editor only holds the points and says when they move; shaping the photo
is the controller's work.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from app.i18n import tr
from core import tone_curve
from ui import theme
from ui.histogram_widget import BINS, SCALE_HEADROOM, SCALE_PERCENTILE, luma_values
from ui.widgets.text import use_tabular_figures

#: The levels a point is read out in.
_LEVELS: Final[int] = 255

#: No point.
_NONE: Final[int] = -1

_DELETE_KEYS: Final[tuple[Qt.Key, ...]] = (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)


class CurvePlot(QWidget):
    """The square the curve is drawn and shaped in."""

    #: The curve as it now is, and whether a point is still under the hand.
    curve_set = Signal(object, bool)
    #: The point the readout should describe has changed.
    readout_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._curve: tone_curve.Curve = tone_curve.STRAIGHT
        self._selected = _NONE
        self._hovered = _NONE
        self._dragging = False
        #: Whether the press now held has changed the curve.
        self._moved = False
        self._backdrop: NDArray[np.int64] | None = None
        self.setFixedSize(theme.CURVE_EDITOR_SIZE, theme.CURVE_EDITOR_SIZE)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setCursor(Qt.CursorShape.CrossCursor)

    # -- what it holds -----------------------------------------------------

    @property
    def curve(self) -> tone_curve.Curve:
        """Return the points as they now are."""
        return self._curve

    @property
    def selected(self) -> int:
        """Return which point is chosen, or -1."""
        return self._selected

    @property
    def dragging(self) -> bool:
        """Return whether a point is under the hand."""
        return self._dragging

    def set_curve(self, curve: tone_curve.Curve) -> None:
        """Show *curve* without announcing it, as undo and presets do."""
        if curve == self._curve:
            return
        self._curve = curve
        self._selected = _NONE
        self._hovered = _NONE
        self._dragging = False
        self.readout_changed.emit()
        self.update()

    def set_backdrop(self, image: NDArray[np.uint8] | None) -> None:
        """Lay the brightness histogram of *image* behind the curve, or none."""
        if image is None or image.size == 0:
            self._backdrop = None
        else:
            self._backdrop = np.bincount(
                luma_values(image).reshape(-1), minlength=BINS
            ).astype(np.int64)
        self.update()

    @property
    def has_backdrop(self) -> bool:
        """Return whether a histogram lies behind the curve."""
        return self._backdrop is not None

    def readout(self) -> tuple[int, int] | None:
        """Return the chosen point in levels, in and out, or ``None``.

        The point under the hand or chosen comes first; failing one, the
        point the pointer rests on.
        """
        index = self._selected if self._selected != _NONE else self._hovered
        if not 0 <= index < len(self._curve):
            return None
        x, y = self._curve[index]
        return round(x * _LEVELS), round(y * _LEVELS)

    # -- where things are --------------------------------------------------

    def plot_rect(self) -> QRectF:
        """Return the square the curve runs across, clear of the edge.

        Inset so that a point on an end or a corner is drawn whole.
        """
        inset = float(theme.CURVE_EDITOR_INSET)
        return QRectF(self.rect()).adjusted(inset, inset, -inset, -inset)

    def position_of(self, x: float, y: float) -> QPointF:
        """Return where the curve's (*x*, *y*) lies in the widget."""
        area = self.plot_rect()
        return QPointF(area.left() + x * area.width(), area.bottom() - y * area.height())

    def _value_at(self, position: QPointF) -> tuple[float, float]:
        """Return the curve's coordinates of a place in the widget."""
        area = self.plot_rect()
        return (
            (position.x() - area.left()) / area.width(),
            (area.bottom() - position.y()) / area.height(),
        )

    def _point_at(self, position: QPointF) -> int:
        """Return the nearest point within reach of *position*, or -1."""
        best, nearest = _NONE, float(theme.CURVE_GRAB_RADIUS)
        for index, (x, y) in enumerate(self._curve):
            place = self.position_of(x, y)
            distance = float(np.hypot(place.x() - position.x(), place.y() - position.y()))
            if distance <= nearest:
                best, nearest = index, distance
        return best

    # -- the mouse and the keyboard ----------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt naming
        """Take hold of the point under the pointer, or add one there."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        position = event.position()
        index = self._point_at(position)
        self._moved = False
        if index == _NONE:
            x, y = self._value_at(position)
            if not 0.0 <= x <= 1.0:
                return
            curve, index = tone_curve.with_point(self._curve, x, y)
            if index == _NONE:
                return
            self._curve = curve
            self._moved = True
        self._selected = index
        self._hovered = index
        self._dragging = True
        self._show_cursor()
        self.readout_changed.emit()
        self.update()
        if self._moved:
            self.curve_set.emit(self._curve, True)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt naming
        """Move the point held, or light the one the pointer rests on."""
        position = event.position()
        if not self._dragging:
            hovered = self._point_at(position)
            if hovered != self._hovered:
                self._hovered = hovered
                self._show_cursor()
                self.readout_changed.emit()
                self.update()
            return
        curve = tone_curve.moved(self._curve, self._selected, *self._value_at(position))
        if curve == self._curve:
            return
        self._curve = curve
        self._moved = True
        self.readout_changed.emit()
        self.update()
        self.curve_set.emit(curve, True)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt naming
        """Let the point go; a curve that moved is now finished."""
        if event.button() != Qt.MouseButton.LeftButton or not self._dragging:
            return
        self._dragging = False
        self._show_cursor()
        if self._moved:
            self._moved = False
            self.curve_set.emit(self._curve, False)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt naming
        """Take away the point double clicked."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._remove(self._point_at(event.position()))

    def leaveEvent(self, event: QEvent) -> None:  # noqa: N802 - Qt naming
        """Stop lighting a point once the pointer has gone."""
        if self._hovered != _NONE and not self._dragging:
            self._hovered = _NONE
            self.readout_changed.emit()
            self.update()
        super().leaveEvent(event)

    def event(self, event: QEvent) -> bool:
        """Keep Delete for the chosen point rather than a window shortcut."""
        if (
            event.type() == QEvent.Type.ShortcutOverride
            and isinstance(event, QKeyEvent)
            and event.key() in _DELETE_KEYS
            and self._can_remove(self._selected)
        ):
            event.accept()
            return True
        return super().event(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt naming
        """Take away the chosen point on Delete."""
        if event.key() in _DELETE_KEYS and self._can_remove(self._selected):
            self._remove(self._selected)
            return
        super().keyPressEvent(event)

    def _can_remove(self, index: int) -> bool:
        """Return whether *index* is a point that may be taken away."""
        return 0 < index < len(self._curve) - 1

    def _remove(self, index: int) -> None:
        """Take one point away and say the curve is finished."""
        if not self._can_remove(index):
            return
        self._curve = tone_curve.without_point(self._curve, index)
        self._selected = _NONE
        self._hovered = _NONE
        self._dragging = False
        self._moved = False
        self._show_cursor()
        self.readout_changed.emit()
        self.update()
        self.curve_set.emit(self._curve, False)

    def _show_cursor(self) -> None:
        """Show the moving hand over a point and the cross elsewhere."""
        over = self._dragging or self._hovered != _NONE
        self.setCursor(
            Qt.CursorShape.SizeAllCursor if over else Qt.CursorShape.CrossCursor
        )

    # -- drawing -----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Draw the recess, the histogram, the grid, the curve and its points."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        area = self.plot_rect()

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.BG_INSET))
        painter.drawRoundedRect(self.rect(), theme.RADIUS_CARD, theme.RADIUS_CARD)

        if self._backdrop is not None:
            self._draw_backdrop(painter, area)
        self._draw_grid(painter, area)
        self._draw_curve(painter, area)
        self._draw_points(painter)

    def _draw_backdrop(self, painter: QPainter, area: QRectF) -> None:
        """Fill the brightness histogram, faint enough to read the curve over."""
        counts = self._backdrop
        assert counts is not None  # the caller has looked
        scale = max(1.0, float(np.percentile(counts, SCALE_PERCENTILE)) * SCALE_HEADROOM)
        path = QPainterPath()
        path.moveTo(area.left(), area.bottom())
        for index in range(BINS):
            x = area.left() + area.width() * index / (BINS - 1)
            height = min(1.0, counts[index] / scale) * area.height()
            path.lineTo(x, area.bottom() - height)
        path.lineTo(area.right(), area.bottom())
        path.closeSubpath()
        fill = QColor(theme.HISTOGRAM_LUMA)
        fill.setAlpha(theme.CURVE_BACKDROP_ALPHA)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        painter.drawPath(path)

    def _draw_grid(self, painter: QPainter, area: QRectF) -> None:
        """Rule the square in quarters and draw the line that changes nothing."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(QPen(QColor(theme.BORDER), theme.BORDER_WIDTH))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for step in range(theme.CURVE_GRID_DIVISIONS + 1):
            along = step / theme.CURVE_GRID_DIVISIONS
            x = round(area.left() + along * area.width())
            y = round(area.top() + along * area.height())
            painter.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
            painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        diagonal = QPen(QColor(theme.TRACK), theme.BORDER_WIDTH, Qt.PenStyle.DashLine)
        painter.setPen(diagonal)
        painter.drawLine(area.bottomLeft(), area.topRight())

    def _draw_curve(self, painter: QPainter, area: QRectF) -> None:
        """Draw the curve itself, one sample to the pixel."""
        samples = max(2, int(area.width()) + 1)
        across = np.linspace(0.0, 1.0, samples)
        heights = tone_curve.evaluate(self._curve, across)
        path = QPainterPath()
        path.moveTo(self.position_of(float(across[0]), float(heights[0])))
        for x, y in zip(across[1:], heights[1:]):
            path.lineTo(self.position_of(float(x), float(y)))
        colour = theme.TEXT_PRIMARY if self.isEnabled() else theme.TEXT_TERTIARY
        painter.setPen(QPen(QColor(colour), theme.CURVE_LINE_WIDTH))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def _draw_points(self, painter: QPainter) -> None:
        """Draw every point: hollow at rest, lit under the pointer, solid when chosen."""
        ring = theme.TEXT_PRIMARY if self.isEnabled() else theme.TEXT_TERTIARY
        painter.setPen(QPen(QColor(ring), theme.CURVE_LINE_WIDTH))
        for index, (x, y) in enumerate(self._curve):
            if index == self._selected:
                fill = theme.ACCENT
            elif index == self._hovered:
                fill = theme.TRACK_PRESSED
            else:
                fill = theme.BG_INSET
            painter.setBrush(QColor(fill))
            painter.drawEllipse(
                self.position_of(x, y), theme.CURVE_POINT_RADIUS, theme.CURVE_POINT_RADIUS
            )


class CurveEditor(QWidget):
    """The curve's square with the line under it that reads the chosen point."""

    #: The curve as it now is, and whether a point is still under the hand.
    curve_set = Signal(object, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.plot = CurvePlot(self)
        self.readout = QLabel(self)
        self.readout.setProperty("role", "caption")
        self.readout.setFixedHeight(theme.CURVE_READOUT_HEIGHT)
        self.readout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        use_tabular_figures(self.readout)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_XS)
        layout.addWidget(self.plot, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.readout)

        self.plot.curve_set.connect(self.curve_set)
        self.plot.readout_changed.connect(self._show_readout)
        self._show_readout()

    @property
    def curve(self) -> tone_curve.Curve:
        """Return the points as they now are."""
        return self.plot.curve

    def set_curve(self, curve: tone_curve.Curve) -> None:
        """Show *curve* without announcing it."""
        self.plot.set_curve(curve)

    def set_backdrop(self, image: NDArray[np.uint8] | None) -> None:
        """Lay the brightness histogram of *image* behind the curve, or none."""
        self.plot.set_backdrop(image)

    def retranslate(self) -> None:
        """Read the line under the square again after a language change."""
        self._show_readout()

    def _show_readout(self) -> None:
        """Read the chosen point in levels, or say how to begin."""
        levels = self.plot.readout()
        if levels is None:
            self.readout.setText(tr("curve.hint"))
            return
        self.readout.setText(tr("curve.readout", input=levels[0], output=levels[1]))
