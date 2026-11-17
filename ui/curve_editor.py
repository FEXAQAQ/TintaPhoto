"""The curves' editor (PLAN.md 2.2 A-34, A-35 and 3.1).

A row of choices over a square: the luminance curve, the curve shared by red,
green and blue, or one of those three alone. The chosen curve is drawn across
the square: how bright a tone was runs left to right, how bright it is to be
runs bottom to top. Behind the curve lies the histogram of the photo as that
curve finds it - brightness for the luminance curve, the three channels for
the shared one, one channel for its own - so a point can be put under the
tones it is meant to move; it does not shift while the curve is being drawn,
since it is what the curve acts on. A curve that has been bent carries a dot
after its name.

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
from core.edit_state import CURVE_NAMES, LUMA_CURVE, RGB_CURVE, EditState
from ui import theme
from ui.histogram_widget import BINS, SCALE_HEADROOM, SCALE_PERCENTILE, luma_values
from ui.widgets.segmented import Segmented
from ui.widgets.text import use_tabular_figures

#: The levels a point is read out in.
_LEVELS: Final[int] = 255

#: No point.
_NONE: Final[int] = -1

_DELETE_KEYS: Final[tuple[Qt.Key, ...]] = (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)

#: The curves on offer, in the order of the row, with the key of each name.
CURVE_CHOICES: Final[tuple[tuple[str, str], ...]] = (
    (LUMA_CURVE, "curve.choice.luma"),
    (RGB_CURVE, "curve.choice.rgb"),
    ("red_curve", "curve.choice.red"),
    ("green_curve", "curve.choice.green"),
    ("blue_curve", "curve.choice.blue"),
)

#: Which rows of the backdrop's counts each curve is drawn over - brightness
#: first, then red, green and blue - and the colour of each row.
_BACKDROP_ROWS: Final[dict[str, tuple[int, ...]]] = {
    LUMA_CURVE: (0,),
    RGB_CURVE: (1, 2, 3),
    "red_curve": (1,),
    "green_curve": (2,),
    "blue_curve": (3,),
}
_ROW_COLOURS: Final[tuple[str, str, str, str]] = (
    theme.HISTOGRAM_LUMA,
    theme.HISTOGRAM_RED,
    theme.HISTOGRAM_GREEN,
    theme.HISTOGRAM_BLUE,
)
#: The colour a curve is drawn in: a channel's own in its colour, the two
#: that act on all three in the colour of text.
_CURVE_COLOURS: Final[dict[str, str]] = {
    LUMA_CURVE: theme.TEXT_PRIMARY,
    RGB_CURVE: theme.TEXT_PRIMARY,
    "red_curve": theme.HISTOGRAM_RED,
    "green_curve": theme.HISTOGRAM_GREEN,
    "blue_curve": theme.HISTOGRAM_BLUE,
}


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
        #: Which curve is drawn, by its name in EditState.
        self._channel = LUMA_CURVE
        #: The backdrop's counts: brightness, then red, green and blue.
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

    @property
    def channel(self) -> str:
        """Return which curve is drawn, by its name in EditState."""
        return self._channel

    def set_channel(self, name: str) -> None:
        """Draw the curve called *name* in its colour, over its histogram."""
        if name == self._channel:
            return
        self._channel = name
        self.update()

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
        """Lay the histograms of *image* behind the curve, or none.

        Brightness and all three channels are counted at once, so that the
        choice of curve only changes which of them is drawn.
        """
        if image is None or image.size == 0:
            self._backdrop = None
        else:
            rows = [luma_values(image).reshape(-1)] + [
                image[..., channel].reshape(-1) for channel in range(3)
            ]
            self._backdrop = np.stack(
                [np.bincount(row, minlength=BINS)[:BINS] for row in rows]
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
        """Fill the curve's histograms, faint enough to read the curve over."""
        counts = self._backdrop
        assert counts is not None  # the caller has looked
        rows = _BACKDROP_ROWS[self._channel]
        shown = counts[list(rows)]
        scale = max(1.0, float(np.percentile(shown, SCALE_PERCENTILE)) * SCALE_HEADROOM)
        painter.setPen(Qt.PenStyle.NoPen)
        for row in rows:
            path = QPainterPath()
            path.moveTo(area.left(), area.bottom())
            for index in range(BINS):
                x = area.left() + area.width() * index / (BINS - 1)
                height = min(1.0, counts[row, index] / scale) * area.height()
                path.lineTo(x, area.bottom() - height)
            path.lineTo(area.right(), area.bottom())
            path.closeSubpath()
            fill = QColor(_ROW_COLOURS[row])
            fill.setAlpha(theme.CURVE_BACKDROP_ALPHA)
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
        painter.setPen(QPen(QColor(self.curve_colour()), theme.CURVE_LINE_WIDTH))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def curve_colour(self) -> str:
        """Return the colour the curve and the rings of its points are drawn in."""
        if not self.isEnabled():
            return theme.TEXT_TERTIARY
        return _CURVE_COLOURS[self._channel]

    def _draw_points(self, painter: QPainter) -> None:
        """Draw every point: hollow at rest, lit under the pointer, solid when chosen."""
        painter.setPen(QPen(QColor(self.curve_colour()), theme.CURVE_LINE_WIDTH))
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
    """The choice of curve, its square and the line under it that reads the chosen point."""

    #: The curve shown as it now is, whether a point is still under the hand,
    #: and which curve it is, by its name in EditState.
    curve_set = Signal(object, bool, str)
    #: Another curve was chosen to be drawn.
    channel_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        #: Every curve the photo holds, the shown one among them.
        self._curves: dict[str, tone_curve.Curve] = {
            name: tone_curve.STRAIGHT for name in CURVE_NAMES
        }
        self.channels = Segmented(CURVE_CHOICES, self)
        self.plot = CurvePlot(self)
        self.readout = QLabel(self)
        self.readout.setProperty("role", "caption")
        self.readout.setFixedHeight(theme.CURVE_READOUT_HEIGHT)
        self.readout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        use_tabular_figures(self.readout)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.channels)
        layout.addSpacing(theme.CURVE_CHOICE_GAP)
        layout.addWidget(self.plot, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(theme.SPACE_XS)
        layout.addWidget(self.readout)

        self.channels.chosen.connect(self._choose)
        self.plot.curve_set.connect(self._on_drawn)
        self.plot.readout_changed.connect(self._show_readout)
        self._show_readout()

    @property
    def channel(self) -> str:
        """Return which curve is shown, by its name in EditState."""
        return self.plot.channel

    @property
    def curve(self) -> tone_curve.Curve:
        """Return the points of the shown curve as they now are."""
        return self.plot.curve

    def curve_of(self, name: str) -> tone_curve.Curve:
        """Return the points of the curve called *name* as the editor holds them."""
        return self._curves[name]

    def set_curve(self, curve: tone_curve.Curve, name: str | None = None) -> None:
        """Hold *curve* as the curve called *name*, without announcing it.

        The shown curve is meant when no name is given.
        """
        name = self.channel if name is None else name
        self._curves[name] = curve
        self.channels.set_marked(name, not tone_curve.is_straight(curve))
        if name == self.channel:
            self.plot.set_curve(curve)

    def set_curves(self, state: EditState) -> None:
        """Hold every curve of *state* without announcing it, as undo and presets do."""
        for name in CURVE_NAMES:
            self.set_curve(state.curve(name), name)

    def set_channel(self, name: str) -> None:
        """Show the curve called *name*, as a click on its name does."""
        if name == self.channel:
            return
        self.channels.set_value(name)
        self._choose(name)

    def set_backdrop(self, image: NDArray[np.uint8] | None) -> None:
        """Lay the histograms of *image* behind the curve, or none."""
        self.plot.set_backdrop(image)

    def retranslate(self) -> None:
        """Read the names and the line under the square again after a language change."""
        self.channels.retranslate()
        self._show_readout()

    def _choose(self, name: str) -> None:
        """Draw the curve called *name* instead of the one shown."""
        self.plot.set_channel(name)
        self.plot.set_curve(self._curves[name])
        self.channel_changed.emit(name)

    def _on_drawn(self, curve: tone_curve.Curve, dragging: bool) -> None:
        """Keep the curve just drawn and pass it on with its name."""
        name = self.channel
        self._curves[name] = curve
        self.channels.set_marked(name, not tone_curve.is_straight(curve))
        self.curve_set.emit(curve, dragging, name)

    def _show_readout(self) -> None:
        """Read the chosen point in levels, or say how to begin."""
        levels = self.plot.readout()
        if levels is None:
            self.readout.setText(tr("curve.hint"))
            return
        self.readout.setText(tr("curve.readout", input=levels[0], output=levels[1]))
