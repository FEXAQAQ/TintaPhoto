"""The 80 ms hover transition of PLAN.md 3.6.

A hover state written in a stylesheet is an instant swap: Qt's stylesheets
have no transitions. A widget that wants the documented 80 ms keeps one of
these instead. It holds how far the hover has come - 0 at rest, 1 under the
pointer - and the widget paints itself with that, which is why the colours
the stylesheet would have swapped are taken out of it and drawn here.

Windows set to show no animation gets the straight swap it asks for.
"""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QObject, QRectF, Qt, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ui import native, theme

#: Closer than this to where it is going, a transition has arrived.
_SETTLED = 0.001


class HoverFade(QObject):
    """How far a widget's hover has come, from 0 at rest to 1 under the pointer."""

    #: The value moved; whatever else follows the hover can listen for it.
    changed = Signal(float)

    def __init__(self, widget: QWidget) -> None:
        super().__init__(widget)
        self._widget = widget
        self._value = 0.0
        self._animation: QVariantAnimation | None = None

    @property
    def value(self) -> float:
        """Return how far the hover has come.

        Zero while the widget is disabled, whatever the pointer is doing: a
        button that cannot be pressed must not answer it (PLAN.md 3.8).
        """
        if not self._widget.isEnabled():
            return 0.0
        return self._value

    def set_hovered(self, hovered: bool) -> None:
        """Slide towards *hovered* over the documented 80 ms (PLAN.md 3.6)."""
        target = 1.0 if hovered else 0.0
        self._stop()
        if not native.animations_enabled():
            self._settle(target)
            return
        if abs(target - self._value) < _SETTLED:
            return
        animation = QVariantAnimation(self)
        animation.setStartValue(float(self._value))
        animation.setEndValue(target)
        animation.setDuration(theme.DURATION_HOVER_MS)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.valueChanged.connect(lambda value: self._apply(float(value)))
        # The animation is owned by this object and let go of when it ends,
        # the way the zoom's is: asking Qt to delete it when it stops would
        # leave the reference here pointing at a deleted object.
        animation.finished.connect(lambda: self._settle(target))
        animation.start()
        self._animation = animation

    def _settle(self, target: float) -> None:
        """Come to rest exactly on *target*, however near the last step got."""
        self._stop()
        self._apply(target)

    def _apply(self, value: float) -> None:
        self._value = value
        self.changed.emit(value)
        self._widget.update()

    def _stop(self) -> None:
        animation = self._animation
        self._animation = None
        if animation is not None:
            animation.stop()
            animation.deleteLater()


def shade(colour: str, strength: float) -> QColor:
    """Return *colour* at *strength* of its full opacity."""
    faded = QColor(colour)
    faded.setAlphaF(max(0.0, min(1.0, strength)))
    return faded


def mix(resting: str, hovered: str, strength: float) -> QColor:
    """Return the colour *strength* of the way from *resting* to *hovered*."""
    start = QColor(resting)
    end = QColor(hovered)
    amount = max(0.0, min(1.0, strength))
    return QColor(
        round(start.red() + (end.red() - start.red()) * amount),
        round(start.green() + (end.green() - start.green()) * amount),
        round(start.blue() + (end.blue() - start.blue()) * amount),
    )


def paint_fill(
    widget: QWidget, strength: float, radius: int, colour: str = theme.BG_RAISED
) -> None:
    """Fill *widget* behind whatever the stylesheet draws over it.

    Called before the stylesheet has its turn, which works because the
    resting background of the widgets that use this is clear: the name and
    the icon are then drawn on top of the colour rather than under it.
    """
    if strength <= 0.0:
        return
    painter = QPainter(widget)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(shade(colour, strength))
    painter.drawRoundedRect(QRectF(widget.rect()), radius, radius)
    painter.end()


def paint_border(widget: QWidget, strength: float, radius: int, colour: str) -> None:
    """Draw the brighter edge over the resting one the stylesheet drew.

    Over rather than under, because the resting border is still there and
    the brighter one has to replace it rather than sit beneath it.
    """
    if strength <= 0.0:
        return
    painter = QPainter(widget)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QPen(shade(colour, strength), theme.BORDER_WIDTH))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    # Inset by half the line, so that it lands inside the widget rather than
    # straddling its edge.
    half = theme.BORDER_WIDTH / 2
    painter.drawRoundedRect(
        QRectF(widget.rect()).adjusted(half, half, -half, -half), radius, radius
    )
    painter.end()
