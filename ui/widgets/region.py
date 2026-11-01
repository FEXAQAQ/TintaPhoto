"""Helper for widgets that paint a background from the stylesheet.

Qt only fills the background of a plain ``QWidget`` subclass from a stylesheet
when ``WA_StyledBackground`` is set. Every named region of the window needs it;
every other container stays transparent so it cannot paint over the region
behind it.
"""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, Qt, QVariantAnimation
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget

from ui import native, theme



def styled_background(widget: QWidget, object_name: str) -> None:
    """Name *widget* and let the stylesheet paint its background."""
    widget.setObjectName(object_name)
    widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)


def set_state(widget: QWidget, name: str, value: bool) -> None:
    """Set a stylesheet property on *widget* and have Qt draw it again.

    Qt does not restyle a widget when a property it selects on changes, so
    every state that lives in the stylesheet has to be followed by this.
    """
    widget.setProperty(name, "true" if value else "false")
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)


class Scrim(QWidget):
    """Dims the window while a dialog is open (PLAN.md 3.11).

    One flat layer of dark, not a blurred or frosted one: PLAN.md 3.4 rules
    those out, and this only has to say that the window behind is not the
    thing to look at. It fades in and out with the dialog rather than
    cutting (PLAN.md 3.6), and simply appears and goes when Windows has
    animation effects turned off.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        #: How far the dim has come in, from 0 (none) to 1 (all of it).
        self._level = 0.0
        self._fade = QVariantAnimation(self)
        self._fade.setDuration(theme.SCRIM_FADE_MS)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade.valueChanged.connect(self._set_level)
        self._fade.finished.connect(self._settle)
        self.hide()

    @property
    def level(self) -> float:
        """Return how far the dim has come in, from 0 to 1."""
        return self._level

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Fill the whole area with the dimming colour."""
        colour = QColor(theme.SCRIM)
        colour.setAlphaF(theme.SCRIM_OPACITY * self._level)
        QPainter(self).fillRect(self.rect(), colour)

    def cover(self, widget: QWidget) -> None:
        """Lie over *widget* and fade in, until :meth:`uncover` is called."""
        if self.isHidden():
            self._level = 0.0
        self.setGeometry(widget.rect())
        self.show()
        self.raise_()
        self._fade_to(1.0)

    def uncover(self) -> None:
        """Fade out, and hide once gone; nothing happens if already going."""
        if not self.isVisible() or (
            self._fade.state() == QVariantAnimation.State.Running
            and self._fade.endValue() == 0.0
        ):
            return
        self._fade_to(0.0)

    def _fade_to(self, level: float) -> None:
        """Move towards *level*, at once when animation effects are off."""
        self._fade.stop()
        if not native.animations_enabled():
            self._set_level(level)
            self._settle()
            return
        self._fade.setStartValue(self._level)
        self._fade.setEndValue(level)
        self._fade.start()

    def _set_level(self, level: object) -> None:
        self._level = float(level)
        self.update()

    def _settle(self) -> None:
        """Hide once the dim has gone all the way out."""
        if self._level <= 0.0:
            self.hide()
