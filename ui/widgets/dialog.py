"""The dialog every window of the program opens (PLAN.md 3.6, 3.11).

A dialog floats over the window, so it moves the way the other floating
things do: it drops a little into place while it fades in, and lifts a
little away while it fades out, with going quicker than coming. When
Windows has animation effects turned off it simply appears and goes.

Closing waits for the fade, so :attr:`Dialog.closing` says the moment the
dialog starts to go: the dim behind it can fade at the same time, and the
export can send its photo on its way without waiting for the dialog.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QEasingCurve, QPoint, QVariantAnimation, Signal
from PySide6.QtWidgets import QDialog, QWidget

from ui import native, theme


class Dialog(QDialog):
    """A modal dialog that fades and slides in and out."""

    #: The dialog has started to close; ``True`` if it was accepted. Sent
    #: before it fades out, whether or not it fades at all.
    closing = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._motion: QVariantAnimation | None = None
        self._home = QPoint()
        self._leaving = False

    @property
    def leaving(self) -> bool:
        """Return whether the dialog is fading away after being closed."""
        return self._leaving

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Drop into place while fading in, once Qt has placed the dialog."""
        super().showEvent(event)
        self._leaving = False
        self._home = self.pos()
        if not native.animations_enabled():
            return
        self._animate(theme.DIALOG_SHOW_MS, self._arrive, self._settle)

    def done(self, result: int) -> None:
        """Lift away while fading out, then close with *result*."""
        if self._leaving:
            return
        self._leaving = True
        self.closing.emit(result == QDialog.DialogCode.Accepted)
        if not self.isVisible() or not native.animations_enabled():
            self._finish(result)
            return
        start = self.windowOpacity()
        if self._motion is not None:
            self._motion.stop()
            self.move(self._home)

        def leave(k: float) -> None:
            self.setWindowOpacity(start * (1.0 - k))
            self.move(self._home - QPoint(0, round(k * theme.DIALOG_LIFT)))

        self._animate(theme.DIALOG_HIDE_MS, leave, lambda: self._finish(result))

    # -- motion ----------------------------------------------------------

    def _arrive(self, k: float) -> None:
        """Show the dialog *k* of the way in: faded, and above its place."""
        self.setWindowOpacity(k)
        self.move(self._home - QPoint(0, round((1.0 - k) * theme.DIALOG_DROP)))

    def _settle(self) -> None:
        """Rest where Qt put the dialog, fully there."""
        self.setWindowOpacity(1.0)
        self.move(self._home)

    def _finish(self, result: int) -> None:
        """Close for good, leaving the dialog as it would be shown again."""
        super().done(result)
        self._settle()
        self._leaving = False

    def _animate(
        self, duration: int, step: Callable[[float], None], finished: Callable[[], None]
    ) -> None:
        """Run *step* from 0 to 1 over *duration* ms, then *finished*."""
        if self._motion is not None:
            self._motion.stop()
        animation = QVariantAnimation(self)
        animation.setDuration(duration)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.valueChanged.connect(lambda value: step(float(value)))
        animation.finished.connect(finished)
        self._motion = animation
        step(0.0)
        animation.start()
