"""The small message that appears below the photo (PLAN.md section 3.6).

Confirmations and errors are shown here for two seconds rather than in a
dialog, so that nothing has to be dismissed and nothing covers the photo. A
second message replaces the first instead of queueing behind it.

A message may carry one thing to click on, which is how a finished export
offers to show the file (PLAN.md 3.11). Those stay longer, because two
seconds is not long enough to reach for something.

It floats over the window, so it rises a little into place while it fades
in and sinks back while it fades out (PLAN.md 3.6); with Windows animation
effects off it simply appears and goes.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QEasingCurve, QPoint, QTimer, Qt, QVariantAnimation
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QGraphicsOpacityEffect, QHBoxLayout, QLabel, QWidget

from ui import native, theme
from ui.widgets.buttons import LinkButton
from ui.widgets.region import styled_background


class Toast(QWidget):
    """A short message that fades away on its own."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        styled_background(self, "Toast")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self._label = QLabel("", self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setMaximumWidth(theme.TOAST_MAX_WIDTH)

        self._action = LinkButton("", parent=self)
        self._action.clicked.connect(self._do_action)
        self._action.hide()
        self._on_click: Callable[[], None] | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(theme.SPACE_LG, theme.SPACE_SM, theme.SPACE_LG, theme.SPACE_SM)
        layout.setSpacing(theme.SPACE_SM)
        layout.addWidget(self._label)
        layout.addWidget(self._action)

        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(theme.DURATION_TOAST_MS)
        self._timer.timeout.connect(self._leave)

        #: How present the message is, from 0 (gone) to 1 (in place).
        self._presence = 0.0
        self._home = QPoint()
        self._motion = QVariantAnimation(self)
        self._motion.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._motion.valueChanged.connect(self._set_presence)
        self._motion.finished.connect(self._settle)
        self.hide()

    @property
    def presence(self) -> float:
        """Return how far the message has come in, from 0 to 1."""
        return self._presence

    def show_message(
        self,
        message: str,
        action: str | None = None,
        on_click: Callable[[], None] | None = None,
    ) -> None:
        """Show *message*, with one thing to click on if there is one."""
        self._label.setText(message)
        # A short message stays on one line; only a long one, such as the
        # explanation of something that went wrong, is allowed to wrap.
        self._label.setWordWrap(
            QFontMetrics(self._label.font()).horizontalAdvance(message)
            > theme.TOAST_MAX_WIDTH
        )

        self._on_click = on_click if action else None
        self._action.setText(action or "")
        self._action.setVisible(bool(action))
        # A message with nothing to click on must not come between the
        # pointer and the photo.
        self.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, not action
        )
        self._timer.setInterval(
            theme.DURATION_TOAST_ACTION_MS if action else theme.DURATION_TOAST_MS
        )
        self.adjustSize()
        if self.isHidden():
            self._presence = 0.0
        self._place()
        self.show()
        self.raise_()
        self._move_to(1.0, theme.TOAST_SHOW_MS)
        self._timer.start()

    def _leave(self) -> None:
        """Sink back and fade out once the message has been up long enough."""
        self._move_to(0.0, theme.TOAST_HIDE_MS)

    def _move_to(self, presence: float, duration: int) -> None:
        self._motion.stop()
        if not native.animations_enabled():
            self._set_presence(presence)
            self._settle()
            return
        self._motion.setDuration(duration)
        self._motion.setStartValue(self._presence)
        self._motion.setEndValue(presence)
        self._motion.start()

    def _set_presence(self, presence: object) -> None:
        self._presence = float(presence)
        self._opacity.setOpacity(self._presence)
        self.move(self._home + QPoint(0, round((1.0 - self._presence) * theme.TOAST_RISE)))

    def _settle(self) -> None:
        if self._presence <= 0.0:
            self.hide()

    def _do_action(self) -> None:
        """Do what the message offered, and let it go."""
        callback = self._on_click
        self._motion.stop()
        self.hide()
        if callback is not None:
            callback()

    def _place(self) -> None:
        """Sit at the bottom centre of whatever this belongs to."""
        parent = self.parentWidget()
        if parent is None:
            return
        self._home = QPoint(
            (parent.width() - self.width()) // 2,
            parent.height() - self.height() - theme.SPACE_XXL,
        )
        self._set_presence(self._presence)

    def parent_resized(self) -> None:
        """Keep the message centred when the window changes size."""
        if self.isVisible():
            self._place()
