"""Showing a focus ring only while the keyboard is in use (PLAN.md 3.6).

A stylesheet's ``:focus`` matches however the focus arrived, so a field
clicked with the mouse was ringed as though it had been tabbed to. These
widgets carry a property instead, which goes true when the focus comes from
the keyboard, or as soon as a key is pressed after a click: by then the user
is working by keyboard and needs to see where what they type will land.
"""

from __future__ import annotations

from typing import Final

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QWidget

from ui.widgets.region import set_state

#: The stylesheet property the ring hangs on.
KEYBOARD_FOCUS: Final[str] = "keyboardFocus"

#: The ways of arriving at a control that count as using the keyboard.
_KEYBOARD_REASONS: Final[frozenset[Qt.FocusReason]] = frozenset(
    {
        Qt.FocusReason.TabFocusReason,
        Qt.FocusReason.BacktabFocusReason,
        Qt.FocusReason.ShortcutFocusReason,
    }
)


class _KeyboardFocusWatcher(QObject):
    """Keeps one widget's :data:`KEYBOARD_FOCUS` property up to date."""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Watch the focus and the keyboard; never swallow the event."""
        kind = event.type()
        if kind == QEvent.Type.FocusIn:
            self._set(watched, event.reason() in _KEYBOARD_REASONS)
        elif kind == QEvent.Type.FocusOut:
            self._set(watched, False)
        elif kind == QEvent.Type.KeyPress:
            self._set(watched, True)
        return False

    @staticmethod
    def _set(widget: QObject, wanted: bool) -> None:
        """Restyle the widget only when the answer has actually changed."""
        if not isinstance(widget, QWidget):
            return
        if (widget.property(KEYBOARD_FOCUS) == "true") != wanted:
            set_state(widget, KEYBOARD_FOCUS, wanted)


def ring_on_keyboard_only(widget: QWidget) -> None:
    """Ring *widget* when the keyboard reaches it, and not when the mouse does."""
    set_state(widget, KEYBOARD_FOCUS, False)
    widget.installEventFilter(_KeyboardFocusWatcher(widget))
