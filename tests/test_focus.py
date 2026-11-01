"""Tests for the keyboard only focus ring (PLAN.md 3.6).

PLAN.md 3.6: 鍵盤焦點顯示 1 px accent 色外框，滑鼠操作時不顯示.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QFocusEvent, QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QLineEdit  # noqa: E402

from ui import theme  # noqa: E402
from ui.widgets.focus import KEYBOARD_FOCUS, ring_on_keyboard_only  # noqa: E402


@pytest.fixture
def field(qt_app: object) -> QLineEdit:
    widget = QLineEdit()
    ring_on_keyboard_only(widget)
    return widget


def _focus(widget: QLineEdit, reason: Qt.FocusReason) -> None:
    QApplication.sendEvent(widget, QFocusEvent(QEvent.Type.FocusIn, reason))


def _ringed(widget: QLineEdit) -> bool:
    return widget.property(KEYBOARD_FOCUS) == "true"


def test_a_field_starts_without_a_ring(field: QLineEdit) -> None:
    assert not _ringed(field)


def test_clicking_a_field_does_not_ring_it(field: QLineEdit) -> None:
    _focus(field, Qt.FocusReason.MouseFocusReason)
    assert not _ringed(field)


def test_tabbing_to_a_field_rings_it(field: QLineEdit) -> None:
    _focus(field, Qt.FocusReason.TabFocusReason)
    assert _ringed(field)


def test_typing_rings_a_field_that_was_clicked(field: QLineEdit) -> None:
    """Once the keyboard is in use, where the typing lands has to show."""
    _focus(field, Qt.FocusReason.MouseFocusReason)
    QApplication.sendEvent(
        field,
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier, "a"),
    )
    assert _ringed(field)


def test_leaving_a_field_takes_the_ring_with_it(field: QLineEdit) -> None:
    _focus(field, Qt.FocusReason.TabFocusReason)
    QApplication.sendEvent(
        field, QFocusEvent(QEvent.Type.FocusOut, Qt.FocusReason.MouseFocusReason)
    )
    assert not _ringed(field)


def test_the_stylesheet_hangs_the_ring_on_the_keyboard() -> None:
    """``:focus`` matches however the focus arrived, which is the bug."""
    qss = theme.build_qss()
    assert 'QLineEdit[keyboardFocus="true"]' in qss
    assert 'QComboBox[keyboardFocus="true"]' in qss
    assert "QLineEdit:focus" not in qss
    assert "QComboBox:focus" not in qss
