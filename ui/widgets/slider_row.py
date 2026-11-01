"""The slider row: name, track and value (PLAN.md sections 3.5 and 3.6).

The most used control in the application, so it carries every interaction the
plan asks for: drag the track, drag the name to scrub, hold shift to go
slowly, double click to go back to where it started, click the number to type
one, and arrow keys once it has the focus. The wheel deliberately does
nothing, so that scrolling the panel cannot change a photo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from PySide6.QtCore import QEvent, QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QWidget

from app.i18n import tr
from ui import theme
from ui.widgets.focus import ring_on_keyboard_only
from ui.widgets.text import format_number, use_tabular_figures

#: Width of the name and the value either side of the track.
NAME_WIDTH: Final[int] = 68
VALUE_WIDTH: Final[int] = 48

#: How much of the range one pixel of dragging covers when scrubbing the name.
SCRUB_RANGE_PER_PIXEL: Final[float] = 1.0 / 200.0
#: Shift slows every way of changing the value by this much (PLAN.md 3.6).
FINE_FACTOR: Final[float] = 0.1
#: One press of an arrow key, as a fraction of the range.
ARROW_STEP: Final[float] = 1.0 / 100.0
#: Decimal places a value is rounded to after being put on a step, which is
#: far finer than any step and coarse enough to clear up binary rounding.
_STEP_PLACES: Final[int] = 9

#: Radius of the dots at the ends of the white balance tracks (PLAN.md 3.6);
#: their colours are tokens like every other colour.
MARK_RADIUS: Final[int] = 2

#: The ways of arriving at a slider that count as using the keyboard, and so
#: show the focus ring (PLAN.md 3.6: not while the mouse is being used).
_KEYBOARD_FOCUS: Final[frozenset[Qt.FocusReason]] = frozenset(
    {
        Qt.FocusReason.TabFocusReason,
        Qt.FocusReason.BacktabFocusReason,
        Qt.FocusReason.ShortcutFocusReason,
    }
)


@dataclass(frozen=True)
class SliderSpec:
    """What one slider is: its range, where it rests and how it reads."""

    name: str
    label_key: str
    minimum: float
    maximum: float
    default: float
    decimals: int = 0
    #: Show a plus or minus in front, as the bipolar sliders do.
    signed: bool = True
    #: Lay the track out in reciprocal temperature, so that a step feels the
    #: same at both ends of the range (PLAN.md section 6 step 3).
    reciprocal: bool = False
    #: Colours of the dots at either end, if it has them.
    marks: tuple[str, str] | None = None
    #: Appended to the number, like the K of a temperature.
    suffix: str = ""
    #: Show the name on the left. A dialog row has a label column of its own,
    #: so the slider there starts at the left edge instead (PLAN.md 3.11).
    show_name: bool = True
    #: The smallest change the slider can make, or zero for one that runs
    #: free. Only the exposure slider has one: PLAN.md 2.2 gives it twentieths
    #: of a stop, which is fine enough that no step can be seen and coarse
    #: enough that the number can be said out loud.
    step: float = 0.0
    #: Where the filled part of the track starts, if not where the slider
    #: rests: a quality runs up from its lowest value rather than out from
    #: where it happens to sit.
    fill_from: float | None = None


class SliderRow(QWidget):
    """One adjustment: its name, its track and its number."""

    #: The adjustment's name, its new value, and whether a drag is still going.
    value_changed = Signal(str, float, bool)

    def __init__(self, spec: SliderSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spec = spec
        self._value = spec.default
        self._dragging = False
        self._scrubbing_from: QPoint | None = None
        self._hovering = False
        #: Whether the focus arrived by keyboard. PLAN.md 3.6 shows the
        #: focus ring then and not while the mouse is being used, so a
        #: click on a slider must not ring it.
        self._keyboard_focus = False
        self.setFixedHeight(theme.SLIDER_ROW_HEIGHT)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._name = QLabel(tr(spec.label_key), self)
        self._name.setProperty("role", "secondary")
        self._name.setFixedWidth(NAME_WIDTH)
        self._name.setCursor(Qt.CursorShape.SizeHorCursor)
        self._name.setToolTip(tr("slider.reset_hint"))
        self._name.setVisible(spec.show_name)

        self._value_label = QLabel("", self)
        self._value_label.setFixedWidth(VALUE_WIDTH)
        self._value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._value_label.setCursor(Qt.CursorShape.IBeamCursor)
        use_tabular_figures(self._value_label)

        self._editor = QLineEdit(self)
        ring_on_keyboard_only(self._editor)
        self._editor.setFixedWidth(VALUE_WIDTH)
        self._editor.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._editor.hide()
        self._editor.editingFinished.connect(self._finish_editing)
        use_tabular_figures(self._editor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_SM)
        layout.addWidget(self._name)
        layout.addStretch(1)
        layout.addWidget(self._value_label)
        layout.addWidget(self._editor)

        self._refresh_text()

    # -- the value -------------------------------------------------------

    @property
    def spec(self) -> SliderSpec:
        """Return what this slider is."""
        return self._spec

    @property
    def value(self) -> float:
        """Return the value shown."""
        return self._value

    def set_value(self, value: float, announce: bool = False) -> None:
        """Show *value*, and say so only if asked.

        A value the user has just arrived at is put on the slider's step
        first, where it has one. A value handed in from a parameter file is
        shown exactly as it is: an older file may hold an exposure between
        two steps, and the photo is developed with what the file really
        says, so the panel must not quietly round it on the screen.
        """
        if announce:
            value = self._on_step(value)
        clamped = max(self._spec.minimum, min(self._spec.maximum, float(value)))
        if clamped == self._value:
            return
        self._value = clamped
        self._refresh_text()
        self.update()
        if announce:
            self.value_changed.emit(self._spec.name, clamped, self._dragging)

    def retranslate(self) -> None:
        """Read the name and the hint again after a language change (U-02).

        The unit after the number lives in the description, so a tab whose
        sliders carry one hands in a fresh description of its own.
        """
        self._name.setText(tr(self._spec.label_key))
        self._name.setToolTip(tr("slider.reset_hint"))
        self._refresh_text()

    def set_spec(self, spec: SliderSpec) -> None:
        """Change what this slider is, as a RAW photo replacing a JPEG does.

        The whole description is replaced rather than only its range: how the
        number reads, which way the track runs and what it rests at all change
        with the kind of photo.
        """
        self._spec = spec
        self._value = max(spec.minimum, min(spec.maximum, self._value))
        self._name.setText(tr(spec.label_key))
        self._name.setVisible(spec.show_name)
        self._refresh_text()
        self.update()

    def reset(self) -> None:
        """Put the slider back where it rests (A-10)."""
        self.set_value(self._spec.default, announce=True)

    def _on_step(self, value: float) -> float:
        """Return *value* on the nearest step, for a slider that has one.

        Rounded again afterwards because three steps of 0.05 come to
        0.15000000000000002 in binary floating point, and that is the number
        that would be written into the parameter file and compared against
        next time to decide whether anything had been changed.
        """
        step = self._spec.step
        if step <= 0.0:
            return value
        return round(round(float(value) / step) * step, _STEP_PLACES)

    def _refresh_text(self) -> None:
        self._value_label.setText(self._format(self._value))

    def _format(self, value: float) -> str:
        text = format_number(value, self._spec.decimals, self._spec.signed)
        return f"{text}{self._spec.suffix}"

    # -- turning a value into a position and back ------------------------

    def _to_position(self, value: float) -> float:
        """Return where *value* sits along the track, from zero to one."""
        low, high = self._spec.minimum, self._spec.maximum
        if high == low:
            return 0.0
        if self._spec.reciprocal:
            # Laid out in reciprocal temperature, where equal steps look like
            # equal changes, while equal steps of kelvin do not. The warm end
            # is on the right, as it is everywhere else: a higher setting
            # says the light was bluer, so the photo comes out warmer.
            warmest, coolest = 1.0 / max(low, 1.0), 1.0 / max(high, 1.0)
            return float((warmest - 1.0 / max(value, 1.0)) / (warmest - coolest))
        return float((value - low) / (high - low))

    def _from_position(self, position: float) -> float:
        """Return the value at *position* along the track."""
        low, high = self._spec.minimum, self._spec.maximum
        position = max(0.0, min(1.0, position))
        if self._spec.reciprocal:
            warmest, coolest = 1.0 / max(low, 1.0), 1.0 / max(high, 1.0)
            return 1.0 / max(warmest - (warmest - coolest) * position, 1e-9)
        return low + (high - low) * position

    # -- geometry --------------------------------------------------------

    def _track_rect(self) -> QRect:
        left = NAME_WIDTH + theme.SPACE_SM if self._spec.show_name else 0
        right = self.width() - VALUE_WIDTH - theme.SPACE_SM
        return QRect(left, 0, max(1, right - left), self.height())

    def _origin(self) -> float:
        """Return the value the filled part of the track starts from."""
        spec = self._spec
        return spec.default if spec.fill_from is None else spec.fill_from

    def _handle_x(self) -> int:
        track = self._track_rect()
        return track.left() + round(self._to_position(self._value) * track.width())

    # -- painting --------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Draw the track, the fill from where it rests, and the handle."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        track = self._track_rect()
        middle = track.center().y()

        # Disabled, the slider goes the greys a disabled switch does
        # (PLAN.md 3.8): the track sinks to the border colour, and the fill
        # and the handle to the tertiary text, so it still says where it is.
        enabled = self.isEnabled()
        hovering = self._hovering and enabled
        painter.setPen(Qt.PenStyle.NoPen)
        if not enabled:
            track_colour = theme.BORDER
        else:
            track_colour = theme.BG_RAISED if hovering else theme.TRACK
        painter.setBrush(QColor(track_colour))
        painter.drawRect(
            QRect(track.left(), middle - theme.SLIDER_TRACK_HEIGHT // 2, track.width(), theme.SLIDER_TRACK_HEIGHT)
        )

        # The fill runs from where the slider rests to where it is now, so
        # that both how far and which way it has moved are visible at a glance.
        origin = track.left() + round(self._to_position(self._origin()) * track.width())
        handle = self._handle_x()
        if handle != origin:
            painter.setBrush(QColor(theme.ACCENT if enabled else theme.TEXT_TERTIARY))
            painter.drawRect(
                QRect(
                    min(origin, handle),
                    middle - theme.SLIDER_TRACK_HEIGHT // 2,
                    abs(handle - origin),
                    theme.SLIDER_TRACK_HEIGHT,
                )
            )

        self._draw_marks(painter, track, middle)

        if self._origin() == 0.0:

            painter.setPen(QPen(QColor(theme.TEXT_TERTIARY), 1))
            painter.drawLine(origin, middle - 4, origin, middle + 4)

        radius = (
            theme.SLIDER_HANDLE_SIZE_HOVER if hovering or self._dragging else theme.SLIDER_HANDLE_SIZE
        ) // 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.TEXT_PRIMARY if enabled else theme.TEXT_TERTIARY))
        painter.drawEllipse(QPoint(handle, middle), radius, radius)

        if self.hasFocus() and self._keyboard_focus:
            painter.setPen(QPen(QColor(theme.ACCENT), theme.FOCUS_RING_WIDTH))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            # Rounded like every other control, and half a pixel in so the
            # line lands on whole pixels instead of straddling the edge.
            inset = theme.FOCUS_RING_WIDTH / 2
            painter.drawRoundedRect(
                QRectF(self.rect()).adjusted(inset, inset, -inset, -inset),
                theme.RADIUS_CONTROL,
                theme.RADIUS_CONTROL,
            )

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Repaint in the disabled greys, or out of them, when that changes."""
        super().changeEvent(event)
        if event.type() == QEvent.Type.EnabledChange:
            if not self.isEnabled():
                self._hovering = False
            self.update()

    def _draw_marks(self, painter: QPainter, track: QRect, middle: int) -> None:
        """Draw the dots that say which way the white balance tracks run."""
        if self._spec.marks is None:
            return
        for colour, x in zip(
            self._spec.marks, (track.left() + MARK_RADIUS, track.right() - MARK_RADIUS)
        ):
            painter.setBrush(QColor(colour))
            painter.drawEllipse(QPoint(x, middle), MARK_RADIUS, MARK_RADIUS)

    # -- the mouse -------------------------------------------------------

    def _is_fine(self, event) -> bool:
        return bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)

    def focusInEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Ring the slider only when the focus came from the keyboard."""
        super().focusInEvent(event)
        self._keyboard_focus = event.reason() in _KEYBOARD_FOCUS
        self.update()

    def focusOutEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Take the ring away with the focus."""
        super().focusOutEvent(event)
        self._keyboard_focus = False
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Start dragging the track, or scrubbing from the name."""
        if event.button() is not Qt.MouseButton.LeftButton:
            return
        position = event.position().toPoint()
        self.setFocus(Qt.FocusReason.MouseFocusReason)

        if position.x() < NAME_WIDTH:
            self._scrubbing_from = position
            self._dragging = True
            return
        if position.x() > self.width() - VALUE_WIDTH:
            self._start_editing()
            return

        self._dragging = True
        self._move_to(position.x(), fine=self._is_fine(event))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Follow the drag, whether on the track or on the name."""
        position = event.position().toPoint()
        if not self._dragging:
            hovering = self._track_rect().contains(position)
            if hovering != self._hovering:
                self._hovering = hovering
                self.update()
            return

        if self._scrubbing_from is not None:
            travelled = position.x() - self._scrubbing_from.x()
            self._scrubbing_from = position
            span = self._spec.maximum - self._spec.minimum
            step = travelled * SCRUB_RANGE_PER_PIXEL * span
            if self._is_fine(event):
                step *= FINE_FACTOR
            self.set_value(self._value + step, announce=True)
            return

        self._move_to(position.x(), fine=self._is_fine(event))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Finish the drag, and say so, which is what gets recorded."""
        if event.button() is not Qt.MouseButton.LeftButton or not self._dragging:
            return
        self._dragging = False
        self._scrubbing_from = None
        self.update()
        self.value_changed.emit(self._spec.name, self._value, False)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Double clicking the name or the handle puts it back (A-10)."""
        position = event.position().toPoint()
        on_handle = abs(position.x() - self._handle_x()) <= theme.SLIDER_HANDLE_SIZE
        if position.x() < NAME_WIDTH or on_handle:
            self._dragging = False
            self.reset()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Stop showing the track as hovered."""
        if self._hovering:
            self._hovering = False
            self.update()
        super().leaveEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Ignore the wheel, so that scrolling the panel cannot change a photo."""
        event.ignore()

    def _move_to(self, x: int, fine: bool) -> None:
        """Set the value from a position on the track."""
        track = self._track_rect()
        position = (x - track.left()) / max(1, track.width())
        target = self._from_position(position)
        if fine:
            target = self._value + (target - self._value) * FINE_FACTOR
        self.set_value(target, announce=True)

    # -- the keyboard ----------------------------------------------------

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Nudge with the arrow keys once the slider has the focus."""
        self._keyboard_focus = True
        key = event.key()
        if key not in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            super().keyPressEvent(event)
            return
        span = self._spec.maximum - self._spec.minimum
        step = span * ARROW_STEP
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            step *= FINE_FACTOR
        if self._spec.step > 0.0:
            # The arrow keys are the fine adjustment of PLAN.md section 8, and
            # on a slider with a step the finest there is is one step. Shift
            # cannot go below it either: there is nothing in between.
            step = self._spec.step
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Down):
            step = -step
        # Along the track rather than along the value, so that a reciprocal
        # track moves as evenly as a straight one.
        position = self._to_position(self._value) + step / max(span, 1e-9)
        self.set_value(self._from_position(position), announce=True)
        self.value_changed.emit(self._spec.name, self._value, False)
        event.accept()

    # -- typing a number -------------------------------------------------

    def _start_editing(self) -> None:
        """Let the number be typed instead of dragged (A-11)."""
        self._editor.setText(format_number(self._value, self._spec.decimals, False))
        self._value_label.hide()
        self._editor.show()
        self._editor.selectAll()
        self._editor.setFocus(Qt.FocusReason.MouseFocusReason)

    def _finish_editing(self) -> None:
        """Take what was typed, if it is a number."""
        if self._editor.isHidden():
            return
        text = self._editor.text().replace("−", "-").strip()
        self._editor.hide()
        self._value_label.show()
        try:
            value = float(text)
        except ValueError:
            self._refresh_text()
            return
        self.set_value(value, announce=True)
        self.value_changed.emit(self._spec.name, self._value, False)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Escape leaves the typed number alone."""
        if event.key() == Qt.Key.Key_Escape and not self._editor.isHidden():
            self._editor.hide()
            self._value_label.show()
            self._refresh_text()
            return
        super().keyReleaseEvent(event)
