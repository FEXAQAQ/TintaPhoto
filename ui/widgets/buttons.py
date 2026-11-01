"""Shared button widgets.

Every clickable element in the interface must show a hover, pressed and
disabled state, and every icon button must carry a tooltip that includes its
keyboard shortcut (PLAN.md 3.6 and 3.8). Centralising the construction here
keeps all buttons of one kind identical.
"""

from __future__ import annotations

import math
from enum import Enum, auto

from PySide6.QtCore import (
    QElapsedTimer,
    QPointF,
    QRectF,
    QSize,
    Qt,
    QTimer,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import QColor, QFontMetricsF, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import QPushButton, QWidget

from app.i18n import tr
from ui import icons, native, theme
from ui.widgets.hover import HoverFade, mix, paint_fill


ROLE_ICON = "icon"
ROLE_ICON_SMALL = "icon_small"
ROLE_PRIMARY = "primary"
#: The one button of the welcome screen, outlined rather than filled: the
#: filled button of PLAN.md 3.5 is the export button.
ROLE_HERO = "hero"
ROLE_TAB = "tab"
ROLE_LINK = "link"
#: One of a set of choices, such as a crop ratio: the chosen one is marked
#: with the accent colour, which PLAN.md 3.4 reserves for the current state.
ROLE_CHOICE = "choice"


def tooltip(label_key: str, shortcut_key: str | None = None) -> str:
    """Return a tooltip for *label_key*, with its shortcut when there is one."""
    label = tr(label_key)
    if shortcut_key is None:
        return label
    return tr("tooltip.with_shortcut", label=label, shortcut=tr(shortcut_key))


class IconButton(QPushButton):
    """A 28 x 28 px borderless button showing an 18 px line icon."""

    def __init__(
        self,
        icon_name: str,
        label_key: str,
        shortcut_key: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._label_key = label_key
        self._shortcut_key = shortcut_key
        self.setProperty("role", ROLE_ICON)
        self.setIcon(icons.load_icon(icon_name))
        self.setIconSize(QSize(theme.ICON_SIZE, theme.ICON_SIZE))
        self.setFixedSize(theme.ICON_BUTTON_SIZE, theme.ICON_BUTTON_SIZE)
        self.setToolTip(tooltip(label_key, shortcut_key))
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)

    def retranslate(self) -> None:
        """Read the tooltip again after a language change (U-02)."""
        self.setToolTip(tooltip(self._label_key, self._shortcut_key))


class SmallIconButton(IconButton):
    """A 20 x 20 px borderless button showing a 12 px line icon.

    For a control tucked beside something it belongs to, such as the
    histogram's arrows, where a full size icon button would crowd it.
    """

    def __init__(
        self,
        icon_name: str,
        label_key: str,
        shortcut_key: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(icon_name, label_key, shortcut_key, parent)
        self.setProperty("role", ROLE_ICON_SMALL)
        self.setIcon(icons.load_icon(icon_name, size=theme.ICON_SIZE_SMALL))
        self.setIconSize(QSize(theme.ICON_SIZE_SMALL, theme.ICON_SIZE_SMALL))
        self.setFixedSize(theme.SMALL_ICON_BUTTON_SIZE, theme.SMALL_ICON_BUTTON_SIZE)


class PrimaryButton(QPushButton):
    """The filled button; only one may be visible at a time (PLAN.md 3.5)."""

    def __init__(
        self,
        label_key: str,
        shortcut_key: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(tr(label_key), parent)
        self._label_key = label_key
        self._shortcut_key = shortcut_key
        self.setProperty("role", ROLE_PRIMARY)
        self.setFixedHeight(theme.BUTTON_HEIGHT)
        self.setToolTip(tooltip(label_key, shortcut_key))
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)

    def retranslate(self) -> None:
        """Read the name and the tooltip again after a language change."""
        self.setText(tr(self._label_key))
        self.setToolTip(tooltip(self._label_key, self._shortcut_key))


class TextButton(QPushButton):
    """An outlined text button, used for the panel action rows."""

    def __init__(self, label_key: str, parent: QWidget | None = None) -> None:
        super().__init__(tr(label_key), parent)
        self._label_key = label_key
        self.setFixedHeight(theme.BUTTON_HEIGHT)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)

    def retranslate(self) -> None:
        """Read the name again after a language change."""
        self.setText(tr(self._label_key))

    def set_label_key(self, label_key: str) -> None:
        """Name the button by another text key, as a choice that becomes a change."""
        self._label_key = label_key
        self.retranslate()


class LinkButton(QPushButton):
    """A borderless left aligned button, used for recent folder entries.

    Its name brightens and a fill comes up under it over the documented
    80 ms (PLAN.md 3.6). The stylesheet asks for the name's colour from the
    palette, which is what lets the transition set it a step at a time.
    """

    def __init__(
        self,
        text: str,
        icon_name: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(text, parent)
        self.setProperty("role", ROLE_LINK)
        self.setFixedHeight(theme.BUTTON_HEIGHT)
        if icon_name is not None:
            self.setIcon(icons.load_icon(icon_name))
            self.setIconSize(QSize(theme.ICON_SIZE, theme.ICON_SIZE))
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._hover = HoverFade(self)
        self._hover.changed.connect(self._follow_hover)
        self._follow_hover(0.0)

    def _follow_hover(self, strength: float) -> None:
        """Take the name the rest of the way to the brighter grey."""
        palette = self.palette()
        palette.setColor(
            QPalette.ColorRole.ButtonText,
            mix(theme.TEXT_SECONDARY, theme.TEXT_PRIMARY, strength),
        )
        self.setPalette(palette)

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().enterEvent(event)
        self._hover.set_hovered(True)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().leaveEvent(event)
        self._hover.set_hovered(False)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Draw the fill under the name the stylesheet draws."""
        paint_fill(self, self._hover.value, theme.RADIUS_CONTROL)
        super().paintEvent(event)


#: Qt's QWIDGETSIZE_MAX, which PySide6 does not export: no limit on a size.
_NO_LIMIT = (1 << 24) - 1

#: How full the fill must be before the button says the export is done; the
#: last sliver is less than a pixel on a button this size.
_FULL = 0.98


class ExportLook(Enum):
    """What the export button is showing (PLAN.md 3.6, X-07)."""

    #: The button as it rests, drawn by the stylesheet.
    REST = auto()
    #: The export has started and the photo is still flying into the button.
    LANDING = auto()
    #: Filling from the left while the export runs.
    WORKING = auto()
    #: Saying the export is done, then fading back to rest.
    DONE = auto()


class ProgressButton(PrimaryButton):
    """The export button, which also shows how far an export has got (X-07).

    PLAN.md 3.6 asks for the progress to appear on the button itself rather
    than in a window over the photo. The moment an export starts the photo
    flies into the button, which flashes as it lands; then the button widens
    to say it is exporting and fills from the left, its words turning dark
    where the fill has passed them. The fill catches up with each step the
    export reports and creeps on between steps, slowing as it goes, so it
    never stands still, never runs past the next step and never goes back.
    When the file is written the button says so with a tick for a moment,
    then fades back to its name.

    With Windows animation effects off there is no flight, no flash and no
    creep: the fill stands at what was reported, and the button changes
    from one look to the next at once.

    An export of several photos (X-09) names how many on the button at rest,
    says which photo it is on while it works, and can be stopped: the
    pointer over the working button turns its words into "stop exporting".
    """

    #: The button went back to its own look; the room kept for it can shrink.
    rested = Signal()

    def __init__(
        self,
        label_key: str,
        shortcut_key: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(label_key, shortcut_key, parent)
        #: The tooltip stays the command's own name, whatever the button says.
        self._tip_key = label_key
        self._label_values: dict[str, object] = {}
        #: What the button says while it works, in place of "exporting", and
        #: the longest it will say, which its width is kept for.
        self._working_words: str | None = None
        self._working_widest: str | None = None
        self._stoppable = False
        self._look = ExportLook.REST
        #: The last step the export reported, or ``None`` with none running.
        self._progress: float | None = None
        #: How far the fill is drawn, which follows the progress smoothly.
        self._shown = 0.0
        #: Milliseconds spent in the current look.
        self._elapsed = 0
        #: When the flying photo lands, counted from the start.
        self._land_ms = 0
        self._finishing = False
        self._animated = True
        self._rest_width = 0
        self._width_from = 0.0
        self._width_to = 0.0
        self._width_elapsed = 0
        self._width_ms = theme.DURATION_PANEL_MS
        self._clock = QElapsedTimer()
        # A looping animation used only as a source of frames: each frame
        # moves the look on by the time that has really passed.
        self._frames = QVariantAnimation(self)
        self._frames.setStartValue(0.0)
        self._frames.setEndValue(1.0)
        self._frames.setDuration(theme.EXPORT_DONE_HOLD_MS)
        self._frames.setLoopCount(-1)
        self._frames.valueChanged.connect(self._on_frame)
        self._hold = QTimer(self)
        self._hold.setSingleShot(True)
        self._hold.setInterval(theme.EXPORT_DONE_HOLD_MS)
        self._hold.timeout.connect(self.rest)

    # -- what the window tells it ---------------------------------------

    @property
    def look(self) -> ExportLook:
        """Return what the button is showing."""
        return self._look

    @property
    def progress(self) -> float | None:
        """Return how far the export has got, or ``None`` if none is running."""
        return self._progress

    @property
    def shown(self) -> float:
        """Return how far the fill is drawn, from 0 to 1."""
        return self._shown

    @property
    def stoppable(self) -> bool:
        """Return whether pressing the working button stops the export."""
        return self._stoppable

    def set_name(self, label_key: str, **values: object) -> None:
        """Name the button at rest by *label_key*, such as "export 5" (X-09)."""
        self._label_key = label_key
        self._label_values = dict(values)
        self.setText(self._label())
        self.update()

    def set_working_words(self, words: str | None, widest: str | None = None) -> None:
        """Say *words* while working instead of "exporting"; ``None`` goes back.

        *widest* is the longest the words will grow to, such as the last
        photo of the count, so that the button does not twitch as it counts.
        """
        self._working_words = words
        self._working_widest = widest if words is not None else None
        self.setText(self._label())
        self.update()

    def set_stoppable(self, stoppable: bool) -> None:
        """Let the working button be pressed to stop the export (X-09)."""
        self._stoppable = stoppable
        self.setText(self._label())
        self.update()

    def start(self, land_ms: int = 0) -> None:
        """Begin an export whose photo lands on the button in *land_ms*.

        Whether the button can be pressed is not decided here: that depends
        on there being a photo open, which only the window knows.
        """
        if self._look is ExportLook.REST:
            self._rest_width = self.sizeHint().width()
        self._hold.stop()
        self._animated = native.animations_enabled()
        self._progress = 0.0
        self._shown = 0.0
        self._finishing = False
        self._land_ms = max(0, land_ms) if self._animated else 0
        self._set_width(self._rest_width, at_once=True)
        self._enter(ExportLook.LANDING if self._land_ms else ExportLook.WORKING)
        if self._animated:
            self._clock.start()
            self._frames.start()

    def set_progress(self, fraction: float | None) -> None:
        """Take *fraction* as the latest step, or ``None`` to go back to rest.

        A step lower than one already reported is ignored, so the fill can
        never go back.
        """
        if fraction is None:
            self.rest()
            return
        if self._look in (ExportLook.REST, ExportLook.DONE):
            self.start()
        fraction = max(0.0, min(1.0, fraction))
        self._progress = max(self._progress or 0.0, fraction)
        if not self._animated:
            self._shown = self._progress
        self.update()

    def finish(self) -> None:
        """Say the file was written, once the fill has reached the end."""
        if self._look in (ExportLook.REST, ExportLook.DONE):
            return
        self._progress = 1.0
        self._finishing = True
        # Nothing is left to stop while the fill runs to its end.
        self._stoppable = False
        if not self._animated:
            self._shown = 1.0
            self._enter(ExportLook.DONE)
            # Nothing moves, but the tick still gives way to the name.
            self._hold.start()

    def rest(self) -> None:
        """Go straight back to the button's own look."""
        self._frames.stop()
        self._hold.stop()
        self._look = ExportLook.REST
        self._progress = None
        self._shown = 0.0
        self._finishing = False
        # What an export of several photos said belongs to that export.
        self._working_words = None
        self._working_widest = None
        self._stoppable = False
        self._width_from = self._width_to
        self.setMinimumWidth(0)
        self.setMaximumWidth(_NO_LIMIT)
        self.setText(self._name())
        self.update()
        self.rested.emit()

    def retranslate(self) -> None:
        """Read the words again, in whichever look the button is."""
        self.setToolTip(tooltip(self._tip_key, self._shortcut_key))
        self.setText(self._label())

    def widest_width(self) -> int:
        """Return the width of the widest look, to keep room for it."""
        self.ensurePolished()
        return max(
            self._width_for(look)
            for look in (ExportLook.REST, ExportLook.WORKING, ExportLook.DONE)
        )

    # -- moving on --------------------------------------------------------

    def _on_frame(self, _value: object) -> None:
        self.advance(self._clock.restart())

    def advance(self, ms: int) -> None:
        """Move the look on by *ms* milliseconds; each frame calls this."""
        self._elapsed += ms
        self._advance_width(ms)
        if self._look is ExportLook.LANDING:
            if self._elapsed >= self._land_ms:
                self._enter(ExportLook.WORKING)
        elif self._look is ExportLook.WORKING:
            self._advance_fill(ms)
            if self._finishing and self._shown >= _FULL:
                self._enter(ExportLook.DONE)
        elif self._look is ExportLook.DONE:
            if self._elapsed >= theme.EXPORT_DONE_HOLD_MS + theme.EXPORT_DONE_FADE_MS:
                self.rest()
                return
            if (
                self._elapsed >= theme.EXPORT_DONE_HOLD_MS
                and self._width_to != self._rest_width
            ):
                self._width_ms = theme.EXPORT_DONE_FADE_MS
                self._set_width(self._rest_width)
        self.update()

    def _advance_fill(self, ms: int) -> None:
        """Let the fill follow the progress: quickly up to it, slowly past it.

        Past the last step it creeps towards a share of what is left, never
        reaching it, so it cannot run past a step that has not come yet.
        """
        if not self._animated:
            return
        progress = self._progress or 0.0
        ceiling = progress + theme.EXPORT_FILL_CREEP_SHARE * (1.0 - progress)
        if self._finishing:
            ceiling = 1.0
        pace = (
            theme.EXPORT_FILL_CATCH_MS
            if self._shown < progress
            else theme.EXPORT_FILL_CREEP_MS
        )
        step = (ceiling - self._shown) * (1.0 - math.exp(-ms / pace))
        self._shown = min(1.0, self._shown + max(0.0, step))

    def _enter(self, look: ExportLook) -> None:
        self._look = look
        self._elapsed = 0
        self.setText(self._label())
        if look is not ExportLook.LANDING:
            self._width_ms = theme.DURATION_PANEL_MS
            self._set_width(self._width_for(look), at_once=not self._animated)

    # -- width --------------------------------------------------------------

    def _width_for(self, look: ExportLook) -> int:
        """Return how wide the button is in *look*, padded as it is at rest."""
        metrics = QFontMetricsF(self.font())
        name = metrics.horizontalAdvance(self._name())
        rest = self.sizeHint().width() if self._look is ExportLook.REST else self._rest_width
        if look is ExportLook.WORKING:
            said = [tr("toolbar.exporting"), self._working_words, self._working_widest]
            if self._stoppable:
                said += [tr("toolbar.stop_export"), tr("toolbar.stopping")]
            words = max(metrics.horizontalAdvance(text) for text in said if text)
        elif look is ExportLook.DONE:
            words = (
                theme.EXPORT_TICK_SIZE
                + theme.SPACE_XS
                + metrics.horizontalAdvance(tr("toolbar.exported"))
            )
        else:
            return rest
        return max(rest, math.ceil(words + rest - name))

    def _set_width(self, width: float, at_once: bool = False) -> None:
        self._width_from = float(width if at_once else self.width())
        self._width_to = float(width)
        self._width_elapsed = 0
        if at_once:
            self.setFixedWidth(round(width))

    def _advance_width(self, ms: int) -> None:
        if self._width_from == self._width_to:
            return
        self._width_elapsed += ms
        k = min(1.0, self._width_elapsed / self._width_ms)
        eased = 1.0 - (1.0 - k) ** 3
        self.setFixedWidth(
            round(self._width_from + (self._width_to - self._width_from) * eased)
        )
        if k >= 1.0:
            self._width_from = self._width_to

    # -- drawing --------------------------------------------------------------

    def _name(self) -> str:
        """Return what the button is called at rest."""
        return tr(self._label_key, **self._label_values)

    def _label(self) -> str:
        if self._look is ExportLook.WORKING:
            if self._stoppable and self.underMouse():
                return tr("toolbar.stop_export")
            return self._working_words or tr("toolbar.exporting")
        if self._look is ExportLook.DONE:
            return tr("toolbar.exported")
        return self._name()

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Offer to stop, over a working button that can be stopped."""
        super().enterEvent(event)
        self.setText(self._label())
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().leaveEvent(event)
        self.setText(self._label())
        self.update()

    def pulse(self) -> float:
        """Return how bright the landing flash is, from 0 to 1."""
        if self._look is not ExportLook.LANDING or not self._land_ms:
            return 0.0
        half = theme.EXPORT_PULSE_MS / 2
        middle = self._land_ms - half
        return max(0.0, 1.0 - abs(self._elapsed - middle) / half)

    def done_fade(self) -> float:
        """Return how far "exported" has faded back to the name, from 0 to 1."""
        if self._look is not ExportLook.DONE:
            return 0.0
        k = (self._elapsed - theme.EXPORT_DONE_HOLD_MS) / theme.EXPORT_DONE_FADE_MS
        k = max(0.0, min(1.0, k))
        return 1.0 - (1.0 - k) ** 3

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Draw the look of the moment; at rest the stylesheet draws it."""
        if self._look is ExportLook.REST:
            super().paintEvent(event)
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setFont(self.font())
        face = QRectF(self.rect())
        shape = QPainterPath()
        shape.addRoundedRect(face, theme.RADIUS_CONTROL, theme.RADIUS_CONTROL)
        painter.setClipPath(shape)
        if self._look is ExportLook.WORKING:
            self._paint_working(painter, face)
        else:
            self._paint_face(painter, face)

    def _paint_working(self, painter: QPainter, face: QRectF) -> None:
        """The dark face, the fill, and the words turning over the fill.

        A button that can be stopped lightens under the pointer and darkens
        again while pressed, as any button does (PLAN.md 3.8).
        """
        colour = theme.BG_RAISED
        if self._stoppable and self.isDown():
            colour = theme.TRACK_PRESSED
        elif self._stoppable and self.underMouse():
            colour = theme.TRACK_HOVER
        painter.fillRect(face, QColor(colour))
        filled = QRectF(face.left(), face.top(), face.width() * self._shown, face.height())
        painter.fillRect(filled, QColor(theme.TEXT_PRIMARY))
        words = self._label()
        painter.save()
        painter.setClipRect(
            QRectF(filled.right(), face.top(), face.width(), face.height()),
            Qt.ClipOperation.IntersectClip,
        )
        painter.setPen(QColor(theme.TEXT_PRIMARY))
        painter.drawText(face, Qt.AlignmentFlag.AlignCenter, words)
        painter.restore()
        painter.save()
        painter.setClipRect(filled, Qt.ClipOperation.IntersectClip)
        painter.setPen(QColor(theme.BG_APP))
        painter.drawText(face, Qt.AlignmentFlag.AlignCenter, words)
        painter.restore()

    def _paint_face(self, painter: QPainter, face: QRectF) -> None:
        """The light face of rest, while the photo lands or once it is done.

        Once done the button can be pressed again, so it shows its hover,
        pressed and disabled states as the stylesheet would.
        """
        ink = QColor(theme.BG_APP)
        if self._look is ExportLook.LANDING:
            colour = QColor(theme.TEXT_PRIMARY)
        elif not self.isEnabled():
            colour, ink = QColor(theme.BG_RAISED), QColor(theme.TEXT_TERTIARY)
        elif self.isDown():
            colour = QColor(theme.TEXT_SECONDARY)
        elif self.underMouse():
            colour = QColor(theme.WHITE)
        else:
            colour = QColor(theme.TEXT_PRIMARY)
        painter.fillRect(face, colour)
        flash = self.pulse() * theme.EXPORT_PULSE_STRENGTH
        if flash > 0.0:
            white = QColor(theme.WHITE)
            white.setAlphaF(flash)
            painter.fillRect(face, white)
        if self._look is ExportLook.LANDING:
            painter.setPen(ink)
            painter.drawText(face, Qt.AlignmentFlag.AlignCenter, self._name())
            return
        fade = self.done_fade()
        painter.setOpacity(1.0 - fade)
        self._paint_done(painter, face, ink)
        painter.setOpacity(fade)
        painter.setPen(ink)
        painter.drawText(face, Qt.AlignmentFlag.AlignCenter, self._name())

    def _paint_done(self, painter: QPainter, face: QRectF, ink: QColor) -> None:
        """A tick, then "exported", centred together."""
        metrics = QFontMetricsF(painter.font())
        words = tr("toolbar.exported")
        tick = theme.EXPORT_TICK_SIZE
        across = tick + theme.SPACE_XS + metrics.horizontalAdvance(words)
        left = face.center().x() - across / 2
        middle = face.center().y()
        # The check of the icon set, on its 24 unit square.
        unit = tick / 24
        path = QPainterPath(QPointF(left + 4 * unit, middle))
        path.lineTo(QPointF(left + 9 * unit, middle + 5 * unit))
        path.lineTo(QPointF(left + 20 * unit, middle - 6 * unit))
        pen = QPen(ink, theme.ICON_STROKE_WIDTH)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        painter.setPen(ink)
        text = QRectF(left + tick + theme.SPACE_XS, face.top(), face.width(), face.height())
        painter.drawText(
            text, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, words
        )
