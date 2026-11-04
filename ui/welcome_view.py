"""The welcome screen shown before any photo is open (PLAN.md 3.7).

The drop of ink from the application icon, the name under it and the
tagline under that; then one column with the "open folder" button and the
folders opened lately, and under them a line saying photos can be dropped
anywhere on the window.

The drop, the name and the tagline are drawn by :class:`OpeningCanvas`
across the whole screen rather than by labels in the column, because the
opening moves them a letter at a time and sends rings out from the drop far
wider than the column. The rest are ordinary widgets that rise into place
one after another through :class:`RiseEffect`.

The opening plays once, when the program starts (PLAN.md 3.6, 3.7): a drop
falls and lands, rings spread from where it landed in the drop's four
colours, the drop grows to its size, the name comes up a letter at a time in
those colours and settles to white, the tagline is laid down from left to
right like a brush stroke, and the button, the list and the hint follow.
Coming back here after closing a folder only fades the screen in, and with
Windows set to show no animation the screen is simply there.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path

from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QSize,
    Qt,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QIcon,
    QKeySequence,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import (
    QGraphicsEffect,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.i18n import tr
from app.settings import RecentFolder
from ui import icons, native, theme
from ui.widgets.buttons import ROLE_HERO, IconButton, tooltip
from ui.widgets.hover import HoverFade, mix, paint_border, paint_fill, shade
from ui.widgets.ink import paint_ink_drop
from ui.widgets.region import styled_background
from ui.widgets.text import TranslatedLabel

# -- the clock -----------------------------------------------------------------


def _share(elapsed: float, starts_at: float, lasts: float) -> float:
    """Return how far a stage has come at *elapsed* milliseconds, from 0 to 1."""
    if lasts <= 0:
        return 1.0 if elapsed >= starts_at else 0.0
    return max(0.0, min(1.0, (elapsed - starts_at) / lasts))


def _ease_out(x: float) -> float:
    """Arrive rather than stop, as everything coming into view does (3.6)."""
    return 1.0 - (1.0 - x) ** 3


def _ease_in(x: float) -> float:
    """Speed up, as something falling does."""
    return x * x


def _ease_in_out(x: float) -> float:
    """Even through the middle, for a stroke that is watched."""
    return 4 * x**3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def _ease_back(x: float) -> float:
    """Overshoot a little and come back, for the drop finding its size."""
    s = 1.6
    x -= 1.0
    return x * x * ((s + 1) * x + s) + 1.0


def opening_length() -> int:
    """Return how long the whole opening lasts, from the stages in the theme."""
    rings = (
        theme.OPENING_FALL_MS
        + (len(theme.INK_COLOURS) - 1) * theme.OPENING_RIPPLE_GAP_MS
        + theme.OPENING_RIPPLE_MS
    )
    rows = (
        theme.OPENING_ROWS_AT_MS
        + (theme.WELCOME_RECENT_COUNT - 1) * theme.OPENING_ROW_GAP_MS
        + theme.OPENING_ITEM_MS
    )
    hint = theme.OPENING_HINT_AT_MS + theme.OPENING_ITEM_MS
    return max(rings, rows, hint)


def _font(pixels: int, weight: int, spacing: float = 0.0) -> QFont:
    """Return the interface's font stack at *pixels*, for text drawn by hand."""
    font = QFont()
    font.setFamilies([name for name in theme.FONT_STACK_UI if name != "sans-serif"])
    font.setPixelSize(pixels)
    font.setWeight(QFont.Weight(weight))
    if spacing:
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    return font


# -- what the opening draws ----------------------------------------------------------


class OpeningCanvas(QWidget):
    """The drop, its rings, the name and the tagline, across the whole screen.

    It lies under everything else on the welcome screen and takes no mouse
    events. *anchor* is the space the column keeps for these three, and is
    where they are drawn; the rings and the falling drop reach past it.
    """

    def __init__(self, anchor: QWidget, parent: QWidget) -> None:
        super().__init__(parent)
        self._anchor = anchor
        #: How far into the opening the screen is, or ``None`` at rest.
        self._elapsed: float | None = None
        #: The whole canvas's opacity, which the fade back in moves.
        self._opacity = 1.0
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def set_elapsed(self, elapsed: float | None) -> None:
        """Draw the opening at *elapsed* milliseconds, or at rest for ``None``."""
        self._elapsed = elapsed
        self.update()

    def set_opacity(self, opacity: float) -> None:
        """Fade everything drawn here, as the screen comes up again."""
        self._opacity = opacity
        self.update()

    @property
    def name(self) -> str:
        """Return the name drawn under the drop."""
        return tr("app.name")

    @property
    def tagline(self) -> str:
        """Return the line drawn under the name."""
        return tr("welcome.tagline")

    @property
    def elapsed(self) -> float | None:
        """Return how far into the opening the canvas is drawn, or ``None`` at rest."""
        return self._elapsed

    def retranslate(self) -> None:
        """Draw the tagline again in the language just chosen (U-02)."""
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Draw every stage where the opening's clock leaves it."""
        t = float(opening_length()) if self._elapsed is None else self._elapsed
        origin = self._anchor.mapTo(self.parentWidget(), QPoint(0, 0))
        centre_x = origin.x() + self._anchor.width() / 2
        top = float(origin.y())
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setOpacity(self._opacity)
        self._paint_drop(painter, t, centre_x, top + theme.WELCOME_DROP_CENTRE)
        self._paint_name(painter, t, centre_x, top + theme.WELCOME_WORDMARK_BASELINE)
        self._paint_tagline(painter, t, centre_x, top + theme.WELCOME_TAGLINE_BASELINE)
        painter.end()

    def _paint_drop(self, painter: QPainter, t: float, x: float, y: float) -> None:
        land = theme.OPENING_FALL_MS
        small = theme.OPENING_FALLING_DROP_HEIGHT
        if t < land:
            # Falling from above the top of the screen, stretched by its
            # speed, and gathering speed as it goes.
            fall = _ease_in(_share(t, 0, land))
            start = -float(small)
            here = start + (y - start) * fall
            paint_ink_drop(painter, QPointF(x, here), small, 0.82, 1.25)
            return
        for index, colour in enumerate(theme.INK_COLOURS):
            spread = _share(t, land + index * theme.OPENING_RIPPLE_GAP_MS, theme.OPENING_RIPPLE_MS)
            if 0.0 < spread < 1.0:
                radius = small / 3 + theme.OPENING_RIPPLE_RADIUS * _ease_out(spread)
                painter.save()
                painter.setOpacity(painter.opacity() * (1.0 - spread) ** 1.6)
                painter.setPen(
                    QPen(QColor(colour), theme.OPENING_RIPPLE_WIDTH * (1.0 - spread) + 0.4)
                )
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(
                    QPointF(x, y + theme.WELCOME_DROP_HEIGHT * 0.3),
                    radius,
                    radius * theme.OPENING_RIPPLE_FLATTEN,
                )
                painter.restore()
        grow = _ease_back(_share(t, land, theme.OPENING_GROW_MS))
        height = small + (theme.WELCOME_DROP_HEIGHT - small) * grow
        paint_ink_drop(painter, QPointF(x, y), height)

    def _paint_name(self, painter: QPainter, t: float, x: float, baseline: float) -> None:
        name = self.name
        font = _font(
            theme.FONT_SIZE_WORDMARK, theme.FONT_WEIGHT_TITLE, theme.WORDMARK_LETTER_SPACING
        )
        metrics = QFontMetricsF(font)
        left = x - metrics.horizontalAdvance(name) / 2
        painter.setFont(font)
        for index, letter in enumerate(name):
            start = theme.OPENING_LETTERS_AT_MS + index * theme.OPENING_LETTER_GAP_MS
            rise = _ease_out(_share(t, start, theme.OPENING_LETTER_MS))
            settle = _ease_out(
                _share(t, start + theme.OPENING_LETTER_SETTLE_AFTER_MS, theme.OPENING_LETTER_MS)
            )
            ink = theme.INK_COLOURS[index % len(theme.INK_COLOURS)]
            if rise > 0.0:
                painter.save()
                painter.setOpacity(painter.opacity() * rise)
                painter.setPen(mix(ink, theme.TEXT_PRIMARY, settle))
                lift = (1.0 - rise) * theme.OPENING_LETTER_RISE
                painter.drawText(QPointF(left, baseline + lift), letter)
                painter.restore()
            left += metrics.horizontalAdvance(name[: index + 1]) - metrics.horizontalAdvance(
                name[:index]
            )

    def _paint_tagline(self, painter: QPainter, t: float, x: float, baseline: float) -> None:
        laid = _ease_in_out(_share(t, theme.OPENING_TAGLINE_AT_MS, theme.OPENING_TAGLINE_MS))
        if laid <= 0.0:
            return
        line = self.tagline
        font = _font(theme.FONT_SIZE_TAGLINE, theme.FONT_WEIGHT_NORMAL)
        metrics = QFontMetricsF(font)
        width = metrics.horizontalAdvance(line)
        left = x - width / 2
        painter.save()
        if laid < 1.0:
            # The stroke's edge runs a little past the end, so the last
            # character is laid down whole rather than cut at the finish.
            reach = (width + theme.SPACE_XXL) * laid
            painter.setClipRect(
                QRectF(left, baseline - metrics.ascent(), reach, metrics.height())
            )
        painter.setFont(font)
        painter.setPen(QColor(theme.TEXT_SECONDARY))
        painter.drawText(QPointF(left, baseline), line)
        painter.restore()


class RiseEffect(QGraphicsEffect):
    """A widget coming up into place: fading in while it rises a few pixels."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._progress = 1.0

    @property
    def progress(self) -> float:
        """Return how far the widget has come up, from 0 to 1."""
        return self._progress

    def set_progress(self, progress: float) -> None:
        """Put the widget *progress* of the way into place."""
        self._progress = progress
        self.update()

    def boundingRectFor(self, rect: QRectF) -> QRectF:  # noqa: N802 - Qt naming
        """Leave room below for the widget while it is still low."""
        return rect.adjusted(0, 0, 0, theme.OPENING_ITEM_RISE)

    def draw(self, painter: QPainter) -> None:  # noqa: D102 - Qt's own hook
        shown = _ease_out(max(0.0, min(1.0, self._progress)))
        if shown <= 0.0:
            return
        # In logical coordinates the source starts at the widget's own
        # corner; the room added below by boundingRectFor is all under it.
        pixmap = self.sourcePixmap(Qt.CoordinateSystem.LogicalCoordinates)
        if pixmap.isNull():
            return
        painter.save()
        painter.setOpacity(shown)
        painter.drawPixmap(QPoint(0, round((1.0 - shown) * theme.OPENING_ITEM_RISE)), pixmap)
        painter.restore()


# -- what can be pressed ----------------------------------------------------------


class OpenFolderButton(QPushButton):
    """The one action: outlined, its icon and name at the left, its shortcut at the right."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("role", ROLE_HERO)
        self.setFixedHeight(theme.WELCOME_BUTTON_HEIGHT)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._hover = HoverFade(self)
        self.retranslate()

    @property
    def label(self) -> str:
        """Return the button's name."""
        return tr("welcome.open_folder")

    def retranslate(self) -> None:
        """Read the name and the tooltip again after a language change (U-02)."""
        self.setAccessibleName(self.label)
        self.setToolTip(tooltip("welcome.open_folder", "shortcut.open_folder"))
        self.update()

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().enterEvent(event)
        self._hover.set_hovered(True)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().leaveEvent(event)
        self._hover.set_hovered(False)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Draw the stylesheet's frame and hover, then the contents over them."""
        strength = self._hover.value
        paint_fill(self, strength, theme.RADIUS_CONTROL)
        super().paintEvent(event)
        paint_border(self, strength, theme.RADIUS_CONTROL, theme.TEXT_TERTIARY)

        enabled = self.isEnabled()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        middle = self.height() / 2
        left = theme.SPACE_LG + theme.SPACE_XS
        size = theme.ICON_SIZE
        icon = icons.load_icon(icons.ICON_FOLDER)
        mode = QIcon.Mode.Normal if enabled else QIcon.Mode.Disabled
        icon.paint(painter, QRect(left, round(middle - size / 2), size, size), mode=mode)
        name_font = _font(theme.FONT_SIZE_TITLE, theme.FONT_WEIGHT_TITLE)
        name_metrics = QFontMetricsF(name_font)
        painter.setFont(name_font)
        painter.setPen(QColor(theme.TEXT_PRIMARY if enabled else theme.TEXT_TERTIARY))
        baseline = middle + (name_metrics.ascent() - name_metrics.descent()) / 2
        painter.drawText(QPointF(left + size + theme.SPACE_SM, baseline), self.label)
        keys_font = _font(theme.FONT_SIZE_LABEL, theme.FONT_WEIGHT_NORMAL)
        keys_metrics = QFontMetricsF(keys_font)
        keys = tr("shortcut.open_folder")
        painter.setFont(keys_font)
        painter.setPen(QColor(theme.TEXT_TERTIARY))
        painter.drawText(
            QPointF(
                self.width() - left - keys_metrics.horizontalAdvance(keys),
                middle + (keys_metrics.ascent() - keys_metrics.descent()) / 2,
            ),
            keys,
        )
        painter.end()


def recent_day(opened: date, today: date) -> str:
    """Return when a folder was opened, the way a person would say it."""
    if opened == today:
        return tr("welcome.recent.today")
    if opened == today - timedelta(days=1):
        return tr("welcome.recent.yesterday")
    months = tr("welcome.recent.months").split(",")
    month_name = months[opened.month - 1].strip() if len(months) == 12 else str(opened.month)
    key = "welcome.recent.date" if opened.year == today.year else "welcome.recent.date_year"
    return tr(key, year=opened.year, month=opened.month, month_name=month_name, day=opened.day)


def recent_count(count: int) -> str:
    """Return how many photos a folder held, as words."""
    if count == 1:
        return tr("welcome.recent.count_one")
    return tr("welcome.recent.count", count=count)


#: Marks a recent folder's row, so the stylesheet can light the remove button
#: on it a step further than an icon button elsewhere.
ROLE_RECENT_ROW = "recent_row"


class RecentFolderRow(QPushButton):
    """One folder opened lately: its name over its path, its count over its day.

    While the pointer is over the row, or the row has the keyboard focus, the
    count and the day make way for a button that takes the folder off the
    list; Delete does the same from the keyboard. Only the record goes: the
    folder and its photos are not touched.
    """

    #: The folder is to be taken off the recent list.
    remove_requested = Signal()

    def __init__(
        self,
        folder: str,
        detail: RecentFolder | None,
        today: date,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.folder = folder
        self.detail = detail
        self._today = today
        self.setFixedHeight(theme.WELCOME_ROW_HEIGHT)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setToolTip(folder)
        self.setAccessibleName(self.name)
        self._hover = HoverFade(self)
        self._hover.changed.connect(lambda _value: self.update())
        self.setProperty("role", ROLE_RECENT_ROW)
        # Kept from the enter and leave events. Passing onto the remove
        # button does not leave the row, since the button is part of it.
        self._pointer_over = False
        self.remove_button = IconButton(
            icons.ICON_X, "welcome.recent.remove", "shortcut.recent_remove", self
        )
        # The row keeps the focus; Delete on it does what the button does.
        self.remove_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.remove_button.clicked.connect(self.remove_requested)
        self.remove_button.hide()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        """Report the row's own height: a button's stylesheet height would win."""
        return QSize(theme.WELCOME_CONTENT_WIDTH, theme.WELCOME_ROW_HEIGHT)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return QSize(theme.WELCOME_BUTTON_HEIGHT, theme.WELCOME_ROW_HEIGHT)

    @property
    def name(self) -> str:
        """Return the folder's own name, or the whole path for a drive."""
        return Path(self.folder).name or self.folder

    def side_text(self) -> tuple[str, str]:
        """Return the count and the day, or nothing for a folder with no record."""
        if self.detail is None:
            return "", ""
        return recent_count(self.detail.count), recent_day(self.detail.day, self._today)

    def retranslate(self) -> None:
        """Say the count and the day again in the language just chosen (U-02)."""
        self.remove_button.retranslate()
        self.update()

    @property
    def offers_removal(self) -> bool:
        """Return whether the remove button is showing in place of the count."""
        return not self.remove_button.isHidden()

    def _refresh_removal(self) -> None:
        shown = self.isEnabled() and (self._pointer_over or self.hasFocus())
        if shown != self.offers_removal:
            self.remove_button.setVisible(shown)
            self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Keep the remove button at the row's right end, centred on it."""
        super().resizeEvent(event)
        size = theme.ICON_BUTTON_SIZE
        self.remove_button.move(
            self.width() - theme.SPACE_SM - size, (self.height() - size) // 2
        )

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().enterEvent(event)
        self._pointer_over = True
        self._hover.set_hovered(True)
        self._refresh_removal()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().leaveEvent(event)
        self._pointer_over = False
        self._hover.set_hovered(False)
        self._refresh_removal()

    def focusInEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().focusInEvent(event)
        self._refresh_removal()

    def focusOutEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().focusOutEvent(event)
        self._refresh_removal()

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().changeEvent(event)
        if event.type() == QEvent.Type.EnabledChange:
            self._refresh_removal()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Delete takes the folder off the list, as the button does."""
        if event.matches(QKeySequence.StandardKey.Delete) and self.isEnabled():
            self.remove_requested.emit()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Draw the row by hand; the stylesheet's button would frame it."""
        enabled = self.isEnabled()
        strength = self._hover.value
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        bounds = QRectF(self.rect())
        if strength > 0.0 or self.isDown():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(
                QColor(theme.TRACK) if self.isDown() else shade(theme.BG_RAISED, strength)
            )
            painter.drawRoundedRect(bounds, theme.RADIUS_CONTROL, theme.RADIUS_CONTROL)
        if self.hasFocus():
            half = theme.FOCUS_RING_WIDTH / 2
            painter.setPen(QPen(QColor(theme.ACCENT), theme.FOCUS_RING_WIDTH))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(
                bounds.adjusted(half, half, -half, -half), theme.RADIUS_CONTROL, theme.RADIUS_CONTROL
            )

        pad = theme.SPACE_MD
        size = theme.ICON_SIZE
        icon = icons.load_icon(icons.ICON_FOLDER)
        icon.paint(
            painter,
            QRect(pad, (self.height() - size) // 2, size, size),
            mode=QIcon.Mode.Normal if enabled else QIcon.Mode.Disabled,
        )
        upper = self.height() / 2 - theme.SPACE_XS / 2
        lower = self.height() / 2 + theme.SPACE_XS / 2
        name_font = _font(theme.FONT_SIZE_RECENT_NAME, theme.FONT_WEIGHT_NORMAL)
        small_font = _font(theme.FONT_SIZE_CAPTION, theme.FONT_WEIGHT_NORMAL)
        count_font = _font(theme.FONT_SIZE_LABEL, theme.FONT_WEIGHT_NORMAL)
        count, day = self.side_text()
        right = self.width() - pad
        side = max(
            QFontMetricsF(count_font).horizontalAdvance(count),
            QFontMetricsF(small_font).horizontalAdvance(day),
        )
        text_left = pad + size + theme.SPACE_MD
        # The name keeps the same room whether the count or the remove
        # button is showing, so it does not jump as the pointer comes over.
        end = min(right - side - theme.SPACE_MD, self.remove_button.x() - theme.SPACE_SM)
        room = max(0.0, end - text_left)
        primary = theme.TEXT_PRIMARY if enabled else theme.TEXT_TERTIARY
        secondary = theme.TEXT_SECONDARY if enabled else theme.TEXT_TERTIARY

        def write(text: str, font: QFont, colour: str, x: float, baseline: float,
                  elide: Qt.TextElideMode | None = None, width: float = 0.0,
                  flush_right: bool = False) -> None:
            metrics = QFontMetricsF(font)
            if elide is not None:
                text = metrics.elidedText(text, elide, width)
            if flush_right:
                x -= metrics.horizontalAdvance(text)
            painter.setFont(font)
            painter.setPen(QColor(colour))
            painter.drawText(QPointF(x, baseline), text)

        name_metrics = QFontMetricsF(name_font)
        small_metrics = QFontMetricsF(small_font)
        write(self.name, name_font, primary, text_left, upper - name_metrics.descent(),
              Qt.TextElideMode.ElideRight, room)
        write(self.folder, small_font, theme.TEXT_TERTIARY, text_left,
              lower + small_metrics.ascent(), Qt.TextElideMode.ElideMiddle, room)
        if count and not self.offers_removal:
            count_metrics = QFontMetricsF(count_font)
            write(count, count_font, secondary, right, upper - count_metrics.descent(),
                  flush_right=True)
            write(day, small_font, theme.TEXT_TERTIARY, right, lower + small_metrics.ascent(),
                  flush_right=True)
        painter.end()


class DropHint(QWidget):
    """The line under the list: photos can be dropped anywhere on the window.

    A small dashed square stands for the drop. While something is dragged
    over the window the line comes up to the accent colour over the
    documented 80 ms, rather than switching to it (PLAN.md 3.6).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._lit = HoverFade(self)
        self._lit.changed.connect(lambda _value: self.update())
        self.setFixedHeight(theme.WELCOME_ROW_HEIGHT)

    @property
    def text(self) -> str:
        """Return what the line says."""
        return tr("welcome.drop_hint")

    def set_active(self, active: bool) -> None:
        """Light the line up, or let it go back to resting."""
        self._lit.set_hovered(active)

    @property
    def lit(self) -> float:
        """Return how far the line has come towards the accent colour."""
        return self._lit.value

    def retranslate(self) -> None:
        """Say the line again in the language just chosen (U-02)."""
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        font = _font(theme.FONT_SIZE_LABEL, theme.FONT_WEIGHT_NORMAL)
        width = QFontMetricsF(font).horizontalAdvance(self.text)
        return QSize(
            round(theme.DROP_HINT_ICON_SIZE + theme.SPACE_SM + width) + 1,
            theme.WELCOME_ROW_HEIGHT,
        )

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        colour = mix(theme.TEXT_TERTIARY, theme.ACCENT, self._lit.value)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        hint = self.sizeHint()
        left = (self.width() - hint.width()) / 2
        size = theme.DROP_HINT_ICON_SIZE
        top = (self.height() - size) / 2
        pen = QPen(colour, theme.BORDER_WIDTH)
        pen.setStyle(Qt.PenStyle.CustomDashLine)
        pen.setDashPattern([value / 2 for value in theme.DROP_ZONE_DASH])
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        half = theme.BORDER_WIDTH / 2
        painter.drawRoundedRect(
            QRectF(left + half, top + half, size - 2 * half, size - 2 * half),
            theme.RADIUS_THUMB / 2,
            theme.RADIUS_THUMB / 2,
        )
        font = _font(theme.FONT_SIZE_LABEL, theme.FONT_WEIGHT_NORMAL)
        metrics = QFontMetricsF(font)
        painter.setFont(font)
        painter.setPen(colour)
        painter.drawText(
            QPointF(
                left + size + theme.SPACE_SM,
                self.height() / 2 + (metrics.ascent() - metrics.descent()) / 2,
            ),
            self.text,
        )
        painter.end()


# -- the screen -------------------------------------------------------------------


class WelcomeView(QWidget):
    """The empty state of the preview area."""

    folder_requested = Signal(str)
    #: A folder is to be taken off the recent list (its record only).
    folder_removed = Signal(str)
    open_folder_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        styled_background(self, "WelcomeView")

        # Everything that can be pressed sits on this page; the canvas under
        # it draws the drop, the name and the tagline, and the opening.
        self._page = QWidget(self)
        self._column = QWidget(self._page)
        self._column.setFixedWidth(theme.WELCOME_CONTENT_WIDTH)

        #: The room the canvas draws the drop, the name and the tagline in.
        self.hero = QWidget(self._column)
        self.hero.setFixedHeight(theme.WELCOME_HERO_HEIGHT)
        self.canvas = OpeningCanvas(self.hero, self)
        self.canvas.lower()

        self.open_folder_button = OpenFolderButton(self._column)
        self.open_folder_button.clicked.connect(self.open_folder_requested)
        # Two things have to be true before a control can be used: the
        # window has something to act on, and the opening has put it on the
        # screen. Both are held here, and `_refresh_controls` is the one
        # place that works out what they come to.
        self._opening_enabled = False
        self._shown_by_opening = True
        self.open_folder_button.setEnabled(False)

        # The recent folders and the space above them are one thing that
        # goes away together: on a first run there is nothing to list, and
        # spacing left behind would open a hole in the screen.
        self._recent_block = QWidget(self._column)
        self.recent_title = TranslatedLabel("welcome.recent", self._recent_block)
        self.recent_title.setProperty("role", "caption")
        # In line with the folder icons of the rows under it.
        self.recent_title.setContentsMargins(theme.SPACE_MD, 0, 0, 0)
        self._recent_list = QWidget(self._recent_block)
        self._recent_layout = QVBoxLayout(self._recent_list)
        self._recent_layout.setContentsMargins(0, 0, 0, 0)
        self._recent_layout.setSpacing(theme.WELCOME_ROW_GAP)
        block_layout = QVBoxLayout(self._recent_block)
        block_layout.setContentsMargins(0, theme.WELCOME_SECTION_GAP, 0, 0)
        block_layout.setSpacing(theme.SPACE_SM)
        block_layout.addWidget(self.recent_title)
        block_layout.addWidget(self._recent_list)

        column_layout = QVBoxLayout(self._column)
        column_layout.setContentsMargins(0, 0, 0, 0)
        column_layout.setSpacing(0)
        column_layout.addWidget(self.hero)
        column_layout.addSpacing(theme.WELCOME_GROUP_GAP)
        column_layout.addWidget(self.open_folder_button)
        column_layout.addWidget(self._recent_block)

        self.drop_hint = DropHint(self._page)

        page_layout = QVBoxLayout(self._page)
        page_layout.setContentsMargins(
            theme.CANVAS_MARGIN, theme.CANVAS_MARGIN, theme.CANVAS_MARGIN, theme.CANVAS_MARGIN
        )
        page_layout.setSpacing(0)
        # The hint follows straight on from the column, centred with it, so
        # the screen reads as one group. Pinned to the bottom of the window it
        # was a footer belonging to nothing; it is not in the column itself
        # because in English it is wider than the column.
        page_layout.addStretch(1)
        page_layout.addWidget(self._column, 0, Qt.AlignmentFlag.AlignHCenter)
        page_layout.addSpacing(theme.WELCOME_GROUP_GAP)
        page_layout.addWidget(self.drop_hint, 0, Qt.AlignmentFlag.AlignHCenter)
        page_layout.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._page)

        # The screen comes up over the documented 120 ms when it is arrived
        # at again (PLAN.md 3.6). Switched off at rest, so that what is drawn
        # at rest is exactly what was drawn before there was a fade at all.
        self._page_opacity = QGraphicsOpacityEffect(self._page)
        self._page_opacity.setEnabled(False)
        self._page.setGraphicsEffect(self._page_opacity)
        self._fade: QVariantAnimation | None = None

        # What rises into place during the opening, each on its own clock.
        self._rises: dict[QWidget, RiseEffect] = {}
        for widget in (self.open_folder_button, self.recent_title, self.drop_hint):
            self._rise(widget)
        self._rows: list[RecentFolderRow] = []
        self._opening: QVariantAnimation | None = None
        self._opened = False

        self.set_recent_folders(())

    def _rise(self, widget: QWidget) -> RiseEffect:
        effect = RiseEffect(widget)
        effect.setEnabled(False)
        widget.setGraphicsEffect(effect)
        self._rises[widget] = effect
        return effect

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Keep the canvas over the whole screen, under the page."""
        super().resizeEvent(event)
        self.canvas.setGeometry(self.rect())

    # -- the folders opened lately ------------------------------------------

    def set_recent_folders(
        self,
        folders: Sequence[str],
        details: Mapping[str, RecentFolder] | None = None,
        today: date | None = None,
    ) -> None:
        """Show up to five recent folders; hide the list when there are none."""
        for row in self._rows:
            self._recent_layout.removeWidget(row)
            self._rises.pop(row, None)
            row.deleteLater()
        self._rows = []
        details = details or {}
        day = today or date.today()
        shown = list(folders)[: theme.WELCOME_RECENT_COUNT]
        self._recent_block.setVisible(bool(shown))
        for folder in shown:
            row = RecentFolderRow(folder, details.get(folder), day, self._recent_list)
            row.clicked.connect(lambda _checked=False, path=folder: self.folder_requested.emit(path))
            row.remove_requested.connect(lambda path=folder: self.folder_removed.emit(path))
            row.setEnabled(self._opening_enabled and self._shown_by_opening)
            effect = self._rise(row)
            if self._opening is not None:
                effect.setEnabled(True)
            self._recent_layout.addWidget(row)
            self._rows.append(row)
        # Settled now rather than on the next turn of the event loop, so the
        # room the list would take goes with it at once.
        layout = self._column.layout()
        if layout is not None:
            layout.activate()

    def recent_rows(self) -> list[RecentFolderRow]:
        """Return the recent folders' rows, for the window and the tests."""
        return list(self._rows)

    # -- coming up -------------------------------------------------------

    @property
    def fading(self) -> bool:
        """Return whether the screen is still coming up."""
        return self._fade is not None

    def fade_in(self) -> None:
        """Bring the screen up from the background over 120 ms (PLAN.md 3.6)."""
        self._stop_fade()
        if not native.animations_enabled():
            return
        animation = QVariantAnimation(self)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setDuration(theme.DURATION_PHOTO_FADE_MS)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.valueChanged.connect(lambda value: self._set_faded(float(value)))
        animation.finished.connect(self._stop_fade)
        self._page_opacity.setEnabled(True)
        self._set_faded(0.0)
        animation.start()
        self._fade = animation

    def _set_faded(self, opacity: float) -> None:
        self._page_opacity.setOpacity(opacity)
        self.canvas.set_opacity(opacity)

    def _stop_fade(self) -> None:
        """End the fade, leaving the screen all the way up."""
        animation = self._fade
        self._fade = None
        if animation is not None:
            animation.stop()
            animation.deleteLater()
        self._set_faded(1.0)
        self._page_opacity.setEnabled(False)

    # -- the opening (PLAN.md 3.7) ---------------------------------------

    @property
    def opening(self) -> bool:
        """Return whether the opening is still playing."""
        return self._opening is not None

    def play_opening(self) -> None:
        """Play the opening, once, from the drop falling to the hint at the bottom.

        One animation drives every stage rather than a timer each, so that
        the stages cannot drift apart; each takes the share of the whole
        that falls inside its own window and eases it itself.
        """
        self._stop_opening()
        if not native.animations_enabled():
            # Windows is set to show no animation, so the screen is simply
            # there (PLAN.md 3.6).
            return
        for effect in self._rises.values():
            effect.setEnabled(True)
        animation = QVariantAnimation(self)
        length = opening_length()
        animation.setStartValue(0.0)
        animation.setEndValue(float(length))
        animation.setDuration(length)
        # Linear here: each stage carries its own ease, and easing the clock
        # as well would ease everything twice.
        animation.setEasingCurve(QEasingCurve.Type.Linear)
        animation.valueChanged.connect(lambda value: self.apply_opening(float(value)))
        animation.finished.connect(self._stop_opening)
        self.apply_opening(0.0)
        animation.start()
        self._opening = animation

    def apply_opening(self, elapsed: float) -> None:
        """Put every stage where *elapsed* milliseconds leaves it."""
        self.canvas.set_elapsed(elapsed)
        item = theme.OPENING_ITEM_MS
        self._rises[self.open_folder_button].set_progress(
            _share(elapsed, theme.OPENING_BUTTON_AT_MS, item)
        )
        self._rises[self.recent_title].set_progress(
            _share(elapsed, theme.OPENING_LIST_AT_MS, item)
        )
        for index, row in enumerate(self._rows):
            self._rises[row].set_progress(
                _share(elapsed, theme.OPENING_ROWS_AT_MS + index * theme.OPENING_ROW_GAP_MS, item)
            )
        self._rises[self.drop_hint].set_progress(_share(elapsed, theme.OPENING_HINT_AT_MS, item))
        self._set_shown_by_opening(elapsed >= theme.OPENING_BUTTON_AT_MS)

    def _stop_opening(self) -> None:
        """End the opening, leaving the screen the way it rests."""
        animation = self._opening
        self._opening = None
        if animation is not None:
            animation.stop()
            animation.deleteLater()
        self.canvas.set_elapsed(None)
        for effect in self._rises.values():
            effect.set_progress(1.0)
            effect.setEnabled(False)
        self._set_shown_by_opening(True)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Come up whenever the screen is arrived at.

        The opening belongs to starting the program: coming back here after
        closing a folder happens often enough in an afternoon's work that
        watching it again would wear (PLAN.md 3.7).
        """
        super().showEvent(event)
        self.canvas.setGeometry(self.rect())
        if self._opened:
            self.fade_in()
            return
        self._opened = True
        self.play_opening()

    def set_drop_target(self, active: bool) -> None:
        """Light the hint up while something is dragged over the window."""
        self.drop_hint.set_active(active)

    def set_opening_enabled(self, enabled: bool) -> None:
        """Let the buttons be used once something can act on them."""
        self._opening_enabled = enabled
        self._refresh_controls()

    def _set_shown_by_opening(self, shown: bool) -> None:
        """Say whether the opening has brought the controls up yet.

        A control nobody can see is not one anybody can be aiming at, and an
        opacity of zero stops none of it: before the button rises into place
        it would answer a click where it was going to be, and Tab would reach
        it just as well (PLAN.md 3.7). Held back rather than hidden, because
        hiding it would take its room out of the layout and move the screen
        for the length of the opening.
        """
        if self._shown_by_opening == shown:
            return
        self._shown_by_opening = shown
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        """Enable what can be used: the window has to have something to act
        on, and the opening has to have put it on the screen."""
        usable = self._opening_enabled and self._shown_by_opening
        self.open_folder_button.setEnabled(usable)
        for row in self._rows:
            row.setEnabled(usable)
