"""The filmstrip along the bottom of the window (PLAN.md section 3.1).

Shows every photo of the open folder in order. The current one is marked with
an outline in the accent colour, photos selected along with it with a quieter
grey one, an edited
one carries a small mark in its bottom right corner, thumbnails that have not
arrived yet are grey blocks rather than spinners (PLAN.md section 3.7), and
the position within the folder is shown on the right.

The wheel moves the strip sideways wherever the pointer is on it. A small
tab standing up from its top left edge, like a bookmark, folds it away; the
tab stays at the bottom of the window to bring it back.

Several photos can be selected with Ctrl and Shift, which is what lets one
set of settings be pasted onto all of them at once (PLAN.md 3.1, P-02). Only
one of them is the current photo: the one being edited and shown.

Right-clicking a photo offers a choosing mode for exporting several photos
(PLAN.md 3.1, X-09): every thumbnail gets a box in its corner, a click ticks
or unticks it without changing the photo being edited, and the export button
then writes the ticked ones.

A cross in the top right corner of the thumbnail under the pointer takes
that photo out of the strip (F-07); the menu takes out several chosen ones.
Only the list changes: the files stay where they are.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QEvent,
    QMimeData,
    QObject,
    QPoint,
    QPointF,
    QPropertyAnimation,
    QRect,
    QRectF,
    QSize,
    Qt,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QDrag,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QToolTip,
    QWidget,
)

from app.i18n import tr
from ui import icons, native, theme
from ui.widgets.buttons import TextButton, tooltip
from ui.widgets.region import styled_background

#: Height of a thumbnail inside the strip, leaving room for the padding.
THUMBNAIL_HEIGHT: Final[int] = 72
#: Width used for a thumbnail whose picture has not arrived yet.
PLACEHOLDER_WIDTH: Final[int] = 96
#: How far one line of the wheel moves the strip. Windows turns one notch
#: into three lines unless told otherwise, which makes it about two
#: thumbnails: enough to get through a folder, little enough to keep track.
WHEEL_LINE: Final[int] = THUMBNAIL_HEIGHT
#: What Qt reports for one notch of an ordinary wheel.
_NOTCH: Final[int] = 120


class Thumbnail(QWidget):
    """One photo in the strip: a picture, or a grey block until it arrives."""

    #: The photo, and the modifier keys held down, as Qt's flags.
    clicked = Signal(Path, int)
    #: The pointer was pressed on this photo and moved away with it.
    drag_started = Signal(Path)
    #: The photo was right-clicked, at this point on the screen.
    menu_requested = Signal(Path, QPoint)
    #: The cross in the corner was clicked (F-07).
    remove_requested = Signal(Path)

    def __init__(self, path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.path = path
        self._pixmap: QPixmap | None = None
        self._current = False
        self._selected = False
        self._edited = False
        #: Whether the box for choosing photos to export is shown, and ticked.
        self._checkable = False
        self._checked = False
        self._hovering = False
        self._pressed = False
        self._press_point = QPoint()
        #: Whether the cross that takes the photo out is offered, and
        #: whether the pointer is over it or pressing it (F-07).
        self._removable = True
        self._over_cross = False
        self._pressing_cross = False
        self.setMouseTracking(True)
        self.setFixedHeight(THUMBNAIL_HEIGHT)
        self.setFixedWidth(PLACEHOLDER_WIDTH)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(path.name)

    def set_picture(self, image: NDArray[np.uint8]) -> None:
        """Show *image*, sized to the height of the strip."""
        height, width = image.shape[:2]
        contiguous = np.ascontiguousarray(image)
        picture = QImage(
            contiguous.data, width, height, 3 * width, QImage.Format.Format_RGB888
        ).copy()
        self._pixmap = QPixmap.fromImage(picture).scaledToHeight(
            THUMBNAIL_HEIGHT, Qt.TransformationMode.SmoothTransformation
        )
        self.setFixedWidth(max(theme.SPACE_XXL, self._pixmap.width()))
        self.update()

    def set_current(self, current: bool) -> None:
        """Mark this photo as the one being edited."""
        if current != self._current:
            self._current = current
            self.update()

    def set_selected(self, selected: bool) -> None:
        """Mark this photo as one of several chosen (PLAN.md 3.1)."""
        if selected != self._selected:
            self._selected = selected
            self.update()

    def set_checkable(self, checkable: bool) -> None:
        """Show or hide the box that ticks this photo for export (X-09)."""
        if checkable != self._checkable:
            self._checkable = checkable
            self.update()

    def set_checked(self, checked: bool) -> None:
        """Tick or untick this photo for export (X-09)."""
        if checked != self._checked:
            self._checked = checked
            self.update()

    def set_removable(self, removable: bool) -> None:
        """Offer the cross that takes this photo out of the strip, or not (F-07)."""
        if removable != self._removable:
            self._removable = removable
            self._over_cross = self._over_cross and removable
            self._pressing_cross = self._pressing_cross and removable
            self.update()

    @property
    def cross_shown(self) -> bool:
        """Return whether the cross is on screen: the pointer is over the photo."""
        return self._removable and self._hovering

    def cross_rect(self) -> QRect:
        """Return where the cross sits, in the top right corner."""
        size = theme.THUMBNAIL_BADGE_SIZE
        return QRect(
            self.rect().right() - size - theme.SPACE_XS + 1,
            self.rect().top() + theme.SPACE_XS,
            size,
            size,
        )

    @property
    def checkable(self) -> bool:
        """Return whether the box for choosing is shown."""
        return self._checkable

    @property
    def checked(self) -> bool:
        """Return whether this photo is ticked for export."""
        return self._checked

    @property
    def picture(self) -> QPixmap | None:
        """Return the thumbnail's picture, once it has arrived."""
        return self._pixmap

    @property
    def selected(self) -> bool:
        """Return whether this photo is one of those chosen."""
        return self._selected

    def set_edited(self, edited: bool) -> None:
        """Mark this photo as having stored adjustments (PLAN.md 3.1)."""
        if edited != self._edited:
            self._edited = edited
            self.setToolTip(
                tr("filmstrip.edited", name=self.path.name) if edited else self.path.name
            )
            self.update()

    @property
    def edited(self) -> bool:
        """Return whether this photo is marked as edited."""
        return self._edited

    def retranslate(self) -> None:
        """Read the tooltip again after a language change (U-02)."""
        self.setToolTip(
            tr("filmstrip.edited", name=self.path.name) if self._edited else self.path.name
        )

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Draw the thumbnail, or a grey block while it is being made."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        area = self.rect()

        if self._pixmap is None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(theme.BG_RAISED))
            painter.drawRoundedRect(area, theme.RADIUS_THUMB, theme.RADIUS_THUMB)
        else:
            shape = QPainterPath()
            shape.addRoundedRect(
                QRectF(area), theme.RADIUS_THUMB, theme.RADIUS_THUMB
            )
            painter.save()
            painter.setClipPath(shape)
            painter.drawPixmap(area, self._pixmap)
            painter.restore()

        if self._edited:
            self._draw_edited_mark(painter, area)

        self._draw_outline(painter, area)

        if self._checkable:
            self._draw_check_box(painter, area)

        if self.cross_shown:
            self._draw_cross(painter)

    def _draw_cross(self, painter: QPainter) -> None:
        """Draw the cross that takes the photo out of the strip (F-07).

        Solid, like the edited mark below it, because the photo underneath
        can be any colour; brighter under the pointer and darker pressed.
        """
        if self._pressing_cross:
            patch, glyph_colour = theme.TRACK, theme.TEXT_PRIMARY
        elif self._over_cross:
            patch, glyph_colour = theme.BG_RAISED, theme.TEXT_PRIMARY
        else:
            patch, glyph_colour = theme.BG_APP, theme.TEXT_SECONDARY
        badge = self.cross_rect()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(patch))
        painter.drawRoundedRect(badge, theme.RADIUS_THUMB, theme.RADIUS_THUMB)
        glyph = theme.ICON_SIZE_SMALL
        inset = (badge.width() - glyph) // 2
        painter.drawPixmap(
            QRect(badge.x() + inset, badge.y() + inset, glyph, glyph),
            icons.render_pixmap(
                icons.ICON_X, glyph_colour, round(glyph * self.devicePixelRatioF())
            ),
        )

    def _draw_outline(self, painter: QPainter, area: QRect) -> None:
        """Outline the thumbnail according to what it is doing right now.

        Which photo is being edited is the one piece of current state in the
        strip, which is what PLAN.md 3.5 keeps the accent colour for; the
        others chosen with it are marked more quietly. A thumbnail under the
        pointer or being pressed gets a thinner line, so that the stronger
        marks are never mistaken for it (PLAN.md 3.8 asks every clickable
        element for all three states).
        """
        if self._current:
            colour, width = theme.ACCENT, theme.TAB_UNDERLINE_HEIGHT
        elif self._selected and not self._checkable:
            # While choosing, the ticks say what is chosen; a grey outline as
            # well would be a second, different choice on the same photos.
            colour, width = theme.TEXT_SECONDARY, theme.TAB_UNDERLINE_HEIGHT
        elif self._pressed:
            colour, width = theme.TEXT_PRIMARY, theme.BORDER_WIDTH
        elif self._hovering:
            colour, width = theme.TEXT_SECONDARY, theme.BORDER_WIDTH
        else:
            return

        pen = QPen(QColor(colour))
        pen.setWidth(width)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        # Half the line's width in from the edge, so that it lands on whole
        # pixels instead of straddling the boundary and coming out grey.
        half = width / 2
        painter.drawRoundedRect(
            QRectF(area).adjusted(half, half, -half, -half),
            theme.RADIUS_THUMB,
            theme.RADIUS_THUMB,
        )

    def _draw_edited_mark(self, painter: QPainter, area: QRect) -> None:
        """Draw the mark in the bottom right corner of an edited photo.

        A solid patch behind the glyph, because the photo underneath can be
        any colour; PLAN.md 3.4 rules out the shadow or translucent layer that
        would otherwise do the job.
        """
        size = theme.THUMBNAIL_BADGE_SIZE
        badge = QRect(
            area.right() - size - theme.SPACE_XS + 1,
            area.bottom() - size - theme.SPACE_XS + 1,
            size,
            size,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.BG_APP))
        painter.drawRoundedRect(badge, theme.RADIUS_THUMB, theme.RADIUS_THUMB)

        glyph = theme.ICON_SIZE_SMALL
        inset = (size - glyph) // 2
        painter.drawPixmap(
            QRect(badge.x() + inset, badge.y() + inset, glyph, glyph),
            icons.render_pixmap(
                icons.ICON_EDITED,
                theme.TEXT_PRIMARY,
                round(glyph * self.devicePixelRatioF()),
            ),
        )

    def _draw_check_box(self, painter: QPainter, area: QRect) -> None:
        """Draw the tick box in the top left corner (X-09).

        Solid, like the edited mark, because the photo underneath can be any
        colour. Ticked, it is filled the way a switch that is on is filled,
        with the tick in the dark of the window.
        """
        size = theme.CHECK_BOX_SIZE
        box = QRectF(area.left() + theme.SPACE_XS, area.top() + theme.SPACE_XS, size, size)
        half = theme.BORDER_WIDTH / 2
        # The edge is as bright as the tick, so an empty box still reads as
        # a box over a dark photo.
        painter.setPen(QPen(QColor(theme.ACCENT), theme.BORDER_WIDTH))
        painter.setBrush(QColor(theme.ACCENT if self._checked else theme.BG_APP))
        painter.drawRoundedRect(
            box.adjusted(half, half, -half, -half),
            theme.RADIUS_CHECK_BOX,
            theme.RADIUS_CHECK_BOX,
        )
        if not self._checked:
            return
        # The check of the icon set, on its 24 unit square.
        unit = size / 24
        left, top = box.left(), box.top()
        tick = QPainterPath(QPointF(left + 6 * unit, top + 12 * unit))
        tick.lineTo(QPointF(left + 10 * unit, top + 16 * unit))
        tick.lineTo(QPointF(left + 18 * unit, top + 8 * unit))
        pen = QPen(QColor(theme.BG_APP), theme.ICON_STROKE_WIDTH)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(tick)

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Ask for the filmstrip's menu where the pointer is (X-09)."""
        self.menu_requested.emit(self.path, event.globalPos())
        event.accept()

    def event(self, event: QEvent) -> bool:
        """Name the cross in its own tooltip, and the photo everywhere else."""
        if event.type() is QEvent.Type.ToolTip and self._over_cross:
            QToolTip.showText(event.globalPos(), tr("filmstrip.remove"), self)
            return True
        return super().event(event)

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Mark the thumbnail the pointer is over."""
        super().enterEvent(event)
        self._hovering = True
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Stop marking it, and let go of a press that ended elsewhere."""
        super().leaveEvent(event)
        self._hovering = False
        self._pressed = False
        self._over_cross = False
        self._pressing_cross = False
        self.update()

    def _set_over_cross(self, over: bool) -> None:
        if over != self._over_cross:
            self._over_cross = over
            # The photo's name was showing; the cross has a name of its own.
            QToolTip.hideText()
            self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Show that the thumbnail, or its cross, is being pressed."""
        if event.button() is Qt.MouseButton.LeftButton and self._over_cross:
            self._pressing_cross = True
            self.update()
            event.accept()
            return
        if event.button() is Qt.MouseButton.LeftButton:
            self._pressed = True
            self._press_point = event.position().toPoint()
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Carry the photo away once the pointer has moved far enough.

        What it is dropped on decides what happens: a half of the collage
        takes it (PLAN.md 3.13). The photos selected along with it go too.
        """
        point = event.position().toPoint()
        if not event.buttons():
            self._set_over_cross(self.cross_shown and self.cross_rect().contains(point))
        if self._pressing_cross:
            return
        if not (self._pressed and event.buttons() & Qt.MouseButton.LeftButton):
            super().mouseMoveEvent(event)
            return
        moved = event.position().toPoint() - self._press_point
        if moved.manhattanLength() < QApplication.startDragDistance():
            return
        self._pressed = False
        self.update()
        self.drag_started.emit(self.path)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Select this photo, on its own or along with the others."""
        if self._pressing_cross:
            self._pressing_cross = False
            self.update()
            if (
                event.button() is Qt.MouseButton.LeftButton
                and self.cross_shown
                and self.cross_rect().contains(event.position().toPoint())
            ):
                self.remove_requested.emit(self.path)
            return
        pressed = self._pressed
        self._pressed = False
        self.update()
        if (
            pressed
            and event.button() is Qt.MouseButton.LeftButton
            and self.rect().contains(event.position().toPoint())
        ):
            self.clicked.emit(self.path, int(event.modifiers().value))


class StripScrollArea(QScrollArea):
    """The strip's scrolling area, which the wheel moves sideways.

    Qt hands an upright turn of the wheel to the vertical scroll bar, and the
    strip has none, so on its own the wheel did nothing at all. A notch
    glides the strip over 160 ms, like the zoom (PLAN.md 3.3); a touchpad,
    which reports exact distances many times a second, moves it directly.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        bar = self.horizontalScrollBar()
        #: One animation, kept for the life of the area and restarted for
        #: each notch, so that a quick run of notches adds up instead of
        #: each one starting again from where the last had got to.
        self._glide = QPropertyAnimation(bar, b"value", self)
        self._glide.setDuration(theme.DURATION_SCROLL_MS)
        self._glide.setEasingCurve(QEasingCurve.Type.OutCubic)

    @property
    def target(self) -> int:
        """Return where the strip is on its way to, or where it is."""
        if self._glide.state() is QPropertyAnimation.State.Running:
            return int(self._glide.endValue())
        return self.horizontalScrollBar().value()

    def stop_gliding(self) -> None:
        """Stop a glide where it is, before the strip is moved some other way."""
        self._glide.stop()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt naming
        """Move the strip sideways, whichever way the wheel was turned."""
        self.scroll_by_wheel(event)

    def scroll_by_wheel(self, event: QWheelEvent) -> None:
        """Move the strip by *event*, a turn of the wheel or a touchpad swipe."""
        bar = self.horizontalScrollBar()
        pixels = event.pixelDelta()
        if not pixels.isNull():
            # A touchpad: follow the fingers exactly, with nothing to smooth.
            delta = pixels.x() if abs(pixels.x()) > abs(pixels.y()) else pixels.y()
            self.stop_gliding()
            bar.setValue(bar.value() - delta)
            event.accept()
            return

        angle = event.angleDelta()
        delta = angle.x() if abs(angle.x()) > abs(angle.y()) else angle.y()
        if delta == 0:
            event.ignore()
            return
        event.accept()
        step = delta * QApplication.wheelScrollLines() * WHEEL_LINE // _NOTCH
        target = max(bar.minimum(), min(bar.maximum(), self.target - step))
        if target == bar.value():
            self.stop_gliding()
            return
        if not native.animations_enabled():
            self.stop_gliding()
            bar.setValue(target)
            return
        self._glide.stop()
        self._glide.setStartValue(bar.value())
        self._glide.setEndValue(target)
        self._glide.start()


class FilmstripTab(QPushButton):
    """The bookmark that folds the filmstrip away and brings it back.

    It stands up from the strip's top edge at its left end, above the first
    thumbnail, the same colour as the strip and joined to it, so it reads as
    a part of the strip rather than a button laid on the photo. Folded, the
    strip is gone and the tab waits at the bottom of the window. At the left
    it keeps clear of the view bar in the middle (PLAN.md 3.3).

    It belongs to the window rather than the strip, because it has to stay
    when the strip is gone, and it follows the strip's top edge by watching
    it, through the fold's animation too.
    """

    def __init__(self, filmstrip: QWidget, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("FilmstripTab")
        self.setFixedSize(theme.FILMSTRIP_TAB_WIDTH, theme.FILMSTRIP_TAB_HEIGHT)
        self.setIconSize(QSize(theme.ICON_SIZE_SMALL, theme.ICON_SIZE_SMALL))
        # Not a keyboard stop: F6 does the same, and taking the focus here
        # would take it away from the photo.
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._filmstrip = filmstrip
        self._collapsed = False
        self.set_collapsed(False)
        filmstrip.installEventFilter(self)
        parent.installEventFilter(self)

    @property
    def collapsed(self) -> bool:
        """Return whether the arrow says the strip is folded away."""
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        """Point the arrow down while the strip is open, up while it is folded."""
        self._collapsed = collapsed
        self.setIcon(
            icons.load_icon(
                icons.ICON_CHEVRON_UP if collapsed else icons.ICON_CHEVRON_DOWN,
                size=theme.ICON_SIZE_SMALL,
            )
        )
        self.retranslate()

    def retranslate(self) -> None:
        """Read the tooltip again after a language change (U-02)."""
        self.setToolTip(
            tooltip(
                "filmstrip.expand" if self._collapsed else "filmstrip.collapse",
                "shortcut.toggle_filmstrip",
            )
        )

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Follow the strip as it moves, folds or the window changes size."""
        if event.type() in (
            QEvent.Type.Move,
            QEvent.Type.Resize,
            QEvent.Type.Show,
            QEvent.Type.Hide,
        ):
            self.follow()
        return False

    def follow(self) -> None:
        """Stand on the strip's top edge, or on the window's bottom without it.

        The tab covers the strip's 1 px top line where they meet, so the
        two read as one piece.
        """
        parent = self.parentWidget()
        strip = self._filmstrip
        floor = parent.height()
        if strip.isVisible():
            floor = min(floor, strip.y() + theme.BORDER_WIDTH)
        self.move(strip.x() + theme.SPACE_MD, floor - self.height())
        self.raise_()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt naming
        """Let the wheel move the strip from here too, while it is open."""
        if self._filmstrip.isVisible():
            self._filmstrip.wheelEvent(event)
        else:
            event.ignore()


class Filmstrip(QWidget):
    """The strip of thumbnails, shown when a folder is open."""

    #: The photo to show and edit.
    photo_selected = Signal(Path)
    #: Every photo now chosen, the current one among them (P-02).
    selection_changed = Signal(object)
    #: Choosing photos to export began or ended (X-09).
    choosing_changed = Signal(bool)
    #: The photos ticked for export, in the order the strip shows them.
    checked_changed = Signal(object)
    #: Photos to take out of the strip, in the order it shows them (F-07).
    remove_requested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        styled_background(self, "Filmstrip")
        self.setFixedHeight(theme.FILMSTRIP_HEIGHT)

        self._thumbnails: dict[Path, Thumbnail] = {}
        self._current: Path | None = None
        self._selected: set[Path] = set()
        #: Choosing photos to export (X-09): whether it is going on, whether
        #: it may (not in the collage), what is ticked, and the photo a
        #: Shift click ticks from.
        self._choosing = False
        self._choosing_allowed = True
        self._checked: set[Path] = set()
        self._anchor: Path | None = None
        #: Whether photos may be taken out; not while several are exported.
        self._removing_allowed = True
        #: Which photo of how many the position reads, so that it can be
        #: written again in another language (U-02).
        self._position_shown: tuple[int, int] | None = None

        self._empty = QLabel(tr("filmstrip.empty"), self)
        self._empty.setProperty("role", "caption")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._strip = QWidget()
        self._strip_layout = QHBoxLayout(self._strip)
        self._strip_layout.setContentsMargins(0, 0, 0, 0)
        self._strip_layout.setSpacing(theme.SPACE_SM)
        self._strip_layout.addStretch(1)

        self._scroll = StripScrollArea(self)
        self._scroll.setWidget(self._strip)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.hide()

        self._position = QLabel("", self)
        styled_background(self._position, "FilmstripPosition")
        self._position.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._position.hide()

        #: Ends choosing photos to export; shown only while choosing.
        self.done_button = TextButton("filmstrip.done", self)
        self.done_button.setToolTip(tooltip("filmstrip.done", "shortcut.crop_cancel"))
        self.done_button.clicked.connect(self.stop_choosing)
        self.done_button.hide()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(
            theme.SPACE_MD, theme.SPACE_MD, theme.SPACE_MD, theme.SPACE_MD
        )
        layout.setSpacing(theme.SPACE_MD)
        layout.addWidget(self._empty, 1)
        layout.addWidget(self._scroll, 1)
        layout.addWidget(self._position, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.done_button, 0, Qt.AlignmentFlag.AlignVCenter)

    # -- contents --------------------------------------------------------

    def set_photos(self, photos: list[Path]) -> None:
        """Rebuild the strip for a newly opened folder."""
        # What was ticked belongs to the folder that was open.
        self.stop_choosing()
        self._scroll.stop_gliding()
        for thumbnail in self._thumbnails.values():
            self._strip_layout.removeWidget(thumbnail)
            thumbnail.deleteLater()
        self._thumbnails.clear()
        self._current = None
        self._selected = set()

        for index, path in enumerate(photos):
            thumbnail = Thumbnail(path, self._strip)
            thumbnail.clicked.connect(self._on_thumbnail_clicked)
            thumbnail.drag_started.connect(self._start_drag)
            thumbnail.menu_requested.connect(self._show_menu)
            thumbnail.remove_requested.connect(
                lambda path: self.remove_requested.emit([path])
            )
            thumbnail.set_removable(self._removing_allowed)
            self._thumbnails[path] = thumbnail
            self._strip_layout.insertWidget(index, thumbnail)

        has_photos = bool(photos)
        self._empty.setVisible(not has_photos)
        self._scroll.setVisible(has_photos)
        self._position.setVisible(has_photos)

    def remove_photos(self, paths: list[Path]) -> None:
        """Take the thumbnails of *paths* out of the strip (F-07).

        The window calls this once the photos have left the list; what is
        current is said by :meth:`set_current` as usual.
        """
        for path in paths:
            thumbnail = self._thumbnails.pop(path, None)
            if thumbnail is None:
                continue
            self._strip_layout.removeWidget(thumbnail)
            thumbnail.deleteLater()
            self._selected.discard(path)
            self._checked.discard(path)
            if self._anchor == path:
                self._anchor = None
            if self._current == path:
                self._current = None
        if self._choosing:
            self._show_checks()
        has_photos = bool(self._thumbnails)
        self._empty.setVisible(not has_photos)
        self._scroll.setVisible(has_photos)
        self._position.setVisible(has_photos)

    def set_removing_allowed(self, allowed: bool) -> None:
        """Offer taking photos out of the strip, or not (F-07)."""
        self._removing_allowed = allowed
        for thumbnail in self._thumbnails.values():
            thumbnail.set_removable(allowed)

    def set_thumbnail(self, path: Path, image: NDArray[np.uint8]) -> None:
        """Show the picture of one photo once it has been made."""
        thumbnail = self._thumbnails.get(path)
        if thumbnail is not None:
            thumbnail.set_picture(image)

    def set_edited_photos(self, edited: set[Path]) -> None:
        """Mark exactly the photos in *edited*, for a newly opened folder."""
        for path, thumbnail in self._thumbnails.items():
            thumbnail.set_edited(path in edited)

    def set_edited(self, path: Path, edited: bool) -> None:
        """Mark or unmark one photo after its parameter file was written."""
        thumbnail = self._thumbnails.get(path)
        if thumbnail is not None:
            thumbnail.set_edited(edited)

    # -- choosing photos (PLAN.md 3.1) -----------------------------------

    def selected_photos(self) -> list[Path]:
        """Return every chosen photo, in the order the strip shows them."""
        return [path for path in self._thumbnails if path in self._selected]

    def thumbnail(self, path: Path) -> Thumbnail | None:
        """Return the thumbnail of *path*, if it is in the strip."""
        return self._thumbnails.get(path)

    def _on_thumbnail_clicked(self, path: Path, modifiers: int) -> None:
        """Choose photos the way every file list does."""
        control = bool(modifiers & int(Qt.KeyboardModifier.ControlModifier.value))
        shift = bool(modifiers & int(Qt.KeyboardModifier.ShiftModifier.value))
        paths = list(self._thumbnails)

        if self._choosing:
            self._tick(path, shift)
            return

        if control and self._current is not None:
            # Add to or take away from the choice, leaving the current photo.
            if path in self._selected and path != self._current:
                self._selected.discard(path)
            else:
                self._selected.add(path)
            self._show_selection()
            self.selection_changed.emit(self.selected_photos())
            return

        if shift and self._current is not None and self._current in paths:
            first = paths.index(self._current)
            last = paths.index(path)
            low, high = sorted((first, last))
            self._selected = set(paths[low : high + 1])
            self._show_selection()
            self.selection_changed.emit(self.selected_photos())
            return

        self.photo_selected.emit(path)

    # -- choosing photos to export (PLAN.md 3.1, X-09) -------------------

    @property
    def choosing(self) -> bool:
        """Return whether photos are being ticked for export."""
        return self._choosing

    def checked_photos(self) -> list[Path]:
        """Return the photos ticked for export, in the order the strip shows them."""
        return [path for path in self._thumbnails if path in self._checked]

    def set_choosing_allowed(self, allowed: bool) -> None:
        """Offer choosing or not; the collage exports itself, not ticked photos."""
        self._choosing_allowed = allowed
        if not allowed:
            self.stop_choosing()

    def start_choosing(self, path: Path | None = None) -> None:
        """Begin ticking photos for export, with *path* ticked to begin with.

        Photos already chosen together with Ctrl or Shift are ticked too, so
        a choice made the usual way carries straight over. A choice of one,
        the current photo, is only where the user happens to be, and is not.
        """
        if self._choosing or not self._choosing_allowed or not self._thumbnails:
            return
        chosen = self.selected_photos()
        checked = set(chosen) if len(chosen) > 1 else set()
        if path is not None and path in self._thumbnails:
            checked.add(path)
        self._choosing = True
        self._checked = checked
        self._anchor = path
        for thumbnail in self._thumbnails.values():
            thumbnail.set_checkable(True)
        self._show_checks()
        self.choosing_changed.emit(True)

    def stop_choosing(self) -> None:
        """Stop ticking photos; what was ticked is let go."""
        if not self._choosing:
            return
        self._choosing = False
        self._checked = set()
        self._anchor = None
        for thumbnail in self._thumbnails.values():
            thumbnail.set_checkable(False)
        self._show_checks()
        self.choosing_changed.emit(False)

    def check_all(self) -> None:
        """Tick every photo of the folder (Ctrl+A while choosing)."""
        if self._choosing:
            self._checked = set(self._thumbnails)
            self._show_checks()

    def check_none(self) -> None:
        """Untick every photo (Ctrl+D while choosing)."""
        if self._choosing:
            self._checked = set()
            self._anchor = None
            self._show_checks()

    def _tick(self, path: Path, shift: bool) -> None:
        """Tick or untick *path*; with Shift, tick everything from the last one."""
        paths = list(self._thumbnails)
        if shift and self._anchor in self._thumbnails:
            first, last = sorted((paths.index(self._anchor), paths.index(path)))
            self._checked |= set(paths[first : last + 1])
        elif path in self._checked:
            self._checked.discard(path)
        else:
            self._checked.add(path)
        self._anchor = path
        self._show_checks()

    def _show_checks(self) -> None:
        """Mark the ticked photos, and say how many there are."""
        for path, thumbnail in self._thumbnails.items():
            thumbnail.set_checked(path in self._checked)
        self.done_button.setVisible(self._choosing)
        self._show_position()
        self.checked_changed.emit(self.checked_photos())

    def _show_menu(self, path: Path, where: QPoint) -> None:
        """Offer choosing photos to export or an end to it (X-09), and taking photos out (F-07)."""
        menu = QMenu(self)
        if self._choosing:
            action = menu.addAction(tr("filmstrip.stop_choosing"))
            action.triggered.connect(self.stop_choosing)
        else:
            action = menu.addAction(tr("filmstrip.choose_for_export"))
            action.setEnabled(self._choosing_allowed)
            action.triggered.connect(lambda _checked=False, chosen=path: self.start_choosing(chosen))
        leaving = self.photos_to_remove(path)
        remove = menu.addAction(
            tr("filmstrip.remove_several", count=len(leaving))
            if len(leaving) > 1
            else tr("filmstrip.remove")
        )
        remove.setEnabled(self._removing_allowed)
        remove.triggered.connect(lambda _checked=False: self.remove_requested.emit(leaving))
        menu.exec(where)
        menu.deleteLater()

    def photos_to_remove(self, path: Path) -> list[Path]:
        """Return what the menu on *path* takes out: the choice it belongs to, or just it.

        While choosing photos to export the grey outlines are not drawn, so
        the menu then speaks only for the photo it was opened on.
        """
        chosen = self.selected_photos()
        if not self._choosing and path in chosen and len(chosen) > 1:
            return chosen
        return [path]

    def _start_drag(self, path: Path) -> None:
        """Carry *path* away, with the photos chosen along with it.

        The drag holds the files as ordinary file links, which is what a
        half of the collage takes (PLAN.md 3.13); the window itself ignores
        a drag that starts inside it, so dropping one back on the photo does
        not open anything.
        """
        chosen = self.selected_photos()
        carried = chosen if path in chosen and len(chosen) > 1 else [path]
        data = QMimeData()
        data.setUrls([QUrl.fromLocalFile(str(photo)) for photo in carried])
        drag = QDrag(self)
        drag.setMimeData(data)
        picture = self._thumbnails[path].picture
        if picture is not None:
            drag.setPixmap(picture)
        drag.exec(Qt.DropAction.CopyAction)

    def _show_selection(self) -> None:
        """Mark the chosen photos."""
        for path, thumbnail in self._thumbnails.items():
            thumbnail.set_selected(path in self._selected)

    def set_current(self, index: int, total: int) -> None:
        """Mark the photo at *index* as current and show the position."""
        paths = list(self._thumbnails)
        if not 0 <= index < len(paths):
            return
        path = paths[index]
        if self._current is not None and self._current in self._thumbnails:
            self._thumbnails[self._current].set_current(False)
        self._thumbnails[path].set_current(True)
        self._current = path
        if path not in self._selected:
            # Moving to a photo that was not part of the choice starts a new
            # one, which is what every file list does.
            self._selected = {path}
            self._show_selection()
        self.selection_changed.emit(self.selected_photos())
        self._position_shown = (index, total)
        self._show_position()
        self._scroll.stop_gliding()
        self._scroll.ensureWidgetVisible(self._thumbnails[path], theme.SPACE_XXL, 0)

    def _show_position(self) -> None:
        """Say where the current photo is, or while choosing, how many are ticked."""
        if self._choosing:
            self._position.setText(tr("filmstrip.checked", count=len(self._checked)))
        elif self._position_shown is not None:
            index, total = self._position_shown
            self._position.setText(tr("filmstrip.position", index=index + 1, total=total))

    def retranslate(self) -> None:
        """Read the empty state and the position again (U-02)."""
        self._empty.setText(tr("filmstrip.empty"))
        self.done_button.setToolTip(tooltip("filmstrip.done", "shortcut.crop_cancel"))
        self._show_position()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt naming
        """Scroll the strip from anywhere on it, the position and edges too."""
        if self._scroll.isVisible():
            self._scroll.scroll_by_wheel(event)
        else:
            event.ignore()

    # -- collapsing ------------------------------------------------------

    def _get_panel_height(self) -> int:
        return self.maximumHeight()

    def _set_panel_height(self, height: int) -> None:
        self.setFixedHeight(max(0, height))

    #: Animated by the window when the filmstrip is collapsed (PLAN.md 3.6).
    panel_height = Property(int, _get_panel_height, _set_panel_height)
