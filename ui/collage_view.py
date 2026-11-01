"""The collage canvas in the middle of the window (PLAN.md 3.13).

Shows the whole canvas as large as the area allows, each photo where
:func:`core.collage.placed_rect` puts it - the same function the export
renders with, so what is seen is what is written. A half with no photo is a
patch that asks for one: click it to pick a file, or drop one on it, from
the filmstrip or from Explorer.

Over a photo, dragging moves it, the wheel zooms it about the pointer (Shift
for a finer step) and a double click puts it back to its fit. Linked, both
photos move together. The seam between the halves can be dragged.

While the second photo is being aligned, it is drawn half see-through over
the first across the whole canvas, and dragging, the wheel and the arrow
keys move and scale it alone.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QWidget

from app.collage_controller import CollageController, CollagePhoto
from app.i18n import tr
from core import collage, decoder
from core.collage import UNALIGNED, Placement
from ui import theme

#: How much one notch of the wheel zooms a photo, and with Shift held.
_WHEEL_ZOOM_PER_NOTCH: Final[float] = 1.05
_WHEEL_ZOOM_FINE: Final[float] = 1.01
#: How much one notch scales the photo being aligned, and with Shift held:
#: two shots of one scene differ by a few percent at most.
_ALIGN_SCALE_PER_NOTCH: Final[float] = 1.005
_ALIGN_SCALE_FINE: Final[float] = 1.001
#: What one notch of a wheel reports, in eighths of a degree.
_WHEEL_NOTCH: Final[float] = 120.0
#: How far an arrow key moves the photo being aligned, in canvas pixels,
#: and with Shift held.
_NUDGE: Final[float] = 1.0
_NUDGE_LARGE: Final[float] = 10.0

_DRAG_PAN: Final[str] = "pan"
_DRAG_SEAM: Final[str] = "seam"
_DRAG_ALIGN: Final[str] = "align"

_ARROWS: Final[dict[int, tuple[float, float]]] = {
    int(Qt.Key.Key_Left): (-1.0, 0.0),
    int(Qt.Key.Key_Right): (1.0, 0.0),
    int(Qt.Key.Key_Up): (0.0, -1.0),
    int(Qt.Key.Key_Down): (0.0, 1.0),
}


class CollageCanvas(QWidget):
    """The canvas, its two halves and the gestures that place the photos."""

    #: The user asked to pick a photo for a half, by its slot.
    choose_requested = Signal(int)

    def __init__(self, collage_controller: CollageController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._collage = collage_controller
        self._pixmaps: dict[int, QPixmap] = {}
        self._hover: int | None = None
        self._hover_seam = False
        self._drop: int | None = None
        self._pressed: int | None = None
        self._drag: str | None = None
        self._press_point = QPointF()
        self._press_placement = Placement()
        self._press_alignment = UNALIGNED
        self._dragged = False
        self._aligning = False

        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(theme.PANEL_WIDTH, theme.PANEL_WIDTH)

        collage_controller.photo_changed.connect(self._on_photo_changed)
        collage_controller.layout_changed.connect(lambda _layout: self.update())
        collage_controller.placement_changed.connect(self.update)

    # -- aligning --------------------------------------------------------

    @property
    def aligning(self) -> bool:
        """Return whether the second photo is being aligned over the first."""
        return self._aligning and self._can_align()

    def set_aligning(self, aligning: bool) -> None:
        """Draw the second photo over the first, and move it alone, or stop."""
        self._aligning = aligning
        self._update_cursor()
        self.update()
        if aligning:
            self.setFocus()

    def _can_align(self) -> bool:
        return self._collage.layout.linked and all(
            self._collage.photo(slot) is not None for slot in range(collage.SLOTS)
        )

    # -- where things are ------------------------------------------------

    def _scale_and_origin(self) -> tuple[float, QPointF]:
        """Return how large a canvas pixel is on screen, and where it starts."""
        width, height = self._collage.layout.canvas_size
        room_w = max(1.0, self.width() - 2.0 * theme.CANVAS_MARGIN)
        room_h = max(1.0, self.height() - 2.0 * theme.CANVAS_MARGIN)
        scale = min(room_w / width, room_h / height)
        origin = QPointF((self.width() - width * scale) / 2.0, (self.height() - height * scale) / 2.0)
        return scale, origin

    def canvas_rect(self) -> QRectF:
        """Return where the canvas is drawn, in widget pixels."""
        scale, origin = self._scale_and_origin()
        width, height = self._collage.layout.canvas_size
        return QRectF(origin.x(), origin.y(), width * scale, height * scale)

    def cell_rect(self, slot: int) -> QRectF:
        """Return where one half is drawn, in widget pixels."""
        cell = collage.cells(self._collage.layout)[slot]
        return self._to_widget((cell.x, cell.y, cell.width, cell.height))

    def _to_widget(self, rect: tuple[float, float, float, float]) -> QRectF:
        scale, origin = self._scale_and_origin()
        left, top, width, height = rect
        return QRectF(
            origin.x() + left * scale, origin.y() + top * scale, width * scale, height * scale
        )

    def _to_canvas(self, point: QPointF) -> tuple[float, float]:
        scale, origin = self._scale_and_origin()
        return (point.x() - origin.x()) / scale, (point.y() - origin.y()) / scale

    def slot_at(self, point: QPointF) -> int | None:
        """Return the half under *point*, or ``None`` over the gap or outside."""
        for slot in range(collage.SLOTS):
            if self.cell_rect(slot).contains(point):
                return slot
        return None

    def seam_position(self) -> float:
        """Return where the seam is drawn, in widget pixels along the split."""
        layout = self._collage.layout
        first = collage.cells(layout)[0]
        scale, origin = self._scale_and_origin()
        if layout.split == collage.SPLIT_SIDE_BY_SIDE:
            return origin.x() + (first.width + layout.gap / 2.0) * scale
        return origin.y() + (first.height + layout.gap / 2.0) * scale

    def _near_seam(self, point: QPointF) -> bool:
        if not self.canvas_rect().contains(point) or self.aligning:
            return False
        along = point.x() if self._collage.layout.split == collage.SPLIT_SIDE_BY_SIDE else point.y()
        reach = theme.COLLAGE_SEAM_GRAB + self._collage.layout.gap * self._scale_and_origin()[0] / 2
        return abs(along - self.seam_position()) <= reach

    # -- being told things -----------------------------------------------

    def _on_photo_changed(self, slot: int, photo: CollagePhoto | None) -> None:
        if photo is None:
            self._pixmaps.pop(slot, None)
        else:
            self._pixmaps[slot] = _to_pixmap(photo.image)
        self._update_cursor()
        self.update()

    # -- drawing ---------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Draw the canvas, its background and each half."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.BG_CANVAS))
        layout = self._collage.layout
        canvas = self.canvas_rect()
        painter.fillRect(canvas, QColor(*collage.BACKGROUND_SRGB8[layout.background]))
        # Smooth scaling for the photos, but no antialiasing on their edges:
        # at a fractional display scale the edge pixel would be half photo
        # and half background, a faint light line down the side of each one.
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        if self.aligning:
            self._draw_aligning(painter, canvas)
            return
        for slot in range(collage.SLOTS):
            area = self.cell_rect(slot)
            pixmap = self._pixmaps.get(slot)
            rect = self._collage.rect(slot)
            if pixmap is None or rect is None:
                self._draw_empty(painter, slot, area)
                continue
            painter.save()
            painter.setClipRect(area)
            painter.drawPixmap(self._to_widget(rect), pixmap, QRectF(pixmap.rect()))
            painter.restore()
            if slot in (self._hover, self._drop) and self.isEnabled() and not self._hover_seam:
                self._draw_outline(painter, area, dashed=slot == self._drop)
        if self.isEnabled() and (self._hover_seam or self._drag == _DRAG_SEAM):
            self._draw_seam(painter, canvas)

    def _draw_aligning(self, painter: QPainter, canvas: QRectF) -> None:
        """Draw both photos over the whole canvas, the second see-through."""
        painter.save()
        painter.setClipRect(canvas)
        for slot in range(collage.SLOTS):
            rect = self._collage.rect(slot)
            pixmap = self._pixmaps.get(slot)
            if rect is None or pixmap is None:
                continue
            painter.setOpacity(theme.COLLAGE_ALIGN_OPACITY if slot == 1 else 1.0)
            painter.drawPixmap(self._to_widget(rect), pixmap, QRectF(pixmap.rect()))
        painter.restore()

    def _draw_seam(self, painter: QPainter, canvas: QRectF) -> None:
        """Mark the seam the pointer is over, or the one being dragged."""
        pen = QPen(QColor(theme.ACCENT))
        pen.setWidth(theme.BORDER_WIDTH)
        painter.setPen(pen)
        at = self.seam_position()
        if self._collage.layout.split == collage.SPLIT_SIDE_BY_SIDE:
            painter.drawLine(QPointF(at, canvas.top()), QPointF(at, canvas.bottom()))
        else:
            painter.drawLine(QPointF(canvas.left(), at), QPointF(canvas.right(), at))

    def _draw_empty(self, painter: QPainter, slot: int, area: QRectF) -> None:
        """Draw a half that has no photo yet, in its hover and pressed states."""
        loading = self._collage.paths[slot] is not None
        if not self.isEnabled() or loading:
            fill = theme.BG_RAISED
        elif slot == self._pressed:
            fill = theme.TRACK_HOVER
        elif slot in (self._hover, self._drop):
            fill = theme.TRACK
        else:
            fill = theme.BG_RAISED
        painter.fillRect(area, QColor(fill))
        if slot == self._drop:
            self._draw_outline(painter, area, dashed=True)

        title = tr("collage.loading") if loading else tr("collage.empty_title")
        hint = "" if loading else tr("collage.empty_hint")
        font = QFont(self.font())
        font.setPixelSize(theme.FONT_SIZE_BODY)
        painter.setFont(font)
        painter.setPen(
            QColor(theme.TEXT_PRIMARY if self.isEnabled() and not loading else theme.TEXT_TERTIARY)
        )
        line = theme.FONT_SIZE_BODY + theme.SPACE_SM
        title_box = QRectF(area.left(), area.center().y() - line, area.width(), line)
        painter.drawText(title_box, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom, title)
        if hint:
            font.setPixelSize(theme.FONT_SIZE_CAPTION)
            painter.setFont(font)
            painter.setPen(QColor(theme.TEXT_TERTIARY))
            hint_box = QRectF(
                area.left() + theme.SPACE_SM,
                area.center().y() + theme.SPACE_XS,
                max(0.0, area.width() - 2 * theme.SPACE_SM),
                line * 2,
            )
            painter.drawText(
                hint_box,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
                hint,
            )

    def _draw_outline(self, painter: QPainter, area: QRectF, dashed: bool) -> None:
        """Mark the half the pointer is over, or the one a drop would land in."""
        pen = QPen(QColor(theme.ACCENT if dashed else theme.TEXT_TERTIARY))
        pen.setWidth(theme.BORDER_WIDTH)
        if dashed:
            pen.setDashPattern(list(theme.DROP_ZONE_DASH))
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        inset = theme.BORDER_WIDTH / 2.0
        painter.drawRect(area.adjusted(inset, inset, -inset, -inset))

    # -- gestures --------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Start dragging a photo, the seam, or the photo being aligned."""
        if event.button() is not Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        point = event.position()
        self.setFocus()
        self._dragged = False
        self._press_point = point
        self._pressed = self.slot_at(point)
        if self.aligning and self.canvas_rect().contains(point):
            self._drag = _DRAG_ALIGN
            self._press_alignment = self._collage.alignment(1)
        elif self._near_seam(point):
            self._drag = _DRAG_SEAM
            self._pressed = None
        elif self._pressed is not None:
            self._drag = _DRAG_PAN
            self._press_placement = self._collage.placement(self._pressed)
        else:
            self._drag = None
        self._update_cursor()
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Carry on the drag that was started, or follow the pointer."""
        point = event.position()
        if self._drag is not None and event.buttons() & Qt.MouseButton.LeftButton:
            moved = point - self._press_point
            if moved.manhattanLength() >= QApplication.startDragDistance():
                self._dragged = True
            if self._dragged:
                self._follow_drag(point, moved)
            return
        hover = self.slot_at(point)
        near_seam = self._near_seam(point)
        if hover != self._hover or near_seam != self._hover_seam:
            self._hover = hover
            self._hover_seam = near_seam
            self.update()
        self._update_cursor()

    def _follow_drag(self, point: QPointF, moved: QPointF) -> None:
        scale, _origin = self._scale_and_origin()
        delta = (moved.x() / scale, moved.y() / scale)
        layout = self._collage.layout
        if self._drag == _DRAG_SEAM:
            canvas_x, canvas_y = self._to_canvas(point)
            width, height = layout.canvas_size
            along = canvas_x / width if layout.split == collage.SPLIT_SIDE_BY_SIDE else canvas_y / height
            self._collage.set_layout(replace(layout, seam=along))
            return
        if self._drag == _DRAG_ALIGN:
            rect = self._collage.rect(1, aligned=False)
            if rect is not None:
                self._collage.set_alignment(collage.nudged(self._press_alignment, delta, rect))
            return
        slot = self._pressed
        size = None if slot is None else self._collage.photo_size(slot)
        if slot is None or size is None:
            return
        self._collage.set_placement(
            slot,
            collage.panned(
                self._press_placement,
                delta,
                size,
                collage.frame(layout, slot),
                layout.fit_used,
            ),
        )

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """A click on an empty half asks for a photo to put there."""
        if event.button() is not Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        slot = self._pressed
        clicked_empty = (
            self._drag != _DRAG_ALIGN
            and slot is not None
            and not self._dragged
            and self.slot_at(event.position()) == slot
            and self._collage.paths[slot] is None
        )
        self._pressed = None
        self._drag = None
        self._dragged = False
        if clicked_empty:
            self.choose_requested.emit(slot)
        self._update_cursor()
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Put back whatever is under the pointer: alignment, seam or fit."""
        point = event.position()
        if self.aligning:
            self._collage.set_alignment(UNALIGNED)
            return
        if self._near_seam(point):
            layout = self._collage.layout
            self._collage.set_layout(replace(layout, seam=collage.SEAM_DEFAULT))
            return
        slot = self.slot_at(point)
        if slot is not None and self._collage.photo_size(slot) is not None:
            self._collage.set_placement(slot, Placement())
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Zoom the photo under the pointer about the pointer (PLAN.md 3.3).

        Shift gives the fine step. Windows may hand a wheel turned with
        Shift held over as a sideways one, so either direction counts.
        """
        delta = event.angleDelta()
        notches = (delta.y() or delta.x()) / _WHEEL_NOTCH
        if notches == 0:
            event.ignore()
            return
        fine = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        anchor = self._to_canvas(event.position())
        if self.aligning:
            rect = self._collage.rect(1, aligned=False)
            if rect is None:
                event.ignore()
                return
            step = _ALIGN_SCALE_FINE if fine else _ALIGN_SCALE_PER_NOTCH
            alignment = self._collage.alignment(1)
            self._collage.set_alignment(
                collage.rescaled(alignment, alignment.scale * step**notches, anchor, rect)
            )
            event.accept()
            return
        slot = self.slot_at(event.position())
        size = None if slot is None else self._collage.photo_size(slot)
        if slot is None or size is None:
            event.ignore()
            return
        layout = self._collage.layout
        placement = self._collage.placement(slot)
        step = _WHEEL_ZOOM_FINE if fine else _WHEEL_ZOOM_PER_NOTCH
        self._collage.set_placement(
            slot,
            collage.zoomed(
                placement,
                placement.zoom * step**notches,
                anchor,
                size,
                collage.frame(layout, slot),
                layout.fit_used,
            ),
        )
        event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """While aligning, the arrow keys move the second photo a pixel at a time."""
        direction = _ARROWS.get(int(event.key()))
        rect = self._collage.rect(1, aligned=False) if self.aligning else None
        if direction is None or rect is None:
            super().keyPressEvent(event)
            return
        step = _NUDGE_LARGE if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else _NUDGE
        self._collage.set_alignment(
            collage.nudged(
                self._collage.alignment(1), (direction[0] * step, direction[1] * step), rect
            )
        )
        event.accept()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Stop marking the half or the seam the pointer was over."""
        super().leaveEvent(event)
        self._hover = None
        self._hover_seam = False
        self.update()

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Draw again when the canvas is enabled or disabled, for its states."""
        super().changeEvent(event)
        self.update()

    def _update_cursor(self) -> None:
        if not self.isEnabled():
            self.unsetCursor()
            return
        if self.aligning:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
            return
        if self._drag == _DRAG_SEAM or (self._drag is None and self._hover_seam):
            vertical = self._collage.layout.split == collage.SPLIT_SIDE_BY_SIDE
            self.setCursor(Qt.CursorShape.SplitHCursor if vertical else Qt.CursorShape.SplitVCursor)
            return
        if self._drag == _DRAG_PAN and self._dragged:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        slot = self._hover if self._pressed is None else self._pressed
        if slot is None:
            self.unsetCursor()
        elif self._collage.photo_size(slot) is not None:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif self._collage.paths[slot] is None:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.unsetCursor()

    # -- dropping photos -------------------------------------------------

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Take a drag that carries photos, and mark where it would land."""
        if not dropped_photos(event):
            event.ignore()
            return
        event.acceptProposedAction()
        self._follow_drop(event.position())

    def dragMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Follow the drag from one half to the other."""
        if self._follow_drop(event.position()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Stop marking a half."""
        super().dragLeaveEvent(event)
        self._drop = None
        self.update()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Put the first photo in the half it was dropped on, a second in the other."""
        slot = self.slot_at(event.position())
        photos = dropped_photos(event)
        self._drop = None
        self.update()
        if slot is None or not photos:
            event.ignore()
            return
        event.acceptProposedAction()
        self._collage.load(slot, photos[0])
        other = 1 - slot
        if len(photos) > 1 and self._collage.paths[other] is None:
            self._collage.load(other, photos[1])

    def _follow_drop(self, point: QPointF) -> bool:
        slot = self.slot_at(point)
        if slot != self._drop:
            self._drop = slot
            self.update()
        return slot is not None


def dropped_photos(event) -> list[Path]:
    """Return the photos a drag carries, as local paths; folders are not photos."""
    data = event.mimeData()
    if not data.hasUrls():
        return []
    photos: list[Path] = []
    for url in data.urls():
        if url.isLocalFile():
            path = Path(url.toLocalFile())
            if decoder.is_supported(path):
                photos.append(path)
    return photos


def _to_pixmap(image: NDArray[np.uint8]) -> QPixmap:
    """Convert an 8 bit RGB array into something Qt can draw."""
    contiguous = np.ascontiguousarray(image)
    height, width, _channels = contiguous.shape
    picture = QImage(
        contiguous.data, width, height, 3 * width, QImage.Format.Format_RGB888
    ).copy()
    return QPixmap.fromImage(picture)
