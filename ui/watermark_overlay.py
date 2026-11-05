"""The watermark drawn over the photo while it is being placed (PLAN.md 3.14).

Like the crop box and the spot brush this is not a widget: the canvas hands
it its painter and its mouse events, in the canvas's own coordinates, with
the rectangle the photo is drawn in. Where the watermark goes is worked out
by :mod:`core.watermark` on the photo's full size frame - the same layout the
export uses - and only scaled onto the screen here, so what is placed is
what is written.

The picture of the watermark is drawn once by the controller and only moved
while it is dragged, so a drag never waits for anything to be drawn again.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen

from core import watermark
from core.watermark import WatermarkSpec
from ui import theme


class WatermarkOverlay(QObject):
    """The watermark on the photo, its outline and the guides it snaps to."""

    #: The watermark was dragged, or put back: where it is now, and whether
    #: the hand has come off (only then is it worth remembering).
    moved = Signal(object, bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._active = False
        self._spec = WatermarkSpec()
        self._image: QImage | None = None
        #: Pixels of the picture per pixel of the full size frame.
        self._scale = 1.0
        self._frame: tuple[int, int] = (0, 0)
        self._hovered = False
        #: How far the press was from the watermark's top left, in frame
        #: pixels, while it is held; ``None`` when it is not.
        self._grab: QPointF | None = None

    # -- what it shows ---------------------------------------------------

    @property
    def active(self) -> bool:
        """Return whether the watermark is being placed."""
        return self._active

    @property
    def dragging(self) -> bool:
        """Return whether the watermark is held."""
        return self._grab is not None

    @property
    def spec(self) -> WatermarkSpec:
        """Return the watermark as it is placed now."""
        return self._spec

    def set_active(self, active: bool) -> None:
        """Start or stop showing the watermark over the photo."""
        self._active = active
        self._grab = None
        self._hovered = False

    def set_frame(self, frame: tuple[int, int]) -> None:
        """Take the size of the photo on screen, at full resolution."""
        self._frame = frame

    def set_spec(self, spec: WatermarkSpec) -> None:
        """Place the watermark as *spec* says, unless it is held."""
        if self._grab is None:
            self._spec = spec.sane()

    def set_picture(
        self, spec: WatermarkSpec, rgba: NDArray[np.uint8] | None, scale: float
    ) -> None:
        """Take the drawn watermark: 8 bit RGBA pixels drawn at *scale* of the frame."""
        self.set_spec(spec)
        if rgba is None:
            self._image = None
            return
        height, width = rgba.shape[:2]
        image = QImage(rgba.data, width, height, rgba.strides[0], QImage.Format.Format_RGBA8888)
        # The pixels belong to the array; the copy keeps them alive.
        self._image = image.copy()
        self._scale = max(scale, 1e-6)

    def _box(self) -> tuple[float, float]:
        """Return the watermark's size in frame pixels."""
        if self._image is None:
            return 0.0, 0.0
        return self._image.width() / self._scale, self._image.height() / self._scale

    def _ratio(self, photo: QRectF) -> float:
        """Return screen pixels per frame pixel."""
        return photo.width() / self._frame[0] if self._frame[0] > 0 else 0.0

    def screen_rect(self, photo: QRectF) -> QRectF:
        """Return where the watermark is drawn on screen, or an empty rectangle."""
        if not self._active or self._image is None or self._frame[0] <= 0 or photo.isEmpty():
            return QRectF()
        box = self._box()
        x, y = watermark.layout(self._spec, self._frame, box)
        ratio = self._ratio(photo)
        return QRectF(
            photo.left() + x * ratio, photo.top() + y * ratio, box[0] * ratio, box[1] * ratio
        )

    def over(self, point: QPointF, photo: QRectF) -> bool:
        """Return whether *point* is on the watermark."""
        return self.screen_rect(photo).contains(point)

    # -- drawing ---------------------------------------------------------

    def paint(self, painter: QPainter, photo: QRectF) -> None:
        """Draw the watermark, and while it is handled its outline and guides."""
        rect = self.screen_rect(photo)
        if rect.isEmpty() or self._image is None:
            return
        painter.save()
        painter.setClipRect(photo)
        if self.dragging:
            self._paint_guides(painter, photo, rect)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(rect, self._image)
        if self._hovered or self.dragging:
            gap = theme.WATERMARK_OUTLINE_GAP
            painter.setPen(
                QPen(QColor(theme.TEXT_TERTIARY), theme.WATERMARK_OUTLINE_WIDTH, Qt.PenStyle.DashLine)
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect.adjusted(-gap, -gap, gap, gap))
        painter.restore()

    def _paint_guides(self, painter: QPainter, photo: QRectF, rect: QRectF) -> None:
        """Draw the lines the watermark has snapped to, stopping short of it."""
        across, down = watermark.guides(self._spec, self._frame, self._box())
        ratio = self._ratio(photo)
        gap = theme.WATERMARK_GUIDE_GAP
        painter.setPen(QPen(QColor(theme.ACCENT), theme.WATERMARK_GUIDE_WIDTH))
        if across is not None:
            x = photo.left() + across * ratio
            painter.drawLine(QPointF(x, photo.top()), QPointF(x, rect.top() - gap))
            painter.drawLine(QPointF(x, rect.bottom() + gap), QPointF(x, photo.bottom()))
        if down is not None:
            y = photo.top() + down * ratio
            painter.drawLine(QPointF(photo.left(), y), QPointF(rect.left() - gap, y))
            painter.drawLine(QPointF(rect.right() + gap, y), QPointF(photo.right(), y))

    # -- the mouse -------------------------------------------------------

    def press(self, point: QPointF, photo: QRectF) -> bool:
        """Take hold of the watermark if *point* is on it."""
        if not self.over(point, photo):
            return False
        x, y = watermark.layout(self._spec, self._frame, self._box())
        self._grab = self._to_frame(point, photo) - QPointF(x, y)
        return True

    def move(self, point: QPointF, photo: QRectF, free: bool = False) -> bool:
        """Drag the watermark, or follow the pointer; return whether to redraw.

        With *free* (Alt held down) it is not pulled onto the lines (W-07).
        """
        if self._grab is None:
            hovered = self.over(point, photo)
            changed = hovered != self._hovered
            self._hovered = hovered
            return changed
        ratio = self._ratio(photo)
        corner = self._to_frame(point, photo) - self._grab
        snap = 0.0 if free or ratio <= 0 else watermark.SNAP_DISTANCE / ratio
        drop = watermark.from_drop(
            self._spec, self._frame, self._box(), (corner.x(), corner.y()), snap
        )
        self._spec = drop.spec
        self.moved.emit(self._spec, False)
        return True

    def release(self, point: QPointF, photo: QRectF) -> bool:
        """Let go of the watermark where it is."""
        if self._grab is None:
            return False
        self._grab = None
        self._hovered = self.over(point, photo)
        self.moved.emit(self._spec, True)
        return True

    def double_click(self, point: QPointF, photo: QRectF) -> bool:
        """Put the watermark back in its usual place, if it was double clicked."""
        if not self.over(point, photo):
            return False
        self._spec = self._spec.with_anchor(watermark.ANCHOR_DEFAULT)
        self.moved.emit(self._spec, True)
        return True

    def leave(self) -> None:
        """The pointer has left the photo."""
        self._hovered = False

    def _to_frame(self, point: QPointF, photo: QRectF) -> QPointF:
        """Return *point* in pixels of the full size frame."""
        ratio = self._ratio(photo)
        if ratio <= 0:
            return QPointF()
        return (point - photo.topLeft()) / ratio
