"""The photo flying into the export button (PLAN.md 3.6, 3.11).

The moment an export is asked for, the photo on screen shrinks to a small
framed thumbnail and flies along a curve into the export button, so the eye
is led to where the progress will appear. It shrinks as it travels, a
little ahead of the travel, so the photo can be followed all the way into
the button rather than collapsing where it stood, rising a little above the
photo on the way and fading over the last part of the flight. Only one picture flies, also for a collage that
writes several files.

It is drawn over the whole window but never takes the pointer, so nothing
waits for it. With Windows animation effects off there is no flight.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, QVariantAnimation
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from ui import native, theme


def _ease_in_out(k: float) -> float:
    """Cubic ease in and out, for the path along the curve."""
    if k < 0.5:
        return 4.0 * k**3
    return 1.0 - (-2.0 * k + 2.0) ** 3 / 2.0


class FlyingPhoto(QWidget):
    """A thumbnail of the photo on its way to the export button."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._picture = QPixmap()
        self._start = QRectF()
        self._target = QPointF()
        self._k = 0.0
        self._drawn = QRectF()
        self._flight = QVariantAnimation(self)
        self._flight.setDuration(theme.EXPORT_FLY_MS)
        self._flight.setStartValue(0.0)
        self._flight.setEndValue(1.0)
        self._flight.valueChanged.connect(self._move_to)
        self._flight.finished.connect(self.stop)
        self.hide()

    @property
    def flying(self) -> bool:
        """Return whether a photo is on its way."""
        return self._flight.state() == QVariantAnimation.State.Running

    def remaining_ms(self) -> int:
        """Return how long until the photo lands, 0 if none is flying."""
        if not self.flying:
            return 0
        return max(0, self._flight.duration() - self._flight.currentTime())

    def launch(self, picture: QPixmap, start: QRectF, target: QPointF) -> None:
        """Fly *picture*, now at *start*, into *target* (parent coordinates)."""
        self.stop()
        if picture.isNull() or start.isEmpty() or not native.animations_enabled():
            return
        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(parent.rect())
        self._picture = picture
        self._start = QRectF(start)
        self._target = QPointF(target)
        self._k = 0.0
        self._drawn = QRectF(start)
        self.show()
        self.raise_()
        self._flight.start()

    def stop(self) -> None:
        """Land at once, leaving nothing on screen."""
        self._flight.stop()
        self._picture = QPixmap()
        self.hide()

    def rect_at(self, k: float) -> QRectF:
        """Return where the thumbnail is *k* of the way through its flight."""
        start, end = self._start, self._target
        end_height = float(theme.EXPORT_FLY_END_HEIGHT)
        end_width = end_height * start.width() / max(1.0, start.height())
        # Along a curve that rises above the photo and bends to the button.
        e = _ease_in_out(k)
        first = start.center()
        bend = QPointF(
            first.x() + (end.x() - first.x()) * 0.55,
            start.top() - theme.EXPORT_FLY_ARC,
        )
        x = (1 - e) ** 2 * first.x() + 2 * (1 - e) * e * bend.x() + e**2 * end.x()
        y = (1 - e) ** 2 * first.y() + 2 * (1 - e) * e * bend.y() + e**2 * end.y()
        # Shrinking a little ahead of the travel. A fourth power here put
        # three quarters of the shrink into the first 120 ms, before the
        # photo had moved, and it read as vanishing on the spot.
        shrink = 1.0 - (1.0 - k) ** 2
        width = start.width() + (end_width - start.width()) * shrink
        height = start.height() + (end_height - start.height()) * shrink
        return QRectF(x - width / 2, y - height / 2, width, height)

    def opacity_at(self, k: float) -> float:
        """Return how solid the thumbnail is *k* of the way through."""
        fade_from = 1.0 - theme.EXPORT_FLY_FADE_SHARE
        return max(0.0, min(1.0, 1.0 - (k - fade_from) / theme.EXPORT_FLY_FADE_SHARE))

    def _move_to(self, value: object) -> None:
        self._k = float(value)
        now = self.rect_at(self._k)
        margin = theme.BORDER_WIDTH * 2
        self.update(
            self._drawn.united(now).adjusted(-margin, -margin, margin, margin).toAlignedRect()
        )
        self._drawn = now

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Draw the thumbnail where it is, framed by a thin light line."""
        if self._picture.isNull():
            return
        target = self.rect_at(self._k)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setOpacity(self.opacity_at(self._k))
        radius = theme.RADIUS_FILTER_THUMB
        clip = QPainterPath()
        clip.addRoundedRect(target, radius, radius)
        painter.setClipPath(clip)
        painter.drawPixmap(target, self._picture, QRectF(self._picture.rect()))
        painter.setClipping(False)
        painter.setPen(QPen(QColor(theme.TEXT_PRIMARY), theme.BORDER_WIDTH))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        half = theme.BORDER_WIDTH / 2
        painter.drawRoundedRect(target.adjusted(half, half, -half, -half), radius, radius)
