"""The drop of ink from the application icon, drawn by hand.

The welcome screen's opening draws it large and the "developing" sign on the
photo draws it small; both take the outline and the colours from the icon's
own artwork (``resources/icons/app.svg``), the one place the interface uses
a gradient (PLAN.md 3.4).
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QTransform

from ui import theme


def ink_drop(centre: QPointF, height: float) -> QPainterPath:
    """Return the icon's drop of ink, *height* tall, its middle at *centre*.

    The outline is the one in ``resources/icons/app.svg``: 256 units square,
    the drop running from 28 to 216 down it.
    """
    path = QPainterPath()
    path.moveTo(128, 28)
    path.cubicTo(128, 28, 64, 108, 64, 152)
    path.arcTo(QRectF(64, 88, 128, 128), 180, 180)
    path.cubicTo(192, 108, 128, 28, 128, 28)
    scale = height / 188.0
    placed = QTransform()
    placed.translate(centre.x(), centre.y())
    placed.scale(scale, scale)
    placed.translate(-128, -122)
    return placed.map(path)


def ink_brush(path: QPainterPath) -> QLinearGradient:
    """Return the drop's own colours across *path*, as the icon lays them."""
    bounds = path.boundingRect()
    gradient = QLinearGradient(
        bounds.left() + 0.1 * bounds.width(),
        bounds.top(),
        bounds.left() + 0.9 * bounds.width(),
        bounds.bottom(),
    )
    for stop, colour in zip(theme.INK_STOPS, theme.INK_COLOURS):
        gradient.setColorAt(stop, QColor(colour))
    return gradient


def paint_ink_drop(
    painter: QPainter, centre: QPointF, height: float, wide: float = 1.0, tall: float = 1.0
) -> None:
    """Fill the drop at *centre*, *height* tall, stretched by *wide* and *tall*."""
    if height <= 0:
        return
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.translate(centre)
    painter.scale(wide, tall)
    path = ink_drop(QPointF(0, 0), height)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(ink_brush(path))
    painter.drawPath(path)
    painter.restore()
