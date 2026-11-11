"""The cuts drawn over the photo while it is being cut into pieces (PLAN.md 3.17).

Like the crop box and the watermark this is not a widget: the canvas hands
it its painter with the rectangle the photo is drawn in. Where the cuts go
is worked out by :mod:`core.split` on the photo's full size frame - the same
pieces the export writes - and only scaled onto the screen here, so what is
shown is what is written.

Each cut is a white line with a dark edge either side, as the spot brush's
outlines are, so it shows on a white wall as well as on a dark sky; each
piece carries the letter its file name ends in.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen

from core import split
from core.split import SplitSpec
from ui import theme


class SplitOverlay(QObject):
    """The cuts across the photo and the letter of each piece."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._shown = False
        self._spec = SplitSpec()
        self._frame: tuple[int, int] = (0, 0)

    @property
    def shown(self) -> bool:
        """Return whether the cuts are drawn over the photo."""
        return self._shown

    @property
    def spec(self) -> SplitSpec:
        """Return the cut being shown."""
        return self._spec

    def set_shown(self, shown: bool) -> None:
        """Draw the cuts over the photo, or stop."""
        self._shown = shown

    def set_spec(self, spec: SplitSpec) -> None:
        """Show the photo cut as *spec* says."""
        self._spec = spec.sane()

    def set_frame(self, frame: tuple[int, int]) -> None:
        """Take the size of the photo on screen, at full resolution."""
        self._frame = frame

    def pieces_on_screen(self, photo: QRectF) -> list[QRectF]:
        """Return where each piece is drawn, for a photo drawn at *photo*."""
        width, height = self._frame
        if width < 1 or height < 1 or photo.isEmpty():
            return []
        across = photo.width() / width
        down = photo.height() / height
        return [
            QRectF(
                photo.x() + rect.x * across,
                photo.y() + rect.y * down,
                rect.width * across,
                rect.height * down,
            )
            for rect in split.rects(self._frame, self._spec)
        ]

    def paint(self, painter: QPainter, photo: QRectF) -> None:
        """Draw the cuts and the letters over the photo at *photo*."""
        if not self._shown:
            return
        pieces = self.pieces_on_screen(photo)
        if not pieces:
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setClipRect(photo)
        lines = self._cuts(pieces, photo)
        halo = QColor(theme.SPOT_HALO)
        halo.setAlpha(theme.SPOT_HALO_ALPHA)
        painter.setPen(QPen(halo, theme.SPLIT_LINE_WIDTH + 2 * theme.SPOT_HALO_WIDTH))
        for start, end in lines:
            painter.drawLine(start, end)
        painter.setPen(QPen(QColor(theme.WHITE), theme.SPLIT_LINE_WIDTH))
        for start, end in lines:
            painter.drawLine(start, end)

        font = QFont(painter.font())
        font.setPixelSize(theme.FONT_SIZE_TITLE)
        font.setWeight(QFont.Weight(theme.FONT_WEIGHT_TITLE))
        painter.setFont(font)
        visible = photo.intersected(QRectF(painter.window()))
        for index, piece in enumerate(pieces):
            shown = piece.intersected(visible)
            if not shown.isEmpty():
                self._draw_letter(painter, split.LETTERS[index], shown)
        painter.restore()

    @staticmethod
    def _cuts(pieces: list[QRectF], photo: QRectF) -> list[tuple[QPointF, QPointF]]:
        """Return the lines where two pieces meet; the photo's own edges are not cuts."""
        lines: list[tuple[QPointF, QPointF]] = []
        downs = sorted({piece.left() for piece in pieces if piece.left() > photo.left()})
        acrosses = sorted({piece.top() for piece in pieces if piece.top() > photo.top()})
        for x in downs:
            lines.append((QPointF(x, photo.top()), QPointF(x, photo.bottom())))
        for y in acrosses:
            lines.append((QPointF(photo.left(), y), QPointF(photo.right(), y)))
        return lines

    @staticmethod
    def _draw_letter(painter: QPainter, letter: str, piece: QRectF) -> None:
        """Write *letter* in the top left corner of *piece*, with a dark edge round it."""
        metrics = QFontMetricsF(painter.font())
        if (
            piece.width() < metrics.horizontalAdvance(letter) + 2 * theme.SPACE_SM
            or piece.height() < metrics.height() + 2 * theme.SPACE_SM
        ):
            return
        origin = QPointF(piece.x() + theme.SPACE_SM, piece.y() + theme.SPACE_SM + metrics.ascent())
        path = QPainterPath()
        path.addText(origin, painter.font(), letter)
        halo = QColor(theme.SPOT_HALO)
        halo.setAlpha(theme.SPOT_HALO_ALPHA)
        painter.setPen(
            QPen(
                halo,
                2 * theme.FACE_LABEL_HALO,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.WHITE))
        painter.drawPath(path)
