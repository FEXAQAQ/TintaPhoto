"""The histogram at the top of the right hand panel (A-12 and A-13).

The three channels are drawn over one another so that a colour cast shows as
one of them leaning away from the others. The arrows either side of the plot
swap them for a single luminance curve, which answers only "how bright", and
back again; the name of the one showing sits above the plot (A-12). The
triangles in the upper corners light up when the photo has pixels with
nothing left in them, and clicking one marks those pixels on the photo
itself.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from app.i18n import tr
from app.settings import HISTOGRAM_LUMA, HISTOGRAM_MODES, HISTOGRAM_RGB
from ui import theme
from ui.widgets.buttons import SmallIconButton
from ui.widgets.region import styled_background

#: One count per possible value of an 8 bit channel.
BINS: Final[int] = 256

#: A channel at or above this has nothing left in it; at or below the other,
#: nothing either (PLAN.md section 6 step 12).
CLIPPED_HIGH: Final[int] = 254
CLIPPED_LOW: Final[int] = 1

#: Size of the triangles in the corners. They sit in the band above the
#: plot, in line with the arrows below them.
TRIANGLE_SIZE: Final[int] = 10

#: How much of the tallest bar the drawing is scaled to. A histogram almost
#: always has one enormous spike, and scaling to it would flatten everything
#: else into the floor, so the scale comes from a high percentile instead.
SCALE_PERCENTILE: Final[float] = 99.0
SCALE_HEADROOM: Final[float] = 1.15

_CHANNEL_COLOURS: Final[tuple[str, str, str]] = (
    theme.HISTOGRAM_RED,
    theme.HISTOGRAM_GREEN,
    theme.HISTOGRAM_BLUE,
)

#: How much each channel adds to brightness: the Rec. 709 weights, which are
#: sRGB's own primaries, applied to the encoded values as a camera's
#: brightness histogram does.
LUMA_WEIGHTS: Final[tuple[float, float, float]] = (0.2126, 0.7152, 0.0722)

#: How much further than the triangle itself a click still finds it.
_TRIANGLE_REACH: Final[int] = 4


def luma_values(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Return the brightness of each pixel of an 8 bit RGB image."""
    weighted = image.astype(np.float32) @ np.asarray(LUMA_WEIGHTS, dtype=np.float32)
    return np.clip(np.rint(weighted), 0, BINS - 1).astype(np.uint8)


class HistogramWidget(QWidget):
    """The histogram, in three channels or one, with its clipping triangles."""

    #: Which ends the user wants marked on the photo.
    clipping_toggled = Signal(bool, bool)
    #: An arrow moved the plot over to the other way of drawing it.
    mode_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        styled_background(self, "Histogram")
        self.setFixedHeight(theme.HISTOGRAM_HEIGHT)
        self.setMouseTracking(True)

        self._counts: NDArray[np.int64] | None = None
        self._luma_counts: NDArray[np.int64] | None = None
        self._mode = HISTOGRAM_RGB
        self._has_clipped_highlights = False
        self._has_clipped_shadows = False
        self._show_highlights = False
        self._show_shadows = False

        # With only two ways of drawing, both arrows go to the other one; the
        # pair says "there is more than one" the way a carousel does.
        self.previous_button = SmallIconButton(
            "chevron-left", "histogram.switch_mode", "shortcut.histogram_mode", self
        )
        self.next_button = SmallIconButton(
            "chevron-right", "histogram.switch_mode", "shortcut.histogram_mode", self
        )
        self.previous_button.clicked.connect(lambda: self.step_mode(-1))
        self.next_button.clicked.connect(lambda: self.step_mode(1))

    # -- contents --------------------------------------------------------

    def set_photo(self, image: NDArray[np.uint8] | None) -> None:
        """Count the values of a developed photo, or show nothing."""
        if image is None:
            self._counts = None
            self._luma_counts = None
            self._has_clipped_highlights = False
            self._has_clipped_shadows = False
            self.update()
            return

        # Every other pixel in each direction is enough for a shape this
        # small, and it is four times less counting (PLAN.md section 6).
        sample = image[::2, ::2]
        counts = np.stack(
            [np.bincount(sample[..., channel].ravel(), minlength=BINS) for channel in range(3)]
        )
        self._counts = counts
        self._luma_counts = np.bincount(luma_values(sample).ravel(), minlength=BINS)
        brightest = sample.max(axis=2)
        darkest = sample.min(axis=2)
        self._has_clipped_highlights = bool((brightest >= CLIPPED_HIGH).any())
        self._has_clipped_shadows = bool((darkest <= CLIPPED_LOW).any())
        self.update()

    @property
    def mode(self) -> str:
        """Return how the plot is drawn: three channels or brightness alone."""
        return self._mode

    def set_mode(self, mode: str) -> None:
        """Draw the plot *mode*'s way; anything unknown means three channels."""
        self._mode = mode if mode in HISTOGRAM_MODES else HISTOGRAM_RGB
        self.update()

    def step_mode(self, step: int) -> None:
        """Move *step* places along the ways of drawing, wrapping round (A-12)."""
        index = HISTOGRAM_MODES.index(self._mode)
        self.set_mode(HISTOGRAM_MODES[(index + step) % len(HISTOGRAM_MODES)])
        self.mode_changed.emit(self._mode)

    def toggle_mode(self) -> None:
        """Go on to the next way of drawing, as the shortcut does (A-12)."""
        self.step_mode(1)

    @property
    def showing_clipping(self) -> tuple[bool, bool]:
        """Return whether each end is being marked on the photo."""
        return self._show_highlights, self._show_shadows

    def toggle_clipping(self) -> None:
        """Turn both markings on or off together, as J does (A-13)."""
        wanted = not (self._show_highlights or self._show_shadows)
        self._show_highlights = wanted
        self._show_shadows = wanted
        self.update()
        self.clipping_toggled.emit(self._show_highlights, self._show_shadows)

    # -- layout ----------------------------------------------------------

    def plot_rect(self) -> QRect:
        """Return where the plot goes: below the title, between the arrows."""
        side = theme.HISTOGRAM_EDGE + theme.SMALL_ICON_BUTTON_SIZE + theme.SPACE_XS
        return QRect(
            side,
            theme.HISTOGRAM_TITLE_HEIGHT,
            self.width() - 2 * side,
            self.height() - theme.HISTOGRAM_TITLE_HEIGHT - theme.PANEL_PADDING,
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Keep the arrows level with the middle of the plot."""
        super().resizeEvent(event)
        size = theme.SMALL_ICON_BUTTON_SIZE
        plot = self.plot_rect()
        top = plot.top() + (plot.height() - size) // 2
        self.previous_button.move(theme.HISTOGRAM_EDGE, top)
        self.next_button.move(self.width() - theme.HISTOGRAM_EDGE - size, top)

    # -- painting --------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Draw the title, the curves and the two corner triangles."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        area = self.plot_rect()

        # The plot sits in a recess, one shade below the panel, so that the
        # channels are read against a surface of their own and not against
        # the sliders below (PLAN.md 3.5: layers are shades of grey).
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.BG_INSET))
        painter.drawRoundedRect(area, theme.RADIUS_CARD, theme.RADIUS_CARD)

        if self._counts is not None and self._luma_counts is not None:
            if self._mode == HISTOGRAM_LUMA:
                self._draw_curves(
                    painter, area, self._luma_counts[np.newaxis], (theme.HISTOGRAM_LUMA,)
                )
            else:
                self._draw_curves(painter, area, self._counts, _CHANNEL_COLOURS)
        self._draw_title(painter)
        self._draw_triangle(painter, left=False, lit=self._has_clipped_highlights, on=self._show_highlights)
        self._draw_triangle(painter, left=True, lit=self._has_clipped_shadows, on=self._show_shadows)

    def _draw_curves(
        self,
        painter: QPainter,
        area: QRect,
        counts: NDArray[np.int64],
        colours: tuple[str, ...],
    ) -> None:
        """Fill one curve per row of *counts*, in the matching colour."""
        scale = max(1.0, float(np.percentile(counts, SCALE_PERCENTILE)) * SCALE_HEADROOM)
        # Three curves are see-through so that their overlaps blend; one
        # alone has nothing to blend with and is drawn more solid.
        alpha = theme.HISTOGRAM_ALPHA if len(colours) > 1 else theme.HISTOGRAM_LUMA_ALPHA

        painter.setPen(Qt.PenStyle.NoPen)
        for channel, colour in enumerate(colours):
            path = QPainterPath()
            path.moveTo(area.left(), area.bottom())
            for index in range(BINS):
                x = area.left() + area.width() * index / (BINS - 1)
                height = min(1.0, counts[channel, index] / scale) * area.height()
                path.lineTo(x, area.bottom() - height)
            path.lineTo(area.right(), area.bottom())
            path.closeSubpath()

            fill = QColor(colour)
            fill.setAlpha(alpha)
            painter.setBrush(fill)
            painter.drawPath(path)

    def _draw_title(self, painter: QPainter) -> None:
        """Name the way the plot is drawn, centred above it."""
        font = painter.font()
        font.setPixelSize(theme.FONT_SIZE_CAPTION)
        painter.setFont(font)
        painter.setPen(QColor(theme.TEXT_SECONDARY))
        band = QRect(0, 0, self.width(), theme.HISTOGRAM_TITLE_HEIGHT)
        painter.drawText(band, Qt.AlignmentFlag.AlignCenter, self.title())

    def title(self) -> str:
        """Return the name of the way the plot is drawn now."""
        return tr("histogram.mode_luma" if self._mode == HISTOGRAM_LUMA else "histogram.mode_rgb")

    def _draw_triangle(self, painter: QPainter, left: bool, lit: bool, on: bool) -> None:
        """Draw one corner triangle: filled when it is marking the photo."""
        rect = self._triangle_rect(left)
        path = QPainterPath()
        if left:
            path.moveTo(rect.right(), rect.top())
            path.lineTo(rect.left(), rect.top())
            path.lineTo(rect.left(), rect.bottom())
        else:
            path.moveTo(rect.left(), rect.top())
            path.lineTo(rect.right(), rect.top())
            path.lineTo(rect.right(), rect.bottom())
        path.closeSubpath()

        if on:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(theme.TEXT_PRIMARY))
        elif lit:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(theme.TEXT_SECONDARY))
        else:
            painter.setPen(QPen(QColor(theme.TEXT_TERTIARY), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def _triangle_rect(self, left: bool) -> QRect:
        top = (theme.HISTOGRAM_TITLE_HEIGHT - TRIANGLE_SIZE) // 2
        if left:
            return QRect(theme.HISTOGRAM_EDGE, top, TRIANGLE_SIZE, TRIANGLE_SIZE)
        return QRect(
            self.width() - theme.HISTOGRAM_EDGE - TRIANGLE_SIZE, top, TRIANGLE_SIZE, TRIANGLE_SIZE
        )

    # -- the mouse -------------------------------------------------------

    def _triangle_at(self, position: QPoint) -> str | None:
        """Return which triangle is under *position*, if either."""
        reach = _TRIANGLE_REACH
        if self._triangle_rect(left=False).adjusted(-reach, -reach, reach, reach).contains(position):
            return "highlights"
        if self._triangle_rect(left=True).adjusted(-reach, -reach, reach, reach).contains(position):
            return "shadows"
        return None

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Clicking a triangle marks that end on the photo (A-13)."""
        if event.button() is not Qt.MouseButton.LeftButton:
            return
        triangle = self._triangle_at(event.position().toPoint())
        if triangle == "highlights":
            self._show_highlights = not self._show_highlights
        elif triangle == "shadows":
            self._show_shadows = not self._show_shadows
        else:
            return
        self.update()
        self.clipping_toggled.emit(self._show_highlights, self._show_shadows)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Explain the triangles when the pointer is over one."""
        triangle = self._triangle_at(event.position().toPoint())
        if triangle == "highlights":
            self.setToolTip(
                tr("tooltip.with_shortcut", label=tr("histogram.clipped_highlights"), shortcut=tr("shortcut.clipping"))
            )
        elif triangle == "shadows":
            self.setToolTip(
                tr("tooltip.with_shortcut", label=tr("histogram.clipped_shadows"), shortcut=tr("shortcut.clipping"))
            )
        else:
            self.setToolTip("")


def clipping_overlay(
    image: NDArray[np.uint8],
    highlights: bool,
    shadows: bool,
    coloured: bool = True,
) -> NDArray[np.uint8]:
    """Return *image* with its clipped pixels marked (A-13).

    Blue where nothing is left in the shadows, red where nothing is left in
    the highlights, which is the way round PLAN.md section 2.2 asks for.
    The other choice in the settings is without hues (PLAN.md 3.10): then
    each mark is the opposite of what it sits on, black over a blown
    highlight and white over a blocked shadow, so that it still shows.
    """
    if not highlights and not shadows:
        return image
    high_mark = theme.DANGER if coloured else theme.MARK_CLIPPED_HIGH
    low_mark = theme.CLIP_SHADOW if coloured else theme.MARK_CLIPPED_LOW
    marked = image.copy()
    if highlights:
        marked[image.max(axis=2) >= CLIPPED_HIGH] = _as_rgb(high_mark)
    if shadows:
        marked[image.min(axis=2) <= CLIPPED_LOW] = _as_rgb(low_mark)
    return marked



def _as_rgb(colour: str) -> tuple[int, int, int]:
    value = QColor(colour)
    return value.red(), value.green(), value.blue()
