"""The central preview: the photo, zooming and panning (PLAN.md 3.1, 3.3).

Zoom is kept as a scale factor and a point of the photo that sits at the
centre of the view. Expressing it that way is what makes zooming around the
pointer simple: the point under the pointer is worked out first and then put
back where it was.

The scale is measured against the photo's own pixels rather than against the
pixels that happen to have been developed: at 100% one pixel of the photo
covers one pixel of the screen, whatever size the preview was developed at.
What arrives to be drawn is either the whole photo, developed at the size it
is shown, or one window of it at the photo's own resolution once the zoom has
gone past what the whole preview can show (PLAN.md 3.3). Either way it is
drawn into the place in the frame it was taken from, so that the two replace
each other without the picture moving.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray
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
from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor, QFontMetricsF, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.i18n import tr
from ui import native, theme
from ui.crop_overlay import CropOverlay
from ui.spot_overlay import SpotOverlay
from ui.face_overlay import FaceOverlay
from ui.liquify_overlay import LiquifyOverlay
from ui.skin_overlay import SkinOverlay
from ui.watermark_overlay import WatermarkOverlay
from ui.welcome_view import WelcomeView
from ui.widgets.buttons import TextButton, tooltip
from ui.widgets.ink import paint_ink_drop
from ui.widgets.region import set_state, styled_background

#: The zoom steps the wheel moves through (PLAN.md section 3.3). Fit to window
#: is a step of its own, held separately because it depends on the window.
ZOOM_STEPS: Final[tuple[float, ...]] = (0.25, 0.33, 0.50, 0.67, 1.00, 1.50, 2.00, 4.00)
ZOOM_ACTUAL: Final[float] = 1.00

#: How much of the photo must stay on screen when panning (PLAN.md 3.3).
MINIMUM_VISIBLE_PX: Final[int] = 24

#: The smallest change a turn of the wheel may make, as a fraction.
#: Fitting the window lands on whatever scale the photo and the window give,
#: which can sit a hair below one of the steps above. Stepping onto it would
#: change the picture by a percent or two and feel like nothing happened, so
#: a step that close is passed over. Neighbouring steps are at least a quarter
#: apart, so nothing else is ever skipped.
MINIMUM_ZOOM_CHANGE: Final[float] = 0.10


class PhotoCanvas(QWidget):
    """Draws the photo, and owns the zoom and pan state."""

    zoom_changed = Signal(float)
    #: How far the photo is zoomed in, which part of it is on screen as
    #: fractions of the frame, and whether the hand has come off the gesture.
    #: What is on screen is developed on its own once it has settled.
    view_changed = Signal(float, object, bool)
    pointer_moved = Signal()
    #: The long edge of the area the photo is drawn in, in pixels.
    resized = Signal(int)
    previous_requested = Signal()
    next_requested = Signal()
    #: Where the white balance eyedropper was clicked, in fractions of the
    #: photo's width and height (A-24).
    white_balance_picked = Signal(float, float)
    #: Delete was pressed with a spot selected (A-32): its index.
    spot_deleted = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        styled_background(self, "PreviewCanvas")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        #: The strip along the bottom that the view bar floats over: the
        #: photo is fitted and centred above it, but a photo larger than the
        #: window carries on underneath it (PLAN.md 3.3).
        self._bottom_reserve = 0

        self._pixmap: QPixmap | None = None
        #: The whole photo as it was last developed, drawn under a window of
        #: it. Panning uncovers ground the window was never developed for,
        #: and this is what stands there until the next one arrives: the
        #: photo, soft, rather than the empty background (PLAN.md 3.3).
        self._backdrop: QPixmap | None = None
        #: The size of the whole photo in its own pixels, which is what the
        #: scale and the pan are measured against.
        self._frame = QSize(0, 0)
        #: The part of the frame the pixmap holds, in the frame's pixels.
        self._source = QRectF()
        #: Screen pixels per pixel of the photo; 1.0 is actual size.
        self._scale = 1.0
        self._fitting = True
        #: The point of the photo drawn at the centre of the view.
        self._centre = QPointF(0.0, 0.0)
        self._panning_from: QPoint | None = None
        self._animation: QVariantAnimation | None = None
        #: How far a newly opened photo has faded in, 1.0 when it is all
        #: there (PLAN.md 3.6).
        self._fade = 1.0
        self._fade_animation: QVariantAnimation | None = None
        #: The picture being taken over by a new one of the same photo, and
        #: where it was drawn: the camera's preview giving way to the photo as
        #: this program develops it (PLAN.md 3.7). ``None`` once it has gone.
        self._previous: QPixmap | None = None
        self._previous_part = QRectF()
        #: How far the new picture has come over the old one, 1.0 when done.
        self._blend = 1.0
        self._blend_animation: QVariantAnimation | None = None

        #: The crop box, drawn over the photo while the crop tab is open.
        self.crop = CropOverlay(self)
        #: In the crop tab the drag belongs to the crop box, so panning moves
        #: to the space bar (PLAN.md 3.3).
        self._space_held = False
        #: Whether the next click on the photo is the white balance
        #: eyedropper's (A-24). The drag goes to the space bar meanwhile, as
        #: it does in the crop tab.
        self._picking = False
        #: The spot removal tool (A-32), drawn over the photo while it is on.
        #: Like the eyedropper it takes the click, and panning goes to the
        #: space bar.
        self.spots = SpotOverlay(self)
        #: The portrait brush and the skin it works on (PLAN.md 3.15).
        self.skin = SkinOverlay(self)
        #: The liquify brush (PLAN.md 3.16); it takes the click while liquify
        #: is open, and panning goes to the space bar.
        self.liquify = LiquifyOverlay(self)
        #: The face's points while they are placed by hand (L-12). A press
        #: on a point drags it; anywhere else still pans.
        self.faces = FaceOverlay(self)
        #: The watermark, drawn over the photo while it is being placed
        #: (PLAN.md 3.14). A press on it drags it; anywhere else still pans.
        self.watermark = WatermarkOverlay(self)

    @property
    def frame_size(self) -> tuple[int, int]:
        """Return the size of the photo on screen, at full resolution."""
        return self._frame.width(), self._frame.height()

    # -- the eyedropper --------------------------------------------------

    @property
    def picking(self) -> bool:
        """Return whether a click on the photo is read for white balance."""
        return self._picking

    def set_picking(self, picking: bool) -> None:
        """Let the next click on the photo set the white balance, or stop."""
        self._picking = picking
        self._update_cursor()

    def _pick_at(self, position: QPointF) -> bool:
        """Read the eyedropper's click, or return False if it missed the photo."""
        if self._frame.isEmpty():
            return False
        point = self.widget_to_photo(position.toPoint())
        across = point.x() / self._frame.width()
        down = point.y() / self._frame.height()
        if not (0.0 <= across <= 1.0 and 0.0 <= down <= 1.0):
            return False
        self.white_balance_picked.emit(across, down)
        return True

    # -- the photo -------------------------------------------------------

    def set_bottom_reserve(self, height: int) -> None:
        """Keep *height* pixels along the bottom out of fitting and centring."""
        if height == self._bottom_reserve:
            return
        self._bottom_reserve = height
        if self._fitting:
            self._apply_scale(self.fit_scale())
        self.update()

    def view_rect(self) -> QRect:
        """Return the part of the canvas the photo is fitted and centred in."""
        return self.rect().adjusted(0, 0, 0, -self._bottom_reserve)

    def _middle(self) -> QPointF:
        return QPointF(self.view_rect().center())

    def photo_rect(self) -> QRectF:
        """Return where the photo is drawn, in this widget's coordinates.

        The crop box lives on the photo rather than on the window, so this is
        what turns the one into the other for the overlay.
        """
        if self._pixmap is None or self._pixmap.isNull() or self._frame.isEmpty():
            return QRectF()
        middle = self._middle()
        return QRectF(
            middle.x() - self._centre.x() * self._scale,
            middle.y() - self._centre.y() * self._scale,
            self._frame.width() * self._scale,
            self._frame.height() * self._scale,
        )

    def set_photo(
        self,
        image: NDArray[np.uint8],
        fade: bool = False,
        frame: tuple[int, int] | None = None,
        region: tuple[float, float, float, float] | None = None,
        fade_ms: int = theme.DURATION_PHOTO_FADE_MS,
        blend: bool = False,
    ) -> None:
        """Show *image*, keeping the zoom if it is the same photo.

        *frame* is the size of the whole photo in its own pixels and *region*
        says which part of it *image* holds, in fractions of that frame; left
        out, the image is taken to be the whole photo at its own size. The
        zoom is kept whenever the frame has not changed, because a window of
        the photo and the whole of it are the same picture shown two ways and
        the view must not jump between them (PLAN.md 3.3).

        With *fade* the photo comes up over the documented 120 ms rather
        than appearing at once (PLAN.md 3.6). That is for a photo the user
        has moved to; the many previews of one photo being adjusted replace
        each other instantly, or the panel would lag behind the slider.
        *fade_ms* says how long it takes.

        With *blend* the picture on screen gives way to this one over
        DURATION_PAGE_FADE_MS instead of being swapped: the camera's own
        preview of a RAW file and the photo as developed here differ in
        colour, and swapped at once the difference reads as a flicker. Only
        when the photo lands where the old picture was; a picture of another
        shape or framing is swapped, since blending the two would show both.
        """
        before_rect = self.photo_rect()
        before_part = self._part_rect()
        before_pixmap = self._pixmap
        height, width = image.shape[:2]
        whole = QSize(*frame) if frame is not None else QSize(width, height)
        if whole.isEmpty():
            whole = QSize(width, height)
        same_photo = self._pixmap is not None and self._frame == whole
        self._pixmap = _to_pixmap(image)
        self._frame = whole
        self.watermark.set_frame((whole.width(), whole.height()))
        self._source = _source_rect(whole, region)
        if region is None:
            # The whole photo: what every window of it is drawn on top of.
            self._backdrop = self._pixmap
        elif not same_photo:
            self._backdrop = None
        if not same_photo:
            self.fit_to_window()
        self._stop_blend()
        if blend and before_pixmap is not None and not fade:
            self._blend_from(before_pixmap, before_part, before_rect)
        if fade:
            # A photo moved to keeps the zoom when it happens to be the same
            # shape as the one before it, so what is on screen has to be said
            # again: it is a different photo now, and nothing else would ask
            # for the window of it (PLAN.md 3.3).
            self._fade_in(fade_ms)
            self._announce_view(settled=True)
        self.update()

    def set_backdrop(self, image: NDArray[np.uint8], frame: tuple[int, int] | None) -> None:
        """Replace the photo drawn under the window, without disturbing it.

        What panning uncovers has to say what the photo is set to now, so it
        is developed again whenever the adjustments move; the window on
        screen is sharper and stays exactly where it is (PLAN.md 3.3).
        """
        if self._pixmap is None:
            return
        whole = QSize(*frame) if frame is not None else self._frame
        if whole != self._frame or self._source == _source_rect(self._frame, None):
            return
        self._backdrop = _to_pixmap(image)
        self.update()

    @property
    def fading(self) -> bool:
        """Return whether a photo is still coming up."""
        return self._fade_animation is not None

    def _fade_in(self, length: int = theme.DURATION_PHOTO_FADE_MS) -> None:
        """Bring the photo up from the background over *length* milliseconds."""
        self._stop_fade()
        if not native.animations_enabled():
            # Windows is set to show no animation, so the photo is simply
            # there (PLAN.md 3.6).
            self._fade = 1.0
            return
        animation = QVariantAnimation(self)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setDuration(length)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.valueChanged.connect(self._apply_fade)
        animation.finished.connect(self._stop_fade)
        self._fade = 0.0
        animation.start()
        self._fade_animation = animation

    def _apply_fade(self, value: float) -> None:
        self._fade = float(value)
        self.update()

    def _stop_fade(self) -> None:
        """End the fade, leaving the photo all the way up."""
        animation = self._fade_animation
        self._fade_animation = None
        if animation is not None:
            animation.stop()
            animation.deleteLater()
        self._fade = 1.0
        self.update()

    @property
    def blending(self) -> bool:
        """Return whether an old picture is still giving way to a new one."""
        return self._blend_animation is not None

    def _blend_from(self, pixmap: QPixmap, part: QRectF, rect: QRectF) -> None:
        """Let *pixmap*, drawn at *part*, give way to the picture just set."""
        if not native.animations_enabled():
            return
        after = self.photo_rect()
        if rect.isEmpty() or after.isEmpty():
            return
        tolerance = 2.0
        if any(
            abs(a - b) > tolerance
            for a, b in (
                (rect.left(), after.left()),
                (rect.top(), after.top()),
                (rect.right(), after.right()),
                (rect.bottom(), after.bottom()),
            )
        ):
            return
        self._previous = pixmap
        self._previous_part = part
        animation = QVariantAnimation(self)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setDuration(theme.DURATION_PAGE_FADE_MS)
        # Watched rather than glanced at, so even through the middle (3.6).
        animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        animation.valueChanged.connect(self._apply_blend)
        animation.finished.connect(self._stop_blend)
        self._blend = 0.0
        animation.start()
        self._blend_animation = animation

    def _apply_blend(self, value: float) -> None:
        self._blend = float(value)
        self.update()

    def _stop_blend(self) -> None:
        """End the cross-fade, leaving only the new picture."""
        animation = self._blend_animation
        self._blend_animation = None
        if animation is not None:
            animation.stop()
            animation.deleteLater()
        self._previous = None
        self._blend = 1.0
        self.update()

    def _part_rect(self) -> QRectF:
        """Return where the developed picture is drawn, in this widget."""
        if self._pixmap is None or self._frame.isEmpty():
            return QRectF()
        middle = self._middle()
        return QRectF(
            middle.x() + (self._source.left() - self._centre.x()) * self._scale,
            middle.y() + (self._source.top() - self._centre.y()) * self._scale,
            self._source.width() * self._scale,
            self._source.height() * self._scale,
        )

    def clear(self) -> None:
        """Forget the photo being shown."""
        self._pixmap = None
        self.update()

    @property
    def has_photo(self) -> bool:
        """Return whether there is a photo to draw."""
        return self._pixmap is not None

    @property
    def scale(self) -> float:
        """Return the scale the photo is drawn at; 1.0 means actual pixels."""
        return self._scale

    @property
    def zooming(self) -> bool:
        """Return whether a zoom is still sliding towards its target."""
        return self._animation is not None

    @property
    def showing_photo_pixels(self) -> bool:
        """Return whether what is drawn holds the photo's own pixels.

        True once a window has been developed at the photo's own resolution,
        and false while the whole photo's preview is being stretched over it.
        """
        if self._pixmap is None or self._source.isEmpty():
            return False
        return self._pixmap.width() >= self._source.width() - 1.0

    def visible_region(self) -> tuple[float, float, float, float] | None:
        """Return the part of the photo on screen, in fractions of the frame.

        ``None`` while the photo is fitted to the window, which is not a part
        of it but all of it: the whole preview is what belongs on screen, and
        there is nothing to be gained from developing it a window at a time.
        """
        if self._pixmap is None or self._frame.isEmpty() or self._fitting:
            return None
        width = self._frame.width()
        height = self._frame.height()
        start = self.widget_to_photo(QPoint(0, 0))
        end = self.widget_to_photo(QPoint(self.width(), self.height()))
        return (
            _clamp(start.x() / width, 0.0, 1.0),
            _clamp(start.y() / height, 0.0, 1.0),
            _clamp(end.x() / width, 0.0, 1.0),
            _clamp(end.y() / height, 0.0, 1.0),
        )

    def _announce_view(self, settled: bool = False) -> None:
        """Say what is on screen, so that it can be developed sharply.

        With *settled* the gesture is over - the wheel has stopped sliding or
        the hand has let go - and there is nothing to be gained by waiting any
        longer before developing what it has come to rest on.
        """
        if self._pixmap is None or self._frame.isEmpty():
            return
        self.view_changed.emit(self._scale, self.visible_region(), settled)

    @property
    def fitting(self) -> bool:

        """Return whether the photo is scaled to fit the window."""
        return self._fitting

    # -- zoom ------------------------------------------------------------

    def fit_scale(self) -> float:
        """Return the scale at which the photo just fits, margins included."""
        if self._pixmap is None or self._pixmap.isNull() or self._frame.isEmpty():
            return 1.0
        available = self.view_rect().size() - QSize(
            2 * theme.CANVAS_MARGIN, 2 * theme.CANVAS_MARGIN
        )
        width = max(1, available.width())
        height = max(1, available.height())
        return min(width / self._frame.width(), height / self._frame.height())

    def fit_to_window(self) -> None:
        """Scale the photo so that all of it is visible."""
        self._fitting = True
        self._centre = self._photo_centre()
        self._apply_scale(self.fit_scale())
        self._announce_view(settled=True)

    def zoom_to(self, scale: float, focus: QPoint | None = None, animate: bool = True) -> None:
        """Zoom to *scale*, keeping the photo point under *focus* in place."""
        if self._pixmap is None:
            return
        self._fitting = False
        target = max(min(scale, ZOOM_STEPS[-1]), min(ZOOM_STEPS[0], self.fit_scale()))
        anchor = focus if focus is not None else self.view_rect().center()
        photo_point = self.widget_to_photo(anchor)

        if animate and native.animations_enabled():
            self._animate_scale(target, anchor, photo_point)
        else:
            self._apply_scale(target, anchor, photo_point)
            self._announce_view(settled=True)

    def view_state(self) -> tuple[bool, float, QPointF]:
        """Return how the photo is shown now: fitted or not, the scale, the middle point."""
        return self._fitting, self._scale, QPointF(self._centre)

    def restore_view(self, state: tuple[bool, float, QPointF]) -> None:
        """Show the photo again as :meth:`view_state` said it was."""
        if self._pixmap is None:
            return
        self._stop_animation()
        fitting, scale, centre = state
        if fitting:
            self.fit_to_window()
            return
        self._fitting = False
        self._apply_scale(scale, self._middle().toPoint(), centre)
        self._announce_view(settled=True)

    def zoom_onto(self, box: tuple[float, float, float, float], share: float) -> None:
        """Zoom so that *box* fills *share* of the view, in its middle.

        *box* is left, top, right and bottom in fractions of the frame. The
        photo is never shown smaller than it fits, nor larger than the
        largest zoom step (L-12: placing a face's points).
        """
        if self._pixmap is None or self._frame.isEmpty():
            return
        left, top, right, bottom = box
        width = max((right - left) * self._frame.width(), 1.0)
        height = max((bottom - top) * self._frame.height(), 1.0)
        view = self.view_rect()
        wanted = share * min(view.width() / width, view.height() / height)
        scale = max(min(wanted, ZOOM_STEPS[-1]), self.fit_scale())
        centre = QPointF(
            (left + right) / 2 * self._frame.width(), (top + bottom) / 2 * self._frame.height()
        )
        self._stop_animation()
        self._fitting = False
        self._apply_scale(scale, self._middle().toPoint(), centre)
        self._announce_view(settled=True)

    def toggle_actual_size(self, focus: QPoint | None = None) -> None:
        """Switch between fitting the window and actual pixels (PLAN.md 3.3)."""
        if self._fitting:
            self.zoom_to(ZOOM_ACTUAL, focus)
        else:
            self._animate_to_fit()

    def zoom_in(self, focus: QPoint | None = None) -> None:
        """Move one step up the zoom scale."""
        self.zoom_to(_next_step(self._scale, up=True), focus)

    def zoom_out(self, focus: QPoint | None = None) -> None:
        """Move one step down the zoom scale, stopping at fit to window."""
        fit = self.fit_scale()
        step = _next_step(self._scale, up=False)
        # Fitting the window is the last step down, and the wheel reaches it
        # from wherever the steps run out. A photo much larger than the window
        # fits at well under the smallest step, so that step is no floor: once
        # no step is left below the current scale, or a step would land a hair
        # above the fitting scale, where it is indistinguishable from it, the
        # wheel goes the whole way there.
        no_step_left = step >= self._scale * (1.0 - MINIMUM_ZOOM_CHANGE)
        if no_step_left or step <= fit * (1.0 + MINIMUM_ZOOM_CHANGE):
            self._animate_to_fit()
            return
        self.zoom_to(step, focus)

    def _animate_to_fit(self) -> None:
        if not native.animations_enabled():
            self.fit_to_window()
            return
        self._fitting = True
        self._animate_scale(self.fit_scale(), None, self._photo_centre())

    def _animate_scale(
        self, target: float, anchor: QPoint | None, photo_point: QPointF
    ) -> None:
        """Slide to *target* over the documented duration (PLAN.md 3.6)."""
        self._stop_animation()
        animation = QVariantAnimation(self)
        animation.setStartValue(float(self._scale))
        animation.setEndValue(float(target))
        animation.setDuration(theme.DURATION_ZOOM_MS)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.valueChanged.connect(
            lambda value: self._apply_scale(float(value), anchor, photo_point)
        )
        animation.finished.connect(self._stop_animation)
        # Only a zoom that has run its course emits this; one cut short by
        # the next turn of the wheel does not, so the photo is never
        # developed for a scale the wheel has already left behind.
        animation.finished.connect(lambda: self._announce_view(settled=True))
        # The animation is owned by this widget and let go of when it ends.
        # Asking Qt to delete it when it stops would leave the reference here
        # pointing at a deleted object, and the next zoom would raise instead
        # of zooming, which is what made the wheel stop working after one turn.
        animation.start()
        self._animation = animation

    def _stop_animation(self) -> None:
        """Stop the running zoom animation, if there is one, and let it go."""
        animation = self._animation
        self._animation = None
        if animation is not None:
            animation.stop()
            animation.deleteLater()

    def _apply_scale(
        self, scale: float, anchor: QPoint | None = None, photo_point: QPointF | None = None
    ) -> None:
        """Set the scale and keep *photo_point* under *anchor*."""
        self._scale = scale
        if anchor is not None and photo_point is not None:
            middle = self._middle()
            offset = (middle - QPointF(anchor)) / scale
            self._centre = photo_point + offset
        self._clamp_centre()
        self.zoom_changed.emit(scale)
        self._announce_view()
        self.update()

    # -- panning ---------------------------------------------------------

    def _photo_centre(self) -> QPointF:
        if self._pixmap is None or self._frame.isEmpty():
            return QPointF(0.0, 0.0)
        return QPointF(self._frame.width() / 2, self._frame.height() / 2)

    def _clamp_centre(self) -> None:
        """Keep at least a corner of the photo inside the view (PLAN.md 3.3)."""
        if self._pixmap is None or self._frame.isEmpty():
            return
        margin = MINIMUM_VISIBLE_PX / max(self._scale, 1e-6)
        half_width = self.width() / (2 * max(self._scale, 1e-6))
        half_height = self.view_rect().height() / (2 * max(self._scale, 1e-6))
        self._centre = QPointF(
            _clamp(
                self._centre.x(),
                margin - half_width,
                self._frame.width() - margin + half_width,
            ),
            _clamp(
                self._centre.y(),
                margin - half_height,
                self._frame.height() - margin + half_height,
            ),
        )

    def widget_to_photo(self, point: QPoint) -> QPointF:
        """Return the point of the photo that is drawn at *point*."""
        middle = self._middle()
        return self._centre + (QPointF(point) - middle) / max(self._scale, 1e-6)

    # -- events ----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Draw what has been developed where it belongs in the frame."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        if self._pixmap is None or self._pixmap.isNull():
            return

        photo = self.photo_rect()
        part = self._part_rect()

        # A photo just moved to comes up out of the background; everything
        # else is drawn at full strength.
        painter.setOpacity(self._fade)
        if self._backdrop is not None and part != photo:
            # Only a window of the photo has been developed, so the whole of
            # it goes underneath: panning then uncovers the photo rather than
            # the empty background, and the sharp window follows (PLAN.md 3.3).
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawPixmap(photo, self._backdrop, QRectF(self._backdrop.rect()))

        # Above actual size the real pixels should be visible rather than
        # blurred together (PLAN.md section 3.3) - but only where they really
        # are the photo's pixels. While a sharper window is still being
        # developed what is on screen is the preview stretched, and showing
        # that pixel for pixel is the mosaic rather than the photograph.
        painter.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform,
            not (self.showing_photo_pixels and self._scale >= ZOOM_ACTUAL),
        )
        if self._previous is not None:
            # The picture being taken over stays underneath at full strength
            # while the new one comes up over it.
            painter.drawPixmap(
                self._previous_part, self._previous, QRectF(self._previous.rect())
            )
            painter.setOpacity(self._fade * self._blend)
        painter.drawPixmap(part, self._pixmap, QRectF(self._pixmap.rect()))
        painter.setOpacity(self._fade)
        self._draw_edge(painter, photo)
        painter.setOpacity(1.0)
        self.crop.paint(painter, photo)
        self.skin.paint(painter, photo)
        self.liquify.paint(painter, photo)
        self.faces.paint(painter, photo)
        self.spots.paint(painter, photo)
        self.watermark.paint(painter, photo)

    def _draw_edge(self, painter: QPainter, photo: QRectF) -> None:
        """Draw the hairline that separates the photo from the background.

        A photo with dark corners otherwise runs into the background and its
        shape is lost. One line of the same grey as every other separator,
        because PLAN.md 3.5 rules out the shadow that would do this elsewhere.
        """
        painter.setPen(QPen(QColor(theme.BORDER), theme.BORDER_WIDTH))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        # Half a pixel in, so the line lands on the photo's own edge rather
        # than straddling it.
        half = theme.BORDER_WIDTH / 2
        painter.drawRect(photo.adjusted(half, half, -half, -half))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Keep the photo fitted while it is in fit mode."""
        super().resizeEvent(event)
        if self._fitting:
            self._apply_scale(self.fit_scale())
        else:
            self._clamp_centre()
            self._announce_view()
        self.resized.emit(max(self.width(), self.height()))

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Zoom around the pointer, one step per notch (PLAN.md 3.3)."""
        if self._pixmap is None:
            return
        steps = event.angleDelta().y()
        if steps == 0:
            return
        position = event.position().toPoint()
        if steps > 0:
            self.zoom_in(position)
        else:
            self.zoom_out(position)
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Take hold of the crop box, or start panning."""
        if event.button() is not Qt.MouseButton.LeftButton or self._pixmap is None:
            return
        if self._crop_has_the_drag() and self.crop.press(
            event.position(), self.photo_rect()
        ):
            self.update()
            return
        if self._picking and not self._space_held:
            self._pick_at(event.position())
            return
        if self.watermark.active and self.watermark.press(event.position(), self.photo_rect()):
            self._update_cursor(event.position())
            self.update()
            return
        if self._spots_have_the_click():
            if self.spots.press(event.position(), self.photo_rect()):
                self._update_cursor(event.position())
                self.update()
                return
        if self._skin_has_the_click() and self.skin.press(event.position(), self.photo_rect()):
            self.update()
            return
        if self._faces_have_the_click() and self.faces.press(event.position(), self.photo_rect()):
            self._update_cursor(event.position())
            self.update()
            return
        if self._liquify_has_the_click():
            self.liquify.set_reversed(
                bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
            )
            if self.liquify.press(event.position(), self.photo_rect()):
                self.update()
                return
        self._panning_from = event.position().toPoint()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Drag the crop box, or pan the photo, or just show the right cursor."""
        self.pointer_moved.emit()
        if self._panning_from is None:
            if self._crop_has_the_drag() and self.crop.move(
                event.position(), self.photo_rect()
            ):
                self.update()
                return
            if self.spots.active:
                # The brush circle follows the pointer, or the spot does.
                self.spots.move(event.position(), self.photo_rect())
                self.update()
            if self.skin.active:
                self.skin.move(event.position(), self.photo_rect())
                self.update()
            if self.liquify.active:
                self.liquify.move(event.position(), self.photo_rect())
                self.update()
            if self.faces.active and self.faces.move(event.position(), self.photo_rect()):
                self.update()
            if self.watermark.active:
                free = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
                if self.watermark.move(event.position(), self.photo_rect(), free):
                    self.update()
            self._update_cursor(event.position())
            return
        position = event.position().toPoint()
        delta = QPointF(position - self._panning_from) / max(self._scale, 1e-6)
        self._panning_from = position
        self._centre -= delta
        self._clamp_centre()
        self._announce_view()
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Let go of the crop box, or stop panning."""
        if event.button() is not Qt.MouseButton.LeftButton:
            return
        was_panning = self._panning_from is not None
        if self._panning_from is None and self.crop.active:
            self.crop.release(event.position(), self.photo_rect())
            self.update()
        if self._panning_from is None and self.spots.release(
            event.position(), self.photo_rect()
        ):
            self.update()
        if self._panning_from is None and self.skin.release(event.position(), self.photo_rect()):
            self.update()
        if self._panning_from is None and self.liquify.release(
            event.position(), self.photo_rect()
        ):
            self.update()
        if self._panning_from is None and self.faces.release(event.position(), self.photo_rect()):
            self.update()
        if self._panning_from is None and self.watermark.release(
            event.position(), self.photo_rect()
        ):
            self.update()
        self._panning_from = None
        self._update_cursor(event.position())
        if was_panning:
            self._announce_view(settled=True)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Switch between fitting the window and actual pixels."""
        if (
            self.crop.active
            or self._picking
            or self.spots.active
            or self.skin.active
            or self.liquify.active
            or (self.faces.active and self.faces.over_handle(event.position(), self.photo_rect()))
        ):
            return
        if self.watermark.active and self.watermark.double_click(
            event.position(), self.photo_rect()
        ):
            self.update()
            return
        if self._pixmap is not None:
            self.toggle_actual_size(event.position().toPoint())

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Move between photos, and zoom, while the preview has the focus.

        PLAN.md section 8 is explicit that the arrow keys only change photo
        when the preview has the focus, so that they can adjust a slider when
        one is focused instead.
        """
        key = event.key()
        if key == Qt.Key.Key_Space and not event.isAutoRepeat():
            # Held down, the space bar hands the drag back to panning while
            # the crop box is on screen (PLAN.md 3.3).
            self._space_held = True
            self._update_cursor()
        elif key == Qt.Key.Key_Left:
            self.previous_requested.emit()
        elif key == Qt.Key.Key_Right:
            self.next_requested.emit()
        elif key == Qt.Key.Key_Z:
            self.toggle_actual_size()
        elif (
            key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
            and self.spots.active
            and self.spots.selected >= 0
        ):
            self.spot_deleted.emit(self.spots.selected)
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def keyReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Give the drag back to the crop box when the space bar is let go."""
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            self._update_cursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _crop_has_the_drag(self) -> bool:
        """Return whether the pointer belongs to the crop box right now."""
        return self.crop.active and not self._space_held

    def _spots_have_the_click(self) -> bool:
        """Return whether a click belongs to the spot removal tool right now."""
        return self.spots.active and not self._space_held

    def _skin_has_the_click(self) -> bool:
        """Return whether a click belongs to the portrait brush right now."""
        return self.skin.active and not self._space_held

    def _liquify_has_the_click(self) -> bool:
        """Return whether a click belongs to the liquify brush right now."""
        return self.liquify.active and not self._space_held

    def _faces_have_the_click(self) -> bool:
        """Return whether a click on a face's point takes hold of it right now."""
        return self.faces.active and not self._space_held

    def set_spot_tool(self, active: bool) -> None:
        """Take up the spot removal tool, or put it down (A-32)."""
        self.spots.set_active(active)
        self._update_cursor()
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Take the brush circle away with the pointer."""
        super().leaveEvent(event)
        if self.spots.active:
            self.spots.set_pointer(None)
            self.update()
        if self.skin.active:
            self.skin.set_pointer(None)
            self.update()
        if self.liquify.active and not self.liquify.painting:
            self.liquify.set_pointer(None)
            self.update()
        if self.faces.active:
            self.faces.leave()
            self.update()
        if self.watermark.active:
            self.watermark.leave()
            self.update()

    def refresh_cursor(self) -> None:
        """Show the cursor for what the pointer would do now, after a tool changed."""
        self._update_cursor()

    def _update_cursor(self, point: QPointF | None = None) -> None:
        """Show what the pointer would do where it is."""
        if self._crop_has_the_drag():
            if self.crop.straightening:
                self.setCursor(Qt.CursorShape.CrossCursor)
            elif point is not None:
                self.setCursor(
                    self.crop.cursor_for(self.crop.handle_at(point, self.photo_rect()))
                )
            return
        if self._picking and not self._space_held:
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        if self._spots_have_the_click():
            over = point is not None and self.spots.over_spot(point, self.photo_rect())
            self.setCursor(
                Qt.CursorShape.SizeAllCursor
                if over or self.spots.dragging
                else Qt.CursorShape.CrossCursor
            )
            return
        if self._faces_have_the_click() and (
            self.faces.dragging
            or (point is not None and self.faces.over_handle(point, self.photo_rect()))
        ):
            self.setCursor(Qt.CursorShape.SizeAllCursor)
            return
        if self._skin_has_the_click() or self._liquify_has_the_click():
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        if self.watermark.active and (
            self.watermark.dragging
            or (point is not None and self.watermark.over(point, self.photo_rect()))
        ):
            self.setCursor(Qt.CursorShape.SizeAllCursor)
            return
        can_pan = self._pixmap is not None and not self._fitting
        self.setCursor(
            Qt.CursorShape.OpenHandCursor if can_pan else Qt.CursorShape.ArrowCursor
        )


class ViewBar(QWidget):
    """The floating bar under the photo: fit, actual size, before and after."""

    fit_requested = Signal()
    actual_size_requested = Signal()
    #: Whether the photo as it was found should be shown (E-06). The button is
    #: held down rather than clicked, like the keyboard shortcut.
    comparing = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        styled_background(self, "ViewBar")
        #: The zoom the bar is showing, so that the percentage can be
        #: written again in another language (U-02).
        self._scale = 1.0

        self.fit_button = TextButton("view.fit", self)
        self.fit_button.clicked.connect(self.fit_requested)
        self.actual_button = TextButton("view.actual", self)
        self.actual_button.clicked.connect(self.actual_size_requested)
        self.compare_button = TextButton("view.compare", self)
        self.compare_button.setToolTip(tooltip("view.compare", "shortcut.compare"))
        self.compare_button.pressed.connect(lambda: self.comparing.emit(True))
        self.compare_button.released.connect(lambda: self.comparing.emit(False))
        self.zoom_label = QLabel("", self)
        self.zoom_label.setProperty("role", "secondary")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setMinimumWidth(theme.SPACE_XXL + theme.SPACE_LG)

        self.setFixedHeight(theme.VIEW_BAR_HEIGHT)

        layout = QHBoxLayout(self)
        # Wider margins at the ends than at the top and bottom: the bar's
        # rounded ends curve in, and a button that reached into the curve
        # would look as if it had been clipped.
        layout.setContentsMargins(theme.SPACE_MD, theme.SPACE_XS, theme.SPACE_MD, theme.SPACE_XS)
        layout.setSpacing(theme.SPACE_XS)
        layout.addWidget(self.fit_button)
        layout.addWidget(self.actual_button)
        layout.addWidget(self.compare_button)
        layout.addWidget(self.zoom_label)

        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity)

    def set_zoom(self, scale: float) -> None:
        """Show the current zoom as a percentage."""
        self._scale = scale
        self.zoom_label.setText(tr("view.zoom_percent", percent=round(scale * 100)))

    def retranslate(self) -> None:
        """Read the percentage and the compare tooltip again (U-02)."""
        self.set_zoom(self._scale)
        self.compare_button.setToolTip(tooltip("view.compare", "shortcut.compare"))

    def set_faded(self, faded: bool) -> None:
        """Dim the bar once the pointer has been away for a while."""
        self._opacity.setOpacity(theme.VIEW_BAR_IDLE_OPACITY if faded else 1.0)


class FadingSnapshot(QWidget):
    """A picture of the page being left, fading out over the page arrived at.

    Leaving the welcome screen for the first photo is a cross-fade rather
    than a cut (PLAN.md 3.6): the welcome screen is pictured as it was, laid
    over the photo page and faded out while the photo fades in under it, and
    the picture goes away once it is gone.
    """

    def __init__(self, picture: QPixmap, parent: QWidget) -> None:
        super().__init__(parent)
        self._picture = picture
        self._opacity = 1.0
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setGeometry(parent.rect())
        animation = QVariantAnimation(self)
        animation.setStartValue(1.0)
        animation.setEndValue(0.0)
        animation.setDuration(theme.DURATION_PAGE_FADE_MS)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.valueChanged.connect(self._apply)
        animation.finished.connect(self.deleteLater)
        self._animation = animation

    @property
    def opacity(self) -> float:
        """Return how much of the old page is still showing."""
        return self._opacity

    def start(self) -> None:
        """Show the picture and start fading it out."""
        self.raise_()
        self.show()
        self._animation.start()

    def _apply(self, value: float) -> None:
        self._opacity = float(value)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Draw the old page at its own size, centred on the area as it is now.

        The panels slide in while this fades, narrowing the area; the welcome
        screen's content is centred, so keeping the picture centred and
        unscaled is where that content would have been.
        """
        painter = QPainter(self)
        painter.setOpacity(self._opacity)
        ratio = self._picture.devicePixelRatio() or 1.0
        width = self._picture.width() / ratio
        height = self._picture.height() / ratio
        painter.drawPixmap(
            QPointF((self.width() - width) / 2, (self.height() - height) / 2), self._picture
        )
        painter.end()


class DevelopingIndicator(QWidget):
    """The small sign that the photo on screen is still being developed.

    A drop of ink breathing at the bottom left of the photo, with the word
    beside it, level with the view bar (PLAN.md 3.7). It says that what is
    on screen - a RAW file's own preview, or nothing yet - is not the photo
    as this program will show it, and that the sliders are on their way.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._shown = 0.0
        self._breath = 0.0
        self._active = False
        self._fade: QVariantAnimation | None = None
        self._breathing = QVariantAnimation(self)
        self._breathing.setStartValue(0.0)
        self._breathing.setEndValue(1.0)
        self._breathing.setDuration(theme.DEVELOPING_BREATH_MS)
        self._breathing.setLoopCount(-1)
        self._breathing.valueChanged.connect(self._apply_breath)
        self.hide()

    @property
    def active(self) -> bool:
        """Return whether the photo is said to be developing."""
        return self._active

    @property
    def shown(self) -> float:
        """Return how far the sign has come up, from 0 to 1."""
        return self._shown

    def retranslate(self) -> None:
        """Say the word again in the language just chosen (U-02)."""
        self.adjustSize()
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        metrics = QFontMetricsF(self.font())
        width = theme.DEVELOPING_DROP_HEIGHT + theme.SPACE_SM
        width += metrics.horizontalAdvance(tr("preview.developing"))
        return QSize(round(width) + 1, theme.BUTTON_HEIGHT)

    def set_active(self, active: bool) -> None:
        """Bring the sign up, or let it go as the developed photo arrives."""
        if active == self._active:
            return
        self._active = active
        self._stop_fade()
        if not native.animations_enabled():
            # Windows is set to show no animation: there or not, and still.
            self._shown = 1.0 if active else 0.0
            self._breath = 1.0
            self.setVisible(active)
            self.update()
            return
        if active:
            self.adjustSize()
            self.show()
            self.raise_()
            self._breathing.start()
        animation = QVariantAnimation(self)
        animation.setStartValue(self._shown)
        animation.setEndValue(1.0 if active else 0.0)
        animation.setDuration(theme.DEVELOPING_FADE_MS if active else theme.DURATION_PAGE_FADE_MS)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.valueChanged.connect(self._apply_shown)
        animation.finished.connect(self._stop_fade)
        animation.start()
        self._fade = animation

    def _apply_shown(self, value: float) -> None:
        self._shown = float(value)
        self.update()

    def _apply_breath(self, value: float) -> None:
        # Once round a cosine a second: in and out, never a jolt.
        self._breath = 0.5 - 0.5 * math.cos(2 * math.pi * float(value))
        self.update()

    def _stop_fade(self) -> None:
        animation = self._fade
        self._fade = None
        if animation is not None:
            animation.stop()
            animation.deleteLater()
        self._shown = 1.0 if self._active else 0.0
        if not self._active:
            self._breathing.stop()
            self.hide()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._shown <= 0.0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        low = theme.DEVELOPING_BREATH_LOW
        size = theme.DEVELOPING_DROP_HEIGHT
        painter.setOpacity(self._shown * (low + (1.0 - low) * self._breath))
        paint_ink_drop(painter, QPointF(size / 2, self.height() / 2), size)
        painter.setOpacity(self._shown)
        metrics = QFontMetricsF(self.font())
        painter.setPen(QColor(theme.TEXT_TERTIARY))
        painter.drawText(
            QPointF(
                size + theme.SPACE_SM,
                self.height() / 2 + (metrics.ascent() - metrics.descent()) / 2,
            ),
            tr("preview.developing"),
        )
        painter.end()


class PreviewArea(QStackedWidget):
    """Switches between the welcome screen and the photo."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        styled_background(self, "PreviewArea")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.welcome = WelcomeView(self)
        self.canvas = PhotoCanvas(self)

        # The corner label that says the photo as it was found is on screen
        # (PLAN.md 3.3). A child of the canvas, so it sits over the photo.
        self.before_label = QLabel(tr("preview.before"), self.canvas)
        styled_background(self.before_label, "BeforeLabel")
        self.before_label.adjustSize()
        self.before_label.hide()

        self._photo_page = QWidget(self)
        styled_background(self._photo_page, "PhotoPage")
        self.view_bar = ViewBar(self._photo_page)

        # The bar floats over the bottom of the photo rather than sitting in
        # a strip of its own under it: a photo larger than the window carries
        # on behind it (PLAN.md 3.3). The canvas keeps that strip out of
        # fitting and centring, so a fitted photo stays clear of the bar.
        layout = QVBoxLayout(self._photo_page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas, 1)
        self.canvas.set_bottom_reserve(theme.VIEW_BAR_HEIGHT + theme.SPACE_MD)
        self._photo_page.installEventFilter(self)

        #: Level with the view bar, at the photo's bottom left (PLAN.md 3.7).
        self.developing = DevelopingIndicator(self._photo_page)
        self.view_bar.installEventFilter(self)
        self._leaving: FadingSnapshot | None = None

        self.addWidget(self.welcome)
        self.addWidget(self._photo_page)
        self.show_welcome()

        self.canvas.zoom_changed.connect(self.view_bar.set_zoom)
        self.view_bar.fit_requested.connect(self.canvas.fit_to_window)
        self.view_bar.actual_size_requested.connect(lambda: self.canvas.zoom_to(ZOOM_ACTUAL))

        # The bar dims two seconds after the pointer leaves (PLAN.md 3.3).
        self._fade_timer = QTimer(self)
        self._fade_timer.setSingleShot(True)
        self._fade_timer.setInterval(theme.DURATION_TOAST_MS)
        self._fade_timer.timeout.connect(lambda: self.view_bar.set_faded(True))
        self.canvas.pointer_moved.connect(self._wake_view_bar)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Keep the fading welcome screen over the whole area as it narrows."""
        super().resizeEvent(event)
        if self._leaving is not None:
            self._leaving.setGeometry(self.rect())
        self._place_before_label()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt naming
        """Keep the developing sign level with the view bar wherever it goes."""
        if watched is self.view_bar and event.type() in (
            QEvent.Type.Move,
            QEvent.Type.Resize,
            QEvent.Type.Show,
        ):
            self._place_developing()
        if (watched is self._photo_page and event.type() in (
            QEvent.Type.Resize,
            QEvent.Type.Show,
        )) or (watched is self.view_bar and event.type() == QEvent.Type.LayoutRequest):
            # The bar is not in a layout, so it is measured again whenever its
            # words change width: the zoom, or another language.
            self._place_view_bar()
        return super().eventFilter(watched, event)

    def _place_view_bar(self) -> None:
        """Float the bar at the bottom centre, over the photo."""
        bar = self.view_bar
        bar.adjustSize()
        page = self._photo_page
        bar.move(
            (page.width() - bar.width()) // 2,
            page.height() - theme.SPACE_MD - bar.height(),
        )
        bar.raise_()

    def _place_developing(self) -> None:
        bar = self.view_bar.geometry()
        self.developing.adjustSize()
        self.developing.move(
            theme.CANVAS_MARGIN,
            bar.top() + (bar.height() - self.developing.height()) // 2,
        )

    def set_developing(self, developing: bool) -> None:
        """Say whether the photo on screen is still being developed (PLAN.md 3.7)."""
        if developing:
            self._place_developing()
        self.developing.set_active(developing)

    @property
    def leaving(self) -> FadingSnapshot | None:
        """Return the welcome screen still fading out, if it is."""
        return self._leaving

    def retranslate(self) -> None:
        """Read the corner label again after a language change (U-02).

        It is sized to its own text and placed by hand, so it has to be
        measured again once the words have changed.
        """
        self.before_label.setText(tr("preview.before"))
        self.before_label.adjustSize()

    def _wake_view_bar(self) -> None:
        """Bring the bar back and start counting again."""
        self.view_bar.set_faded(False)
        self._fade_timer.start()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Start the countdown as soon as the pointer leaves the preview."""
        super().leaveEvent(event)
        self._fade_timer.start()

    def set_drop_target(self, active: bool) -> None:
        """Show or hide the outline that says a drag would land here (3.7)."""
        for widget in (self, self._photo_page):
            set_state(widget, "dropTarget", active)
        self.welcome.set_drop_target(active)

    def show_welcome(self) -> None:
        """Show the welcome screen."""
        self.setCurrentWidget(self.welcome)

    def show_photo(
        self,
        image: NDArray[np.uint8],
        fade: bool = False,
        frame: tuple[int, int] | None = None,
        region: tuple[float, float, float, float] | None = None,
        blend: bool = False,
    ) -> None:
        """Show a developed photo, fading it in if it is a photo moved to.

        *frame* and *region* say how large the whole photo is and which part
        of it *image* holds, which is how a window of a zoomed in photo finds
        its place (PLAN.md 3.3). *blend* lets the picture on screen give way
        to this one, for the developed photo replacing a camera's preview.
        Arriving from the welcome screen cross-fades the two pages.
        """
        leaving_welcome = self.currentWidget() is self.welcome
        if leaving_welcome and native.animations_enabled():
            # The welcome screen fades out over the photo page while the
            # first photo fades in under it, rather than being cut (3.6).
            snapshot = FadingSnapshot(self.welcome.grab(), self)
            snapshot.destroyed.connect(lambda _object=None: self._forget_leaving())
            self._leaving = snapshot
        self.canvas.set_photo(
            image,
            fade=fade or leaving_welcome,
            frame=frame,
            region=region,
            fade_ms=theme.DURATION_PAGE_FADE_MS if leaving_welcome else theme.DURATION_PHOTO_FADE_MS,
            blend=blend,
        )
        self.setCurrentWidget(self._photo_page)
        if self._leaving is not None and leaving_welcome:
            self._leaving.start()
        self.view_bar.set_zoom(self.canvas.scale)

    def _forget_leaving(self) -> None:
        self._leaving = None

    def set_backdrop(
        self, image: NDArray[np.uint8], frame: tuple[int, int] | None = None
    ) -> None:
        """Bring the photo under a zoomed in window up to date (PLAN.md 3.3)."""
        self.canvas.set_backdrop(image, frame)

    def set_background(self, choice: str) -> None:
        """Paint the area behind the photo in one of the three greys (3.10)."""
        for widget in (self, self.canvas):
            widget.setProperty("background", choice)
            style = widget.style()
            style.unpolish(widget)
            style.polish(widget)
        self.update()

    def set_showing_before(self, before: bool) -> None:

        """Show or hide the corner label of the before and after view (E-06)."""
        if before:
            self.before_label.adjustSize()
            self._place_before_label()
            self.before_label.raise_()
        self.before_label.setVisible(before)

    def _place_before_label(self) -> None:
        """Put the label in the photo's top right corner.

        The top left is the info card's (PLAN.md 3.9), and the two can be up
        at once.
        """
        self.before_label.move(
            self.canvas.width() - self.before_label.width() - theme.CANVAS_MARGIN,
            theme.CANVAS_MARGIN,
        )


def _source_rect(
    frame: QSize, region: tuple[float, float, float, float] | None
) -> QRectF:
    """Return where a developed image sits in its frame, in frame pixels."""
    if region is None:
        return QRectF(0.0, 0.0, float(frame.width()), float(frame.height()))
    left, top, right, bottom = region
    return QRectF(
        left * frame.width(),
        top * frame.height(),
        max(right - left, 0.0) * frame.width(),
        max(bottom - top, 0.0) * frame.height(),
    )


def _to_pixmap(image: NDArray[np.uint8]) -> QPixmap:
    """Convert an 8 bit RGB array into something Qt can draw."""
    contiguous = np.ascontiguousarray(image)
    height, width, _channels = contiguous.shape
    # The QImage borrows the array's memory, so it is copied before the array
    # goes out of scope.
    picture = QImage(
        contiguous.data, width, height, 3 * width, QImage.Format.Format_RGB888
    ).copy()
    return QPixmap.fromImage(picture)


def _next_step(scale: float, up: bool) -> float:
    """Return the next zoom step that is far enough from *scale* to be seen."""
    if up:
        for step in ZOOM_STEPS:
            if step > scale * (1.0 + MINIMUM_ZOOM_CHANGE):
                return step
        return ZOOM_STEPS[-1]
    for step in reversed(ZOOM_STEPS):
        if step < scale * (1.0 - MINIMUM_ZOOM_CHANGE):
            return step
    return ZOOM_STEPS[0]


def _clamp(value: float, low: float, high: float) -> float:
    if low > high:
        return (low + high) / 2
    return max(low, min(high, value))
