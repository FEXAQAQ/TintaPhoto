"""The controller that connects the interface to the image core (PLAN.md 5).

The interface never decodes or processes a photo itself; it asks for one and
listens for what comes back. Everything slow happens on a background thread,
and the results arrive as signals.

Opening a photo happens in two stages, so that something appears at once: the
preview the camera stored inside a RAW file is shown within a few
milliseconds, and the properly decoded photo replaces it when it is ready
(PLAN.md section 3.7).

Editing is non-destructive from end to end: the photo is read and never
written, the adjustments are stored in a parameter file beside it a second
after the last change (E-02), and they are read back when it is opened again
(E-03). Undo and redo live in :mod:`app.history`, one history per photo.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Final, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, QTimer, Signal

from app.history import History
from app.i18n import tr
from app.paths import (
    APP_NAME,
    APP_VERSION,
    earlier_sidecar_dir,
    filters_dir,
    presets_dir,
    sidecar_dir,
    thumbnail_cache_dir,
)
from app.settings import Settings
from app.workers import TaskRunner
from core import color_management as cm
from core import (
    decoder,
    exporter,
    face_detector,
    geometry,
    healing,
    lens,
    metadata,
    pipeline,
    presets,
    sidecar,
    skin_range,
    thumbnails,
    watermark,
)
from core.auto_light import AUTO_NAMES
from core.auto_light import auto_light as find_auto_light
from core.edit_state import DETAIL_NAMES, NUMERIC_NAMES, RESET_NAMES, Edit, EditState
from core.filters import NO_FILTER, Look, load_catalogue
from core.exporter import ExportOptions
from core.faces import Face

from core.geometry import Geometry
from core.healing import Spot
from core.skin_range import SkinRange, SkinStroke
from core.lens import LensSettings
from core.cache import ThumbnailCache
from core.decoder import DecodedPhoto, DecodeError
from core.metadata import PhotoMetadata
from core.presets import Preset
from core.watermark import WatermarkSpec

logger = logging.getLogger(__name__)

#: One task of a group runs at a time, and a new request replaces whatever
#: was waiting rather than queueing behind it (see TaskRunner).
_GROUP_PREVIEW = "preview"
#: The camera's own preview has its own group, so that submitting the real
#: decode right after it does not abandon it before it has been shown.
_GROUP_QUICK_PREVIEW = "quick-preview"
#: The full size decode a zoomed in preview needs has a group of its own: it
#: takes seconds, and a preview asked for in the meantime must not abandon it
#: (PLAN.md 3.3).
_GROUP_FULL = "full-size"
#: The whole photo developed to stand under a zoomed in window has its own
#: group as well: it follows the window rather than competing with it, and a
#: newer one replaces it the way every other preview does (PLAN.md 3.3).
_GROUP_BACKDROP = "backdrop"
_GROUP_METADATA = "metadata"
_GROUP_EDITED = "edited"
#: Only the newest estimate of the exported file's size is of any interest.
_GROUP_ESTIMATE = "estimate"
#: Only the newest click of the white balance eyedropper counts.
_GROUP_PICK = "white-balance-pick"
#: Only the newest press of the automatic light counts.
_GROUP_AUTO_LIGHT = "auto-light"
#: Reading the lens database once, ahead of the first photo.
_GROUP_LENS_DATABASE = "lens-database"
#: Looking for the faces of the photo now open (PLAN.md 6 step 2.6); only
#: the newest photo's search is worth waiting for.
_GROUP_FACES = "faces"
#: The small pictures on the filter tab: only the newest set is wanted.
_GROUP_LOOK_THUMBNAILS = "look-thumbnails"
#: Finding where a new spot takes its texture from. No group: every click
#: of the spot removal tool is a spot of its own and none may be dropped.
_GROUP_SPOT_SOURCE: Final[None] = None
#: Showing the skin the portrait sliders work on (PLAN.md 3.15): only the
#: newest is wanted.
_GROUP_SKIN_SHOWN = "skin-shown"
#: How many times a look thumbnail's side the photo is developed at along
#: its long edge, so that even a 3:1 panorama still has the side across its
#: short one.
LOOK_THUMBNAIL_REACH: Final[int] = 3
#: How many lens corrected photos are kept: the preview, its draft, the
#: proxy and the full size photo behind a zoomed in window.
_LENSED_KEPT = 4


def _save_group(photo: Path) -> str:
    """Return the task group a save of *photo* belongs to.

    One group per photo: a newer save of the same photo supersedes an older
    one, but switching photo quickly must never drop the save of the one being
    left behind.
    """
    return f"sidecar:{photo}"


#: How long after the last change the parameter file is written (E-02).
AUTOSAVE_DELAY_MS: Final[int] = 1000

#: What a developed preview is for.
_EDITED: Final[str] = "edited"
_BEFORE: Final[str] = "before"

#: What a :class:`_Kept` holds: a framed photo, or step 7's local brightness.
_Made = TypeVar("_Made")


#: The adjustments that rest at different places on a RAW photo and on a
#: JPEG, and so are only copied between photos of the same kind (PLAN.md 7.2).
_KIND_NAMES: Final[frozenset[str]] = frozenset(presets.WHITE_BALANCE_NAMES) | frozenset(
    DETAIL_NAMES
)


class _Kept(Generic[_Made]):
    """One made thing, kept until the key it was made for changes.

    The photo with its geometry applied, which PLAN.md section 6 asks to be
    cached so that moving a colour slider does not rotate the photo again,
    and step 7's local brightness, which a zoomed in preview works out over
    the whole frame and then reuses for every window of it. The work happens
    on the thread that asks for it, which is a background one, and two of
    those can briefly overlap when a request is abandoned, hence the lock
    around the single entry.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: object = None
        self._made: _Made | None = None

    def get(self, key: object, make: Callable[[], _Made | None]) -> _Made | None:
        """Return what was made for *key*, making it if it is not held."""
        with self._lock:
            if key == self._key and self._made is not None:
                return self._made
        made = make()
        with self._lock:
            self._key = key
            self._made = made
        return made

    def clear(self) -> None:
        """Forget what is held, for instance when another photo is shown."""
        with self._lock:
            self._key = None
            self._made = None


class _KeptFew(Generic[_Made]):
    """A few made things, the most recently used kept when there are too many.

    For the lens corrected photo (PLAN.md 6 step 1.5), which the preview
    needs at several sizes at once - the one on screen, the draft under a
    moving hand, the proxy and the full size photo behind a zoomed in window
    - and which only the lens settings change. Thread safe like
    :class:`_Kept`.
    """

    def __init__(self, capacity: int) -> None:
        self._lock = threading.Lock()
        self._capacity = capacity
        self._made: dict[object, _Made] = {}

    def get(self, key: object, make: Callable[[], _Made]) -> _Made:
        """Return what was made for *key*, making it if it is not held."""
        with self._lock:
            found = self._made.pop(key, None)
            if found is not None:
                self._made[key] = found  # most recently used goes last
                return found
        made = make()
        with self._lock:
            self._made[key] = made
            while len(self._made) > self._capacity:
                self._made.pop(next(iter(self._made)))
        return made

    def clear(self) -> None:
        """Forget everything held, for instance when another photo is shown."""
        with self._lock:
            self._made.clear()


#: Long edge the preview is developed at while a slider is being dragged.
#: Half the width is a quarter of the pixels, which is what keeps up with a
#: hand moving a slider; the full size follows as soon as it is let go.
DRAFT_FRACTION: Final[float] = 0.5
#: However small the window, the draft never goes below this.
DRAFT_MINIMUM_LONG_EDGE: Final[int] = 480

#: Fitting the window lands on whatever resolution the photo and the window
#: give, which can sit a hair either side of what the preview holds. A window
#: is only worth developing when it asks for more than that by this much.
_SCALE_SLACK: Final[float] = 0.01

#: A preview of the whole photo held on to for the before and after key: the
#: photo it belongs to, the pixels, and the frame they were developed to fill.
#: The frame is kept with them because it is what says they can still be
#: shown: the crop tab fills a different one (PLAN.md 3.2).
_KeptPreview = tuple[Path, NDArray[np.uint8], tuple[int, int] | None]

#: How long after the last zoom or pan what is on screen is developed. The
#: zoom slides for 160 ms and panning reports every pixel the hand moves, so
#: waiting for it to settle is what keeps a gesture from asking for a window
#: it has already left (PLAN.md 3.3).
VIEW_SETTLE_MS: Final[int] = 120


@dataclass(frozen=True)
class Preview:
    """A developed preview, ready to be shown."""

    path: Path
    image: NDArray[np.uint8]
    #: Whether this is the camera's own preview rather than the real decode.
    provisional: bool
    #: Whether this was developed small to keep up with a drag.
    draft: bool = False
    #: Whether this is the photo as it was found, shown while the before and
    #: after key is held down (E-06).
    before: bool = False
    #: The size the photo has at full resolution with its geometry applied:
    #: the frame the zoom is measured against, so that 100% means one photo
    #: pixel to one screen pixel (PLAN.md 3.3).
    frame: tuple[int, int] | None = None
    #: Which part of that frame these pixels cover, as fractions, when only
    #: what is on screen was developed; ``None`` for the whole photo.
    region: tuple[float, float, float, float] | None = None
    #: Whether this is the whole photo for underneath a zoomed in window
    #: rather than what is to be shown: panning uncovers it, so it has to
    #: say what the photo is set to now (PLAN.md 3.3).
    backdrop: bool = False


@dataclass(frozen=True)
class WatermarkPicture:
    """The watermark drawn for the screen (PLAN.md 3.14)."""

    #: 8 bit RGBA, the watermark's colour in every pixel and its coverage in
    #: the alpha.
    rgba: NDArray[np.uint8]
    #: Pixels of the picture per pixel of the photo's full size frame.
    scale: float
    #: There was Chinese text and no Chinese font to draw it with.
    missing_glyphs: bool


class Controller(QObject):
    """Owns the open folder, the current photo and everything in flight."""

    folder_opened = Signal(list)
    #: The adjustments of the photo now open, and the white balance the
    #: camera recorded, which is where its sliders rest.
    adjustments_ready = Signal(object, object)
    #: Adjustments the sliders must show without announcing them again:
    #: undo, redo and reset (E-04, E-05).
    state_restored = Signal(object)
    #: The geometry of the photo now open, for the crop tab (C-01 to C-06)
    #: and the lens correction group of the adjust tab.
    geometry_changed = Signal(object)
    #: The lens profile of the photo now open, or ``None`` when the database
    #: does not know its lens, and whether it is a RAW file (PLAN.md 2.2).
    lens_ready = Signal(object, bool)
    #: How many faces the photo now open holds, or ``None`` while they are
    #: still being looked for (PLAN.md 2.2, A-29).
    faces_ready = Signal(object)
    #: Whether there is now something to undo, and something to redo.
    history_changed = Signal(bool, bool)
    #: The index of the photo now being shown, and how many there are.
    selection_changed = Signal(int, int)
    preview_ready = Signal(object)
    metadata_ready = Signal(object)
    thumbnail_ready = Signal(Path, object)
    #: The saved presets, whenever the list of them changes (P-04 to P-06).
    presets_changed = Signal(object)
    #: Whether there are settings on the clipboard to paste (P-01, P-02).
    clipboard_changed = Signal(bool)
    #: Something that has just happened, in one translated sentence.
    announcement = Signal(str)
    #: The eyedropper's spot could not be used (A-24); the message saying so
    #: goes out as an announcement as well.
    white_balance_refused = Signal()
    #: Which photos of the open folder have been edited (PLAN.md 3.1).
    edited_photos_ready = Signal(object)
    #: One photo's edited state, after its parameter file was written.
    edited_changed = Signal(Path, bool)
    #: A sentence to show the user, already translated.
    error = Signal(str)
    #: How far the export has got, from 0 to 1 (X-07).
    export_progress = Signal(float)
    #: The finished export, as a :class:`core.exporter.ExportResult`.
    export_finished = Signal(object)
    #: One sentence saying the export did not happen, already translated.
    export_failed = Signal(str)
    #: Roughly how large the exported file would be, in bytes (PLAN.md 3.11).
    export_estimate_ready = Signal(int)
    #: The photo now open as each film look would leave it, by the look's
    #: identifier ("" for none), as square 8 bit pictures for the filter
    #: tab; empty when the photo changes and they are not known yet.
    look_thumbnails_ready = Signal(object)
    #: A stroke of the spot removal tool (A-32) has been mended and added,
    #: by its index in the geometry's spots.
    spot_added = Signal(int)
    #: The skin the portrait sliders work on, as it lies on the frame on
    #: screen: 0 to 255 for each pixel, at a size of its own, or ``None``
    #: when there is none (PLAN.md 3.15).
    skin_shown = Signal(object)

    def __init__(self, settings: Settings, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._runner = TaskRunner(self)
        self._runner.completed.connect(self._on_completed)
        self._runner.failed.connect(self._on_failed)
        self._runner.discarded.connect(self._on_discarded)
        self._cache = ThumbnailCache(thumbnail_cache_dir(), settings.cache_limit_bytes)
        #: The film looks that ship with the program, by their identifier.
        #: Read once: they are files on disk that nothing changes while the
        #: program runs, and every develop needs the one it is wearing.
        self._filters: dict[str, Look] = {
            look.identifier: look for look in load_catalogue(filters_dir())
        }

        self._photos: list[Path] = []
        self._index = -1
        self._decoded: dict[Path, DecodedPhoto] = {}
        self._handlers: dict[int, object] = {}
        #: The decodes under way, by token, so a photo is never decoded twice
        #: at once: clicking a photo that is being preloaded waits for that.
        self._decoding: dict[int, Path] = {}
        #: Counts the folders asked to be opened; a listing that finishes after
        #: a newer one was asked for is dropped, and stops early if it can.
        self._listing = 0
        self._state = EditState()
        self._geometry = Geometry()
        #: Where the current photo rests when nothing is adjusted: what reset
        #: goes back to and what the before and after view shows.
        self._reference = EditState()
        #: Its geometry the same way: the whole frame, and the lens corrected
        #: for a RAW file and not for a JPEG (PLAN.md 2.2).
        self._reference_geometry = Geometry()
        #: The geometry the crop tab was entered with, which cancel goes back
        #: to (PLAN.md 3.2).
        self._geometry_before_crop = Geometry()
        #: While the crop tab is open the whole rotated photo is shown, with
        #: the crop box drawn over it rather than cut out of it.
        self._cropping = False
        self._framed: _Kept[DecodedPhoto] = _Kept()
        #: The photo with its lens corrected (PLAN.md 6 step 1.5), at each
        #: size the preview uses it at. Only the lens settings change it, so
        #: dragging a crop or an angle, or panning a zoomed in view, starts
        #: from what is kept here instead of correcting the lens again.
        self._lensed: _KeptFew[DecodedPhoto] = _KeptFew(_LENSED_KEPT)
        #: The faces found in each photo, by the photo and its lens
        #: correction, which moves them a little (PLAN.md 6 step 2.6). Read
        #: by the threads that develop; only this thread writes to it.
        self._faces: dict[tuple[Path, LensSettings], tuple[Face, ...]] = {}
        #: The same for the zoomed in path, which frames a different photo:
        #: one kept each, or the two would keep throwing the other away.
        self._framed_window: _Kept[DecodedPhoto] = _Kept()
        #: Step 7's local brightness over the whole frame, which every window
        #: of it is then cut from (PLAN.md 3.3).
        self._surroundings: _Kept[NDArray[np.float32]] = _Kept()
        #: The framed photo after noise reduction and sharpening (PLAN.md 6
        #: step 2.5), for the whole preview and for the window. The step only
        #: depends on its own four sliders, so moving any other one goes on
        #: from what is kept here instead of doing it again.
        self._detailed: _Kept[DecodedPhoto] = _Kept()
        self._detailed_window: _Kept[DecodedPhoto] = _Kept()
        #: The film grain of the whole preview and of the window (PLAN.md 6
        #: step 11.5). It depends only on the frame, the window and the
        #: photo's seed, so every slider, the grain's own included, moves
        #: without it being worked out again.
        self._grain: _Kept[NDArray[np.float32]] = _Kept()
        self._grain_window: _Kept[NDArray[np.float32]] = _Kept()
        #: The skin joined to the faces over the whole frame, for whitening
        #: (PLAN.md 6 step 10), for the whole preview and for the window. It
        #: depends only on the framed photo and its faces, so no slider
        #: works it out again; the window takes its part of it, as it does
        #: of step 7's local brightness.
        self._body: _Kept[NDArray[np.float32]] = _Kept()
        self._body_window: _Kept[NDArray[np.float32]] = _Kept()
        self._shot_white_balance: tuple[float, float] = (0.0, 0.0)
        self._preview_long_edge = decoder.PROXY_LONG_EDGE
        self._sized: dict[tuple[Path, int], DecodedPhoto] = {}
        #: The fonts the watermark is drawn in, found the first time it is.
        self._watermark_fonts: watermark.Fonts | None = None

        #: How many screen pixels one pixel of the photo takes up, and which
        #: part of the frame is on screen, in fractions of it (PLAN.md 3.3).
        #: Zero means the interface has not said yet.
        self._zoom = 0.0
        self._visible: tuple[float, float, float, float] | None = None
        #: The photo at full resolution, decoded only once the zoom has gone
        #: past what the proxy can show, and only for the photo on screen.
        self._full: tuple[Path, DecodedPhoto] | None = None
        self._full_token: int | None = None
        #: Whether what was last asked for was a window rather than the whole
        #: photo, so that zooming back out asks for the whole one again.
        self._showing_window = False
        #: Whether the photo standing under the window still shows what the
        #: adjustments say; the next window to arrive brings it up to date.
        self._backdrop_stale = False
        self._display_transform: object | None = None

        self._history = History()
        self._application = sidecar.Application(APP_NAME, APP_VERSION)
        self._store = sidecar_dir()
        # A handful of small files at most, so this is quick enough here.
        sidecar.adopt(earlier_sidecar_dir(), self._store)
        #: Adjustments changed but not yet written; emptied by the save timer.
        self._unsaved: dict[Path, Edit] = {}
        self._edited: set[Path] = set()
        #: The unknown fields each photo's parameter file holds, so that
        #: saving cannot throw away what this version does not understand.
        self._carried: dict[Path, dict[str, Any]] = {}

        self._presets_dir = presets_dir()
        self._presets: list[Preset] = []
        #: The adjustments of P-01, kept until the program closes. Only the
        #: colour: a crop belongs to the photo it was made on (P-03).
        self._clipboard: EditState | None = None

        #: True from the moment an export is handed to a thread until its
        #: result comes back, so that a second one cannot start on top of it.
        self._exporting = False
        #: What the info tab was told about the photo now open, which is also
        #: where the export dialog reads its full size from.
        self._info: PhotoMetadata | None = None

        self._comparing = False

        #: How many pixels across the filter tab's thumbnails are, or 0 while
        #: the tab is out of sight and nobody would see them.
        self._look_thumbnail_side = 0
        #: What the thumbnails last asked for were made of, so that the same
        #: photo, adjustments and size are not developed twice.
        self._look_thumbnails_key: object = None

        #: The last fully developed preview, and the photo as it was found, so
        #: that holding the before and after key is instant the second time.
        self._developed: _KeptPreview | None = None
        self._before: _KeptPreview | None = None

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(AUTOSAVE_DELAY_MS)
        self._save_timer.timeout.connect(self._write_pending)

        self._view_timer = QTimer(self)
        self._view_timer.setSingleShot(True)
        self._view_timer.setInterval(VIEW_SETTLE_MS)
        self._view_timer.timeout.connect(self._develop_window)

        # The lens database takes about half a second to read. Reading it now,
        # while the welcome screen is up, means the first photo opened does
        # not wait for it (PLAN.md 6 step 1.5).
        self._runner.submit(lambda: lens.database(), _GROUP_LENS_DATABASE)

    @property
    def settings(self) -> Settings:
        """Return the application settings."""
        return self._settings

    @property
    def state(self) -> EditState:
        """Return the adjustments of the photo now open."""
        return self._state

    @property
    def geometry(self) -> Geometry:
        """Return the crop and rotation of the photo now open."""
        return self._geometry

    @property
    def edit(self) -> Edit:
        """Return everything the photo now open carries."""
        return Edit(self._state, self._geometry)

    @property
    def photo_size(self) -> tuple[int, int]:
        """Return the size the crop is measured against, before any turning.

        The proxy rather than the original, which has the same shape; the crop
        is held in fractions, so only the shape matters.
        """
        path = self.current
        decoded = self._decoded.get(path) if path is not None else None
        if decoded is None:
            return (1, 1)
        return decoded.size

    @property
    def cropping(self) -> bool:
        """Return whether the crop tab is open."""
        return self._cropping

    def set_display_profile(self, profile: bytes | None) -> None:
        """Say which screen the photo is being shown on (PLAN.md 6 step 13).

        Only the preview is corrected for the screen; an exported file is
        always plain sRGB, so that it looks right on somebody else's.
        """
        self._display_transform = cm.build_display_transform(profile)
        # Both kept previews were developed for the old screen.
        self._developed = None
        self._before = None
        path = self.current
        decoded = self._decoded.get(path) if path is not None else None
        if path is not None and decoded is not None:
            self._request_develop(path, decoded, draft=False)

    def set_preview_long_edge(self, long_edge: int) -> None:
        """Say how large the preview is shown, so it is developed that size.

        The photo on screen is developed again at the new size. Opening a
        folder is when this matters most: the first photo is asked for while
        the preview is still being laid out and reports a tiny size, and the
        real size only arrives a moment later; without this the first photo
        stayed stretched from that tiny preview until another was chosen.
        """
        long_edge = max(DRAFT_MINIMUM_LONG_EDGE, min(decoder.PROXY_LONG_EDGE, int(long_edge)))
        if long_edge == self._preview_long_edge:
            return
        self._preview_long_edge = long_edge
        self._sized.clear()
        self._framed.clear()
        self._detailed.clear()
        self._grain.clear()
        self._body.clear()
        self._surroundings.clear()
        self._developed = None
        self._before = None
        path = self.current
        decoded = self._decoded.get(path) if path is not None else None
        if path is None or decoded is None:
            return
        if self._comparing:
            self._request_develop(
                path, decoded, draft=False, state=self._reference, purpose=_BEFORE
            )
        else:
            self._request_develop(path, decoded, draft=False)

    # -- zooming in (PLAN.md 3.3 and section 4) --------------------------

    def set_view(
        self,
        zoom: float,
        visible: tuple[float, float, float, float] | None,
        settled: bool = False,
    ) -> None:
        """Say how far the preview is zoomed in and what part of it is on screen.

        *zoom* is how many screen pixels one pixel of the photo takes up, so
        one means actual size; *visible* is the part of the frame on screen,
        in fractions of it. Once the zoom goes past what the whole preview
        holds, only what is on screen is developed, and from the full size
        photo rather than from the proxy.

        *settled* says the gesture is over, and then there is nothing to wait
        for: the window is developed at once rather than after the pause that
        keeps a moving hand from asking for windows it has already left.
        """
        self._zoom = max(float(zoom), 0.0)
        self._visible = visible
        if self._zoomed_in():
            if settled:
                self._view_timer.stop()
                self._develop_window()
            else:
                self._view_timer.start()
            return
        self._view_timer.stop()
        if self._showing_window:
            # Back inside what the whole preview can show, so it is the whole
            # photo that belongs on screen again.
            self._showing_window = False
            self._develop_current()

    def _zoomed_in(self) -> bool:
        """Return whether the zoom asks for more than the whole preview holds."""
        if self._cropping or self._visible is None:
            return False
        path = self.current
        decoded = self._decoded.get(path) if path is not None else None
        if decoded is None:
            return False
        return self._window_scale() > self._preview_scale(decoded) * (1.0 + _SCALE_SLACK)

    def _window_scale(self, draft: bool = False) -> float:
        """Return what resolution a window wants, as a fraction of the photo's.

        Past one screen pixel to one photo pixel there is no more detail to
        be had, so that is where it stops; PLAN.md 3.3 asks for the real
        pixels to be visible from there on.
        """
        scale = min(self._zoom, 1.0)
        return scale * DRAFT_FRACTION if draft else scale

    def _preview_scale(self, decoded: DecodedPhoto) -> float:
        """Return what resolution the whole photo preview holds, the same way."""
        shown = min(self._preview_long_edge, max(decoded.size))
        return shown / max(max(decoded.full_size), 1)

    def _frame_of(self, path: Path | None) -> tuple[int, int] | None:
        """Return the size the photo on screen has at full resolution.

        The frame the zoom is measured against: turned, rotated and cropped,
        but never shrunk. While the crop tab is open it is the whole rotated
        frame instead, because that is what is on screen (PLAN.md 3.2).
        """
        decoded = self._decoded.get(path) if path is not None else None
        if decoded is None:
            return None
        if self._cropping:
            return geometry.frame_size(decoded.full_size, self._geometry)
        return geometry.output_size(decoded.full_size, self._geometry)

    def _develop_window(self) -> None:
        """Develop what is on screen, once the zoom and the pan have settled."""
        path = self.current
        decoded = self._decoded.get(path) if path is not None else None
        if path is None or decoded is None or not self._zoomed_in():
            return
        comparing = self._comparing
        self._request_window(
            path,
            decoded,
            draft=False,
            state=self._reference if comparing else None,
            purpose=_BEFORE if comparing else _EDITED,
        )

    def _request_window(
        self,
        path: Path,
        decoded: DecodedPhoto,
        draft: bool,
        state: EditState | None,
        purpose: str,
    ) -> None:
        """Develop the part of the photo on screen, and nothing else.

        Step 7's local brightness is worked out once over the whole frame and
        every window is cut from it, so the only thing a window still has to
        be grown by is the narrow reach of step 4's mask; it is cut back to
        size afterwards and nothing shows a seam where it happened to end
        (PLAN.md section 4).
        """
        visible = self._visible
        frame = self._frame_of(path)
        if visible is None or frame is None:
            return

        source, scale = self._window_source(path, decoded, self._window_scale(draft))
        inner = geometry.Window.around(frame, _in_pixels(frame, visible)).scaled(scale)
        values = self._state if state is None else state
        shape = self._geometry_for(purpose)
        outer = inner.grown(
            pipeline.blur_margin(
                self._with_faces(path, source, shape),
                values,
                inner.frame,
                surroundings_apart=True,
                scale=scale,
            )
        )
        framed = self._framed_window
        detailed = self._detailed_window
        kept_grain = self._grain_window
        kept_body = self._body_window
        kept_field = self._surroundings
        # One framed photo is kept, so the key says which photo was framed
        # and how: its size tells the proxy and the full size one apart, and
        # whether its faces had been found yet tells the two framings apart.
        token = (path, source.size, geometry.frame_key(shape), (path, shape.lens) in self._faces)
        skin = geometry.framed_skin(shape, source.size)
        transform = self._display_transform
        # The local brightness is a blur a hundredth of the frame wide, so it
        # is worked out on the proxy rather than on the full size photo. A
        # draft is rougher still: under a moving hand it comes from the same
        # small photo the whole preview is developed from.
        field_long_edge = self._preview_long_edge if draft else max(decoded.size)
        field_from = self._sized_photo(path, decoded, field_long_edge)
        field_token = (path, shape, field_long_edge, pipeline.surroundings_key(values))

        corrected = self._corrected_lens

        def develop_window(
            photo: DecodedPhoto = source,
            values: EditState = values,
            screen: object = transform,
            shape: Geometry = shape,
            key: object = token,
            around: geometry.Window = outer,
            asked: geometry.Window = inner,
            field_photo: DecodedPhoto = field_from,
            field_key: object = field_token,
            look: Look | None = self._look_for(values),
            skin: SkinRange | None = skin,
        ) -> NDArray[np.uint8]:
            """Develop the window, on whichever thread takes this on."""
            whole = framed.get(
                key, lambda: geometry.apply(corrected(path, photo, shape), shape)
            )
            assert whole is not None  # the maker always returns a photo
            # The framed photo is kept from one stroke of the portrait brush
            # to the next; the newest skin range is put on it here.
            whole = replace(whole, skin=skin)
            field = kept_field.get(
                field_key,
                lambda: pipeline.surroundings_field(
                    geometry.apply(corrected(path, field_photo, shape), shape), values
                ),
            )
            part = detailed.get(
                (key, around, pipeline.detail_key(values)),
                lambda: pipeline.with_detail(geometry.window_of(whole, around), values),
            )
            assert part is not None  # the maker always returns a photo
            part = replace(part, skin=skin)
            grain = (
                kept_grain.get((key, around), lambda: pipeline.grain_field(part, around))
                if pipeline.has_grain(values)
                else None
            )
            body = (
                kept_body.get((key, skin), lambda: pipeline.body_field(whole))
                if pipeline.needs_body(values, whole)
                else None
            )
            developed = pipeline.develop(
                part,
                values,
                window=around,
                surroundings=None if field is None else around.take(field),
                look=look,
                grain=grain,
                body=body,
            )
            return around.trim(cm.apply_display_transform(developed, screen), asked)

        self._showing_window = True
        self._submit(
            develop_window,
            _GROUP_PREVIEW,
            partial(
                self._on_window_developed,
                frame=frame,
                region=_in_fractions(inner),
                values=values,
                before=purpose == _BEFORE,
            ),
            path,
        )

    def _corrected_lens(self, path: Path, photo: DecodedPhoto, shape: Geometry) -> DecodedPhoto:
        """Return *photo* with its lens corrected, keeping it for next time.

        Called on the background thread that develops the preview; the
        geometry step that follows sees a photo already corrected and does
        not correct it again (PLAN.md 6 step 1.5). The photo comes back
        carrying its faces, once they have been found (step 2.6).
        """
        corrected = self._lensed.get(
            (path, photo.size, shape.lens), lambda: lens.correct(photo, shape.lens)
        )
        return self._with_faces(path, corrected, shape)

    def _with_faces(self, path: Path, photo: DecodedPhoto, shape: Geometry) -> DecodedPhoto:
        """Return *photo* carrying the faces found in it, when they are known."""
        found = self._faces.get((path, shape.lens))
        return photo if found is None else replace(photo, faces=found)

    # -- faces (PLAN.md 2.2, A-29, and 6 step 2.6) -------------------------

    def _request_faces(self, path: Path, decoded: DecodedPhoto) -> None:
        """Look for the faces of the photo now open, unless they are known.

        Once for each photo and lens correction: on the proxy, corrected the
        way the preview is, so the faces lie where the preview has them. A
        newer search replaces one still waiting, so switching photos quickly
        only looks at the photo that stays.
        """
        key = (path, self._geometry.lens)
        if key in self._faces:
            if path == self.current:
                self.faces_ready.emit(len(self._faces[key]))
            return
        if path == self.current:
            self.faces_ready.emit(None)
        shape = self._geometry
        corrected = self._corrected_lens

        def find(photo: DecodedPhoto = decoded) -> tuple[Face, ...]:
            return face_detector.find(corrected(path, photo, shape))

        self._submit(find, _GROUP_FACES, partial(self._on_faces, settings=shape.lens), path)

    def _on_faces(
        self, path: Path, found: tuple[Face, ...], settings: LensSettings
    ) -> None:
        """Keep the faces found, and show them if the skin slider wants them."""
        self._faces[(path, settings)] = found
        if path != self.current or settings != self._geometry.lens:
            return
        self.faces_ready.emit(len(found))
        if not found:
            return
        # What was framed and sharpened so far was done without the faces,
        # and would be used again the next time the skin slider moves.
        for kept in (self._framed, self._framed_window, self._detailed, self._detailed_window):
            kept.clear()
        decoded = self._decoded.get(path)
        if decoded is not None and self._state.uses_faces:
            self._request_develop(path, decoded, draft=False)

    def faces_of_current(self) -> tuple[Face, ...] | None:
        """Return the faces of the photo now open, or ``None`` when unknown."""
        path = self.current
        if path is None:
            return None
        return self._faces.get((path, self._geometry.lens))

    def _window_source(
        self, path: Path, decoded: DecodedPhoto, wanted: float
    ) -> tuple[DecodedPhoto, float]:
        """Return the photo a window is cut from, and what resolution it holds.

        The proxy is enough until the zoom goes past it. Beyond that the full
        size photo is wanted, and decoding one takes seconds, so the proxy is
        stretched over the window until it arrives (PLAN.md 3.3).
        """
        proxy = max(decoded.size) / max(max(decoded.full_size), 1)
        if wanted <= proxy:
            return decoded, wanted
        full = self._full
        if full is not None and full[0] == path:
            return full[1], wanted
        self._request_full_decode(path)
        return decoded, proxy

    def _request_full_decode(self, path: Path) -> None:
        """Decode *path* at full resolution, for the zoomed in preview."""
        if self._full_token is not None:
            return
        if self._full is not None and self._full[0] == path:
            return
        self._full_token = self._submit(
            lambda: decoder.decode(path), _GROUP_FULL, self._on_full_decoded, path
        )

    def _on_full_decoded(self, path: Path, decoded: DecodedPhoto) -> None:
        """Keep the full size photo and show the sharp window it allows."""
        self._full_token = None
        if path == self.current:
            self._full = (path, decoded)
            self._framed_window.clear()
            self._detailed_window.clear()
            self._grain_window.clear()
            self._body_window.clear()
        # Developed again either way: the photo may have been moved on from
        # while this was running, and the one now on screen is then still
        # waiting for a decode of its own, which this is what starts.
        self._develop_window()

    def _forget_full_size(self) -> None:
        """Let go of the full size photo; a proxy is small, this is not."""
        self._full = None
        self._framed_window.clear()
        self._detailed_window.clear()
        self._grain_window.clear()
        self._body_window.clear()
        self._surroundings.clear()
        # Its lens corrected copy is as large; the small ones are quick to redo.
        self._lensed.clear()

    def _on_window_developed(
        self,
        path: Path,
        image: NDArray[np.uint8],
        frame: tuple[int, int],
        region: tuple[float, float, float, float],
        values: EditState,
        before: bool,
    ) -> None:
        """Show a window of the photo, saying where it was taken from."""
        if path != self.current or before != self._comparing:
            return
        self.preview_ready.emit(
            Preview(
                path=path,
                image=image,
                provisional=False,
                before=before,
                frame=frame,
                region=region,
            )
        )
        if self._backdrop_stale:
            # Now that what the user is looking at is on screen, the photo
            # underneath it is brought up to date for the next pan.
            self._backdrop_stale = False
            self._request_backdrop(path, values, before)

    def _request_backdrop(self, path: Path, values: EditState, before: bool) -> None:
        """Develop the whole photo for underneath the window (PLAN.md 3.3).

        At the size the whole photo is shown at, which is all it is ever
        stretched from, and in its own group so that it waits for no window
        and no window waits for it.
        """
        decoded = self._decoded.get(path)
        if decoded is None:
            return
        self._submit_whole(
            path,
            decoded,
            values,
            self._preview_long_edge,
            _GROUP_BACKDROP,
            partial(self._on_backdrop_developed, before=before),
            self._geometry_for(_BEFORE if before else _EDITED),
        )

    def _on_backdrop_developed(
        self,
        path: Path,
        image: NDArray[np.uint8],
        frame: tuple[int, int] | None,
        before: bool,
    ) -> None:
        """Hand over the photo that stands under the window, if it still does."""
        if path != self.current or before != self._comparing or not self._showing_window:
            return
        self.preview_ready.emit(
            Preview(
                path=path,
                image=image,
                provisional=False,
                before=before,
                frame=frame,
                backdrop=True,
            )
        )

    @property
    def photos(self) -> list[Path]:
        """Return the photos in the filmstrip, in the order they are shown."""
        return list(self._photos)

    @property
    def index(self) -> int:
        """Return the position of the current photo, or -1 if there is none."""
        return self._index

    @property
    def current(self) -> Path | None:
        """Return the photo being shown, if any."""
        if 0 <= self._index < len(self._photos):
            return self._photos[self._index]
        return None

    # -- opening ---------------------------------------------------------

    def open(self, path: Path) -> None:
        """Open a file or a folder, whichever *path* is (F-01, F-02)."""
        self.open_paths([path])

    def open_paths(self, paths: Sequence[Path]) -> None:
        """Open what was chosen or dropped (F-01, F-02).

        The first path decides: a folder opens with every photo in it, and
        files open by themselves, the folders they sit in left alone.
        """
        if not paths:
            return
        first = paths[0]
        if first.is_dir():
            self.open_folder(first)
        elif first.is_file():
            self.open_files([path for path in paths if path.is_file()])
        else:
            self.error.emit(tr("error.not_found", name=first.name))

    def open_file(self, path: Path) -> None:
        """Open one photo by itself (F-01)."""
        self.open_files([path])

    def open_files(self, paths: Sequence[Path]) -> None:
        """Open just these photos, in name order, showing the first one chosen (F-01).

        Files that are not photos are passed over; if none is a photo, the
        first one is named in the message.
        """
        chosen = list(dict.fromkeys(path for path in paths if decoder.is_supported(path)))
        if not chosen:
            if paths:
                self.error.emit(tr("error.unsupported", name=paths[0].name))
            return
        photos = sorted(chosen, key=_sort_key)
        shown = photos.index(chosen[0])

        def opened(_photos: list[Path]) -> None:
            self._set_photos(None, photos)
            self.select(shown)

        # Handed over the way a listed folder is, so a folder still being
        # listed cannot take the place of these photos once they are open.
        self._deliver_photos(lambda _wanted: photos, photos[0], opened)

    def open_folder(self, folder: Path) -> None:
        """Open every supported photo in *folder* and the folders below it (F-02)."""

        def opened(photos: list[Path]) -> None:
            if not photos:
                self.error.emit(tr("error.empty_folder", name=folder.name))
                return
            self._set_photos(folder, photos)
            self.select(0)

        self._list_folder(folder, True, opened)

    def _list_folder(
        self, folder: Path, recursive: bool, then: Callable[[list[Path]], None]
    ) -> None:
        """List the photos in *folder* in the background and hand them to *then*.

        A folder of folders can hold tens of thousands of files, so the walk
        stays off the interface thread; a newer request makes it stop.
        """
        self._deliver_photos(
            lambda wanted: list_photos(folder, recursive, wanted), folder, then
        )

    def _deliver_photos(
        self,
        find: Callable[[Callable[[], bool]], list[Path]],
        path: Path,
        then: Callable[[list[Path]], None],
    ) -> None:
        """Find photos in the background and hand them to *then*.

        Only the newest request is handed over; *find* is given a function
        saying whether it is still wanted, so a long walk can stop early.
        """
        self._listing += 1
        listing = self._listing

        def wanted() -> bool:
            return listing == self._listing

        def deliver(_path: Path, photos: list[Path]) -> None:
            if wanted():
                then(photos)

        self._submit(partial(find, wanted), None, deliver, path)

    def _set_photos(self, folder: Path | None, photos: list[Path]) -> None:
        """Make *photos* the ones in the filmstrip.

        A *folder* goes on the recent list; photos opened by themselves only
        leave their folder as the place the open dialogs start in.
        """
        self._write_pending()
        self._photos = photos
        self._index = -1
        self._decoded.clear()
        self._carried.clear()
        self._history.forget_all_but(set(photos))
        if folder is not None:
            self._settings.remember_folder(str(folder), len(photos))
        else:
            self._settings.remember_browse_folder(str(photos[0].parent))
        self.folder_opened.emit(list(photos))
        self._request_edited_marks(photos)

    def _request_edited_marks(self, photos: list[Path]) -> None:
        """Find out which photos are edited, for the filmstrip (PLAN.md 3.1)."""
        if not photos:
            self._edited = set()
            self.edited_photos_ready.emit(set())
            return
        store = self._store
        self._submit(
            lambda: sidecar.edited_photos(photos, store),
            _GROUP_EDITED,
            self._on_edited_marks,
            photos[0],
        )

    def _on_edited_marks(self, _path: Path, edited: set[Path]) -> None:
        self._edited = set(edited)
        self.edited_photos_ready.emit(set(edited))

    # -- moving between photos -------------------------------------------

    def select(self, index: int) -> None:
        """Show the photo at *index*."""
        if not self._photos or not 0 <= index < len(self._photos):
            return
        # The photo being left behind must not lose the second it was still
        # waiting to be saved in.
        self._write_pending()
        self._comparing = False
        self._cropping = False
        self._before = None
        self._developed = None
        self._framed.clear()
        self._lensed.clear()
        self._detailed.clear()
        self._grain.clear()
        # The view belongs to the photo that is leaving: the preview will say
        # where the new one sits as soon as it has fitted it to the window.
        self._view_timer.stop()
        self._zoom = 0.0
        self._visible = None
        self._showing_window = False
        self._backdrop_stale = False
        self._forget_full_size()
        self._index = index
        path = self._photos[index]
        self._look_thumbnails_key = None
        self.look_thumbnails_ready.emit({})
        self.selection_changed.emit(index, len(self._photos))
        self._forget_distant_photos()
        self._show(path)
        self._request_metadata(path)

    def select_next(self) -> None:
        """Show the next photo, if there is one."""
        self.select(min(self._index + 1, len(self._photos) - 1))

    def select_previous(self) -> None:
        """Show the previous photo, if there is one."""
        self.select(max(self._index - 1, 0))

    # -- the preview -----------------------------------------------------

    def _show(self, path: Path) -> None:
        """Show *path*, using what is already decoded if possible."""
        decoded = self._decoded.get(path)
        if decoded is not None:
            self._runner.abandon(_GROUP_QUICK_PREVIEW)
            self._announce_adjustments(path, decoded)
            self._request_develop(path, decoded, draft=False)
            self._preload_neighbours()
            return

        if decoder.is_raw(path):
            self._submit(
                lambda: _camera_preview(path),
                _GROUP_QUICK_PREVIEW,
                self._on_camera_preview,
                path,
            )
        # Not in a group: a group would make this photo wait for the one
        # clicked before it to finish decoding, which for a RAW is seconds.
        self._start_decode(path)

    def _start_decode(self, path: Path) -> None:
        """Decode *path* to its proxy, unless that is already under way.

        Decodes are never grouped, so none waits behind another; clicking
        quickly along the strip cannot pile up preloads either, because they
        only start once the photo being looked at has arrived.
        """
        if path in self._decoding.values():
            return
        token = self._submit(lambda: _decode(path), None, self._on_decoded, path)
        self._decoding[token] = path

    def _on_camera_preview(
        self, path: Path, found: tuple[NDArray[np.uint8], tuple[int, int]] | None
    ) -> None:
        """Show the camera's own preview while the real decode is running.

        It goes out with the size the photo has in full, read from the file
        alongside it, so the zoom is measured against the photo from the
        start rather than jumping when the decode arrives (PLAN.md 3.3).
        """
        if found is None or path != self.current or path in self._decoded:
            return
        image, full_size = found
        self._emit_whole(path, image, full_size, provisional=True)

    def _kept_whole(self, kept: _KeptPreview | None, path: Path) -> bool:
        """Return whether a kept preview of the whole photo can be shown now.

        The developed photo and the photo as it was found are kept so that the
        before and after key is instant the second time round. Zoomed in they
        are the wrong pixels: what belongs on screen is one window of the
        photo, developed at the resolution the screen can show. So are they
        once the frame has changed: entering or leaving the crop tab is what
        does that, and a kept preview of the other tab's frame would be shown
        at the wrong size (PLAN.md 3.2).
        """
        if kept is None or kept[0] != path or self._zoomed_in():
            return False
        return kept[2] == self._frame_of(path)

    def _emit_whole(
        self,
        path: Path,
        image: NDArray[np.uint8],
        frame: tuple[int, int] | None,
        provisional: bool = False,
        draft: bool = False,
        before: bool = False,
    ) -> None:
        """Announce a preview of the whole photo, with the frame it fills.

        The frame is what the zoom is measured against, so every preview
        carries it: the one the photo was developed for rather than the one
        wanted now, which may already have moved on. Nothing is known about
        it until the photo is decoded, which is why the camera's own preview
        may go out without one.
        """
        self.preview_ready.emit(
            Preview(
                path=path,
                image=image,
                provisional=provisional,
                draft=draft,
                before=before,
                frame=frame,
            )
        )

    def _on_decoded(self, path: Path, decoded: DecodedPhoto) -> None:
        """Take a freshly decoded photo and ask for it to be developed.

        Only once the current photo is decoded do its neighbours start:
        decoding them alongside it would make the photo being waited for
        share the processor with two that nobody is looking at yet.
        """
        if path not in self._neighbourhood():
            return
        self._decoded[path] = decoded
        if path != self.current:
            return
        self._announce_adjustments(path, decoded)
        self._request_develop(path, decoded, draft=False)
        self._preload_neighbours()

    def _announce_adjustments(self, path: Path, decoded: DecodedPhoto) -> None:
        """Set the sliders up for a photo that has just been opened.

        The adjustments come from three places, in order: the history, if this
        photo has been edited since the program started; the parameter file
        beside it, which is how an edit survives a restart (E-03); and
        otherwise the photo as the camera left it.
        """
        self._shot_white_balance = pipeline.shot_white_balance(decoded)
        self._reference = EditState.for_photo(decoded.is_raw, *self._shot_white_balance)
        self._reference_geometry = _reference_geometry(decoded)

        remembered = self._history.current(path) if self._history.knows(path) else None
        edit = remembered if remembered is not None else self._stored_edit(path)
        self._state = edit.adjustments
        self._geometry = edit.geometry
        self._history.begin(path, edit)

        self.adjustments_ready.emit(self._state, self._shot_white_balance)
        self.lens_ready.emit(decoded.lens, decoded.is_raw)
        self.geometry_changed.emit(self._geometry)
        self._announce_history()
        self._request_faces(path, decoded)

    @property
    def filters(self) -> tuple[Look, ...]:
        """Return the film looks that can be put on a photo, in file order."""
        return tuple(self._filters.values())

    def _look_for(self, state: EditState) -> Look | None:
        """Return the film look *state* is wearing, or ``None`` for none."""
        return self._filters.get(state.filter) if state.filter else None

    def _as_installed(self, state: EditState) -> EditState:
        """Return *state* with a filter this copy does not have taken off.

        A parameter file can name a filter that is not here - written by a
        newer version, or carried from another machine. The photo is shown
        without it rather than refused, and the name is not kept, so that the
        next save does not pass on a filter nobody can render.
        """
        if state.filter and state.filter not in self._filters:
            logger.warning("The filter %r is not installed", state.filter)
            return state.with_filter(NO_FILTER)
        return state

    @property
    def reference_edit(self) -> Edit:
        """Return the photo as it was found: no adjustments, no crop (E-05)."""
        return Edit(self._reference, self._reference_geometry)

    def _stored_edit(self, path: Path) -> Edit:
        """Return what the parameter file of *path* says, or the reference (E-03)."""
        stored = sidecar.read(path, self._reference, self._store)
        if stored is None:
            return self.reference_edit
        self._carried[path] = stored.extra
        return stored.edit.with_adjustments(self._as_installed(stored.edit.adjustments))

    # -- what other parts of the program need to know about a photo -----

    @property
    def display_transform(self) -> object | None:
        """Return the conversion to the screen's colours (PLAN.md 6 step 13)."""
        return self._display_transform

    def remembered_edit(self, path: Path) -> Edit | None:
        """Return the edit this run's history holds for *path*, if any.

        Newer than the parameter file by up to the autosave delay, so the
        collage asks here first (PLAN.md 2.8). Called on the interface thread.
        """
        return self._history.current(path) if self._history.knows(path) else None

    def stored_edit_of(self, path: Path, decoded: DecodedPhoto) -> Edit:
        """Return what the parameter file of *path* says, or the photo untouched.

        Reads only files and the list of installed filters, so it may be
        called on a background thread, unlike the history.
        """
        reference = EditState.for_photo(decoded.is_raw, *pipeline.shot_white_balance(decoded))
        stored = sidecar.read(path, reference, self._store)
        if stored is None:
            return Edit(reference, _reference_geometry(decoded))
        return stored.edit.with_adjustments(self._as_installed(stored.edit.adjustments))

    def look_for(self, state: EditState) -> Look | None:
        """Return the film look *state* is wearing, or ``None`` for none."""
        return self._look_for(state)

    def _announce_history(self) -> None:
        """Say whether undo and redo have anything to do."""
        path = self.current
        if path is None:
            self.history_changed.emit(False, False)
            return
        self.history_changed.emit(self._history.can_undo(path), self._history.can_redo(path))

    # -- adjusting -------------------------------------------------------

    def set_adjustments(self, state: EditState, dragging: bool = False) -> None:
        """Apply new slider values to the photo being shown (A-09).

        While the slider is still moving the preview is developed at half the
        width, which is a quarter of the pixels, so that it keeps up with the
        hand; the full size one follows as soon as the slider is let go.

        A change that is finished - a slider let go, a number typed, a key
        pressed - becomes one step of history and starts the save timer. A
        change still under the hand does neither, which is what makes a whole
        drag cost a single undo (E-04).
        """
        self._state = state
        path = self.current
        decoded = self._decoded.get(path) if path is not None else None
        if path is None or decoded is None:
            return
        if not dragging:
            self._record(path, self.edit)
        self._request_develop(path, decoded, draft=dragging)

    def set_adjustment(self, name: str, value: float, dragging: bool = False) -> None:
        """Move one slider of the photo being shown, keeping everything else.

        The adjust tab sends only the slider it moved, and it is applied to
        the photo's own state here. Sending its whole state instead put back
        whatever the tab had last been told: a filter chosen on the filter tab
        since then was taken off again by the next slider moved.
        """
        self.set_adjustments(self._state.with_value(name, value), dragging)

    def set_lens(self, settings: LensSettings, dragging: bool = False) -> None:
        """Change the lens correction of the photo being shown (PLAN.md 2.2).

        It is part of the geometry, since it changes the frame the crop is
        measured against, and goes through the same path as a crop does.
        """
        self.set_geometry(replace(self._geometry, lens=settings), dragging)

    def set_geometry(self, geometry: Geometry, dragging: bool = False) -> None:
        """Apply a new crop, rotation or quarter turn (C-01 to C-06).

        Like the sliders, a change that is still under the hand only updates
        the preview; the finished one becomes a step of history and is saved.
        The end of a drag arrives with the value the preview already shows,
        and that is exactly the one that has to be recorded, so nothing is
        skipped here for being unchanged - the history itself decides whether
        a step was really taken.
        """
        self._geometry = geometry
        path = self.current
        decoded = self._decoded.get(path) if path is not None else None
        if path is None or decoded is None:
            return
        if not dragging:
            self._record(path, self.edit)
            # The lens correction moves the faces; dragging its sliders does
            # not wait for them, letting go looks again.
            self._request_faces(path, decoded)
        self._before = None  # it carries the old crop
        self._request_develop(path, decoded, draft=dragging)

    def _record(self, path: Path, edit: Edit) -> None:
        """Make *edit* a step of history and schedule the parameter file."""
        if self._history.push(path, edit):
            self._announce_history()
        self._schedule_save(path, edit)

    # -- copying and pasting (P-01 to P-03) ------------------------------

    @property
    def has_copied_adjustments(self) -> bool:
        """Return whether there are settings waiting to be pasted."""
        return self._clipboard is not None

    def copy_adjustments(self) -> bool:
        """Take the colour settings of the photo being shown (P-01)."""
        if self.current is None:
            return False
        self._clipboard = self._state
        self.clipboard_changed.emit(True)
        self.announcement.emit(tr("toast.settings_copied"))
        return True

    def paste_adjustments(self, photos: list[Path] | None = None) -> int:
        """Put the copied settings on *photos*, or on the one being shown (P-02).

        The photo on screen goes through the ordinary path, so the paste is
        one step of history like any other change (P-05). The rest are not
        open - nothing is known about them beyond their parameter file - so
        their settings are changed in that file directly.
        """
        copied = self._clipboard
        current = self.current
        if copied is None or current is None:
            return 0

        targets = list(photos) if photos else [current]
        if current not in targets:
            targets.append(current)

        pasted = 0
        for photo in targets:
            if photo == current:
                self.set_adjustments(self._pasted_onto(copied, self._state))
                pasted += 1
                continue
            if self._paste_into_file(photo, copied):
                pasted += 1

        self.announcement.emit(tr("toast.settings_pasted", count=pasted))
        return pasted

    @staticmethod
    def _pasted_onto(copied: EditState, state: EditState) -> EditState:
        """Return *state* with the copied settings on it.

        The temperature and tint of a RAW photo are in kelvin and those of a
        JPEG are a nudge from where the camera left it, so the pair is only
        carried across between photos of the same kind (PLAN.md 7.2). The
        detail group is the same: a RAW rests with some sharpening and a JPEG
        with none, so a RAW's resting values would sharpen a JPEG twice.
        """
        changed = state
        for name in NUMERIC_NAMES:
            if name in _KIND_NAMES and copied.is_raw != state.is_raw:
                continue
            changed = changed.with_value(name, getattr(copied, name))
        return changed.with_filter(copied.filter)

    def _paste_into_file(self, photo: Path, copied: EditState) -> bool:
        """Change the stored adjustments of a photo that is not open."""
        same_kind = decoder.is_raw(photo) == copied.is_raw
        values: dict[str, float | str] = {
            name: float(getattr(copied, name))
            for name in NUMERIC_NAMES
            if same_kind or name not in _KIND_NAMES
        }
        values["filter"] = copied.filter
        result = _update_sidecar(photo, values, self._application, self._store)
        if result.failed:
            return False
        self._note_edited(photo, result)
        return True

    # -- presets (P-04 to P-06) ------------------------------------------

    @property
    def presets(self) -> list[Preset]:
        """Return the saved presets, in the order they are shown."""
        return list(self._presets)

    def load_presets(self) -> None:
        """Read the presets folder and announce what is in it."""
        self._presets = presets.load_all(self._presets_dir)
        self.presets_changed.emit(self.presets)

    def save_preset(self, name: str) -> Preset | None:
        """Save what the photo is set to now as a preset (P-04)."""
        if self.current is None:
            return None
        preset = presets.create(name, self._state, self._reference)
        if not preset.name:
            return None
        saved = presets.save(preset, self._presets_dir)
        if saved is None:
            self.error.emit(tr("error.cannot_save_preset", name=preset.name))
            return None
        self.load_presets()
        self.announcement.emit(tr("toast.preset_saved", name=saved.name))
        return saved

    def rename_preset(self, preset: Preset, name: str) -> Preset | None:
        """Give a preset another name (P-06)."""
        renamed = presets.rename(preset, name, self._presets_dir)
        if renamed is None:
            return None
        self.load_presets()
        return renamed

    def delete_preset(self, preset: Preset) -> bool:
        """Remove a preset (P-06)."""
        removed = presets.delete(preset, self._presets_dir)
        self.load_presets()
        return removed

    def apply_preset(self, preset: Preset) -> None:
        """Put a preset's adjustments on the photo being shown (P-05).

        Through the ordinary path, so it is one step of history and is saved
        like any other change.
        """
        if self.current is None:
            return
        self.set_adjustments(self._as_installed(preset.apply_to(self._state)))
        self.state_restored.emit(self._state)
        self.announcement.emit(tr("toast.preset_applied", name=preset.name))

    # -- the crop tab (PLAN.md 3.2) --------------------------------------

    def set_cropping(self, cropping: bool) -> None:
        """Open or leave the crop tab.

        While it is open the whole rotated photo is shown so that the box can
        be dragged over it; leaving applies the crop, which is what switching
        to another tab usually does in photo editors and what PLAN.md 3.2 asks for.
        """
        if cropping == self._cropping:
            return
        self._cropping = cropping
        if cropping:
            self._geometry_before_crop = self._geometry
            if self._geometry.crop.is_whole_frame:
                # A photo that has never been cropped starts with the largest
                # box its shape allows, rather than with nothing to drag.
                self._geometry = replace(
                    self._geometry, crop=geometry.start_crop(self.photo_size, self._geometry)
                )
                self.geometry_changed.emit(self._geometry)
        self._before = None
        self._develop_current()

    def cancel_crop(self) -> None:
        """Leave the crop tab, putting the geometry back (PLAN.md 3.2, Esc)."""
        geometry_before = self._geometry_before_crop
        self._cropping = False
        if geometry_before != self._geometry:
            self._geometry = geometry_before
            self.geometry_changed.emit(self._geometry)
            path = self.current
            if path is not None:
                self._record(path, self.edit)
        self._before = None
        self._develop_current()

    def reset_crop(self) -> None:
        """Put the crop and rotation back to the whole photo (PLAN.md 3.2).

        The lens correction, the mended spots, the portrait's skin range and
        the liquify strokes are not the crop tab's and are left as they are.
        """
        fresh = Geometry(
            lens=self._geometry.lens,
            spots=self._geometry.spots,
            skin=self._geometry.skin,
            liquify=self._geometry.liquify,
        )
        if self._cropping:
            fresh = replace(fresh, crop=geometry.start_crop(self.photo_size, fresh))
        self.set_geometry(fresh)
        self.geometry_changed.emit(self._geometry)

    # -- spot removal (PLAN.md 2.2, A-32, and 6 step 1.6) -----------------

    def _spot_frame(self) -> tuple[tuple[int, int], Geometry, bool]:
        """Return the photo's size, its geometry and whether the crop is shown."""
        return self.photo_size, self._geometry, not self._cropping

    def spots_on_screen(
        self,
    ) -> list[tuple[list[tuple[float, float]], list[tuple[float, float]], float]]:
        """Return each stroke as it is shown: its path, its source's path and radius.

        The points are in fractions of the frame on screen (turned, rotated
        and, outside the crop tab, cropped), and the radius is a fraction of
        that frame's long edge.
        """
        size, shape, crop = self._spot_frame()
        scale = geometry.shown_scale(size, shape, crop)
        return [
            (
                [geometry.shown_point(point, size, shape, crop) for point in spot.points],
                [geometry.shown_point(point, size, shape, crop) for point in spot.source_points],
                spot.radius / scale,
            )
            for spot in shape.spots
        ]

    def shown_radius(self, radius: float) -> float:
        """Return a brush radius as a fraction of the long edge of the frame on screen."""
        size, shape, crop = self._spot_frame()
        return radius / geometry.shown_scale(size, shape, crop)

    def add_spot(self, points: Sequence[tuple[float, float]], radius: float) -> None:
        """Mend what the brush painted over, or the dot it clicked (A-32).

        *points* is the path the pointer took, in fractions of the frame on
        screen, and *radius* the brush's, a fraction of the photo's long
        edge. Where the texture comes from is worked out in the background on
        the proxy, mended as it is so far, and the stroke is added when it
        arrives, as one step of history.
        """
        path = self.current
        decoded = self._decoded.get(path) if path is not None else None
        if path is None or decoded is None or not points:
            return
        size, shape, crop = self._spot_frame()
        on_photo = [geometry.source_point(point, size, shape, crop) for point in points]
        on_photo = [
            (min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)) for x, y in on_photo
        ]
        radius = healing.clamp_radius(radius)
        stroke = healing.thinned(on_photo, radius, size)
        corrected = self._corrected_lens

        def find(photo: DecodedPhoto = decoded, frame: Geometry = shape) -> tuple[float, float]:
            """Choose the source, on whichever thread takes this on."""
            mended = healing.heal(corrected(path, photo, frame), frame.spots)
            return healing.find_offset(mended.image, stroke, radius)

        self._submit(
            find,
            _GROUP_SPOT_SOURCE,
            partial(self._on_spot_source, points=stroke, radius=radius),
            path,
        )

    def _on_spot_source(
        self,
        path: Path,
        offset: tuple[float, float],
        points: tuple[tuple[float, float], ...],
        radius: float,
    ) -> None:
        """Add the stroke whose source has been found."""
        if path != self.current:
            return
        spot = Spot(points=points, radius=radius, offset=offset)
        self._set_spots(self._geometry.spots + (spot,))
        self.spot_added.emit(len(self._geometry.spots) - 1)

    def move_spot(
        self, index: int, point: tuple[float, float], source: bool, dragging: bool = False
    ) -> None:
        """Move one stroke, or only where it takes its texture from.

        *point* is where the stroke's first point, or its source's, goes, in
        fractions of the frame on screen. Moving the stroke takes its source
        along at the same offset, the way the other editors do; moving the
        source leaves the stroke where it is.
        """
        spots = self._geometry.spots
        if not 0 <= index < len(spots):
            return
        size, shape, crop = self._spot_frame()
        where = geometry.source_point(point, size, shape, crop)
        spot = spots[index]
        moved = spot.with_source(where) if source else spot.moved_to(where)
        self._set_spots(spots[:index] + (moved,) + spots[index + 1 :], dragging)

    def resize_spot(self, index: int, radius: float, dragging: bool = False) -> None:
        """Change the size of one spot; *radius* is a fraction of the long edge."""
        spots = self._geometry.spots
        if not 0 <= index < len(spots):
            return
        resized = spots[index].with_radius(radius)
        if resized == spots[index] and dragging:
            return
        self._set_spots(spots[:index] + (resized,) + spots[index + 1 :], dragging)

    def remove_spot(self, index: int) -> None:
        """Stop mending one spot."""
        spots = self._geometry.spots
        if 0 <= index < len(spots):
            self._set_spots(spots[:index] + spots[index + 1 :])

    def clear_spots(self) -> None:
        """Stop mending every spot, as one step of history."""
        if self._geometry.spots:
            self._set_spots(())

    def _set_spots(self, spots: tuple[Spot, ...], dragging: bool = False) -> None:
        """Give the photo these spots, the way any change of geometry is given."""
        self.set_geometry(replace(self._geometry, spots=spots), dragging)
        self.geometry_changed.emit(self._geometry)

    # -- the portrait range (PLAN.md 2.10 and 3.15) -----------------------

    @property
    def skin(self) -> SkinRange:
        """Return the skin range of the photo now open."""
        return self._geometry.skin

    def add_skin_stroke(
        self,
        points: Sequence[tuple[float, float]],
        radius: float,
        adding: bool,
        snap: bool,
    ) -> None:
        """Add to the skin the portrait sliders work on, or take from it.

        *points* is the path the pointer took, in fractions of the frame on
        screen, and *radius* the brush's, a fraction of the photo's long
        edge, as for :meth:`add_spot`. One stroke is one step of history.
        """
        if self.current is None or not points:
            return
        size, shape, crop = self._spot_frame()
        on_photo = [geometry.source_point(point, size, shape, crop) for point in points]
        on_photo = [(min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)) for x, y in on_photo]
        radius = skin_range.clamp_radius(radius)
        stroke = SkinStroke(
            points=skin_range.thinned(on_photo, radius, size),
            radius=radius,
            adding=adding,
            snap=snap,
        )
        self._set_skin(self._geometry.skin.with_stroke(stroke))

    def set_skin_auto(self, auto: bool) -> None:
        """Say whether the skin beside the faces is found by itself."""
        if self.current is not None and self._geometry.skin.auto != auto:
            self._set_skin(replace(self._geometry.skin, auto=auto))

    def reset_skin(self) -> None:
        """Go back to the skin found by itself, every stroke gone, as one step."""
        if self.current is not None and not self._geometry.skin.is_default:
            self._set_skin(skin_range.DEFAULT_RANGE)

    def _set_skin(self, skin: SkinRange) -> None:
        """Give the photo this skin range, the way any change of geometry is given."""
        self.set_geometry(replace(self._geometry, skin=skin))
        self.geometry_changed.emit(self._geometry)

    def request_skin_shown(self) -> None:
        """Work out the skin on screen in the background, for :attr:`skin_shown`.

        The same small field the preview whitens with, kept with it, so
        showing it after a stroke costs nothing the preview has not paid.
        """
        path = self.current
        decoded = self._decoded.get(path) if path is not None else None
        if path is None or decoded is None:
            self.skin_shown.emit(None)
            return
        long_edge = self._preview_long_edge
        sized = self._sized_photo(path, decoded, long_edge)
        shape = self._geometry
        whole = self._cropping
        key = (path, long_edge, geometry.frame_key(shape), whole, (path, shape.lens) in self._faces)
        skin = geometry.framed_skin(shape, sized.size, crop=not whole)
        corrected = self._corrected_lens
        kept_body = self._body

        def shown(photo: DecodedPhoto = sized) -> NDArray[np.uint8] | None:
            """Work the skin out, on whichever thread takes this on."""
            framed = self._framed.get(
                key,
                lambda: geometry.apply(corrected(path, photo, shape), shape, crop=not whole),
            )
            assert framed is not None  # the maker always returns a photo
            framed = replace(framed, skin=skin)
            if skin_range.reaches_nothing(framed.faces, framed.skin):
                return None
            field = kept_body.get((key, skin), lambda: pipeline.body_field(framed))
            if field is None:
                return None
            combined = skin_range.combined(field)
            return np.round(np.clip(combined, 0.0, 1.0) * 255.0).astype(np.uint8)

        self._submit(shown, _GROUP_SKIN_SHOWN, self._on_skin_shown, path)

    def _on_skin_shown(self, path: Path, skin: NDArray[np.uint8] | None) -> None:
        """Hand the skin on screen over, if it is still the photo's."""
        if path == self.current:
            self.skin_shown.emit(skin)

    # -- undo, redo and reset --------------------------------------------

    def undo(self) -> None:
        """Go back one step on the photo being shown (E-04)."""
        self._restore(self._history.undo(self.current) if self.current else None)

    def redo(self) -> None:
        """Go forward one step on the photo being shown (E-04)."""
        self._restore(self._history.redo(self.current) if self.current else None)

    def reset(self) -> None:
        """Put the adjust tab's sliders back to where the photo was found (E-05).

        Only that tab's, whose button this is: the filter, the portrait, the
        lens correction and the crop each belong to another tab and stay as
        they are (PLAN.md 3.1). Recorded as a step of its own, so that it can
        be undone like anything else.
        """
        path = self.current
        if path is None:
            return
        state = self._state
        for name in RESET_NAMES:
            state = state.with_value(name, getattr(self._reference, name))
        self._state = state
        self._record(path, self.edit)
        self.state_restored.emit(self._state)
        self._develop_current()

    def pick_white_balance(self, point: tuple[float, float]) -> None:
        """Balance the photo so that the spot at *point* turns grey (A-24).

        *point* is where the eyedropper was clicked, in fractions of the photo
        as the adjust tab shows it: turned, rotated and cropped. The spot is
        read off the preview proxy in the background, and what comes back
        moves the two white balance sliders as one step of history, the way
        any finished slider change does.
        """
        path = self.current
        decoded = self._decoded.get(path) if path is not None else None
        if path is None or decoded is None:
            return

        def pick(
            photo: DecodedPhoto = decoded,
            shape: Geometry = self._geometry,
            where: tuple[float, float] = point,
        ) -> pipeline.WhiteBalancePick:
            """Read the spot, on whichever thread takes this on."""
            return pipeline.pick_white_balance(geometry.apply(photo, shape), where)

        self._submit(pick, _GROUP_PICK, self._on_white_balance_picked, path)

    def auto_light(self) -> None:
        """Set the light group's sliders from the photo itself (PLAN.md 3.1, section 6).

        Worked out in the background on a small copy of the photo as it is
        framed and cropped now, by :func:`core.auto_light.auto_light`. What
        comes back moves the five sliders as one step of history, the way the
        eyedropper's answer does.
        """
        path = self.current
        decoded = self._decoded.get(path) if path is not None else None
        if path is None or decoded is None:
            return
        values = self._state
        shape = self._geometry
        look = self._look_for(values)
        with_faces = self._with_faces

        def work() -> EditState | None:
            """Measure and search, on whichever thread takes this on."""
            try:
                sized = decoder.resize_photo(decoded, pipeline.AUTO_LONG_EDGE)
                corrected = with_faces(path, lens.correct(sized, shape.lens), shape)
                framed = geometry.apply(corrected, shape, crop=True)
                return find_auto_light(framed, values, look)
            except Exception:  # noqa: BLE001 - the photo stays as it was
                logger.exception("The automatic light of %s could not be worked out", path.name)
                return None

        self._submit(work, _GROUP_AUTO_LIGHT, self._on_auto_light, path)

    def _on_auto_light(self, path: Path, found: EditState | None) -> None:
        """Put the five sliders found on the photo, as it is by now."""
        if path != self.current:
            return
        if found is None:
            self.announcement.emit(tr("toast.auto_light_failed"))
            return
        # Only the light sliders are taken from the answer: anything moved
        # while it was being worked out stays as it was moved.
        state = self._state
        for name in AUTO_NAMES:
            state = state.with_value(name, getattr(found, name))
        self.announcement.emit(tr("toast.auto_light"))
        if state == self._state:
            return
        self.state_restored.emit(state)
        self.set_adjustments(state)

    def _on_white_balance_picked(self, path: Path, picked: pipeline.WhiteBalancePick) -> None:
        if path != self.current:
            return
        if picked.problem is not None:
            self.announcement.emit(tr(f"toast.pick_{picked.problem}"))
            self.white_balance_refused.emit()
            return
        state = self._state.with_value("temperature", picked.temperature).with_value(
            "tint", picked.tint
        )
        if state == self._state:
            return
        self.state_restored.emit(state)
        self.set_adjustments(state)

    def _restore(self, edit: Edit | None) -> None:
        """Show *edit* again after an undo or a redo."""
        path = self.current
        if edit is None or path is None:
            return
        geometry_changed = edit.geometry != self._geometry
        self._state = edit.adjustments
        self._geometry = edit.geometry
        self._announce_history()
        self.state_restored.emit(edit.adjustments)
        if geometry_changed:
            self.geometry_changed.emit(edit.geometry)
            self._before = None
        self._schedule_save(path, edit)
        self._develop_current()

    def _develop_current(self) -> None:
        """Develop the photo being shown at full preview size."""
        path = self.current
        decoded = self._decoded.get(path) if path is not None else None
        if path is not None and decoded is not None:
            self._request_develop(path, decoded, draft=False)

    @property
    def can_undo(self) -> bool:
        """Return whether the photo being shown has something to undo."""
        path = self.current
        return path is not None and self._history.can_undo(path)

    @property
    def can_redo(self) -> bool:
        """Return whether the photo being shown has something to redo."""
        path = self.current
        return path is not None and self._history.can_redo(path)

    # -- before and after (E-06) -----------------------------------------

    @property
    def comparing(self) -> bool:
        """Return whether the photo as it was found is being shown."""
        return self._comparing

    def set_comparing(self, comparing: bool) -> None:
        """Show the photo as it was found while the key is held (E-06).

        What "as it was found" means is the state the photo opens in: the
        camera's own white balance and no adjustments. Crop and rotation will
        still apply when they arrive in stage 4, as PLAN.md section 3.3 asks.
        """
        if comparing == self._comparing:
            return
        self._comparing = comparing
        path = self.current
        decoded = self._decoded.get(path) if path is not None else None
        if path is None or decoded is None:
            self._comparing = False
            return

        if not comparing:
            if self._kept_whole(self._developed, path):
                assert self._developed is not None  # kept for this photo
                self._emit_whole(path, self._developed[1], self._developed[2])
            else:
                self._request_develop(path, decoded, draft=False)
            return

        if self._kept_whole(self._before, path):
            assert self._before is not None  # kept for this photo
            self._emit_whole(path, self._before[1], self._before[2], before=True)
            return
        # The crop still applies; only the colour goes back (PLAN.md 3.3).
        self._request_develop(
            path, decoded, draft=False, state=self._reference, purpose=_BEFORE
        )

    def _request_develop(
        self,
        path: Path,
        decoded: DecodedPhoto,
        draft: bool,
        state: EditState | None = None,
        purpose: str = _EDITED,
    ) -> None:
        """Develop the photo in the background at the size now wanted.

        *state* is the current one unless another is given, which is how the
        before and after view asks for the photo as it was found.

        Zoomed in past what the whole preview holds, it is only the part on
        screen that is developed, by way of :meth:`_request_window`.
        """
        if not draft and state is None:
            self._request_look_thumbnails()
        if self._zoomed_in():
            if not draft:
                # The photo underneath is what panning uncovers, so it is out
                # of date as soon as the adjustments move. It is developed
                # after this window rather than with it, to keep off the
                # thread that is making what the user is looking at.
                self._backdrop_stale = True
            self._request_window(path, decoded, draft, state, purpose)
            return
        self._showing_window = False
        before = state is not None and purpose == _BEFORE
        long_edge = self._preview_long_edge
        if draft:
            long_edge = max(
                DRAFT_MINIMUM_LONG_EDGE, int(self._preview_long_edge * DRAFT_FRACTION)
            )
        if before:
            handler = self._on_before_developed
        elif draft:
            handler = self._on_developed_draft
        else:
            handler = self._on_developed
        self._submit_whole(
            path,
            decoded,
            self._state if state is None else state,
            long_edge,
            _GROUP_PREVIEW,
            handler,
            self._geometry_for(purpose),
        )

    def _geometry_for(self, purpose: str) -> Geometry:
        """Return the geometry a preview for *purpose* is framed with.

        The photo as it was found keeps its crop (PLAN.md 3.3) but not its
        mended spots or its liquify strokes: comparing is how the mending
        and the reshaping are checked (A-32, L-08).
        """
        if purpose == _BEFORE and (self._geometry.spots or self._geometry.liquify):
            return replace(self._geometry, spots=(), liquify=())
        return self._geometry

    def _submit_whole(
        self,
        path: Path,
        decoded: DecodedPhoto,
        values: EditState,
        long_edge: int,
        group: str,
        handler: Callable[..., None],
        shape: Geometry | None = None,
    ) -> None:
        """Develop all of *decoded* at *long_edge* and hand it to *handler*.

        The size the photo fills on screen is worked out here, where the
        request is made, and travels with it to the handler. Working it out
        when the outcome arrives instead would read the crop tab as it is by
        then: a preview still being developed when the tab is switched would
        be handed the other tab's frame, and the photo would be scaled to a
        frame it does not fill (PLAN.md 3.2, 3.3).
        """
        sized = self._sized_photo(path, decoded, long_edge)
        transform = self._display_transform
        frame = self._geometry if shape is None else shape
        whole = self._cropping
        key = (path, long_edge, geometry.frame_key(frame), whole, (path, frame.lens) in self._faces)
        skin = geometry.framed_skin(frame, sized.size, crop=not whole)
        on_screen = self._frame_of(path)
        detailed = self._detailed
        kept_grain = self._grain
        kept_body = self._body
        corrected = self._corrected_lens

        def develop_preview(
            photo: DecodedPhoto = sized,
            values: EditState = values,
            screen: object = transform,
            shape: Geometry = frame,
            cut: bool = whole,
            token: object = key,
            look: Look | None = self._look_for(values),
            skin: SkinRange | None = skin,
        ) -> NDArray[np.uint8]:
            """Develop the whole photo, on whichever thread takes this on."""
            framed = self._framed.get(
                token,
                lambda: geometry.apply(corrected(path, photo, shape), shape, crop=not cut),
            )
            assert framed is not None  # the maker always returns a photo
            # Kept from one stroke of the portrait brush to the next; the
            # newest skin range is put on it here.
            framed = replace(framed, skin=skin)
            sharpened = detailed.get(
                (token, pipeline.detail_key(values)),
                lambda: pipeline.with_detail(framed, values),
            )
            assert sharpened is not None  # the maker always returns a photo
            sharpened = replace(sharpened, skin=skin)
            grain = (
                kept_grain.get(token, lambda: pipeline.grain_field(framed))
                if pipeline.has_grain(values)
                else None
            )
            body = (
                kept_body.get((token, skin), lambda: pipeline.body_field(framed))
                if pipeline.needs_body(values, framed)
                else None
            )
            return cm.apply_display_transform(
                pipeline.develop(sharpened, values, look=look, grain=grain, body=body), screen
            )

        self._submit(develop_preview, group, partial(handler, frame=on_screen), path)

    # -- the filter tab's thumbnails --------------------------------------

    def set_look_thumbnail_side(self, side: int) -> None:
        """Say how many pixels across the filter tab's thumbnails are shown.

        Zero means the tab is out of sight: nothing is developed for it then,
        and the thumbnails catch up the moment it comes back.
        """
        self._look_thumbnail_side = max(0, int(side))
        self._request_look_thumbnails()

    def _request_look_thumbnails(self) -> None:
        """Develop the photo now open once for every film look, in the background.

        Each is the photo with every adjustment and the crop it has now,
        wearing one look at the strength the slider shows, which is exactly
        what clicking that look would give. It goes through the same
        pipeline as the preview and the export, only smaller (PLAN.md 5).
        """
        side = self._look_thumbnail_side
        path = self.current
        decoded = self._decoded.get(path) if path is not None else None
        if not side or path is None or decoded is None or not self._filters:
            return
        values = self._state
        shape = self._geometry
        transform = self._display_transform
        known_faces = (path, shape.lens) in self._faces
        key = (path, values, shape, side, transform, known_faces)
        if key == self._look_thumbnails_key:
            return
        self._look_thumbnails_key = key
        looks = dict(self._filters)
        long_edge = min(decoder.PROXY_LONG_EDGE, side * LOOK_THUMBNAIL_REACH)
        with_faces = self._with_faces

        def develop_thumbnails() -> dict[str, NDArray[np.uint8]] | None:
            """Develop every look, on whichever thread takes this on."""
            try:
                sized = decoder.resize_photo(decoded, long_edge)
                corrected = with_faces(path, lens.correct(sized, shape.lens), shape)
                framed = geometry.apply(corrected, shape, crop=True)
                detailed = pipeline.with_detail(framed, values)
                made: dict[str, NDArray[np.uint8]] = {}
                for identifier in (NO_FILTER, *looks):
                    image = pipeline.develop(
                        detailed, values.with_filter(identifier), look=looks.get(identifier)
                    )
                    made[identifier] = _centre_square(
                        cm.apply_display_transform(image, transform)
                    )
                return made
            except Exception:  # noqa: BLE001 - a thumbnail must never trouble the user
                logger.exception("The filter thumbnails of %s could not be made", path.name)
                return None

        self._submit(
            develop_thumbnails, _GROUP_LOOK_THUMBNAILS, self._on_look_thumbnails, path
        )

    def _on_look_thumbnails(
        self, path: Path, thumbnails: dict[str, NDArray[np.uint8]] | None
    ) -> None:
        if path != self.current:
            return
        if thumbnails is None:
            # Asking again for the same thing is allowed after a failure.
            self._look_thumbnails_key = None
            return
        self.look_thumbnails_ready.emit(thumbnails)

    def _sized_photo(self, path: Path, decoded: DecodedPhoto, long_edge: int) -> DecodedPhoto:
        """Return *decoded* at *long_edge*, keeping the copy for next time."""
        key = (path, long_edge)
        sized = self._sized.get(key)
        if sized is None:
            sized = decoder.resize_photo(decoded, long_edge)
            # Only the sizes of the photo on screen are worth keeping.
            self._sized = {
                existing: photo
                for existing, photo in self._sized.items()
                if existing[0] == path
            }
            self._sized[key] = sized
        return sized

    def _on_developed(
        self, path: Path, image: NDArray[np.uint8], frame: tuple[int, int] | None
    ) -> None:
        if path != self.current:
            return
        self._developed = (path, image, frame)
        if not self._comparing:
            self._emit_whole(path, image, frame)

    def _on_developed_draft(
        self, path: Path, image: NDArray[np.uint8], frame: tuple[int, int] | None
    ) -> None:
        if path == self.current and not self._comparing:
            self._emit_whole(path, image, frame, draft=True)

    def _on_before_developed(
        self, path: Path, image: NDArray[np.uint8], frame: tuple[int, int] | None
    ) -> None:
        """Show the photo as it was found, if the key is still held down."""
        if path != self.current:
            return
        self._before = (path, image, frame)
        if self._comparing:
            self._emit_whole(path, image, frame, before=True)

    def _preload_neighbours(self) -> None:
        """Decode the photos either side, so moving on feels instant."""
        if not self._settings.preload_neighbours:
            return
        for offset in (1, -1):
            index = self._index + offset
            if 0 <= index < len(self._photos):
                path = self._photos[index]
                if path not in self._decoded:
                    self._start_decode(path)

    def _neighbourhood(self) -> set[Path]:
        """Return the photos worth keeping decoded: this one and its two neighbours."""
        return {
            self._photos[index]
            for index in range(self._index - 1, self._index + 2)
            if 0 <= index < len(self._photos)
        }

    def _forget_sized_copies(self) -> None:
        """Drop the resized copies of photos that are no longer on screen."""
        keep = self._neighbourhood()
        self._sized = {key: photo for key, photo in self._sized.items() if key[0] in keep}

    def _forget_distant_photos(self) -> None:
        """Drop decoded photos that are no longer next to the current one.

        A proxy is a few tens of megabytes, so only three are ever held
        (PLAN.md section 4).
        """
        keep = self._neighbourhood()
        for path in list(self._decoded):
            if path not in keep:
                del self._decoded[path]
        for key in list(self._faces):
            if key[0] not in keep:
                del self._faces[key]
        if self._full is not None and self._full[0] != self.current:
            self._forget_full_size()
        self._forget_sized_copies()

    # -- the info tab ----------------------------------------------------

    def _request_metadata(self, path: Path) -> None:
        self._info = None
        self._submit(lambda: metadata.read(path), _GROUP_METADATA, self._on_metadata, path)

    def _on_metadata(self, path: Path, info: PhotoMetadata) -> None:
        if path == self.current:
            self._info = info
            self.metadata_ready.emit(info)

    # -- exporting (PLAN.md 2.6) -----------------------------------------

    @property
    def exporting(self) -> bool:
        """Return whether an export is running (X-07)."""
        return self._exporting

    @property
    def full_photo_size(self) -> tuple[int, int]:
        """Return the size of the photo now open, at its own resolution.

        The preview works on a proxy, but the export dialog has to say what
        the exported file will measure, which is the real thing. It comes
        from the photo's own details; until those have been read, the proxy's
        shape is the best that can be said.
        """
        info = self._info
        if info is not None and info.width and info.height:
            return info.width, info.height
        return self.photo_size

    def export_output_size(self, options: ExportOptions) -> tuple[int, int]:
        """Return the size the exported file would have (PLAN.md 3.11)."""
        return exporter.output_size(self.full_photo_size, self._geometry, options)

    def export_will_copy(self, options: ExportOptions) -> bool:
        """Return whether the original file would be copied as it is (X-08)."""
        path = self.current
        if path is None:
            return False
        return exporter.can_copy_original(
            path, self._state, self._geometry, options, self.faces_of_current()
        )

    def export(self, target: Path, options: ExportOptions) -> bool:
        """Write the photo now open to *target* in the background (X-07).

        Returns whether the export was started: one runs at a time, which is
        what the progress on the toolbar button reports.
        """
        path = self.current
        if path is None or self._exporting:
            return False
        self._exporting = True
        self.export_progress.emit(0.0)
        state = self._state
        frame = self._geometry
        found = self.faces_of_current()
        self._submit(
            lambda source=path, where=target, values=state, shape=frame: self._run_export(
                source, where, values, shape, options, found
            ),
            None,
            self._on_exported,
            path,
        )
        return True

    def _run_export(
        self,
        source: Path,
        target: Path,
        state: EditState,
        frame: Geometry,
        options: ExportOptions,
        found: tuple[Face, ...] | None = None,
    ) -> exporter.ExportResult | None:
        """Write the file on a background thread; never raise from there.

        Returning ``None`` rather than letting the error out is what keeps
        the button's progress from being left half way: the result arrives
        either way (PLAN.md section 10 rule 11).
        """
        try:
            return exporter.export(
                source,
                target,
                state,
                frame,
                options,
                progress=self.export_progress.emit,
                look=self._look_for(state),
                faces=found,
            )
        except Exception:  # noqa: BLE001 - the user is told, the log has the rest
            logger.exception("Exporting %s failed", source)
            return None

    def _on_exported(self, path: Path, result: exporter.ExportResult | None) -> None:
        """Say how the export went, whichever photo is now on screen."""
        self._exporting = False
        if result is None:
            self.export_progress.emit(0.0)
            self.export_failed.emit(tr("error.cannot_export", name=path.name))
            return
        self.export_finished.emit(result)

    def request_export_estimate(self, options: ExportOptions) -> None:
        """Work out roughly how large the exported file would be (PLAN.md 3.11).

        The preview proxy is compressed with the settings asked for, which
        takes long enough to be worth doing off the interface thread; only
        the newest answer is of any interest.
        """
        path = self.current
        developed = self._developed
        if path is None or developed is None or developed[0] != path:
            self.export_estimate_ready.emit(0)
            return
        if self.export_will_copy(options):
            try:
                self.export_estimate_ready.emit(path.stat().st_size)
            except OSError:
                self.export_estimate_ready.emit(0)
            return
        width, height = self.export_output_size(options)
        preview = developed[1]
        self._submit(
            lambda image=preview, pixels=width * height: exporter.estimate_size(
                image, pixels, options
            ),
            _GROUP_ESTIMATE,
            self._on_export_estimate,
            path,
        )

    def _on_export_estimate(self, _path: Path, size: int) -> None:
        self.export_estimate_ready.emit(int(size))

    # -- the watermark (PLAN.md 2.9, 3.14) -------------------------------

    def watermark_picture(
        self, spec: WatermarkSpec, frame: tuple[int, int]
    ) -> WatermarkPicture | None:
        """Draw *spec* for the photo on screen, whose full size is *frame*.

        Drawn for the frame shrunk to the preview's size, as the photo on
        screen is, rather than at full size: a few milliseconds, so it is
        done at once, and the overlay only moves the result while it is
        dragged. ``None`` when there is nothing to draw.
        """
        if min(frame) <= 0:
            return None
        if self._watermark_fonts is None:
            self._watermark_fonts = watermark.load_fonts()
        scale = min(1.0, decoder.PROXY_LONG_EDGE / max(frame))
        shown = (max(1, round(frame[0] * scale)), max(1, round(frame[1] * scale)))
        sprite = watermark.rasterise(spec, shown, self._watermark_fonts)
        if sprite is None:
            return None
        return WatermarkPicture(
            rgba=watermark.to_rgba(sprite),
            scale=shown[0] / frame[0],
            missing_glyphs=sprite.missing_glyphs,
        )


    # -- thumbnails ------------------------------------------------------

    def request_thumbnails(self, paths: list[Path]) -> None:
        """Make the thumbnails of *paths*, nearest to the current photo first."""
        for path in paths:
            self._submit(lambda p=path: self._thumbnail(p), None, self._on_thumbnail, path)

    def _thumbnail(self, path: Path) -> NDArray[np.uint8] | None:
        """Return the thumbnail of *path*, from the cache when it is there."""
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        thumbnail = thumbnails.make(path)
        self._cache.put(path, thumbnail)
        return thumbnail

    def _on_thumbnail(self, path: Path, thumbnail: NDArray[np.uint8] | None) -> None:
        if thumbnail is not None:
            self.thumbnail_ready.emit(path, thumbnail)

    def clear_thumbnail_cache(self) -> None:
        """Empty the thumbnail cache on disk."""
        self._cache.clear()

    def thumbnail_cache_usage(self) -> int:
        """Return how much room the cached thumbnails take (PLAN.md 3.10)."""
        return self._cache.usage_bytes()

    def set_thumbnail_cache_limit(self, limit_bytes: int) -> None:
        """Change how much room the cache may take, trimming it at once."""
        self._cache.set_limit(limit_bytes)


    # -- the parameter file (E-02) ---------------------------------------

    def _schedule_save(self, path: Path, edit: Edit) -> None:
        """Write the parameters of *path* a second after the last change.

        A second of quiet is what PLAN.md E-02 asks for, so that dragging a
        slider does not write a file thirty times a second. Whatever is still
        waiting is written when the photo changes or the window closes.
        """
        self._unsaved[path] = edit
        self._save_timer.start()

    def _write_pending(self) -> None:
        """Write every parameter file that is waiting, in the background."""
        self._save_timer.stop()
        waiting = self._unsaved
        self._unsaved = {}
        for path, edit in waiting.items():
            arguments = self._save_arguments(path, edit)
            self._submit(
                lambda values=arguments: _save_sidecar(*values),
                _save_group(path),
                self._on_saved,
                path,
            )

    def flush(self) -> None:
        """Write anything still waiting at once, without the background.

        Used when the window is closing: the thread pool is about to be shut
        down, so a save handed to it might never run.
        """
        self._save_timer.stop()
        waiting = self._unsaved
        self._unsaved = {}
        for path, edit in waiting.items():
            self._on_saved(path, _save_sidecar(*self._save_arguments(path, edit)))

    def _save_arguments(
        self, path: Path, edit: Edit
    ) -> tuple[Path, Edit, Edit, sidecar.Application, Path, dict]:
        """Collect everything one save needs, while still on this thread."""
        return (
            path,
            edit,
            self.reference_edit,
            self._application,
            self._store,
            self._carried.get(path, {}),
        )

    def _on_saved(self, path: Path, result: sidecar.SaveResult) -> None:
        """React to a finished save: the filmstrip mark, or what went wrong."""
        if result.failed:
            self.error.emit(tr("error.cannot_save_parameters", name=path.name))
            return
        self._note_edited(path, result)

    def _note_edited(self, path: Path, result: sidecar.SaveResult) -> None:
        """Keep the filmstrip's mark in step with the parameter file."""
        edited = result.path is not None
        if edited == (path in self._edited):
            return
        if edited:
            self._edited.add(path)
        else:
            self._edited.discard(path)
        self.edited_changed.emit(path, edited)

    def is_edited(self, path: Path) -> bool:
        """Return whether *path* has adjustments stored (PLAN.md 3.1)."""
        return path in self._edited

    # -- running work ----------------------------------------------------

    def _submit(self, function, group: str | None, handler, path: Path) -> int:
        token = self._runner.submit(function, group)
        self._handlers[token] = (handler, path)
        return token

    def _on_completed(self, token: int, result: object) -> None:
        self._decoding.pop(token, None)
        entry = self._handlers.pop(token, None)
        if entry is None:
            return
        handler, path = entry
        handler(path, result)

    def _on_discarded(self, token: int) -> None:
        """Let go of a request that was replaced before anything came of it."""
        self._decoding.pop(token, None)
        self._handlers.pop(token, None)
        if token == self._full_token:
            self._full_token = None

    def _on_failed(self, token: int, message: str) -> None:
        self._decoding.pop(token, None)
        entry = self._handlers.pop(token, None)
        if token == self._full_token:
            # The proxy is on screen and stays there, so all that is lost is
            # the sharper window: there is nothing to trouble the user with.
            self._full_token = None
            logger.info("A photo could not be decoded at full size: %s", message)
            return
        if entry is None:
            return
        _handler, path = entry
        logger.info("Work on %s failed: %s", path, message)
        if path == self.current:
            self.error.emit(tr("error.cannot_open", name=path.name))

    def shutdown(self) -> None:
        """Stop background work; called when the window closes."""
        self.flush()
        self._runner.shutdown()


def _centre_square(image: NDArray[Any]) -> NDArray[Any]:
    """Return the largest square in the middle of *image*."""
    height, width = image.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return np.ascontiguousarray(image[top : top + side, left : left + side])


def _reference_geometry(decoded: DecodedPhoto) -> Geometry:
    """Return the geometry *decoded* opens with: whole, lens as it should start."""
    return Geometry(lens=LensSettings.for_photo(decoded.is_raw))


def _sort_key(path: Path) -> tuple[str, str]:
    """Order photos by name, ignoring case, with the extension as a tie break."""
    return (path.stem.casefold(), path.suffix.casefold())


def _folder_sort_key(
    root: Path, path: Path
) -> tuple[tuple[str, ...], tuple[str, str]]:
    """Order photos by the folder they sit in below *root*, then by name."""
    try:
        folders = path.parent.relative_to(root).parts
    except ValueError:
        folders = path.parent.parts
    return (tuple(part.casefold() for part in folders), _sort_key(path))


#: Folders Windows keeps for itself, which never hold the user's photos.
_SYSTEM_FOLDERS: Final[frozenset[str]] = frozenset(
    {"$recycle.bin", "system volume information"}
)


def list_photos(
    folder: Path, recursive: bool = True, wanted: Callable[[], bool] = lambda: True
) -> list[Path]:
    """Return the supported photos in *folder*, grouped by folder and ordered by name.

    With *recursive* the folders below *folder* are searched too: the photos
    directly in *folder* come first, then each subfolder in name order. A RAW
    and a JPEG of the same shot are two separate photos, which falls out of
    sorting by the whole file name (F-05). Hidden folders (a leading dot) and
    Windows' own system folders are skipped, links to folders are not
    followed so a loop cannot trap the walk, and a folder that cannot be read
    is logged and passed over. The walk ends early once *wanted* says no.
    """

    def skipped(error: OSError) -> None:
        logger.info("%s could not be listed: %s", error.filename, error)

    photos: list[Path] = []
    for current, folders, names in os.walk(folder, onerror=skipped):
        if not wanted():
            return []
        base = Path(current)
        photos.extend(
            base / name for name in names if decoder.is_supported(base / name)
        )
        if not recursive:
            break
        folders[:] = [
            name
            for name in folders
            if not name.startswith(".") and name.casefold() not in _SYSTEM_FOLDERS
        ]
    return sorted(photos, key=partial(_folder_sort_key, folder))


def _in_pixels(
    frame: tuple[int, int], fractions: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Return a rectangle given in fractions of *frame* in its pixels."""
    width, height = frame
    left, top, right, bottom = fractions
    return (left * width, top * height, right * width, bottom * height)


def _in_fractions(window: geometry.Window) -> tuple[float, float, float, float]:
    """Return where *window* sits in its frame, in fractions of it."""
    width, height = window.frame
    return (
        window.left / width,
        window.top / height,
        window.right / width,
        window.bottom / height,
    )


def _camera_preview(path: Path) -> tuple[NDArray[np.uint8], tuple[int, int]] | None:
    """Return the preview the camera stored inside a RAW file, and the photo's size."""
    return decoder.embedded_preview_and_size(path)


def _decode(path: Path) -> DecodedPhoto:
    """Decode a photo to its preview proxy."""
    return decoder.decode_proxy(path)


def _update_sidecar(
    photo: Path,
    values: dict[str, float],
    application: sidecar.Application,
    store: Path,
) -> sidecar.SaveResult:
    """Change some stored adjustments, reporting rather than raising."""
    try:
        return sidecar.update_adjustments(photo, values, application, store)
    except Exception:  # noqa: BLE001 - a paste must never take anything down
        logger.exception("The parameters of %s could not be stored", photo)
        return sidecar.SaveResult(failed=True)


def _save_sidecar(
    path: Path,
    edit: Edit,
    reference: Edit,
    application: sidecar.Application,
    store: Path,
    extra: dict[str, Any],
) -> sidecar.SaveResult:
    """Store the parameters of one photo, reporting rather than raising.

    A failed save must never reach the generic task failure path, which tells
    the user a photo could not be opened; it has its own message.
    """
    try:
        return sidecar.save(
            path,
            edit,
            reference,
            application,
            store,
            extra=extra,
        )
    except Exception:  # noqa: BLE001 - a save must never take anything down
        logger.exception("The parameters of %s could not be stored", path)
        return sidecar.SaveResult(failed=True)


__all__ = ["Controller", "Preview", "DecodeError"]
