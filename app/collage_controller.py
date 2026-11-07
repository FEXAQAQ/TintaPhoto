"""The collage's side of the controller layer (PLAN.md 2.8, 3.13).

Holds the photos, where each one sits and how the canvas is laid out,
and does the slow work - developing a photo for the screen, writing the
finished collage - on background threads.

Each photo goes in as it was edited: its adjustments, film filter, crop and
rotation, developed by the same pipeline as everywhere else. The edit is read
from this run's history when there is one, which may be newer than the
parameter file by the autosave delay, and otherwise from the parameter file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, Signal

from app.controller import Controller
from app.i18n import tr
from app.settings import Settings
from app.workers import TaskRunner
from core import collage, decoder, exporter, face_detector, pipeline
from core import color_management as cm
from core.collage import UNALIGNED, Alignment, Layout, Placement
from core.edit_state import Edit
from core.exporter import ExportOptions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CollageExportResult:
    """Where the collage was written, and its carousel's single photos."""

    path: Path
    #: A collage is always developed, never copied (X-08).
    copied: bool = False
    #: Each photo on a canvas of its own, in the order they are posted.
    singles: tuple[Path, ...] = ()

#: Developing two photos at once is as much as the collage ever asks for.
_THREADS: Final[int] = 2

#: How far the export has got at the start, over all the photos, and at the end.
_PROGRESS_START: Final[float] = 0.05
_PROGRESS_PHOTOS: Final[float] = 0.8
_PROGRESS_DONE: Final[float] = 1.0

#: What a piece of work that is not about one slot is filed under.
_EXPORT: Final[int] = -1
_ESTIMATE: Final[int] = -2


@dataclass(frozen=True)
class CollagePhoto:
    """One photo developed for the screen."""

    path: Path
    #: 8 bit sRGB, already converted for the screen, at the preview's size.
    image: NDArray[np.uint8]


class CollageController(QObject):
    """The photos of the collage and everything done with them."""

    #: A slot, and the photo now in it or ``None`` while one is on its way.
    photo_changed = Signal(int, object)
    #: The layout changed; the window redraws.
    layout_changed = Signal(object)
    #: A photo was moved, zoomed or aligned; the canvas and the sliders follow.
    placement_changed = Signal()
    #: A sentence to show the user, already translated.
    error = Signal(str)
    export_progress = Signal(float)
    #: The finished export, as a :class:`CollageExportResult`.
    export_finished = Signal(object)
    export_failed = Signal(str)
    #: Roughly how large the file would be, in bytes (PLAN.md 3.11).
    export_estimate_ready = Signal(int)

    def __init__(
        self, controller: Controller, settings: Settings, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._settings = settings
        self._runner = TaskRunner(self, max_threads=_THREADS)
        self._runner.completed.connect(self._on_completed)
        self._runner.failed.connect(self._on_failed)
        self._runner.discarded.connect(self._on_discarded)
        self._handlers: dict[int, tuple[Any, int, Path]] = {}

        self._layout = settings.collage.layout
        self._paths: list[Path | None] = [None] * collage.SLOTS
        self._photos: list[CollagePhoto | None] = [None] * collage.SLOTS
        self._placements: list[Placement] = [Placement()] * collage.SLOTS
        #: Linked, one placement for all photos, how each later one is
        #: aligned against the first, and which of them is being aligned
        #: (PLAN.md 2.8 K-11, K-12).
        self._shared = Placement()
        self._alignments: list[Alignment] = [UNALIGNED] * collage.SLOTS
        self._aligning_slot = 1
        self._exporting = False

    # -- what the collage holds ------------------------------------------

    @property
    def layout(self) -> Layout:
        """Return how the canvas is shaped and split."""
        return self._layout

    @property
    def count(self) -> int:
        """Return how many pieces the canvas is split into."""
        return self._layout.count

    @property
    def paths(self) -> tuple[Path | None, ...]:
        """Return the photo in each piece in use, first left or top.

        Photos put in pieces a smaller count leaves out are kept, and come
        back when the count goes up again, but are not part of the collage.
        """
        return tuple(self._paths[: self.count])

    @property
    def aligning_slot(self) -> int:
        """Return the linked photo the alignment tools move (PLAN.md 2.8 K-12)."""
        return min(self._aligning_slot, self.count - 1)

    def set_aligning_slot(self, slot: int) -> None:
        """Choose which linked photo after the first the alignment tools move."""
        slot = max(1, min(self.count - 1, int(slot)))
        if slot == self._aligning_slot:
            return
        self._aligning_slot = slot
        self.placement_changed.emit()

    def photo(self, slot: int) -> CollagePhoto | None:
        """Return the developed photo in *slot*, once it has arrived."""
        return self._photos[slot]

    def placement(self, slot: int) -> Placement:
        """Return where the photo in *slot* sits; linked, all share one.

        Linked, the shared placement is zoomed in as far as it takes for
        every aligned photo to cover its piece (PLAN.md 2.8 K-12).
        """
        if not self._layout.linked:
            return self._placements[slot]
        sizes = [self.photo_size(index) for index in range(self.count)]
        alignments = [self.alignment(index) for index in range(self.count)]
        return collage.linked_placement(self._layout, self._shared, sizes, alignments)

    def alignment(self, slot: int) -> Alignment:
        """Return how the photo in *slot* is aligned; never the first one."""
        if self._layout.linked and 1 <= slot < self.count:
            return self._alignments[slot]
        return UNALIGNED

    def photo_size(self, slot: int) -> tuple[int, int] | None:
        """Return the size of the photo in *slot*, once it has arrived."""
        photo = self._photos[slot]
        if photo is None:
            return None
        return int(photo.image.shape[1]), int(photo.image.shape[0])

    def rect(self, slot: int, aligned: bool = True) -> tuple[float, float, float, float] | None:
        """Return where the photo in *slot* lands on the canvas, in canvas pixels."""
        size = self.photo_size(slot)
        if size is None:
            return None
        alignment = self.alignment(slot) if aligned else UNALIGNED
        return collage.placed_rect(self._layout, slot, size, self.placement(slot), alignment)

    @property
    def complete(self) -> bool:
        """Return whether every piece has its photo, ready to be written."""
        return all(photo is not None for photo in self._photos[: self.count])

    @property
    def exporting(self) -> bool:
        """Return whether the collage is being written."""
        return self._exporting

    # -- changing it -----------------------------------------------------

    def set_layout(self, layout: Layout) -> None:
        """Shape the canvas anew; each photo keeps its zoom and focus."""
        layout = layout.sane()
        if layout == self._layout:
            return
        self._layout = layout
        self._settings.collage.remember(layout)
        self.layout_changed.emit(layout)
        # Linking or unlinking, or a different count, changes which
        # placement is in force.
        self.placement_changed.emit()

    def set_placement(self, slot: int, placement: Placement) -> None:
        """Move or zoom the photo in *slot*; linked, all move together."""
        if self._layout.linked:
            self._shared = placement
        else:
            self._placements[slot] = placement
        self.placement_changed.emit()

    def set_alignment(self, alignment: Alignment) -> None:
        """Move or scale the photo being aligned against the first (linked only)."""
        self._alignments[self.aligning_slot] = alignment.sane()
        self.placement_changed.emit()

    def set_zoom(self, slot: int, zoom: float) -> None:
        """Zoom the photo in *slot* about the middle of what it is placed against."""
        size = self.photo_size(slot)
        if size is None:
            size = next(
                (found for index in range(self.count) if (found := self.photo_size(index))),
                None,
            )
        if size is None:
            return
        frame = collage.frame(self._layout, slot)
        self.set_placement(
            slot,
            collage.zoomed(
                self.placement(slot), zoom, frame.centre, size, frame, self._layout.fit_used
            ),
        )

    def set_alignment_scale(self, scale: float) -> None:
        """Scale the photo being aligned about the middle of the canvas."""
        slot = self.aligning_slot
        rect = self.rect(slot, aligned=False)
        if rect is None:
            return
        centre = collage.canvas_cell(self._layout).centre
        self.set_alignment(collage.rescaled(self.alignment(slot), scale, centre, rect))

    def load(self, slot: int, path: Path) -> None:
        """Put *path* in *slot*, developed with its own edit, in the background."""
        if not decoder.is_supported(path):
            self.error.emit(tr("error.cannot_open", name=path.name))
            return
        self._paths[slot] = path
        self._photos[slot] = None
        self._placements[slot] = Placement()
        self._alignments[slot] = UNALIGNED
        self.photo_changed.emit(slot, None)
        self._request(slot, path)

    def fill(self, paths: list[Path]) -> None:
        """Put *paths* in the empty pieces in use, in order, and leave the rest."""
        for path in paths:
            empty = [slot for slot in range(self.count) if self._paths[slot] is None]
            if not empty:
                return
            if path not in self.paths:
                self.load(empty[0], path)

    def refresh(self) -> None:
        """Develop every photo again, in case it was edited meanwhile.

        Called when the collage is opened again: the photo may have been
        adjusted in the editor since it was put here. Where it sits is kept.
        """
        for slot, path in enumerate(self._paths):
            if path is not None:
                self._request(slot, path)

    def _request(self, slot: int, path: Path) -> None:
        remembered = self._controller.remembered_edit(path)
        transform = self._controller.display_transform

        def develop_for_screen(
            source: Path = path, edit: Edit | None = remembered, screen: object = transform
        ) -> CollagePhoto:
            photo = decoder.decode_proxy(source)
            chosen = edit if edit is not None else self._controller.stored_edit_of(source, photo)
            photo = face_detector.attach(photo, chosen.adjustments, chosen.geometry)
            image = pipeline.render(
                photo,
                chosen.adjustments,
                chosen.geometry,
                look=self._controller.look_for(chosen.adjustments),
            )
            return CollagePhoto(source, cm.apply_display_transform(image, screen))

        token = self._runner.submit(develop_for_screen, group=f"collage-{slot}")
        self._handlers[token] = (self._on_developed, slot, path)

    def _on_developed(self, slot: int, path: Path, photo: CollagePhoto) -> None:
        if self._paths[slot] != path:
            return
        self._photos[slot] = photo
        self.photo_changed.emit(slot, photo)

    # -- writing it ------------------------------------------------------

    def default_name(self) -> str:
        """Return the name the collage is offered under: all its photos' names."""
        stems = [path.stem for path in self.paths if path is not None]
        return "_".join(stems) + collage.FILE_SUFFIX

    def default_folder(self) -> Path | None:
        """Return the folder of the first photo, where the collage is offered."""
        for path in self.paths:
            if path is not None:
                return path.parent
        return None

    def output_size(self) -> tuple[int, int]:
        """Return the size the file will have; always the canvas."""
        return self._layout.canvas_size

    def export(self, target: Path, options: ExportOptions, singles: bool = False) -> bool:
        """Write the collage to *target* in the background (X-07).

        With *singles*, each photo is also written on a canvas of its own,
        for a carousel: the collage first, then each photo in order (K-13).
        """
        if self._exporting or not self.complete:
            return False
        self._exporting = True
        self.export_progress.emit(0.0)
        edits = [
            None if path is None else self._controller.remembered_edit(path)
            for path in self.paths
        ]
        job = _ExportJob(
            controller=self._controller,
            layout=self._layout,
            paths=list(self.paths),
            edits=edits,
            placements=[self.placement(slot) for slot in range(self.count)],
            alignments=[self.alignment(slot) for slot in range(self.count)],
            options=options.sane(),
            target=target,
            singles=singles,
            progress=self.export_progress.emit,
        )
        token = self._runner.submit(job.run)
        self._handlers[token] = (self._on_exported, _EXPORT, target)
        return True

    def request_export_estimate(self, options: ExportOptions) -> None:
        """Work out roughly how large the file would be (PLAN.md 3.11).

        The collage is laid out from the photos the screen shows and
        compressed with the settings asked for. Its pixels are the file's
        own size already, so nothing has to be scaled.
        """
        if not self.complete:
            return
        layout = self._layout
        images = [
            None if photo is None else photo.image for photo in self._photos[: self.count]
        ]
        placements = [self.placement(slot) for slot in range(self.count)]
        alignments = [self.alignment(slot) for slot in range(self.count)]

        def estimate() -> int:
            canvas = collage.render(layout, images, placements, alignments)
            width, height = layout.canvas_size
            return exporter.estimate_size(canvas, width * height, options)

        token = self._runner.submit(estimate, group="collage-estimate")
        self._handlers[token] = (self._on_estimated, _ESTIMATE, Path())

    def _on_estimated(self, _slot: int, _path: Path, size: object) -> None:
        self.export_estimate_ready.emit(int(size))

    def _on_exported(self, _slot: int, target: Path, result: object) -> None:
        self._exporting = False
        if result is None:
            self.export_progress.emit(0.0)
            self.export_failed.emit(tr("error.cannot_export", name=target.name))
            return
        self.export_finished.emit(result)

    # -- running work ----------------------------------------------------

    def _on_completed(self, token: int, result: object) -> None:
        entry = self._handlers.pop(token, None)
        if entry is None:
            return
        handler, slot, path = entry
        handler(slot, path, result)

    def _on_discarded(self, token: int) -> None:
        self._handlers.pop(token, None)

    def _on_failed(self, token: int, message: str) -> None:
        entry = self._handlers.pop(token, None)
        if entry is None:
            return
        _handler, slot, path = entry
        logger.info("Collage work on %s failed: %s", path, message)
        if slot == _ESTIMATE:
            return
        if slot == _EXPORT:
            # The export catches its own errors, so this is not expected.
            self._exporting = False
            self.export_failed.emit(tr("error.cannot_export", name=path.name))
            return
        if self._paths[slot] == path:
            self._paths[slot] = None
            self._photos[slot] = None
            self.photo_changed.emit(slot, None)
        self.error.emit(tr("error.cannot_open", name=path.name))

    def shutdown(self) -> None:
        """Stop background work; called when the window closes."""
        self._runner.shutdown()


@dataclass
class _ExportJob:
    """Everything one export needs, taken on the interface thread."""

    controller: Controller
    layout: Layout
    paths: list[Path | None]
    edits: list[Edit | None]
    placements: list[Placement]
    alignments: list[Alignment]
    options: ExportOptions
    target: Path
    progress: Any
    singles: bool = False

    def run(self) -> CollageExportResult | None:
        """Develop each photo at full size, lay them out and write the files.

        One photo at a time, and only the piece it fills - and its own canvas,
        for a carousel - is kept, so the memory needed is one developed photo
        rather than all of them. Returns ``None`` rather than raising, so the
        progress is never left half way.
        """
        try:
            self.progress(_PROGRESS_START)
            depth = self.options.depth
            dither = depth == pipeline.EIGHT_BIT
            halves: list[NDArray[np.float32] | None] = []
            singles: list[NDArray[Any]] = []
            for index, (path, edit, placement, alignment) in enumerate(
                zip(self.paths, self.edits, self.placements, self.alignments)
            ):
                if path is None:
                    halves.append(None)
                    continue
                photo = decoder.decode(path)
                chosen = edit if edit is not None else self.controller.stored_edit_of(path, photo)
                photo = face_detector.attach(photo, chosen.adjustments, chosen.geometry)
                developed = pipeline.render(
                    photo,
                    chosen.adjustments,
                    chosen.geometry,
                    bits=pipeline.SIXTEEN_BIT,
                    look=self.controller.look_for(chosen.adjustments),
                )
                del photo
                halves.append(
                    collage.render_slot(self.layout, index, developed, placement, alignment)
                )
                if self.singles:
                    single = collage.render_single(self.layout, developed, placement)
                    singles.append(collage.encode(single, depth, dither))
                del developed
                self.progress(
                    _PROGRESS_START + _PROGRESS_PHOTOS * (index + 1) / max(1, len(self.paths))
                )
            canvas = collage.encode(collage.compose(self.layout, halves), depth, dither)
            written = exporter.export_pixels(canvas, self.target, self.options).path
            # Named after the collage as it was written, so that a second
            # export's photos sit beside its own collage, not the first one's.
            single_paths = tuple(
                exporter.export_pixels(
                    pixels,
                    written.with_name(f"{written.stem}{suffix}{written.suffix}"),
                    self.options,
                ).path
                for pixels, suffix in zip(singles, collage.SINGLE_SUFFIXES)
            )
            self.progress(_PROGRESS_DONE)
            return CollageExportResult(path=written, singles=single_paths)
        except Exception:  # noqa: BLE001 - the user is told, the log has the rest
            logger.exception("Exporting the collage to %s failed", self.target)
            return None
