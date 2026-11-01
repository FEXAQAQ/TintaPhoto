"""Exporting several photos at once (PLAN.md 2.6 X-09, 3.11).

The photos ticked in the filmstrip are written one after another on a
background thread, each with its own adjustments and with the settings the
export dialog chose for all of them. One photo is decoded and developed at a
time, so the memory needed is that of a single export.

Each photo's edit is taken on the interface thread when the export starts:
the photo being edited from the controller, any other photo visited in this
run from the history, which can be newer than its parameter file by the
autosave delay. A photo the history does not know has its parameter file
read on the background thread, which needs the decoded photo for the
reference white balance. Edits made after the export started are not used.

A photo that cannot be written is logged and skipped, and the others carry
on; the user is told at the end how many made it. The export can be asked to
stop: the photo being written is finished first, so no half written file is
left behind.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from PySide6.QtCore import QObject, Signal

from app.controller import Controller
from app.workers import TaskRunner
from core import decoder, exporter
from core.decoder import DecodedPhoto
from core.edit_state import Edit, EditState
from core.exporter import ExportOptions
from core.faces import Face
from core.filters import Look

logger = logging.getLogger(__name__)

#: One photo at a time: the memory of one export, and the order kept.
_THREADS: Final[int] = 1


@dataclass(frozen=True)
class BatchItem:
    """One photo to write, with the edit it is written with."""

    path: Path
    #: The edit to use, or ``None`` to read the photo's parameter file.
    edit: Edit | None = None
    #: The faces the preview found, for the photo being edited.
    faces: tuple[Face, ...] | None = None


@dataclass(frozen=True)
class BatchExportResult:
    """How an export of several photos went."""

    #: Where each photo that made it was written, in the order written.
    written: tuple[Path, ...]
    #: The photos that could not be written.
    failed: tuple[Path, ...]
    #: How many photos were asked for.
    total: int
    #: Whether the export was stopped before every photo had its turn.
    stopped: bool = False


def batch_target(source: Path, folder: Path, suffix: str, options: ExportOptions) -> Path:
    """Return the name *source* is written under: its own name and *suffix*.

    A name already taken is left alone; the exporter writes beside it under
    ``_2``, ``_3`` and so on (X-05).
    """
    return folder / f"{source.stem}{suffix}{options.extension}"


def run_batch(
    items: Sequence[BatchItem],
    folder: Path,
    suffix: str,
    options: ExportOptions,
    edit_of: Callable[[Path, DecodedPhoto], Edit],
    look_for: Callable[[EditState], Look | None],
    stop: threading.Event,
    progress: Callable[[float], None],
    advanced: Callable[[int, int], None],
) -> BatchExportResult:
    """Write every photo of *items* into *folder*, one after another.

    *edit_of* reads a photo's parameter file, for an item that carries no
    edit, and *look_for* finds the film look an edit names. *progress* hears
    how far the whole export has got, from 0 to 1, and *advanced* which photo
    is being written, counted from 1, and of how many. Never raises.
    """
    total = len(items)
    written: list[Path] = []
    failed: list[Path] = []
    for index, item in enumerate(items):
        if stop.is_set():
            return BatchExportResult(tuple(written), tuple(failed), total, stopped=True)
        advanced(index + 1, total)

        def step(fraction: float, done: int = index) -> None:
            progress((done + fraction) / total)

        photo: DecodedPhoto | None = None
        try:
            photo = decoder.decode(item.path)
            edit = item.edit if item.edit is not None else edit_of(item.path, photo)
            result = exporter.export_decoded(
                photo,
                item.path,
                batch_target(item.path, folder, suffix, options),
                edit.adjustments,
                edit.geometry,
                options,
                progress=step,
                look=look_for(edit.adjustments),
                faces=item.faces,
            )
            written.append(result.path)
        except Exception:  # noqa: BLE001 - one photo must not stop the others
            logger.exception("Exporting %s as part of several failed", item.path)
            failed.append(item.path)
        finally:
            # Let go before the next photo is decoded, or two would be held.
            photo = None
        progress((index + 1) / total)
    return BatchExportResult(tuple(written), tuple(failed), total)


class BatchExporter(QObject):
    """Writes several photos in the background, one after another (X-09)."""

    #: How far the whole export has got, from 0 to 1.
    progress = Signal(float)
    #: Which photo is being written, counted from 1, and of how many.
    advanced = Signal(int, int)
    #: The finished export, as a :class:`BatchExportResult`.
    finished = Signal(object)

    def __init__(self, controller: Controller, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._runner = TaskRunner(self, max_threads=_THREADS)
        self._runner.completed.connect(self._on_completed)
        self._runner.failed.connect(self._on_failed)
        self._stop = threading.Event()
        self._exporting = False
        self._total = 0

    @property
    def exporting(self) -> bool:
        """Return whether an export of several photos is running."""
        return self._exporting

    @property
    def stopping(self) -> bool:
        """Return whether the running export has been asked to stop."""
        return self._exporting and self._stop.is_set()

    def export(
        self, paths: Sequence[Path], folder: Path, suffix: str, options: ExportOptions
    ) -> bool:
        """Start writing *paths* into *folder*; return whether it started.

        One export of several runs at a time. Called on the interface thread,
        which is where each photo's edit is taken.
        """
        if self._exporting or not paths:
            return False
        items = [self._item(path) for path in paths]
        self._stop.clear()
        self._exporting = True
        self._total = len(items)
        self.progress.emit(0.0)
        controller = self._controller
        chosen = options.sane()
        self._runner.submit(
            lambda: run_batch(
                items,
                folder,
                suffix,
                chosen,
                controller.stored_edit_of,
                controller.look_for,
                self._stop,
                self.progress.emit,
                self.advanced.emit,
            )
        )
        return True

    def stop(self) -> None:
        """Stop once the photo being written is finished."""
        if self._exporting:
            self._stop.set()

    def shutdown(self) -> None:
        """Stop and wait for the photo being written; called on closing."""
        self._stop.set()
        self._runner.shutdown()

    def _item(self, path: Path) -> BatchItem:
        """Return *path* with the newest edit there is for it."""
        controller = self._controller
        if path == controller.current:
            return BatchItem(
                path, Edit(controller.state, controller.geometry), controller.faces_of_current()
            )
        return BatchItem(path, controller.remembered_edit(path))

    def _on_completed(self, _token: int, result: object) -> None:
        self._exporting = False
        self.finished.emit(result)

    def _on_failed(self, _token: int, message: str) -> None:
        # run_batch catches its own errors, so this is not expected.
        logger.error("The export of several photos failed: %s", message)
        self._exporting = False
        self.finished.emit(BatchExportResult((), (), self._total))
