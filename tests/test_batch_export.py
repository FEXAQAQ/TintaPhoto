"""Tests for exporting several photos at once (PLAN.md 2.6 X-09).

The job runs here directly, on the test's own thread, with small JPEGs
written for the purpose; the background thread and its signals are the task
runner's business and are tested with it.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.batch_export import BatchExportResult, BatchItem, batch_target, run_batch
from core.decoder import DecodedPhoto
from core.edit_state import Edit, EditState
from core.exporter import FORMAT_PNG, ExportOptions


def _photo(path: Path, grey: int = 120) -> Path:
    Image.new("RGB", (32, 24), (grey, grey, grey)).save(path, "JPEG", quality=95)
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Calls:
    """Collects what the job reports, the way the window would hear it."""

    def __init__(self) -> None:
        self.progress: list[float] = []
        self.advanced: list[tuple[int, int]] = []
        self.read: list[Path] = []

    def edit_of(self, path: Path, _photo: DecodedPhoto) -> Edit:
        self.read.append(path)
        return Edit(EditState(exposure=0.5))


def _run(
    items: list[BatchItem],
    folder: Path,
    calls: _Calls,
    *,
    suffix: str = "_edit",
    options: ExportOptions | None = None,
    stop: threading.Event | None = None,
) -> BatchExportResult:
    return run_batch(
        items,
        folder,
        suffix,
        options or ExportOptions(),
        calls.edit_of,
        lambda _state: None,
        stop or threading.Event(),
        calls.progress.append,
        lambda index, total: calls.advanced.append((index, total)),
    )


@pytest.fixture
def photos(tmp_path: Path) -> list[Path]:
    folder = tmp_path / "Photos 京都"
    folder.mkdir()
    return [_photo(folder / f"DSC000{index}.jpg", 60 + 40 * index) for index in range(3)]


def test_every_photo_is_written_under_its_own_name_and_the_suffix(
    photos: list[Path], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    items = [BatchItem(path, Edit(EditState(exposure=0.3))) for path in photos]

    result = _run(items, out, _Calls())

    assert [path.name for path in result.written] == [
        "DSC0000_edit.jpg",
        "DSC0001_edit.jpg",
        "DSC0002_edit.jpg",
    ]
    assert all(path.exists() for path in result.written)
    assert result.failed == ()
    assert not result.stopped
    assert result.total == 3


def test_the_originals_are_never_touched(photos: list[Path]) -> None:
    before = [_digest(path) for path in photos]
    items = [BatchItem(path, Edit(EditState(exposure=0.3))) for path in photos]

    # Written into the photos' own folder, with no suffix at all: the names
    # are taken by the originals, so the exports go beside them.
    result = _run(items, photos[0].parent, _Calls(), suffix="")

    assert [_digest(path) for path in photos] == before
    assert [path.name for path in result.written] == [
        "DSC0000_2.jpg",
        "DSC0001_2.jpg",
        "DSC0002_2.jpg",
    ]


def test_a_photo_without_an_edit_has_its_parameter_file_read(
    photos: list[Path], tmp_path: Path
) -> None:
    calls = _Calls()
    items = [BatchItem(photos[0], Edit(EditState())), BatchItem(photos[1])]

    result = _run(items, tmp_path / "out", calls, options=ExportOptions(format=FORMAT_PNG))

    assert calls.read == [photos[1]]
    with Image.open(result.written[1]) as brighter:
        assert np.asarray(brighter).mean() > 100, "the stored exposure was not used"


def test_an_untouched_jpeg_is_copied_as_it_is(photos: list[Path], tmp_path: Path) -> None:
    result = _run([BatchItem(photos[0], Edit(EditState()))], tmp_path / "out", _Calls())

    assert result.written[0].read_bytes() == photos[0].read_bytes()


def test_a_photo_that_fails_does_not_stop_the_others(
    photos: list[Path], tmp_path: Path
) -> None:
    broken = photos[0].with_name("broken.jpg")
    broken.write_bytes(b"not a photo")
    items = [BatchItem(photos[0]), BatchItem(broken), BatchItem(photos[2])]

    result = _run(items, tmp_path / "out", _Calls())

    assert result.failed == (broken,)
    assert [path.name for path in result.written] == ["DSC0000_edit.jpg", "DSC0002_edit.jpg"]


def test_stopping_finishes_the_photo_being_written_and_no_more(
    photos: list[Path], tmp_path: Path
) -> None:
    stop = threading.Event()
    calls = _Calls()

    def advanced(index: int, total: int) -> None:
        calls.advanced.append((index, total))
        if index == 2:
            stop.set()

    result = run_batch(
        [BatchItem(path) for path in photos],
        tmp_path / "out",
        "_edit",
        ExportOptions(),
        calls.edit_of,
        lambda _state: None,
        stop,
        calls.progress.append,
        advanced,
    )

    assert result.stopped
    assert len(result.written) == 2
    assert calls.advanced == [(1, 3), (2, 3)]


def test_progress_covers_the_whole_export_and_never_goes_back(
    photos: list[Path], tmp_path: Path
) -> None:
    calls = _Calls()

    _run([BatchItem(path) for path in photos], tmp_path / "out", calls)

    assert calls.progress == sorted(calls.progress)
    assert calls.progress[-1] == pytest.approx(1.0)
    assert calls.advanced == [(1, 3), (2, 3), (3, 3)]


def test_the_name_follows_the_format() -> None:
    target = batch_target(
        Path("D:/Photos/DSC01234.ARW"), Path("E:/Out"), "_web", ExportOptions(format=FORMAT_PNG)
    )
    assert target == Path("E:/Out/DSC01234_web.png")
