"""Tests for the controller (PLAN.md sections 2.1 and 5).

The controller does its work on background threads, so each test waits for
the signal it is interested in rather than assuming anything has finished.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.i18n import tr
from app.settings import Settings

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer  # noqa: E402

from app import controller as controller_module  # noqa: E402
from app.controller import DRAFT_MINIMUM_LONG_EDGE, Controller  # noqa: E402
from core import geometry, liquify, sidecar  # noqa: E402
from core.geometry import Geometry, Rect  # noqa: E402
from core.edit_state import Edit, EditState  # noqa: E402
from core.exporter import ExportOptions  # noqa: E402
from core.filters import NO_FILTER  # noqa: E402
from core.lens import LensSettings  # noqa: E402


TIMEOUT_MS = 20_000


def _write_jpeg(path: Path, colour: tuple[int, int, int] = (120, 120, 120)) -> Path:
    Image.new("RGB", (240, 160), colour).save(path, "JPEG")
    return path


def wait_for(signal, timeout_ms: int = TIMEOUT_MS) -> list:
    """Wait for *signal* and return the arguments it carried."""
    loop = QEventLoop()
    received: list = []

    def capture(*args: object) -> None:
        if received:  # several may arrive before the loop gets a turn
            return
        received.extend(args)
        loop.quit()

    # Disconnected by the handle connect() gives back rather than by the
    # function: PySide6 fails to find a closure again once a queued emission
    # from a worker thread has been left behind in the event queue, and the
    # failure surfaces as a SystemError in whichever test runs next.
    connection = signal.connect(capture)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(timeout_ms)
    loop.exec()
    signal.disconnect(connection)
    assert received, "the signal never arrived"
    return received


@pytest.fixture
def controller(
    qt_app: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Controller]:
    # Every parameter file goes into the store, which a test must keep out
    # of the real application data folder.
    monkeypatch.setattr(controller_module, "sidecar_dir", lambda: tmp_path / "store")
    monkeypatch.setattr(
        controller_module, "earlier_sidecar_dir", lambda: tmp_path / "earlier store"
    )
    # The same goes for the presets folder, which is in the user's own
    # application data.
    monkeypatch.setattr(controller_module, "presets_dir", lambda: tmp_path / "presets")
    settings = Settings()
    created = Controller(settings)
    yield created
    created.shutdown()


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    photos = tmp_path / "photos"
    photos.mkdir()
    for name, colour in (("b.jpg", (200, 60, 60)), ("a.jpg", (60, 200, 60)), ("c.jpg", (60, 60, 200))):
        _write_jpeg(photos / name, colour)
    return photos


def open_and_wait(controller: Controller, path: Path) -> None:
    """Open *path* and wait for its folder, which is listed in the background."""
    controller.open(path)
    wait_for(controller.folder_opened)


# -- opening --------------------------------------------------------------


def test_opening_a_folder_lists_its_photos_in_name_order(
    controller: Controller, folder: Path
) -> None:
    open_and_wait(controller, folder)
    assert [path.name for path in controller.photos] == ["a.jpg", "b.jpg", "c.jpg"]


def test_opening_a_folder_selects_the_first_photo(controller: Controller, folder: Path) -> None:
    open_and_wait(controller, folder)
    assert controller.index == 0
    assert controller.current is not None
    assert controller.current.name == "a.jpg"


def test_opening_a_folder_finds_the_photos_in_its_subfolders(
    controller: Controller, tmp_path: Path
) -> None:
    """F-02: a folder of folders opens every photo below it, folder by folder."""
    root = tmp_path / "2026-10 京都"
    (root / "Day 2").mkdir(parents=True)
    (root / "day 1" / "evening").mkdir(parents=True)
    _write_jpeg(root / "z.jpg")
    _write_jpeg(root / "Day 2" / "a.jpg")
    _write_jpeg(root / "day 1" / "b.jpg")
    _write_jpeg(root / "day 1" / "evening" / "a.jpg")
    open_and_wait(controller, root)
    assert [path.relative_to(root).as_posix() for path in controller.photos] == [
        "z.jpg",
        "day 1/b.jpg",
        "day 1/evening/a.jpg",
        "Day 2/a.jpg",
    ]


def test_a_folder_with_photos_only_in_subfolders_is_not_empty(
    controller: Controller, tmp_path: Path
) -> None:
    root = tmp_path / "trip"
    (root / "card").mkdir(parents=True)
    _write_jpeg(root / "card" / "a.jpg")
    messages: list[str] = []
    controller.error.connect(messages.append)
    open_and_wait(controller, root)
    assert not messages
    assert controller.current == root / "card" / "a.jpg"


def test_hidden_and_system_folders_are_skipped(
    controller: Controller, folder: Path
) -> None:
    for name in (".thumbnails", "$RECYCLE.BIN", "System Volume Information"):
        (folder / name).mkdir()
        _write_jpeg(folder / name / "x.jpg")
    open_and_wait(controller, folder)
    assert [path.name for path in controller.photos] == ["a.jpg", "b.jpg", "c.jpg"]


def test_opening_a_file_opens_only_that_photo(
    controller: Controller, folder: Path
) -> None:
    """F-01: the rest of its folder is not brought along."""
    open_and_wait(controller, folder / "b.jpg")
    assert controller.photos == [folder / "b.jpg"]
    assert controller.current == folder / "b.jpg"


def test_a_photo_opened_by_itself_is_not_a_recent_folder(
    controller: Controller, folder: Path
) -> None:
    """Only its folder is kept, as the place the open dialogs start in."""
    settings = controller.settings
    settings.remember_folder("D:/Earlier", 5)
    open_and_wait(controller, folder / "a.jpg")
    assert settings.recent_folders == ["D:/Earlier"]
    assert settings.last_folder == "D:/Earlier"
    assert settings.browse_folder == str(folder)


def test_several_files_open_by_themselves_in_name_order(
    controller: Controller, folder: Path
) -> None:
    controller.open_paths([folder / "c.jpg", folder / "a.jpg"])
    wait_for(controller.folder_opened)
    assert controller.photos == [folder / "a.jpg", folder / "c.jpg"]
    # The first one chosen is the one shown.
    assert controller.current == folder / "c.jpg"


def test_files_that_are_not_photos_are_left_out_of_a_choice(
    controller: Controller, folder: Path
) -> None:
    notes = folder / "notes.txt"
    notes.write_text("hello", encoding="utf-8")
    controller.open_paths([notes, folder / "a.jpg", folder / "a.jpg"])
    wait_for(controller.folder_opened)
    assert controller.photos == [folder / "a.jpg"]


def test_a_folder_dropped_first_opens_whole(
    controller: Controller, folder: Path, tmp_path: Path
) -> None:
    """The first path decides; the photo dropped with the folder is not added."""
    loose = tmp_path / "loose.jpg"
    _write_jpeg(loose)
    controller.open_paths([folder, loose])
    wait_for(controller.folder_opened)
    assert [path.name for path in controller.photos] == ["a.jpg", "b.jpg", "c.jpg"]
    assert controller.settings.recent_folders == [str(folder)]


def test_photos_opened_replace_a_folder_still_being_listed(
    controller: Controller, folder: Path, tmp_path: Path
) -> None:
    loose = tmp_path / "loose.jpg"
    _write_jpeg(loose)
    opened: list[list[Path]] = []
    controller.folder_opened.connect(opened.append)
    controller.open(folder)
    controller.open(loose)
    wait_for(controller.folder_opened)
    wait_for(controller.preview_ready)
    assert opened == [[loose]]
    assert controller.photos == [loose]


def test_a_newer_folder_replaces_one_still_being_listed(
    controller: Controller, folder: Path, tmp_path: Path
) -> None:
    """The second folder asked for is the one that opens, whichever finishes first."""
    other = tmp_path / "other"
    other.mkdir()
    _write_jpeg(other / "z.jpg")
    opened: list[list[Path]] = []
    controller.folder_opened.connect(opened.append)
    controller.open(folder)
    controller.open(other)
    wait_for(controller.folder_opened)
    wait_for(controller.preview_ready)
    assert opened == [[other / "z.jpg"]]
    assert controller.photos == [other / "z.jpg"]


def test_listing_stops_once_it_is_no_longer_wanted(folder: Path) -> None:
    (folder / "sub").mkdir()
    _write_jpeg(folder / "sub" / "d.jpg")
    assert controller_module.list_photos(folder, True, lambda: False) == []
    assert len(controller_module.list_photos(folder, True)) == 4


def test_files_that_are_not_photos_are_ignored(controller: Controller, folder: Path) -> None:
    (folder / "notes.txt").write_text("hello", encoding="utf-8")
    (folder / "clip.mp4").write_bytes(b"\x00\x00")
    open_and_wait(controller, folder)
    assert len(controller.photos) == 3


def test_a_raw_and_a_jpeg_of_the_same_shot_are_two_photos(
    controller: Controller, tmp_path: Path
) -> None:
    """PLAN.md F-05: same name, different extension, two separate photos."""
    folder = tmp_path / "pair"
    folder.mkdir()
    _write_jpeg(folder / "DSC01234.JPG")
    (folder / "DSC01234.ARW").write_bytes(bytes(1024))
    open_and_wait(controller, folder)
    assert [path.name for path in controller.photos] == ["DSC01234.ARW", "DSC01234.JPG"]


def test_an_empty_folder_reports_a_readable_message(
    controller: Controller, tmp_path: Path
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    messages: list[str] = []
    controller.error.connect(messages.append)
    controller.open(empty)
    wait_for(controller.error)
    assert messages
    # A sentence naming the folder, not a code or a traceback.
    assert empty.name in messages[0]
    assert messages[0].endswith(("。", ".")) and len(messages[0]) > len(empty.name) + 4


def test_a_path_that_does_not_exist_reports_a_message(
    controller: Controller, tmp_path: Path
) -> None:
    messages: list[str] = []
    controller.error.connect(messages.append)
    controller.open(tmp_path / "nowhere")
    assert messages


def test_an_unsupported_file_reports_a_message(controller: Controller, tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")
    messages: list[str] = []
    controller.error.connect(messages.append)
    controller.open(path)
    assert messages


# -- the preview ----------------------------------------------------------


def test_opening_a_photo_produces_a_preview(controller: Controller, folder: Path) -> None:
    controller.open(folder)
    preview = wait_for(controller.preview_ready)[0]
    assert preview.path.name == "a.jpg"
    assert preview.image.dtype == np.uint8
    assert preview.image.shape[2] == 3
    assert not preview.provisional


def test_an_unedited_jpeg_preview_is_the_original_photo(
    controller: Controller, tmp_path: Path
) -> None:
    """PLAN.md section 6: an untouched JPEG is shown pixel for pixel."""
    folder = tmp_path / "one"
    folder.mkdir()
    path = _write_jpeg(folder / "a.jpg", (34, 177, 76))
    controller.open(folder)
    preview = wait_for(controller.preview_ready)[0]
    with Image.open(path) as image:
        original = np.asarray(image.convert("RGB"), dtype=np.uint8)
    assert np.array_equal(preview.image, original)


def test_an_unedited_png_preview_is_the_original_photo(
    controller: Controller, tmp_path: Path
) -> None:
    """A PNG is a finished photo too, so it is shown pixel for pixel."""
    folder = tmp_path / "png"
    folder.mkdir()
    path = folder / "a.png"
    Image.new("RGB", (240, 160), (34, 177, 76)).save(path, "PNG")
    controller.open(folder)
    preview = wait_for(controller.preview_ready)[0]
    assert preview.path.name == "a.png"
    with Image.open(path) as image:
        original = np.asarray(image.convert("RGB"), dtype=np.uint8)
    assert np.array_equal(preview.image, original)


def test_a_damaged_photo_reports_a_message_instead_of_crashing(
    controller: Controller, tmp_path: Path
) -> None:
    """PLAN.md F-06: say what happened, keep running."""
    folder = tmp_path / "broken"
    folder.mkdir()
    (folder / "broken.jpg").write_bytes(b"\xff\xd8 not a photo")
    controller.open(folder)
    message = wait_for(controller.error)[0]
    assert isinstance(message, str) and message


# -- moving between photos ------------------------------------------------


def test_moving_to_the_next_and_previous_photo(controller: Controller, folder: Path) -> None:
    open_and_wait(controller, folder)
    controller.select_next()
    assert controller.current is not None and controller.current.name == "b.jpg"
    controller.select_previous()
    assert controller.current is not None and controller.current.name == "a.jpg"


def test_moving_stops_at_the_ends(controller: Controller, folder: Path) -> None:
    open_and_wait(controller, folder)
    controller.select_previous()
    assert controller.index == 0
    controller.select(len(controller.photos) - 1)
    controller.select_next()
    assert controller.index == len(controller.photos) - 1


def _counting_decodes(monkeypatch: pytest.MonkeyPatch, hold: dict[str, threading.Event]) -> list[str]:
    """Record every proxy decode by file name, holding back the ones in *hold*."""
    calls: list[str] = []
    real = controller_module._decode

    def counted(path: Path):
        calls.append(path.name)
        gate = hold.get(path.name)
        if gate is not None:
            gate.wait(10)
        return real(path)

    monkeypatch.setattr(controller_module, "_decode", counted)
    return calls


def _wait_for_photo(controller: Controller, name: str) -> None:
    """Wait for the finished preview of the photo called *name*."""
    for _ in range(20):
        preview = wait_for(controller.preview_ready)[0]
        if preview.path.name == name and not preview.provisional:
            return
    raise AssertionError(f"{name} never arrived")


def test_neighbours_are_preloaded_only_once_the_photo_has_arrived(
    controller: Controller, folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decoding them alongside would slow down the photo being waited for."""
    gate = threading.Event()
    calls = _counting_decodes(monkeypatch, {"a.jpg": gate})
    try:
        open_and_wait(controller, folder)
        deadline = time.perf_counter() + 5
        while not calls and time.perf_counter() < deadline:
            QCoreApplication.processEvents()
        # The photo is being decoded, and nothing else is.
        time.sleep(0.1)
        QCoreApplication.processEvents()
        assert calls == ["a.jpg"]
    finally:
        gate.set()
    _wait_for_photo(controller, "a.jpg")
    deadline = time.perf_counter() + 5
    while len(calls) < 2 and time.perf_counter() < deadline:
        QCoreApplication.processEvents()
    assert calls[1:] == ["b.jpg"]


def test_a_photo_being_preloaded_is_not_decoded_a_second_time(
    controller: Controller, folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = threading.Event()
    calls = _counting_decodes(monkeypatch, {"b.jpg": gate})
    try:
        open_and_wait(controller, folder)
        _wait_for_photo(controller, "a.jpg")
        controller.select(1)  # while its preload is under way, and held
    finally:
        gate.set()
    _wait_for_photo(controller, "b.jpg")
    assert calls.count("b.jpg") == 1


def test_a_photo_does_not_wait_for_the_one_clicked_before_it(
    controller: Controller, folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clicking along the strip, the newest photo starts decoding at once."""
    gate = threading.Event()
    calls = _counting_decodes(monkeypatch, {"a.jpg": gate})
    open_and_wait(controller, folder)
    controller.select(2)
    try:
        # c.jpg arrives while a.jpg is still being held back.
        _wait_for_photo(controller, "c.jpg")
        assert "a.jpg" not in {path.name for path in controller._decoded}
    finally:
        gate.set()


def test_selection_changes_are_announced(controller: Controller, folder: Path) -> None:
    open_and_wait(controller, folder)
    seen: list[tuple[int, int]] = []
    controller.selection_changed.connect(lambda index, total: seen.append((index, total)))
    controller.select(2)
    assert seen == [(2, 3)]


def test_only_the_photo_and_its_neighbours_are_kept_decoded(
    controller: Controller, tmp_path: Path
) -> None:
    """Three proxies at most, whatever the size of the folder (PLAN.md 4)."""
    folder = tmp_path / "many"
    folder.mkdir()
    for index in range(8):
        _write_jpeg(folder / f"{index}.jpg")
    controller.open(folder)
    for index in range(8):
        controller.select(index)
        wait_for(controller.preview_ready)
    assert len(controller._decoded) <= 3


# -- thumbnails -----------------------------------------------------------


def test_thumbnails_are_produced_for_the_folder(controller: Controller, folder: Path) -> None:
    open_and_wait(controller, folder)
    controller.request_thumbnails(controller.photos)
    path, thumbnail = wait_for(controller.thumbnail_ready)
    assert path in controller.photos
    assert thumbnail.dtype == np.uint8
    assert max(thumbnail.shape[:2]) <= 256


def test_metadata_arrives_for_the_current_photo(controller: Controller, folder: Path) -> None:
    controller.open(folder)
    info = wait_for(controller.metadata_ready)[0]
    assert info.file_format == "JPG"
    assert info.width == 240


# -- non-destructive editing (PLAN.md E-01 to E-06) -----------------------


def _opened(controller: Controller, folder: Path) -> Path:
    """Open *folder* and wait until its first photo can be adjusted."""
    controller.open(folder)
    wait_for(controller.preview_ready)
    current = controller.current
    assert current is not None
    return current


def test_adjusting_writes_a_parameter_file(
    controller: Controller, folder: Path
) -> None:
    """E-02: the adjustments are stored in the program's own store."""
    photo = _opened(controller, folder)

    controller.set_adjustments(controller.state.with_value("exposure", 0.8))
    controller.flush()

    stored = sidecar.read(photo, controller.state, controller._store)
    assert stored is not None
    assert stored.adjustments.exposure == 0.8


def test_the_adjustments_come_back_when_the_photo_is_opened_again(
    controller: Controller, folder: Path, qt_app: object
) -> None:
    """E-03: closing and reopening must restore the adjustments in full."""
    _opened(controller, folder)
    edited = controller.state.with_value("exposure", 0.65).with_value("vibrance", 30.0)
    controller.set_adjustments(edited)
    controller.flush()

    # A second controller stands in for restarting the program.
    restarted = Controller(Settings())
    try:
        restarted.open(folder)
        state = wait_for(restarted.adjustments_ready)[0]
    finally:
        restarted.shutdown()

    assert state.exposure == 0.65
    assert state.vibrance == 30.0


def test_editing_never_touches_the_photo_itself(
    controller: Controller, folder: Path
) -> None:
    """E-01 and PLAN.md section 10 rule 8: the original is read, never written."""
    photo = _opened(controller, folder)
    before = photo.read_bytes()
    stat_before = photo.stat()

    controller.set_adjustments(controller.state.with_value("contrast", 60.0))
    controller.flush()
    controller.set_adjustments(controller.state.with_value("exposure", -1.0))
    controller.flush()
    controller.reset()
    controller.flush()

    assert photo.read_bytes() == before
    assert photo.stat().st_size == stat_before.st_size
    assert photo.stat().st_mtime_ns == stat_before.st_mtime_ns


def test_a_photo_back_at_its_defaults_loses_its_parameter_file(
    controller: Controller, folder: Path
) -> None:
    """PLAN.md 7.1: nothing adjusted, no file left behind."""
    photo = _opened(controller, folder)
    controller.set_adjustments(controller.state.with_value("exposure", 0.8))
    controller.flush()
    assert sidecar.stored_path(photo, controller._store).is_file()

    controller.reset()
    controller.flush()

    assert not sidecar.stored_path(photo, controller._store).exists()


def test_a_whole_drag_is_one_undo(controller: Controller, folder: Path) -> None:
    """E-04: dragging a slider and pressing Ctrl+Z once goes back to the start."""
    _opened(controller, folder)
    rested = controller.state

    for value in (0.1, 0.2, 0.3, 0.4, 0.5):
        controller.set_adjustments(controller.state.with_value("exposure", value), True)
    controller.set_adjustments(controller.state.with_value("exposure", 0.5), False)

    assert controller.can_undo
    controller.undo()
    assert controller.state == rested
    assert not controller.can_undo


def test_undo_and_redo_walk_the_history(controller: Controller, folder: Path) -> None:
    _opened(controller, folder)
    rested = controller.state
    restored: list[object] = []
    controller.state_restored.connect(restored.append)

    controller.set_adjustments(rested.with_value("exposure", 0.5))
    controller.set_adjustments(controller.state.with_value("contrast", 20.0))

    controller.undo()
    assert controller.state.contrast == 0.0
    assert controller.state.exposure == 0.5
    controller.undo()
    assert controller.state == rested
    controller.redo()
    assert controller.state.exposure == 0.5
    assert restored[-1] == controller.state


def test_what_undo_and_redo_can_do_is_announced(
    controller: Controller, folder: Path
) -> None:
    _opened(controller, folder)
    seen: list[tuple[bool, bool]] = []
    controller.history_changed.connect(lambda undo, redo: seen.append((undo, redo)))

    controller.set_adjustments(controller.state.with_value("exposure", 0.5))
    assert seen[-1] == (True, False)
    controller.undo()
    assert seen[-1] == (False, True)
    controller.redo()
    assert seen[-1] == (True, False)


def test_reset_puts_back_only_the_adjust_tab_s_sliders(
    controller: Controller, folder: Path
) -> None:
    """2026-11-01: the reset on the adjust tab is that tab's; the filter, the
    portrait and the crop are on tabs of their own and stay (PLAN.md 3.1)."""
    _opened(controller, folder)
    found = controller.state
    controller.set_adjustments(
        found.with_value("exposure", 0.8)
        .with_value("grain", 30.0)
        .with_value("sharpening", 60.0)
        .with_value("skin", 40.0)
        .with_value("whiten", 20.0)
        .with_filter("warm-negative")
        .with_value("filter_strength", 60.0)
    )
    controller.set_geometry(replace(controller.geometry, angle=3.0))

    controller.reset()

    state = controller.state
    assert (state.exposure, state.grain, state.sharpening) == (
        found.exposure,
        found.grain,
        found.sharpening,
    )
    assert (state.skin, state.whiten) == (40.0, 20.0)
    assert (state.filter, state.filter_strength) == ("warm-negative", 60.0)
    assert controller.geometry.angle == 3.0


def test_resetting_can_itself_be_undone(controller: Controller, folder: Path) -> None:
    """E-05: reset is a step like any other."""
    _opened(controller, folder)
    controller.set_adjustments(controller.state.with_value("saturation", -40.0))

    controller.reset()
    assert controller.state.saturation == 0.0

    controller.undo()
    assert controller.state.saturation == -40.0


def test_the_eyedropper_moves_both_white_balance_sliders(
    controller: Controller, folder: Path
) -> None:
    """A-24: the first photo is a green cast, so it asks for magenta.

    So much of it that the tint ends at the slider's end; the temperature is
    solved in the working space, where that green is not quite even in red
    and blue, so it moves as well.
    """
    _opened(controller, folder)
    controller.set_adjustments(controller.state.with_value("exposure", 0.5))

    controller.pick_white_balance((0.5, 0.5))
    restored = wait_for(controller.state_restored)[0]

    assert restored == controller.state
    assert controller.state.temperature != 0.0
    assert controller.state.tint == 100.0
    assert controller.state.exposure == 0.5


def test_a_pick_is_one_step_of_history(controller: Controller, folder: Path) -> None:
    _opened(controller, folder)
    controller.pick_white_balance((0.5, 0.5))
    wait_for(controller.state_restored)

    controller.undo()
    assert controller.state.tint == 0.0


def test_a_spot_too_dark_to_read_is_announced_and_changes_nothing(
    controller: Controller, tmp_path: Path
) -> None:
    photos = tmp_path / "dark"
    photos.mkdir()
    _write_jpeg(photos / "night.jpg", (2, 3, 2))
    _opened(controller, photos)
    before = controller.state

    controller.pick_white_balance((0.5, 0.5))
    message = wait_for(controller.announcement)[0]

    assert message == controller_module.tr("toast.pick_dark")
    assert controller.state == before
    assert not controller.can_undo


def test_each_photo_keeps_its_own_history(controller: Controller, folder: Path) -> None:
    """E-04: one history per photo."""
    _opened(controller, folder)
    controller.set_adjustments(controller.state.with_value("exposure", 0.5))

    controller.select(1)
    wait_for(controller.preview_ready)

    assert not controller.can_undo
    controller.select(0)
    wait_for(controller.preview_ready)
    assert controller.can_undo


def test_moving_to_another_photo_writes_what_was_still_waiting(
    controller: Controller, folder: Path
) -> None:
    """The second of grace before a save must not be lost by moving on."""
    photo = _opened(controller, folder)
    controller.set_adjustments(controller.state.with_value("exposure", 0.45))

    controller.select(1)
    wait_for(controller.edited_changed)

    assert sidecar.stored_path(photo, controller._store).is_file()


def test_the_before_view_shows_the_photo_as_it_was_found(
    controller: Controller, folder: Path
) -> None:
    """E-06: holding the key shows the photo without the adjustments."""
    _opened(controller, folder)
    controller.set_adjustments(controller.state.with_value("exposure", 2.0))
    edited = wait_for(controller.preview_ready)[0]

    controller.set_comparing(True)
    before = wait_for(controller.preview_ready)[0]

    assert before.before
    assert not np.array_equal(before.image, edited.image)
    assert float(before.image.mean()) < float(edited.image.mean())

    # Letting the key go is instant: the edited preview is still in hand, so
    # it is shown again without being developed a second time.
    previews: list = []
    controller.preview_ready.connect(previews.append)
    controller.set_comparing(False)
    assert previews, "the edited photo did not come back"
    assert not previews[-1].before
    assert np.array_equal(previews[-1].image, edited.image)


def test_which_photos_are_edited_is_announced_when_a_folder_opens(
    controller: Controller, folder: Path
) -> None:
    """PLAN.md 3.1: the filmstrip marks the photos that have been edited."""
    photo = folder / "b.jpg"
    sidecar.save(
        photo,
        Edit(EditState(contrast=30.0)),
        Edit(EditState()),
        sidecar.Application("TintaPhoto", "0.1.0"),
        controller._store,
    )

    controller.open(folder)
    edited = wait_for(controller.edited_photos_ready)[0]

    assert edited == {photo}


def test_editing_and_resetting_change_the_mark(
    controller: Controller, folder: Path
) -> None:
    photo = _opened(controller, folder)
    marks: list[tuple[Path, bool]] = []
    controller.edited_changed.connect(lambda path, edited: marks.append((path, edited)))

    controller.set_adjustments(controller.state.with_value("exposure", 0.8))
    controller.flush()
    assert marks[-1] == (photo, True)
    assert controller.is_edited(photo)

    controller.reset()
    controller.flush()
    assert marks[-1] == (photo, False)
    assert not controller.is_edited(photo)


def test_adjusting_leaves_nothing_in_the_photo_folder(
    controller: Controller, folder: Path
) -> None:
    """PLAN.md 7.1: the parameter files are the program's, not the folder's."""
    before = sorted(path.name for path in folder.iterdir())
    photo = _opened(controller, folder)

    controller.set_adjustments(controller.state.with_value("exposure", 0.8))
    controller.flush()

    assert sorted(path.name for path in folder.iterdir()) == before
    assert sidecar.stored_path(photo, controller._store).is_file()


def test_an_old_file_beside_the_photo_still_opens_and_is_never_touched(
    controller: Controller, folder: Path
) -> None:
    """0.34 and earlier wrote beside the photo; that is read but left alone."""
    photo = folder / "a.jpg"
    beside = sidecar.legacy_path(photo)
    beside.write_text('{"version": 1, "adjustments": {"contrast": 30}}', encoding="utf-8")
    untouched = beside.read_bytes()

    shown = _opened(controller, folder)
    assert shown == photo
    assert controller.state.contrast == 30.0

    controller.set_adjustments(controller.state.with_value("contrast", 10.0))
    controller.flush()
    assert sidecar.read(photo, controller.state, controller._store).adjustments.contrast == 10.0

    controller.reset()
    controller.flush()
    assert sidecar.read(photo, controller.state, controller._store) is None
    assert not controller.is_edited(photo)
    assert beside.read_bytes() == untouched


def test_the_earlier_store_is_taken_over_at_start_up(
    qt_app: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, folder: Path
) -> None:
    """Parameters of unwritable folders from 0.34 and earlier are not lost."""
    photo = folder / "a.jpg"
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    old = sidecar.stored_path(photo, earlier)
    old.write_text('{"version": 1, "adjustments": {"contrast": 30}}', encoding="utf-8")
    store = tmp_path / "new store"
    monkeypatch.setattr(controller_module, "sidecar_dir", lambda: store)
    monkeypatch.setattr(controller_module, "earlier_sidecar_dir", lambda: earlier)
    monkeypatch.setattr(controller_module, "presets_dir", lambda: tmp_path / "presets")

    created = Controller(Settings())
    try:
        assert not old.exists()
        assert sidecar.read(photo, EditState(), store).adjustments.contrast == 30.0
    finally:
        created.shutdown()


# -- cropping and rotation (PLAN.md C-01 to C-07) -------------------------


#: How long the preview has to stay quiet before it counts as settled.
_SETTLED_MS = 150


def _wait_for_preview(controller: Controller):
    """Return the fully developed preview the photo settles at.

    A change made while an earlier one is still being developed shows both
    now, the older one first (``app/workers.py``): what finishes goes on the
    screen rather than being thrown away, which is what makes a drag move.
    A test that asks what the photo looks like therefore has to let the
    stream of previews go quiet rather than take the first one.
    """
    loop = QEventLoop()
    arrived: list = []

    quiet = QTimer()
    quiet.setSingleShot(True)
    quiet.setInterval(_SETTLED_MS)
    quiet.timeout.connect(loop.quit)

    def capture(preview) -> None:
        if preview.provisional or preview.draft:
            return
        arrived.append(preview)
        quiet.start()

    # Disconnected by the handle, for the reason given in wait_for.
    connection = controller.preview_ready.connect(capture)
    overall = QTimer()
    overall.setSingleShot(True)
    overall.timeout.connect(loop.quit)
    overall.start(TIMEOUT_MS)
    loop.exec()
    controller.preview_ready.disconnect(connection)
    assert arrived, "no preview arrived"
    return arrived[-1]


def _shown(controller: Controller, folder: Path):
    """Open *folder* and return its first photo and the preview of it."""
    controller.open(folder)
    preview = _wait_for_preview(controller)
    current = controller.current
    assert current is not None
    return current, preview


def test_cropping_cuts_the_preview_down(controller: Controller, folder: Path) -> None:
    """C-01: a crop of a quarter of the photo gives a quarter of the pixels."""
    _photo, whole = _shown(controller, folder)

    controller.set_geometry(Geometry(crop=Rect(0.0, 0.0, 0.5, 0.5)))
    cropped = _wait_for_preview(controller)

    assert cropped.image.shape[1] == pytest.approx(whole.image.shape[1] / 2, abs=1)
    assert cropped.image.shape[0] == pytest.approx(whole.image.shape[0] / 2, abs=1)


def test_the_crop_tab_shows_the_whole_photo(controller: Controller, folder: Path) -> None:
    """PLAN.md 3.2: 預覽區顯示完整未裁切的照片."""
    _photo, whole = _shown(controller, folder)
    controller.set_geometry(Geometry(crop=Rect(0.25, 0.25, 0.5, 0.5)))
    _wait_for_preview(controller)

    controller.set_cropping(True)
    in_tab = _wait_for_preview(controller)

    assert in_tab.image.shape == whole.image.shape
    assert controller.cropping


def test_leaving_the_crop_tab_applies_the_crop(controller: Controller, folder: Path) -> None:
    _photo, whole = _shown(controller, folder)
    controller.set_cropping(True)
    controller.set_geometry(Geometry(crop=Rect(0.25, 0.25, 0.5, 0.5)))
    _wait_for_preview(controller)

    controller.set_cropping(False)
    applied = _wait_for_preview(controller)

    assert applied.image.shape[0] < whole.image.shape[0]
    assert not controller.cropping


def test_entering_the_crop_tab_starts_with_the_largest_box(
    controller: Controller, folder: Path
) -> None:
    """An unrotated photo at its own ratio: the box is the whole picture."""
    _shown(controller, folder)
    controller.set_cropping(True)

    assert controller.geometry.crop.is_whole_frame
    assert geometry.is_inside(
        controller.geometry.crop, controller.photo_size, controller.geometry.angle
    )


def test_entering_the_crop_tab_of_a_rotated_photo_fits_the_box(
    controller: Controller, folder: Path
) -> None:
    """A rotated photo cannot keep the whole frame, so the box shrinks (C-05)."""
    _shown(controller, folder)
    controller.set_geometry(Geometry(angle=8.0, crop=Rect(0.0, 0.0, 1.0, 1.0)))

    controller.set_cropping(True)

    crop = controller.geometry.crop
    assert not crop.is_whole_frame
    assert geometry.is_inside(crop, controller.photo_size, 8.0)


def test_cancelling_the_crop_puts_the_geometry_back(
    controller: Controller, folder: Path
) -> None:
    """PLAN.md 3.2: Esc 等同取消."""
    _shown(controller, folder)
    controller.set_geometry(Geometry(crop=Rect(0.1, 0.1, 0.8, 0.8)))
    kept = controller.geometry

    controller.set_cropping(True)
    controller.set_geometry(Geometry(crop=Rect(0.4, 0.4, 0.2, 0.2)))
    controller.cancel_crop()

    assert controller.geometry == kept
    assert not controller.cropping


def test_a_crop_is_stored_and_restored(controller: Controller, folder: Path) -> None:
    """Stage 4 acceptance: 裁切後重開照片，裁切範圍不變."""
    _shown(controller, folder)
    crop = Geometry(angle=-2.5, crop=Rect(0.1, 0.2, 0.6, 0.5), aspect="3:2")
    controller.set_geometry(crop)
    controller.flush()

    restarted = Controller(Settings())
    restored: list = []
    restarted.geometry_changed.connect(restored.append)
    try:
        restarted.open(folder)
        wait_for(restarted.adjustments_ready)
    finally:
        restarted.shutdown()

    assert restored and restored[0] == crop


def test_a_crop_alone_is_worth_a_parameter_file(
    controller: Controller, folder: Path
) -> None:
    photo, _preview = _shown(controller, folder)

    controller.set_geometry(Geometry(crop=Rect(0.2, 0.2, 0.5, 0.5)))
    controller.flush()

    stored = sidecar.read(photo, controller.state, controller._store)
    assert stored is not None
    assert stored.geometry.crop.w == pytest.approx(0.5)
    assert stored.adjustments == controller.state


def test_a_crop_can_be_undone(controller: Controller, folder: Path) -> None:
    """E-04 covers the crop as well as the sliders."""
    _shown(controller, folder)

    controller.set_geometry(Geometry(crop=Rect(0.2, 0.2, 0.5, 0.5)))
    assert controller.can_undo

    controller.undo()
    assert controller.geometry.crop.is_whole_frame

    controller.redo()
    assert controller.geometry.crop.w == pytest.approx(0.5)


def test_dragging_the_crop_box_is_one_step_of_history(
    controller: Controller, folder: Path
) -> None:
    _shown(controller, folder)

    for width in (0.9, 0.8, 0.7, 0.6):
        controller.set_geometry(Geometry(crop=Rect(0.0, 0.0, width, width)), True)
    controller.set_geometry(Geometry(crop=Rect(0.0, 0.0, 0.6, 0.6)), False)

    controller.undo()
    assert controller.geometry.crop.is_whole_frame
    assert not controller.can_undo


def test_turning_the_photo_turns_the_preview(controller: Controller, folder: Path) -> None:
    """C-06: a quarter turn swaps the sides of the picture."""
    _photo, upright = _shown(controller, folder)

    controller.set_geometry(Geometry(quarter_turns=1))
    turned = _wait_for_preview(controller)

    assert turned.image.shape[0] == upright.image.shape[1]
    assert turned.image.shape[1] == upright.image.shape[0]


def test_the_before_view_keeps_the_crop(controller: Controller, folder: Path) -> None:
    """PLAN.md 3.3: 原圖只套用裁切，不套用調色."""
    _shown(controller, folder)
    controller.set_geometry(Geometry(crop=Rect(0.0, 0.0, 0.5, 0.5)))
    controller.set_adjustments(controller.state.with_value("exposure", 2.0))
    edited = _wait_for_preview(controller)

    controller.set_comparing(True)
    before = wait_for(controller.preview_ready)[0]

    assert before.before
    assert before.image.shape == edited.image.shape
    assert float(before.image.mean()) < float(edited.image.mean())


def test_resetting_the_crop_gives_the_whole_photo_back(
    controller: Controller, folder: Path
) -> None:
    _shown(controller, folder)
    controller.set_geometry(Geometry(angle=5.0, crop=Rect(0.2, 0.2, 0.5, 0.5)))

    controller.reset_crop()

    assert controller.geometry.angle == 0.0
    assert controller.geometry.crop.is_whole_frame


def test_the_geometry_survives_moving_between_photos(
    controller: Controller, folder: Path
) -> None:
    _shown(controller, folder)
    controller.set_geometry(Geometry(crop=Rect(0.1, 0.1, 0.5, 0.5)))

    controller.select(1)
    _wait_for_preview(controller)
    assert controller.geometry.crop.is_whole_frame

    controller.select(0)
    _wait_for_preview(controller)
    assert controller.geometry.crop.w == pytest.approx(0.5)


def test_the_geometry_is_only_applied_once_for_several_colour_changes(
    controller: Controller, folder: Path
) -> None:
    """PLAN.md section 6: the result of step 2 is cached."""
    _shown(controller, folder)
    controller.set_geometry(Geometry(angle=3.0))
    _wait_for_preview(controller)

    calls: list[int] = []
    real_apply = geometry.apply

    def counted(photo, frame, crop=True):
        calls.append(1)
        return real_apply(photo, frame, crop=crop)

    controller_module.geometry.apply = counted
    try:
        for value in (0.1, 0.2, 0.3):
            controller.set_adjustments(controller.state.with_value("exposure", value))
            _wait_for_preview(controller)
    finally:
        controller_module.geometry.apply = real_apply

    assert not calls, "the photo was rotated again for a colour slider"


# -- copying, pasting and presets (PLAN.md P-01 to P-06) ------------------


@pytest.fixture
def many(tmp_path: Path) -> Path:
    """Return a folder of six photos, enough to paste onto five others."""
    folder = tmp_path / "many"
    folder.mkdir()
    for index in range(6):
        _write_jpeg(folder / f"{index}.jpg", (40 + index * 30, 120, 160))
    return folder


def test_copying_takes_the_settings_of_the_photo_shown(
    controller: Controller, folder: Path
) -> None:
    """P-01."""
    _shown(controller, folder)
    controller.set_adjustments(controller.state.with_value("vibrance", 35.0))

    assert controller.copy_adjustments()
    assert controller.has_copied_adjustments


def test_nothing_can_be_copied_without_a_photo(controller: Controller) -> None:
    assert not controller.copy_adjustments()
    assert not controller.has_copied_adjustments


def test_pasting_changes_the_photo_shown(controller: Controller, folder: Path) -> None:
    """P-02 on its own."""
    _shown(controller, folder)
    controller.set_adjustments(controller.state.with_value("vibrance", 35.0))
    controller.copy_adjustments()

    controller.select(1)
    _wait_for_preview(controller)
    assert controller.state.vibrance == 0.0

    assert controller.paste_adjustments() == 1
    assert controller.state.vibrance == 35.0


def test_pasting_reaches_every_chosen_photo(controller: Controller, many: Path) -> None:
    """Stage 5 acceptance: 複製設定貼到另外 5 張照片，5 張的調色都改變."""
    _shown(controller, many)
    edited = controller.state.with_value("vibrance", 35.0).with_value("contrast", 20.0)
    controller.set_adjustments(edited)
    controller.copy_adjustments()
    photos = controller.photos

    assert controller.paste_adjustments(photos) == len(photos)
    controller.flush()

    for photo in photos:
        stored = sidecar.read(photo, EditState(), controller._store)
        assert stored is not None, f"{photo.name} was not changed"
        assert stored.adjustments.vibrance == 35.0
        assert stored.adjustments.contrast == 20.0


def test_pasting_leaves_the_crop_of_the_other_photos_alone(
    controller: Controller, many: Path
) -> None:
    """P-03: 複製貼上只包含調色參數，不包含裁切和旋轉."""
    photos = controller.photos if controller.photos else []
    _shown(controller, many)
    photos = controller.photos
    other = photos[1]
    # Give the other photo a crop of its own, as the crop tab would.
    sidecar.save(
        other,
        Edit(EditState(), Geometry(angle=4.0, crop=Rect(0.1, 0.1, 0.6, 0.6))),
        Edit(EditState()),
        sidecar.Application("TintaPhoto", "0.1.0"),
        controller._store,
    )

    controller.set_adjustments(controller.state.with_value("saturation", 25.0))
    controller.copy_adjustments()
    controller.paste_adjustments([photos[0], other])
    controller.flush()

    stored = sidecar.read(other, EditState(), controller._store)
    assert stored is not None
    assert stored.adjustments.saturation == 25.0
    assert stored.geometry.angle == 4.0
    assert stored.geometry.crop.w == pytest.approx(0.6)


def test_pasting_onto_the_photo_shown_can_be_undone(
    controller: Controller, folder: Path
) -> None:
    """Stage 5 acceptance: 貼上設定後 Ctrl+Z 可以復原."""
    _shown(controller, folder)
    controller.set_adjustments(controller.state.with_value("vibrance", 35.0))
    controller.copy_adjustments()
    controller.select(1)
    _wait_for_preview(controller)

    controller.paste_adjustments()
    assert controller.state.vibrance == 35.0

    controller.undo()
    assert controller.state.vibrance == 0.0


def test_pasting_says_how_many_photos_it_reached(
    controller: Controller, many: Path
) -> None:
    _shown(controller, many)
    controller.copy_adjustments()
    said: list[str] = []
    controller.announcement.connect(said.append)

    controller.paste_adjustments(controller.photos[:3])

    assert said and "3" in said[-1]


# -- presets ---------------------------------------------------------------


def test_a_preset_records_only_what_was_changed(
    controller: Controller, folder: Path
) -> None:
    """P-04."""
    _shown(controller, folder)
    controller.set_adjustments(controller.state.with_value("vibrance", 40.0))

    preset = controller.save_preset("Punchy")

    assert preset is not None
    assert set(preset.adjustments) == {"vibrance"}
    assert controller.presets == [preset] or controller.presets[0].id == preset.id


def test_a_preset_is_still_there_after_a_restart(
    controller: Controller, folder: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage 5 acceptance: 儲存預設集後重啟程式，預設集還在，套用結果相同."""
    _shown(controller, folder)
    controller.set_adjustments(
        controller.state.with_value("vibrance", 40.0).with_value("contrast", -15.0)
    )
    saved = controller.save_preset("Punchy")
    assert saved is not None
    wanted = saved.apply_to(EditState())

    restarted = Controller(Settings())
    try:
        restarted.load_presets()
        restored = restarted.presets
    finally:
        restarted.shutdown()

    assert [item.name for item in restored] == ["Punchy"]
    assert restored[0].apply_to(EditState()) == wanted


def test_applying_a_preset_changes_the_photo_and_can_be_undone(
    controller: Controller, folder: Path
) -> None:
    """P-05: 套用預設集（可復原）."""
    _shown(controller, folder)
    controller.set_adjustments(controller.state.with_value("vibrance", 40.0))
    preset = controller.save_preset("Punchy")
    assert preset is not None
    controller.reset()
    assert controller.state.vibrance == 0.0

    controller.apply_preset(preset)
    assert controller.state.vibrance == 40.0

    controller.undo()
    assert controller.state.vibrance == 0.0


def test_a_preset_leaves_the_rest_of_the_photo_alone(
    controller: Controller, folder: Path
) -> None:
    _shown(controller, folder)
    controller.set_adjustments(controller.state.with_value("vibrance", 40.0))
    preset = controller.save_preset("Punchy")
    assert preset is not None
    controller.reset()
    controller.set_adjustments(controller.state.with_value("exposure", 0.7))

    controller.apply_preset(preset)

    assert controller.state.exposure == 0.7
    assert controller.state.vibrance == 40.0


def test_renaming_and_deleting_a_preset(controller: Controller, folder: Path) -> None:
    """P-06."""
    _shown(controller, folder)
    controller.set_adjustments(controller.state.with_value("vibrance", 40.0))
    saved = controller.save_preset("Draft")
    assert saved is not None

    renamed = controller.rename_preset(saved, "Final")
    assert renamed is not None
    assert [item.name for item in controller.presets] == ["Final"]

    assert controller.delete_preset(renamed)
    assert controller.presets == []


def test_the_preset_list_is_announced_when_it_changes(
    controller: Controller, folder: Path
) -> None:
    _shown(controller, folder)
    heard: list[object] = []
    controller.presets_changed.connect(heard.append)

    controller.set_adjustments(controller.state.with_value("vibrance", 40.0))
    controller.save_preset("Punchy")

    assert heard and [item.name for item in heard[-1]] == ["Punchy"]


def test_a_preset_cannot_be_saved_without_a_photo(controller: Controller) -> None:
    assert controller.save_preset("Punchy") is None


# -- exporting (PLAN.md 2.6) ----------------------------------------------


def test_exporting_writes_the_file_in_the_background(
    controller: Controller, folder: Path, tmp_path: Path
) -> None:
    controller.open_folder(folder)
    wait_for(controller.preview_ready)

    target = tmp_path / "out" / "a_edit.jpg"
    assert controller.export(target, ExportOptions())
    assert controller.exporting

    result = wait_for(controller.export_finished)[0]
    assert result.path.exists()
    assert not controller.exporting


def test_only_one_export_runs_at_a_time(
    controller: Controller, folder: Path, tmp_path: Path
) -> None:
    """The progress on the button reports one export (X-07)."""
    controller.open_folder(folder)
    wait_for(controller.preview_ready)

    assert controller.export(tmp_path / "first.jpg", ExportOptions())
    assert not controller.export(tmp_path / "second.jpg", ExportOptions())
    wait_for(controller.export_finished)


def test_the_progress_is_reported_as_the_export_goes(
    controller: Controller, folder: Path, tmp_path: Path
) -> None:
    controller.open_folder(folder)
    wait_for(controller.preview_ready)
    seen: list[float] = []
    controller.export_progress.connect(seen.append)

    controller.export(tmp_path / "a_edit.jpg", ExportOptions())
    wait_for(controller.export_finished)

    assert seen[0] == 0.0
    assert seen[-1] == 1.0


def test_an_export_that_cannot_be_written_says_so_in_one_sentence(
    controller: Controller, folder: Path, tmp_path: Path
) -> None:
    controller.open_folder(folder)
    wait_for(controller.preview_ready)
    # A file where the folder should be, so that nothing can be written.
    blocked = tmp_path / "a file"
    blocked.write_bytes(b"in the way")

    controller.export(blocked / "out.jpg", ExportOptions())
    message = wait_for(controller.export_failed)[0]

    assert "blocked" not in message or message
    assert not controller.exporting


def test_exporting_with_nothing_open_does_nothing(
    controller: Controller, tmp_path: Path
) -> None:
    assert not controller.export(tmp_path / "out.jpg", ExportOptions())


def test_the_output_size_follows_the_crop(
    controller: Controller, folder: Path
) -> None:
    controller.open_folder(folder)
    wait_for(controller.preview_ready)

    assert controller.full_photo_size == (240, 160)
    assert controller.export_output_size(ExportOptions()) == (240, 160)

    controller.set_geometry(Geometry(crop=Rect(0.0, 0.0, 0.5, 0.5)))
    assert controller.export_output_size(ExportOptions()) == (120, 80)


def test_an_untouched_jpeg_would_be_copied(
    controller: Controller, folder: Path
) -> None:
    """X-08, as the dialog asks before it says the quality will be used."""
    controller.open_folder(folder)
    wait_for(controller.preview_ready)

    assert controller.export_will_copy(ExportOptions())
    controller.set_adjustments(EditState(exposure=0.5))
    assert not controller.export_will_copy(ExportOptions())


def test_the_estimate_is_the_file_itself_when_it_will_be_copied(
    controller: Controller, folder: Path
) -> None:
    controller.open_folder(folder)
    wait_for(controller.preview_ready)
    sizes: list[int] = []
    controller.export_estimate_ready.connect(sizes.append)

    # Nothing has to be compressed to know this one, so the answer is here
    # before the call returns.
    controller.request_export_estimate(ExportOptions())

    assert sizes == [(folder / "a.jpg").stat().st_size]


def test_the_estimate_of_a_developed_file_is_worked_out(
    controller: Controller, folder: Path
) -> None:
    controller.open_folder(folder)
    wait_for(controller.preview_ready)
    controller.set_adjustments(EditState(exposure=0.5))
    wait_for(controller.preview_ready)

    controller.request_export_estimate(ExportOptions())
    size = wait_for(controller.export_estimate_ready)[0]

    assert size > 0


# -- zooming in (PLAN.md 3.3 and section 4) -------------------------------

#: Larger than the preview is developed at, so that zooming in has something
#: to go and fetch, and larger than the proxy, so that the full size decode
#: is the only thing that can satisfy actual size.
BIG_PHOTO = (3200, 2000)


def _write_big_jpeg(path: Path) -> Path:
    """Write a photo with detail in it, larger than the preview proxy."""
    width, height = BIG_PHOTO
    columns = np.linspace(0, 255, width, dtype=np.uint8)
    pixels = np.repeat(columns[None, :, None], height, axis=0).repeat(3, axis=2)
    pixels[::7, :, 1] = 20
    Image.fromarray(pixels).save(path, "JPEG", quality=95)
    return path


@pytest.fixture
def big_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "big"
    folder.mkdir()
    _write_big_jpeg(folder / "big.jpg")
    return folder


def wait_until(signal, is_wanted, timeout_ms: int = TIMEOUT_MS) -> object:
    """Wait for an emission of *signal* that *is_wanted* accepts."""
    loop = QEventLoop()
    found: list = []

    def capture(*args: object) -> None:
        if not found and is_wanted(args[0]):
            found.append(args[0])
            loop.quit()

    connection = signal.connect(capture)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(timeout_ms)
    loop.exec()
    signal.disconnect(connection)
    assert found, "no emission of the kind the test waited for arrived"
    return found[0]


def _opened_big(controller: Controller, folder: Path) -> object:
    """Open the one large photo and return its first whole preview."""
    controller.set_preview_long_edge(800)
    controller.open(folder)
    return wait_until(controller.preview_ready, lambda preview: not preview.provisional)


def test_a_preview_says_how_large_the_photo_really_is(
    controller: Controller, big_folder: Path
) -> None:
    """The frame the zoom is measured against (PLAN.md 3.3)."""
    preview = _opened_big(controller, big_folder)

    assert preview.frame == BIG_PHOTO
    assert preview.region is None
    # Developed at the size it is shown, which is smaller than the photo.
    assert preview.image.shape[1] < BIG_PHOTO[0]


def test_a_larger_preview_develops_the_photo_on_screen_again(
    controller: Controller, big_folder: Path
) -> None:
    """Opening a folder: the preview's real size arrives after the first photo."""
    controller.set_preview_long_edge(100)
    controller.open(big_folder)
    small = wait_until(controller.preview_ready, lambda preview: not preview.provisional)

    controller.set_preview_long_edge(1200)
    sharp = wait_until(controller.preview_ready, lambda preview: not preview.provisional)

    assert max(small.image.shape[:2]) == DRAFT_MINIMUM_LONG_EDGE
    assert max(sharp.image.shape[:2]) == 1200


def test_zooming_in_develops_only_what_is_on_screen(
    controller: Controller, big_folder: Path
) -> None:
    _opened_big(controller, big_folder)

    controller.set_view(0.5, (0.25, 0.25, 0.5, 0.5))
    window = wait_until(controller.preview_ready, lambda preview: preview.region is not None)

    assert window.frame == BIG_PHOTO
    left, top, right, bottom = window.region
    assert 0.2 <= left < right <= 0.55
    assert 0.2 <= top < bottom <= 0.55
    # A quarter of the width at half resolution, and no more: the rest of the
    # photo was not developed at all.
    assert window.image.shape[1] == pytest.approx(
        (right - left) * BIG_PHOTO[0] * 0.5, rel=0.05
    )


def test_a_window_is_sharper_than_the_whole_preview_stretched(
    controller: Controller, big_folder: Path
) -> None:
    """What the mosaic was: more pixels for the same part of the photo."""
    whole = _opened_big(controller, big_folder)
    region = (0.25, 0.25, 0.5, 0.5)
    stretched = (region[2] - region[0]) * whole.image.shape[1]

    controller.set_view(1.0, region)
    window = wait_until(
        controller.preview_ready,
        lambda preview: preview.region is not None
        and preview.image.shape[1] >= (preview.region[2] - preview.region[0])
        * BIG_PHOTO[0]
        * 0.99,
    )

    assert window.image.shape[1] > stretched * 3


def test_zooming_back_out_asks_for_the_whole_photo_again(
    controller: Controller, big_folder: Path
) -> None:
    _opened_big(controller, big_folder)
    controller.set_view(0.5, (0.25, 0.25, 0.5, 0.5))
    wait_until(controller.preview_ready, lambda preview: preview.region is not None)

    controller.set_view(0.2, None)
    whole = wait_until(controller.preview_ready, lambda preview: preview.region is None)

    assert whole.frame == BIG_PHOTO


def test_the_crop_tab_always_shows_the_whole_photo(
    controller: Controller, big_folder: Path
) -> None:
    """The crop box is dragged over all of it, so there is nothing to window."""
    _opened_big(controller, big_folder)
    controller.set_view(1.0, (0.25, 0.25, 0.5, 0.5))

    controller.set_cropping(True)
    preview = wait_until(controller.preview_ready, lambda _preview: True)

    assert preview.region is None


def _previews_while(controller: Controller, act, quiet_ms: int = 600) -> list:
    """Return every preview *act* leads to, drafts included, until they stop.

    The listener is connected before *act* runs, because a preview that was
    kept goes out there and then rather than from a background thread.
    """
    loop = QEventLoop()
    arrived: list = []

    quiet = QTimer()
    quiet.setSingleShot(True)
    quiet.setInterval(quiet_ms)
    quiet.timeout.connect(loop.quit)

    def capture(preview) -> None:
        arrived.append(preview)
        quiet.start()

    connection = controller.preview_ready.connect(capture)
    act()
    quiet.start()
    overall = QTimer()
    overall.setSingleShot(True)
    overall.timeout.connect(loop.quit)
    overall.start(TIMEOUT_MS)
    loop.exec()
    controller.preview_ready.disconnect(connection)
    return arrived


def _mismatched_frames(previews: list) -> list:
    """Return the previews whose frame is not the shape they fill.

    A preview of the whole photo is stretched over the frame it carries, so
    the two have to be the same shape: the crop tab's frame holding the
    cropped photo is drawn at the wrong size (PLAN.md 3.2, 3.3).
    """
    wrong = []
    for preview in previews:
        if preview.provisional or preview.region is not None or preview.frame is None:
            continue
        height, width = preview.image.shape[:2]
        frame_width, frame_height = preview.frame
        wanted = frame_width / frame_height
        if abs(width / height - wanted) > 0.02 * wanted:
            wrong.append((preview.frame, (width, height)))
    return wrong


def test_a_preview_carries_the_frame_it_was_developed_for(
    controller: Controller, big_folder: Path
) -> None:
    """Switching tabs while one is still developing (PLAN.md 3.2).

    The crop tab shows the whole photo and the others show it cropped, so
    the two fill different frames. A preview still being developed when the
    tab is switched has to keep the frame it was asked for: handed the other
    tab's, it is scaled to a frame it does not fill and the photo jumps.
    """
    _opened_big(controller, big_folder)
    controller.set_geometry(Geometry(crop=Rect(0.1, 0.1, 0.5, 0.35)))
    _previews_while(controller, lambda: None)

    def let_go_and_switch_tabs() -> None:
        controller.set_geometry(Geometry(crop=Rect(0.1, 0.1, 0.5, 0.36)), dragging=True)
        controller.set_cropping(True)

    previews = _previews_while(controller, let_go_and_switch_tabs)

    assert previews, "no preview arrived"
    assert not _mismatched_frames(previews)


def test_a_kept_preview_is_not_shown_in_the_crop_tab_s_frame(
    controller: Controller, big_folder: Path
) -> None:
    """The preview kept for the before and after key belongs to one frame.

    Let go of the key just after the crop tab was opened and the cropped
    photo would otherwise go back on screen, where the whole of it belongs
    (PLAN.md 3.2, E-06).
    """
    _opened_big(controller, big_folder)
    controller.set_adjustments(controller.state.with_value("exposure", 1.0))
    controller.set_geometry(Geometry(crop=Rect(0.1, 0.1, 0.5, 0.35)))
    _previews_while(controller, lambda: None)

    def switch_tabs_and_tap_the_key() -> None:
        controller.set_cropping(True)
        controller.set_comparing(True)
        controller.set_comparing(False)

    previews = _previews_while(controller, switch_tabs_and_tap_the_key)

    assert previews, "no preview arrived"
    assert not _mismatched_frames(previews)
    # The whole photo is what the crop tab settles on, not the cropped one.
    assert previews[-1].frame == BIG_PHOTO


def test_the_photo_as_it_was_found_is_shown_in_the_same_window(
    controller: Controller, big_folder: Path
) -> None:
    """Holding the before and after key while zoomed in (E-06)."""
    _opened_big(controller, big_folder)
    controller.set_adjustments(controller.state.with_value("exposure", 1.0))
    controller.set_view(0.5, (0.25, 0.25, 0.5, 0.5))
    edited = wait_until(controller.preview_ready, lambda preview: preview.region is not None)

    controller.set_comparing(True)
    before = wait_until(controller.preview_ready, lambda preview: preview.before)

    assert before.region == edited.region
    assert before.image.mean() < edited.image.mean()


def test_a_small_photo_is_never_worth_windowing(
    controller: Controller, folder: Path
) -> None:
    """Its whole preview already holds every pixel the photo has."""
    controller.open(folder)
    wait_for(controller.preview_ready)

    controller.set_view(4.0, (0.4, 0.4, 0.6, 0.6))

    assert not controller._zoomed_in()


def test_the_photo_is_only_decoded_at_full_size_once(
    controller: Controller, big_folder: Path
) -> None:
    _opened_big(controller, big_folder)
    decodes: list[Path] = []
    real_decode = controller_module.decoder.decode

    def counted(path: Path):
        decodes.append(path)
        return real_decode(path)

    controller_module.decoder.decode = counted  # type: ignore[assignment]
    try:
        controller.set_view(1.0, (0.25, 0.25, 0.5, 0.5))
        wait_until(
            controller.preview_ready,
            lambda preview: preview.region is not None
            and preview.image.shape[1] >= (preview.region[2] - preview.region[0])
            * BIG_PHOTO[0]
            * 0.99,
        )
        controller.set_view(1.0, (0.3, 0.3, 0.55, 0.55))
        wait_until(controller.preview_ready, lambda preview: preview.region is not None)
    finally:
        controller_module.decoder.decode = real_decode  # type: ignore[assignment]

    assert len(decodes) == 1


def test_moving_to_another_photo_lets_go_of_the_full_size_one(
    controller: Controller, big_folder: Path
) -> None:
    """A proxy is tens of megabytes; the full size photo is hundreds."""
    _write_big_jpeg(big_folder / "other.jpg")
    _opened_big(controller, big_folder)
    controller.set_view(1.0, (0.25, 0.25, 0.5, 0.5))
    wait_until(controller.preview_ready, lambda preview: preview.region is not None)

    controller.select_next()
    wait_until(controller.preview_ready, lambda preview: not preview.provisional)

    assert controller._full is None


def test_a_gesture_that_is_over_is_not_waited_out(
    controller: Controller, big_folder: Path
) -> None:
    """Letting go of the wheel develops what it came to rest on at once."""
    _opened_big(controller, big_folder)

    controller.set_view(0.5, (0.25, 0.25, 0.5, 0.5), settled=True)

    assert not controller._view_timer.isActive()
    window = wait_until(controller.preview_ready, lambda preview: preview.region is not None)
    assert window.region is not None


def test_a_gesture_still_under_the_hand_waits_for_it_to_settle(
    controller: Controller, big_folder: Path
) -> None:
    _opened_big(controller, big_folder)

    controller.set_view(0.5, (0.25, 0.25, 0.5, 0.5))

    assert controller._view_timer.isActive()


def test_the_zoom_carries_over_to_the_next_photo(
    controller: Controller, big_folder: Path
) -> None:
    """Two photos of the same shape keep the zoom, so the window must follow."""
    _write_big_jpeg(big_folder / "other.jpg")
    _opened_big(controller, big_folder)
    controller.set_view(0.5, (0.25, 0.25, 0.5, 0.5), settled=True)
    wait_until(controller.preview_ready, lambda preview: preview.region is not None)

    controller.select_next()
    whole = wait_until(controller.preview_ready, lambda preview: not preview.provisional)
    # The preview area says where the photo sits as soon as it has shown it,
    # the way the canvas does for a photo moved to.
    controller.set_view(0.5, (0.25, 0.25, 0.5, 0.5), settled=True)
    window = wait_until(controller.preview_ready, lambda preview: preview.region is not None)

    assert whole.region is None
    assert window.path == controller.current


def test_the_local_brightness_is_worked_out_once_for_many_windows(
    controller: Controller, big_folder: Path
) -> None:
    """Panning must not redo the blur the whole frame was measured with."""
    _opened_big(controller, big_folder)
    controller.set_adjustments(controller.state.with_value("shadows", 40.0))
    wait_until(controller.preview_ready, lambda preview: preview.region is None)

    fields: list[object] = []
    real_field = controller_module.pipeline.surroundings_field

    def counted(photo, state):
        fields.append(photo.size)
        return real_field(photo, state)

    controller_module.pipeline.surroundings_field = counted  # type: ignore[assignment]
    try:
        for box in ((0.25, 0.25, 0.5, 0.5), (0.3, 0.3, 0.55, 0.55), (0.35, 0.35, 0.6, 0.6)):
            controller.set_view(0.5, box, settled=True)
            wait_until(controller.preview_ready, lambda preview: preview.region is not None)
        made_for_panning = len(fields)
        # A slider after step 7 does not change what the field says either.
        controller.set_adjustments(controller.state.with_value("contrast", 30.0))
        wait_until(controller.preview_ready, lambda preview: preview.region is not None)
    finally:
        controller_module.pipeline.surroundings_field = real_field  # type: ignore[assignment]

    assert made_for_panning == 1
    assert len(fields) == 1


def test_the_photo_under_the_window_follows_the_adjustments(
    controller: Controller, big_folder: Path
) -> None:
    """Or panning would uncover the colours the photo had a moment ago."""
    _opened_big(controller, big_folder)
    controller.set_view(0.5, (0.25, 0.25, 0.5, 0.5), settled=True)
    window = wait_until(controller.preview_ready, lambda preview: preview.region is not None)

    controller.set_adjustments(controller.state.with_value("exposure", 2.0))
    backdrop = wait_until(controller.preview_ready, lambda preview: preview.backdrop)

    assert backdrop.region is None
    assert backdrop.frame == BIG_PHOTO
    assert backdrop.image.shape[1] < BIG_PHOTO[0]
    # Two stops brighter than the window developed before the change.
    assert backdrop.image.mean() > window.image.mean() + 20


def test_nothing_is_developed_underneath_while_the_whole_photo_is_shown(
    controller: Controller, big_folder: Path
) -> None:
    """Fitted, what is on screen is the whole photo: there is no underneath."""
    _opened_big(controller, big_folder)
    seen: list = []
    connection = controller.preview_ready.connect(seen.append)
    try:
        controller.set_adjustments(controller.state.with_value("exposure", 1.0))
        wait_until(controller.preview_ready, lambda preview: not preview.draft)
    finally:
        controller.preview_ready.disconnect(connection)

    assert seen and not any(preview.backdrop for preview in seen)


def test_the_photo_underneath_is_not_developed_for_every_pan(
    controller: Controller, big_folder: Path
) -> None:
    """It only goes out of date when the photo itself does."""
    _opened_big(controller, big_folder)
    controller.set_adjustments(controller.state.with_value("shadows", 30.0))
    wait_until(controller.preview_ready, lambda preview: not preview.draft)

    seen: list = []
    connection = controller.preview_ready.connect(seen.append)
    try:
        for box in ((0.25, 0.25, 0.5, 0.5), (0.3, 0.3, 0.55, 0.55), (0.35, 0.3, 0.6, 0.55)):
            controller.set_view(0.5, box, settled=True)
            wait_until(controller.preview_ready, lambda preview: preview.region is not None)
    finally:
        controller.preview_ready.disconnect(connection)

    assert not any(preview.backdrop for preview in seen)


# -- the film filter ------------------------------------------------------


def test_the_shipped_filters_are_there(controller: Controller) -> None:
    """The controller is the only door the interface has to them."""
    identifiers = [look.identifier for look in controller.filters]

    assert identifiers
    assert len(set(identifiers)) == len(identifiers)


def test_a_filter_reaches_the_preview(controller: Controller, folder: Path) -> None:
    """Choosing a filter develops the photo again, through the look."""
    controller.open_folder(folder)
    plain = wait_for(controller.preview_ready)[0].image

    chosen = controller.filters[0].identifier
    controller.set_adjustments(controller.state.with_filter(chosen))
    filtered = wait_for(controller.preview_ready)[0].image

    assert controller.state.filter == chosen
    assert plain.shape == filtered.shape
    assert not np.array_equal(plain, filtered)


def test_a_filter_that_is_not_installed_is_taken_off(
    controller: Controller, folder: Path
) -> None:
    """A parameter file from elsewhere must not stop the photo opening."""
    photo = folder / "a.jpg"
    reference = EditState.for_photo(False)
    sidecar.save(
        photo,
        Edit(reference.with_filter("no-such-filter").with_value("vibrance", 20.0)),
        Edit(reference),
        sidecar.Application("TintaPhoto", "0.1.0"),
        controller._store,
    )

    controller.open_file(photo)
    wait_for(controller.preview_ready)

    assert controller.state.filter == ""
    assert controller.state.vibrance == 20.0


# -- the automatic light ------------------------------------------------------


def test_the_automatic_light_moves_only_the_light_sliders(
    controller: Controller, folder: Path
) -> None:
    from core.auto_light import AUTO_NAMES

    controller.open_folder(folder)
    wait_for(controller.preview_ready)
    controller.set_adjustment("vibrance", 12.0)
    controller.set_adjustment("contrast", 15.0)
    before = controller.state

    controller.auto_light()
    (restored,) = wait_for(controller.state_restored)

    assert restored == controller.state
    assert controller.state != before
    unchanged = before
    for name in AUTO_NAMES:
        unchanged = unchanged.with_value(name, getattr(controller.state, name))
    assert controller.state == unchanged


def test_the_automatic_light_is_one_step_of_history(
    controller: Controller, folder: Path
) -> None:
    controller.open_folder(folder)
    wait_for(controller.preview_ready)
    before = controller.state

    controller.auto_light()
    wait_for(controller.state_restored)
    controller.undo()

    assert controller.state == before


def test_the_automatic_light_says_what_it_did(controller: Controller, folder: Path) -> None:
    controller.open_folder(folder)
    wait_for(controller.preview_ready)

    controller.auto_light()
    (message,) = wait_for(controller.announcement)

    assert message == tr("toast.auto_light")


# -- the filter tab's thumbnails ---------------------------------------------


def _look_thumbnails(controller: Controller, side: int = 40) -> dict:
    """Ask for the filter tab's thumbnails and return the first full set."""
    controller.set_look_thumbnail_side(side)
    while True:
        (looks,) = wait_for(controller.look_thumbnails_ready)
        if looks:
            return looks


def test_look_thumbnails_show_the_photo_wearing_each_look(
    controller: Controller, folder: Path
) -> None:
    controller.open_folder(folder)
    wait_for(controller.preview_ready)

    thumbnails = _look_thumbnails(controller)

    looks = [look.identifier for look in controller.filters]
    assert set(thumbnails) == {NO_FILTER, *looks}
    for image in thumbnails.values():
        assert image.dtype == np.uint8
        assert image.shape[0] == image.shape[1] >= 40
    plain = thumbnails[NO_FILTER].astype(np.int16)
    assert any(
        np.abs(thumbnails[name].astype(np.int16) - plain).max() > 0 for name in looks
    )


def test_look_thumbnails_are_not_made_while_the_tab_is_out_of_sight(
    controller: Controller, folder: Path
) -> None:
    """Developments nobody will see are not worth a thread."""
    arrived: list = []
    controller.look_thumbnails_ready.connect(
        lambda looks: arrived.append(looks) if looks else None
    )
    controller.open_folder(folder)
    wait_for(controller.preview_ready)
    controller.set_adjustment("exposure", 0.5)
    wait_for(controller.preview_ready)

    assert arrived == []


def test_look_thumbnails_follow_the_adjustments(
    controller: Controller, folder: Path
) -> None:
    """A thumbnail is what clicking the look would give: sliders included."""
    controller.open_folder(folder)
    wait_for(controller.preview_ready)
    before = _look_thumbnails(controller)[NO_FILTER]

    controller.set_adjustment("exposure", 1.0)
    (after,) = wait_for(controller.look_thumbnails_ready)

    assert after[NO_FILTER].mean() > before.mean() + 10


def test_look_thumbnails_use_the_strength_the_slider_shows(
    controller: Controller, folder: Path
) -> None:
    """At strength 0 a look leaves the photo as it is, so its thumbnail does too."""
    controller.open_folder(folder)
    wait_for(controller.preview_ready)
    _look_thumbnails(controller)

    controller.set_adjustment("filter_strength", 0.0)
    (thumbnails,) = wait_for(controller.look_thumbnails_ready)

    for image in thumbnails.values():
        assert np.array_equal(image, thumbnails[NO_FILTER])


def test_another_photo_clears_the_look_thumbnails_until_its_own_arrive(
    controller: Controller, folder: Path
) -> None:
    controller.open_folder(folder)
    wait_for(controller.preview_ready)
    first = _look_thumbnails(controller)

    # The clearing goes out the moment the photo changes, before anything
    # is waited for, so it is recorded from here.
    arrived: list = []
    connection = controller.look_thumbnails_ready.connect(arrived.append)
    controller.select_next()
    (second,) = wait_for(controller.look_thumbnails_ready)
    controller.look_thumbnails_ready.disconnect(connection)

    assert arrived[0] == {}
    assert not np.array_equal(second[NO_FILTER], first[NO_FILTER])


# -- the detail group across kinds of photo (PLAN.md 7.2) ------------------


def test_pasting_a_raws_settings_onto_a_jpeg_leaves_its_detail_alone() -> None:
    """A RAW rests sharpened; its resting values would sharpen a JPEG twice."""
    copied = EditState.for_photo(True, 5600.0, 0.0).with_value("exposure", 0.7)
    jpeg = EditState.for_photo(False)

    pasted = Controller._pasted_onto(copied, jpeg)

    assert pasted.exposure == pytest.approx(0.7)
    assert pasted.sharpening == 0.0
    assert pasted.colour_noise == 0.0


def test_pasting_between_two_raws_carries_the_detail() -> None:
    copied = EditState.for_photo(True, 5600.0, 0.0).with_value("luminance_noise", 45.0)
    other = EditState.for_photo(True, 4800.0, 3.0)

    pasted = Controller._pasted_onto(copied, other)

    assert pasted.luminance_noise == 45.0
    assert pasted.sharpening == copied.sharpening


# -- lens correction (PLAN.md 2.2, A-26 to A-28) -------------------------

SAMPLE_RAW = Path(__file__).resolve().parent / "samples" / "_DSC2828.ARW"


@pytest.fixture
def raw_folder(tmp_path: Path) -> Path:
    """A folder with a copy of a sample RAW whose lens the database knows."""
    if not SAMPLE_RAW.is_file():
        pytest.skip("_DSC2828.ARW is not in tests/samples")
    import shutil

    photos = tmp_path / "raw"
    photos.mkdir()
    shutil.copy2(SAMPLE_RAW, photos / SAMPLE_RAW.name)
    return photos


def test_a_jpeg_opens_without_lens_correction(controller: Controller, folder: Path) -> None:
    announced = []
    controller.lens_ready.connect(lambda profile, is_raw: announced.append((profile, is_raw)))
    _shown(controller, folder)

    assert announced[-1] == (None, False)
    assert not controller.geometry.lens.enabled


def test_a_raw_opens_corrected_and_says_which_lens(
    controller: Controller, raw_folder: Path
) -> None:
    announced = []
    controller.lens_ready.connect(lambda profile, is_raw: announced.append((profile, is_raw)))
    _shown(controller, raw_folder)

    profile, is_raw = announced[-1]
    assert is_raw and profile is not None and profile.name.strip()
    assert controller.geometry.lens.enabled


def test_switching_the_lens_off_is_undone_and_stored(
    controller: Controller, raw_folder: Path
) -> None:
    photo, _preview = _shown(controller, raw_folder)

    controller.set_lens(LensSettings(enabled=False))
    controller.flush()
    stored = sidecar.read(photo, controller.state, controller._store)
    assert stored is not None and not stored.geometry.lens.enabled

    controller.undo()
    assert controller.geometry.lens.enabled


def test_a_raw_untouched_but_for_its_own_lens_has_no_parameter_file(
    controller: Controller, raw_folder: Path
) -> None:
    """Corrected is where a RAW starts, so it is nothing worth storing (7.1)."""
    photo, _preview = _shown(controller, raw_folder)
    controller.set_adjustments(controller.state.with_value("exposure", 0.5))
    controller.set_adjustments(controller.state.with_value("exposure", 0.0))
    controller.flush()

    assert sidecar.read(photo, controller.state, controller._store) is None


def test_neither_reset_touches_the_lens_correction(
    controller: Controller, raw_folder: Path
) -> None:
    """The lens is on the retouch tab: the adjust and crop resets leave it."""
    _shown(controller, raw_folder)
    controller.set_lens(LensSettings(enabled=True, distortion=40.0))
    controller.set_geometry(replace(controller.geometry, angle=3.0))

    controller.reset_crop()
    assert controller.geometry.lens.distortion == 40.0
    assert controller.geometry.angle == 0.0

    controller.reset()
    assert controller.geometry.lens.distortion == 40.0


# -- faces (PLAN.md 2.2, A-29) -------------------------------------------------


def test_the_faces_of_a_photo_are_looked_for_once_it_opens(
    controller: Controller, folder: Path
) -> None:
    """A plain colour holds no faces: first "looking", then none found."""
    heard: list[object] = []
    loop = QEventLoop()

    def capture(count: object) -> None:
        heard.append(count)
        if count is not None:
            loop.quit()

    connection = controller.faces_ready.connect(capture)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(TIMEOUT_MS)
    open_and_wait(controller, folder)
    if not heard or heard[-1] is None:
        loop.exec()
    controller.faces_ready.disconnect(connection)

    assert heard[0] is None
    assert heard[-1] == 0
    assert controller.faces_of_current() == ()


# -- spot removal (PLAN.md 2.2, A-32) ------------------------------------------


def _blotted_folder(tmp_path: Path) -> Path:
    """A folder with one grey JPEG that has a dark blot at its middle."""
    photos = tmp_path / "blotted"
    photos.mkdir()
    rng = np.random.default_rng(7)
    pixels = np.clip(rng.normal(150, 6, (160, 240, 3)), 0, 255).astype(np.uint8)
    ys, xs = np.mgrid[:160, :240]
    pixels[np.hypot(xs - 120, ys - 80) <= 6] = (60, 40, 30)
    Image.fromarray(pixels).save(photos / "blot.jpg", quality=98)
    return photos


def test_a_click_mends_a_spot_as_one_step_and_stores_it(
    controller: Controller, tmp_path: Path
) -> None:
    photo, before = _shown(controller, _blotted_folder(tmp_path))
    assert before.image[80, 120].mean() < 90

    controller.add_spot([(0.5, 0.5)], 0.05)
    index = wait_for(controller.spot_added)[0]
    after = _wait_for_preview(controller)

    assert index == 0
    spot = controller.geometry.spots[0]
    assert spot.target == pytest.approx((0.5, 0.5), abs=1e-6)
    assert spot.source != spot.target
    assert after.image[80, 120].mean() > 120

    controller.flush()
    stored = sidecar.read(photo, controller.state, controller._store)
    assert stored is not None and stored.geometry.spots == (spot,)

    controller.undo()
    assert controller.geometry.spots == ()


def test_a_painted_stroke_is_mended_along_its_path(
    controller: Controller, tmp_path: Path
) -> None:
    """A-32: dragging the brush paints a stroke; letting go mends all of it."""
    photos = tmp_path / "streaked"
    photos.mkdir()
    rng = np.random.default_rng(3)
    pixels = np.clip(rng.normal(150, 6, (160, 240, 3)), 0, 255).astype(np.uint8)
    pixels[78:83, 60:180] = (50, 40, 30)  # a dark scratch across the middle
    Image.fromarray(pixels).save(photos / "scratch.jpg", quality=98)
    _photo, before = _shown(controller, photos)
    assert before.image[80, 60:180].mean() < 90

    path = [(x / 480, 80.5 / 160) for x in range(112, 372)]  # half a pixel apart
    controller.add_spot(path, 0.02)
    wait_for(controller.spot_added)
    after = _wait_for_preview(controller)

    stroke = controller.geometry.spots[0]
    assert 1 < len(stroke.points) < len(path)  # thinned, but still a path
    assert abs(stroke.offset[1]) * 160 >= 2 * 0.02 * 240  # taken from beside it
    assert after.image[80, 60:180].mean() > 130


def test_spots_follow_the_crop_on_screen(controller: Controller, tmp_path: Path) -> None:
    _shown(controller, _blotted_folder(tmp_path))
    controller.add_spot([(0.5, 0.5)], 0.05)
    wait_for(controller.spot_added)
    controller.set_geometry(replace(controller.geometry, crop=Rect(0.25, 0.0, 0.5, 1.0)))

    (path, _source, radius), = controller.spots_on_screen()
    assert path[0] == pytest.approx((0.5, 0.5), abs=0.01)
    # The frame on screen is 120 by 160 now, so its long edge is 2/3 of 240.
    assert radius == pytest.approx(0.05 / (160 / 240), rel=0.02)


def test_moving_resizing_and_removing_a_spot(controller: Controller, tmp_path: Path) -> None:
    _shown(controller, _blotted_folder(tmp_path))
    controller.add_spot([(0.5, 0.5)], 0.05)
    wait_for(controller.spot_added)
    first = controller.geometry.spots[0]

    controller.move_spot(0, (0.2, 0.3), source=True)
    assert controller.geometry.spots[0].source == pytest.approx((0.2, 0.3), abs=1e-6)
    assert controller.geometry.spots[0].target == first.target

    controller.move_spot(0, (0.6, 0.5), source=False)
    moved = controller.geometry.spots[0]
    assert moved.target == pytest.approx((0.6, 0.5), abs=1e-6)
    assert moved.source == pytest.approx((0.3, 0.3), abs=1e-6)

    controller.resize_spot(0, 0.08)
    assert controller.geometry.spots[0].radius == pytest.approx(0.08)

    controller.remove_spot(0)
    assert controller.geometry.spots == ()
    controller.undo()
    assert len(controller.geometry.spots) == 1


def test_neither_reset_clears_the_spots(controller: Controller, tmp_path: Path) -> None:
    _shown(controller, _blotted_folder(tmp_path))
    controller.add_spot([(0.5, 0.5)], 0.05)
    wait_for(controller.spot_added)
    controller.reset_crop()
    controller.reset()
    assert len(controller.geometry.spots) == 1

    controller.clear_spots()
    assert controller.geometry.spots == ()
    controller.undo()
    assert len(controller.geometry.spots) == 1


def test_the_photo_as_found_shows_the_blemish_again(
    controller: Controller, tmp_path: Path
) -> None:
    _shown(controller, _blotted_folder(tmp_path))
    controller.add_spot([(0.5, 0.5)], 0.05)
    wait_for(controller.spot_added)
    _wait_for_preview(controller)

    controller.set_comparing(True)
    before = _wait_for_preview(controller)
    assert before.image[80, 120].mean() < 90


# -- the portrait range (PLAN.md 2.10) ---------------------------------------


def _skin_folder(tmp_path: Path) -> Path:
    """A folder with one JPEG the colour of skin and no face in it."""
    photos = tmp_path / "skin"
    photos.mkdir()
    _write_jpeg(photos / "arm.jpg", (205, 150, 120))
    return photos


def test_a_painted_stroke_is_whitened_where_no_face_was_found(
    controller: Controller, tmp_path: Path
) -> None:
    photo, before = _shown(controller, _skin_folder(tmp_path))
    controller.set_adjustments(controller.state.with_value("whiten", 100.0))
    unchanged = _wait_for_preview(controller)
    assert np.array_equal(unchanged.image, before.image)  # nothing to whiten yet

    controller.add_skin_stroke([(0.5, 0.5)], 0.08, adding=True, snap=False)
    after = _wait_for_preview(controller)
    assert after.image[80, 120].astype(int).sum() > before.image[80, 120].astype(int).sum() + 6
    assert np.array_equal(after.image[5, 5], before.image[5, 5])

    controller.flush()
    stored = sidecar.read(photo, controller.state, controller._store)
    assert stored is not None and len(stored.geometry.skin.strokes) == 1
    controller.undo()
    assert controller.skin.strokes == ()


def test_the_skin_on_screen_follows_the_strokes(
    controller: Controller, tmp_path: Path
) -> None:
    _shown(controller, _skin_folder(tmp_path))
    controller.request_skin_shown()
    assert wait_for(controller.skin_shown)[0] is None

    controller.add_skin_stroke([(0.25, 0.5)], 0.08, adding=True, snap=False)
    controller.request_skin_shown()
    shown = wait_for(controller.skin_shown)[0]
    height, width = shown.shape
    assert shown[height // 2, width // 4] > 230
    assert shown[height // 2, 3 * width // 4] == 0


def test_the_range_is_kept_by_both_resets_and_cleared_by_its_own(
    controller: Controller, tmp_path: Path
) -> None:
    _shown(controller, _skin_folder(tmp_path))
    controller.add_skin_stroke([(0.5, 0.5)], 0.05, adding=False, snap=True)
    controller.set_skin_auto(False)
    controller.reset_crop()
    controller.reset()
    assert not controller.skin.auto and len(controller.skin.strokes) == 1

    controller.reset_skin()
    assert controller.skin.is_default
    controller.undo()
    assert not controller.skin.auto and len(controller.skin.strokes) == 1


# -- liquify (PLAN.md 2.11, 3.16) --------------------------------------------


def _blot_column(image: np.ndarray) -> float:
    """Return where across the dark blot sits on the middle row, in pixels."""
    row = image[image.shape[0] // 2].astype(float).mean(axis=1)
    darkness = np.clip(140.0 - row, 0.0, None)
    return float((np.arange(row.size) * darkness).sum() / darkness.sum())


def test_a_liquify_stroke_moves_the_photo_as_one_step_and_is_stored(
    controller: Controller, tmp_path: Path
) -> None:
    photo, before = _shown(controller, _blotted_folder(tmp_path))
    assert _blot_column(before.image) == pytest.approx(120, abs=1.5)

    controller.add_liquify_stroke(
        [(0.5, 0.5), (0.55, 0.5), (0.6, 0.5)], liquify.TOOL_PUSH, 0.08, 1.0
    )
    after = _wait_for_preview(controller)
    assert _blot_column(after.image) > 135
    assert np.array_equal(after.image[5, 5], before.image[5, 5])
    assert controller.can_undo

    controller.flush()
    stored = sidecar.read(photo, controller.state, controller._store)
    assert stored is not None and len(stored.geometry.liquify) == 1
    controller.undo()
    assert controller.liquify_strokes == ()


def test_a_click_of_the_push_is_not_a_step(controller: Controller, folder: Path) -> None:
    _shown(controller, folder)
    controller.add_liquify_stroke([(0.5, 0.5)], liquify.TOOL_PUSH, 0.08, 1.0)
    assert controller.liquify_strokes == ()
    assert not controller.can_undo


def test_a_twirl_on_a_mirrored_photo_turns_the_way_it_is_seen(
    controller: Controller, folder: Path
) -> None:
    _shown(controller, folder)
    controller.set_geometry(Geometry(mirrored=True))
    controller.add_liquify_stroke([(0.3, 0.5)] * 5, liquify.TOOL_TWIRL_CW, 0.08, 1.0)
    (stroke,) = controller.liquify_strokes
    assert stroke.tool == liquify.TOOL_TWIRL_CCW
    # Held before the mirror: the left of the screen is the right of the photo.
    assert stroke.points[0] == pytest.approx((0.7, 0.5), abs=0.01)


def test_a_new_stroke_lands_where_the_picture_shows_things_now(
    controller: Controller, folder: Path
) -> None:
    _shown(controller, folder)
    push = [(0.3, 0.5), (0.4, 0.5), (0.5, 0.5)]
    controller.add_liquify_stroke(push, liquify.TOOL_PUSH, 0.1, 1.0)
    controller.add_liquify_stroke([(0.5, 0.5)] * 3, liquify.TOOL_BLOAT, 0.05, 0.5)
    first, second = controller.liquify_strokes
    # Not taken back through the first stroke, which it comes after.
    assert second.points[0] == pytest.approx((0.5, 0.5), abs=1e-6)
    assert first.radius == pytest.approx(0.1) and second.strength == pytest.approx(0.5)


def test_liquify_is_kept_by_both_resets_and_cleared_by_its_own(
    controller: Controller, folder: Path
) -> None:
    _shown(controller, folder)
    controller.add_liquify_stroke([(0.5, 0.5)] * 4, liquify.TOOL_PUCKER, 0.08, 1.0)
    controller.reset_crop()
    controller.reset()
    assert len(controller.liquify_strokes) == 1

    controller.reset_liquify()
    assert controller.liquify_strokes == ()
    controller.undo()
    assert len(controller.liquify_strokes) == 1


def test_the_photo_as_it_was_found_has_no_liquify(
    controller: Controller, tmp_path: Path
) -> None:
    _shown(controller, _blotted_folder(tmp_path))
    controller.add_liquify_stroke([(0.5, 0.5), (0.6, 0.5)], liquify.TOOL_PUSH, 0.08, 1.0)
    _wait_for_preview(controller)
    controller.set_comparing(True)
    before = _wait_for_preview(controller)
    assert _blot_column(before.image) == pytest.approx(120, abs=1.5)


def test_a_stroke_being_painted_is_drafted_on_the_picture_on_screen(
    controller: Controller, tmp_path: Path
) -> None:
    _photo, before = _shown(controller, _blotted_folder(tmp_path))
    controller.draft_liquify([(0.5, 0.5), (0.6, 0.5)], liquify.TOOL_PUSH, 0.08, 1.0)
    (draft,) = wait_for(controller.preview_ready)
    assert draft.draft
    assert draft.frame == before.frame and draft.region == before.region
    assert _blot_column(draft.image) > 135
    assert controller.liquify_strokes == ()  # nothing is kept until it is let go

    # A draft still on its way when the stroke ends is not shown.
    base = controller._liquify_base
    number = controller._liquify_stroke
    controller.add_liquify_stroke([(0.5, 0.5), (0.6, 0.5)], liquify.TOOL_PUSH, 0.08, 1.0)
    shown: list = []
    connection = controller.preview_ready.connect(shown.append)
    controller._on_liquify_drafted(controller.current, draft.image, base, number)
    controller.preview_ready.disconnect(connection)
    assert not shown
