"""Measure the performance targets of PLAN.md section 11.

Run it from the project root, with photos in ``tests/samples``:

    python tests\\benchmark.py

PLAN.md section 11 asks for this at the end of stages 2, 6 and 7.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
import tempfile
import time

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# As main.py does, so that the numbers are the program's.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.controller import Controller  # noqa: E402
from app.paths import filters_dir, sidecar_dir  # noqa: E402
from app.i18n import init_translator  # noqa: E402
from app.settings import Settings  # noqa: E402
from core import decoder, exporter, pipeline, sidecar  # noqa: E402
from core.edit_state import EditState  # noqa: E402
from core.filters import load_catalogue  # noqa: E402
from core.exporter import ExportOptions  # noqa: E402
from core.geometry import Geometry, Rect  # noqa: E402

SAMPLES = PROJECT_ROOT / "tests" / "samples"
#: The window size the preview is measured at, as a long edge in pixels.
PREVIEW_LONG_EDGE = 1280

#: How long the simulated drag lasts, and how often it sends a new value:
#: about sixty a second, which is what a hand on a mouse produces.
DRAG_SECONDS = 1.2
DRAG_CADENCE_MS = 16

#: The drag sweeps the exposure slider back and forth over this many stops,
#: in this many steps per sweep, so that no two previews in a row are the
#: same piece of work and nothing can come out of a cache.
DRAG_AMPLITUDE_EV = 1.0
DRAG_PERIOD = 24

#: How many times each crop or rotation is applied; the median is kept.
GEOMETRY_RUNS = 5

#: PLAN.md section 11 sets the thumbnail target for a folder of this many
#: photos. A smaller folder is still measured, but the result is reported
#: rather than held to the target, where it would mean nothing.
THUMBNAIL_TARGET_COUNT = 300
THUMBNAIL_TARGET_FIRST = 30.0
THUMBNAIL_TARGET_CACHED = 2.0

#: How often the stall timer should fire while the thumbnails are made. How
#: late it actually fires is how long the interface was unable to answer.
STALL_TICK_MS = 10


@dataclass
class Result:
    """One measurement against its target."""

    name: str
    measured: float
    target: float
    unit: str

    @property
    def passed(self) -> bool:
        return self.measured <= self.target

    def line(self) -> str:
        mark = "通過" if self.passed else "未達標"
        return f"  {self.name:<34s}{self.measured:>9.2f} {self.unit:<4s} 目標 {self.target:>6.1f}  {mark}"


class _Counters(ctypes.Structure):
    """PROCESS_MEMORY_COUNTERS, as Windows fills it in."""

    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _memory_counter(field: str) -> float:
    """Return one of the process memory counters, in megabytes."""
    try:
        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        # The handle has to go across as a pointer: it is a pseudo handle of
        # all ones, and sent as a plain int it arrives truncated and the call
        # quietly reports nothing.
        handle = ctypes.c_void_p(ctypes.windll.kernel32.GetCurrentProcess())  # type: ignore[attr-defined]
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
            handle, ctypes.byref(counters), ctypes.c_uint32(counters.cb)
        )
        if not ok:
            return float("nan")
        return float(getattr(counters, field)) / 1024 / 1024
    except (AttributeError, OSError):
        return float("nan")


def memory_used_mb() -> float:
    """Return how much memory this process is using, in megabytes."""
    return _memory_counter("WorkingSetSize")


def peak_memory_used_mb() -> float:
    """Return the most memory this process has used, in megabytes.

    The working set at the end of an export says nothing about what it took
    while it ran, since the buffers are given back the moment it is done.
    """
    return _memory_counter("PeakWorkingSetSize")


@dataclass
class Drag:
    """What one simulated drag managed to get onto the screen."""

    seconds: float
    #: Changes the hand sent, whether or not the screen kept up.
    sent: int
    #: Previews that arrived while it was dragging.
    shown: int
    #: Milliseconds between one preview arriving and the next.
    gaps: list[float]
    #: Milliseconds from letting go to the full size preview.
    tail: float

    @property
    def median_gap(self) -> float:
        """How long the screen went between refreshes, in milliseconds.

        This is the number PLAN.md section 11 sets a target for: what the
        hand sees, not what one update costs when nothing else is waiting.
        """
        return float(np.median(self.gaps)) if self.gaps else float("inf")

    @property
    def worst_gap(self) -> float:
        return max(self.gaps) if self.gaps else float("inf")

    @property
    def rate(self) -> float:
        """Previews a second."""
        return self.shown / self.seconds if self.seconds else 0.0

    @property
    def superseded(self) -> int:
        """Changes that never got their own preview.

        Not a fault: a change that arrives while an older one is still being
        developed replaces it, which is what keeps the screen on the value
        the hand is on now rather than working through a backlog. The number
        grows with how fast the hand moves and says nothing on its own - the
        gaps and the tail are what tell you whether the drag felt good.
        """
        return max(self.sent - self.shown, 0)


def _drag_value(step: int) -> float:
    """Return the exposure the hand is on at *step*, sweeping back and forth."""
    half = DRAG_PERIOD // 2
    position = step % DRAG_PERIOD
    if position > half:
        position = DRAG_PERIOD - position
    return DRAG_AMPLITUDE_EV * position / half


def wait_for_preview(signal, accept, timeout: float = 60.0) -> float:
    """Return how long the first preview *accept* says yes to took, in seconds."""
    loop = QEventLoop()
    arrived: list[float] = []
    start = time.perf_counter()

    def capture(preview: object) -> None:
        if arrived or not accept(preview):
            return
        arrived.append(time.perf_counter() - start)
        loop.quit()

    signal.connect(capture)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(int(timeout * 1000))
    loop.exec()
    signal.disconnect(capture)
    return arrived[0] if arrived else float("inf")


def time_a_drag(controller, seconds: float = DRAG_SECONDS) -> Drag:
    """Drag a slider the way a hand does, and see what the screen got.

    The changes go out on a timer and are never waited for, because a hand
    does not wait either. That is the whole point: what matters while
    dragging is how often the screen can refresh when the changes arrive
    faster than a preview can be developed, and how far behind it is left
    when the hand stops.

    Until 2026-10-23 this sent one change, waited for its preview, and sent
    the next, which measured what a single update costs with the machine
    otherwise idle - a number that cannot show congestion at all, and said
    the target was met however bad the drag was.
    """
    base = controller.state
    shown: list[float] = []
    sent = 0
    loop = QEventLoop()

    def arrived(preview: object) -> None:
        if getattr(preview, "draft", False):
            shown.append(time.perf_counter())

    def send() -> None:
        nonlocal sent
        if time.perf_counter() - start >= seconds:
            loop.quit()
            return
        sent += 1
        controller.set_adjustments(
            base.with_value("exposure", _drag_value(sent)), dragging=True
        )

    controller.preview_ready.connect(arrived)
    timer = QTimer()
    timer.setInterval(DRAG_CADENCE_MS)
    timer.timeout.connect(send)
    start = time.perf_counter()
    timer.start()
    loop.exec()
    timer.stop()
    elapsed = time.perf_counter() - start
    controller.preview_ready.disconnect(arrived)

    # Letting go. What follows is the full size preview, and how long it
    # takes is how long the screen disagrees with the slider.
    controller.set_adjustments(base.with_value("exposure", _drag_value(sent)))
    tail = wait_for_preview(
        controller.preview_ready, lambda preview: not getattr(preview, "draft", True)
    )

    return Drag(
        seconds=elapsed,
        sent=sent,
        shown=len(shown),
        gaps=[(later - earlier) * 1000 for earlier, later in zip(shown, shown[1:])],
        tail=tail * 1000,
    )


#: The crop and rotation changes the preview is measured against, as a name
#: for the report and the geometry to apply. The target is held to whichever
#: of them turns out to be slowest, so that it cannot be met by picking the
#: cheap one: cropping away half the photo is less work than turning it.
GEOMETRY_CASES = {
    "裁切到中央 70%": Geometry(crop=Rect(0.15, 0.15, 0.7, 0.7)),
    "轉 90 度": Geometry(quarter_turns=1),
    "拉直 4.5 度": Geometry(angle=4.5),
    "裁切＋轉＋拉直": Geometry(quarter_turns=1, angle=4.5, crop=Rect(0.1, 0.15, 0.7, 0.6)),
}


def time_a_geometry_change(controller, geometry: Geometry) -> float:
    """Return the median milliseconds from a crop or rotation to its preview.

    Every run puts the photo back to the whole frame first and waits for
    that preview, so what is measured is the change to *geometry* itself
    rather than the step from whatever the previous case left behind.
    """
    timings = []
    for _ in range(GEOMETRY_RUNS):
        controller.set_geometry(Geometry())
        wait_for(controller.preview_ready, final_only=True)
        controller.set_geometry(geometry)
        timings.append(wait_for(controller.preview_ready, final_only=True) * 1000)
    return float(np.median(timings))


@dataclass
class Thumbnails:
    """What making a folder's thumbnails took, twice over."""

    count: int
    #: How many of them actually arrived, in case one cannot be made.
    shown: int
    #: Seconds to make every thumbnail with nothing cached.
    first: float
    #: Seconds to show them all again, from the cache on disk.
    cached: float
    #: The longest the interface could not answer, in milliseconds.
    worst_stall: float


def time_the_thumbnails(controller, photos: list[Path]) -> Thumbnails:
    """Make every thumbnail in the folder, cold then from the cache.

    The interface may not freeze while it happens (PLAN.md 11), so a timer
    that should fire every ``STALL_TICK_MS`` runs alongside and the worst
    lateness is kept: that is how long the main thread was busy at a stretch,
    which is what a frozen window is.
    """
    stalls: list[float] = []

    def run(expected: int) -> tuple[float, int]:
        arrived: set[Path] = set()
        loop = QEventLoop()

        def ready(path: Path, _thumbnail: object) -> None:
            arrived.add(path)
            if len(arrived) >= expected:
                loop.quit()

        ticked = [time.perf_counter()]

        def tick() -> None:
            now = time.perf_counter()
            stalls.append((now - ticked[0]) * 1000 - STALL_TICK_MS)
            ticked[0] = now

        stall_timer = QTimer()
        stall_timer.setInterval(STALL_TICK_MS)
        stall_timer.timeout.connect(tick)
        timeout = QTimer()
        timeout.setSingleShot(True)
        timeout.timeout.connect(loop.quit)

        controller.thumbnail_ready.connect(ready)
        start = time.perf_counter()
        ticked[0] = start
        stall_timer.start()
        # A photo whose thumbnail cannot be made never arrives, so the wait
        # has to end on its own rather than hang the whole run.
        timeout.start(120_000)
        controller.request_thumbnails(list(photos))
        loop.exec()
        elapsed = time.perf_counter() - start
        stall_timer.stop()
        controller.thumbnail_ready.disconnect(ready)
        return elapsed, len(arrived)

    controller.clear_thumbnail_cache()
    first, shown = run(len(photos))
    cached, _ = run(len(photos))
    return Thumbnails(
        count=len(photos),
        shown=shown,
        first=first,
        cached=cached,
        worst_stall=max(stalls) if stalls else 0.0,
    )


def use_a_scratch_profile() -> Path:
    """Point the application's own data directories at a throwaway folder.

    The settings file, the presets and the thumbnail cache live under
    %APPDATA% and %LOCALAPPDATA% (``app/paths.py`` reads them on every call).
    A benchmark must not reach into the ones the person uses: measuring the
    thumbnails means clearing the cache first, and opening a photo leaves
    tests/samples as the folder the application would reopen.
    """
    scratch = Path(tempfile.mkdtemp(prefix="tintaphoto-benchmark-"))
    os.environ["APPDATA"] = str(scratch / "roaming")
    os.environ["LOCALAPPDATA"] = str(scratch / "local")
    return scratch


def forget_adjustments(photos: list[Path]) -> None:
    """Remove the parameter files the measurements leave behind.

    Letting go of a slider and changing a crop are finished changes, so the
    controller saves them, which is what it is supposed to do. Left there,
    the next run opens the photo with those adjustments already on it and
    measures something else: on 2026-10-23 a leftover +0.25 EV was being
    developed during the open measurement. The store is in the scratch
    profile, and an old file beside a sample from 0.34 and earlier is hidden
    rather than deleted (PLAN.md 7.1).
    """
    for photo in photos:
        sidecar.delete(photo, sidecar_dir())


def time_one_open(controller, sample: Path, timeout: float = 60.0) -> tuple[float, float]:
    """Open *sample* once and return when each of its two previews appeared.

    Both numbers come from the same open: first the preview the camera stored
    inside the RAW file, then the photo itself. Taking them from two opens
    instead, which is what this did until 2026-10-23, has the second decode
    racing the first - every open clears the decoded photo, so the first open
    warms nothing - and measured almost twice what one open takes.
    """
    quick: list[float] = []
    full: list[float] = []
    loop = QEventLoop()

    def capture(*args: object) -> None:
        now = time.perf_counter() - start
        if args and getattr(args[0], "provisional", False):
            if not quick:
                quick.append(now)
            return
        if not full:
            full.append(now)
            loop.quit()

    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(int(timeout * 1000))

    controller.preview_ready.connect(capture)
    start = time.perf_counter()
    controller.open(sample)
    if not full:
        loop.exec()
    controller.preview_ready.disconnect(capture)

    return (
        quick[0] if quick else float("inf"),
        full[0] if full else float("inf"),
    )


def wait_for(signal, timeout: float = 60.0, final_only: bool = False) -> float:
    """Wait for *signal* and return how long it took, in seconds.

    With *final_only*, the preview the camera stored inside the RAW file does
    not count: what is being measured is the photo itself appearing.
    """
    loop = QEventLoop()
    arrived: list[float] = []
    start = time.perf_counter()

    def capture(*args: object) -> None:
        if final_only and args and getattr(args[0], "provisional", False):
            return
        if not arrived:
            arrived.append(time.perf_counter() - start)
            loop.quit()

    signal.connect(capture)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(int(timeout * 1000))
    loop.exec()
    signal.disconnect(capture)
    return arrived[0] if arrived else float("inf")


def wait_for_window(signal, sharp: bool, timeout: float = 60.0) -> float:
    """Return how long the next window of the zoomed in photo took to arrive.

    With *sharp*, only a window developed at the photo's own resolution
    counts; the stretched proxy that comes first while the full size photo is
    being decoded does not (PLAN.md 3.3).
    """
    loop = QEventLoop()
    arrived: list[float] = []
    start = time.perf_counter()

    def capture(preview: object) -> None:
        region = getattr(preview, "region", None)
        frame = getattr(preview, "frame", None)
        if arrived or region is None or frame is None:
            return
        wanted = (region[2] - region[0]) * frame[0]
        if sharp and preview.image.shape[1] < wanted * 0.99:
            return
        arrived.append(time.perf_counter() - start)
        loop.quit()

    signal.connect(capture)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(int(timeout * 1000))
    loop.exec()
    signal.disconnect(capture)
    return arrived[0] if arrived else float("inf")


def raw_samples() -> list[Path]:
    return sorted({*SAMPLES.glob("*.ARW"), *SAMPLES.glob("*.arw")})


def main() -> int:
    photos = raw_samples()
    if not photos:
        print("tests/samples 裡沒有 ARW 檔案，無法量測。")
        return 1

    if QApplication.instance() is None:
        QApplication(sys.argv[:1])
    scratch = use_a_scratch_profile()
    init_translator("zh_TW")
    results: list[Result] = []
    sample = photos[0]

    # Whatever a previous run saved beside the photos, so that the open
    # below is measured on the photo as the camera left it.
    forget_adjustments(photos)

    print(f"素材：{sample.name}，預覽長邊 {PREVIEW_LONG_EDGE} px\n")

    # -- opening a photo ---------------------------------------------
    controller = Controller(Settings())
    controller.set_preview_long_edge(PREVIEW_LONG_EDGE)
    quick, elapsed = time_one_open(controller, sample)
    results.append(Result("開啟一張 ARW 到顯示預覽", elapsed, 3.0, "秒"))
    print(f"  （參考）相機內嵌預覽先出現 {quick * 1000:>9.1f} 毫秒")
    print()

    # -- dragging a slider -------------------------------------------
    drag = time_a_drag(controller)
    results.append(Result("拖曳調色滑桿的預覽更新", drag.median_gap, 100.0, "毫秒"))
    print(
        f"  （參考）拖曳 {drag.seconds:.1f} 秒：送出 {drag.sent} 次變更，"
        f"畫面更新 {drag.shown} 次，{drag.superseded} 次被後來的請求取代"
    )
    print(f"  （參考）更新率 {drag.rate:>9.1f} 次／秒")
    print(f"  （參考）更新間隔最長 {drag.worst_gap:>9.1f} 毫秒")
    print(f"  （參考）放手到全尺寸預覽 {drag.tail:>9.1f} 毫秒")

    # The same drag with a film filter on. The filter is the one step that
    # runs over every pixel for a reason the sliders cannot undo, so the
    # target of PLAN.md section 11 has to be met with one on as well.
    if controller.filters:
        worn = controller.filters[0].identifier
        controller.set_adjustments(controller.state.with_filter(worn))
        wait_for(controller.preview_ready, final_only=True)
        filtered = time_a_drag(controller)
        results.append(
            Result(f"拖曳滑桿的預覽更新（濾鏡 {worn}）", filtered.median_gap, 100.0, "毫秒")
        )
        print(
            f"  （參考）同上但戴著濾鏡：更新率 {filtered.rate:.1f} 次／秒、"
            f"最長間隔 {filtered.worst_gap:.1f} 毫秒、放手後 {filtered.tail:.1f} 毫秒"
        )
        controller.set_adjustments(controller.state.with_filter(""))
        wait_for(controller.preview_ready, final_only=True)

    # And with film grain on (PLAN.md 6 step 11.5). Its field is kept by the
    # controller, so the drag should cost what it costs without it plus the
    # adding of the grain itself.
    controller.set_adjustments(controller.state.with_value("grain", 50.0))
    wait_for(controller.preview_ready, final_only=True)
    grained = time_a_drag(controller)
    results.append(Result("拖曳滑桿的預覽更新（顆粒 50）", grained.median_gap, 100.0, "毫秒"))
    print(
        f"  （參考）同上但加了顆粒：更新率 {grained.rate:.1f} 次／秒、"
        f"最長間隔 {grained.worst_gap:.1f} 毫秒、放手後 {grained.tail:.1f} 毫秒"
    )
    controller.set_adjustments(controller.state.with_value("grain", 0.0))
    wait_for(controller.preview_ready, final_only=True)
    print()

    # -- changing the crop or the angle (PLAN.md 11) -----------------
    geometry_timings = {
        name: time_a_geometry_change(controller, geometry)
        for name, geometry in GEOMETRY_CASES.items()
    }
    slowest = max(geometry_timings, key=geometry_timings.get)
    results.append(
        Result("變更裁切或旋轉後的預覽更新", geometry_timings[slowest], 200.0, "毫秒")
    )
    for name, median in geometry_timings.items():
        mark = "（最慢，列入上表）" if name == slowest else ""
        print(f"  （參考）{name:<16s}{median:>9.1f} 毫秒{mark}")
    print()
    # Back to the whole photo, or the export below would be measured on a
    # crop of it.
    controller.set_geometry(Geometry())
    wait_for(controller.preview_ready, final_only=True)

    # -- moving to a photo already decoded ---------------------------
    if len(photos) > 1:
        controller.open_folder(SAMPLES)
        wait_for(controller.preview_ready, final_only=True)
        controller.select_next()
        wait_for(controller.preview_ready, final_only=True)
        controller.select_previous()
        wait_for(controller.preview_ready, final_only=True)
        controller.select_next()
        elapsed = wait_for(controller.preview_ready, final_only=True)
        results.append(Result("切換到下一張（已開過）", elapsed, 1.0, "秒"))

    # Before the thumbnails and the zoom below, or this would be the memory
    # of a session that has done rather more than open one photo.
    results.append(Result("記憶體用量（開啟一張照片）", memory_used_mb() / 1024, 2.0, "GB"))

    # -- the folder's thumbnails (PLAN.md 11) ------------------------
    gallery = sorted(
        path for path in SAMPLES.iterdir() if path.is_file() and decoder.is_supported(path)
    )
    thumbs = time_the_thumbnails(controller, gallery)
    if thumbs.count >= THUMBNAIL_TARGET_COUNT:
        results.append(
            Result("資料夾縮圖全部出現（首次）", thumbs.first, THUMBNAIL_TARGET_FIRST, "秒")
        )
        results.append(
            Result("資料夾縮圖全部出現（快取）", thumbs.cached, THUMBNAIL_TARGET_CACHED, "秒")
        )
    else:
        print(
            f"  （參考）{thumbs.count} 張縮圖：首次 {thumbs.first:.2f} 秒、"
            f"快取 {thumbs.cached:.2f} 秒"
        )
        print(
            f"  （參考）PLAN 的 30／2 秒目標要 {THUMBNAIL_TARGET_COUNT} 張才算得準，"
            f"現在只有 {thumbs.count} 張，所以不列入上表"
        )
    if thumbs.shown < thumbs.count:
        print(f"  （注意）{thumbs.count - thumbs.shown} 張縮圖做不出來")
    print(f"  （參考）產生縮圖時介面最長停頓 {thumbs.worst_stall:>9.1f} 毫秒")
    print()

    # -- zooming in past the proxy (PLAN.md 3.3) ---------------------
    # Settled, the way the preview says it once the wheel has stopped.
    controller.set_view(1.0, (0.40, 0.40, 0.60, 0.60), settled=True)
    stretched = wait_for_window(controller.preview_ready, sharp=False)
    first = stretched + wait_for_window(controller.preview_ready, sharp=True)
    print(f"  （參考）放大到 100%，代理圖先頂上  {stretched * 1000:>9.1f} 毫秒")
    print(f"  （參考）第一次放大，含原尺寸解碼    {first:>9.2f} 秒")
    controller.set_view(1.0, (0.45, 0.45, 0.65, 0.65), settled=True)
    panned = wait_for_window(controller.preview_ready, sharp=True)
    results.append(Result("放大後平移，換成清晰畫面", panned * 1000, 300.0, "毫秒"))
    print(f"  （參考）放大到 100% 後的記憶體     {memory_used_mb() / 1024:>9.2f} GB")
    controller.set_view(0.2, None)
    wait_for(controller.preview_ready)

    # -- exporting at full size (PLAN.md 2.6, section 11) ------------
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / exporter.default_name(sample)
        start = time.perf_counter()
        controller.export(target, ExportOptions(quality=90))
        elapsed = wait_for(controller.export_finished, timeout=120.0)
        results.append(Result("匯出原尺寸 JPG", elapsed, 10.0, "秒"))
        print(f"  （參考）匯出耗時 {time.perf_counter() - start:>9.2f} 秒（含訊號往返）")
        for label, options in (
            ("16 位元 TIFF", ExportOptions(format=exporter.FORMAT_TIFF, bits=16)),
            ("16 位元 PNG", ExportOptions(format=exporter.FORMAT_PNG, bits=16)),
        ):
            controller.export(Path(folder) / label.replace(" ", "_"), options)
            taken = wait_for(controller.export_finished, timeout=120.0)
            print(f"  （參考）匯出 {label:<12s}{taken:>9.2f} 秒")
        print(f"  （參考）匯出後的記憶體尖峰 {peak_memory_used_mb() / 1024:>9.2f} GB")

    controller.shutdown()

    # -- the pipeline itself, for the record -------------------------
    photo = decoder.decode_proxy(sample, PREVIEW_LONG_EDGE)
    untouched = EditState.for_photo(True, *pipeline.shot_white_balance(photo))
    moved = (
        untouched.with_value("exposure", 0.5)
        .with_value("contrast", 40.0)
        .with_value("shadows", 60.0)
        .with_value("saturation", 30.0)
    )
    # The film filter is measured apart from the sliders because it is the
    # one step a photo can be developed entirely without: with no filter
    # chosen the step is not entered at all, so the first two numbers here
    # are what every photo that wears none still costs.
    looks = {look.identifier: look for look in load_catalogue(filters_dir())}
    first = next(iter(looks), "")
    wearing = untouched.with_filter(first)
    whole = decoder.decode_proxy(sample, decoder.PROXY_LONG_EDGE)
    measurements = [
        (f"原樣（{PREVIEW_LONG_EDGE} px）", photo, untouched, None),
        (f"四個滑桿都動（{PREVIEW_LONG_EDGE} px）", photo, moved, None),
        (f"加上顆粒（{PREVIEW_LONG_EDGE} px）", photo, untouched.with_value("grain", 50.0), None),
        (
            f"加上顆粒（{decoder.PROXY_LONG_EDGE} px）",
            whole_proxy := decoder.decode_proxy(sample, decoder.PROXY_LONG_EDGE),
            untouched.with_value("grain", 50.0),
            None,
        ),
    ]
    if first:
        measurements += [
            (f"加上濾鏡（{PREVIEW_LONG_EDGE} px）", photo, wearing, looks[first]),
            (f"原樣（{decoder.PROXY_LONG_EDGE} px）", whole, untouched, None),
            (f"加上濾鏡（{decoder.PROXY_LONG_EDGE} px）", whole, wearing, looks[first]),
        ]
    for label, source, values, look in measurements:
        best = min(
            (
                lambda: (
                    lambda s: (
                        pipeline.develop(source, values, look=look),
                        time.perf_counter() - s,
                    )[1]
                )(time.perf_counter())
            )()
            for _ in range(3)
        )
        print(f"  （參考）pipeline.develop {label:<22s}{best * 1000:>9.1f} 毫秒")

    forget_adjustments(photos)
    shutil.rmtree(scratch, ignore_errors=True)

    print()
    for result in results:
        print(result.line())
    failed = [result for result in results if not result.passed]
    print()
    print("全部達標。" if not failed else f"{len(failed)} 項未達標。")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
