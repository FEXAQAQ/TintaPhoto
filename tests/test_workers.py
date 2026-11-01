"""Tests for the background thread pool (PLAN.md section 5).

What a group promises is what the preview depends on: one task of it at a
time, a new request replacing whatever was waiting, and whatever finishes
reaching the window unless something newer has already reached it. A slider
under the hand submits a request every few milliseconds, so getting this
wrong either fills the pool with previews nobody will see or shows none of
them at all.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402

from app.workers import TaskRunner  # noqa: E402

GROUP = "preview"
#: Long enough that the main thread can submit several requests while one
#: task is running, short enough not to slow the suite down.
WORK_MS = 60


class Watcher:
    """A piece of work that records how many of its kind run at once."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.live = 0
        self.most_at_once = 0
        self.ran: list[int] = []

    def work(self, label: int, milliseconds: int = WORK_MS):
        """Return a function that takes a while and reports *label*."""

        def run() -> int:
            with self.lock:
                self.live += 1
                self.most_at_once = max(self.most_at_once, self.live)
                self.ran.append(label)
            try:
                time.sleep(milliseconds / 1000)
                return label
            finally:
                with self.lock:
                    self.live -= 1

        return run


@pytest.fixture
def runner(qt_app: object) -> Iterator[TaskRunner]:
    task_runner = TaskRunner()
    yield task_runner
    task_runner.shutdown()


def settle(qt_app: object, milliseconds: int) -> None:
    """Let the pool work and its signals arrive."""
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()
    qt_app.processEvents()  # type: ignore[attr-defined]


def _collect(runner: TaskRunner) -> tuple[list[int], list[int]]:
    """Return the lists the runner's signals fill in."""
    delivered: list[int] = []
    dropped: list[int] = []
    runner.completed.connect(lambda token, result: delivered.append(result))
    runner.discarded.connect(dropped.append)
    return delivered, dropped


# -- one at a time --------------------------------------------------------


def test_a_group_runs_one_task_at_a_time(qt_app: object, runner: TaskRunner) -> None:
    """Eleven previews at once is what made one of them eight times slower."""
    watcher = Watcher()
    for label in range(6):
        runner.submit(watcher.work(label), GROUP)
    settle(qt_app, WORK_MS * 8)
    assert watcher.most_at_once == 1


def test_only_the_newest_waiting_request_of_a_group_runs(
    qt_app: object, runner: TaskRunner
) -> None:
    """A slider moved six times must not develop six previews."""
    watcher = Watcher()
    for label in range(6):
        runner.submit(watcher.work(label), GROUP)
    settle(qt_app, WORK_MS * 8)
    # The first one starts at once; the rest replace each other while it
    # runs, so the last is the only other that ever runs.
    assert watcher.ran == [0, 5]


def test_work_without_a_group_still_runs_alongside(
    qt_app: object, runner: TaskRunner
) -> None:
    """Thumbnails must keep every thread busy; only groups are queues."""
    if runner.thread_count < 2:
        pytest.skip("a single thread cannot overlap anything")
    watcher = Watcher()
    for label in range(runner.thread_count):
        runner.submit(watcher.work(label))
    settle(qt_app, WORK_MS * 6)
    assert watcher.most_at_once > 1


def test_a_task_that_fails_still_lets_the_waiting_one_run(
    qt_app: object, runner: TaskRunner
) -> None:
    def explode() -> None:
        raise ValueError("no")

    watcher = Watcher()
    failures: list[str] = []
    runner.failed.connect(lambda token, message: failures.append(message))
    runner.submit(explode, GROUP)
    runner.submit(watcher.work(1), GROUP)
    settle(qt_app, WORK_MS * 6)
    assert failures == ["no"]
    assert watcher.ran == [1]


# -- what finishes is shown ----------------------------------------------


def test_a_finished_task_is_delivered_although_a_newer_one_is_waiting(
    qt_app: object, runner: TaskRunner
) -> None:
    """The drag's whole point: the frame that is ready goes on the screen.

    The first task is still running when the second is asked for, which is
    what happens on every mouse move. Dropping its result because something
    newer exists is what left the preview frozen for a whole drag.
    """
    watcher = Watcher()
    delivered, _dropped = _collect(runner)
    runner.submit(watcher.work(0), GROUP)
    settle(qt_app, WORK_MS // 3)
    runner.submit(watcher.work(1), GROUP)
    settle(qt_app, WORK_MS * 6)
    assert delivered == [0, 1]


def test_what_is_delivered_never_goes_backwards(
    qt_app: object, runner: TaskRunner
) -> None:
    """A slow early task must not land on top of a newer one."""
    watcher = Watcher()
    delivered, _dropped = _collect(runner)
    for label, milliseconds in ((0, WORK_MS * 3), (1, WORK_MS)):
        runner.submit(watcher.work(label, milliseconds), GROUP)
        settle(qt_app, WORK_MS // 3)
    settle(qt_app, WORK_MS * 8)
    assert delivered == sorted(delivered)


def test_a_replaced_request_is_announced_so_it_can_be_let_go(
    qt_app: object, runner: TaskRunner
) -> None:
    """The controller holds a handler per token; a dropped one must be freed."""
    watcher = Watcher()
    _delivered, dropped = _collect(runner)
    tokens = [runner.submit(watcher.work(label), GROUP) for label in range(4)]
    settle(qt_app, WORK_MS * 8)
    assert sorted(dropped) == tokens[1:3]


def test_abandoning_a_group_drops_what_is_waiting_and_what_is_running(
    qt_app: object, runner: TaskRunner
) -> None:
    watcher = Watcher()
    delivered, _dropped = _collect(runner)
    runner.submit(watcher.work(0), GROUP)
    runner.submit(watcher.work(1), GROUP)
    settle(qt_app, WORK_MS // 3)
    runner.abandon(GROUP)
    settle(qt_app, WORK_MS * 6)
    assert delivered == []


def test_a_request_after_abandoning_is_delivered_as_usual(
    qt_app: object, runner: TaskRunner
) -> None:
    watcher = Watcher()
    delivered, _dropped = _collect(runner)
    runner.submit(watcher.work(0), GROUP)
    runner.abandon(GROUP)
    runner.submit(watcher.work(1), GROUP)
    settle(qt_app, WORK_MS * 8)
    assert delivered == [1]


def test_two_groups_do_not_wait_for_each_other(
    qt_app: object, runner: TaskRunner
) -> None:
    """The camera's quick preview must not sit behind a develop."""
    if runner.thread_count < 2:
        pytest.skip("a single thread cannot overlap anything")
    watcher = Watcher()
    runner.submit(watcher.work(0), "one")
    runner.submit(watcher.work(1), "two")
    settle(qt_app, WORK_MS * 6)
    assert watcher.most_at_once == 2


def test_nothing_starts_once_the_runner_has_shut_down(qt_app: object) -> None:
    """A result still on its way at closing time must not set off more work."""
    task_runner = TaskRunner()
    task_runner.shutdown()
    ran: list[str] = []
    task_runner.submit(lambda: ran.append("free"))
    task_runner.submit(lambda: ran.append("grouped"), GROUP)
    settle(qt_app, WORK_MS * 2)
    assert ran == []
