"""Running slow work off the interface thread (PLAN.md section 5).

Decoding, thumbnails and developing a preview all take long enough to freeze
the window if they ran on it. They are handed to a thread pool here, and their
results come back as Qt signals on the interface thread.

Requests can be grouped, and a group is a queue of one: only one task of a
group runs at a time, and a request made while one is running replaces
whatever was waiting rather than queueing behind it. Without that, a slider
under the hand fills every thread in the pool with previews nobody will ever
see, and each of them slows down the one that will be seen - measured on a
2560 px preview, eleven at once turned 178 ms into 1505 ms.

What finishes is delivered, even when a newer request is already waiting: it
is the newest answer there is until that one arrives, and showing it is what
makes a drag move rather than sit still. Only an outcome older than one
already delivered is dropped, so what reaches the window never goes
backwards.

Work submitted without a group has none of this: thumbnails and the export
run alongside each other and alongside everything else.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from itertools import count
from threading import Lock
from typing import Any, Final

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, SignalInstance

logger = logging.getLogger(__name__)

#: One thread is left for the interface and the rest do the work.
_RESERVED_THREADS: Final[int] = 1


class TaskRunner(QObject):
    """A thread pool whose results arrive as signals on the main thread."""

    #: The token the work was submitted with, and what the function returned.
    completed = Signal(object, object)
    #: The token, and a message meant for the log rather than for the user.
    failed = Signal(object, str)
    #: A token nothing will ever arrive for: it was replaced before it ran, or
    #: it finished after something newer had already been delivered. Whoever
    #: is holding on to the token can let go of it.
    discarded = Signal(object)

    def __init__(self, parent: QObject | None = None, max_threads: int | None = None) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        available = QThreadPool.globalInstance().maxThreadCount()
        self._pool.setMaxThreadCount(
            max_threads if max_threads is not None else max(1, available - _RESERVED_THREADS)
        )
        self._tokens = count()
        self._lock = Lock()
        #: The groups with a task running.
        self._running: set[str] = set()
        #: The one task of a group that is waiting for the running one to end.
        self._waiting: dict[str, tuple[int, Callable[[], Any]]] = {}
        #: The newest token of each group whose outcome has been delivered.
        self._delivered: dict[str, int] = {}
        self._closing = False

    @property
    def thread_count(self) -> int:
        """Return how many tasks can run at once."""
        return self._pool.maxThreadCount()

    def submit(self, function: Callable[[], Any], group: str | None = None) -> int:
        """Run *function* in the background and return its token.

        With a *group*, the call either starts at once or becomes the one
        task waiting for that group, replacing whatever was waiting before.
        """
        token = next(self._tokens)
        replaced: int | None = None
        with self._lock:
            if self._closing:
                # A result that was already on its way when the window closed
                # can still ask for more; nothing more is started.
                return token
            if group is None:
                starting = True
            elif group in self._running:
                waiting = self._waiting.get(group)
                replaced = None if waiting is None else waiting[0]
                self._waiting[group] = (token, function)
                starting = False
            else:
                self._running.add(group)
                starting = True
        if replaced is not None:
            self.discarded.emit(replaced)
        if starting:
            self._pool.start(_Task(self, token, group, function))
        return token

    def abandon(self, group: str) -> None:
        """Declare that nothing pending in *group* is wanted any more."""
        with self._lock:
            waiting = self._waiting.pop(group, None)
            # What is already running cannot be stopped, but it belongs to
            # before this moment, so its outcome must not be delivered.
            self._delivered[group] = next(self._tokens)
        if waiting is not None:
            self.discarded.emit(waiting[0])

    def shutdown(self) -> None:
        """Stop accepting work and wait for what is running to end."""
        with self._lock:
            self._closing = True
            self._waiting.clear()
        self._pool.clear()
        self._pool.waitForDone()

    # -- what a task asks on its way out ---------------------------------

    def _may_deliver(self, token: int, group: str | None) -> bool:
        """Return whether this outcome is newer than the last one delivered."""
        if group is None:
            return True
        with self._lock:
            if token <= self._delivered.get(group, -1):
                return False
            self._delivered[group] = token
            return True

    def _hand_on(self, group: str | None) -> None:
        """Start whatever was waiting for *group*, or let the group go."""
        if group is None:
            return
        with self._lock:
            waiting = None if self._closing else self._waiting.pop(group, None)
            if waiting is None:
                self._running.discard(group)
                return
        self._pool.start(_Task(self, waiting[0], group, waiting[1]))


class _Task(QRunnable):
    """One piece of work, with the bookkeeping that lets it be abandoned."""

    def __init__(
        self, runner: TaskRunner, token: int, group: str | None, function: Callable[[], Any]
    ) -> None:
        super().__init__()
        self._runner = runner
        self._token = token
        self._group = group
        self._function = function
        self.setAutoDelete(True)

    def run(self) -> None:  # noqa: D102 - QRunnable's own documented hook
        try:
            try:
                result = self._function()
            except Exception as error:  # noqa: BLE001 - a task must never take the pool down
                logger.exception("A background task failed")
                self._announce(self._runner.failed, str(error))
                return
            self._announce(self._runner.completed, result)
        finally:
            # The next task of the group starts after the outcome of this one
            # has gone out, so that a finished preview reaches the window
            # before the pool is busy again.
            self._runner._hand_on(self._group)

    def _announce(self, signal: SignalInstance, payload: Any) -> None:
        """Send the outcome out, unless something newer has already gone."""
        if self._runner._may_deliver(self._token, self._group):
            signal.emit(self._token, payload)
        else:
            self._runner.discarded.emit(self._token)
