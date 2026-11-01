"""Undo and redo, one history per photo (PLAN.md E-04).

Each photo keeps its own list of edits and a position in it. Undoing steps
the position back, redoing steps it forward, and recording a new edit throws
away anything ahead of the position - the usual shape, which is what makes
undo predictable. An edit is the adjustments and the geometry together, so a
crop and a slider are undone the same way.

The one thing worth care is that dragging a slider must cost a single step,
not one per pixel the hand moves. That is solved before it gets here: the
controller only records when a change is finished, which for a slider means
the moment it is let go (PLAN.md section 9, stage 3). The preview still
follows the hand; the history simply does not watch it.

Nothing here is written to disk. A history lasts as long as the program runs
(E-04), and plain Python is enough for it: no Qt, no threads, no images, just
the small frozen states themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.edit_state import Edit

#: How many steps one photo remembers. PLAN.md E-04 asks for at least a
#: hundred; a state is a handful of numbers, so there is no reason to be
#: miserly, and the oldest are dropped once the limit is reached.
HISTORY_LIMIT: int = 200


@dataclass
class _PhotoHistory:
    """The states of one photo, and where in them we are."""

    states: list[Edit]
    position: int = 0

    @property
    def current(self) -> Edit:
        """Return the state now in force."""
        return self.states[self.position]

    @property
    def can_undo(self) -> bool:
        """Return whether there is an earlier state to go back to."""
        return self.position > 0

    @property
    def can_redo(self) -> bool:
        """Return whether there is a later state to go forward to."""
        return self.position < len(self.states) - 1

    def push(self, state: Edit) -> bool:
        """Record *state*; return whether it was a change worth recording."""
        if state == self.current:
            return False
        del self.states[self.position + 1 :]
        self.states.append(state)
        if len(self.states) > HISTORY_LIMIT:
            # Drop the oldest; the position moves with the window.
            del self.states[0 : len(self.states) - HISTORY_LIMIT]
        self.position = len(self.states) - 1
        return True


@dataclass
class History:
    """The undo histories of every photo opened since the program started."""

    _photos: dict[Path, _PhotoHistory] = field(default_factory=dict)

    def begin(self, photo: Path, state: Edit) -> None:
        """Start a history for *photo* at *state*, if it has none yet.

        Called when a photo is opened, with the edit it opened in - which for
        an edited photo is what its parameter file says, so that undo goes
        back towards how it was found rather than to an empty slate.
        """
        if photo not in self._photos:
            self._photos[photo] = _PhotoHistory(states=[state])

    def forget_all_but(self, photos: set[Path]) -> None:
        """Keep the histories of *photos* and drop the rest.

        Used when another folder is opened: nothing in the old one can be
        undone any more, and holding the states would only be clutter.
        """
        self._photos = {
            path: history for path, history in self._photos.items() if path in photos
        }

    def knows(self, photo: Path) -> bool:
        """Return whether *photo* has a history yet."""
        return photo in self._photos

    def current(self, photo: Path) -> Edit | None:
        """Return the state *photo* is in, or ``None`` if it has no history."""
        history = self._photos.get(photo)
        return history.current if history is not None else None

    def push(self, photo: Path, state: Edit) -> bool:
        """Record a finished change to *photo*.

        Returns whether anything was recorded: a change that lands on the
        state already in force - letting a slider go without having moved it,
        resetting a photo that is already at its defaults - is not a step.
        """
        history = self._photos.get(photo)
        if history is None:
            self._photos[photo] = _PhotoHistory(states=[state])
            return False
        return history.push(state)

    def can_undo(self, photo: Path) -> bool:
        """Return whether *photo* has something to undo."""
        history = self._photos.get(photo)
        return history is not None and history.can_undo

    def can_redo(self, photo: Path) -> bool:
        """Return whether *photo* has something to redo."""
        history = self._photos.get(photo)
        return history is not None and history.can_redo

    def undo(self, photo: Path) -> Edit | None:
        """Step *photo* back one state and return it, or ``None`` if it cannot."""
        history = self._photos.get(photo)
        if history is None or not history.can_undo:
            return None
        history.position -= 1
        return history.current

    def redo(self, photo: Path) -> Edit | None:
        """Step *photo* forward one state and return it, or ``None``."""
        history = self._photos.get(photo)
        if history is None or not history.can_redo:
            return None
        history.position += 1
        return history.current

    def depth(self, photo: Path) -> int:
        """Return how many states *photo* holds; for tests and diagnostics."""
        history = self._photos.get(photo)
        return len(history.states) if history is not None else 0


__all__ = ["HISTORY_LIMIT", "History"]
