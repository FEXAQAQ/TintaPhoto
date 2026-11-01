"""Tests for undo and redo (PLAN.md E-04)."""

from __future__ import annotations

from pathlib import Path

from app.history import HISTORY_LIMIT, History
from core.edit_state import EditState

PHOTO = Path("D:/Photos/DSC01234.ARW")
OTHER = Path("D:/Photos/DSC01235.ARW")

#: Where a RAW photo rests: the camera's own white balance, nothing adjusted.
REST = EditState.for_photo(True, temperature=5600.0, tint=0.0)


def _with(exposure: float) -> EditState:
    return REST.with_value("exposure", exposure)


def test_a_new_photo_has_nothing_to_undo() -> None:
    history = History()
    history.begin(PHOTO, REST)

    assert not history.can_undo(PHOTO)
    assert not history.can_redo(PHOTO)
    assert history.current(PHOTO) == REST


def test_a_photo_that_was_never_opened_is_unknown() -> None:
    history = History()

    assert not history.knows(PHOTO)
    assert history.current(PHOTO) is None
    assert history.undo(PHOTO) is None
    assert history.redo(PHOTO) is None


def test_undo_goes_back_to_the_previous_state() -> None:
    history = History()
    history.begin(PHOTO, REST)
    history.push(PHOTO, _with(1.0))

    assert history.can_undo(PHOTO)
    assert history.undo(PHOTO) == REST
    assert history.current(PHOTO) == REST


def test_redo_goes_forward_again() -> None:
    history = History()
    history.begin(PHOTO, REST)
    history.push(PHOTO, _with(1.0))
    history.undo(PHOTO)

    assert history.can_redo(PHOTO)
    assert history.redo(PHOTO) == _with(1.0)
    assert not history.can_redo(PHOTO)


def test_a_new_change_after_an_undo_drops_what_was_ahead() -> None:
    history = History()
    history.begin(PHOTO, REST)
    history.push(PHOTO, _with(1.0))
    history.undo(PHOTO)

    history.push(PHOTO, _with(-1.0))

    assert not history.can_redo(PHOTO)
    assert history.current(PHOTO) == _with(-1.0)
    assert history.undo(PHOTO) == REST


def test_a_change_that_changes_nothing_is_not_a_step() -> None:
    """Letting a slider go without moving it must not cost an undo."""
    history = History()
    history.begin(PHOTO, REST)

    assert not history.push(PHOTO, REST)
    assert not history.can_undo(PHOTO)
    assert history.depth(PHOTO) == 1


def test_one_step_per_finished_change() -> None:
    """A drag reaches the history once, because only the end is recorded."""
    history = History()
    history.begin(PHOTO, REST)

    # What the controller does with a drag: the preview follows every value,
    # the history is only told the one the slider was let go at.
    history.push(PHOTO, _with(0.75))

    assert history.depth(PHOTO) == 2
    assert history.undo(PHOTO) == REST


def test_each_photo_has_its_own_history() -> None:
    history = History()
    history.begin(PHOTO, REST)
    history.begin(OTHER, REST)
    history.push(PHOTO, _with(1.0))

    assert history.can_undo(PHOTO)
    assert not history.can_undo(OTHER)
    assert history.current(OTHER) == REST


def test_opening_a_photo_again_keeps_its_history() -> None:
    history = History()
    history.begin(PHOTO, REST)
    history.push(PHOTO, _with(1.0))

    history.begin(PHOTO, REST)

    assert history.can_undo(PHOTO)
    assert history.current(PHOTO) == _with(1.0)


def test_at_least_a_hundred_steps_are_remembered() -> None:
    """PLAN.md E-04 asks for at least 100 steps."""
    history = History()
    history.begin(PHOTO, REST)
    for step in range(1, 151):
        history.push(PHOTO, _with(step * 0.01))

    for _ in range(100):
        assert history.undo(PHOTO) is not None


def test_the_oldest_steps_are_dropped_once_the_limit_is_reached() -> None:
    history = History()
    history.begin(PHOTO, REST)
    for step in range(HISTORY_LIMIT + 50):
        history.push(PHOTO, _with(step * 0.01))

    assert history.depth(PHOTO) == HISTORY_LIMIT
    assert history.current(PHOTO) == _with((HISTORY_LIMIT + 49) * 0.01)


def test_opening_another_folder_forgets_the_old_histories() -> None:
    history = History()
    history.begin(PHOTO, REST)
    history.push(PHOTO, _with(1.0))
    history.begin(OTHER, REST)

    history.forget_all_but({OTHER})

    assert not history.knows(PHOTO)
    assert history.knows(OTHER)


def test_pushing_to_an_unknown_photo_starts_its_history() -> None:
    history = History()

    assert not history.push(PHOTO, _with(1.0))
    assert history.current(PHOTO) == _with(1.0)
    assert not history.can_undo(PHOTO)
