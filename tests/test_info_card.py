"""Tests for the info card's coming and going (PLAN.md 3.6 and 3.9).

It comes the way the application menu beside the info button does: it fades
in where it rests, and it is gone at once when taken away. With Windows set
to show no animation it is simply there or not.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

from ui import native, theme  # noqa: E402
from ui.info_card import InfoCard  # noqa: E402


def _wait(predicate, timeout_ms: int = 4000) -> bool:
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        loop = QEventLoop()
        QTimer.singleShot(5, loop.quit)
        loop.exec()
        if predicate():
            return True
    return predicate()


@pytest.fixture
def card(qt_app: object, monkeypatch: pytest.MonkeyPatch) -> Iterator[InfoCard]:
    monkeypatch.setattr(native, "animations_enabled", lambda: True)
    host = QWidget()
    host.resize(900, 700)
    made = InfoCard(host)
    made.hide()
    host.show()
    yield made
    host.close()
    host.deleteLater()


def test_the_card_fades_in_where_it_rests_like_the_menu(card: InfoCard) -> None:
    card.set_shown(True)
    assert card.wanted and card.isVisible()
    assert card.moving, "the card was there all at once"
    assert card.graphicsEffect().opacity() == 0.0

    seen: list[tuple[int, int, float]] = []

    def watch() -> bool:
        if card.moving:
            seen.append((card.x(), card.y(), card.graphicsEffect().opacity()))
        return not card.moving

    assert _wait(watch)
    assert seen, "it never faded"
    assert {(x, y) for x, y, _ in seen} == {(card.home().x(), card.home().y())}, (
        "it moved rather than faded in place"
    )
    fades = [fade for _, _, fade in seen]
    assert fades == sorted(fades), "it faded on its way in"
    assert card.pos() == card.home()
    assert not card.graphicsEffect().isEnabled(), "an effect left on at rest"


def test_it_fades_in_as_long_as_the_menu_does() -> None:
    # Qt fades a menu in over 150 ms (qFadeEffect's default), evenly.
    assert theme.INFO_CARD_FADE_MS == 150


def test_taken_away_it_is_gone_at_once_like_the_menu(card: InfoCard) -> None:
    card.set_shown(True)
    assert _wait(lambda: not card.moving)

    card.set_shown(False)
    assert not card.wanted
    assert not card.isVisible() and not card.moving


def test_taken_away_half_way_in_it_is_gone_and_comes_back_whole(card: InfoCard) -> None:
    card.set_shown(True)
    card.set_shown(False)
    assert not card.wanted
    assert not card.isVisible() and not card.moving

    card.set_shown(True)
    assert _wait(lambda: not card.moving)
    assert card.isVisible() and card.pos() == card.home()
    assert not card.graphicsEffect().isEnabled()


def test_with_no_animation_the_card_is_simply_there_or_not(
    card: InfoCard, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(native, "animations_enabled", lambda: False)
    card.set_shown(True)
    assert card.isVisible() and not card.moving and card.pos() == card.home()
    card.set_shown(False)
    assert not card.isVisible() and not card.moving
