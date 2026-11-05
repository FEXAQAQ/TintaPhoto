"""Tests for the welcome screen (PLAN.md 3.6 and 3.7).

The drop, the name and the tagline are drawn by the opening canvas; the
two buttons, the recent folders and the drop hint are widgets that rise into
place. The opening is driven by hand here, frame by frame, through
``apply_opening``, and the screen is grabbed whole and measured, so that
what is checked is what is drawn.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import date

import pytest

pytest.importorskip("PySide6.QtWidgets")

import numpy as np  # noqa: E402
from PySide6.QtCore import QEventLoop, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from app.i18n import LANGUAGE_EN, init_translator, tr  # noqa: E402
from app.settings import RecentFolder  # noqa: E402
from ui import native, theme  # noqa: E402
from ui.welcome_view import (  # noqa: E402
    WelcomeView,
    opening_length,
    recent_count,
    recent_day,
)
from ui.widgets.hover import mix  # noqa: E402

#: Long enough for a transition to finish under a full test run, where it
#: can take longer than its own duration to get going.
_PATIENCE_MS = 4000

TODAY = date(2026, 10, 31)
KYOTO = "D:/Photos/2026-10 Kyoto"
LAKE = "D:/Photos/2026-09 Sun Moon Lake"


def _settle(predicate, timeout_ms: int = _PATIENCE_MS) -> bool:
    """Turn the event loop until *predicate* holds, or the time runs out."""
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        loop = QEventLoop()
        QTimer.singleShot(theme.DURATION_HOVER_MS // 2 or 1, loop.quit)
        loop.exec()
        if predicate():
            return True
    return predicate()


@pytest.fixture
def welcome(qt_app: object) -> Iterator[WelcomeView]:
    view = WelcomeView()
    view.resize(960, 720)
    view.set_recent_folders(
        (KYOTO, LAKE),
        {KYOTO: RecentFolder(148, "2026-10-30"), LAKE: RecentFolder(1, "2025-09-30")},
        today=TODAY,
    )
    view.set_opening_enabled(True)
    view.show()
    assert _settle(lambda: not view.opening and not view.fading)
    yield view
    view.close()
    view.deleteLater()


def _screen(welcome: WelcomeView) -> np.ndarray:
    """Return the whole screen as it is drawn, as rows of RGB."""
    image = welcome.grab().toImage().convertToFormat(QImage.Format.Format_RGB32)
    height, width = image.height(), image.width()
    rows = np.asarray(image.bits()).reshape(height, image.bytesPerLine())
    return rows[:, : width * 4].reshape(height, width, 4)[..., :3].astype(float)


def _area(welcome: WelcomeView, screen: np.ndarray, x: float, y: float, w: float, h: float):
    scale = screen.shape[1] / welcome.width()
    return screen[
        max(0, round(y * scale)) : round((y + h) * scale),
        max(0, round(x * scale)) : round((x + w) * scale),
    ]


def _of(welcome: WelcomeView, widget) -> tuple[float, float, float, float]:
    corner = widget.mapTo(welcome, widget.rect().topLeft())
    return corner.x(), corner.y(), widget.width(), widget.height()


def _ink(area: np.ndarray) -> float:
    """Return how much light is put on the background there."""
    return float(np.clip(area.mean(axis=2) - 20, 0, None).sum())


def _colour(area: np.ndarray) -> float:
    """Return how much colour is carried there."""
    return float((area.max(axis=2) - area.min(axis=2)).sum())


def _frame(welcome: WelcomeView, elapsed: float) -> np.ndarray:
    """Put the opening at *elapsed* milliseconds, hold it, and grab it."""
    for effect in welcome._rises.values():
        effect.setEnabled(True)
    welcome.apply_opening(float(elapsed))
    return _screen(welcome)


def _hero(welcome: WelcomeView) -> tuple[float, float, float, float]:
    return _of(welcome, welcome.hero)


def _drop_area(welcome: WelcomeView) -> tuple[float, float, float, float]:
    x, y, w, _h = _hero(welcome)
    size = theme.WELCOME_DROP_HEIGHT
    return x + w / 2 - size / 2, y + theme.WELCOME_DROP_CENTRE - size / 2, size, size


def _name_area(welcome: WelcomeView) -> tuple[float, float, float, float]:
    x, y, w, _h = _hero(welcome)
    top = y + theme.WELCOME_WORDMARK_BASELINE - theme.FONT_SIZE_WORDMARK
    return x - 80, top, w + 160, theme.FONT_SIZE_WORDMARK + theme.SPACE_SM


def _tagline_area(welcome: WelcomeView) -> tuple[float, float, float, float]:
    x, y, w, _h = _hero(welcome)
    top = y + theme.WELCOME_TAGLINE_BASELINE - theme.FONT_SIZE_TAGLINE - theme.SPACE_XS
    return x - 120, top, w + 240, theme.FONT_SIZE_TAGLINE + theme.SPACE_MD


# -- what rests on the screen --------------------------------------------------


def test_the_drop_name_and_tagline_are_drawn_in_the_room_kept_for_them(
    welcome: WelcomeView,
) -> None:
    screen = _screen(welcome)

    assert _colour(_area(welcome, screen, *_drop_area(welcome))) > 1e4, "no coloured drop"
    assert _ink(_area(welcome, screen, *_name_area(welcome))) > 1e4, "no name"
    assert _ink(_area(welcome, screen, *_tagline_area(welcome))) > 1e3, "no tagline"


def test_at_rest_the_name_is_white_not_ink(welcome: WelcomeView) -> None:
    """The letters come up in the drop's colours and settle to white."""
    area = _area(welcome, _screen(welcome), *_name_area(welcome))
    bright = area[area.mean(axis=2) > 200]
    assert bright.size
    assert float((bright.max(axis=1) - bright.min(axis=1)).mean()) < 12


def test_each_button_names_its_shortcut(welcome: WelcomeView) -> None:
    for button, label, keys in (
        (welcome.open_file_button, "welcome.open_file", "shortcut.open_file"),
        (welcome.open_folder_button, "welcome.open_folder", "shortcut.open_folder"),
    ):
        assert button.label == tr(label)
        assert tr(keys) in button.toolTip()
        assert button.height() == theme.WELCOME_BUTTON_HEIGHT
        assert button.width() == theme.WELCOME_BUTTON_WIDTH
        assert button.width() % theme.UNIT == 0


def test_open_file_sits_left_of_open_folder_across_the_column(welcome: WelcomeView) -> None:
    welcome._stop_opening()
    file_x, file_y, file_w, _h = _of(welcome, welcome.open_file_button)
    folder_x, folder_y, folder_w, _h = _of(welcome, welcome.open_folder_button)
    _hero_x, _hero_y, hero_w, _hero_h = _hero(welcome)
    assert file_y == folder_y
    assert folder_x - (file_x + file_w) == theme.WELCOME_BUTTON_GAP
    assert file_w + theme.WELCOME_BUTTON_GAP + folder_w == hero_w == theme.WELCOME_CONTENT_WIDTH


def test_each_button_asks_for_its_own_dialog(welcome: WelcomeView) -> None:
    welcome._stop_opening()
    asked: list[str] = []
    welcome.open_file_requested.connect(lambda: asked.append("file"))
    welcome.open_folder_requested.connect(lambda: asked.append("folder"))
    QTest.mouseClick(welcome.open_file_button, Qt.MouseButton.LeftButton)
    QTest.mouseClick(welcome.open_folder_button, Qt.MouseButton.LeftButton)
    assert asked == ["file", "folder"]


def test_a_button_centres_its_icon_and_whole_name(welcome: WelcomeView) -> None:
    """The shortcut is in the tooltip only, so the icon and the name sit
    in the middle of the button, the name not cut short."""
    welcome._stop_opening()
    screen = _screen(welcome)
    for button in welcome.buttons():
        left, name = button.content_layout()
        assert name == button.label
        assert left >= theme.SPACE_LG
        x, y, w, h = _of(welcome, button)
        inside = _area(welcome, screen, x + 4, y + 4, w - 8, h - 8)
        lit = np.nonzero(np.clip(inside.mean(axis=2) - 60, 0, None).sum(axis=0))[0]
        margin_left = lit[0]
        margin_right = inside.shape[1] - 1 - lit[-1]
        assert abs(margin_left - margin_right) <= 4 * screen.shape[1] / welcome.width()


def test_a_recent_folder_says_its_name_path_count_and_day(welcome: WelcomeView) -> None:
    kyoto, lake = welcome.recent_rows()
    assert kyoto.name == "2026-10 Kyoto"
    assert kyoto.toolTip() == KYOTO
    assert kyoto.side_text() == (tr("welcome.recent.count", count=148), tr("welcome.recent.yesterday"))
    assert lake.side_text()[0] == tr("welcome.recent.count_one")


def test_rows_are_as_tall_as_the_theme_says(welcome: WelcomeView) -> None:
    """A button's stylesheet height squashed them to 28 px once, and the
    name and the path were drawn over each other."""
    for row in welcome.recent_rows():
        assert row.height() == theme.WELCOME_ROW_HEIGHT


def test_a_folder_remembered_by_an_older_version_shows_no_count(qt_app: object) -> None:
    view = WelcomeView()
    view.set_recent_folders(("D:/old",))
    (row,) = view.recent_rows()
    assert row.side_text() == ("", "")
    view.deleteLater()


def test_clicking_a_folder_opens_it(welcome: WelcomeView) -> None:
    asked: list[str] = []
    welcome.folder_requested.connect(asked.append)

    QTest.mouseClick(welcome.recent_rows()[1], Qt.MouseButton.LeftButton)

    assert asked == [LAKE]


def _point_at(row, over: bool) -> None:
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QEnterEvent
    from PySide6.QtWidgets import QApplication

    if over:
        point = QPointF(row.width() / 2, row.height() / 2)
        QApplication.sendEvent(row, QEnterEvent(point, point, point))
    else:
        QApplication.sendEvent(row, QEvent(QEvent.Type.Leave))


def test_the_remove_button_shows_only_while_the_pointer_is_over_a_row(
    welcome: WelcomeView,
) -> None:
    kyoto, lake = welcome.recent_rows()
    assert not kyoto.offers_removal and not lake.offers_removal

    _point_at(kyoto, True)
    assert kyoto.offers_removal
    assert kyoto.remove_button.isVisibleTo(kyoto)
    assert not lake.offers_removal

    _point_at(kyoto, False)
    assert not kyoto.offers_removal


def test_the_remove_button_sits_at_the_right_end_in_place_of_the_count(
    welcome: WelcomeView,
) -> None:
    row = welcome.recent_rows()[0]
    button = row.remove_button
    assert button.width() == button.height() == theme.ICON_BUTTON_SIZE
    assert button.geometry().right() + 1 == row.width() - theme.SPACE_SM
    assert button.geometry().center().y() == row.rect().center().y()


def test_the_remove_button_says_what_it_does_and_its_key(welcome: WelcomeView) -> None:
    button = welcome.recent_rows()[0].remove_button
    assert tr("welcome.recent.remove") in button.toolTip()
    assert tr("shortcut.recent_remove") in button.toolTip()


def test_the_remove_button_asks_for_that_folder_to_go(welcome: WelcomeView) -> None:
    removed: list[str] = []
    opened: list[str] = []
    welcome.folder_removed.connect(removed.append)
    welcome.folder_requested.connect(opened.append)
    lake = welcome.recent_rows()[1]
    _point_at(lake, True)

    QTest.mouseClick(lake.remove_button, Qt.MouseButton.LeftButton)

    assert removed == [LAKE]
    assert opened == [], "removing a folder must not open it"


def test_delete_on_a_row_asks_for_that_folder_to_go(welcome: WelcomeView) -> None:
    removed: list[str] = []
    welcome.folder_removed.connect(removed.append)

    QTest.keyClick(welcome.recent_rows()[0], Qt.Key.Key_Delete)

    assert removed == [KYOTO]


def test_a_disabled_row_offers_no_removal(welcome: WelcomeView) -> None:
    row = welcome.recent_rows()[0]
    _point_at(row, True)
    row.setEnabled(False)
    assert not row.offers_removal
    removed: list[str] = []
    welcome.folder_removed.connect(removed.append)
    QTest.keyClick(row, Qt.Key.Key_Delete)
    assert removed == []


def test_the_list_goes_away_on_a_first_run(welcome: WelcomeView) -> None:
    welcome.set_recent_folders(())
    assert welcome.recent_rows() == []
    assert not welcome._recent_block.isVisibleTo(welcome)


def test_no_more_than_five_folders_are_listed(welcome: WelcomeView) -> None:
    welcome.set_recent_folders([f"D:/Photos/{index}" for index in range(8)])
    assert len(welcome.recent_rows()) == theme.WELCOME_RECENT_COUNT


# -- saying when ------------------------------------------------------------------


def test_a_day_is_said_the_way_a_person_would() -> None:
    assert recent_day(TODAY, TODAY) == tr("welcome.recent.today")
    assert recent_day(date(2026, 10, 30), TODAY) == tr("welcome.recent.yesterday")
    assert recent_day(date(2026, 10, 12), TODAY) == "10 月 12 日"
    assert recent_day(date(2025, 9, 30), TODAY) == "2025 年 9 月 30 日"


def test_english_names_the_month(qt_app: object) -> None:
    init_translator(LANGUAGE_EN)
    try:
        assert recent_day(date(2026, 10, 12), TODAY) == "Oct 12"
        assert recent_day(date(2025, 9, 30), TODAY) == "Sep 30, 2025"
        assert recent_count(1) == "1 photo"
        assert recent_count(148) == "148 photos"
    finally:
        init_translator("zh_TW")


# -- the pointer and a drag (PLAN.md 3.6) ----------------------------------------


def test_a_row_and_the_buttons_answer_the_pointer_over_80_ms(welcome: WelcomeView) -> None:
    for widget in (*welcome.buttons(), welcome.recent_rows()[0]):
        widget._hover.set_hovered(True)
        assert widget._hover.value < 1.0, "the hover switched at once"
        assert _settle(lambda w=widget: w._hover.value == 1.0)
        widget._hover.set_hovered(False)
        assert _settle(lambda w=widget: w._hover.value == 0.0)


def test_the_drop_hint_comes_up_to_the_accent_while_something_is_dragged(
    welcome: WelcomeView,
) -> None:
    welcome.set_drop_target(True)
    assert _settle(lambda: welcome.drop_hint.lit == 1.0)
    welcome.set_drop_target(False)
    assert _settle(lambda: welcome.drop_hint.lit == 0.0)
    assert mix(theme.TEXT_TERTIARY, theme.ACCENT, 1.0).name() == theme.ACCENT.lower()


# -- the opening (PLAN.md 3.7) ----------------------------------------------------


def test_the_opening_lasts_as_long_as_its_last_stage() -> None:
    assert opening_length() >= theme.OPENING_HINT_AT_MS + theme.OPENING_ITEM_MS
    assert opening_length() >= (
        theme.OPENING_FALL_MS
        + (len(theme.INK_COLOURS) - 1) * theme.OPENING_RIPPLE_GAP_MS
        + theme.OPENING_RIPPLE_MS
    )


def test_starting_the_program_plays_the_opening_and_leaves_nothing_behind(
    qt_app: object,
) -> None:
    view = WelcomeView()
    view.resize(960, 720)
    try:
        view.show()
        assert view.opening, "the screen was there all at once"
        assert _settle(lambda: not view.opening)
        assert view.canvas.elapsed is None
        for effect in view._rises.values():
            assert not effect.isEnabled()
            assert effect.progress == 1.0
    finally:
        view.close()
        view.deleteLater()


def test_the_drop_falls_before_it_lands(welcome: WelcomeView) -> None:
    """Halfway through the fall it is above where it comes to rest."""
    welcome._stop_opening()
    x, y, w, h = _drop_area(welcome)
    above = (x, y - 3 * h, w, 2 * h)

    falling = _frame(welcome, theme.OPENING_FALL_MS * 0.6)
    assert _colour(_area(welcome, falling, *above)) > _colour(
        _area(welcome, falling, x, y, w, h)
    )

    landed = _frame(welcome, theme.OPENING_FALL_MS + theme.OPENING_GROW_MS)
    assert _colour(_area(welcome, landed, x, y, w, h)) > 1e4
    welcome._stop_opening()


def test_rings_spread_from_where_it_lands_past_the_column(welcome: WelcomeView) -> None:
    """The rings reach wider than the column: the canvas is the whole screen."""
    welcome._stop_opening()
    x, y, w, _h = _of(welcome, welcome._column)
    drop_y = _drop_area(welcome)[1]
    outside = (x + w + 8, drop_y - 40, 120, 160)

    before = _frame(welcome, theme.OPENING_FALL_MS - 1)
    spreading = _frame(welcome, theme.OPENING_FALL_MS + theme.OPENING_RIPPLE_MS * 0.55)
    gone = _frame(welcome, opening_length())

    assert _colour(_area(welcome, before, *outside)) == 0
    assert _colour(_area(welcome, spreading, *outside)) > 500
    assert _colour(_area(welcome, gone, *outside)) == 0
    welcome._stop_opening()


def test_the_name_comes_up_from_its_first_letter_in_ink(welcome: WelcomeView) -> None:
    welcome._stop_opening()
    x, y, w, h = _name_area(welcome)
    left, right = (x, y, w / 2, h), (x + w / 2, y, w / 2, h)

    # The rings cross this band too, the same on both sides; the letters are
    # what makes the left half outweigh the right.
    before = _frame(welcome, theme.OPENING_LETTERS_AT_MS - 1)
    early = _frame(welcome, theme.OPENING_LETTERS_AT_MS + 3 * theme.OPENING_LETTER_GAP_MS)
    assert _ink(_area(welcome, early, *left)) > _ink(_area(welcome, before, *left)) + 1e3
    assert _ink(_area(welcome, early, *left)) > _ink(_area(welcome, early, *right)) + 1e3
    assert _colour(_area(welcome, early, *left)) > 1e3, "the letters arrived without ink"
    welcome._stop_opening()


def test_the_tagline_is_laid_down_from_left_to_right(welcome: WelcomeView) -> None:
    welcome._stop_opening()
    x, y, w, h = _tagline_area(welcome)
    half = _frame(welcome, theme.OPENING_TAGLINE_AT_MS + theme.OPENING_TAGLINE_MS / 2)
    left = _ink(_area(welcome, half, x, y, w / 2, h))
    right = _ink(_area(welcome, half, x + w / 2, y, w / 2, h))
    assert left > 1e3
    assert right < left / 4
    welcome._stop_opening()


def test_the_rest_rises_into_place_in_order(welcome: WelcomeView) -> None:
    welcome._stop_opening()
    row = _of(welcome, welcome.recent_rows()[0])
    hint = _of(welcome, welcome.drop_hint)

    at_button = _frame(welcome, theme.OPENING_BUTTON_AT_MS + theme.OPENING_ITEM_MS)
    for button in welcome.buttons():
        assert _ink(_area(welcome, at_button, *_of(welcome, button))) > 1e3
    assert _ink(_area(welcome, at_button, *hint)) < 50

    done = _frame(welcome, opening_length())
    assert _ink(_area(welcome, done, *row)) > 1e3
    assert _ink(_area(welcome, done, *hint)) > 1e2
    welcome._stop_opening()


def test_nothing_can_be_pressed_before_the_opening_has_shown_it(
    welcome: WelcomeView,
) -> None:
    """An opacity of zero hides a control from the eye and from nothing else."""
    asked: list[str] = []
    welcome.open_file_requested.connect(lambda: asked.append("file"))
    welcome.open_folder_requested.connect(lambda: asked.append("open"))
    welcome.folder_requested.connect(asked.append)

    _frame(welcome, theme.OPENING_TAGLINE_AT_MS)
    QTest.mouseClick(welcome.open_file_button, Qt.MouseButton.LeftButton)
    QTest.mouseClick(welcome.open_folder_button, Qt.MouseButton.LeftButton)
    QTest.mouseClick(welcome.recent_rows()[0], Qt.MouseButton.LeftButton)
    for _ in range(6):
        QTest.keyClick(welcome, Qt.Key.Key_Tab)
    QTest.keyClick(welcome, Qt.Key.Key_Return)
    assert asked == [], f"a control nobody could see answered: {asked}"

    _frame(welcome, theme.OPENING_BUTTON_AT_MS)
    QTest.mouseClick(welcome.open_folder_button, Qt.MouseButton.LeftButton)
    assert asked == ["open"]
    welcome._stop_opening()


def test_holding_the_screen_back_does_not_move_it(welcome: WelcomeView) -> None:
    welcome._stop_opening()
    settled = _hero(welcome)
    _frame(welcome, 0)
    welcome.layout().activate()
    assert _hero(welcome) == settled
    welcome._stop_opening()


def test_the_opening_is_played_once_and_not_again(welcome: WelcomeView) -> None:
    """Coming back after closing a folder plays the plain fade."""
    welcome.hide()
    welcome.show()
    assert not welcome.opening
    assert welcome.fading
    assert _settle(lambda: not welcome.fading)


def test_nothing_animates_when_windows_is_set_to_show_no_animation(
    qt_app: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PLAN.md 3.6: every transition becomes a straight switch."""
    monkeypatch.setattr(native, "animations_enabled", lambda: False)
    view = WelcomeView()
    view.resize(960, 720)
    view.set_opening_enabled(True)
    try:
        view.show()
        assert not view.opening, "the opening played anyway"
        assert not view.fading
        assert view.canvas.elapsed is None
        assert all(button.isEnabled() for button in view.buttons())
        view.set_drop_target(True)
        assert view.drop_hint.lit == 1.0
    finally:
        view.close()
        view.deleteLater()


def test_the_drop_hint_follows_the_list_rather_than_the_bottom_of_the_window(
    welcome: WelcomeView,
) -> None:
    """Pinned to the bottom it read as a footer belonging to nothing."""
    _x, top, _w, height = _of(welcome, welcome._column)
    _hx, hint_top, _hw, _hh = _of(welcome, welcome.drop_hint)
    assert hint_top == top + height + theme.WELCOME_GROUP_GAP

    welcome.resize(welcome.width(), welcome.height() + 300)
    welcome.layout().activate()
    _x, top, _w, height = _of(welcome, welcome._column)
    _hx, hint_top, _hw, _hh = _of(welcome, welcome.drop_hint)
    assert hint_top == top + height + theme.WELCOME_GROUP_GAP
