"""Tests for the filter tab of the right hand panel (the film looks).

A look is one of a set with exactly one taken, so the list has a "none" at
the top and the current one is marked. Nothing happens on hover; a click
puts the look on. The saved presets are in the adjust tab
(tests/test_presets_group.py).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from core.edit_state import EditState
from core.filters import NO_FILTER, Look

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QEvent, QPointF, QSize, Qt  # noqa: E402
from PySide6.QtGui import QEnterEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from app.i18n import tr  # noqa: E402
from ui.looks_tab import LooksTab, look_name  # noqa: E402
from ui import theme  # noqa: E402
from ui.inspector import Inspector, InspectorTab  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOOKS = (
    Look.identity("warm-negative"),
    Look.identity("rich-negative"),
    Look.identity("cool-slide"),
)
RAW_REST = EditState.for_photo(True, temperature=5600.0, tint=0.0)


@pytest.fixture
def tab(qt_app: object) -> LooksTab:
    page = LooksTab()
    page.set_filters(LOOKS)
    page.set_photo(RAW_REST)
    return page


def _hover(row: object, entered: bool = True) -> None:
    point = QPointF(1.0, 1.0)
    if entered:
        row.enterEvent(QEnterEvent(point, point, point))
    else:
        row.leaveEvent(QEvent(QEvent.Type.Leave))


# -- the list --------------------------------------------------------------


def test_the_list_offers_none_and_every_installed_look(tab: LooksTab) -> None:
    assert list(tab.rows()) == [NO_FILTER, "warm-negative", "rich-negative", "cool-slide"]


def test_the_list_says_so_when_there_are_no_looks(qt_app: object) -> None:
    page = LooksTab()
    page.set_filters(())
    page.set_photo(RAW_REST)

    assert page.rows() == {}


def test_rebuilding_the_list_does_not_leave_the_old_rows_behind(tab: LooksTab) -> None:
    tab.set_filters((Look.identity("warm-negative"),))

    assert list(tab.rows()) == [NO_FILTER, "warm-negative"]


def test_the_look_in_use_is_the_one_marked(tab: LooksTab) -> None:
    tab.set_state(RAW_REST.with_filter("rich-negative"))

    marked = [name for name, row in tab.rows().items() if row.isChecked()]
    assert marked == ["rich-negative"]


def test_no_look_in_use_marks_the_none_row(tab: LooksTab) -> None:
    marked = [name for name, row in tab.rows().items() if row.isChecked()]
    assert marked == [NO_FILTER]


# -- the category the looks sit in ----------------------------------------


def test_the_looks_sit_in_a_category_and_the_other_two_rows_do_not(
    tab: LooksTab,
) -> None:
    """A category holds what belongs to it. "None" is the photo wearing
    nothing at all and the strength belongs to whichever look is on, so
    neither is a film look and neither stands under that header."""
    group = tab.groups()[0]

    for identifier, row in tab.rows().items():
        inside = group.isAncestorOf(row)
        assert inside == bool(identifier), f"{identifier or 'none'} is in the wrong place"
    assert not group.isAncestorOf(tab.strength_row())


def test_the_category_is_named_rather_than_repeating_the_tab(tab: LooksTab) -> None:
    """The tab is called Filter and so was its one group, which left the
    word said twice and no layer between the tab and the looks."""
    assert tab.groups()[0].header.text() == tr("filters.group.film")
    assert tab.groups()[0].header.text() != tr("panel.tab.filters")


def test_each_look_sits_under_its_own_category(qt_app: object) -> None:
    page = LooksTab()
    medium = replace(Look.identity("natural-medium-format"), category="medium-format")
    page.set_filters((*LOOKS, medium))
    page.set_photo(RAW_REST)

    film, medium_group = page.groups()
    assert film.header.text() == tr("filters.group.film")
    assert medium_group.header.text() == tr("filters.group.medium-format")
    assert medium_group.isAncestorOf(page.rows()["natural-medium-format"])
    assert not film.isAncestorOf(page.rows()["natural-medium-format"])
    assert film.isAncestorOf(page.rows()["cool-slide"])


def test_the_categories_keep_their_order_whatever_order_the_files_come_in(
    qt_app: object,
) -> None:
    page = LooksTab()
    page.set_filters(
        (
            replace(Look.identity("street-standard"), category="street"),
            replace(Look.identity("natural-medium-format"), category="medium-format"),
            Look.identity("cool-slide"),
        )
    )

    assert [group.header.text() for group in page.groups()] == [
        tr("filters.group.film"),
        tr("filters.group.medium-format"),
        tr("filters.group.street"),
    ]


def test_a_category_with_nothing_installed_is_not_shown(tab: LooksTab) -> None:
    assert [group.header.text() for group in tab.groups()] == [tr("filters.group.film")]


def test_rebuilding_the_list_moves_a_look_out_of_its_old_category(qt_app: object) -> None:
    page = LooksTab()
    page.set_filters((replace(Look.identity("x"), category="medium-format"),))
    page.set_filters((Look.identity("x"),))

    (film,) = page.groups()
    assert film.isAncestorOf(page.rows()["x"])


def test_a_row_outside_the_category_still_lines_up_with_the_ones_inside(
    tab: LooksTab,
) -> None:
    """A group gives its rows the panel's side padding; a row standing
    outside one has to be given the same padding or it runs to the edges."""
    tab.resize(theme.PANEL_WIDTH, 400)
    tab.show()
    try:
        QTest.qWaitForWindowExposed(tab)
        inside = tab.rows()["warm-negative"]
        for row in (tab.rows()[NO_FILTER], tab.strength_row()):
            assert row.mapTo(tab, row.rect().topLeft()).x() == (
                inside.mapTo(tab, inside.rect().topLeft()).x()
            )
    finally:
        tab.hide()


# -- choosing --------------------------------------------------------------


def test_clicking_a_look_asks_for_it(tab: LooksTab) -> None:
    chosen: list[str] = []
    tab.chosen.connect(chosen.append)

    QTest.mouseClick(tab.rows()["warm-negative"], Qt.MouseButton.LeftButton)

    assert chosen == ["warm-negative"]


def test_clicking_the_look_already_on_asks_for_nothing(tab: LooksTab) -> None:
    tab.set_state(RAW_REST.with_filter("warm-negative"))
    chosen: list[str] = []
    tab.chosen.connect(chosen.append)

    QTest.mouseClick(tab.rows()["warm-negative"], Qt.MouseButton.LeftButton)

    assert chosen == []
    assert tab.rows()["warm-negative"].isChecked()


def test_clicking_none_takes_the_look_off(tab: LooksTab) -> None:
    tab.set_state(RAW_REST.with_filter("warm-negative"))
    chosen: list[str] = []
    tab.chosen.connect(chosen.append)

    QTest.mouseClick(tab.rows()[NO_FILTER], Qt.MouseButton.LeftButton)

    assert chosen == [NO_FILTER]


# -- no preview on hover --------------------------------------------------


def test_resting_on_a_look_changes_nothing(tab: LooksTab) -> None:
    """A preview that came and went as the pointer crossed the list made the
    photo flicker; a look now goes on only when it is clicked."""
    chosen: list[str] = []
    tab.chosen.connect(chosen.append)

    _hover(tab.rows()["rich-negative"])
    _hover(tab.rows()["rich-negative"], entered=False)

    assert chosen == []
    assert not hasattr(tab, "previewed")


# -- the strength ----------------------------------------------------------


def test_the_strength_waits_until_a_look_is_on(tab: LooksTab) -> None:
    assert not tab.strength_row().isEnabled()

    tab.set_state(RAW_REST.with_filter("warm-negative"))
    assert tab.strength_row().isEnabled()


def test_the_strength_follows_the_photo(tab: LooksTab) -> None:
    tab.set_state(RAW_REST.with_filter("warm-negative").with_value("filter_strength", 40.0))

    assert tab.strength_row().value == pytest.approx(40.0)


def test_following_the_photo_does_not_announce_a_change(tab: LooksTab) -> None:
    """Undo and presets set the tab; only the user's hand reports a change."""
    changes: list[tuple[float, bool]] = []
    tab.strength_changed.connect(lambda value, dragging: changes.append((value, dragging)))

    tab.set_state(RAW_REST.with_filter("warm-negative").with_value("filter_strength", 25.0))

    assert changes == []


# -- names -----------------------------------------------------------------


def test_a_look_is_called_what_the_translations_call_it() -> None:
    assert look_name("warm-negative") == tr("filters.name.warm-negative")
    assert look_name(NO_FILTER) == tr("filters.none")


def test_a_look_nobody_has_translated_is_called_by_its_own_name() -> None:
    """A filter file dropped in by hand still reads as something."""
    assert look_name("xyzzy") == "XYZZY"


def test_every_shipped_look_has_a_name_in_both_languages() -> None:
    from core.filters import load_catalogue

    from app.paths import filters_dir

    for language in ("zh_TW", "en"):
        path = PROJECT_ROOT / "resources" / "i18n" / f"{language}.json"
        strings = json.loads(path.read_text(encoding="utf-8-sig"))
        for look in load_catalogue(filters_dir()):
            key = f"filters.name.{look.identifier}"
            assert key in strings, f"{language} has no name for {look.identifier}"


# -- the tab in the panel --------------------------------------------------


def test_the_panel_has_the_filter_tab_right_after_adjust(qt_app: object) -> None:
    """A look is the same kind of decision as the adjustments, so it sits
    beside them; the presets are no longer a tab of their own."""
    assert [tab.name for tab in InspectorTab] == ["ADJUST", "LOOKS", "RETOUCH", "CROP"]


def test_the_looks_tab_can_be_shown(qt_app: object) -> None:
    panel = Inspector()
    panel.set_filters(LOOKS)

    panel.set_current_tab(InspectorTab.LOOKS)

    assert panel.current_tab is InspectorTab.LOOKS
    assert list(panel.looks_tab.rows()) == [NO_FILTER, "warm-negative", "rich-negative", "cool-slide"]


def test_the_tab_strip_still_fits_the_panel_in_both_languages(qt_app: object) -> None:
    """The tabs share 300 px, and English is the tight one.

    Measured with Inter when there were five: with 4 px of side padding they
    wanted 312 px and Presets lost its last letter; with none they wanted
    272 px. Four leave room, but this still fails if a tab is added that does
    not fit, which is the point of it.
    """
    from PySide6.QtGui import QFontDatabase

    from app.i18n import init_translator
    from app.paths import resources_dir
    from ui import theme

    # The measurement is only worth anything in the font that ships, which
    # the program registers at start-up and a test does not. It is taken out
    # again afterwards: left in, it changes how wide every string in every
    # later test is, and a test that reads a pixel of the window fails.
    registered = [
        QFontDatabase.addApplicationFont(str(font))
        for font in sorted((resources_dir() / "fonts").glob("*.tt[fc]"))
    ]

    try:
        for language in ("zh_TW", "en"):
            init_translator(language)
            panel = Inspector()
            panel.setStyleSheet(theme.build_qss())
            panel.ensurePolished()
            wanted = sum(
                panel.tab_buttons()[tab].sizeHint().width() for tab in InspectorTab
            )
            assert wanted <= theme.PANEL_WIDTH, (
                f"{language}: the tab strip wants {wanted} px of {theme.PANEL_WIDTH}"
            )
    finally:
        init_translator("zh_TW")
        for identifier in registered:
            QFontDatabase.removeApplicationFont(identifier)


# -- the tab joined to the controller --------------------------------------


def _window_with_a_photo(qt_app: object, photo: Path):
    """Return a window with *photo* open, waiting properly for it to arrive."""
    from PIL import Image
    from PySide6.QtCore import QEventLoop, QTimer

    from app.settings import Settings
    from ui.main_window import MainWindow

    Image.new("RGB", (240, 160), (120, 110, 100)).save(photo, "JPEG")
    window = MainWindow(Settings())

    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    connection = window.controller.preview_ready.connect(lambda *_: loop.quit())
    window.controller.open_file(photo)
    timer.start(20_000)
    loop.exec()
    window.controller.preview_ready.disconnect(connection)
    assert window.controller.current == photo, "the photo never opened"
    return window


def test_choosing_a_look_in_the_window_puts_it_on_the_photo(
    qt_app: object, tmp_path: Path
) -> None:
    """The one thing neither the tab nor the controller can prove alone."""
    window = _window_with_a_photo(qt_app, tmp_path / "a.jpg")
    try:
        installed = window.inspector.looks_tab.rows()
        assert installed, "no filters were handed to the tab"
        wanted = next(name for name in installed if name)

        window.inspector.set_current_tab(InspectorTab.LOOKS)
        QTest.mouseClick(installed[wanted], Qt.MouseButton.LeftButton)

        assert window.controller.state.filter == wanted
    finally:
        window.controller.shutdown()
        window.hide()
        window.deleteLater()


def test_the_strength_slider_reaches_the_photo(qt_app: object, tmp_path: Path) -> None:
    window = _window_with_a_photo(qt_app, tmp_path / "b.jpg")
    try:
        wanted = next(name for name in window.inspector.looks_tab.rows() if name)
        window.controller.set_adjustments(window.controller.state.with_filter(wanted))

        window.inspector.looks_tab.strength_changed.emit(30.0, False)

        assert window.controller.state.filter_strength == pytest.approx(30.0)
    finally:
        window.controller.shutdown()
        window.hide()
        window.deleteLater()


# -- the thumbnails ---------------------------------------------------------


def _square(colour: tuple[int, int, int], side: int = 80) -> np.ndarray:
    return np.full((side, side, 3), colour, dtype=np.uint8)


def test_each_row_holds_a_thumbnail_at_its_left(tab: LooksTab) -> None:
    """The looks are told apart by eye, so the photo comes before the name."""
    for row in tab.rows().values():
        assert row.height() == theme.FILTER_ROW_HEIGHT
        assert row.iconSize().height() == theme.FILTER_THUMBNAIL_SIZE
        assert row.property("thumbnail") is True
        assert not row.icon().isNull()


def test_a_row_waits_with_an_empty_patch_until_its_photo_arrives(tab: LooksTab) -> None:
    assert not any(row.has_thumbnail() for row in tab.rows().values())

    tab.set_thumbnails({NO_FILTER: _square((10, 20, 30)), "cool-slide": _square((40, 50, 60))})

    shown = {name for name, row in tab.rows().items() if row.has_thumbnail()}
    assert shown == {NO_FILTER, "cool-slide"}


def test_the_thumbnail_is_the_picture_it_was_given(tab: LooksTab) -> None:
    """The square is the photo itself, not a tint laid over the patch."""
    tab.set_thumbnails({"rich-negative": _square((200, 40, 40))})

    icon = tab.rows()["rich-negative"].icon()
    side = theme.FILTER_THUMBNAIL_SIZE
    picture = icon.pixmap(QSize(side + theme.FILTER_THUMBNAIL_GAP, side)).toImage()
    # The middle of the square, which is the left part of the picture: the
    # gap before the name is the rest of it.
    across = picture.width() * side // (2 * (side + theme.FILTER_THUMBNAIL_GAP))
    middle = picture.pixelColor(across, picture.height() // 2)
    assert (middle.red(), middle.green(), middle.blue()) == (200, 40, 40)


def test_clearing_the_thumbnails_brings_the_empty_patches_back(tab: LooksTab) -> None:
    """Another photo is opening: the old photo's looks must not stay on show."""
    tab.set_thumbnails({name: _square((90, 90, 90)) for name in tab.rows()})
    tab.set_thumbnails({})

    assert not any(row.has_thumbnail() for row in tab.rows().values())


def test_a_rebuilt_list_shows_the_thumbnails_it_already_had(tab: LooksTab) -> None:
    tab.set_thumbnails({"warm-negative": _square((90, 60, 30))})
    tab.set_filters(LOOKS)

    assert tab.rows()["warm-negative"].has_thumbnail()



# -- the strength stays in reach -------------------------------------------


def test_the_strength_is_pinned_below_the_list(tab: LooksTab) -> None:
    """However far the list is scrolled, the strength stays in reach."""
    area = tab.scroll_area()

    assert not area.widget().isAncestorOf(tab.strength_row())
    assert tab.isAncestorOf(tab.strength_row())


def test_the_strength_and_the_list_go_away_with_the_photo(tab: LooksTab) -> None:
    tab.set_photo(None)

    assert not tab.strength_row().isVisibleTo(tab)
    assert not tab.scroll_area().isVisibleTo(tab)
