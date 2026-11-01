"""Tests for the collage's controller, panel and place in the window (PLAN.md 3.13)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.settings import CollageSettings, Settings, WindowState
from core import collage, sidecar
from core.collage import Layout, Placement
from core.edit_state import Edit, EditState
from core.exporter import ExportOptions
from core.geometry import Geometry

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402

from ui.collage_panel import CollagePanel  # noqa: E402
from ui.export_dialog import ExportDialog  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402

TIMEOUT_MS = 20_000


def _jpeg(path: Path, colour: tuple[int, int, int], size: tuple[int, int] = (600, 400)) -> Path:
    Image.new("RGB", size, colour).save(path, "JPEG", quality=95)
    return path


def _wait(condition, timeout_ms: int = TIMEOUT_MS) -> None:
    """Let the event loop run until *condition* holds."""
    deadline = time.perf_counter() + timeout_ms / 1000
    while not condition():
        assert time.perf_counter() < deadline, "the condition never came true"
        loop = QEventLoop()
        QTimer.singleShot(10, loop.quit)
        loop.exec()


@pytest.fixture
def window(qt_app: object) -> Iterator[MainWindow]:
    settings = Settings(window=WindowState(x=0, y=0, width=1440, height=900))
    main_window = MainWindow(settings)
    main_window.show()
    main_window.resize(1440, 900)
    qt_app.processEvents()  # type: ignore[attr-defined]
    yield main_window
    main_window.collage.shutdown()
    main_window.hide()
    main_window.deleteLater()


# -- entering and leaving ------------------------------------------------------


def test_the_toolbar_button_swaps_the_editor_for_the_collage_and_back(
    window: MainWindow,
) -> None:
    button = window.toolbar.collage_button
    button.click()
    assert window.in_collage
    assert button.isChecked()
    assert window.collage_canvas.isVisible() and window.collage_panel.isVisible()
    assert not window.preview.isVisible() and not window.inspector.isVisible()

    button.click()
    assert not window.in_collage
    assert not button.isChecked()
    assert window.preview.isVisible() and not window.collage_canvas.isVisible()


def test_escape_leaves_the_collage(window: MainWindow) -> None:
    window.enter_collage()
    window.on_escape()
    assert not window.in_collage
    assert not window.toolbar.collage_button.isChecked()


def test_the_icon_button_names_its_shortcut(window: MainWindow) -> None:
    from app.i18n import tr

    assert tr("shortcut.collage") in window.toolbar.collage_button.toolTip()


def test_editing_shortcuts_do_nothing_while_the_collage_is_open(window: MainWindow) -> None:
    window.enter_collage()
    ran: list[str] = []
    window._on_shortcut("shortcut.crop", lambda: ran.append("crop"))
    window._on_shortcut("shortcut.undo", lambda: ran.append("undo"))
    window._on_shortcut("shortcut.export", lambda: ran.append("export"))
    assert ran == ["export"]


def test_export_waits_for_both_photos(window: MainWindow, tmp_path: Path) -> None:
    window.enter_collage()
    assert not window.toolbar.export_button.isEnabled()
    window.collage.load(0, _jpeg(tmp_path / "a.jpg", (200, 40, 40)))
    _wait(lambda: window.collage.photo(0) is not None)
    assert not window.toolbar.export_button.isEnabled()
    window.collage.load(1, _jpeg(tmp_path / "b.jpg", (40, 40, 200)))
    _wait(lambda: window.collage.complete)
    assert window.toolbar.export_button.isEnabled()
    assert window.collage_panel.name_labels[1].text() == "b.jpg"


def test_photos_chosen_together_in_the_filmstrip_go_straight_in(
    window: MainWindow, tmp_path: Path
) -> None:
    folder = tmp_path / "photos"
    folder.mkdir()
    for name, colour in (("a.jpg", (200, 0, 0)), ("b.jpg", (0, 200, 0)), ("c.jpg", (0, 0, 200))):
        _jpeg(folder / name, colour)
    window.controller.open(folder)
    _wait(lambda: bool(window.controller.photos))
    window.filmstrip.set_photos(window.controller.photos)
    window.filmstrip._selected = {folder / "b.jpg", folder / "c.jpg"}
    window.enter_collage()
    assert window.collage.paths == (folder / "b.jpg", folder / "c.jpg")


def test_a_thumbnail_clicked_in_the_collage_fills_the_next_empty_half(
    window: MainWindow, tmp_path: Path
) -> None:
    path = _jpeg(tmp_path / "a.jpg", (10, 10, 10))
    window.enter_collage()
    window._on_thumbnail_clicked(path)
    assert window.collage.paths == (path, None)


def test_a_drag_from_inside_the_window_is_not_opened(window: MainWindow) -> None:
    """A thumbnail on its way to the collage must not reopen its folder."""

    class FromInside:
        """A drag whose source is a widget of this program."""

        accepted = False

        def source(self) -> object:
            return window.filmstrip

        def ignore(self) -> None:
            self.accepted = False

        def acceptProposedAction(self) -> None:  # noqa: N802 - Qt naming
            self.accepted = True

    event = FromInside()
    window.dragEnterEvent(event)
    assert not event.accepted


# -- the controller ------------------------------------------------------------


def test_each_photo_goes_in_with_its_own_edit(window: MainWindow, tmp_path: Path) -> None:
    """A photo darkened in the editor is darker in the collage too."""
    plain = _jpeg(tmp_path / "plain.jpg", (128, 128, 128))
    darker = _jpeg(tmp_path / "darker.jpg", (128, 128, 128))
    edited = EditState.for_photo(False).with_value("exposure", -1.0)
    application = sidecar.Application("TintaPhoto", "test")
    reference = Edit(EditState.for_photo(False), Geometry())
    sidecar.save(darker, Edit(edited, Geometry()), reference, application)
    assert sidecar.is_edited(darker)
    window.collage.load(0, plain)
    window.collage.load(1, darker)
    _wait(lambda: window.collage.complete)
    first = window.collage.photo(0).image.mean()
    second = window.collage.photo(1).image.mean()
    assert second < first - 20


def test_the_layout_is_remembered_in_the_settings(window: MainWindow) -> None:
    layout = Layout(orientation=collage.ORIENTATION_PORTRAIT, split=collage.SPLIT_STACKED, gap=6)
    window.collage.set_layout(layout)
    stored = window._settings.collage
    assert stored.layout == layout.sane()
    again = CollageSettings.from_dict(stored.to_dict())
    assert again.layout == layout.sane()


def test_a_damaged_layout_in_the_settings_falls_back_to_the_default() -> None:
    settings = CollageSettings.from_dict({"orientation": "round", "gap": "lots"})
    assert settings.layout == Layout()
    assert Settings.from_dict({"collage": 5}).collage.layout == Layout()


def test_the_collage_is_written_at_the_canvas_size(
    window: MainWindow, tmp_path: Path
) -> None:
    window.collage.set_layout(Layout(orientation=collage.ORIENTATION_PORTRAIT, gap=10))
    window.collage.load(0, _jpeg(tmp_path / "left.jpg", (220, 30, 30)))
    window.collage.load(1, _jpeg(tmp_path / "right.jpg", (30, 30, 220), (400, 600)))
    _wait(lambda: window.collage.complete)
    window.collage.set_placement(1, Placement(zoom=2.0))
    finished: list[object] = []
    window.collage.export_finished.connect(finished.append)
    target = tmp_path / "out" / "both.jpg"
    assert window.collage.export(target, ExportOptions())
    _wait(lambda: bool(finished))
    written = finished[0].path
    assert written == target and target.exists()
    with Image.open(target) as image:
        assert image.size == (1080, 1350)
        assert image.info.get("icc_profile")
        pixels = np.asarray(image.convert("RGB"))
    first, second = collage.cells(window.collage.layout)
    assert pixels[675, first.width // 2, 0] > 180
    assert pixels[675, second.x + second.width // 2, 2] > 180
    assert np.all(pixels[675, first.width + 2 : second.x - 2] > 245)

    # Nothing is overwritten (X-05).
    finished.clear()
    assert window.collage.export(target, ExportOptions())
    _wait(lambda: bool(finished))
    assert finished[0].path.name == "both_2.jpg"


def test_the_collage_offers_both_photos_names(window: MainWindow, tmp_path: Path) -> None:
    window.collage.load(0, _jpeg(tmp_path / "DSC0001.jpg", (1, 1, 1)))
    window.collage.load(1, _jpeg(tmp_path / "DSC0002.jpg", (1, 1, 1)))
    assert window.collage.default_name() == "DSC0001_DSC0002_collage"
    assert window.collage.default_folder() == tmp_path


def test_a_file_that_is_not_a_photo_is_refused(window: MainWindow, tmp_path: Path) -> None:
    messages: list[str] = []
    window.collage.error.connect(messages.append)
    window.collage.load(0, tmp_path / "notes.txt")
    assert window.collage.paths == (None, None)
    assert messages


# -- the panel and the dialog -------------------------------------------------


def test_the_panel_names_the_halves_after_the_split(qt_app: object) -> None:
    panel = CollagePanel()
    announced: list[Layout] = []
    panel.layout_changed.connect(announced.append)
    from app.i18n import tr

    assert panel._row_labels[0].text() == tr("collage.slot.left")
    panel.split_choice.buttons()[collage.SPLIT_STACKED].click()
    assert announced and announced[-1].split == collage.SPLIT_STACKED
    assert panel._row_labels[0].text() == tr("collage.slot.top")
    assert panel.size_label.text() == tr("export.pixels", width=1920, height=1080)


def test_the_panel_shows_without_announcing(qt_app: object) -> None:
    panel = CollagePanel()
    announced: list[Layout] = []
    panel.layout_changed.connect(announced.append)
    panel.set_layout(Layout(orientation=collage.ORIENTATION_PORTRAIT, gap=12))
    assert not announced
    assert panel.layout_value == Layout(orientation=collage.ORIENTATION_PORTRAIT, gap=12)


def test_the_export_dialog_for_a_collage_has_a_fixed_size(
    qt_app: object, tmp_path: Path
) -> None:
    settings = Settings().export
    settings.long_edge = 2000
    settings.suffix = "_mine"
    dialog = ExportDialog(
        tmp_path / "a.jpg",
        settings,
        collage_name="a_b_collage",
        collage_size=(1920, 1080),
    )
    assert not dialog.size_choice.isVisibleTo(dialog)
    assert not dialog.keep_metadata_switch.isVisibleTo(dialog)
    assert dialog.options.long_edge is None
    assert dialog.target == tmp_path / "a_b_collage.jpg"
    chosen = dialog.chosen_settings
    # A collage does not change what a single photo is exported as.
    assert chosen.long_edge == 2000 and chosen.suffix == "_mine"


# -- gestures on the canvas ---------------------------------------------------


def _filled(window: MainWindow, tmp_path: Path) -> None:
    window.enter_collage()
    window.collage.load(0, _jpeg(tmp_path / "a.jpg", (200, 40, 40), (900, 600)))
    window.collage.load(1, _jpeg(tmp_path / "b.jpg", (40, 40, 200), (900, 600)))
    _wait(lambda: window.collage.complete)


def test_clicking_an_empty_half_asks_for_a_photo(window: MainWindow) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    window.enter_collage()
    # The window would open the file picker, which waits for a person.
    window.collage_canvas.choose_requested.disconnect(window.choose_collage_photo)
    asked: list[int] = []
    window.collage_canvas.choose_requested.connect(asked.append)
    centre = window.collage_canvas.cell_rect(1).center().toPoint()
    QTest.mouseClick(window.collage_canvas, Qt.MouseButton.LeftButton, pos=centre)
    assert asked == [1]


def test_dragging_a_photo_moves_it_and_clicking_it_asks_nothing(
    window: MainWindow, tmp_path: Path
) -> None:
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    _filled(window, tmp_path)
    canvas = window.collage_canvas
    asked: list[int] = []
    canvas.choose_requested.connect(asked.append)
    start = canvas.cell_rect(0).center().toPoint()
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(canvas, start + QPoint(60, 0))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=start + QPoint(60, 0))
    assert window.collage.placement(0).focus_x < 0.5
    assert window.collage.placement(1) == Placement()
    assert not asked


def test_the_wheel_zooms_the_photo_under_it_and_a_double_click_resets_it(
    window: MainWindow, tmp_path: Path
) -> None:
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtTest import QTest

    _filled(window, tmp_path)
    canvas = window.collage_canvas
    centre = canvas.cell_rect(1).center()
    event = QWheelEvent(
        centre,
        QPointF(canvas.mapToGlobal(centre.toPoint())),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    canvas.wheelEvent(event)
    assert window.collage.placement(1).zoom > 1.0
    assert window.collage.placement(0).zoom == 1.0
    QTest.mouseDClick(canvas, Qt.MouseButton.LeftButton, pos=centre.toPoint())
    assert window.collage.placement(1) == Placement()


def test_leaving_the_collage_gives_the_toolbar_back_to_the_photo(
    window: MainWindow, tmp_path: Path
) -> None:
    from app.i18n import tr

    window.enter_collage()
    assert window.toolbar.title_label.text() == tr("collage.title")
    window.leave_collage()
    assert window.toolbar.title_label.text() == tr("toolbar.no_photo")
    assert not window.toolbar.export_button.isEnabled()


# -- finer zoom, linking, aligning and the seam (PLAN.md 2.8 K-06, K-10 to K-12)


def _wheel(canvas, point, notches: int, shift: bool = False) -> None:
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    modifiers = Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
    event = QWheelEvent(
        point,
        QPointF(canvas.mapToGlobal(point.toPoint())),
        QPoint(0, 0),
        QPoint(0, 120 * notches),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    canvas.wheelEvent(event)


def _drag(canvas, start, offset) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(canvas, start + offset)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=start + offset)


def test_one_notch_of_the_wheel_zooms_five_percent_and_one_percent_with_shift(
    window: MainWindow, tmp_path: Path
) -> None:
    _filled(window, tmp_path)
    canvas = window.collage_canvas
    centre = canvas.cell_rect(0).center()
    _wheel(canvas, centre, 1)
    assert window.collage.placement(0).zoom == pytest.approx(1.05)
    _wheel(canvas, centre, 1, shift=True)
    assert window.collage.placement(0).zoom == pytest.approx(1.05 * 1.01)


def test_the_zoom_slider_zooms_and_follows_the_wheel(window: MainWindow, tmp_path: Path) -> None:
    _filled(window, tmp_path)
    slider = window.collage_panel.zoom_sliders[1]
    slider.set_value(150.0, announce=True)
    assert window.collage.placement(1).zoom == pytest.approx(1.5)
    _wheel(window.collage_canvas, window.collage_canvas.cell_rect(0).center(), 2)
    assert window.collage_panel.zoom_sliders[0].value == pytest.approx(100 * 1.05**2)


def test_linked_photos_move_together(window: MainWindow, tmp_path: Path) -> None:
    from PySide6.QtCore import QPoint

    _filled(window, tmp_path)
    window.collage_panel.linked_switch.click()
    assert window.collage.layout.linked
    assert not window.collage_panel.zoom_sliders[1].isVisibleTo(window.collage_panel)
    canvas = window.collage_canvas
    _wheel(canvas, canvas.cell_rect(1).center(), 4)
    zoomed_focus = window.collage.placement(0).focus_x
    _drag(canvas, canvas.cell_rect(1).center().toPoint(), QPoint(40, 0))
    assert window.collage.placement(0) == window.collage.placement(1)
    assert window.collage.placement(0).focus_x < zoomed_focus
    first, second = window.collage.rect(0), window.collage.rect(1)
    assert first == pytest.approx(second)


def test_aligning_moves_only_the_second_photo(window: MainWindow, tmp_path: Path) -> None:
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    _filled(window, tmp_path)
    window.collage_panel.linked_switch.click()
    window.collage_panel.aligning_switch.click()
    canvas = window.collage_canvas
    assert canvas.aligning
    before_first = window.collage.rect(0)
    _drag(canvas, canvas.cell_rect(0).center().toPoint(), QPoint(30, 0))
    assert window.collage.rect(0) == pytest.approx(before_first)
    assert window.collage.alignment(1).shift_x > 0

    nudged_from = window.collage.rect(1)
    QTest.keyClick(canvas, Qt.Key.Key_Down)
    assert window.collage.rect(1)[1] == pytest.approx(nudged_from[1] + 1.0)

    # Esc ends aligning first, and only then the collage.
    window.on_escape()
    assert not canvas.aligning and window.in_collage
    window.on_escape()
    assert not window.in_collage


def test_the_seam_can_be_dragged_and_put_back(window: MainWindow, tmp_path: Path) -> None:
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtTest import QTest

    _filled(window, tmp_path)
    canvas = window.collage_canvas
    seam_at = QPointF(canvas.seam_position(), canvas.canvas_rect().center().y()).toPoint()
    canvas_width = canvas.canvas_rect().width()
    _drag(canvas, seam_at, QPoint(round(canvas_width * 0.1), 0))
    assert window.collage.layout.seam == pytest.approx(0.6, abs=0.01)
    assert window.collage_panel.seam_slider.value == pytest.approx(60.0, abs=1.0)
    seam_at = QPointF(canvas.seam_position(), canvas.canvas_rect().center().y()).toPoint()
    QTest.mouseDClick(canvas, Qt.MouseButton.LeftButton, pos=seam_at)
    assert window.collage.layout.seam == collage.SEAM_DEFAULT


def test_the_linked_collage_is_written_as_it_is_shown(window: MainWindow, tmp_path: Path) -> None:
    """Left half of the first photo, right half of the second, aligned."""
    gradient = np.zeros((600, 900, 3), dtype=np.uint8)
    gradient[..., 0] = np.linspace(0, 255, 900, dtype=np.uint8)[None, :]
    Image.fromarray(gradient).save(tmp_path / "night.png")
    Image.fromarray(255 - gradient).save(tmp_path / "day.png")
    window.collage.set_layout(Layout(linked=True))
    window.collage.load(0, tmp_path / "night.png")
    window.collage.load(1, tmp_path / "day.png")
    _wait(lambda: window.collage.complete)
    window.collage.set_alignment(collage.Alignment(shift_x=0.02))
    finished: list[object] = []
    window.collage.export_finished.connect(finished.append)
    assert window.collage.export(tmp_path / "out.png", ExportOptions(format="png"))
    _wait(lambda: bool(finished))
    written = np.asarray(Image.open(finished[0].path).convert("RGB")).astype(int)
    expected = collage.render(
        window.collage.layout,
        [gradient, 255 - gradient],
        [window.collage.placement(0), window.collage.placement(1)],
        [collage.UNALIGNED, collage.Alignment(shift_x=0.02)],
    ).astype(int)
    assert np.abs(written - expected).mean() < 1.5


# -- the carousel (PLAN.md 2.8 K-13) ------------------------------------------


def test_a_carousel_writes_each_photo_on_a_canvas_of_its_own(
    window: MainWindow, tmp_path: Path
) -> None:
    window.collage.set_layout(Layout(gap=10))
    window.collage.load(0, _jpeg(tmp_path / "left.jpg", (220, 30, 30)))
    window.collage.load(1, _jpeg(tmp_path / "right.jpg", (30, 30, 220), (400, 600)))
    _wait(lambda: window.collage.complete)
    finished: list[object] = []
    window.collage.export_finished.connect(finished.append)
    target = tmp_path / "out" / "a.b_collage.jpg"
    assert window.collage.export(target, ExportOptions(), singles=True)
    _wait(lambda: bool(finished))
    result = finished[0]
    assert result.path == target
    assert [path.name for path in result.singles] == [
        "a.b_collage_a.jpg",
        "a.b_collage_b.jpg",
    ]
    for path, channel in zip(result.singles, (0, 2)):
        with Image.open(path) as image:
            assert image.size == (1920, 1080)
            assert image.info.get("icc_profile")
            pixels = np.asarray(image.convert("RGB"))
        # A portrait photo filling a landscape canvas: no background shows.
        assert pixels[540, 960, channel] > 180
        assert pixels[540, 4, channel] > 180

    # A second export's photos sit beside its own collage.
    finished.clear()
    assert window.collage.export(target, ExportOptions(), singles=True)
    _wait(lambda: bool(finished))
    assert finished[0].path.name == "a.b_collage_2.jpg"
    assert [path.name for path in finished[0].singles] == [
        "a.b_collage_2_a.jpg",
        "a.b_collage_2_b.jpg",
    ]


def test_without_a_carousel_only_the_collage_is_written(
    window: MainWindow, tmp_path: Path
) -> None:
    window.collage.load(0, _jpeg(tmp_path / "left.jpg", (220, 30, 30)))
    window.collage.load(1, _jpeg(tmp_path / "right.jpg", (30, 30, 220)))
    _wait(lambda: window.collage.complete)
    finished: list[object] = []
    window.collage.export_finished.connect(finished.append)
    out = tmp_path / "out"
    assert window.collage.export(out / "c.jpg", ExportOptions())
    _wait(lambda: bool(finished))
    assert finished[0].singles == ()
    assert [path.name for path in out.iterdir()] == ["c.jpg"]


def test_the_carousel_switch_is_only_offered_for_a_collage(
    qt_app: object, tmp_path: Path
) -> None:
    settings = Settings().export
    photo = ExportDialog(tmp_path / "a.jpg", settings)
    assert not photo.carousel_switch.isVisibleTo(photo)
    assert not photo.carousel
    collage_dialog = ExportDialog(
        tmp_path / "a.jpg",
        settings,
        collage_name="a_b_collage",
        collage_size=(1920, 1080),
        carousel=True,
    )
    assert collage_dialog.carousel_switch.isVisibleTo(collage_dialog)
    assert collage_dialog.carousel
    collage_dialog.carousel_switch.setChecked(False)
    assert not collage_dialog.carousel


def test_the_carousel_choice_is_remembered() -> None:
    stored = CollageSettings()
    stored.carousel = True
    assert CollageSettings.from_dict(stored.to_dict()).carousel
    assert not CollageSettings.from_dict({"carousel": "yes"}).carousel
