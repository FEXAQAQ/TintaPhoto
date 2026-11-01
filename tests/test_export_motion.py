"""Tests for the motion around an export (PLAN.md 3.6 and 3.11).

The dim and the dialog fade in and out, the photo flies into the export
button, the button fills smoothly while the file is written and then says
it is done, and the message under the photo rises into place. With Windows
set to show no animation every one of these changes at once.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QEventLoop, QPointF, QRectF, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QColor, QPixmap  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

from app.i18n import tr  # noqa: E402
from ui import native, theme  # noqa: E402
from ui.widgets.buttons import ExportLook, ProgressButton  # noqa: E402
from ui.widgets.dialog import Dialog  # noqa: E402
from ui.widgets.flight import FlyingPhoto  # noqa: E402
from ui.widgets.region import Scrim  # noqa: E402
from ui.widgets.toast import Toast  # noqa: E402

FRAME_MS = 16


def _wait(predicate, timeout_ms: int = 4000) -> bool:
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        loop = QEventLoop()
        QTimer.singleShot(5, loop.quit)
        loop.exec()
        if predicate():
            return True
    return predicate()


def _run(button: ProgressButton, ms: int) -> None:
    """Move *button* on by *ms*, a frame at a time."""
    for _ in range(ms // FRAME_MS):
        button.advance(FRAME_MS)


@pytest.fixture
def animated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(native, "animations_enabled", lambda: True)


@pytest.fixture
def still(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(native, "animations_enabled", lambda: False)


@pytest.fixture
def button(qt_app: object) -> Iterator[ProgressButton]:
    made = ProgressButton("toolbar.export", "shortcut.export")
    yield made
    made.rest()
    made.deleteLater()


def _stop_frames(button: ProgressButton) -> None:
    button._frames.stop()


# -- the export button ---------------------------------------------------


def test_the_button_waits_for_the_photo_to_land_then_flashes(
    animated: None, button: ProgressButton
) -> None:
    button.start(theme.EXPORT_FLY_MS)
    _stop_frames(button)
    assert button.look is ExportLook.LANDING
    assert button.text() == tr("toolbar.export"), "still its name while the photo flies"

    button.advance(theme.EXPORT_FLY_MS - theme.EXPORT_PULSE_MS // 2)
    assert button.pulse() == pytest.approx(1.0)
    button.advance(theme.EXPORT_PULSE_MS // 2)
    assert button.look is ExportLook.WORKING
    assert button.text() == tr("toolbar.exporting")


def test_the_button_widens_to_say_it_is_exporting(
    animated: None, button: ProgressButton
) -> None:
    rest = button.sizeHint().width()
    button.start()
    _stop_frames(button)
    _run(button, theme.DURATION_PANEL_MS * 2)
    assert button.width() > rest
    assert button.width() <= button.widest_width()


def test_the_fill_creeps_between_steps_but_never_past_the_next(
    animated: None, button: ProgressButton
) -> None:
    """The exporter's steps are 0.05, 0.45, 0.85 and 1 (core/exporter.py)."""
    button.start()
    _stop_frames(button)
    button.set_progress(0.05)
    _run(button, 400)
    assert button.shown > 0.05, "it keeps moving between steps"
    _run(button, 20_000)
    assert button.shown < 0.45, "it never reaches a step that has not come"

    button.set_progress(0.45)
    _run(button, 20_000)
    assert 0.45 <= button.shown < 0.85


def test_the_fill_never_goes_back(animated: None, button: ProgressButton) -> None:
    button.start()
    _stop_frames(button)
    button.set_progress(0.45)
    _run(button, 1000)
    before = button.shown

    button.set_progress(0.05)
    assert button.progress == pytest.approx(0.45)
    _run(button, 1000)
    assert button.shown >= before


def test_a_finished_export_shows_a_tick_then_the_name_again(
    animated: None, button: ProgressButton
) -> None:
    button.start()
    _stop_frames(button)
    button.set_progress(0.45)
    button.finish()
    for _ in range(100):
        if button.look is ExportLook.DONE:
            break
        button.advance(FRAME_MS)
    assert button.look is ExportLook.DONE, "the fill reaches the end quickly"
    assert button.text() == tr("toolbar.exported")

    _run(button, theme.EXPORT_DONE_HOLD_MS + FRAME_MS * 2)
    assert 0.0 < button.done_fade() < 1.0, "the words cross-fade back"
    _run(button, theme.EXPORT_DONE_FADE_MS + FRAME_MS)
    assert button.look is ExportLook.REST
    assert button.text() == tr("toolbar.export")
    assert button.progress is None


def test_the_fill_turns_the_words_dark_where_it_has_passed(
    animated: None, button: ProgressButton
) -> None:
    button.resize(120, theme.BUTTON_HEIGHT)
    button.start()
    _stop_frames(button)
    button.set_progress(0.45)
    _run(button, 1000)
    picture = button.grab().toImage()
    left = QColor(picture.pixel(2, picture.height() // 2))
    right = QColor(picture.pixel(picture.width() - 3, picture.height() // 2))
    assert left.lightness() > 200, "filled with the light of the button"
    assert right.lightness() < 80, "the rest is the dark of the track"


def test_without_animation_the_button_changes_at_once(
    still: None, button: ProgressButton
) -> None:
    button.start(theme.EXPORT_FLY_MS)
    assert button.look is ExportLook.WORKING, "nothing flies, so nothing lands"
    button.set_progress(0.45)
    assert button.shown == pytest.approx(0.45), "no creep"

    button.finish()
    assert button.look is ExportLook.DONE
    assert button._hold.isActive(), "the tick still gives way to the name"
    button._hold.timeout.emit()
    assert button.look is ExportLook.REST


# -- the flight ---------------------------------------------------------------


@pytest.fixture
def flight(qt_app: object) -> Iterator[FlyingPhoto]:
    host = QWidget()
    host.resize(1280, 800)
    made = FlyingPhoto(host)
    yield made
    made.stop()
    host.deleteLater()


def _picture() -> QPixmap:
    picture = QPixmap(60, 40)
    picture.fill(QColor("#808080"))
    return picture


START = QRectF(80, 70, 820, 546)
TARGET = QPointF(1240, 22)


def test_the_photo_shrinks_into_the_button_along_a_curve(
    animated: None, flight: FlyingPhoto
) -> None:
    flight.launch(_picture(), START, TARGET)
    assert flight.rect_at(0.0) == START
    end = flight.rect_at(1.0)
    assert end.center().x() == pytest.approx(TARGET.x())
    assert end.center().y() == pytest.approx(TARGET.y())
    assert end.height() == pytest.approx(theme.EXPORT_FLY_END_HEIGHT)
    assert end.width() / end.height() == pytest.approx(START.width() / START.height())
    # It shrinks as it goes: a quarter of the way it is still clearly the
    # photo, half way it is less than half its size.
    assert flight.rect_at(0.25).height() > START.height() * 0.5
    middle = flight.rect_at(0.5)
    assert middle.height() < START.height() * 0.5
    assert middle.center().y() < START.center().y(), "it rises on its way"


def test_the_photo_fades_only_at_the_end(animated: None, flight: FlyingPhoto) -> None:
    flight.launch(_picture(), START, TARGET)
    assert flight.opacity_at(0.5) == 1.0
    assert flight.opacity_at(1.0 - theme.EXPORT_FLY_FADE_SHARE) == pytest.approx(1.0)
    assert flight.opacity_at(1.0) == pytest.approx(0.0)


def test_the_flight_never_takes_the_pointer_and_lands_on_time(
    animated: None, flight: FlyingPhoto
) -> None:
    flight.launch(_picture(), START, TARGET)
    assert flight.flying
    assert flight.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert 0 < flight.remaining_ms() <= theme.EXPORT_FLY_MS
    assert _wait(lambda: not flight.flying)
    assert flight.isHidden()
    assert flight.remaining_ms() == 0


def test_without_animation_nothing_flies(still: None, flight: FlyingPhoto) -> None:
    flight.launch(_picture(), START, TARGET)
    assert not flight.flying
    assert flight.isHidden()


# -- the dim, the dialog and the message --------------------------------------


@pytest.fixture
def host(qt_app: object) -> Iterator[QWidget]:
    made = QWidget()
    made.resize(800, 600)
    made.show()
    yield made
    made.close()
    made.deleteLater()


def test_the_dim_fades_in_and_out(animated: None, host: QWidget) -> None:
    scrim = Scrim(host)
    scrim.cover(host)
    assert scrim.level < 1.0, "it comes in over a fade"
    assert _wait(lambda: scrim.level == 1.0)

    scrim.uncover()
    assert scrim.isVisible(), "it goes over a fade"
    assert _wait(scrim.isHidden)
    assert scrim.level == 0.0


def test_without_animation_the_dim_simply_comes_and_goes(still: None, host: QWidget) -> None:
    scrim = Scrim(host)
    scrim.cover(host)
    assert scrim.level == 1.0
    scrim.uncover()
    assert scrim.isHidden()


def test_a_dialog_drops_in_and_lifts_away(animated: None, host: QWidget) -> None:
    dialog = Dialog(host)
    dialog.resize(400, 300)
    said: list[bool] = []
    dialog.closing.connect(said.append)
    dialog.show()
    assert dialog.windowOpacity() < 1.0
    assert _wait(lambda: dialog.windowOpacity() == 1.0)
    home = dialog.pos()

    dialog.accept()
    assert said == [True], "it says it is going before it fades"
    assert dialog.isVisible() and dialog.leaving
    assert _wait(lambda: dialog.windowOpacity() < 1.0 and dialog.pos().y() < home.y())
    assert _wait(dialog.isHidden)
    assert dialog.result() == Dialog.DialogCode.Accepted


def test_a_dialog_closed_twice_closes_once(animated: None, host: QWidget) -> None:
    dialog = Dialog(host)
    said: list[bool] = []
    dialog.closing.connect(said.append)
    dialog.show()
    dialog.accept()
    dialog.reject()
    assert said == [True]
    assert _wait(dialog.isHidden)
    assert dialog.result() == Dialog.DialogCode.Accepted


def test_without_animation_a_dialog_simply_goes(still: None, host: QWidget) -> None:
    dialog = Dialog(host)
    dialog.show()
    assert dialog.windowOpacity() == 1.0
    dialog.reject()
    assert dialog.isHidden()


def test_the_message_rises_into_place_and_sinks_away(animated: None, host: QWidget) -> None:
    toast = Toast(host)
    toast.show_message("Saved")
    assert toast.presence < 1.0
    assert _wait(lambda: toast.presence == 1.0)
    home = toast.pos()
    assert home.y() == host.height() - toast.height() - theme.SPACE_XXL

    toast._leave()
    assert _wait(lambda: toast.pos().y() > home.y())
    assert _wait(toast.isHidden)


def test_without_animation_the_message_is_simply_there(still: None, host: QWidget) -> None:
    toast = Toast(host)
    toast.show_message("Saved")
    assert toast.presence == 1.0
    toast._leave()
    assert toast.isHidden()
