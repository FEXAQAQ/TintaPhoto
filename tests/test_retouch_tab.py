"""Tests for the retouch tab: spot removal, portrait and lens correction (PLAN.md 2.2, 3.1).

The portrait and lens groups used to be at the bottom of the adjust tab; they
have a tab of their own because neither is wanted on every photo
(2026-11-01). Spot removal joined them the same day (A-32).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from core import healing, lens
from core.edit_state import EditState
from core.lens import LensProfile, LensSettings

pytest.importorskip("PySide6.QtWidgets")

from app.i18n import tr  # noqa: E402
from ui.retouch_tab import RetouchTab  # noqa: E402

SHOT = (5234.0, 7.6)


@pytest.fixture
def tab(qt_app: object) -> Iterator[RetouchTab]:
    widget = RetouchTab()
    widget.resize(300, 400)
    widget.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    yield widget
    widget.hide()
    widget.deleteLater()


def test_spot_removal_comes_first_then_the_portrait_then_the_lens(tab: RetouchTab) -> None:
    assert [group.header.text() for group in tab.groups()] == [
        tr("adjust.group.spots"),
        tr("adjust.group.portrait"),
        tr("adjust.group.lens"),
    ]
    assert list(tab.rows()) == ["skin", "whiten"]


# -- spot removal (A-32) -------------------------------------------------------


def test_the_spot_tool_is_a_button_on_the_group_header(tab: RetouchTab) -> None:
    button = tab.spot_button
    assert button.isCheckable()
    assert button.parent() is tab.groups()[0].header
    assert "Q" in button.toolTip()


def test_the_size_slider_counts_thousandths_of_the_long_edge(tab: RetouchTab) -> None:
    seen: list[tuple[float, bool]] = []
    tab.spot_size_set.connect(lambda radius, dragging: seen.append((radius, dragging)))
    assert tab.spot_size == healing.RADIUS_DEFAULT

    tab.spot_size_row.set_value(20.0, announce=True)
    assert seen[-1] == (pytest.approx(0.02), False)
    assert tab.spot_size == pytest.approx(0.02)

    seen.clear()
    tab.set_spot_size(0.5)  # shown, not announced, and held in range
    assert not seen
    assert tab.spot_size == healing.RADIUS_MAX


def test_the_caption_counts_the_spots_and_clearing_waits_for_one(tab: RetouchTab) -> None:
    tab.set_photo(EditState.for_photo(False))
    assert tab.spots_caption.text() == tr("adjust.spots.hint")
    assert not tab.clear_spots_button.isEnabled()

    tab.set_spot_count(1)
    assert tab.spots_caption.text() == tr("adjust.spots.one")
    tab.set_spot_count(3)
    assert tab.spots_caption.text() == tr("adjust.spots.many", count=3)
    assert tab.clear_spots_button.isEnabled()

    cleared: list[bool] = []
    tab.spots_cleared.connect(lambda: cleared.append(True))
    tab.clear_spots_button.click()
    assert cleared == [True]


def test_a_portrait_slider_says_what_changed(tab: RetouchTab) -> None:
    tab.set_photo(EditState.for_photo(False))
    tab.set_faces(1)
    seen: list[tuple[str, float, bool]] = []
    tab.value_set.connect(lambda name, value, dragging: seen.append((name, value, dragging)))

    tab.rows()["skin"].set_value(30.0, announce=True)

    assert seen[-1] == ("skin", pytest.approx(30.0), False)


def test_undo_and_reset_show_the_portrait_without_announcing_it(tab: RetouchTab) -> None:
    tab.set_photo(EditState.for_photo(False))
    seen: list[object] = []
    tab.value_set.connect(lambda *args: seen.append(args))

    tab.set_state(EditState.for_photo(False).with_value("whiten", 25.0))

    assert tab.rows()["whiten"].value == pytest.approx(25.0)
    assert not seen


def test_without_a_photo_the_groups_are_hidden(tab: RetouchTab) -> None:
    tab.set_photo(None)
    assert all(group.isHidden() or not group.isVisibleTo(tab) for group in tab.groups())


# -- the lens correction group (PLAN.md 2.2, A-26 to A-28) -----------------


def _profile(distortion: bool = True, vignetting: bool = True) -> LensProfile:
    return LensProfile(
        name="Test 24-70mm",
        distortion=("ptlens", (0.0, -0.01, 0.0)) if distortion else None,
        tca=None,
        vignetting=("pa", (-0.3, 0.0, 0.0)) if vignetting else None,
        distortion_unit_mm=12.0,
        vignetting_unit_mm=21.6,
        sensor_diagonal_mm=43.3,
    )


def test_a_raw_with_a_known_lens_can_be_corrected(tab: RetouchTab) -> None:
    tab.set_photo(EditState.for_photo(True, *SHOT))
    tab.set_lens_profile(_profile(), True)
    tab.set_lens(LensSettings.for_photo(True))

    assert tab.lens_switch.isEnabled()
    assert tab.lens_switch.isChecked()
    assert tab.lens_caption.text() == "Test 24-70mm"
    assert all(row.isEnabled() for row in tab.lens_rows().values())


@pytest.mark.parametrize(("name", "key"), [("Zoom 15-45mm", "adjust.lens.camera"), ("", "adjust.lens.camera_unnamed")])
def test_the_camera_s_own_data_is_named_as_such(tab: RetouchTab, name: str, key: str) -> None:
    tab.set_photo(EditState.for_photo(True, *SHOT))
    tab.set_lens_profile(lens.camera_profile(name, ((0.5, 1.0), (1.0, 4.0)), None, 0.98), True)
    tab.set_lens(LensSettings.for_photo(True))

    assert tab.lens_switch.isChecked()
    assert tab.lens_caption.text() == tr(key, lens=name)
    if name:
        assert name in tab.lens_caption.text()
    rows = tab.lens_rows()
    assert rows["distortion"].isEnabled()
    assert not rows["vignetting"].isEnabled()


def test_a_jpeg_says_the_camera_corrected_it(tab: RetouchTab) -> None:
    tab.set_photo(EditState.for_photo(False))
    tab.set_lens_profile(None, False)
    tab.set_lens(LensSettings.for_photo(False))

    assert not tab.lens_switch.isEnabled()
    assert not tab.lens_switch.isChecked()
    assert tab.lens_caption.text() == tr("adjust.lens.jpeg")
    assert not any(row.isEnabled() for row in tab.lens_rows().values())


def test_an_unknown_lens_says_so_and_cannot_be_switched_on(tab: RetouchTab) -> None:
    tab.set_photo(EditState.for_photo(True, *SHOT))
    tab.set_lens_profile(None, True)
    tab.set_lens(LensSettings.for_photo(True))

    assert not tab.lens_switch.isEnabled()
    assert tab.lens_caption.text() == tr("adjust.lens.unknown")


def test_switching_the_lens_off_announces_it_and_greys_the_sliders(tab: RetouchTab) -> None:
    tab.set_lens_profile(_profile(), True)
    tab.set_lens(LensSettings.for_photo(True))
    sent: list[tuple[LensSettings, bool]] = []
    tab.lens_set.connect(lambda settings, dragging: sent.append((settings, dragging)))

    tab.lens_switch.click()

    assert sent == [(LensSettings(enabled=False), False)]
    assert not any(row.isEnabled() for row in tab.lens_rows().values())


def test_a_lens_slider_sends_the_whole_lens_setting(tab: RetouchTab) -> None:
    tab.set_lens_profile(_profile(), True)
    tab.set_lens(LensSettings(enabled=True, distortion=100.0, vignetting=100.0))
    sent: list[tuple[LensSettings, bool]] = []
    tab.lens_set.connect(lambda settings, dragging: sent.append((settings, dragging)))

    tab.lens_rows()["vignetting"].set_value(40.0, announce=True)

    assert sent[-1][0] == LensSettings(enabled=True, distortion=100.0, vignetting=40.0)


def test_a_profile_without_vignetting_greys_only_that_slider(tab: RetouchTab) -> None:
    tab.set_lens_profile(_profile(vignetting=False), True)
    tab.set_lens(LensSettings.for_photo(True))

    assert tab.lens_rows()["distortion"].isEnabled()
    assert not tab.lens_rows()["vignetting"].isEnabled()


# -- skin enhancement (PLAN.md 2.2, A-29) ------------------------------------


def test_the_portrait_group_says_how_many_faces_it_found(tab: RetouchTab) -> None:
    tab.set_photo(EditState.for_photo(False))
    skin = tab.rows()["skin"]

    tab.set_faces(None)
    assert tab.faces_caption.text() == tr("adjust.faces.looking")
    assert skin.isEnabled()

    tab.set_faces(2)
    assert tab.faces_caption.text() == tr("adjust.faces.many", count=2)
    assert skin.isEnabled()

    tab.set_faces(1)
    assert tab.faces_caption.text() == tr("adjust.faces.one")


def test_the_skin_slider_is_greyed_out_without_faces(tab: RetouchTab) -> None:
    tab.set_photo(EditState.for_photo(True, *SHOT))
    tab.set_faces(0)
    assert tab.faces_caption.text() == tr("adjust.faces.none")
    assert not tab.rows()["skin"].isEnabled()


def test_the_whitening_slider_sits_under_skin_and_is_greyed_out_with_it(tab: RetouchTab) -> None:
    tab.set_photo(EditState.for_photo(False))
    rows = tab.rows()
    assert rows["whiten"].y() > rows["skin"].y()
    tab.set_faces(0)
    assert not rows["whiten"].isEnabled()
    tab.set_faces(1)
    assert rows["whiten"].isEnabled()
