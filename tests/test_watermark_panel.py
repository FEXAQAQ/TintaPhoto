"""Tests for the watermark's panel (PLAN.md 3.14)."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6.QtWidgets")

from core import watermark  # noqa: E402
from core.watermark import WatermarkSpec  # noqa: E402
from ui import theme  # noqa: E402
from ui.watermark_panel import WatermarkPanel  # noqa: E402


@pytest.fixture
def panel(qt_app: object) -> WatermarkPanel:
    return WatermarkPanel()


def _changes(panel: WatermarkPanel) -> list[WatermarkSpec]:
    seen: list[WatermarkSpec] = []
    panel.spec_changed.connect(seen.append)
    return seen


def test_it_is_as_wide_as_the_inspector(panel: WatermarkPanel) -> None:
    assert panel.width() == theme.PANEL_WIDTH


def test_it_shows_the_watermark_it_is_given(panel: WatermarkPanel) -> None:
    spec = WatermarkSpec(
        text="© me", bold=True, italic=True, colour="black", opacity=40,
        size=6.5, anchor="top", margin=9,
    )
    seen = _changes(panel)
    panel.set_state(True, spec)
    assert seen == []
    assert panel.enabled
    assert panel.text_field.text() == "© me"
    assert panel.bold_button.isChecked() and panel.italic_button.isChecked()
    assert panel.colour_choice.value == "black"
    assert panel.size_slider.value == pytest.approx(6.5)
    assert panel.opacity_slider.value == pytest.approx(40)
    assert panel.margin_slider.value == pytest.approx(9)
    assert panel.anchor_picker.anchor == "top"
    assert panel.spec == spec


def test_typing_says_what_the_watermark_is_now(panel: WatermarkPanel) -> None:
    seen = _changes(panel)
    panel.text_field.setText("© 2026")
    panel.text_field.textEdited.emit("© 2026")
    panel.bold_button.click()
    assert [spec.text for spec in seen] == ["© 2026", "© 2026"]
    assert seen[-1].bold


def test_choosing_a_place_clears_the_drag(panel: WatermarkPanel) -> None:
    panel.set_state(False, WatermarkSpec(text="x", anchor="top", offset=(0.1, 0.0)))
    assert panel.anchor_picker.anchor is None
    seen = _changes(panel)
    panel.anchor_picker.chosen.emit("left")
    assert seen[-1].anchor == "left" and seen[-1].offset == (0.0, 0.0)


def test_reset_keeps_only_the_text(panel: WatermarkPanel) -> None:
    panel.set_state(True, WatermarkSpec(text="me", bold=True, size=9.0, anchor="top"))
    seen = _changes(panel)
    panel.reset_button.click()
    assert seen[-1] == WatermarkSpec(text="me")
    assert not panel.bold_button.isChecked()
    assert panel.anchor_picker.anchor == watermark.ANCHOR_DEFAULT


def test_the_switch_is_announced_on_its_own(panel: WatermarkPanel) -> None:
    switched: list[bool] = []
    panel.enabled_changed.connect(switched.append)
    panel.enabled_switch.click()
    assert switched == [True]


def test_a_missing_chinese_font_is_mentioned(panel: WatermarkPanel) -> None:
    panel.show()
    try:
        panel.set_missing_font(True)
        assert panel.missing_font_label.isVisible()
        panel.set_missing_font(False)
        assert not panel.missing_font_label.isVisible()
    finally:
        panel.hide()
