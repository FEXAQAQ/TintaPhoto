"""The right hand panel while the watermark is open (PLAN.md 3.14).

Takes the inspector's place, at the same width and in the same style, as the
collage's panel does: a bar naming the mode, the switch that decides whether
exports carry the watermark, a group for the text and how it looks, a group
for where it goes, and an action row along the bottom.

The panel only says what was chosen: the window keeps the watermark in the
settings, and the photo's overlay draws it.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget

from app.i18n import tr
from core import watermark
from core.watermark import WatermarkSpec
from ui import theme
from ui.widgets.anchor_picker import AnchorPicker
from ui.widgets.buttons import ROLE_CHOICE, TextButton, tooltip
from ui.widgets.focus import ring_on_keyboard_only
from ui.widgets.group import ControlGroup
from ui.widgets.region import styled_background
from ui.widgets.scroll import scrolled
from ui.widgets.segmented import Segmented
from ui.widgets.slider_row import SliderRow, SliderSpec
from ui.widgets.switch import Switch
from ui.widgets.text import TranslatedLabel


def size_spec() -> SliderSpec:
    """Return the size slider: a line's height in per cent of the short edge (W-05)."""
    return SliderSpec(
        name="size",
        label_key="watermark.size",
        minimum=watermark.SIZE_MIN,
        maximum=watermark.SIZE_MAX,
        default=watermark.SIZE_DEFAULT,
        decimals=watermark.SIZE_DECIMALS,
        signed=False,
        fill_from=watermark.SIZE_MIN,
    )


def opacity_spec() -> SliderSpec:
    """Return the opacity slider (W-06)."""
    return SliderSpec(
        name="opacity",
        label_key="watermark.opacity",
        minimum=float(watermark.OPACITY_MIN),
        maximum=float(watermark.OPACITY_MAX),
        default=float(watermark.OPACITY_DEFAULT),
        signed=False,
        fill_from=float(watermark.OPACITY_MIN),
    )


def margin_spec() -> SliderSpec:
    """Return the margin slider (W-07)."""
    return SliderSpec(
        name="margin",
        label_key="watermark.margin",
        minimum=float(watermark.MARGIN_MIN),
        maximum=float(watermark.MARGIN_MAX),
        default=float(watermark.MARGIN_DEFAULT),
        signed=False,
        fill_from=float(watermark.MARGIN_MIN),
    )


class StyleToggle(QPushButton):
    """A 28 px square that turns bold or italic on and off (W-04).

    Its letter is set the way it sets the text, so B is bold and I leans.
    """

    def __init__(self, mark_key: str, label_key: str, style: str, parent: QWidget) -> None:
        super().__init__(tr(mark_key), parent)
        self._mark_key = mark_key
        self._label_key = label_key
        self.setProperty("role", ROLE_CHOICE)
        self.setCheckable(True)
        self.setFixedSize(theme.BUTTON_HEIGHT, theme.BUTTON_HEIGHT)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setStyleSheet(style)
        self.setToolTip(tooltip(label_key))

    def retranslate(self) -> None:
        """Read the letter and the tooltip again after a language change."""
        self.setText(tr(self._mark_key))
        self.setToolTip(tooltip(self._label_key))


class WatermarkPanel(QWidget):
    """The watermark's choices, in the inspector's place."""

    #: The watermark the controls now describe.
    spec_changed = Signal(object)
    #: Whether exported photos carry the watermark (W-09).
    enabled_changed = Signal(bool)
    #: Back to editing.
    leave_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        styled_background(self, "WatermarkPanel")
        self.setFixedWidth(theme.PANEL_WIDTH)
        self._spec = WatermarkSpec()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_title())

        content = QWidget(self)
        body = QVBoxLayout(content)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_switch(content))
        body.addWidget(self._build_text_group(content))
        body.addWidget(self._build_position_group(content))
        body.addStretch(1)
        outer.addWidget(scrolled(content, "TabPage", self), 1)
        outer.addWidget(self._build_action_row())

        self.set_state(False, self._spec)

    # -- building --------------------------------------------------------

    def _build_title(self) -> QWidget:
        """The bar along the top, where the tab strip is when editing."""
        bar = QWidget(self)
        styled_background(bar, "TabBar")
        bar.setFixedHeight(theme.TAB_BAR_HEIGHT)
        row = QHBoxLayout(bar)
        row.setContentsMargins(theme.PANEL_PADDING, 0, theme.PANEL_PADDING, 0)
        row.addWidget(TranslatedLabel("watermark.title", bar))
        row.addStretch(1)
        return bar

    def _build_switch(self, parent: QWidget) -> QWidget:
        """Whether exports carry it: first, since it decides whether the rest counts."""
        holder = QWidget(parent)
        row = QHBoxLayout(holder)
        row.setContentsMargins(
            theme.PANEL_PADDING, theme.SPACE_MD, theme.PANEL_PADDING, theme.SPACE_MD
        )
        row.setSpacing(theme.SPACE_SM)
        row.addWidget(TranslatedLabel("watermark.enabled", holder), 1)
        self.enabled_switch = Switch(holder)
        self.enabled_switch.toggled.connect(self.enabled_changed.emit)
        row.addWidget(self.enabled_switch)
        return holder

    def _labelled(self, group: ControlGroup, label_key: str) -> QHBoxLayout:
        """Start a row whose name sits in the fixed column on the left."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACE_SM)
        label = TranslatedLabel(label_key, group)
        label.setProperty("role", "secondary")
        label.setFixedWidth(theme.DIALOG_LABEL_WIDTH)
        row.addWidget(label)
        group.add_layout(row)
        return row

    def _build_text_group(self, parent: QWidget) -> ControlGroup:
        """The text, and how it looks (W-03 to W-06)."""
        group = ControlGroup("watermark.group.text", parent=parent)
        self.text_field = QLineEdit(group)
        ring_on_keyboard_only(self.text_field)
        self.text_field.setFixedHeight(theme.BUTTON_HEIGHT)
        self.text_field.setMaxLength(watermark.TEXT_MAX_LENGTH)
        self.text_field.setPlaceholderText(tr("watermark.text_placeholder"))
        self.text_field.textEdited.connect(lambda _text: self._changed())
        group.add_row(self.text_field)

        style = self._labelled(group, "watermark.style")
        self.bold_button = StyleToggle(
            "watermark.bold_mark",
            "watermark.bold",
            f"font-weight: {theme.FONT_WEIGHT_TITLE};",
            group,
        )
        self.italic_button = StyleToggle(
            "watermark.italic_mark", "watermark.italic", "font-style: italic;", group
        )
        for button in (self.bold_button, self.italic_button):
            button.clicked.connect(lambda _checked: self._changed())
            style.addWidget(button)
        style.addStretch(1)

        self.colour_choice = Segmented(
            [
                (watermark.COLOUR_WHITE, "watermark.colour.white"),
                (watermark.COLOUR_BLACK, "watermark.colour.black"),
            ],
            group,
        )
        self.colour_choice.chosen.connect(lambda _value: self._changed())
        self._labelled(group, "watermark.colour").addWidget(self.colour_choice, 1)

        self.size_slider = SliderRow(size_spec(), group)
        self.size_slider.value_changed.connect(lambda *_: self._changed())
        group.add_row(self.size_slider)
        self.opacity_slider = SliderRow(opacity_spec(), group)
        self.opacity_slider.value_changed.connect(lambda *_: self._changed())
        group.add_row(self.opacity_slider)

        self.missing_font_label = TranslatedLabel("watermark.no_cjk_font", group)
        self.missing_font_label.setProperty("role", "caption")
        self.missing_font_label.setWordWrap(True)
        self.missing_font_label.hide()
        group.add_row(self.missing_font_label)
        return group

    def _build_position_group(self, parent: QWidget) -> ControlGroup:
        """Where it goes: the nine places, and how far in from the edges (W-07)."""
        group = ControlGroup("watermark.group.position", parent=parent)
        row = QHBoxLayout()
        row.setContentsMargins(0, theme.SPACE_XS, 0, theme.SPACE_XS)
        row.setSpacing(theme.SPACE_MD)
        self.anchor_picker = AnchorPicker(group)
        self.anchor_picker.chosen.connect(self._on_anchor)
        row.addWidget(self.anchor_picker, 0, Qt.AlignmentFlag.AlignTop)
        hint = TranslatedLabel("watermark.hint", group)
        hint.setProperty("role", "caption")
        hint.setWordWrap(True)
        row.addWidget(hint, 1, Qt.AlignmentFlag.AlignTop)
        group.add_layout(row)

        self.margin_slider = SliderRow(margin_spec(), group)
        self.margin_slider.value_changed.connect(lambda *_: self._changed())
        group.add_row(self.margin_slider)
        return group

    def _build_action_row(self) -> QWidget:
        """The row along the bottom, as on every tab of the inspector."""
        row = QWidget(self)
        styled_background(row, "PanelActionRow")
        row.setFixedHeight(theme.PANEL_ACTION_ROW_HEIGHT)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(
            theme.PANEL_PADDING, theme.SPACE_SM, theme.PANEL_PADDING, theme.SPACE_SM
        )
        layout.setSpacing(theme.SPACE_SM)
        self.reset_button = TextButton("panel.action.reset", row)
        self.reset_button.clicked.connect(self._reset)
        layout.addWidget(self.reset_button, 1)
        self.leave_button = TextButton("watermark.leave", row)
        self.leave_button.setToolTip(tooltip("watermark.leave", "shortcut.watermark"))
        self.leave_button.clicked.connect(self.leave_requested.emit)
        layout.addWidget(self.leave_button, 1)
        return row

    # -- being told things -----------------------------------------------

    @property
    def spec(self) -> WatermarkSpec:
        """Return the watermark the controls describe."""
        return self._spec

    @property
    def enabled(self) -> bool:
        """Return whether the switch says exports carry the watermark."""
        return self.enabled_switch.isChecked()

    def set_state(self, enabled: bool, spec: WatermarkSpec) -> None:
        """Show *spec* and the switch, without announcing either."""
        spec = spec.sane()
        self._spec = spec
        self.enabled_switch.blockSignals(True)
        self.enabled_switch.setChecked(enabled)
        self.enabled_switch.blockSignals(False)
        if self.text_field.text() != spec.text:
            self.text_field.setText(spec.text)
        self.bold_button.setChecked(spec.bold)
        self.italic_button.setChecked(spec.italic)
        self.colour_choice.set_value(spec.colour)
        self.size_slider.set_value(spec.size)
        self.opacity_slider.set_value(float(spec.opacity))
        self.margin_slider.set_value(float(spec.margin))
        self.anchor_picker.set_anchor(watermark.anchor_of(spec))

    def set_spec(self, spec: WatermarkSpec) -> None:
        """Show *spec*, moved on the photo, without announcing it."""
        self.set_state(self.enabled, spec)

    def set_missing_font(self, missing: bool) -> None:
        """Say that the Chinese in the text cannot be drawn on this computer."""
        self.missing_font_label.setVisible(missing)

    def retranslate(self) -> None:
        """Read the placeholder and the leave button's tooltip again (U-02)."""
        self.text_field.setPlaceholderText(tr("watermark.text_placeholder"))
        self.leave_button.setToolTip(tooltip("watermark.leave", "shortcut.watermark"))

    # -- reacting --------------------------------------------------------

    def _changed(self) -> None:
        """Take on what the controls say and announce it."""
        self._spec = WatermarkSpec(
            kind=self._spec.kind,
            text=self.text_field.text(),
            bold=self.bold_button.isChecked(),
            italic=self.italic_button.isChecked(),
            colour=self.colour_choice.value,
            opacity=round(self.opacity_slider.value),
            size=round(self.size_slider.value, watermark.SIZE_DECIMALS),
            anchor=self._spec.anchor,
            offset=self._spec.offset,
            margin=round(self.margin_slider.value),
        ).sane()
        self.spec_changed.emit(self._spec)

    def _on_anchor(self, anchor: str) -> None:
        """Put the watermark exactly on the place chosen in the picker."""
        self._spec = self._spec.with_anchor(anchor)
        self.spec_changed.emit(self._spec)

    def _reset(self) -> None:
        """Keep the text and put everything else back (W-11)."""
        self.set_state(self.enabled, self._spec.with_style_reset())
        self.spec_changed.emit(self._spec)
