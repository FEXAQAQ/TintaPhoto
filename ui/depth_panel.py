"""The right hand panel while the depth of field is open (PLAN.md 2.13, 3.18).

Takes the inspector's place, at the same width and in the same style, as the
portrait's panel does: a bar naming the mode, the blur and how out of focus
light looks, the range that stays sharp - its shape, how gradually the blur
grows beyond it, and the brush that keeps a place sharp or blurs it - and an
action row along the bottom.

The panel only says what was chosen: the controller keeps the depth of field
with the photo, and the photo's overlay draws the shape and the brush.
"""

from __future__ import annotations

from typing import Final

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.i18n import tr
from core import depth_of_field, skin_range
from core.depth_of_field import DEFAULT_DEPTH, DepthOfField
from ui import theme
from ui.portrait_panel import BRUSH_SIZE_SCALE, BrushChoice
from ui.widgets.buttons import TextButton, tooltip
from ui.widgets.group import ControlGroup
from ui.widgets.region import styled_background
from ui.widgets.scroll import scrolled
from ui.widgets.segmented import Segmented
from ui.widgets.slider_row import SliderRow, SliderSpec
from ui.widgets.switch import Switch
from ui.widgets.text import TranslatedLabel

#: The brush's two jobs: keep what it covers sharp, or blur it.
BRUSH_SHARP: Final[str] = "sharp"
BRUSH_BLUR: Final[str] = "blur"

#: The three sliders the depth of field holds, by name.
SPECS: Final[dict[str, SliderSpec]] = {
    "amount": SliderSpec(
        name="amount",
        label_key="depth.amount",
        minimum=depth_of_field.AMOUNT_MIN,
        maximum=depth_of_field.AMOUNT_MAX,
        default=0.0,
        signed=False,
    ),
    "highlights": SliderSpec(
        name="highlights",
        label_key="depth.highlights",
        minimum=depth_of_field.HIGHLIGHTS_MIN,
        maximum=depth_of_field.HIGHLIGHTS_MAX,
        default=depth_of_field.HIGHLIGHTS_DEFAULT,
        signed=False,
        fill_from=depth_of_field.HIGHLIGHTS_MIN,
    ),
    "transition": SliderSpec(
        name="transition",
        label_key="depth.transition",
        minimum=depth_of_field.TRANSITION_MIN,
        maximum=depth_of_field.TRANSITION_MAX,
        default=depth_of_field.TRANSITION_DEFAULT,
        signed=False,
        fill_from=depth_of_field.TRANSITION_MIN,
    ),
}

#: The brush slider counts thousandths of the long edge, as the portrait's does.
BRUSH_SIZE_SPEC: Final[SliderSpec] = SliderSpec(
    name="brush_size",
    label_key="depth.brush_size",
    minimum=skin_range.RADIUS_MIN * BRUSH_SIZE_SCALE,
    maximum=skin_range.RADIUS_MAX * BRUSH_SIZE_SCALE,
    default=skin_range.RADIUS_DEFAULT * BRUSH_SIZE_SCALE,
    signed=False,
    fill_from=skin_range.RADIUS_MIN * BRUSH_SIZE_SCALE,
)


class DepthPanel(QWidget):
    """The depth of field's blur and the range that stays sharp, in the inspector's place."""

    #: One slider moved: amount, highlights or transition, its value, and
    #: whether a drag is still going on.
    value_set = Signal(str, float, bool)
    #: The sharp shape was chosen: an ellipse or a band.
    shape_chosen = Signal(str)
    #: The shape of out of focus light was chosen: a circle or a hexagon.
    bokeh_chosen = Signal(str)
    #: The brush was taken up, put down or given the other job.
    brush_changed = Signal()
    #: The brush size changed: the radius as a fraction of the long edge,
    #: and whether a drag is still going on.
    brush_size_set = Signal(float, bool)
    #: Whether the range is shown over the photo.
    showing_changed = Signal(bool)
    #: The sharp shape back where it starts, every stroke gone.
    reset_requested = Signal()
    #: Back to editing.
    leave_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        styled_background(self, "DepthPanel")
        self.setFixedWidth(theme.PANEL_WIDTH)
        self._depth = DEFAULT_DEPTH
        self._rows: dict[str, SliderRow] = {}
        self._groups: list[ControlGroup] = []
        self._labels: list[QWidget] = []
        self._brush_size = skin_range.RADIUS_DEFAULT

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_title())
        content = QWidget(self)
        body = QVBoxLayout(content)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addSpacing(theme.SPACE_SM)
        body.addWidget(self._build_effect_group(content))
        body.addWidget(self._build_range_group(content))
        body.addStretch(1)
        outer.addWidget(scrolled(content, "TabPage", self), 1)
        outer.addWidget(self._build_action_row())

        self.set_brush_size(self._brush_size)
        self._show_depth()

    # -- building --------------------------------------------------------

    def _build_title(self) -> QWidget:
        """The bar along the top, where the tab strip is when editing."""
        bar = QWidget(self)
        styled_background(bar, "TabBar")
        bar.setFixedHeight(theme.TAB_BAR_HEIGHT)
        row = QHBoxLayout(bar)
        row.setContentsMargins(theme.PANEL_PADDING, 0, theme.PANEL_PADDING, 0)
        row.addWidget(TranslatedLabel("depth.title", bar))
        row.addStretch(1)
        return bar

    def _slider(self, group: ControlGroup, name: str) -> SliderRow:
        """Add the slider *name* to *group*."""
        row = SliderRow(SPECS[name], group)
        row.value_changed.connect(self._on_row_changed)
        self._rows[name] = row
        group.add_row(row)
        return row

    def _labelled(self, group: ControlGroup, label_key: str, control: QWidget) -> None:
        """Add *control* with its name in the fixed column on the left."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACE_SM)
        label = TranslatedLabel(label_key, group)
        label.setProperty("role", "secondary")
        label.setFixedWidth(theme.DIALOG_LABEL_WIDTH)
        row.addWidget(label)
        row.addWidget(control, 1)
        group.add_layout(row)

    def _build_effect_group(self, parent: QWidget) -> ControlGroup:
        """How much is blurred, and how out of focus light looks."""
        group = ControlGroup("depth.group.effect", first=True, parent=parent)
        self._groups.append(group)
        self._slider(group, "amount")
        self.bokeh_choice = Segmented(
            [
                (depth_of_field.BOKEH_CIRCLE, "depth.bokeh.circle"),
                (depth_of_field.BOKEH_HEXAGON, "depth.bokeh.hexagon"),
            ],
            group,
        )
        self.bokeh_choice.chosen.connect(self.bokeh_chosen.emit)
        self._labelled(group, "depth.bokeh", self.bokeh_choice)
        self._slider(group, "highlights")
        return group

    def _build_range_group(self, parent: QWidget) -> ControlGroup:
        """What stays sharp: the shape, the transition beyond it, and the brush."""
        group = ControlGroup("depth.group.range", parent=parent)
        self._groups.append(group)
        self.shape_choice = Segmented(
            [
                (depth_of_field.SHAPE_ELLIPSE, "depth.shape.ellipse"),
                (depth_of_field.SHAPE_BAND, "depth.shape.band"),
            ],
            group,
        )
        self.shape_choice.chosen.connect(self.shape_chosen.emit)
        self._labelled(group, "depth.shape", self.shape_choice)
        self._slider(group, "transition")

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACE_SM)
        self.sharp_button = BrushChoice("depth.sharp", "shortcut.portrait_add", group)
        self.blur_button = BrushChoice("depth.blur", "shortcut.portrait_remove", group)
        self.sharp_button.clicked.connect(lambda: self._on_choice(BRUSH_SHARP))
        self.blur_button.clicked.connect(lambda: self._on_choice(BRUSH_BLUR))
        row.addWidget(self.sharp_button, 1)
        row.addWidget(self.blur_button, 1)
        group.add_layout(row)

        self.brush_size_row = SliderRow(BRUSH_SIZE_SPEC, group)
        self.brush_size_row.value_changed.connect(self._on_brush_size)
        group.add_row(self.brush_size_row)

        show = QHBoxLayout()
        show.setContentsMargins(0, 0, 0, 0)
        show.setSpacing(theme.SPACE_SM)
        show.addWidget(TranslatedLabel("depth.show", group), 1)
        self.show_switch = Switch(group)
        self.show_switch.setToolTip(tooltip("depth.show", "shortcut.portrait_show"))
        self.show_switch.toggled.connect(self.showing_changed.emit)
        show.addWidget(self.show_switch)
        group.add_layout(show)

        #: How to use it, or how many strokes there are.
        self.caption = QLabel("", group)
        self.caption.setProperty("role", "caption")
        self.caption.setWordWrap(True)
        group.add_row(self.caption)
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
        self.reset_button = TextButton("depth.reset", row)
        self.reset_button.clicked.connect(self.reset_requested.emit)
        layout.addWidget(self.reset_button, 1)
        self.leave_button = TextButton("depth.leave", row)
        self.leave_button.setToolTip(tooltip("depth.leave", "shortcut.depth"))
        self.leave_button.clicked.connect(self.leave_requested.emit)
        layout.addWidget(self.leave_button, 1)
        return row

    # -- what it says ----------------------------------------------------

    @property
    def brush(self) -> str | None:
        """Return what the brush does, or ``None`` while it is put down."""
        if self.sharp_button.isChecked():
            return BRUSH_SHARP
        if self.blur_button.isChecked():
            return BRUSH_BLUR
        return None

    @property
    def brush_size(self) -> float:
        """Return the brush's radius, as a fraction of the long edge."""
        return self._brush_size

    @property
    def showing(self) -> bool:
        """Return whether the range is shown over the photo."""
        return self.show_switch.isChecked()

    def rows(self) -> dict[str, SliderRow]:
        """Return the sliders by name."""
        return dict(self._rows)

    def groups(self) -> tuple[ControlGroup, ...]:
        """Return the groups, in display order."""
        return tuple(self._groups)

    # -- being told things -----------------------------------------------

    def set_depth(self, depth: DepthOfField) -> None:
        """Show the photo's depth of field without announcing it."""
        self._depth = depth
        self._show_depth()

    def set_brush(self, brush: str | None) -> None:
        """Take the brush up for *brush*, or put it down for ``None``, announcing it."""
        self.sharp_button.setChecked(brush == BRUSH_SHARP)
        self.blur_button.setChecked(brush == BRUSH_BLUR)
        self.brush_changed.emit()

    def set_brush_size(self, radius: float) -> None:
        """Show *radius* on the size slider without announcing it."""
        self._brush_size = skin_range.clamp_radius(radius)
        self.brush_size_row.blockSignals(True)
        self.brush_size_row.set_value(self._brush_size * BRUSH_SIZE_SCALE)
        self.brush_size_row.blockSignals(False)

    def set_showing(self, showing: bool) -> None:
        """Show the range over the photo or stop, announcing it."""
        self.show_switch.setChecked(showing)

    def retranslate(self) -> None:
        """Read the sliders and the captions again after a language change (U-02)."""
        for name, row in self._rows.items():
            row.set_spec(SPECS[name])
        self.brush_size_row.set_spec(BRUSH_SIZE_SPEC)
        self.set_brush_size(self._brush_size)
        for button in (self.sharp_button, self.blur_button):
            button.retranslate()
        self.shape_choice.retranslate()
        self.bokeh_choice.retranslate()
        self.reset_button.retranslate()
        self.leave_button.retranslate()
        self.leave_button.setToolTip(tooltip("depth.leave", "shortcut.depth"))
        self.show_switch.setToolTip(tooltip("depth.show", "shortcut.portrait_show"))
        self._show_depth()

    def _show_depth(self) -> None:
        depth = self._depth
        for name, row in self._rows.items():
            row.blockSignals(True)
            row.set_value(getattr(depth, name))
            row.blockSignals(False)
        self.shape_choice.set_value(depth.shape)
        self.bokeh_choice.set_value(depth.bokeh)
        count = len(depth.strokes)
        if not depth.blurs:
            caption = tr("depth.hint.amount")
        elif count == 0:
            caption = tr("depth.hint")
        elif count == 1:
            caption = tr("depth.strokes.one")
        else:
            caption = tr("depth.strokes.many", count=count)
        self.caption.setText(caption)
        fresh = DEFAULT_DEPTH.with_shape(depth.shape)
        self.reset_button.setEnabled(
            bool(depth.strokes)
            or (depth.centre, depth.handle, depth.ratio)
            != (fresh.centre, fresh.handle, fresh.ratio)
        )

    # -- reacting --------------------------------------------------------

    def _on_choice(self, brush: str) -> None:
        """Take the brush up for *brush*; pressing the one already pressed puts it down."""
        pressed = self.sharp_button if brush == BRUSH_SHARP else self.blur_button
        self.set_brush(brush if pressed.isChecked() else None)

    def _on_row_changed(self, name: str, value: float, dragging: bool) -> None:
        self._depth = self._depth.with_values(**{name: value})
        self._set_titles_dimmed(dragging)
        self.value_set.emit(name, value, dragging)

    def _on_brush_size(self, _name: str, value: float, dragging: bool) -> None:
        self._brush_size = skin_range.clamp_radius(value / BRUSH_SIZE_SCALE)
        self._set_titles_dimmed(dragging)
        self.brush_size_set.emit(self._brush_size, dragging)

    def _set_titles_dimmed(self, dimmed: bool) -> None:
        """Fade the group titles while a slider is moving (PLAN.md 3.6)."""
        for group in self._groups:
            group.header.set_faded(dimmed)
