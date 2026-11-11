"""The right hand panel while a collage is open (PLAN.md 3.13).

Takes the inspector's place, at the same width and in the same style: a bar
naming the mode, three groups that fold away like the inspector's, and an
action row along the bottom. The panel only says what was chosen; the
collage controller keeps it and the canvas draws it.

The groups follow the order the choices are made in: the shape of the canvas,
how and into how many pieces it is split, then the photos - how they fit,
whether they are one scene, how far each is zoomed and how the later ones are
aligned against the first - then the gap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.i18n import tr
from core import collage
from core.collage import Layout
from ui import theme
from ui.widgets.buttons import TextButton, tooltip
from ui.widgets.group import ControlGroup
from ui.widgets.region import styled_background
from ui.widgets.scroll import scrolled
from ui.widgets.segmented import Segmented
from ui.widgets.slider_row import SliderRow, SliderSpec
from ui.widgets.switch import Switch
from ui.widgets.text import TranslatedLabel, elide_middle

#: Zoom, seam and alignment are shown as percentages.
_PERCENT: Final[float] = 100.0


def gap_spec() -> SliderSpec:
    """Return the gap slider: the same control as every other slider."""
    return SliderSpec(
        name="gap",
        label_key="collage.gap",
        minimum=float(collage.GAP_MIN),
        maximum=float(collage.GAP_MAX),
        default=float(collage.GAP_MIN),
        signed=False,
        suffix=tr("collage.pixels_suffix"),
        step=1.0,
        fill_from=float(collage.GAP_MIN),
    )


def seam_spec() -> SliderSpec:
    """Return the seam slider, resting in the middle (PLAN.md 2.8 K-10)."""
    return SliderSpec(
        name="seam",
        label_key="collage.seam",
        minimum=collage.SEAM_MIN * _PERCENT,
        maximum=collage.SEAM_MAX * _PERCENT,
        default=collage.SEAM_DEFAULT * _PERCENT,
        decimals=1,
        signed=False,
        suffix=tr("collage.percent_suffix"),
    )


def zoom_spec(label_key: str) -> SliderSpec:
    """Return a zoom slider, from the fit at 100% up (PLAN.md 2.8 K-06)."""
    return SliderSpec(
        name="zoom",
        label_key=label_key,
        minimum=collage.ZOOM_MIN * _PERCENT,
        maximum=collage.ZOOM_MAX * _PERCENT,
        default=collage.ZOOM_MIN * _PERCENT,
        decimals=1,
        signed=False,
        suffix=tr("collage.percent_suffix"),
        fill_from=collage.ZOOM_MIN * _PERCENT,
    )


def alignment_scale_spec() -> SliderSpec:
    """Return the slider that scales the photo being aligned against the first."""
    return SliderSpec(
        name="alignment_scale",
        label_key="collage.align_scale",
        minimum=collage.ALIGN_SCALE_MIN * _PERCENT,
        maximum=collage.ALIGN_SCALE_MAX * _PERCENT,
        default=_PERCENT,
        decimals=1,
        signed=False,
        suffix=tr("collage.percent_suffix"),
    )


#: The photo rows and their zoom sliders are numbered 1 to 4, left to right
#: or top to bottom - a grid read like a page - whatever the count and the
#: split (PLAN.md 3.13).
_SLOT_LABEL: Final[str] = "collage.slot"


def _zoom_label(slot: int) -> str:
    """Return the name of the zoom slider of the photo in *slot*."""
    return f"collage.zoom.{slot + 1}"


def _count_value(count: int) -> str:
    """Return the value of the count choice that stands for *count* pieces."""
    return str(count)


class CollagePanel(QWidget):
    """The collage's choices, in the inspector's place."""

    #: The layout the controls now describe.
    layout_changed = Signal(object)
    #: The user wants to pick the photo for a piece, by its slot.
    choose_requested = Signal(int)
    #: A zoom slider moved: the slot, and the zoom as a factor of the fit.
    zoom_changed = Signal(int, float)
    #: Aligning was turned on or off.
    aligning_toggled = Signal(bool)
    #: Which linked photo after the first is being aligned, by its slot.
    aligning_slot_changed = Signal(int)
    #: The scale of the photo being aligned against the first, as a factor.
    alignment_scale_changed = Signal(float)
    #: Back to editing photos one at a time.
    leave_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        styled_background(self, "CollagePanel")
        self.setFixedWidth(theme.PANEL_WIDTH)
        self._layout = Layout()
        #: The count a split side by side or stacked had, for when a grid,
        #: which is always four, is left again.
        self._line_count = self._layout.count
        self._paths: tuple[Path | None, ...] = (None,) * collage.SLOTS
        self._row_labels: list[QLabel] = []
        self._name_rows: list[list[QWidget]] = []
        self._fit_widgets: list[QWidget] = []
        self._linked_widgets: list[QWidget] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_title())

        content = QWidget(self)
        body = QVBoxLayout(content)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_layout_group())
        body.addWidget(self._build_photos_group())
        body.addWidget(self._build_spacing_group())
        body.addStretch(1)
        outer.addWidget(scrolled(content, "TabPage", self), 1)
        outer.addWidget(self._build_action_row())

        self.set_layout(self._layout)

    # -- building --------------------------------------------------------

    def _build_title(self) -> QWidget:
        """The bar along the top, where the tab strip is when editing."""
        bar = QWidget(self)
        styled_background(bar, "TabBar")
        bar.setFixedHeight(theme.TAB_BAR_HEIGHT)
        row = QHBoxLayout(bar)
        row.setContentsMargins(theme.PANEL_PADDING, 0, theme.PANEL_PADDING, 0)
        title = TranslatedLabel("collage.title", bar)
        row.addWidget(title)
        row.addStretch(1)
        return bar

    def _labelled(
        self, group: ControlGroup, label_key: str, control: QWidget
    ) -> list[QWidget]:
        """Add *control* with its name in a fixed column on the left."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACE_SM)
        label = TranslatedLabel(label_key, group)
        label.setProperty("role", "secondary")
        label.setFixedWidth(theme.DIALOG_LABEL_WIDTH)
        row.addWidget(label)
        row.addWidget(control, 1)
        group.add_layout(row)
        return [label, control]

    def _switch_row(self, group: ControlGroup, label_key: str, switch: Switch) -> list[QWidget]:
        """Add a switch with its name on the left, as in the export dialog."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACE_SM)
        label = TranslatedLabel(label_key, group)
        row.addWidget(label, 1)
        row.addWidget(switch)
        group.add_layout(row)
        return [label, switch]

    def _build_layout_group(self) -> ControlGroup:
        """The shape of the canvas, how it is split, into how many and where."""
        group = ControlGroup("collage.group.layout", parent=self)
        self.platform_choice = Segmented(
            [(platform, f"collage.platform.{platform}") for platform in collage.PLATFORMS],
            group,
        )
        self.platform_choice.chosen.connect(lambda _value: self._changed())
        self._labelled(group, "collage.platform", self.platform_choice)

        # One row of canvases for each platform; only the chosen one shows.
        self.canvas_choices: dict[str, Segmented] = {}
        self._canvas_rows: dict[str, list[QWidget]] = {}
        for platform, canvases in collage.PLATFORM_CANVASES.items():
            choice = Segmented(
                [(canvas, f"collage.canvas.{canvas}") for canvas in canvases], group
            )
            choice.chosen.connect(lambda _value: self._changed())
            self.canvas_choices[platform] = choice
            self._canvas_rows[platform] = self._labelled(group, "collage.canvas", choice)

        self.split_choice = Segmented(
            [
                (collage.SPLIT_SIDE_BY_SIDE, "collage.split.side_by_side"),
                (collage.SPLIT_STACKED, "collage.split.stacked"),
                (collage.SPLIT_GRID, "collage.split.grid"),
            ],
            group,
        )
        self.split_choice.chosen.connect(lambda _value: self._changed())
        self._labelled(group, "collage.split", self.split_choice)

        self.count_choice = Segmented(
            [(_count_value(count), f"collage.count.{count}") for count in collage.COUNTS],
            group,
        )
        self.count_choice.chosen.connect(lambda _value: self._changed())
        self._count_row = self._labelled(group, "collage.count", self.count_choice)

        # Two pieces have one seam, which a slider moves; three or four, and
        # a grid, are moved on the canvas, where each seam can be taken hold of.
        self.seam_slider = SliderRow(seam_spec(), group)
        self.seam_slider.value_changed.connect(lambda *_: self._changed())
        group.add_row(self.seam_slider)

        self.size_label = QLabel("", group)
        self.size_label.setProperty("role", "caption")
        self.size_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        group.add_row(self.size_label)
        return group

    def _build_photos_group(self) -> ControlGroup:
        """The photos: which, whether one scene, how they fit, zoom and align."""
        group = ControlGroup("collage.group.photos", parent=self)

        self.name_labels: list[QLabel] = []
        self.choose_buttons: list[TextButton] = []
        for slot in range(collage.SLOTS):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(theme.SPACE_SM)
            label = QLabel("", group)
            label.setProperty("role", "secondary")
            label.setFixedWidth(theme.DIALOG_LABEL_WIDTH)
            row.addWidget(label)
            self._row_labels.append(label)
            name = QLabel("", group)
            row.addWidget(name, 1)
            self.name_labels.append(name)
            button = TextButton("collage.choose", group)
            button.clicked.connect(lambda _checked=False, s=slot: self.choose_requested.emit(s))
            row.addWidget(button)
            self.choose_buttons.append(button)
            self._name_rows.append([label, name, button])
            group.add_layout(row)

        self.linked_switch = Switch(group)
        self.linked_switch.toggled.connect(lambda _on: self._changed())
        self._switch_row(group, "collage.linked", self.linked_switch)

        self.fit_choice = Segmented(
            [
                (collage.FIT_COVER, "collage.fit.cover"),
                (collage.FIT_CONTAIN, "collage.fit.contain"),
            ],
            group,
        )
        self.fit_choice.chosen.connect(lambda _value: self._changed())
        self._fit_widgets = self._labelled(group, "collage.fit", self.fit_choice)

        self.zoom_sliders: list[SliderRow] = []
        for slot in range(collage.SLOTS):
            slider = SliderRow(zoom_spec(_zoom_label(slot)), group)
            slider.value_changed.connect(
                lambda _name, value, _dragging, s=slot: self.zoom_changed.emit(s, value / _PERCENT)
            )
            group.add_row(slider)
            self.zoom_sliders.append(slider)

        self.aligning_switch = Switch(group)
        self.aligning_switch.toggled.connect(self._on_aligning)
        self._linked_widgets += self._switch_row(group, "collage.aligning", self.aligning_switch)
        # Which photo is aligned against the first, when there is a choice.
        self.align_target_choice = Segmented(
            [(str(slot), f"collage.align_target.{slot + 1}") for slot in range(1, collage.SLOTS)],
            group,
        )
        self.align_target_choice.chosen.connect(
            lambda value: self.aligning_slot_changed.emit(int(value))
        )
        self._align_target_widgets = self._labelled(
            group, "collage.align_target", self.align_target_choice
        )
        self.alignment_slider = SliderRow(alignment_scale_spec(), group)
        self.alignment_slider.value_changed.connect(
            lambda _name, value, _dragging: self.alignment_scale_changed.emit(value / _PERCENT)
        )
        group.add_row(self.alignment_slider)
        self._linked_widgets.append(self.alignment_slider)

        self.hint = QLabel("", group)
        self.hint.setProperty("role", "caption")
        self.hint.setWordWrap(True)
        group.add_row(self.hint)
        return group

    def _build_spacing_group(self) -> ControlGroup:
        """The gap between the pieces and its colour."""
        group = ControlGroup("collage.group.spacing", parent=self)
        self.gap_slider = SliderRow(gap_spec(), group)
        self.gap_slider.value_changed.connect(lambda *_: self._changed())
        group.add_row(self.gap_slider)

        self.background_choice = Segmented(
            [
                (collage.BACKGROUND_WHITE, "collage.background.white"),
                (collage.BACKGROUND_BLACK, "collage.background.black"),
            ],
            group,
        )
        self.background_choice.chosen.connect(lambda _value: self._changed())
        self._labelled(group, "collage.background", self.background_choice)
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
        self.leave_button = TextButton("collage.leave", row)
        self.leave_button.setToolTip(tooltip("collage.leave", "shortcut.crop_cancel"))
        self.leave_button.clicked.connect(self.leave_requested.emit)
        layout.addWidget(self.leave_button, 1)
        return row

    # -- being told things -----------------------------------------------

    @property
    def layout_value(self) -> Layout:
        """Return the layout the controls now describe.

        A different count starts from even pieces; with two, the seam is the
        slider's; with more, the seams stay where the canvas left them. A
        grid is always four, and leaving it goes back to the count before it;
        going into a grid or out of one starts from even pieces too.
        """
        split = self.split_choice.value
        was_grid = self._layout.split == collage.SPLIT_GRID
        if split == collage.SPLIT_GRID:
            count = collage.GRID_COUNT
        elif was_grid:
            count = self._line_count
        else:
            count = int(self.count_choice.value)
        if (split == collage.SPLIT_GRID) != was_grid or count != self._layout.count:
            seams: tuple[float, ...] = ()
        elif split == collage.SPLIT_GRID:
            seams = self._layout.seams
        elif count == collage.COUNT_MIN:
            seams = (self.seam_slider.value / _PERCENT,)
        else:
            seams = self._layout.seams
        platform = self.platform_choice.value
        if platform == self._layout.platform:
            canvas = self.canvas_choices[platform].value
        else:
            canvas = collage.with_platform(self._layout, platform).canvas
        return Layout(
            platform=platform,
            canvas=canvas,
            split=split,
            fit=self.fit_choice.value,
            gap=round(self.gap_slider.value),
            background=self.background_choice.value,
            linked=self.linked_switch.isChecked(),
            count=count,
            seams=seams,
        ).sane()

    @property
    def aligning(self) -> bool:
        """Return whether a linked photo is being aligned."""
        return self.aligning_switch.isChecked() and self._layout.linked

    def set_layout(self, layout: Layout) -> None:
        """Show *layout* without announcing it."""
        layout = layout.sane()
        self._layout = layout
        grid = layout.split == collage.SPLIT_GRID
        if not grid:
            self._line_count = layout.count
        self.platform_choice.set_value(layout.platform)
        for platform, choice in self.canvas_choices.items():
            visible = platform == layout.platform
            for widget in self._canvas_rows[platform]:
                widget.setVisible(visible)
            if visible:
                choice.set_value(layout.canvas)
        self.split_choice.set_value(layout.split)
        self.count_choice.set_value(_count_value(layout.count))
        # A grid is always four: the count shows it but cannot be changed.
        for widget in self._count_row:
            widget.setEnabled(not grid)
        self.fit_choice.set_value(layout.fit)
        self.gap_slider.set_value(float(layout.gap))
        self.background_choice.set_value(layout.background)
        if not grid and layout.count == collage.COUNT_MIN:
            self.seam_slider.set_value(layout.seams[0] * _PERCENT)
        self.linked_switch.blockSignals(True)
        self.linked_switch.setChecked(layout.linked)
        self.linked_switch.blockSignals(False)
        self._show_mode()
        self._show_names()

    def set_paths(self, paths: tuple[Path | None, ...]) -> None:
        """Show which photo is in each piece."""
        self._paths = tuple(paths)
        self._show_names()

    def show_placements(
        self, zooms: tuple[float, ...], alignment_scale: float, aligning_slot: int = 1
    ) -> None:
        """Show each photo's zoom and the aligned one's scale, without announcing."""
        for slider, zoom in zip(self.zoom_sliders, zooms):
            slider.set_value(zoom * _PERCENT)
        self.alignment_slider.set_value(alignment_scale * _PERCENT)
        self.align_target_choice.set_value(str(aligning_slot))

    def set_aligning(self, aligning: bool) -> None:
        """Turn aligning on or off from elsewhere, saying so."""
        self.aligning_switch.setChecked(aligning)

    def _show_mode(self) -> None:
        """Show the controls that belong to linked or separate photos."""
        linked = self._layout.linked
        for widget in self._fit_widgets:
            widget.setVisible(not linked)
        for widget in self._linked_widgets:
            widget.setVisible(linked)
        if not linked and self.aligning_switch.isChecked():
            self.aligning_switch.setChecked(False)
        count = self._layout.count
        grid = self._layout.split == collage.SPLIT_GRID
        self.seam_slider.setVisible(not grid and count == collage.COUNT_MIN)
        # Linked, one zoom moves every photo.
        for slot, slider in enumerate(self.zoom_sliders):
            slider.setVisible(slot == 0 or (not linked and slot < count))
        self.zoom_sliders[0].set_spec(zoom_spec("collage.zoom" if linked else _zoom_label(0)))
        # With two photos only the second can be aligned, so there is no choice.
        for widget in self._align_target_widgets:
            widget.setVisible(linked and count > collage.COUNT_MIN)
        for value, button in self.align_target_choice.buttons().items():
            button.setVisible(int(value) < count)
        self._show_hint()

    def _show_hint(self) -> None:
        if self.aligning:
            key = "collage.hint.aligning"
        elif self._layout.linked:
            key = "collage.hint.linked"
        else:
            key = "collage.hint"
        self.hint.setText(tr(key))

    def _show_names(self) -> None:
        width, height = self._layout.canvas_size
        use = tr(f"collage.canvas_use.{self._layout.platform}.{self._layout.canvas}")
        self.size_label.setText(
            tr("collage.canvas_caption", use=use, size=tr("export.pixels", width=width, height=height))
        )
        for slot, (label, name, button) in enumerate(
            zip(self._row_labels, self.name_labels, self.choose_buttons)
        ):
            for widget in self._name_rows[slot]:
                widget.setVisible(slot < self._layout.count)
            label.setText(tr(_SLOT_LABEL, number=slot + 1))
            path = self._paths[slot] if slot < len(self._paths) else None
            if path is None:
                name.setText(tr("collage.empty_slot"))
                name.setProperty("role", "caption")
                name.setToolTip("")
            else:
                name.setText(elide_middle(path.name, name, self._name_width(button)))
                name.setProperty("role", "")
                name.setToolTip(str(path))
            name.style().unpolish(name)
            name.style().polish(name)
            button.set_label_key("collage.replace" if path is not None else "collage.choose")

    def _name_width(self, button: TextButton) -> int:
        """Return the room a photo's name has between its label and button."""
        return (
            theme.PANEL_WIDTH
            - 2 * theme.PANEL_PADDING
            - theme.SCROLLBAR_WIDTH
            - theme.DIALOG_LABEL_WIDTH
            - 2 * theme.SPACE_SM
            - button.sizeHint().width()
        )

    def retranslate(self) -> None:
        """Read the names that are not translated labels again (U-02)."""
        for slider in (self.gap_slider, self.seam_slider, self.alignment_slider, *self.zoom_sliders):
            slider.retranslate()
        self.leave_button.setToolTip(tooltip("collage.leave", "shortcut.crop_cancel"))
        self._show_mode()
        self._show_names()

    def _on_aligning(self, _on: bool) -> None:
        self._show_hint()
        self.aligning_toggled.emit(self.aligning)

    def _changed(self) -> None:
        layout = self.layout_value
        if layout == self._layout:
            return
        # Shown in full, since another platform shows another row of canvases.
        self.set_layout(layout)
        self.layout_changed.emit(layout)
