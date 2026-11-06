"""The right hand panel while the portrait is open (PLAN.md 2.10, 3.15).

Takes the inspector's place, at the same width and in the same style, as the
watermark's panel does: a bar naming the mode, a line saying how many faces
were found, the two portrait sliders (A-29, A-30), the skin they work on and
the brush that shapes it, and an action row along the bottom.

The panel only says what was chosen: the controller keeps the skin range
with the photo, and the photo's overlay paints and shows it.
"""

from __future__ import annotations

from typing import Final

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app.i18n import tr
from core import skin_range
from core.edit_state import PORTRAIT_NAMES, EditState
from core.skin_range import SkinRange
from ui import theme
from ui.adjust_tab import JPEG_SPECS
from ui.widgets.buttons import ROLE_CHOICE, TextButton, tooltip
from ui.widgets.group import ControlGroup
from ui.widgets.region import styled_background
from ui.widgets.scroll import scrolled
from ui.widgets.slider_row import SliderRow, SliderSpec
from ui.widgets.switch import Switch
from ui.widgets.text import TranslatedLabel

#: The brush's two jobs.
BRUSH_ADD: Final[str] = "add"
BRUSH_REMOVE: Final[str] = "remove"

#: The brush slider counts thousandths of the long edge, as the spot brush's
#: does (:mod:`core.skin_range`).
BRUSH_SIZE_SCALE: Final[float] = 1000.0
BRUSH_SIZE_SPEC: Final[SliderSpec] = SliderSpec(
    name="brush_size",
    label_key="portrait.brush_size",
    minimum=skin_range.RADIUS_MIN * BRUSH_SIZE_SCALE,
    maximum=skin_range.RADIUS_MAX * BRUSH_SIZE_SCALE,
    default=skin_range.RADIUS_DEFAULT * BRUSH_SIZE_SCALE,
    signed=False,
    fill_from=skin_range.RADIUS_MIN * BRUSH_SIZE_SCALE,
)


class BrushChoice(QPushButton):
    """One of the brush's two jobs: pressed while the brush does it."""

    def __init__(self, label_key: str, shortcut_key: str, parent: QWidget) -> None:
        super().__init__(tr(label_key), parent)
        self._label_key = label_key
        self._shortcut_key = shortcut_key
        self.setProperty("role", ROLE_CHOICE)
        self.setCheckable(True)
        self.setFixedHeight(theme.BUTTON_HEIGHT)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setToolTip(tooltip(label_key, shortcut_key))

    def retranslate(self) -> None:
        """Read the name and the tooltip again after a language change."""
        self.setText(tr(self._label_key))
        self.setToolTip(tooltip(self._label_key, self._shortcut_key))


class PortraitPanel(QWidget):
    """The portrait sliders and the skin they work on, in the inspector's place."""

    #: One portrait slider moved: the adjustment, its value, and whether a
    #: drag is still going on, the same as the adjust tab's sliders.
    value_set = Signal(str, float, bool)
    #: Whether the skin beside the faces is found by itself.
    auto_set = Signal(bool)
    #: The brush was taken up, put down or given the other job.
    brush_changed = Signal()
    #: The brush size changed: the radius as a fraction of the long edge,
    #: and whether a drag is still going on.
    brush_size_set = Signal(float, bool)
    #: Whether the skin is shown over the photo.
    showing_changed = Signal(bool)
    #: Back to the skin found by itself.
    reset_requested = Signal()
    #: Back to editing.
    leave_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        styled_background(self, "PortraitPanel")
        self.setFixedWidth(theme.PANEL_WIDTH)
        self._state = EditState()
        self._rows: dict[str, SliderRow] = {}
        self._groups: list[ControlGroup] = []
        self._faces: int | None = None
        self._skin = SkinRange()
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
        self._show_faces()
        self._show_skin()

    # -- building --------------------------------------------------------

    def _build_title(self) -> QWidget:
        """The bar along the top, where the tab strip is when editing."""
        bar = QWidget(self)
        styled_background(bar, "TabBar")
        bar.setFixedHeight(theme.TAB_BAR_HEIGHT)
        row = QHBoxLayout(bar)
        row.setContentsMargins(theme.PANEL_PADDING, 0, theme.PANEL_PADDING, 0)
        row.addWidget(TranslatedLabel("portrait.title", bar))
        row.addStretch(1)
        return bar

    def _switch_row(self, group: ControlGroup, label_key: str) -> Switch:
        """Add a row with a name on the left and a switch on the right."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACE_SM)
        row.addWidget(TranslatedLabel(label_key, group), 1)
        switch = Switch(group)
        switch.setToolTip(tr(label_key))
        row.addWidget(switch)
        group.add_layout(row)
        return switch

    def _build_effect_group(self, parent: QWidget) -> ControlGroup:
        """The skin and whitening sliders, and how many faces were found."""
        group = ControlGroup("portrait.group.effect", first=True, parent=parent)
        self._groups.append(group)
        for name in PORTRAIT_NAMES:
            row = SliderRow(JPEG_SPECS[name], group)
            row.value_changed.connect(self._on_row_changed)
            self._rows[name] = row
            group.add_row(row)
        #: How many faces were found, and what that means for the sliders.
        self.faces_caption = QLabel("", group)
        self.faces_caption.setProperty("role", "caption")
        self.faces_caption.setWordWrap(True)
        group.add_row(self.faces_caption)
        return group

    def _build_range_group(self, parent: QWidget) -> ControlGroup:
        """The skin the sliders work on: found by itself, and the brush."""
        group = ControlGroup("portrait.group.range", parent=parent)
        self._groups.append(group)
        self.auto_switch = self._switch_row(group, "portrait.auto")
        self.auto_switch.toggled.connect(self.auto_set.emit)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACE_SM)
        self.add_button = BrushChoice("portrait.add", "shortcut.portrait_add", group)
        self.remove_button = BrushChoice("portrait.remove", "shortcut.portrait_remove", group)
        self.add_button.clicked.connect(lambda: self._on_choice(BRUSH_ADD))
        self.remove_button.clicked.connect(lambda: self._on_choice(BRUSH_REMOVE))
        row.addWidget(self.add_button, 1)
        row.addWidget(self.remove_button, 1)
        group.add_layout(row)

        self.brush_size_row = SliderRow(BRUSH_SIZE_SPEC, group)
        self.brush_size_row.value_changed.connect(self._on_brush_size)
        group.add_row(self.brush_size_row)
        self.snap_switch = self._switch_row(group, "portrait.snap")
        self.snap_switch.setChecked(True)
        self.show_switch = self._switch_row(group, "portrait.show")
        self.show_switch.setToolTip(tooltip("portrait.show", "shortcut.portrait_show"))
        self.show_switch.toggled.connect(self.showing_changed.emit)

        #: How to paint, or how many strokes there are.
        self.strokes_caption = QLabel("", group)
        self.strokes_caption.setProperty("role", "caption")
        self.strokes_caption.setWordWrap(True)
        group.add_row(self.strokes_caption)
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
        self.reset_button = TextButton("portrait.reset", row)
        self.reset_button.clicked.connect(self.reset_requested.emit)
        layout.addWidget(self.reset_button, 1)
        self.leave_button = TextButton("portrait.leave", row)
        self.leave_button.setToolTip(tooltip("portrait.leave", "shortcut.portrait"))
        self.leave_button.clicked.connect(self.leave_requested.emit)
        layout.addWidget(self.leave_button, 1)
        return row

    # -- what it says ----------------------------------------------------

    @property
    def brush(self) -> str | None:
        """Return what the brush does, or ``None`` while it is put down."""
        if self.add_button.isChecked():
            return BRUSH_ADD
        if self.remove_button.isChecked():
            return BRUSH_REMOVE
        return None

    @property
    def brush_size(self) -> float:
        """Return the brush's radius, as a fraction of the long edge."""
        return self._brush_size

    @property
    def snap(self) -> bool:
        """Return whether the brush keeps to the colour of skin."""
        return self.snap_switch.isChecked()

    @property
    def showing(self) -> bool:
        """Return whether the skin is shown over the photo."""
        return self.show_switch.isChecked()

    def rows(self) -> dict[str, SliderRow]:
        """Return the portrait sliders by adjustment name."""
        return dict(self._rows)

    def groups(self) -> tuple[ControlGroup, ...]:
        """Return the groups, in display order."""
        return tuple(self._groups)

    # -- being told things -----------------------------------------------

    def set_state(self, state: EditState) -> None:
        """Show the sliders of *state* without announcing them."""
        self._state = state
        for name, row in self._rows.items():
            row.blockSignals(True)
            row.set_value(getattr(state, name))
            row.blockSignals(False)

    def set_faces(self, count: int | None) -> None:
        """Say how many faces the photo holds, or ``None`` while looking."""
        self._faces = count
        self._show_faces()

    def set_skin(self, skin: SkinRange) -> None:
        """Show the photo's skin range without announcing it."""
        self._skin = skin
        self._show_skin()

    def set_brush(self, brush: str | None) -> None:
        """Take the brush up for *brush*, or put it down for ``None``, announcing it."""
        self.add_button.setChecked(brush == BRUSH_ADD)
        self.remove_button.setChecked(brush == BRUSH_REMOVE)
        self.brush_changed.emit()

    def set_brush_size(self, radius: float) -> None:
        """Show *radius* on the size slider without announcing it."""
        self._brush_size = skin_range.clamp_radius(radius)
        self.brush_size_row.blockSignals(True)
        self.brush_size_row.set_value(self._brush_size * BRUSH_SIZE_SCALE)
        self.brush_size_row.blockSignals(False)

    def set_showing(self, showing: bool) -> None:
        """Show the skin over the photo or stop, announcing it."""
        self.show_switch.setChecked(showing)

    def retranslate(self) -> None:
        """Read the sliders and the captions again after a language change (U-02)."""
        for name, row in self._rows.items():
            row.set_spec(JPEG_SPECS[name])
            row.blockSignals(True)
            row.set_value(getattr(self._state, name))
            row.blockSignals(False)
        self.brush_size_row.set_spec(BRUSH_SIZE_SPEC)
        self.set_brush_size(self._brush_size)
        for button in (self.add_button, self.remove_button):
            button.retranslate()
        self.reset_button.retranslate()
        self.leave_button.retranslate()
        self.leave_button.setToolTip(tooltip("portrait.leave", "shortcut.portrait"))
        for switch, key in (
            (self.auto_switch, "portrait.auto"),
            (self.snap_switch, "portrait.snap"),
        ):
            switch.setToolTip(tr(key))
        self.show_switch.setToolTip(tooltip("portrait.show", "shortcut.portrait_show"))
        self._show_faces()
        self._show_skin()

    def _show_faces(self) -> None:
        count = self._faces
        if count is None:
            caption = tr("adjust.faces.looking")
        elif count == 0:
            caption = tr("portrait.faces.none")
        elif count == 1:
            caption = tr("adjust.faces.one")
        else:
            caption = tr("adjust.faces.many", count=count)
        self.faces_caption.setText(caption)

    def _show_skin(self) -> None:
        skin = self._skin
        self.auto_switch.blockSignals(True)
        self.auto_switch.setChecked(skin.auto)
        self.auto_switch.blockSignals(False)
        count = len(skin.strokes)
        if count == 0:
            caption = tr("portrait.hint")
        elif count == 1:
            caption = tr("portrait.strokes.one")
        else:
            caption = tr("portrait.strokes.many", count=count)
        self.strokes_caption.setText(caption)
        self.reset_button.setEnabled(not skin.is_default)

    # -- reacting --------------------------------------------------------

    def _on_choice(self, brush: str) -> None:
        """Take the brush up for *brush*; pressing the one already pressed puts it down."""
        pressed = self.add_button if brush == BRUSH_ADD else self.remove_button
        self.set_brush(brush if pressed.isChecked() else None)

    def _on_row_changed(self, name: str, value: float, dragging: bool) -> None:
        self._state = self._state.with_value(name, value)
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
