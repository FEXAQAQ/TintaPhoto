"""The right hand panel while liquify is open (PLAN.md 2.11, 3.16).

Takes the inspector's place, at the same width and in the same style, as the
portrait's panel does: a bar naming the mode, the seven tools of the brush,
its size and strength, a line saying how to paint or how many strokes there
are, the five face sliders (L-11) with how many faces they found, and an
action row along the bottom.

The panel only says what was chosen: the controller keeps the strokes with
the photo, and the photo's overlay paints them.
"""

from __future__ import annotations

from typing import Final

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.i18n import tr
from core import liquify
from ui import theme
from ui.widgets.buttons import ROLE_CHOICE, TextButton, tooltip
from ui.widgets.group import ControlGroup
from ui.widgets.region import styled_background
from ui.widgets.scroll import scrolled
from ui.widgets.slider_row import SliderRow, SliderSpec
from ui.widgets.text import TranslatedLabel

#: The size slider counts thousandths of the long edge, as the spot and the
#: portrait brushes do (L-03).
BRUSH_SIZE_SCALE: Final[float] = 1000.0
BRUSH_SIZE_SPEC: Final[SliderSpec] = SliderSpec(
    name="brush_size",
    label_key="liquify.brush_size",
    minimum=liquify.RADIUS_MIN * BRUSH_SIZE_SCALE,
    maximum=liquify.RADIUS_MAX * BRUSH_SIZE_SCALE,
    default=liquify.RADIUS_DEFAULT * BRUSH_SIZE_SCALE,
    signed=False,
    fill_from=liquify.RADIUS_MIN * BRUSH_SIZE_SCALE,
)
#: The strength slider counts hundredths (L-04).
STRENGTH_SCALE: Final[float] = 100.0
STRENGTH_SPEC: Final[SliderSpec] = SliderSpec(
    name="strength",
    label_key="liquify.strength",
    minimum=liquify.STRENGTH_MIN * STRENGTH_SCALE,
    maximum=liquify.STRENGTH_MAX * STRENGTH_SCALE,
    default=liquify.STRENGTH_DEFAULT * STRENGTH_SCALE,
    signed=False,
    fill_from=liquify.STRENGTH_MIN * STRENGTH_SCALE,
)

#: The face sliders (L-11): −100 to +100, resting at 0.
FACE_SPECS: Final[dict[str, SliderSpec]] = {
    name: SliderSpec(
        name=name,
        label_key=f"liquify.face.{name}",
        minimum=-liquify.FACE_LIMIT,
        maximum=liquify.FACE_LIMIT,
        default=0.0,
    )
    for name in liquify.FACE_NAMES
}

#: The tools in the rows the panel lays them out in (PLAN.md 3.16).
TOOL_ROWS: Final[tuple[tuple[str, ...], ...]] = (
    (liquify.TOOL_PUSH, liquify.TOOL_BLOAT, liquify.TOOL_PUCKER),
    (liquify.TOOL_TWIRL_CW, liquify.TOOL_TWIRL_CCW),
    (liquify.TOOL_RECONSTRUCT, liquify.TOOL_SMOOTH),
)
#: The tool the brush starts as.
DEFAULT_TOOL: Final[str] = liquify.TOOL_PUSH


class ToolChoice(QPushButton):
    """One of the brush's tools: pressed while the brush is that tool."""

    def __init__(self, tool: str, parent: QWidget) -> None:
        super().__init__(tr(f"liquify.tool.{tool}"), parent)
        self.tool = tool
        self.setProperty("role", ROLE_CHOICE)
        self.setCheckable(True)
        self.setFixedHeight(theme.BUTTON_HEIGHT)
        # The width comes from the row alone, so that the buttons of a row
        # are the same width whatever their names (PLAN.md 3.4).
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.retranslate()

    def retranslate(self) -> None:
        """Read the name and the tooltip again after a language change."""
        self.setText(tr(f"liquify.tool.{self.tool}"))
        self.setToolTip(tooltip(f"liquify.tool.{self.tool}.tip"))


class LiquifyPanel(QWidget):
    """The liquify brush's tools, size and strength, in the inspector's place."""

    #: Another tool was chosen.
    tool_changed = Signal(str)
    #: The brush size changed: the radius as a fraction of the long edge,
    #: and whether a drag is still going on.
    brush_size_set = Signal(float, bool)
    #: One face slider moved: its name, its value, and whether a drag is
    #: still going on.
    face_value_set = Signal(str, float, bool)
    #: Every stroke taken off the photo and the face sliders put to rest.
    reset_requested = Signal()
    #: Back to editing.
    leave_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        styled_background(self, "LiquifyPanel")
        self.setFixedWidth(theme.PANEL_WIDTH)
        self._groups: list[ControlGroup] = []
        self._brush_size = liquify.RADIUS_DEFAULT
        self._strength = liquify.STRENGTH_DEFAULT
        self._strokes = 0
        self._face = liquify.DEFAULT_FACE_SHAPE
        self._faces: int | None = None
        self.face_rows: dict[str, SliderRow] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_title())
        content = QWidget(self)
        body = QVBoxLayout(content)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addSpacing(theme.SPACE_SM)
        body.addWidget(self._build_tool_group(content))
        body.addWidget(self._build_brush_group(content))
        body.addWidget(self._build_face_group(content))
        body.addStretch(1)
        outer.addWidget(scrolled(content, "TabPage", self), 1)
        outer.addWidget(self._build_action_row())

        self.set_tool(DEFAULT_TOOL)
        self.set_brush_size(self._brush_size)
        self.set_strength(self._strength)
        self._show_strokes()
        self._show_faces()

    # -- building --------------------------------------------------------

    def _build_title(self) -> QWidget:
        """The bar along the top, where the tab strip is when editing."""
        bar = QWidget(self)
        styled_background(bar, "TabBar")
        bar.setFixedHeight(theme.TAB_BAR_HEIGHT)
        row = QHBoxLayout(bar)
        row.setContentsMargins(theme.PANEL_PADDING, 0, theme.PANEL_PADDING, 0)
        row.addWidget(TranslatedLabel("liquify.title", bar))
        row.addStretch(1)
        return bar

    def _build_tool_group(self, parent: QWidget) -> ControlGroup:
        """The seven tools, one always pressed."""
        group = ControlGroup("liquify.group.tools", first=True, parent=parent)
        self._groups.append(group)
        self._choices = QButtonGroup(self)
        self._choices.setExclusive(True)
        self.tool_buttons: dict[str, ToolChoice] = {}
        # Each row shares the panel's width evenly, so the rows run edge to
        # edge alike, and the rows are as far apart as the buttons in them.
        rows = QVBoxLayout()
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(theme.SPACE_SM)
        for tools in TOOL_ROWS:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(theme.SPACE_SM)
            for tool in tools:
                button = ToolChoice(tool, group)
                button.clicked.connect(lambda _checked=False, name=tool: self._on_choice(name))
                self._choices.addButton(button)
                self.tool_buttons[tool] = button
                row.addWidget(button, 1)
            rows.addLayout(row)
        group.add_layout(rows)
        return group

    def _build_brush_group(self, parent: QWidget) -> ControlGroup:
        """The brush's size and strength, and how to paint."""
        group = ControlGroup("liquify.group.brush", parent=parent)
        self._groups.append(group)
        self.brush_size_row = SliderRow(BRUSH_SIZE_SPEC, group)
        self.brush_size_row.value_changed.connect(self._on_brush_size)
        group.add_row(self.brush_size_row)
        self.strength_row = SliderRow(STRENGTH_SPEC, group)
        self.strength_row.value_changed.connect(self._on_strength)
        group.add_row(self.strength_row)
        #: How to paint, or how many strokes there are.
        self.strokes_caption = QLabel("", group)
        self.strokes_caption.setProperty("role", "caption")
        self.strokes_caption.setWordWrap(True)
        group.add_row(self.strokes_caption)
        return group

    def _build_face_group(self, parent: QWidget) -> ControlGroup:
        """The five face sliders, and how many faces they work on."""
        group = ControlGroup("liquify.group.face", parent=parent)
        self._groups.append(group)
        for name in liquify.FACE_NAMES:
            row = SliderRow(FACE_SPECS[name], group)
            row.value_changed.connect(self._on_face_row)
            self.face_rows[name] = row
            group.add_row(row)
        #: How many faces were found, and what that means for the sliders.
        self.faces_caption = QLabel("", group)
        self.faces_caption.setProperty("role", "caption")
        self.faces_caption.setWordWrap(True)
        group.add_row(self.faces_caption)
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
        self.reset_button = TextButton("liquify.reset", row)
        self.reset_button.clicked.connect(self.reset_requested.emit)
        layout.addWidget(self.reset_button, 1)
        self.leave_button = TextButton("liquify.leave", row)
        self.leave_button.setToolTip(tooltip("liquify.leave", "shortcut.liquify"))
        self.leave_button.clicked.connect(self.leave_requested.emit)
        layout.addWidget(self.leave_button, 1)
        return row

    # -- what it says ----------------------------------------------------

    @property
    def tool(self) -> str:
        """Return the tool the brush is."""
        for tool, button in self.tool_buttons.items():
            if button.isChecked():
                return tool
        return DEFAULT_TOOL

    @property
    def brush_size(self) -> float:
        """Return the brush's radius, as a fraction of the long edge."""
        return self._brush_size

    @property
    def strength(self) -> float:
        """Return how hard the brush works, from 0 to 1."""
        return self._strength

    def groups(self) -> tuple[ControlGroup, ...]:
        """Return the groups, in display order."""
        return tuple(self._groups)

    # -- being told things -----------------------------------------------

    def set_tool(self, tool: str) -> None:
        """Press the button of *tool* without announcing it."""
        button = self.tool_buttons.get(tool, self.tool_buttons[DEFAULT_TOOL])
        button.setChecked(True)

    def set_strokes(self, count: int) -> None:
        """Say how many strokes the photo has."""
        self._strokes = max(0, count)
        self._show_strokes()

    def set_face_shape(self, face: liquify.FaceShape) -> None:
        """Show the photo's face sliders without announcing them."""
        self._face = face
        for name, row in self.face_rows.items():
            row.blockSignals(True)
            row.set_value(face.value(name))
            row.blockSignals(False)
        self._show_strokes()

    def set_faces(self, count: int | None) -> None:
        """Say how many faces the photo holds, or ``None`` while looking."""
        self._faces = count
        self._show_faces()

    def set_brush_size(self, radius: float) -> None:
        """Show *radius* on the size slider without announcing it."""
        self._brush_size = liquify.clamp_radius(radius)
        self.brush_size_row.blockSignals(True)
        self.brush_size_row.set_value(self._brush_size * BRUSH_SIZE_SCALE)
        self.brush_size_row.blockSignals(False)

    def set_strength(self, strength: float) -> None:
        """Show *strength* on its slider without announcing it."""
        self._strength = liquify.clamp_strength(strength)
        self.strength_row.blockSignals(True)
        self.strength_row.set_value(self._strength * STRENGTH_SCALE)
        self.strength_row.blockSignals(False)

    def retranslate(self) -> None:
        """Read the sliders and the caption again after a language change (U-02)."""
        self.brush_size_row.set_spec(BRUSH_SIZE_SPEC)
        self.strength_row.set_spec(STRENGTH_SPEC)
        self.set_brush_size(self._brush_size)
        self.set_strength(self._strength)
        for button in self.tool_buttons.values():
            button.retranslate()
        self.reset_button.retranslate()
        self.leave_button.retranslate()
        self.leave_button.setToolTip(tooltip("liquify.leave", "shortcut.liquify"))
        for name, row in self.face_rows.items():
            row.set_spec(FACE_SPECS[name])
        self.set_face_shape(self._face)
        self._show_strokes()
        self._show_faces()

    def _show_faces(self) -> None:
        count = self._faces
        if count is None:
            caption = tr("adjust.faces.looking")
        elif count == 0:
            caption = tr("liquify.faces.none")
        elif count == 1:
            caption = tr("adjust.faces.one")
        else:
            caption = tr("adjust.faces.many", count=count)
        self.faces_caption.setText(caption)
        # With no face there is nothing for the sliders to shape (L-11).
        for row in self.face_rows.values():
            row.setEnabled(bool(count))

    def _show_strokes(self) -> None:
        count = self._strokes
        if count == 0:
            caption = tr("liquify.hint")
        elif count == 1:
            caption = tr("liquify.strokes.one")
        else:
            caption = tr("liquify.strokes.many", count=count)
        self.strokes_caption.setText(caption)
        self.reset_button.setEnabled(count > 0 or not self._face.is_default)

    # -- reacting --------------------------------------------------------

    def _on_choice(self, tool: str) -> None:
        self.set_tool(tool)
        self.tool_changed.emit(tool)

    def _on_brush_size(self, _name: str, value: float, dragging: bool) -> None:
        self._brush_size = liquify.clamp_radius(value / BRUSH_SIZE_SCALE)
        self._set_titles_dimmed(dragging)
        self.brush_size_set.emit(self._brush_size, dragging)

    def _on_face_row(self, name: str, value: float, dragging: bool) -> None:
        self._face = self._face.with_value(name, value)
        self._show_strokes()
        self._set_titles_dimmed(dragging)
        self.face_value_set.emit(name, value, dragging)

    def _on_strength(self, _name: str, value: float, dragging: bool) -> None:
        self._strength = liquify.clamp_strength(value / STRENGTH_SCALE)
        self._set_titles_dimmed(dragging)

    def _set_titles_dimmed(self, dimmed: bool) -> None:
        """Fade the group titles while a slider is moving (PLAN.md 3.6)."""
        for group in self._groups:
            group.header.set_faded(dimmed)
