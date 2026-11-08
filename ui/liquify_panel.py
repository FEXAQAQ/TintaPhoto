"""The right hand panel while liquify is open (PLAN.md 2.11, 3.16).

Takes the inspector's place, at the same width and in the same style, as the
portrait's panel does. Liquify is two ways of working, and the panel shows
one at a time, chosen at the top:

* **Face** - sliders, nothing to paint. A line says how many faces were
  found; the sliders come in two groups, the face's shape and its
  features, and each says under its track what its two ends do. Under
  them two buttons act on the faces: putting the sliders to rest, and
  placing their points by hand (L-12), which is done on the same page -
  the sliders fade and the buttons become cancel and done. With no face
  found the page says so and offers the brush instead.
* **Brush** - a line saying how to paint, the six tools with a sentence on
  the one chosen, and the brush's size and strength.

An action row runs along the bottom. The panel only says what was chosen:
the controller keeps the strokes and the sliders with the photo, and the
photo's overlays paint them.
"""

from __future__ import annotations

from typing import Final

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
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
from ui.widgets.segmented import Segmented
from ui.widgets.slider_row import SliderEnds, SliderRow, SliderSpec
from ui.widgets.text import TranslatedLabel

#: The two pages, in the order the switch shows them.
PAGE_FACE: Final[str] = "face"
PAGE_BRUSH: Final[str] = "brush"

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
#: The face sliders in their two groups: the face's shape, then its
#: features, the shape first since it is what is reached for most.
FACE_GROUPS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("liquify.group.shape", ("face_width", "chin")),
    ("liquify.group.features", ("eyes", "nose", "mouth")),
)

#: The tools in the rows the panel lays them out in (PLAN.md 3.16). The two
#: twirls are one button: clockwise, and anticlockwise with Alt held.
TOOL_ROWS: Final[tuple[tuple[str, ...], ...]] = (
    (liquify.TOOL_PUSH, liquify.TOOL_BLOAT, liquify.TOOL_PUCKER),
    (liquify.TOOL_TWIRL_CW, liquify.TOOL_RECONSTRUCT, liquify.TOOL_SMOOTH),
)
#: The text key each tool is named and described by.
TOOL_KEYS: Final[dict[str, str]] = {
    liquify.TOOL_PUSH: "liquify.tool.push",
    liquify.TOOL_BLOAT: "liquify.tool.bloat",
    liquify.TOOL_PUCKER: "liquify.tool.pucker",
    liquify.TOOL_TWIRL_CW: "liquify.tool.twirl",
    liquify.TOOL_RECONSTRUCT: "liquify.tool.reconstruct",
    liquify.TOOL_SMOOTH: "liquify.tool.smooth",
}
#: The tool the brush starts as.
DEFAULT_TOOL: Final[str] = liquify.TOOL_PUSH


class ToolChoice(QPushButton):
    """One of the brush's tools: pressed while the brush is that tool."""

    def __init__(self, tool: str, parent: QWidget) -> None:
        super().__init__(tr(TOOL_KEYS[tool]), parent)
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
        self.setText(tr(TOOL_KEYS[self.tool]))
        self.setToolTip(tooltip(f"{TOOL_KEYS[self.tool]}.tip"))


def _half_button(key: str, parent: QWidget) -> TextButton:
    """Return a text button that takes its width from its row, not its words."""
    button = TextButton(key, parent)
    button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    return button


def _caption(parent: QWidget, primary: bool = False) -> QLabel:
    """Return a wrapping line of small text, in the primary colour if asked."""
    label = QLabel("", parent)
    if not primary:
        label.setProperty("role", "caption")
    label.setWordWrap(True)
    return label


class LiquifyPanel(QWidget):
    """Liquify's two pages, the face sliders and the brush, in the inspector's place."""

    #: The other page was chosen.
    page_changed = Signal(str)
    #: Another tool was chosen.
    tool_changed = Signal(str)
    #: The brush size changed: the radius as a fraction of the long edge,
    #: and whether a drag is still going on.
    brush_size_set = Signal(float, bool)
    #: One face slider moved: its name, its value, and whether a drag is
    #: still going on.
    face_value_set = Signal(str, float, bool)
    #: The pointer came onto a face slider (its name) or left them all
    #: (an empty name), so the photo can show where it works.
    face_hovered = Signal(str)
    #: Placing the face's points by hand was started or finished (L-12).
    editing_toggled = Signal(bool)
    #: Placing the face's points was called off: they go back as they were.
    editing_cancelled = Signal()
    #: The face's points put back where they were found (L-12).
    face_marks_reset_requested = Signal()
    #: The face sliders put back to rest, the strokes left alone.
    face_reset_requested = Signal()
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
        self._faces_edited = False
        self._editing = False
        self.face_rows: dict[str, SliderRow] = {}
        self.face_ends: dict[str, SliderEnds] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_title())
        content = QWidget(self)
        body = QVBoxLayout(content)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_switch(content))
        self.face_page = self._build_face_page(content)
        self.brush_page = self._build_brush_page(content)
        body.addWidget(self.face_page)
        body.addWidget(self.brush_page)
        body.addStretch(1)
        outer.addWidget(scrolled(content, "TabPage", self), 1)
        outer.addWidget(self._build_action_row())

        self.set_tool(DEFAULT_TOOL)
        self.set_brush_size(self._brush_size)
        self.set_strength(self._strength)
        self.set_page(PAGE_FACE)
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

    def _build_switch(self, parent: QWidget) -> QWidget:
        """The two pages to choose between, across the top."""
        holder = QWidget(parent)
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(
            theme.PANEL_PADDING, theme.SPACE_MD, theme.PANEL_PADDING, theme.SPACE_XS
        )
        self.page_switch = Segmented(
            ((PAGE_FACE, "liquify.page.face"), (PAGE_BRUSH, "liquify.page.brush")), holder
        )
        self.page_switch.chosen.connect(self._on_page)
        layout.addWidget(self.page_switch)
        return holder

    def _build_face_page(self, parent: QWidget) -> QWidget:
        """The face sliders, how many faces they work on, and what to do without one."""
        page = QWidget(parent)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # How many faces.
        status = QWidget(page)
        row = QHBoxLayout(status)
        row.setContentsMargins(theme.PANEL_PADDING, theme.SPACE_XS, theme.PANEL_PADDING, 0)
        self.faces_caption = _caption(status)
        row.addWidget(self.faces_caption, 1)
        layout.addWidget(status)
        self._face_status = status

        # The sliders, in their two groups, each with what its ends do.
        self._face_groups: list[ControlGroup] = []
        for index, (title, names) in enumerate(FACE_GROUPS):
            group = ControlGroup(title, first=index == 0, parent=page)
            self._groups.append(group)
            self._face_groups.append(group)
            for name in names:
                slider = SliderRow(FACE_SPECS[name], group)
                slider.value_changed.connect(self._on_face_row)
                slider.installEventFilter(self)
                self.face_rows[name] = slider
                group.add_row(slider)
                ends = SliderEnds(
                    (f"liquify.face.{name}.low", f"liquify.face.{name}.high"), parent=group
                )
                self.face_ends[name] = ends
                group.add_row(ends)
            layout.addWidget(group)

        # Under the sliders: what they work on, and two buttons that act on
        # the faces. Placing the points by hand (L-12) happens on this same
        # page: the sliders stay, faded, and the two buttons become the way
        # out - cancel, and done on the right, where the panel's own is.
        footer = QWidget(page)
        column = QVBoxLayout(footer)
        column.setContentsMargins(
            theme.PANEL_PADDING, theme.SPACE_SM, theme.PANEL_PADDING, theme.SPACE_SM
        )
        column.setSpacing(theme.SPACE_SM)
        self.every_face_caption = _caption(footer)
        column.addWidget(self.every_face_caption)
        self.editing_caption = _caption(footer, primary=True)
        column.addWidget(self.editing_caption)
        self.reset_faces_button = _half_button("liquify.reset_faces", footer)
        self.reset_faces_button.clicked.connect(self.face_marks_reset_requested.emit)
        self.reset_face_button = _half_button("liquify.reset_face", footer)
        self.reset_face_button.clicked.connect(self.face_reset_requested.emit)
        self.edit_faces_button = _half_button("liquify.edit_faces", footer)
        self.edit_faces_button.clicked.connect(lambda: self._on_editing(True))
        self.cancel_editing_button = _half_button("liquify.edit_faces.cancel", footer)
        self.cancel_editing_button.clicked.connect(self._on_cancel_editing)
        self.finish_editing_button = _half_button("liquify.edit_faces.done", footer)
        self.finish_editing_button.clicked.connect(lambda: self._on_editing(False))
        self._tooltips()
        # Two even columns: back to found alone on the row above, as wide as
        # the buttons under it; each pair shares its cells with the other,
        # one pair shown at a time.
        buttons = QGridLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setHorizontalSpacing(theme.SPACE_SM)
        buttons.setVerticalSpacing(theme.SPACE_SM)
        buttons.setColumnStretch(0, 1)
        buttons.setColumnStretch(1, 1)
        buttons.addWidget(self.reset_faces_button, 0, 0)
        buttons.addWidget(self.reset_face_button, 1, 0)
        buttons.addWidget(self.cancel_editing_button, 1, 0)
        buttons.addWidget(self.edit_faces_button, 1, 1)
        buttons.addWidget(self.finish_editing_button, 1, 1)
        column.addLayout(buttons)
        layout.addWidget(footer)
        self._face_footer = footer

        # With no face found: why that may be, and the brush instead.
        missing = QWidget(page)
        column = QVBoxLayout(missing)
        column.setContentsMargins(
            theme.PANEL_PADDING, theme.SPACE_SM, theme.PANEL_PADDING, theme.SPACE_SM
        )
        column.setSpacing(theme.SPACE_XS)
        self.no_face_title = _caption(missing, primary=True)
        column.addWidget(self.no_face_title)
        self.no_face_caption = _caption(missing)
        column.addWidget(self.no_face_caption)
        column.addSpacing(theme.SPACE_XS)
        self.use_brush_button = TextButton("liquify.use_brush", missing)
        self.use_brush_button.clicked.connect(lambda: self._on_page(PAGE_BRUSH))
        column.addWidget(self.use_brush_button)
        layout.addWidget(missing)
        self._face_missing = missing
        return page

    def _build_brush_page(self, parent: QWidget) -> QWidget:
        """How to paint, the tools with a sentence on the one chosen, and the brush."""
        page = QWidget(parent)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        hint = QWidget(page)
        row = QHBoxLayout(hint)
        row.setContentsMargins(theme.PANEL_PADDING, theme.SPACE_XS, theme.PANEL_PADDING, 0)
        self.brush_hint = _caption(hint)
        row.addWidget(self.brush_hint)
        layout.addWidget(hint)

        group = ControlGroup("liquify.group.tools", first=True, parent=page)
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
            line = QHBoxLayout()
            line.setContentsMargins(0, 0, 0, 0)
            line.setSpacing(theme.SPACE_SM)
            for tool in tools:
                button = ToolChoice(tool, group)
                button.clicked.connect(lambda _checked=False, name=tool: self._on_choice(name))
                self._choices.addButton(button)
                self.tool_buttons[tool] = button
                line.addWidget(button, 1)
            rows.addLayout(line)
        group.add_layout(rows)
        #: What the chosen tool does, so nobody has to hover to find out.
        self.tool_caption = _caption(group, primary=True)
        group.add_row(self.tool_caption)
        layout.addWidget(group)

        group = ControlGroup("liquify.group.brush", parent=page)
        self._groups.append(group)
        self.brush_size_row = SliderRow(BRUSH_SIZE_SPEC, group)
        self.brush_size_row.value_changed.connect(self._on_brush_size)
        group.add_row(self.brush_size_row)
        self.strength_row = SliderRow(STRENGTH_SPEC, group)
        self.strength_row.value_changed.connect(self._on_strength)
        group.add_row(self.strength_row)
        #: How many strokes there are.
        self.strokes_caption = _caption(group)
        group.add_row(self.strokes_caption)
        layout.addWidget(group)
        return page

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
    def page(self) -> str:
        """Return the page on show: :data:`PAGE_FACE` or :data:`PAGE_BRUSH`."""
        return self.page_switch.value

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

    @property
    def editing(self) -> bool:
        """Return whether the face's points are being placed by hand."""
        return self._editing

    def groups(self) -> tuple[ControlGroup, ...]:
        """Return the groups of both pages, in display order."""
        return tuple(self._groups)

    # -- being told things -----------------------------------------------

    def set_page(self, page: str) -> None:
        """Show *page* without announcing it."""
        self.page_switch.set_value(page)
        self.face_page.setVisible(self.page == PAGE_FACE)
        self.brush_page.setVisible(self.page == PAGE_BRUSH)

    def set_tool(self, tool: str) -> None:
        """Press the button of *tool* without announcing it."""
        button = self.tool_buttons.get(tool, self.tool_buttons[DEFAULT_TOOL])
        button.setChecked(True)
        self.tool_caption.setText(tr(f"{TOOL_KEYS[button.tool]}.tip"))

    def set_strokes(self, count: int) -> None:
        """Say how many strokes the photo has."""
        self._strokes = max(0, count)
        self._show_strokes()

    def set_face_shape(self, face: liquify.FaceShape) -> None:
        """Show the photo's face sliders without announcing them."""
        self._face = face
        self._faces_edited = any(marks.edited for marks in face.faces)
        for name, row in self.face_rows.items():
            row.blockSignals(True)
            row.set_value(face.value(name))
            row.blockSignals(False)
        self._show_strokes()
        self._show_faces()

    def set_faces(self, count: int | None) -> None:
        """Say how many faces the photo holds, or ``None`` while looking."""
        self._faces = count
        self._show_faces()

    def set_editing(self, editing: bool) -> None:
        """Show the page for placing the points, or the sliders, without announcing it."""
        self._editing = editing
        self._show_faces()

    def set_faces_edited(self, edited: bool) -> None:
        """Say whether any face's points were placed by hand."""
        self._faces_edited = edited
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
        """Read every word again after a language change (U-02)."""
        self.page_switch.retranslate()
        self.brush_size_row.set_spec(BRUSH_SIZE_SPEC)
        self.strength_row.set_spec(STRENGTH_SPEC)
        self.set_brush_size(self._brush_size)
        self.set_strength(self._strength)
        for button in self.tool_buttons.values():
            button.retranslate()
        self.set_tool(self.tool)
        for button in (
            self.reset_button,
            self.leave_button,
            self.reset_face_button,
            self.edit_faces_button,
            self.cancel_editing_button,
            self.finish_editing_button,
            self.reset_faces_button,
            self.use_brush_button,
        ):
            button.retranslate()
        self.leave_button.setToolTip(tooltip("liquify.leave", "shortcut.liquify"))
        self._tooltips()
        for name, row in self.face_rows.items():
            row.set_spec(FACE_SPECS[name])
        for ends in self.face_ends.values():
            ends.retranslate()
        self.set_face_shape(self._face)
        self._show_strokes()
        self._show_faces()

    def _tooltips(self) -> None:
        """Give the face page's buttons their tooltips, with the keys that do the same."""
        self.edit_faces_button.setToolTip(tooltip("liquify.edit_faces.tip"))
        self.reset_faces_button.setToolTip(tooltip("liquify.reset_faces.tip"))
        self.cancel_editing_button.setToolTip(
            tooltip("liquify.edit_faces.cancel.tip", "liquify.edit_faces.keys.cancel")
        )
        self.finish_editing_button.setToolTip(
            tooltip("liquify.edit_faces.done.tip", "liquify.edit_faces.keys.done")
        )

    def _show_faces(self) -> None:
        """Show the sliders, the page for placing the points, or the page for no face."""
        count = self._faces
        if count is None:
            caption = tr("adjust.faces.looking")
        elif count == 1:
            caption = tr("adjust.faces.one")
        else:
            caption = tr("adjust.faces.many", count=count)
        self.faces_caption.setText(caption)
        # Faces kept with the sliders still have points to place before the
        # search has finished again (L-12).
        editing = self._editing
        missing = count == 0 and not self._face.faces and not editing
        self._face_status.setVisible(not missing)
        for group in self._face_groups:
            group.setVisible(not missing)
        self._face_footer.setVisible(not missing)
        self._face_missing.setVisible(missing)
        # While the points are placed the photo is shown without liquify,
        # so a slider moved then would change nothing to be seen.
        usable = (bool(count) or bool(self._face.faces)) and not editing
        for row in self.face_rows.values():
            row.setEnabled(usable)
        for ends in self.face_ends.values():
            ends.setEnabled(usable)
        self.every_face_caption.setVisible(not editing)
        self.editing_caption.setVisible(editing)
        self.reset_faces_button.setVisible(editing)
        self.reset_face_button.setVisible(not editing)
        self.edit_faces_button.setVisible(not editing)
        self.cancel_editing_button.setVisible(editing)
        self.finish_editing_button.setVisible(editing)
        self.edit_faces_button.setEnabled(bool(count) or bool(self._face.faces))
        self.reset_face_button.setEnabled(not self._face.is_default)
        self.reset_faces_button.setEnabled(self._faces_edited)
        self.every_face_caption.setText(tr("liquify.every_face"))
        self.editing_caption.setText(tr("liquify.editing_hint"))
        self.no_face_title.setText(tr("liquify.faces.none"))
        self.no_face_caption.setText(tr("liquify.faces.none.why"))
        self.brush_hint.setText(tr("liquify.hint"))

    def _show_strokes(self) -> None:
        count = self._strokes
        if count == 0:
            caption = tr("liquify.strokes.none")
        elif count == 1:
            caption = tr("liquify.strokes.one")
        else:
            caption = tr("liquify.strokes.many", count=count)
        self.strokes_caption.setText(caption)
        self.reset_button.setEnabled(count > 0 or not self._face.is_default)

    # -- reacting --------------------------------------------------------

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt naming
        """Say which face slider the pointer is over, so the photo can show where it works."""
        if self._editing:
            # The sliders are put down while the points are placed.
            pass
        elif event.type() == QEvent.Type.Enter:
            for name, row in self.face_rows.items():
                if row is watched:
                    self.face_hovered.emit(name)
        elif event.type() == QEvent.Type.Leave and watched in self.face_rows.values():
            self.face_hovered.emit("")
        return super().eventFilter(watched, event)

    def _on_page(self, page: str) -> None:
        self.set_page(page)
        self.page_changed.emit(page)

    def _on_choice(self, tool: str) -> None:
        self.set_tool(tool)
        self.tool_changed.emit(tool)

    def _on_brush_size(self, _name: str, value: float, dragging: bool) -> None:
        self._brush_size = liquify.clamp_radius(value / BRUSH_SIZE_SCALE)
        self._set_titles_dimmed(dragging)
        self.brush_size_set.emit(self._brush_size, dragging)

    def _on_editing(self, editing: bool) -> None:
        self.set_editing(editing)
        self.editing_toggled.emit(editing)

    def _on_cancel_editing(self) -> None:
        self.set_editing(False)
        self.editing_cancelled.emit()

    def _on_face_row(self, name: str, value: float, dragging: bool) -> None:
        self._face = self._face.with_value(name, value)
        self._show_strokes()
        self.reset_face_button.setEnabled(not self._face.is_default)
        self._set_titles_dimmed(dragging)
        self.face_value_set.emit(name, value, dragging)

    def _on_strength(self, _name: str, value: float, dragging: bool) -> None:
        self._strength = liquify.clamp_strength(value / STRENGTH_SCALE)
        self._set_titles_dimmed(dragging)

    def _set_titles_dimmed(self, dimmed: bool) -> None:
        """Fade the group titles while a slider is moving (PLAN.md 3.6)."""
        for group in self._groups:
            group.header.set_faded(dimmed)
