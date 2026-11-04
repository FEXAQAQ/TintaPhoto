"""The crop tab of the right hand panel (PLAN.md 3.2).

Three groups, top to bottom, each under a header that folds it away: the
seven ratios in three columns, the angle slider, and the orientation tools - straighten
and the quarter turns on one row, the two flips under them. The tab only says what
the user asked for; what that does to the crop box is worked out by
:mod:`core.geometry` and applied by the controller.
"""

from __future__ import annotations

from typing import Final

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.i18n import tr
from core import geometry
from core.geometry import Geometry
from ui import icons, theme
from ui.widgets.buttons import ROLE_CHOICE, TextButton
from ui.widgets.group import ControlGroup
from ui.widgets.slider_row import SliderRow, SliderSpec

#: How the ratios are laid out (PLAN.md 3.2: three columns).
_COLUMNS: Final[int] = 3

#: The text key of each ratio button.
_ASPECT_LABELS: Final[dict[str, str]] = {
    geometry.ASPECT_ORIGINAL: "crop.aspect.original",
    geometry.ASPECT_FREE: "crop.aspect.free",
    geometry.ASPECT_SQUARE: "crop.aspect.square",
    geometry.ASPECT_3_2: "crop.aspect.3_2",
    geometry.ASPECT_4_5: "crop.aspect.4_5",
    geometry.ASPECT_16_9: "crop.aspect.16_9",
    geometry.ASPECT_9_16: "crop.aspect.9_16",
}

#: PLAN.md 3.2 turns the chosen ratio on its side when it is clicked again,
#: but these three have nothing to turn: "original" is the name of the whole
#: picture, "free" has no shape, and a square is the same either way.
#: Clicking one of them again simply applies it afresh. X still turns the
#: crop over, whichever ratio it is.
_RECLICK_APPLIES_AGAIN: Final[frozenset[str]] = frozenset(
    {geometry.ASPECT_ORIGINAL, geometry.ASPECT_FREE, geometry.ASPECT_SQUARE}
)


def angle_spec() -> SliderSpec:
    """Return the rotation slider, the same component as the adjust tab (C-03)."""
    return SliderSpec(
        name="angle",
        label_key="crop.angle",
        minimum=geometry.ANGLE_MIN,
        maximum=geometry.ANGLE_MAX,
        default=0.0,
        decimals=1,
        suffix=tr("crop.degrees"),
    )


class CropTab(QWidget):
    """The ratios, the angle and the orientation tools."""

    #: A ratio was chosen, by its name in :data:`core.geometry.ASPECTS`.
    aspect_chosen = Signal(str)
    #: The chosen ratio was clicked again, or X was pressed (PLAN.md 3.2).
    orientation_flipped = Signal()
    #: A new angle, and whether the slider is still being dragged.
    angle_changed = Signal(float, bool)
    #: The straighten tool was armed or disarmed (C-04).
    straighten_toggled = Signal(bool)
    #: A quarter turn, -1 to the left and +1 to the right (C-06).
    turn_requested = Signal(int)
    #: A flip, top to bottom when true and left to right when false (C-08).
    mirror_requested = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._geometry = Geometry()
        self._aspect_buttons: dict[str, QPushButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)


        self._content = QWidget(self)
        content = QVBoxLayout(self._content)
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)
        content.addSpacing(theme.SPACE_SM)
        self._group_widgets = (
            self._build_aspects(),
            self._build_angle(),
            self._build_orientation(),
        )
        for group in self._group_widgets:
            content.addWidget(group)
        layout.addWidget(self._content)
        layout.addStretch(1)

        self.set_photo(None)

    # -- building --------------------------------------------------------

    def _build_aspects(self) -> ControlGroup:
        """Build the ratio buttons, three to a row."""
        group = ControlGroup("crop.group.aspect", first=True, parent=self)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(theme.SPACE_XS)
        self._aspect_group = QButtonGroup(self)
        self._aspect_group.setExclusive(True)
        for index, aspect in enumerate(geometry.ASPECTS):
            button = QPushButton(tr(_ASPECT_LABELS[aspect]))
            button.setProperty("role", ROLE_CHOICE)
            button.setCheckable(True)
            button.setFixedHeight(theme.BUTTON_HEIGHT)
            button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
            button.clicked.connect(lambda _checked, name=aspect: self._on_aspect(name))
            self._aspect_group.addButton(button, index)
            self._aspect_buttons[aspect] = button
            grid.addWidget(button, index // _COLUMNS, index % _COLUMNS)
        group.add_layout(grid)
        return group

    def _build_angle(self) -> ControlGroup:
        """Build the rotation slider (C-03)."""
        group = ControlGroup("crop.group.angle", parent=self)

        self.angle_row = SliderRow(angle_spec())
        self.angle_row.value_changed.connect(
            lambda _name, value, dragging: self.angle_changed.emit(value, dragging)
        )
        group.add_row(self.angle_row)
        return group

    def _build_orientation(self) -> ControlGroup:
        """Build the straighten, quarter turn and flip buttons (C-04, C-06, C-08)."""
        group = ControlGroup("crop.group.orientation", parent=self)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACE_XS)

        self.straighten_button = TextButton("crop.straighten")
        self.straighten_button.setCheckable(True)
        self.straighten_button.setProperty("role", ROLE_CHOICE)
        self.straighten_button.setToolTip(tr("crop.straighten.hint"))
        self.straighten_button.toggled.connect(self.straighten_toggled)

        self.turn_left_button = TextButton("crop.turn_left")
        self.turn_left_button.setIcon(icons.load_icon(icons.ICON_TURN_LEFT))
        self.turn_left_button.clicked.connect(lambda: self.turn_requested.emit(-1))

        self.turn_right_button = TextButton("crop.turn_right")
        self.turn_right_button.setIcon(icons.load_icon(icons.ICON_TURN_RIGHT))
        self.turn_right_button.clicked.connect(lambda: self.turn_requested.emit(1))

        for button in (self.straighten_button, self.turn_left_button, self.turn_right_button):
            row.addWidget(button, 1)
        group.add_layout(row)

        # Five buttons do not fit one row of the panel, so the flips have
        # one of their own, under the turns they belong with.
        flips = QHBoxLayout()
        flips.setContentsMargins(0, 0, 0, 0)
        flips.setSpacing(theme.SPACE_XS)

        self.mirror_horizontal_button = TextButton("crop.mirror_horizontal")
        self.mirror_horizontal_button.setIcon(icons.load_icon(icons.ICON_FLIP_HORIZONTAL))
        self.mirror_horizontal_button.clicked.connect(
            lambda: self.mirror_requested.emit(False)
        )

        self.mirror_vertical_button = TextButton("crop.mirror_vertical")
        self.mirror_vertical_button.setIcon(icons.load_icon(icons.ICON_FLIP_VERTICAL))
        self.mirror_vertical_button.clicked.connect(lambda: self.mirror_requested.emit(True))

        for button in (self.mirror_horizontal_button, self.mirror_vertical_button):
            # Halves of the row whatever the labels say: "Flip horizontal"
            # would otherwise push its neighbour narrower.
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            flips.addWidget(button, 1)
        group.add_layout(flips)
        return group

    def retranslate(self) -> None:
        """Read the ratios, the slider and the orientation tools again (U-02).

        The slider gets a whole fresh description, because the degree sign
        after its number is part of it.
        """
        for aspect, button in self._aspect_buttons.items():
            button.setText(tr(_ASPECT_LABELS[aspect]))
        self.angle_row.set_spec(angle_spec())
        self.straighten_button.setToolTip(tr("crop.straighten.hint"))

    # -- state -----------------------------------------------------------

    def set_photo(self, geometry_now: Geometry | None) -> None:
        """Show the geometry of a photo, or the empty state."""
        self._content.setVisible(geometry_now is not None)
        if geometry_now is None:
            return
        self.set_geometry(geometry_now)

    def set_geometry(self, geometry_now: Geometry) -> None:
        """Show *geometry_now* without announcing it again."""
        self._geometry = geometry_now
        button = self._aspect_buttons.get(geometry_now.aspect)
        if button is not None:
            button.setChecked(True)
        self.angle_row.blockSignals(True)
        self.angle_row.set_value(geometry_now.angle)
        self.angle_row.blockSignals(False)

    def set_straightening(self, armed: bool) -> None:
        """Show whether the straighten tool is armed, without announcing it."""
        self.straighten_button.blockSignals(True)
        self.straighten_button.setChecked(armed)
        self.straighten_button.blockSignals(False)

    def groups(self) -> tuple[ControlGroup, ...]:
        """Return the three collapsible groups, in display order."""
        return self._group_widgets

    def aspect_buttons(self) -> dict[str, QPushButton]:
        """Return the ratio buttons by the name of their ratio."""
        return dict(self._aspect_buttons)

    def flip_orientation(self) -> None:
        """Turn the chosen ratio on its side (X, PLAN.md 3.2)."""
        self.orientation_flipped.emit()

    # -- reacting --------------------------------------------------------

    def _on_aspect(self, aspect: str) -> None:
        """Choose a ratio, or turn the chosen one on its side (PLAN.md 3.2).

        Three of them have nothing to turn over, so clicking them again
        applies them afresh instead: pressing "original" twice gives the
        whole picture both times rather than standing it on its side.
        """
        if aspect == self._geometry.aspect and aspect not in _RECLICK_APPLIES_AGAIN:
            self.orientation_flipped.emit()
            return
        self.aspect_chosen.emit(aspect)
