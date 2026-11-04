"""The adjust tab: the saved presets, then the sliders in five groups (PLAN.md 2.2 and 3.1).

The presets come first: a preset is a set of slider values, so clicking one
moves the sliders right underneath it, where what it changed can be seen and
adjusted further. Its header carries the button that saves a new one.

White balance, light, colour, effects and detail follow, each under a header
that folds its group away. The light header carries the wand that sets the
light sliders from the photo itself. The white balance pair changes range
depending on the photo: a RAW file is adjusted in kelvin from where the
camera was, a JPEG only relatively from where it already is. The white
balance header also carries the eyedropper (A-24), which sets the pair from
a spot on the photo.

Portrait and lens correction are on the retouch tab (``ui/retouch_tab.py``):
neither is wanted on every photo.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import QMenu, QVBoxLayout, QWidget

from app.i18n import tr
from core import color_management as cm
from core import edit_state
from core.edit_state import EditState
from core.presets import Preset
from ui import icons, theme
from ui.presets import NameDialog, PresetRow
from ui.theme import MARK_BLUE, MARK_GREEN, MARK_MAGENTA, MARK_YELLOW
from ui.widgets.buttons import IconButton
from ui.widgets.group import ControlGroup
from ui.widgets.slider_row import SliderRow, SliderSpec
from ui.widgets.text import empty_caption

#: The five groups of sliders and what is in each (PLAN.md section 3.1).
GROUPS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("adjust.group.white_balance", ("temperature", "tint")),
    (
        "adjust.group.light",
        ("exposure", "contrast", "highlights", "shadows", "whites", "blacks"),
    ),
    ("adjust.group.colour", ("vibrance", "saturation")),
    ("adjust.group.effects", ("vignette", "vignette_midpoint", "grain")),
    (
        "adjust.group.detail",
        ("sharpening", "sharpen_masking", "luminance_noise", "colour_noise"),
    ),
)


def _amount(
    name: str, label_key: str, maximum: float = edit_state.DETAIL_MAX, default: float = 0.0
) -> SliderSpec:
    """Return a slider that runs up from nothing, like the detail group's.

    It reads without a sign and fills from its low end, the way the
    vignette's midpoint does: there is no other way for it to go.
    """
    return SliderSpec(
        name=name,
        label_key=label_key,
        minimum=edit_state.DETAIL_MIN,
        maximum=maximum,
        default=default,
        signed=False,
        fill_from=edit_state.DETAIL_MIN,
    )


def _sharpening(default: float = 0.0) -> SliderSpec:
    """Return the sharpening slider: softens below zero, sharpens above.

    It reads with a sign like the bipolar sliders, and its track fills from
    zero rather than from where a RAW photo rests, so that which way it
    points - sharper or softer - is what the fill shows.
    """
    return SliderSpec(
        name="sharpening",
        label_key="adjust.sharpening",
        minimum=edit_state.SHARPENING_MIN,
        maximum=edit_state.SHARPENING_MAX,
        default=default,
        fill_from=0.0,
    )


def _relative(name: str, label_key: str) -> SliderSpec:
    return SliderSpec(
        name=name,
        label_key=label_key,
        minimum=edit_state.TONE_MIN,
        maximum=edit_state.TONE_MAX,
        default=0.0,
    )


#: How each slider reads for a JPEG, whose white balance is only relative.
JPEG_SPECS: Final[dict[str, SliderSpec]] = {
    "temperature": SliderSpec(
        name="temperature",
        label_key="adjust.temperature",
        minimum=edit_state.RELATIVE_MIN,
        maximum=edit_state.RELATIVE_MAX,
        default=0.0,
        marks=(MARK_BLUE, MARK_YELLOW),
    ),
    "tint": SliderSpec(
        name="tint",
        label_key="adjust.tint",
        minimum=edit_state.RELATIVE_MIN,
        maximum=edit_state.RELATIVE_MAX,
        default=0.0,
        marks=(MARK_GREEN, MARK_MAGENTA),
    ),
    "exposure": SliderSpec(
        name="exposure",
        label_key="adjust.exposure",
        minimum=edit_state.EXPOSURE_MIN_EV,
        maximum=edit_state.EXPOSURE_MAX_EV,
        default=0.0,
        decimals=2,
        step=edit_state.EXPOSURE_STEP_EV,
    ),
    "contrast": _relative("contrast", "adjust.contrast"),
    "highlights": _relative("highlights", "adjust.highlights"),
    "shadows": _relative("shadows", "adjust.shadows"),
    "whites": _relative("whites", "adjust.whites"),
    "blacks": _relative("blacks", "adjust.blacks"),
    "vibrance": _relative("vibrance", "adjust.vibrance"),
    "saturation": _relative("saturation", "adjust.saturation"),
    "vignette": _relative("vignette", "adjust.vignette"),
    "vignette_midpoint": SliderSpec(
        name="vignette_midpoint",
        label_key="adjust.vignette_midpoint",
        minimum=edit_state.MIDPOINT_MIN,
        maximum=edit_state.MIDPOINT_MAX,
        default=edit_state.MIDPOINT_DEFAULT,
        # How far out the vignette reaches is a size, not an amount either
        # way from a middle, so it reads without a sign and its track fills
        # from the low end.
        signed=False,
        fill_from=edit_state.MIDPOINT_MIN,
    ),
    "grain": _amount("grain", "adjust.grain", edit_state.GRAIN_MAX),
    "skin": _amount("skin", "adjust.skin", edit_state.SKIN_MAX),
    "whiten": _amount("whiten", "adjust.whiten", edit_state.WHITEN_MAX),
    "sharpening": _sharpening(),
    "sharpen_masking": _amount("sharpen_masking", "adjust.sharpen_masking"),
    "luminance_noise": _amount("luminance_noise", "adjust.luminance_noise"),
    "colour_noise": _amount("colour_noise", "adjust.colour_noise"),
}


def raw_specs(
    shot_temperature: float, shot_tint: float, in_kelvin: bool = True
) -> dict[str, SliderSpec]:
    """Return the sliders for a RAW photo, resting at what the camera used.

    With *in_kelvin* false the temperature is shown as a nudge either way
    from what the camera recorded instead, which is the other choice in the
    settings (PLAN.md 3.10). The photo still carries kelvin; only the slider
    reads differently.
    """
    specs = dict(JPEG_SPECS)
    specs["temperature"] = SliderSpec(
        name="temperature",
        label_key="adjust.temperature",
        minimum=edit_state.TEMPERATURE_MIN_K,
        maximum=edit_state.TEMPERATURE_MAX_K,
        default=shot_temperature,
        signed=False,
        reciprocal=True,
        suffix=tr("adjust.kelvin"),
        marks=(MARK_BLUE, MARK_YELLOW),
    ) if in_kelvin else SliderSpec(
        name="temperature",
        label_key="adjust.temperature",
        minimum=edit_state.RELATIVE_MIN,
        maximum=edit_state.RELATIVE_MAX,
        default=0.0,
        marks=(MARK_BLUE, MARK_YELLOW),
    )
    specs["tint"] = SliderSpec(
        name="tint",
        label_key="adjust.tint",
        minimum=edit_state.TINT_MIN,
        maximum=edit_state.TINT_MAX,
        default=shot_tint,
        marks=(MARK_GREEN, MARK_MAGENTA),
    )
    # A RAW rests with some sharpening and some colour noise taken out, so a
    # double click on either puts it back there rather than at zero.
    specs["sharpening"] = _sharpening(edit_state.RAW_SHARPENING_DEFAULT)
    specs["colour_noise"] = _amount(
        "colour_noise",
        "adjust.colour_noise",
        default=edit_state.RAW_COLOUR_NOISE_DEFAULT,
    )
    return specs


class AdjustTab(QWidget):
    """The sliders, and the state they add up to."""

    #: One slider moved: the adjustment, the value the photo should hold, and
    #: whether a drag is still going on. Only the one value travels, so that
    #: whatever else the photo holds - the filter chosen on its own tab, say -
    #: is taken from the photo and not from what this tab last heard.
    value_set = Signal(str, float, bool)
    #: A preset to put on (P-05).
    preset_applied = Signal(object)
    #: A preset to rename, with the name it should now have (P-06).
    preset_renamed = Signal(object, str)
    #: A preset to remove (P-06).
    preset_deleted = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = EditState()
        self._reference = EditState()
        self._rows: dict[str, SliderRow] = {}
        self._group_widgets: list[ControlGroup] = []
        self._preset_rows: list[PresetRow] = []
        self._shot: tuple[float, float] = (0.0, 0.0)
        #: Whether a RAW photo's temperature slider reads in kelvin
        #: (PLAN.md 3.10).
        self._in_kelvin = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)


        self._groups = QWidget(self)
        groups_layout = QVBoxLayout(self._groups)
        groups_layout.setContentsMargins(0, 0, 0, 0)
        groups_layout.setSpacing(0)
        groups_layout.addSpacing(theme.SPACE_SM)
        #: The white balance eyedropper (A-24). It stays pressed while the
        #: next click on the photo is waiting to be read; the window owns
        #: what the click does, since the photo is not in this panel.
        self.pick_button = IconButton(
            icons.ICON_PIPETTE, "adjust.pick_white_balance", "shortcut.white_balance_pick"
        )
        self.pick_button.setCheckable(True)
        #: The automatic light: sets the light group's sliders from the photo.
        self.auto_button = IconButton(
            icons.ICON_WAND, "adjust.auto_light", "shortcut.auto_light"
        )
        groups_layout.addWidget(self._build_presets_group())
        for title_key, names in GROUPS:
            group = ControlGroup(title_key, parent=self._groups)
            if title_key == "adjust.group.white_balance":
                group.header.set_action(self.pick_button)
            elif title_key == "adjust.group.light":
                group.header.set_action(self.auto_button)
            self._group_widgets.append(group)
            for name in names:
                row = SliderRow(JPEG_SPECS[name])
                row.value_changed.connect(self._on_row_changed)
                self._rows[name] = row
                group.add_row(row)
            groups_layout.addWidget(group)
        layout.addWidget(self._groups)
        layout.addStretch(1)

        self.set_photo(None)

    def _build_presets_group(self) -> ControlGroup:
        """The saved presets, with the button that saves one on the header."""
        group = ControlGroup("adjust.group.presets", first=True, parent=self._groups)
        self._group_widgets.append(group)
        #: Saves what the photo is set to now as a preset (P-04); the window
        #: asks for the name, since that is a dialog over the whole window.
        self.save_preset_button = IconButton(
            icons.ICON_PLUS, "panel.action.save_preset", "shortcut.new_preset"
        )
        group.header.set_action(self.save_preset_button)
        # Shown until there is a preset, so the first one can be found.
        self._no_presets = empty_caption("presets.empty", group, padded=False)
        # Read from the left like the rows it stands in for.
        self._no_presets.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        group.add_row(self._no_presets)
        self.presets_group = group
        return group

    # -- state -----------------------------------------------------------

    @property
    def state(self) -> EditState:
        """Return what the sliders add up to."""
        return self._state

    @property
    def reference(self) -> EditState:
        """Return the state the photo opened in, which reset goes back to."""
        return self._reference

    def set_photo(self, state: EditState | None, shot: tuple[float, float] = (0.0, 0.0)) -> None:
        """Show the adjustments of a photo, or the empty state.

        Changing photo rebuilds the two white balance sliders, because what
        they mean and where they rest depends on whether it is RAW.
        """
        self._groups.setVisible(state is not None)
        if state is None:
            self.pick_button.setChecked(False)
            return

        self._shot = shot
        self._state = state
        self._reference = EditState.for_photo(state.is_raw, *shot)
        self._rebuild_rows()

    def set_temperature_unit(self, in_kelvin: bool) -> None:
        """Show a RAW photo's temperature in kelvin, or as a nudge (3.10)."""
        if in_kelvin == self._in_kelvin:
            return
        self._in_kelvin = in_kelvin
        if self._state.is_raw:
            self._rebuild_rows()

    def retranslate(self) -> None:
        """Read the sliders again after a language change (U-02).

        Their descriptions are rebuilt rather than only their names, because
        the unit after a RAW photo's temperature is part of the description.
        """
        self._rebuild_rows()

    def _rebuild_rows(self) -> None:
        """Give every slider the description it should have, and its value."""
        specs = (
            raw_specs(*self._shot, in_kelvin=self._in_kelvin)
            if self._state.is_raw
            else JPEG_SPECS
        )
        for name, row in self._rows.items():
            row.set_spec(specs[name])
            row.blockSignals(True)
            row.set_value(self._shown(name, getattr(self._state, name)))
            row.blockSignals(False)
        self._update_masking()

    def set_state(self, state: EditState) -> None:
        """Show *state* without announcing it, as undo and presets do."""
        self._state = state
        for name, row in self._rows.items():
            row.blockSignals(True)
            row.set_value(self._shown(name, getattr(state, name)))
            row.blockSignals(False)
        self._update_masking()

    def _update_masking(self) -> None:
        """Let the masking be moved only while there is sharpening to mask.

        At zero there is nothing to keep off the flat areas, and below zero
        the photo is softened, which the masking does not take part in.
        """
        self._rows["sharpen_masking"].setEnabled(self._state.sharpening > 0.0)

    def _shown(self, name: str, value: float) -> float:
        """Return what the slider for *name* should show for *value*."""
        if name != "temperature" or self._in_kelvin or not self._state.is_raw:
            return value
        return cm.kelvin_to_relative(
            value, self._shot[0], edit_state.TEMPERATURE_MIN_K, edit_state.TEMPERATURE_MAX_K
        )

    def _held(self, name: str, value: float) -> float:
        """Return what the photo should hold for a slider showing *value*."""
        if name != "temperature" or self._in_kelvin or not self._state.is_raw:
            return value
        return cm.relative_to_kelvin(
            value, self._shot[0], edit_state.TEMPERATURE_MIN_K, edit_state.TEMPERATURE_MAX_K
        )

    def rows(self) -> dict[str, SliderRow]:
        """Return the sliders by adjustment name."""
        return dict(self._rows)

    def groups(self) -> tuple[ControlGroup, ...]:
        """Return the collapsible groups, in display order."""
        return tuple(self._group_widgets)

    # -- the saved presets (P-04 to P-06) ---------------------------------

    def set_presets(self, presets: Sequence[Preset]) -> None:
        """Show the saved presets, or the line that says how to save one."""
        for row in self._preset_rows:
            self.presets_group.remove_row(row)
            row.deleteLater()
        self._preset_rows = []
        for index, preset in enumerate(presets):
            row = PresetRow(preset, self.presets_group)
            row.clicked.connect(
                lambda _checked=False, item=preset: self.preset_applied.emit(item)
            )
            row.menu_requested.connect(self._show_preset_menu)
            self.presets_group.insert_row(index, row)
            self._preset_rows.append(row)
        self._no_presets.setVisible(not presets)

    def preset_rows(self) -> list[PresetRow]:
        """Return the preset rows in display order."""
        return list(self._preset_rows)

    def preset_row(self, preset_id: str) -> PresetRow | None:
        """Return the row of the preset *preset_id*, opening its group if folded."""
        for row in self._preset_rows:
            if row.preset.id == preset_id:
                self.presets_group.header.setChecked(True)
                return row
        return None

    def _show_preset_menu(self, preset: Preset, where: QPoint) -> None:
        """Offer to rename or delete this preset (P-06)."""
        menu = QMenu(self)
        rename = menu.addAction(tr("presets.rename"))
        remove = menu.addAction(tr("presets.delete"))
        chosen = menu.exec(where)
        if chosen is rename:
            name = NameDialog.ask("presets.rename", preset.name, self)
            if name:
                self.preset_renamed.emit(preset, name)
        elif chosen is remove:
            self.preset_deleted.emit(preset)

    # -- reacting --------------------------------------------------------

    def _on_row_changed(self, name: str, value: float, dragging: bool) -> None:
        held = self._held(name, value)
        self._state = self._state.with_value(name, held)
        if name == "sharpening":
            self._update_masking()

        self._set_titles_dimmed(dragging)
        self.value_set.emit(name, held, dragging)

    def _set_titles_dimmed(self, dimmed: bool) -> None:
        """Fade the group titles while a slider is moving (PLAN.md 3.6)."""
        for group in self._group_widgets:
            group.header.set_faded(dimmed)
