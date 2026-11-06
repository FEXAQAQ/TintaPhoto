"""The retouch tab: spot removal and lens correction (PLAN.md 2.2, 3.1).

Both take out a flaw rather than set the look of the photo, and neither is
wanted on every photo: spots are mended where there is dust or a blemish,
and the lens correction is set once for a lens and left. So they sit on a
tab of their own rather than lengthening the adjust tab, whose groups are
the ones every photo goes through. The portrait sliders were here too until
they got a mode of their own, with the brush that shapes the skin they work
on (PLAN.md 3.15).

The spot removal group (A-32) holds the tool's button on its header, the
size of the brush, a line saying how many spots are mended and a button
that clears them. The tool itself works on the photo, which is the window's.

The lens correction group (A-26 to A-28) is not part of the adjustments: it
belongs to the geometry, because it changes the frame the crop is measured
against, so it travels on a signal of its own.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.i18n import tr
from core import healing, lens
from core.edit_state import EditState
from core.lens import LensProfile, LensSettings
from ui import icons, theme
from ui.widgets.buttons import IconButton, TextButton
from ui.widgets.group import ControlGroup
from ui.widgets.slider_row import SliderRow, SliderSpec
from ui.widgets.switch import Switch
from ui.widgets.text import TranslatedLabel

def _lens_amount(name: str, label_key: str) -> SliderSpec:
    """Return one of the lens correction amounts: a percentage of the profile."""
    return SliderSpec(
        name=name,
        label_key=label_key,
        minimum=lens.AMOUNT_MIN,
        maximum=lens.AMOUNT_MAX,
        default=lens.AMOUNT_DEFAULT,
        signed=False,
        fill_from=lens.AMOUNT_MIN,
    )


#: The lens correction's two sliders (PLAN.md 2.2, A-27 and A-28), by the
#: field of :class:`core.lens.LensSettings` each one sets.
LENS_SPECS: Final[dict[str, SliderSpec]] = {
    "distortion": _lens_amount("distortion", "adjust.lens.distortion"),
    "vignetting": _lens_amount("vignetting", "adjust.lens.vignetting"),
}


#: The brush slider counts thousandths of the long edge: 1 is a speck of
#: sensor dust, 100 a tenth of the frame (:mod:`core.healing`).
SPOT_SIZE_SCALE: Final[float] = 1000.0
SPOT_SIZE_SPEC: Final[SliderSpec] = SliderSpec(
    name="spot_size",
    label_key="adjust.spots.size",
    minimum=healing.RADIUS_MIN * SPOT_SIZE_SCALE,
    maximum=healing.RADIUS_MAX * SPOT_SIZE_SCALE,
    default=healing.RADIUS_DEFAULT * SPOT_SIZE_SCALE,
    signed=False,
    fill_from=healing.RADIUS_MIN * SPOT_SIZE_SCALE,
)


class RetouchTab(QWidget):
    """Spot removal and the lens correction."""

    #: The lens correction changed (PLAN.md 2.2): the new
    #: :class:`core.lens.LensSettings` and whether a drag is still going on.
    lens_set = Signal(object, bool)
    #: The brush size changed (A-32): the radius as a fraction of the long
    #: edge, and whether a drag is still going on.
    spot_size_set = Signal(float, bool)
    #: The button that clears every mended spot was pressed.
    spots_cleared = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._group_widgets: list[ControlGroup] = []
        self._lens = LensSettings()
        self._lens_profile: LensProfile | None = None
        self._lens_raw = False
        #: How many spots are mended (A-32), and the size the slider shows.
        self._spot_count = 0
        self._spot_size = healing.RADIUS_DEFAULT

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._groups = QWidget(self)
        groups_layout = QVBoxLayout(self._groups)
        groups_layout.setContentsMargins(0, 0, 0, 0)
        groups_layout.setSpacing(0)
        groups_layout.addSpacing(theme.SPACE_SM)
        groups_layout.addWidget(self._build_spot_group())
        groups_layout.addWidget(self._build_lens_group())
        layout.addWidget(self._groups)
        layout.addStretch(1)

        self._show_spots()
        self.set_photo(None)

    def _build_spot_group(self) -> ControlGroup:
        """The spot removal tool: its button, its size and what it has done."""
        group = ControlGroup("adjust.group.spots", first=True, parent=self._groups)
        self._group_widgets.append(group)
        #: Takes up the tool, the way the eyedropper is taken up (A-24): it
        #: stays pressed while clicks on the photo mend spots.
        self.spot_button = IconButton(icons.ICON_SPOT, "adjust.spot_tool", "shortcut.spot_tool")
        self.spot_button.setCheckable(True)
        group.header.set_action(self.spot_button)

        self.spot_size_row = SliderRow(SPOT_SIZE_SPEC)
        self.spot_size_row.value_changed.connect(self._on_spot_size)
        group.add_row(self.spot_size_row)

        #: How many spots are mended, or how to mend the first.
        self.spots_caption = QLabel("", group)
        self.spots_caption.setProperty("role", "caption")
        self.spots_caption.setWordWrap(True)
        group.add_row(self.spots_caption)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.clear_spots_button = TextButton("adjust.spots.clear", group)
        self.clear_spots_button.clicked.connect(self.spots_cleared)
        row.addWidget(self.clear_spots_button)
        row.addStretch(1)
        group.add_layout(row)
        return group

    def _build_lens_group(self) -> ControlGroup:
        """The lens correction: a switch, which lens it is, and two amounts."""
        group = ControlGroup("adjust.group.lens", parent=self._groups)
        self._group_widgets.append(group)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACE_SM)
        self._lens_switch_label = TranslatedLabel("adjust.lens.enable", group)
        row.addWidget(self._lens_switch_label, 1)
        self.lens_switch = Switch(group)
        self.lens_switch.setToolTip(tr("adjust.lens.enable"))
        self.lens_switch.toggled.connect(self._on_lens_switched)
        row.addWidget(self.lens_switch)
        group.add_layout(row)

        #: Which lens the profile is for, or why there is none.
        self.lens_caption = QLabel("", group)
        self.lens_caption.setProperty("role", "caption")
        self.lens_caption.setWordWrap(True)
        group.add_row(self.lens_caption)

        self._lens_rows: dict[str, SliderRow] = {}
        for name, spec in LENS_SPECS.items():
            slider = SliderRow(spec)
            slider.value_changed.connect(self._on_lens_amount)
            self._lens_rows[name] = slider
            group.add_row(slider)
        return group

    # -- state -----------------------------------------------------------

    def set_photo(self, state: EditState | None) -> None:
        """Show the groups for a photo, or the empty state."""
        self._groups.setVisible(state is not None)

    def retranslate(self) -> None:
        """Read the sliders again after a language change (U-02)."""
        self.lens_switch.setToolTip(tr("adjust.lens.enable"))
        self.spot_size_row.set_spec(SPOT_SIZE_SPEC)
        self.set_spot_size(self._spot_size)
        self.clear_spots_button.retranslate()
        self._show_lens()
        self._show_spots()

    def lens_rows(self) -> dict[str, SliderRow]:
        """Return the lens correction's sliders, by the setting each one sets."""
        return dict(self._lens_rows)

    def groups(self) -> tuple[ControlGroup, ...]:
        """Return the collapsible groups, in display order."""
        return tuple(self._group_widgets)

    # -- spot removal (PLAN.md 2.2, A-32) --------------------------------

    @property
    def spot_size(self) -> float:
        """Return the size the slider shows, as a fraction of the long edge."""
        return self._spot_size

    def set_spot_size(self, radius: float) -> None:
        """Show *radius* on the size slider without announcing it."""
        self._spot_size = healing.clamp_radius(radius)
        self.spot_size_row.blockSignals(True)
        self.spot_size_row.set_value(self._spot_size * SPOT_SIZE_SCALE)
        self.spot_size_row.blockSignals(False)

    def set_spot_count(self, count: int) -> None:
        """Say how many spots the photo has mended."""
        self._spot_count = count
        self._show_spots()

    def _show_spots(self) -> None:
        count = self._spot_count
        if count == 0:
            caption = tr("adjust.spots.hint")
        elif count == 1:
            caption = tr("adjust.spots.one")
        else:
            caption = tr("adjust.spots.many", count=count)
        self.spots_caption.setText(caption)
        self.clear_spots_button.setEnabled(count > 0)

    def _on_spot_size(self, _name: str, value: float, dragging: bool) -> None:
        self._spot_size = healing.clamp_radius(value / SPOT_SIZE_SCALE)
        self._set_titles_dimmed(dragging)
        self.spot_size_set.emit(self._spot_size, dragging)

    # -- lens correction (PLAN.md 2.2, A-26 to A-28) ---------------------

    @property
    def lens(self) -> LensSettings:
        """Return the lens correction the controls show."""
        return self._lens

    def set_lens_profile(self, profile: LensProfile | None, is_raw: bool) -> None:
        """Say which lens the photo now open was taken with, if it is known.

        Only a RAW file with a known lens can be corrected: a JPEG has been
        corrected by the camera already, and an unknown lens has no profile.
        """
        self._lens_profile = profile
        self._lens_raw = is_raw
        self._show_lens()

    def set_lens(self, settings: LensSettings) -> None:
        """Show *settings* without announcing them, as undo and reset do."""
        self._lens = settings
        self._show_lens()

    def _show_lens(self) -> None:
        """Bring the switch, the caption and the two sliders up to date."""
        profile = self._lens_profile
        usable = self._lens_raw and profile is not None
        self.lens_switch.blockSignals(True)
        self.lens_switch.setChecked(usable and self._lens.enabled)
        self.lens_switch.blockSignals(False)
        self.lens_switch.setEnabled(usable)
        self._lens_switch_label.setEnabled(usable)
        if not self._lens_raw:
            caption = tr("adjust.lens.jpeg")
        elif profile is None:
            caption = tr("adjust.lens.unknown")
        elif profile.from_camera:
            caption = (
                tr("adjust.lens.camera", lens=profile.name)
                if profile.name
                else tr("adjust.lens.camera_unnamed")
            )
        else:
            caption = profile.name
        self.lens_caption.setText(caption)
        for name, slider in self._lens_rows.items():
            slider.blockSignals(True)
            slider.set_value(getattr(self._lens, name))
            slider.blockSignals(False)
            # A profile can hold one correction and not the other.
            measured = profile is not None and (
                profile.distortion if name == "distortion" else profile.vignetting
            ) is not None
            slider.setEnabled(usable and self._lens.enabled and measured)

    # -- reacting --------------------------------------------------------

    def _on_lens_switched(self, on: bool) -> None:
        self._lens = replace(self._lens, enabled=on)
        self._show_lens()
        self.lens_set.emit(self._lens, False)

    def _on_lens_amount(self, name: str, value: float, dragging: bool) -> None:
        self._lens = self._lens.with_amount(name, value)
        self._set_titles_dimmed(dragging)
        self.lens_set.emit(self._lens, dragging)

    def _set_titles_dimmed(self, dimmed: bool) -> None:
        """Fade the group titles while a slider is moving (PLAN.md 3.6)."""
        for group in self._group_widgets:
            group.header.set_faded(dimmed)
