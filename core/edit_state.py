"""The adjustments applied to one photo (PLAN.md section 2.2).

The numbers the sliders hold, and the name of a film look. The ranges and defaults
are the ones in PLAN.md section 2.2, and they differ between RAW and JPEG
because a RAW file carries a real colour temperature in kelvin while a JPEG
has already been balanced and can only be nudged.

A state is stored beside the photo by :mod:`core.sidecar`, which is why it
can be built from a dictionary as well as read into one. Cropping and
rotation live next to it rather than inside it, in :class:`Edit`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Final

from core.filters import NO_FILTER
from core.geometry import DEFAULT_GEOMETRY, Geometry

# --------------------------------------------------------------------------
# Ranges (PLAN.md section 2.2)
# --------------------------------------------------------------------------

#: Colour temperature in kelvin, for RAW.
TEMPERATURE_MIN_K: Final[float] = 2000.0
TEMPERATURE_MAX_K: Final[float] = 50000.0
#: Tint for RAW: green at one end, magenta at the other.
TINT_MIN: Final[float] = -150.0
TINT_MAX: Final[float] = 150.0

#: A JPEG has no temperature of its own, so both of its white balance
#: sliders are relative and centred on zero.
RELATIVE_MIN: Final[float] = -100.0
RELATIVE_MAX: Final[float] = 100.0

EXPOSURE_MIN_EV: Final[float] = -5.0
EXPOSURE_MAX_EV: Final[float] = 5.0
EXPOSURE_STEP_EV: Final[float] = 0.05

TONE_MIN: Final[float] = -100.0
TONE_MAX: Final[float] = 100.0

#: The vignette's midpoint is a size rather than an amount, so it runs from
#: one end of the frame to the other instead of either way from a middle.
MIDPOINT_MIN: Final[float] = 0.0
MIDPOINT_MAX: Final[float] = 100.0
MIDPOINT_DEFAULT: Final[float] = 50.0

#: How much of a film look is applied, as a percentage. It rests at the
#: whole of it: choosing a filter and getting nothing would be a puzzle.
STRENGTH_MIN: Final[float] = 0.0
STRENGTH_MAX: Final[float] = 100.0
STRENGTH_DEFAULT: Final[float] = 100.0

#: The detail sliders (PLAN.md 2.2) are amounts with nothing below zero,
#: except sharpening, which goes below zero to soften the photo instead.
#: Sharpening reaches past 100 the way it does in other editors, because a
#: soft lens or a photo meant for a small screen can take more.
SHARPENING_MIN: Final[float] = -100.0
SHARPENING_MAX: Final[float] = 150.0
DETAIL_MIN: Final[float] = 0.0
DETAIL_MAX: Final[float] = 100.0
#: Where a RAW photo's sharpening and colour noise rest. A RAW straight off
#: the sensor is softer than the camera's own JPEG, which was sharpened, and
#: carries coloured speckles in its shadows that the camera took out; these
#: are the amounts other editors start a RAW at for the same reasons. A JPEG
#: has had both done already and rests at zero.
RAW_SHARPENING_DEFAULT: Final[float] = 40.0
RAW_COLOUR_NOISE_DEFAULT: Final[float] = 25.0

#: Film grain (PLAN.md 2.2, A-25) is an amount with nothing below zero. 50 is
#: the camera's weak grain and 100 its strong one, which measured twice as
#: strong (PLAN.md 6 step 11.5).
GRAIN_MIN: Final[float] = 0.0
GRAIN_MAX: Final[float] = 100.0

#: Skin enhancement (PLAN.md 2.2, A-29) is an amount with nothing below zero:
#: how much of the smoothing and of the brighter, less yellow skin is laid
#: over the faces the photo holds.
SKIN_MIN: Final[float] = 0.0
SKIN_MAX: Final[float] = 100.0
#: Whitening (PLAN.md 2.2, A-30) likewise: how much lighter and less yellow
#: the skin of the faces, and the skin joined to them, is made.
WHITEN_MIN: Final[float] = 0.0
WHITEN_MAX: Final[float] = 100.0


@dataclass(frozen=True)
class EditState:
    """The adjustments of PLAN.md section 2.2.

    ``temperature`` and ``tint`` mean different things for the two kinds of
    photo: kelvin and the DNG tint scale for RAW, and a relative nudge from
    -100 to +100 for JPEG. ``is_raw`` says which, so that a state knows how
    to read its own numbers.
    """

    temperature: float = 0.0
    tint: float = 0.0
    exposure: float = 0.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    vibrance: float = 0.0
    saturation: float = 0.0
    vignette: float = 0.0
    #: How much of the frame the vignette leaves alone, as a percentage of
    #: the way out to the corner.
    vignette_midpoint: float = MIDPOINT_DEFAULT
    #: How much film grain is laid over the photo (PLAN.md 6 step 11.5).
    grain: float = 0.0
    #: How much the skin of the faces is smoothed and brightened (PLAN.md 6
    #: steps 2.6 and 10).
    skin: float = 0.0
    #: How much the skin is whitened (PLAN.md 6 step 10).
    whiten: float = 0.0
    #: Which film look is on the photo, by the name of its file in
    #: ``resources/filters``, or empty for none. The only adjustment that is
    #: a name rather than a number: it chooses a measured rendering rather
    #: than setting an amount of one (:mod:`core.filters`).
    filter: str = NO_FILTER
    #: How much of that look is applied, as a percentage.
    filter_strength: float = STRENGTH_DEFAULT
    #: The detail group (PLAN.md 2.2 and section 6 step 2.5): how much the
    #: finest detail is sharpened - below zero, softened - how much of the
    #: photo the sharpening is kept off - only edges are sharpened at 100 -
    #: and how much noise is taken out of the brightness and out of the colour.
    sharpening: float = 0.0
    sharpen_masking: float = 0.0
    luminance_noise: float = 0.0
    colour_noise: float = 0.0
    #: Whether temperature and tint are in kelvin or relative.
    is_raw: bool = False

    @classmethod
    def for_photo(
        cls, is_raw: bool, temperature: float = 0.0, tint: float = 0.0
    ) -> "EditState":
        """Return the untouched state of a photo.

        For a RAW photo the white balance starts at what the camera recorded,
        which is why it has to be passed in; for a JPEG it starts at zero.
        """
        if not is_raw:
            return cls(is_raw=False)
        return cls(
            is_raw=True,
            temperature=temperature,
            tint=tint,
            sharpening=RAW_SHARPENING_DEFAULT,
            colour_noise=RAW_COLOUR_NOISE_DEFAULT,
        )

    # -- questions the pipeline and the panel ask -------------------------

    @property
    def is_neutral_colour(self) -> bool:
        """Return whether every tone and colour slider but white balance is at rest.

        The detail group is not asked about here: a RAW photo does not rest
        at zero in it, so it is compared with where the photo started in
        :meth:`detail_is_untouched` instead.

        The filter's strength is not asked about, for the same reason as the
        vignette's midpoint (PLAN.md 2.2): with no filter chosen it changes
        nothing, so a photo that only has it moved is still untouched.
        """
        return (
            self.filter == NO_FILTER
            and self.exposure == 0.0
            and self.contrast == 0.0
            and self.highlights == 0.0
            and self.shadows == 0.0
            and self.whites == 0.0
            and self.blacks == 0.0
            and self.vibrance == 0.0
            and self.saturation == 0.0
            and self.vignette == 0.0
            and self.grain == 0.0
            and self.skin == 0.0
            and self.whiten == 0.0
        )

    @property
    def uses_faces(self) -> bool:
        """Return whether a slider that works on faces has been moved (A-29, A-30)."""
        return self.skin != 0.0 or self.whiten != 0.0

    def without_faces(self) -> "EditState":
        """Return this state as it acts on a photo known to hold no faces.

        The sliders that work on faces change nothing there.
        """
        return replace(self, skin=0.0, whiten=0.0)

    def is_untouched(self, reference: "EditState") -> bool:
        """Return whether nothing has been changed from *reference*.

        White balance cannot be compared against zero for a RAW photo, whose
        untouched value is whatever the camera recorded, so the state it
        started from is what it is compared against.
        """
        return (
            self.is_neutral_colour
            and self.temperature == reference.temperature
            and self.tint == reference.tint
            and self.detail_is_untouched(reference)
        )

    def detail_is_untouched(self, reference: "EditState") -> bool:
        """Return whether the detail group is where *reference* has it.

        A RAW photo rests with some sharpening and some colour noise taken
        out, so this is a comparison and not a test against zero. The masking
        only counts while there is sharpening for it to mask, for the same
        reason as the vignette's midpoint (PLAN.md 2.2); softening, below
        zero, has no edges to keep to and does not use it.
        """
        return (
            self.sharpening == reference.sharpening
            and self.luminance_noise == reference.luminance_noise
            and self.colour_noise == reference.colour_noise
            and (self.sharpening <= 0.0 or self.sharpen_masking == reference.sharpen_masking)
        )

    def temperature_range(self) -> tuple[float, float]:
        """Return the range of the temperature slider for this kind of photo."""
        if self.is_raw:
            return TEMPERATURE_MIN_K, TEMPERATURE_MAX_K
        return RELATIVE_MIN, RELATIVE_MAX

    def tint_range(self) -> tuple[float, float]:
        """Return the range of the tint slider for this kind of photo."""
        if self.is_raw:
            return TINT_MIN, TINT_MAX
        return RELATIVE_MIN, RELATIVE_MAX

    # -- changing ---------------------------------------------------------

    def with_value(self, name: str, value: float) -> "EditState":
        """Return a copy with one adjustment set to *value*, within its range."""
        low, high = self.range_of(name)
        return replace(self, **{name: max(low, min(high, float(value)))})

    def with_filter(self, identifier: str) -> "EditState":
        """Return a copy wearing the film look *identifier*, or none for ""."""
        return replace(self, filter=str(identifier))

    def range_of(self, name: str) -> tuple[float, float]:
        """Return the lowest and highest value *name* may take."""
        if name == "filter_strength":
            return STRENGTH_MIN, STRENGTH_MAX
        if name == "temperature":
            return self.temperature_range()
        if name == "tint":
            return self.tint_range()
        if name == "exposure":
            return EXPOSURE_MIN_EV, EXPOSURE_MAX_EV
        if name == "vignette_midpoint":
            return MIDPOINT_MIN, MIDPOINT_MAX
        if name == "grain":
            return GRAIN_MIN, GRAIN_MAX
        if name == "skin":
            return SKIN_MIN, SKIN_MAX
        if name == "whiten":
            return WHITEN_MIN, WHITEN_MAX
        if name == "sharpening":
            return SHARPENING_MIN, SHARPENING_MAX
        if name in DETAIL_NAMES:
            return DETAIL_MIN, DETAIL_MAX
        if name in _TONE_NAMES:
            return TONE_MIN, TONE_MAX
        raise KeyError(f"{name} is not an adjustment")

    def reset(self, reference: "EditState") -> "EditState":
        """Return the state with everything back to *reference* (E-05)."""
        return reference

    def to_dict(self) -> dict[str, Any]:
        """Return the adjustments as a dictionary, for the sidecar file."""
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "is_raw"
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], reference: "EditState") -> "EditState":
        """Build a state from stored adjustments (PLAN.md section 7.1).

        A field that is missing or not a number falls back to *reference*,
        which is where the photo rests when nothing has been changed; this is
        what makes a parameter file written by an older version readable, and
        what keeps a RAW photo's white balance at the camera's own value
        rather than at zero. Values outside the slider's range are brought
        back into it, so that a hand edited file cannot push an adjustment
        somewhere the interface could never take it.
        """
        state = reference
        for name in NUMERIC_NAMES:
            number = as_float(data.get(name))
            if number is None:
                continue
            state = state.with_value(name, number)
        chosen = data.get("filter")
        if isinstance(chosen, str):
            state = state.with_filter(chosen)
        return state


@dataclass(frozen=True)
class Edit:
    """Everything one photo carries: its adjustments and its geometry.

    The two are kept apart because they are used apart. The pipeline's colour
    steps take the adjustments alone (PLAN.md section 6, steps 3 to 12), the
    geometry is applied before them at step 2, and copying settings from one
    photo to another takes the adjustments without the crop (P-03). What they
    share is this: the history remembers both, and the parameter file stores
    both.
    """

    adjustments: EditState
    geometry: Geometry = DEFAULT_GEOMETRY

    def with_adjustments(self, adjustments: EditState) -> "Edit":
        """Return this edit with different adjustments."""
        return replace(self, adjustments=adjustments)

    def is_untouched(self, reference: "Edit") -> bool:
        """Return whether nothing has been changed from *reference*.

        An untouched photo has nothing worth storing, which is what makes its
        parameter file go away (PLAN.md 7.1). The lens correction is compared
        with *reference* rather than with off, since a RAW photo starts with
        it on (PLAN.md 2.2). A mended spot is a change like any other, and
        so is a skin range shaped by the brush (PLAN.md 2.10).
        """
        return (
            self.adjustments.is_untouched(reference.adjustments)
            and self.geometry.is_default
            and self.geometry.lens == reference.geometry.lens
            and not self.geometry.spots
            and self.geometry.skin.is_default
        )


#: The adjustments, in the order the panel shows them (PLAN.md section 2.2).
ADJUSTMENT_NAMES: Final[tuple[str, ...]] = (
    "temperature",
    "tint",
    "exposure",
    "contrast",
    "highlights",
    "shadows",
    "whites",
    "blacks",
    "vibrance",
    "saturation",
    "skin",
    "whiten",
    "vignette",
    "vignette_midpoint",
    "grain",
    "sharpening",
    "sharpen_masking",
    "luminance_noise",
    "colour_noise",
)

#: The portrait sliders (PLAN.md 2.2, A-29 and A-30). They sit on the retouch
#: tab, follow the faces of one photo, and so are left out of presets and of
#: the adjust tab's reset (PLAN.md 3.1, 7.2).
PORTRAIT_NAMES: Final[tuple[str, ...]] = ("skin", "whiten")

#: What the adjust tab's reset puts back (E-05): its own sliders, and none of
#: what the filter, retouch or crop tab holds (PLAN.md 3.1).
RESET_NAMES: Final[tuple[str, ...]] = tuple(
    name for name in ADJUSTMENT_NAMES if name not in PORTRAIT_NAMES
)

#: The detail group. A RAW photo and a JPEG rest at different places in it,
#: so, like the white balance, it is only carried from one photo to another
#: of the same kind when settings are copied (PLAN.md 7.2).
DETAIL_NAMES: Final[tuple[str, ...]] = (
    "sharpening",
    "sharpen_masking",
    "luminance_noise",
    "colour_noise",
)

#: Every adjustment that is a number, which is the panel's sliders plus the
#: filter's strength. The parameter file, the presets and copy and paste all
#: work from this one rather than from the panel's list, because the filter
#: is on a tab of its own and is still an adjustment (PLAN.md 7.1 and 7.2).
#: The filter itself is a name, not a number, and is carried beside it.
NUMERIC_NAMES: Final[tuple[str, ...]] = ADJUSTMENT_NAMES + ("filter_strength",)

_TONE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "contrast",
        "highlights",
        "shadows",
        "whites",
        "blacks",
        "vibrance",
        "saturation",
        "vignette",
    }
)


def as_float(value: Any) -> float | None:
    """Return *value* as a float, or ``None`` when it is not a number.

    What comes out of a parameter file or a preset is whatever was in the
    JSON, so a value has to be read rather than trusted: a string, a list, a
    boolean, a not-a-number or an infinity all have to come back as nothing
    instead of reaching an adjustment (PLAN.md 7.1).
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number
