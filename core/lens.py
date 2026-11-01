"""Lens correction from the Lensfun database (PLAN.md 2.2 and section 6 step 1.5).

A lens bends straight lines (distortion), lets the colours land at slightly
different distances from the centre (lateral chromatic aberration) and
darkens the corners (vignetting). The Lensfun project measures all three for
a few thousand lenses and publishes the results as XML, which is bundled in
``resources/lensfun`` under CC BY-SA 3.0. Only the data is used: the models
below are implemented here from Lensfun's published formulas, and no Lensfun
code is.

The three models, as Lensfun defines them, all describe what the lens did:

* distortion, ``ptlens``: a point at radius ``ru`` on a perfect picture
  landed at ``rd = ru (a ru³ + b ru² + c ru + 1 − a − b − c)``; ``poly3`` and
  ``poly5`` are the same idea with other polynomials;
* chromatic aberration, ``poly3``: red (and blue) landed at
  ``r (b r² + c r + v)`` where green landed at ``r``; ``linear`` is ``k r``;
* vignetting, ``pa``: the light at radius ``r`` was multiplied by
  ``1 + k1 r² + k2 r⁴ + k3 r⁶``.

So correcting is direct, with nothing to invert: every pixel of the corrected
picture is looked up where the lens put it, and its light is divided by what
the lens took away.

Some cameras also write their own measurements of the lens into the RAW file
(:mod:`core.embedded_lens` reads them). Those come as tables over the radius
rather than as polynomials - the ``table`` models below - and when a file has
them they are used instead of the database: they were made for that lens on
that body, at that focal length.

Radii are physical. Distortion and aberration measure 1 as half the short
side of the frame the lens was calibrated on, vignetting as half its
diagonal. The calibration names its crop factor and aspect ratio, the photo
has its own, so a pixel is first turned into millimetres on the sensor and
then into the calibration's units. A lens measured on one body is therefore
corrected properly on another of a different size.

Lookups happen once, when the photo is decoded, and give a
:class:`LensProfile`: the coefficients at that focal length and aperture.
What the user can change is a :class:`LensSettings`, kept with the geometry
because it changes the frame the crop is measured against.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import cv2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from core.decoder import DecodedPhoto

logger = logging.getLogger(__name__)

#: Where the database ships. Found from this file rather than from the
#: application, because a photo is decoded - and its lens looked up - by the
#: exporter and the collage as well as by the controller; in a packaged build
#: ``core`` and ``resources`` are unpacked side by side in the same way.
DATABASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "resources" / "lensfun"

#: The diagonal of a 36 x 24 mm frame, which a crop factor divides.
FULL_FRAME_DIAGONAL_MM: Final[float] = math.hypot(36.0, 24.0)

#: Lensfun's aspect ratio when a lens entry names none.
DEFAULT_ASPECT: Final[float] = 1.5

#: The two amount sliders run from no correction to twice the profile's, with
#: the profile itself at 100 (PLAN.md 2.2, A-27 and A-28).
AMOUNT_MIN: Final[float] = 0.0
AMOUNT_MAX: Final[float] = 200.0
AMOUNT_DEFAULT: Final[float] = 100.0

#: The darkest a corner is believed to be, as a fraction of the centre. A
#: profile is a polynomial fit and can run towards zero just past the corner
#: it was measured to; this stops the correction from blowing up there.
VIGNETTING_FLOOR: Final[float] = 0.1

#: Rows corrected at a time. Working in strips keeps the coordinate arrays
#: small: for a 24 megapixel photo a whole frame of them would be several
#: hundred megabytes (PLAN.md 11, memory).
STRIP_ROWS: Final[int] = 256

#: How far apart, in pixels, the corrections are worked out exactly. In
#: between they are interpolated linearly: every one of them is a smooth
#: function of the radius, so the error that leaves is a few thousandths of
#: a pixel, for a few hundred times less work than doing every pixel. A
#: small image gets closer nodes, so that there are always at least
#: :data:`_NODES_ACROSS` of them along its short side.
NODE_SPACING: Final[int] = 16
_NODES_ACROSS: Final[int] = 64

#: How finely each edge of the frame is checked when working out how far to
#: zoom so that no empty edge shows (C-05).
_EDGE_SAMPLES: Final[int] = 65

#: Lateral chromatic aberration smaller than this, in pixels of the image
#: being corrected, is left alone: a fringe a fifth of a pixel wide cannot
#: be seen, and leaving it lets the three colours be moved together, which
#: is a third of the work. On a screen sized preview that is nearly always
#: the case; at full size it depends on the lens.
INVISIBLE_ABERRATION_PX: Final[float] = 0.2

#: The zoom search's range and how many halvings it takes.
_SCALE_LOW: Final[float] = 0.5
_SCALE_HIGH: Final[float] = 2.0
_SCALE_STEPS: Final[int] = 40


# --------------------------------------------------------------------------
# What the user sets
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LensSettings:
    """Whether the lens is corrected, and how much of each correction.

    *distortion* and *vignetting* are percentages of what the profile says.
    Chromatic aberration has no slider: it is either right or not there, so
    it simply follows the switch.
    """

    enabled: bool = False
    distortion: float = AMOUNT_DEFAULT
    vignetting: float = AMOUNT_DEFAULT

    @classmethod
    def for_photo(cls, is_raw: bool) -> "LensSettings":
        """Return where a photo opened for the first time starts.

        A RAW file is corrected, so that it frames like the camera's own JPEG.
        A JPEG has been corrected by the camera already, and correcting it
        again would bend it the other way, so it is left alone.
        """
        return cls(enabled=is_raw)

    def with_amount(self, name: str, value: float) -> "LensSettings":
        """Return a copy with ``distortion`` or ``vignetting`` set, within range."""
        if name not in ("distortion", "vignetting"):
            raise KeyError(f"{name} is not a lens correction amount")
        return replace(self, **{name: _clamp(float(value), AMOUNT_MIN, AMOUNT_MAX)})

    def to_dict(self) -> dict[str, Any]:
        """Return the settings as stored in the parameter file (PLAN.md 7.1)."""
        return {
            "enabled": self.enabled,
            "distortion": self.distortion,
            "vignetting": self.vignetting,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "LensSettings":
        """Build settings from stored data; anything odd falls back.

        A parameter file written before lens correction existed has no lens
        block at all and reads as switched off: the photo was framed and
        cropped without the correction, and turning it on now would move what
        the crop holds (PLAN.md 7.1).
        """
        if not isinstance(data, dict):
            return cls()
        return cls(
            enabled=data.get("enabled") is True,
            distortion=_amount(data.get("distortion")),
            vignetting=_amount(data.get("vignetting")),
        )


DEFAULT_SETTINGS: Final[LensSettings] = LensSettings()


# --------------------------------------------------------------------------
# What the file says about how it was taken
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LensShot:
    """The facts a lookup needs, as read from the photo's file."""

    lens_model: str
    lens_maker: str = ""
    camera_maker: str = ""
    camera_model: str = ""
    focal_length: float = 0.0
    aperture: float = 0.0
    #: The 35 mm equivalent focal length, which gives the crop factor of a
    #: camera the database does not know.
    focal_length_35mm: float = 0.0


@dataclass(frozen=True)
class LensProfile:
    """One lens's corrections at the focal length and aperture of one photo.

    Hashable, so that the work that follows from it - how far the corrected
    picture has to be zoomed - can be remembered.
    """

    #: How the lens is named in the panel.
    name: str
    #: The distortion model and its coefficients, or ``None``.
    distortion: tuple[str, tuple[float, ...]] | None
    #: The aberration model and its red and blue coefficients, or ``None``.
    tca: tuple[str, tuple[float, ...], tuple[float, ...]] | None
    #: The vignetting model and its coefficients, or ``None``.
    vignetting: tuple[str, tuple[float, ...]] | None
    #: One unit of distortion and aberration radius, in millimetres.
    distortion_unit_mm: float
    #: One unit of vignetting radius, in millimetres.
    vignetting_unit_mm: float
    #: The diagonal of the photo's own sensor, in millimetres. Only the
    #: ratios of these three lengths matter, so a camera's own tables, whose
    #: radii are fractions of the frame, give them in any unit they like.
    sensor_diagonal_mm: float
    #: Whether the corrections are the camera's own, read from the file,
    #: rather than the database's.
    from_camera: bool = False

    @property
    def corrects_geometry(self) -> bool:
        """Return whether there is anything to move, not only to brighten."""
        return self.distortion is not None or self.tca is not None


# --------------------------------------------------------------------------
# The database
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Lens:
    """One lens entry of the database, as calibrated."""

    maker: str
    #: Every name it goes by, the one the camera writes first.
    models: tuple[str, ...]
    display: str
    crop_factor: float
    aspect: float
    rectilinear: bool
    #: (focal length, model, coefficients)
    distortion: tuple[tuple[float, str, tuple[float, ...]], ...]
    #: (focal length, model, red coefficients, blue coefficients)
    tca: tuple[tuple[float, str, tuple[float, ...], tuple[float, ...]], ...]
    #: (focal length, aperture, distance, k1, k2, k3)
    vignetting: tuple[tuple[float, float, float, float, float, float], ...]

    @property
    def is_calibrated(self) -> bool:
        """Return whether the entry holds any correction at all."""
        return bool(self.distortion or self.tca or self.vignetting)


@dataclass(frozen=True)
class _Camera:
    """One camera body, for its crop factor."""

    maker: str
    models: tuple[str, ...]
    crop_factor: float


@dataclass(frozen=True)
class Database:
    """The lenses and cameras of a Lensfun database."""

    lenses: tuple[_Lens, ...]
    cameras: tuple[_Camera, ...]

    @classmethod
    def load(cls, directory: Path) -> "Database":
        """Read every ``*.xml`` file in *directory*.

        A file that cannot be read is logged and skipped, so that one broken
        file costs its own lenses and nothing else.
        """
        lenses: list[_Lens] = []
        cameras: list[_Camera] = []
        for path in sorted(directory.glob("*.xml")):
            try:
                root = ElementTree.parse(path).getroot()
            except (OSError, ElementTree.ParseError):
                logger.exception("The lens database file %s could not be read", path.name)
                continue
            for element in root.findall("camera"):
                camera = _parse_camera(element)
                if camera is not None:
                    cameras.append(camera)
            for element in root.findall("lens"):
                lens = _parse_lens(element)
                if lens is not None and lens.is_calibrated:
                    lenses.append(lens)
        return cls(lenses=tuple(lenses), cameras=tuple(cameras))

    def find_lens(self, shot: LensShot, crop_factor: float | None) -> _Lens | None:
        """Return the entry for the lens *shot* was taken with, if there is one.

        The name is compared the way the camera writes it, ignoring case and
        spacing; failing that, ignoring everything but letters and digits.
        Among several entries for one name, the one from the lens's own maker
        and the one calibrated nearest the camera's crop factor win: Lensfun
        warns that a profile measured on a smaller sensor does not reach the
        corners of a larger one.
        """
        wanted = _loose(shot.lens_model)
        if not wanted:
            return None
        candidates = [
            lens for lens in self.lenses if any(_loose(m) == wanted for m in lens.models)
        ]
        if not candidates:
            compact = _compact(shot.lens_model)
            candidates = [
                lens
                for lens in self.lenses
                if any(_compact(m) == compact for m in lens.models)
            ]
        candidates = [lens for lens in candidates if lens.rectilinear]
        if not candidates:
            return None

        maker = shot.lens_maker.strip().lower()

        def rank(lens: _Lens) -> tuple[int, int, float]:
            same_maker = bool(maker) and _makers_match(maker, lens.maker.lower())
            if crop_factor is None:
                return (0 if same_maker else 1, 0, 0.0)
            too_small = lens.crop_factor > crop_factor * 1.02
            return (
                0 if same_maker else 1,
                1 if too_small else 0,
                abs(lens.crop_factor - crop_factor),
            )

        return min(candidates, key=rank)

    def crop_factor_of(self, shot: LensShot) -> float | None:
        """Return the crop factor of the camera *shot* was taken with.

        The database's own figure when it knows the body; otherwise the ratio
        of the 35 mm equivalent to the real focal length the file records.
        """
        model = _loose(shot.camera_model)
        maker = shot.camera_maker.strip().lower()
        if model:
            for camera in self.cameras:
                if any(_loose(name) == model for name in camera.models) and (
                    not maker or _makers_match(maker, camera.maker.lower())
                ):
                    return camera.crop_factor
        if shot.focal_length > 0 and shot.focal_length_35mm > 0:
            return shot.focal_length_35mm / shot.focal_length
        return None

    def profile_for(self, shot: LensShot) -> LensProfile | None:
        """Return the corrections for *shot*, or ``None`` for an unknown lens."""
        crop_factor = self.crop_factor_of(shot)
        lens = self.find_lens(shot, crop_factor)
        if lens is None:
            return None
        if crop_factor is None or crop_factor <= 0:
            # A body nobody knows, in a file without an equivalent focal
            # length: the best guess is the format the lens was made for.
            crop_factor = lens.crop_factor
        return _profile(lens, shot, crop_factor)


_loaded: dict[Path, Database] = {}
_loading = threading.Lock()


def database(directory: Path = DATABASE_DIR) -> Database:
    """Return the database in *directory*, reading it the first time only.

    Reading the whole of it takes about half a second, so the controller
    starts it on a background thread as soon as the program opens. A decode
    that asks before it is done waits for that one reading instead of
    starting a second.
    """
    with _loading:
        found = _loaded.get(directory)
        if found is None:
            found = Database.load(directory)
            _loaded[directory] = found
        return found


def profile_for(shot: LensShot | None, directory: Path = DATABASE_DIR) -> LensProfile | None:
    """Return the corrections for *shot* from the bundled database.

    Never raises: a lens that cannot be looked up is simply not corrected.
    """
    if shot is None or not shot.lens_model.strip():
        return None
    try:
        return database(directory).profile_for(shot)
    except (OSError, ValueError, ArithmeticError):
        logger.exception("The lens %r could not be looked up", shot.lens_model)
        return None


def _parse_camera(element: ElementTree.Element) -> _Camera | None:
    """Return one ``<camera>`` entry, or ``None`` if it lacks what is needed."""
    crop = _number(element.findtext("cropfactor"))
    models = tuple(m.text.strip() for m in element.findall("model") if m.text)
    if crop is None or crop <= 0 or not models:
        return None
    return _Camera(maker=_untranslated(element, "maker"), models=models, crop_factor=crop)


def _parse_lens(element: ElementTree.Element) -> _Lens | None:
    """Return one ``<lens>`` entry, or ``None`` if it lacks what is needed."""
    models = [m for m in element.findall("model") if m.text]
    crop = _number(element.findtext("cropfactor"))
    if not models or crop is None or crop <= 0:
        return None
    lens_type = (element.findtext("type") or "rectilinear").strip()
    maker = _untranslated(element, "maker")
    english = next((m.text.strip() for m in models if m.get("lang") == "en"), "")
    own = next((m.text.strip() for m in models if m.get("lang") is None), models[0].text.strip())
    display = english or own
    if maker and maker.lower() not in display.lower():
        display = f"{maker} {display}"

    distortion: list[tuple[float, str, tuple[float, ...]]] = []
    tca: list[tuple[float, str, tuple[float, ...], tuple[float, ...]]] = []
    vignetting: list[tuple[float, float, float, float, float, float]] = []
    for calibration in element.findall("calibration"):
        for entry in calibration.findall("distortion"):
            parsed = _parse_distortion(entry)
            if parsed is not None:
                distortion.append(parsed)
        for entry in calibration.findall("tca"):
            parsed_tca = _parse_tca(entry)
            if parsed_tca is not None:
                tca.append(parsed_tca)
        for entry in calibration.findall("vignetting"):
            parsed_vignetting = _parse_vignetting(entry)
            if parsed_vignetting is not None:
                vignetting.append(parsed_vignetting)

    return _Lens(
        maker=maker,
        models=tuple(m.text.strip() for m in models),
        display=display,
        crop_factor=crop,
        aspect=_aspect(element.findtext("aspect-ratio")),
        rectilinear=lens_type == "rectilinear",
        distortion=tuple(sorted(distortion, key=lambda item: item[0])),
        tca=tuple(sorted(tca, key=lambda item: item[0])),
        vignetting=tuple(sorted(vignetting)),
    )


#: The coefficients each model is written with, in the order they are used.
_DISTORTION_TERMS: Final[dict[str, tuple[str, ...]]] = {
    "ptlens": ("a", "b", "c"),
    "poly3": ("k1",),
    "poly5": ("k1", "k2"),
}
_TCA_TERMS: Final[dict[str, tuple[tuple[str, ...], tuple[str, ...]]]] = {
    "poly3": (("vr", "cr", "br"), ("vb", "cb", "bb")),
    "linear": (("kr",), ("kb",)),
}
#: What a coefficient that is left out stands for.
_TCA_DEFAULTS: Final[dict[str, float]] = {"vr": 1.0, "vb": 1.0, "kr": 1.0, "kb": 1.0}


def _parse_distortion(entry: ElementTree.Element) -> tuple[float, str, tuple[float, ...]] | None:
    model = entry.get("model", "")
    focal = _number(entry.get("focal"))
    terms = _DISTORTION_TERMS.get(model)
    if terms is None or focal is None:
        return None
    return focal, model, tuple(_number(entry.get(term)) or 0.0 for term in terms)


def _parse_tca(
    entry: ElementTree.Element,
) -> tuple[float, str, tuple[float, ...], tuple[float, ...]] | None:
    model = entry.get("model", "")
    focal = _number(entry.get("focal"))
    terms = _TCA_TERMS.get(model)
    if terms is None or focal is None:
        return None

    def read(names: tuple[str, ...]) -> tuple[float, ...]:
        values = []
        for name in names:
            value = _number(entry.get(name))
            values.append(_TCA_DEFAULTS.get(name, 0.0) if value is None else value)
        return tuple(values)

    return focal, model, read(terms[0]), read(terms[1])


def _parse_vignetting(
    entry: ElementTree.Element,
) -> tuple[float, float, float, float, float, float] | None:
    if entry.get("model") != "pa":
        return None
    focal = _number(entry.get("focal"))
    aperture = _number(entry.get("aperture"))
    if focal is None or aperture is None or aperture <= 0:
        return None
    distance = _number(entry.get("distance")) or 1000.0
    k1, k2, k3 = (_number(entry.get(name)) or 0.0 for name in ("k1", "k2", "k3"))
    return focal, aperture, distance, k1, k2, k3


def _untranslated(element: ElementTree.Element, tag: str) -> str:
    """Return the text of the *tag* child that has no language of its own."""
    children = element.findall(tag)
    for child in children:
        if child.get("lang") is None and child.text:
            return child.text.strip()
    return children[0].text.strip() if children and children[0].text else ""


def _aspect(text: str | None) -> float:
    """Return an ``<aspect-ratio>`` as long side over short side."""
    if not text:
        return DEFAULT_ASPECT
    try:
        if ":" in text:
            long_side, short_side = (float(part) for part in text.split(":", 1))
            ratio = long_side / short_side
        else:
            ratio = float(text)
    except (ValueError, ZeroDivisionError):
        return DEFAULT_ASPECT
    if not math.isfinite(ratio) or ratio <= 0:
        return DEFAULT_ASPECT
    return max(ratio, 1.0 / ratio)


def _loose(name: str) -> str:
    """Return *name* lower case with its spacing made uniform."""
    return " ".join(name.lower().split())


def _compact(name: str) -> str:
    """Return only the letters and digits of *name*, lower case."""
    return re.sub(r"[^0-9a-z]", "", name.lower())


def _makers_match(one: str, other: str) -> bool:
    """Return whether two maker names are the same company.

    A file can write ``MAKER CORPORATION`` in capitals where the database
    writes ``Maker``, so one only has to start the other.
    """
    return one.startswith(other) or other.startswith(one)


# --------------------------------------------------------------------------
# One photo's profile
# --------------------------------------------------------------------------


def _profile(lens: _Lens, shot: LensShot, crop_factor: float) -> LensProfile:
    """Return *lens*'s corrections at the focal length and aperture of *shot*."""
    focal = shot.focal_length if shot.focal_length > 0 else _middle_focal(lens)
    calibrated_diagonal = FULL_FRAME_DIAGONAL_MM / lens.crop_factor
    short_side = calibrated_diagonal / math.sqrt(1.0 + lens.aspect * lens.aspect)
    return LensProfile(
        name=lens.display,
        distortion=_distortion_at(lens.distortion, focal),
        tca=_tca_at(lens.tca, focal),
        vignetting=_vignetting_at(lens.vignetting, focal, shot.aperture),
        distortion_unit_mm=short_side / 2.0,
        vignetting_unit_mm=calibrated_diagonal / 2.0,
        sensor_diagonal_mm=FULL_FRAME_DIAGONAL_MM / crop_factor,
    )


def _middle_focal(lens: _Lens) -> float:
    """Return a focal length to use when the file records none."""
    focals = [item[0] for item in lens.distortion] or [item[0] for item in lens.vignetting]
    return float(np.median(focals)) if focals else 0.0


def _between(focals: list[float], focal: float) -> tuple[int, int, float]:
    """Return the two entries either side of *focal* and how far between.

    Past either end the nearest entry is used as it is: a profile says
    nothing about focal lengths it was not measured at.
    """
    if focal <= focals[0]:
        return 0, 0, 0.0
    if focal >= focals[-1]:
        last = len(focals) - 1
        return last, last, 0.0
    upper = next(index for index, value in enumerate(focals) if value >= focal)
    lower = upper - 1
    span = focals[upper] - focals[lower]
    return lower, upper, 0.0 if span <= 0 else (focal - focals[lower]) / span


def _mix(one: tuple[float, ...], other: tuple[float, ...], weight: float) -> tuple[float, ...]:
    """Return the coefficients a *weight* of the way from *one* to *other*."""
    return tuple(a + (b - a) * weight for a, b in zip(one, other))


def _distortion_at(
    entries: tuple[tuple[float, str, tuple[float, ...]], ...], focal: float
) -> tuple[str, tuple[float, ...]] | None:
    """Return the distortion at *focal*, interpolated between calibrations."""
    if not entries:
        return None
    lower, upper, weight = _between([item[0] for item in entries], focal)
    low, high = entries[lower], entries[upper]
    if low[1] != high[1]:
        # Two models cannot be blended term by term; take the nearer one.
        nearer = low if weight < 0.5 else high
        return nearer[1], nearer[2]
    return low[1], _mix(low[2], high[2], weight)


def _tca_at(
    entries: tuple[tuple[float, str, tuple[float, ...], tuple[float, ...]], ...], focal: float
) -> tuple[str, tuple[float, ...], tuple[float, ...]] | None:
    """Return the chromatic aberration at *focal*, interpolated."""
    if not entries:
        return None
    lower, upper, weight = _between([item[0] for item in entries], focal)
    low, high = entries[lower], entries[upper]
    if low[1] != high[1]:
        nearer = low if weight < 0.5 else high
        return nearer[1], nearer[2], nearer[3]
    return low[1], _mix(low[2], high[2], weight), _mix(low[3], high[3], weight)


def _vignetting_at(
    entries: tuple[tuple[float, float, float, float, float, float], ...],
    focal: float,
    aperture: float,
) -> tuple[str, tuple[float, ...]] | None:
    """Return the vignetting at *focal* and *aperture*.

    Within each calibrated focal length the farthest focus distance is used,
    since the file does not say where the lens was focused and most photos
    are not close ups. The apertures are then interpolated in stops, which is
    how the light changes, and the two nearest focal lengths in millimetres.
    """
    if not entries:
        return None
    by_focal: dict[float, list[tuple[float, float, float, float, float, float]]] = {}
    for entry in entries:
        by_focal.setdefault(entry[0], []).append(entry)

    def at_aperture(group: list[tuple[float, float, float, float, float, float]]) -> tuple[float, ...]:
        farthest = max(entry[2] for entry in group)
        rows = sorted(
            (entry for entry in group if entry[2] == farthest), key=lambda entry: entry[1]
        )
        stops = [math.log2(entry[1]) for entry in rows]
        wanted = math.log2(aperture) if aperture > 0 else stops[0]
        lower, upper, weight = _between(stops, wanted)
        return _mix(rows[lower][3:], rows[upper][3:], weight)

    focals = sorted(by_focal)
    lower, upper, weight = _between(focals, focal)
    return "pa", _mix(at_aperture(by_focal[focals[lower]]), at_aperture(by_focal[focals[upper]]), weight)


def camera_profile(
    name: str,
    distortion: tuple[tuple[float, ...], tuple[float, ...]] | None,
    vignetting: tuple[tuple[float, ...], tuple[float, ...]] | None,
    unit: float,
) -> LensProfile:
    """Return a profile from the tables a camera wrote into its file.

    Each table is its radii and, at each of them, a value: for *distortion*
    how many percent further out than it should the lens put that radius,
    for *vignetting* how many percent of the centre's light is left there.
    *unit* is the radius the tables call 1, as a fraction of half the
    frame's diagonal. In between the radii the values are interpolated
    linearly in the square of the radius, which is how both grow, from none
    at the centre; past the last one the last value holds.
    """

    def table(radii: tuple[float, ...], values: tuple[float, ...], centre: float) -> tuple[float, ...]:
        squares = (0.0, *(radius * radius for radius in radii))
        return (*squares, centre, *values)

    return LensProfile(
        name=name,
        distortion=None if distortion is None else ("table", table(*distortion, 0.0)),
        tca=None,
        vignetting=None if vignetting is None else ("table", table(*vignetting, 100.0)),
        distortion_unit_mm=unit,
        vignetting_unit_mm=unit,
        sensor_diagonal_mm=2.0,
        from_camera=True,
    )


def _look_up(terms: tuple[float, ...], radius: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return a ``table`` model's value at each *radius*."""
    half = len(terms) // 2
    return np.interp(radius * radius, terms[:half], terms[half:]).astype(np.float32)


# --------------------------------------------------------------------------
# Correcting a photo (PLAN.md section 6 step 1.5)
# --------------------------------------------------------------------------


def _distortion_factor(
    profile: LensProfile, radius: NDArray[np.float32], amount: float
) -> NDArray[np.float32] | float:
    """Return how far out the lens put each radius, as a multiple of it.

    *amount* is the fraction of the profile to apply: 0 leaves every point
    where it is, 1 is the profile, 2 twice as much.
    """
    if profile.distortion is None or amount == 0.0:
        return 1.0
    model, terms = profile.distortion
    if model == "ptlens":
        a, b, c = terms
        factor = ((a * radius + b) * radius + c) * radius + (1.0 - a - b - c)
    elif model == "poly3":
        (k1,) = terms
        factor = (1.0 - k1) + k1 * radius * radius
    elif model == "table":
        factor = 1.0 + _look_up(terms, radius) / np.float32(100.0)
    else:  # poly5
        k1, k2 = terms
        square = radius * radius
        factor = 1.0 + (k1 + k2 * square) * square
    return 1.0 + amount * (factor - 1.0)


def _aberration_factors(
    profile: LensProfile, radius: NDArray[np.float32] | float
) -> tuple[NDArray[np.float32] | float, NDArray[np.float32] | float]:
    """Return how far out red and blue landed where green landed at *radius*."""
    if profile.tca is None:
        return 1.0, 1.0
    model, red, blue = profile.tca
    if model == "linear":
        return red[0], blue[0]

    def factor(terms: tuple[float, ...]) -> NDArray[np.float32] | float:
        v, c, b = terms
        return (b * radius + c) * radius + v

    return factor(red), factor(blue)


def _vignetting_gain(
    profile: LensProfile, radius: NDArray[np.float32], amount: float
) -> NDArray[np.float32] | None:
    """Return what to multiply the light by at *radius*, or ``None`` for nothing."""
    if profile.vignetting is None or amount == 0.0:
        return None
    model, terms = profile.vignetting
    if model == "table":
        falloff = _look_up(terms, radius) / np.float32(100.0)
    else:  # pa
        k1, k2, k3 = terms
        square = radius * radius
        falloff = 1.0 + square * (k1 + square * (k2 + square * k3))
    falloff = np.maximum(falloff, np.float32(VIGNETTING_FLOOR))
    return np.power(falloff, np.float32(-amount)).astype(np.float32)


@dataclass(frozen=True)
class _Plan:
    """What correcting one photo involves, worked out once."""

    #: Millimetres on the sensor per pixel of the full size photo.
    mm_per_pixel: float
    #: The full size the photo's own pixels are measured against.
    full_size: tuple[int, int]
    #: How far the corrected picture is zoomed so that no edge is empty.
    scale: float
    distortion: float
    vignetting: float
    moves: bool
    brightens: bool


@lru_cache(maxsize=32)
def _plan(profile: LensProfile, settings: LensSettings, full_size: tuple[int, int]) -> _Plan:
    """Return how to correct a photo of *full_size* taken with *profile*.

    Worked out from the full size alone, so that the preview proxy, a window
    of the full photo and the exported file all get the same zoom and every
    pixel lands in the same place in each.
    """
    width, height = full_size
    mm_per_pixel = profile.sensor_diagonal_mm / math.hypot(width, height)
    distortion = settings.distortion / 100.0 if profile.distortion is not None else 0.0
    vignetting = settings.vignetting / 100.0 if profile.vignetting is not None else 0.0
    moves = distortion != 0.0 or profile.tca is not None
    plan = _Plan(
        mm_per_pixel=mm_per_pixel,
        full_size=full_size,
        scale=1.0,
        distortion=distortion,
        vignetting=vignetting,
        moves=moves,
        brightens=vignetting != 0.0,
    )
    if moves:
        plan = replace(plan, scale=_fill_scale(profile, plan))
    return plan


def _sources(
    profile: LensProfile, plan: _Plan, dx: NDArray[np.float32], dy: NDArray[np.float32]
) -> tuple[
    tuple[NDArray[np.float32], NDArray[np.float32]],
    tuple[NDArray[np.float32], NDArray[np.float32]],
    tuple[NDArray[np.float32], NDArray[np.float32]],
    NDArray[np.float32],
]:
    """Return where red, green and blue come from for output offsets *dx*, *dy*.

    Offsets are in pixels of the full size photo from the centre of the
    frame, and so are the sources. The fourth array is the radius the green
    source sits at in vignetting units, for the brightness correction.
    """
    to_units = np.float32(plan.mm_per_pixel / profile.distortion_unit_mm / plan.scale)
    ux = dx / np.float32(plan.scale)
    uy = dy / np.float32(plan.scale)
    radius = np.sqrt(dx * dx + dy * dy) * to_units
    bend = _distortion_factor(profile, radius, plan.distortion)
    green = (ux * bend, uy * bend)
    landed = radius * bend
    red_factor, blue_factor = _aberration_factors(profile, landed)
    red = (green[0] * red_factor, green[1] * red_factor)
    blue = (green[0] * blue_factor, green[1] * blue_factor)
    vignetting_radius = landed * np.float32(
        profile.distortion_unit_mm / profile.vignetting_unit_mm
    )
    return red, green, blue, np.asarray(vignetting_radius, dtype=np.float32)


def _fill_scale(profile: LensProfile, plan: _Plan) -> float:
    """Return the least zoom at which every edge of the frame has picture.

    Straightening a barrel shaped lens pulls the corners in and pushes the
    middle of each edge out of the frame, and a pincushion lens the other
    way round. The frame's outline is sampled and the zoom found by halving,
    which works for the wavy "moustache" shapes too, as long as the lens does
    not fold the picture back on itself.
    """
    width, height = plan.full_size
    half_w = (width - 1) / 2.0
    half_h = (height - 1) / 2.0
    along = np.linspace(-1.0, 1.0, _EDGE_SAMPLES, dtype=np.float32)
    dx = np.concatenate(
        [along * half_w, along * half_w, np.full_like(along, -half_w), np.full_like(along, half_w)]
    )
    dy = np.concatenate(
        [np.full_like(along, -half_h), np.full_like(along, half_h), along * half_h, along * half_h]
    )

    def fits(scale: float) -> bool:
        trial = replace(plan, scale=scale)
        for sx, sy in _sources(profile, trial, dx, dy)[:3]:
            if np.any(np.abs(sx) > half_w + 1e-3) or np.any(np.abs(sy) > half_h + 1e-3):
                return False
        return True

    low, high = _SCALE_LOW, _SCALE_HIGH
    if fits(low):
        return low
    if not fits(high):
        logger.warning("%s bends too far to fill the frame; it is zoomed as far as allowed", profile.name)
        return high
    for _ in range(_SCALE_STEPS):
        middle = (low + high) / 2.0
        if fits(middle):
            high = middle
        else:
            low = middle
    return high


def is_active(photo: "DecodedPhoto", settings: LensSettings | None) -> bool:
    """Return whether correcting *photo* with *settings* would change it."""
    if settings is None or not settings.enabled or photo.lens is None:
        return False
    plan = _plan(photo.lens, settings, photo.full_size)
    return plan.moves or plan.brightens


def correct(photo: "DecodedPhoto", settings: LensSettings | None) -> "DecodedPhoto":
    """Return *photo* with its lens corrected (PLAN.md section 6 step 1.5).

    The photo comes back the same size: the corrected picture is zoomed just
    far enough to fill the frame, so nothing downstream - the crop, the
    rotation, the sizes the export dialog reports - has to know it happened.
    What comes back no longer carries the profile, so it cannot be corrected
    twice. A photo with nothing to correct comes back as it is.
    """
    if not is_active(photo, settings):
        return photo
    assert photo.lens is not None and settings is not None
    profile = photo.lens
    plan = _plan(profile, settings, photo.full_size)

    image = photo.image
    height, width = image.shape[:2]
    spacing = max(1, min(NODE_SPACING, min(width, height) // _NODES_ACROSS))
    strip_rows = spacing * max(1, STRIP_ROWS // spacing)
    nodes = _nodes(profile, plan, (width, height), spacing)

    out = np.empty_like(image)
    clipped = np.empty_like(photo.clipped)
    mask = photo.clipped.astype(np.uint8) if plan.moves else None
    # Each colour moved on its own only when the aberration can be seen.
    apart = plan.moves and nodes.aberration > INVISIBLE_ABERRATION_PX
    planes = cv2.split(image) if apart else []

    for top in range(0, height, strip_rows):
        bottom = min(height, top + strip_rows)

        def spread(values: NDArray[np.float32]) -> NDArray[np.float32]:
            return _spread(values, top, bottom, width, spacing)

        gain = None if nodes.gain is None else spread(nodes.gain)
        target = out[top:bottom]
        if apart:
            assert mask is not None
            channels = []
            for channel, (map_x, map_y) in enumerate(nodes.sources):
                xs, ys = spread(map_x), spread(map_y)
                moved = cv2.remap(
                    planes[channel], xs, ys, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
                )
                channels.append(moved if gain is None else cv2.multiply(moved, gain))
                if channel == 1:
                    clipped[top:bottom] = _landed(mask, xs, ys)
            cv2.merge(channels, dst=target)
        elif plan.moves:
            assert mask is not None
            xs, ys = spread(nodes.sources[1][0]), spread(nodes.sources[1][1])
            cv2.remap(
                image, xs, ys, cv2.INTER_CUBIC, dst=target, borderMode=cv2.BORDER_REPLICATE
            )
            if gain is not None:
                cv2.multiply(target, cv2.merge((gain, gain, gain)), dst=target)
            clipped[top:bottom] = _landed(mask, xs, ys)
        else:
            assert gain is not None  # with nothing to move, there is light to restore
            cv2.multiply(image[top:bottom], cv2.merge((gain, gain, gain)), dst=target)
            clipped[top:bottom] = photo.clipped[top:bottom]

    return replace(photo, image=out, clipped=clipped, lens=None)


def _landed(
    mask: NDArray[np.uint8], xs: NDArray[np.float32], ys: NDArray[np.float32]
) -> NDArray[np.bool_]:
    """Return the clipped pixels moved the way the picture was."""
    return cv2.remap(mask, xs, ys, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0


def _spread(
    values: NDArray[np.float32], top: int, bottom: int, width: int, spacing: int
) -> NDArray[np.float32]:
    """Return rows *top* to *bottom* of the image interpolated from its *values* at the nodes.

    The nodes sit half a step in from where OpenCV's resize expects pixel
    centres, with one node of margin on every side (see :func:`_nodes`), so
    stretching a block of them by exactly *spacing* puts every pixel at its
    true place between nodes. Resizing interpolates floating point values
    with floating point weights, which matters here: the values are positions
    in pixels, and rounding the weights would move them.
    """
    first = top // spacing
    last = (bottom - 1) // spacing + 3
    block = values[first:last]
    stretched = cv2.resize(
        block,
        (block.shape[1] * spacing, block.shape[0] * spacing),
        interpolation=cv2.INTER_LINEAR,
    )
    offset = top - first * spacing + spacing
    return stretched[offset:offset + bottom - top, spacing:spacing + width]


@dataclass(frozen=True)
class _Nodes:
    """The corrections worked out exactly on a coarse grid of one image."""

    #: For red, green and blue, where each node's light comes from, as the
    #: column and the row of the image.
    sources: tuple[tuple[NDArray[np.float32], NDArray[np.float32]], ...]
    #: What each node's light is multiplied by, or ``None`` for nothing.
    gain: NDArray[np.float32] | None
    #: How far, at most, red or blue is moved apart from green, in pixels.
    aberration: float


def _nodes(profile: LensProfile, plan: _Plan, size: tuple[int, int], spacing: int) -> _Nodes:
    """Return the corrections on a grid of nodes over an image of *size*.

    Node *j* stands at pixel ``(j - 0.5) * spacing - 0.5``: that is where
    OpenCV's resize puts the centre of a source pixel once it has been
    stretched by the spacing, shifted by one node so that there is a node of
    margin before the first pixel and after the last.
    """
    width, height = size
    full_width, full_height = plan.full_size
    across = width / full_width
    down = height / full_height
    count_x = (width - 1) // spacing + 3
    count_y = (height - 1) // spacing + 3
    xs = (np.arange(count_x, dtype=np.float64) - 0.5) * spacing - 0.5
    ys = (np.arange(count_y, dtype=np.float64) - 0.5) * spacing - 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    centre_x, centre_y = (width - 1) / 2.0, (height - 1) / 2.0
    # Offsets from the centre in pixels of the full size photo, which is what
    # the plan is worked out in, so that every size lands alike.
    dx = ((grid_x - centre_x) / across).astype(np.float32)
    dy = ((grid_y - centre_y) / down).astype(np.float32)
    red, green, blue, vignetting_radius = _sources(profile, plan, dx, dy)

    sources = tuple(
        (
            (np.asarray(sx, dtype=np.float64) * across + centre_x).astype(np.float32),
            (np.asarray(sy, dtype=np.float64) * down + centre_y).astype(np.float32),
        )
        for sx, sy in (red, green, blue)
    )
    gain = _vignetting_gain(profile, vignetting_radius, plan.vignetting)
    green_x, green_y = sources[1]
    aberration = max(
        float(np.max(np.hypot(sx - green_x, sy - green_y))) for sx, sy in (sources[0], sources[2])
    )
    return _Nodes(
        sources=sources,
        gain=None if gain is None else gain.astype(np.float32),
        aberration=aberration,
    )


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _number(text: Any) -> float | None:
    """Return *text* as a finite float, or ``None``."""
    if text is None or isinstance(text, bool):
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _amount(value: Any) -> float:
    """Return a stored amount, or the default when it is not a number."""
    number = _number(value)
    return AMOUNT_DEFAULT if number is None else _clamp(number, AMOUNT_MIN, AMOUNT_MAX)


def _clamp(value: float, low: float, high: float) -> float:
    """Return *value* held between *low* and *high*."""
    return max(low, min(high, value))
