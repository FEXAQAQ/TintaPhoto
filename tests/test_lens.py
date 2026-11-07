"""Lens correction (PLAN.md 2.2, A-26 to A-28, and section 6 step 1.5)."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from core import decoder, geometry, lens, sidecar
from core.decoder import DecodedPhoto
from core.edit_state import Edit, EditState
from core.geometry import Geometry
from core.lens import LensProfile, LensSettings, LensShot

SAMPLES = Path(__file__).resolve().parent / "samples"

#: A small database written for these tests, so that the arithmetic can be
#: checked against numbers chosen by hand rather than against Lensfun's.
SMALL_DATABASE = """<lensdatabase version="2">
    <camera>
        <maker>Maker</maker>
        <model>Body One</model>
        <mount>Test</mount>
        <cropfactor>1.5</cropfactor>
    </camera>
    <lens>
        <maker>Maker</maker>
        <model>Zoom 20-40mm</model>
        <mount>Test</mount>
        <cropfactor>1.5</cropfactor>
        <calibration>
            <distortion model="ptlens" focal="20" a="0.01" b="-0.04" c="0.0"/>
            <distortion model="ptlens" focal="40" a="0.03" b="0.00" c="0.02"/>
            <tca model="poly3" focal="20" vr="1.001" vb="0.999"/>
            <vignetting model="pa" focal="20" aperture="4" distance="1000" k1="-0.4" k2="0" k3="0"/>
            <vignetting model="pa" focal="20" aperture="8" distance="1000" k1="-0.2" k2="0" k3="0"/>
            <vignetting model="pa" focal="20" aperture="8" distance="1" k1="-0.9" k2="0" k3="0"/>
        </calibration>
    </lens>
    <lens>
        <maker>Maker</maker>
        <model>MK18-55mmF3.5</model>
        <model lang="en">MK 18-55mm f/3.5</model>
        <mount>Test</mount>
        <cropfactor>1.5</cropfactor>
        <calibration>
            <distortion model="poly5" focal="18" k1="-0.02" k2="0.001"/>
        </calibration>
    </lens>
    <lens>
        <maker>Maker</maker>
        <model>Fish 8mm</model>
        <mount>Test</mount>
        <type>fisheye</type>
        <cropfactor>1.5</cropfactor>
        <calibration>
            <distortion model="poly3" focal="8" k1="-0.1"/>
        </calibration>
    </lens>
    <lens>
        <maker>Maker</maker>
        <model>Prime 50mm</model>
        <mount>Test</mount>
        <cropfactor>1.5</cropfactor>
    </lens>
</lensdatabase>
"""


@pytest.fixture
def small(tmp_path: Path) -> lens.Database:
    (tmp_path / "test.xml").write_text(SMALL_DATABASE, encoding="utf-8")
    return lens.Database.load(tmp_path)


def _shot(model: str = "Zoom 20-40mm", focal: float = 20.0, aperture: float = 8.0) -> LensShot:
    return LensShot(
        lens_model=model,
        camera_maker="MAKER CORPORATION",
        camera_model="Body One",
        focal_length=focal,
        aperture=aperture,
    )


def _profile(**changes: object) -> LensProfile:
    """Return a full frame profile with nothing in it, for changing."""
    base = LensProfile(
        name="Test",
        distortion=None,
        tca=None,
        vignetting=None,
        distortion_unit_mm=12.0,
        vignetting_unit_mm=math.hypot(36.0, 24.0) / 2.0,
        sensor_diagonal_mm=math.hypot(36.0, 24.0),
    )
    return replace(base, **changes)


def _photo(image: np.ndarray, profile: LensProfile | None, **changes: object) -> DecodedPhoto:
    return DecodedPhoto(
        image=image.astype(np.float32),
        is_raw=True,
        clipped=np.zeros(image.shape[:2], dtype=bool),
        lens=profile,
        **changes,
    )


ON = LensSettings(enabled=True)


# -- the database -----------------------------------------------------------


def test_a_lens_name_is_shown_with_its_maker(small: lens.Database) -> None:
    """A database's English name can leave the maker out; the panel puts it in."""
    profile = small.profile_for(_shot("MK18-55mmF3.5", focal=18.0))
    assert profile is not None
    assert profile.name == "Maker MK 18-55mm f/3.5"
    assert profile.distortion == ("poly5", (-0.02, 0.001))


def test_a_photo_without_a_lens_name_has_no_profile() -> None:
    """Some bodies keep the lens only in their maker notes, which LibRaw does not read."""
    assert lens.profile_for(LensShot(lens_model="")) is None
    assert lens.profile_for(None) is None


def test_the_bundled_database_is_read() -> None:
    database = lens.database()
    assert len(database.lenses) > 1000
    assert len(database.cameras) > 500


def test_the_camera_gives_the_crop_factor(small: lens.Database) -> None:
    assert small.crop_factor_of(_shot()) == pytest.approx(1.5)


def test_an_unknown_camera_falls_back_on_the_equivalent_focal_length(
    small: lens.Database,
) -> None:
    shot = replace(_shot(), camera_model="Body Two", focal_length=20.0, focal_length_35mm=30.0)
    assert small.crop_factor_of(shot) == pytest.approx(1.5)


def test_lens_names_are_matched_loosely(small: lens.Database) -> None:
    assert small.profile_for(_shot("zoom  20-40MM")) is not None
    assert small.profile_for(_shot("Zoom 20-40 mm")) is not None
    assert small.profile_for(_shot("Zoom 24-70mm")) is None


def test_a_fisheye_or_an_uncalibrated_lens_is_not_offered(small: lens.Database) -> None:
    """Straightening a fisheye with a rectilinear model would bend it wrong."""
    assert small.profile_for(_shot("Fish 8mm", focal=8.0)) is None
    assert small.profile_for(_shot("Prime 50mm", focal=50.0)) is None


def test_distortion_is_interpolated_between_focal_lengths(small: lens.Database) -> None:
    at_ends = small.profile_for(_shot(focal=20.0)), small.profile_for(_shot(focal=40.0))
    middle = small.profile_for(_shot(focal=30.0))
    beyond = small.profile_for(_shot(focal=60.0))
    assert at_ends[0] is not None and at_ends[1] is not None
    assert middle is not None and beyond is not None

    assert at_ends[0].distortion == ("ptlens", (0.01, -0.04, 0.0))
    assert middle.distortion is not None
    assert middle.distortion[1] == pytest.approx((0.02, -0.02, 0.01))
    # Past the end the nearest calibration is used as it is.
    assert beyond.distortion == at_ends[1].distortion


def test_vignetting_uses_the_far_focus_and_interpolates_in_stops(
    small: lens.Database,
) -> None:
    """f/5.6 is one stop from f/4 and one from f/8: half way between."""
    at_eight = small.profile_for(_shot(aperture=8.0))
    between = small.profile_for(_shot(aperture=4.0 * math.sqrt(2.0)))
    assert at_eight is not None and between is not None

    assert at_eight.vignetting is not None and between.vignetting is not None
    assert at_eight.vignetting[0] == between.vignetting[0] == "pa"
    assert at_eight.vignetting[1] == pytest.approx((-0.2, 0.0, 0.0))
    assert between.vignetting[1] == pytest.approx((-0.3, 0.0, 0.0))


def test_units_follow_the_calibration_and_the_photo(small: lens.Database) -> None:
    profile = small.profile_for(_shot())
    assert profile is not None
    diagonal = math.hypot(36.0, 24.0) / 1.5
    short_side = diagonal / math.sqrt(1.0 + 1.5**2)
    assert profile.distortion_unit_mm == pytest.approx(short_side / 2.0)
    assert profile.vignetting_unit_mm == pytest.approx(diagonal / 2.0)
    assert profile.sensor_diagonal_mm == pytest.approx(diagonal)


def test_a_broken_database_file_costs_only_its_own_lenses(tmp_path: Path) -> None:
    (tmp_path / "good.xml").write_text(SMALL_DATABASE, encoding="utf-8")
    (tmp_path / "broken.xml").write_text("<lensdatabase><lens>", encoding="utf-8")
    database = lens.Database.load(tmp_path)
    assert database.profile_for(_shot()) is not None


# -- the settings -----------------------------------------------------------


def test_a_raw_starts_corrected_and_a_jpeg_does_not() -> None:
    assert LensSettings.for_photo(True).enabled
    assert not LensSettings.for_photo(False).enabled


def test_settings_survive_the_parameter_file() -> None:
    settings = LensSettings(enabled=True, distortion=60.0, vignetting=150.0)
    assert LensSettings.from_dict(json.loads(json.dumps(settings.to_dict()))) == settings


def test_an_older_parameter_file_reads_as_switched_off() -> None:
    """Its crop was made without the correction (PLAN.md 7.1)."""
    assert LensSettings.from_dict(None) == LensSettings(enabled=False)
    assert Geometry.from_dict({"angle": 1.0}).lens == LensSettings(enabled=False)


def test_odd_stored_values_fall_back_or_are_held_in_range() -> None:
    read = LensSettings.from_dict({"enabled": "yes", "distortion": 900, "vignetting": "x"})
    assert read == LensSettings(enabled=False, distortion=200.0, vignetting=100.0)
    assert LensSettings().with_amount("distortion", -5.0).distortion == 0.0
    with pytest.raises(KeyError):
        LensSettings().with_amount("enabled", 1.0)


# -- correcting -------------------------------------------------------------


def test_nothing_happens_without_a_profile_or_with_the_switch_off() -> None:
    image = np.full((40, 60, 3), 0.5, dtype=np.float32)
    bare = _photo(image, None)
    profiled = _photo(image, _profile(vignetting=("pa", (-0.5, 0.0, 0.0))))

    assert lens.correct(bare, ON) is bare
    assert lens.correct(profiled, LensSettings(enabled=False)) is profiled
    assert lens.correct(profiled, None) is profiled
    assert lens.correct(profiled, replace(ON, vignetting=0.0)) is profiled


def test_vignetting_is_taken_out_of_an_evenly_lit_wall() -> None:
    """A flat field darkened by the profile comes back flat."""
    height, width = 200, 300
    profile = _profile(vignetting=("pa", (-0.5, 0.1, -0.05)))
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    half_diagonal = math.hypot(width, height) / 2.0
    radius = np.hypot(x - (width - 1) / 2.0, y - (height - 1) / 2.0) / half_diagonal
    falloff = 1.0 - 0.5 * radius**2 + 0.1 * radius**4 - 0.05 * radius**6
    wall = np.repeat((0.4 * falloff)[..., None], 3, axis=2)

    corrected = lens.correct(_photo(wall, profile), ON)

    assert corrected.image.shape == wall.shape
    # Within a thousandth: the gain is interpolated between nodes.
    assert np.abs(corrected.image / 0.4 - 1.0).max() < 1e-3
    assert corrected.lens is None  # it cannot be corrected twice


def test_half_the_amount_takes_out_half_the_falloff_in_stops() -> None:
    profile = _profile(vignetting=("pa", (-0.75, 0.0, 0.0)))
    wall = np.full((101, 151, 3), 0.25, dtype=np.float32)
    full = lens.correct(_photo(wall, profile), ON).image
    half = lens.correct(_photo(wall, profile), replace(ON, vignetting=50.0)).image
    corner_full, corner_half = math.log2(full[0, 0, 1] / 0.25), math.log2(half[0, 0, 1] / 0.25)
    assert corner_half == pytest.approx(corner_full / 2.0, rel=1e-3)


def _distort(scene: np.ndarray, profile: LensProfile) -> np.ndarray:
    """Return *scene* as the lens of *profile* would have recorded it.

    For every recorded pixel the point of the scene that landed there is
    found by inverting the distortion numerically, which is the one thing
    the correction itself never has to do.
    """
    height, width = scene.shape[:2]
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    dx, dy = x - (width - 1) / 2.0, y - (height - 1) / 2.0
    mm_per_pixel = profile.sensor_diagonal_mm / math.hypot(width, height)
    landed = np.hypot(dx, dy) * mm_per_pixel / profile.distortion_unit_mm
    assert profile.distortion is not None
    a, b, c = profile.distortion[1]
    radius = landed.copy()
    for _ in range(30):  # Newton on rd(ru) - landed = 0
        value = radius * (a * radius**3 + b * radius**2 + c * radius + 1 - a - b - c) - landed
        slope = 4 * a * radius**3 + 3 * b * radius**2 + 2 * c * radius + (1 - a - b - c)
        radius -= value / slope
    ratio = np.divide(radius, landed, out=np.ones_like(radius), where=landed > 0)
    map_x = (dx * ratio + (width - 1) / 2.0).astype(np.float32)
    map_y = (dy * ratio + (height - 1) / 2.0).astype(np.float32)
    return cv2.remap(scene, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _scene(height: int, width: int) -> np.ndarray:
    """A smooth pattern of straight stripes both ways, easy to resample."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    value = 0.5 + 0.2 * np.sin(x / 9.0) + 0.2 * np.sin(y / 7.0)
    return np.repeat(value[..., None], 3, axis=2).astype(np.float32)


@pytest.mark.parametrize("terms", [(0.0, -0.05, 0.0), (0.02, 0.03, 0.0), (0.03, -0.06, 0.02)])
def test_a_distorted_scene_comes_back_straight(terms: tuple[float, float, float]) -> None:
    """Barrel, pincushion and a moustache: the lines come back where they were.

    The correction also zooms by its own fill scale, so the scene is compared
    zoomed by the same amount about the centre.
    """
    height, width = 240, 360
    profile = _profile(distortion=("ptlens", terms))
    scene = _scene(height, width)
    recorded = _distort(scene, profile)

    corrected = lens.correct(_photo(recorded, profile), ON).image
    scale = lens._plan(profile, ON, (width, height)).scale
    # warpAffine inverts its matrix: this one shows the scene at scene(p / scale).
    zoom = np.float32([[scale, 0, (1 - scale) * (width - 1) / 2],
                       [0, scale, (1 - scale) * (height - 1) / 2]])
    expected = cv2.warpAffine(scene, zoom, (width, height), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    inner = (slice(20, -20), slice(20, -20))
    assert np.abs(corrected[inner] - expected[inner]).max() < 0.02


@pytest.mark.parametrize("terms", [(0.0, -0.05, 0.0), (0.02, 0.03, 0.0), (0.03, -0.06, 0.02)])
def test_the_corrected_frame_has_picture_up_to_every_edge(
    terms: tuple[float, float, float],
) -> None:
    """C-05: no empty edge, and no more zoom than that needs."""
    width, height = 600, 400
    profile = _profile(distortion=("ptlens", terms), tca=("poly3", (1.002, 0.0, 0.0), (0.998, 0.0, 0.0)))
    plan = lens._plan(profile, ON, (width, height))
    half_w, half_h = (width - 1) / 2.0, (height - 1) / 2.0
    along = np.linspace(-1.0, 1.0, 401, dtype=np.float32)
    dx = np.concatenate([along * half_w, along * half_w, np.full_like(along, -half_w), np.full_like(along, half_w)])
    dy = np.concatenate([np.full_like(along, -half_h), np.full_like(along, half_h), along * half_h, along * half_h])

    def worst(scale: float) -> float:
        sources = lens._sources(profile, replace(plan, scale=scale), dx, dy)[:3]
        return max(max(np.abs(sx).max() / half_w, np.abs(sy).max() / half_h) for sx, sy in sources)

    assert worst(plan.scale) <= 1.0 + 1e-4
    assert worst(plan.scale * 0.995) > 1.0


def test_red_and_blue_are_moved_onto_green() -> None:
    """Lateral aberration: a channel scaled by the lens is scaled back."""
    height, width = 240, 360
    profile = _profile(tca=("linear", (1.004,), (0.996,)))
    scene = _scene(height, width)
    recorded = scene.copy()
    for channel, factor in ((0, 1.004), (2, 0.996)):
        grow = np.float32([[factor, 0, (1 - factor) * (width - 1) / 2],
                           [0, factor, (1 - factor) * (height - 1) / 2]])
        recorded[..., channel] = cv2.warpAffine(scene[..., channel], grow, (width, height),
                                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    assert np.abs(recorded[..., 0] - recorded[..., 1]).max() > 0.02

    corrected = lens.correct(_photo(recorded, profile), ON).image
    inner = (slice(20, -20), slice(20, -20))
    assert np.abs(corrected[inner][..., 0] - corrected[inner][..., 1]).max() < 0.01
    assert np.abs(corrected[inner][..., 2] - corrected[inner][..., 1]).max() < 0.01


def test_the_proxy_and_the_full_size_photo_are_corrected_alike() -> None:
    """Every pixel lands in the same place at any size (PLAN.md 6, preview = export)."""
    profile = _profile(distortion=("ptlens", (0.02, -0.05, 0.01)), vignetting=("pa", (-0.4, 0.0, 0.0)))
    full = _scene(800, 1200)
    proxy = cv2.resize(full, (600, 400), interpolation=cv2.INTER_AREA)

    big = lens.correct(_photo(full, profile), ON).image
    small = lens.correct(_photo(proxy, profile, shrunk_from=(1200, 800)), ON).image

    shrunk = cv2.resize(big, (600, 400), interpolation=cv2.INTER_AREA)
    inner = (slice(10, -10), slice(10, -10))
    assert np.abs(shrunk[inner] - small[inner]).mean() < 0.003


def test_the_clipped_mask_moves_with_the_picture() -> None:
    profile = _profile(distortion=("ptlens", (0.0, -0.08, 0.0)))
    image = np.full((200, 300, 3), 0.3, dtype=np.float32)
    clipped = np.zeros((200, 300), dtype=bool)
    clipped[95:105, 145:155] = True  # in the middle, where nothing moves much
    photo = replace(_photo(image, profile), clipped=clipped)

    corrected = lens.correct(photo, ON)
    assert corrected.clipped.dtype == np.bool_
    assert corrected.clipped.shape == clipped.shape
    assert corrected.clipped[100, 150]
    assert abs(int(corrected.clipped.sum()) - 100) < 40


def test_the_geometry_step_corrects_the_lens_first() -> None:
    profile = _profile(vignetting=("pa", (-0.5, 0.0, 0.0)))
    photo = _photo(np.full((40, 60, 3), 0.2, dtype=np.float32), profile)

    plain = geometry.apply(photo, Geometry())
    corrected = geometry.apply(photo, Geometry(lens=ON))
    rotated = geometry.apply(photo, Geometry(lens=ON, angle=3.0), crop=False)

    assert plain is photo
    assert corrected.image[0, 0, 1] > 0.2
    assert rotated.lens is None


# -- the edit and its parameter file ----------------------------------------


def test_a_raw_is_untouched_at_its_own_lens_setting() -> None:
    reference = Edit(EditState.for_photo(True, 5000.0, 0.0), Geometry(lens=LensSettings.for_photo(True)))
    switched_off = replace(reference, geometry=Geometry(lens=LensSettings(enabled=False)))
    assert reference.is_untouched(reference)
    assert not switched_off.is_untouched(reference)


def test_the_lens_setting_is_stored_with_the_geometry(tmp_path: Path) -> None:
    photo = tmp_path / "DSC0001.ARW"
    photo.write_bytes(b"raw")
    state = EditState.for_photo(True, 5000.0, 0.0)
    reference = Edit(state, Geometry(lens=LensSettings.for_photo(True)))
    edit = Edit(state, Geometry(lens=LensSettings(enabled=True, distortion=50.0)))
    application = sidecar.Application("TintaPhoto", "test")

    store = tmp_path / "store"
    sidecar.save(photo, edit, reference, application, store)
    stored = sidecar.read(photo, state, store)

    assert stored is not None
    assert stored.geometry.lens == LensSettings(enabled=True, distortion=50.0)


def test_pasting_onto_a_raw_without_a_file_keeps_its_lens_on(tmp_path: Path) -> None:
    """Writing the first parameter file must not switch the correction off."""
    raw = tmp_path / "DSC0002.ARW"
    jpeg = tmp_path / "DSC0002.JPG"
    raw.write_bytes(b"raw")
    jpeg.write_bytes(b"jpeg")
    application = sidecar.Application("TintaPhoto", "test")

    store = tmp_path / "store"
    sidecar.update_adjustments(raw, {"exposure": 0.5}, application, store)
    sidecar.update_adjustments(jpeg, {"exposure": 0.5}, application, store)

    assert sidecar.read(raw, EditState.for_photo(True), store).geometry.lens.enabled  # type: ignore[union-attr]
    assert not sidecar.read(jpeg, EditState.for_photo(False), store).geometry.lens.enabled  # type: ignore[union-attr]


# -- real photos --------------------------------------------------------------


@pytest.mark.parametrize("name", ["_DSC2828.ARW", "DSCF3335.RAF"])
def test_a_sample_raw_is_decoded_with_its_lens_profile(name: str) -> None:
    """Both sample RAW files were taken with lenses the database knows.

    One is a TIFF underneath and one is not, which are the two ways the
    camera's name and equivalent focal length have to be read.
    """
    path = SAMPLES / name
    if not path.is_file():
        pytest.skip(f"{name} is not in tests/samples")
    photo = decoder.decode_proxy(path)
    assert photo.lens is not None
    assert photo.lens.name.strip()
    assert photo.lens.distortion is not None and photo.lens.vignetting is not None


def test_a_sample_jpeg_has_no_lens_profile() -> None:
    path = SAMPLES / "_DSC2828.JPG"
    if not path.is_file():
        pytest.skip("_DSC2828.JPG is not in tests/samples")
    assert decoder.decode_proxy(path).lens is None
