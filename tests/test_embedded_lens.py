"""Tests for the lens corrections a camera stores in its RAW file (PLAN.md 6 step 1.5)."""

from __future__ import annotations

import struct
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from core import decoder, embedded_lens, lens
from core.lens import LensSettings

SAMPLES = Path(__file__).resolve().parent / "samples"

RADII = (0.5, 0.75, 1.0)
STRETCH = (1.0, 2.25, 4.0)
LIGHT = (98.0, 95.0, 90.0)


def _rationals(values: tuple[float, ...], order: str) -> bytes:
    """Return *values* as signed TIFF fractions."""
    out = b""
    for value in values:
        fraction = Fraction(value).limit_denominator(1 << 20)
        out += struct.pack(order + "ii", fraction.numerator, fraction.denominator)
    return out


def _raf(
    distortion: tuple[float, ...] | None = (0.0, *RADII, *STRETCH),
    vignetting: tuple[float, ...] | None = (0.0, *RADII, *LIGHT),
    section_magic: bytes | None = None,
    order: str = "<",
) -> bytes:
    """Return a file laid out as a RAF is, holding only what the reader looks at.

    The header points at a section that begins with a TIFF header; its first
    directory points with tag 0xF000 to the camera's own directory, which
    holds the two tables as signed fractions.
    """
    start = 128
    tables = [(tag, values) for tag, values in ((0xF00B, distortion), (0xF010, vignetting)) if values]
    # Section layout: TIFF header (8), first directory, camera directory, table data.
    first_at = 8
    first_size = 2 + 12 + 4
    camera_at = first_at + first_size
    camera_size = 2 + 12 * len(tables) + 4
    data_at = camera_at + camera_size

    data = b""
    entries = b""
    for tag, values in tables:
        entries += struct.pack(order + "HHII", tag, 10, len(values), data_at + len(data))
        data += _rationals(values, order)
    if section_magic is None:
        section_magic = b"II*\0" if order == "<" else b"MM\0*"
    section = section_magic + struct.pack(order + "I", first_at)
    section += struct.pack(order + "H", 1) + struct.pack(order + "HHII", 0xF000, 4, 1, camera_at)
    section += struct.pack(order + "I", 0)
    section += struct.pack(order + "H", len(tables)) + entries + struct.pack(order + "I", 0)
    section += data

    header = bytearray(start)
    header[100:108] = struct.pack(">II", start, len(section))
    return bytes(header) + section


def _write(tmp_path: Path, content: bytes, name: str = "DSCF0001.RAF") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


# -- reading the file ---------------------------------------------------------


def test_both_tables_are_read(tmp_path: Path) -> None:
    profile = embedded_lens.profile_for(_write(tmp_path, _raf()), "Zoom 15-45mm")

    assert profile is not None
    assert profile.from_camera
    assert profile.name == "Zoom 15-45mm"
    assert profile.tca is None
    assert profile.distortion is not None and profile.distortion[0] == "table"
    assert profile.vignetting is not None and profile.vignetting[0] == "table"
    assert profile.distortion_unit_mm == profile.vignetting_unit_mm == embedded_lens.TABLE_UNIT
    assert profile.sensor_diagonal_mm == pytest.approx(2.0)


@pytest.mark.parametrize("order", ["<", ">"], ids=["little endian", "big endian"])
def test_the_tables_hold_their_radii_and_values(tmp_path: Path, order: str) -> None:
    """The reader follows the byte order the section's TIFF header names."""
    path = _write(tmp_path, _raf(order=order))
    with path.open("rb") as handle:
        tables = embedded_lens._read_tables(handle)
    assert tables is not None
    distortion, vignetting = tables
    assert distortion == (pytest.approx(RADII, abs=1e-6), pytest.approx(STRETCH, abs=1e-6))
    assert vignetting == (pytest.approx(RADII, abs=1e-6), pytest.approx(LIGHT, abs=1e-6))


def test_only_a_raf_is_looked_at(tmp_path: Path) -> None:
    assert embedded_lens.profile_for(_write(tmp_path, _raf(), "DSC0001.ARW"), "x") is None


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"\0" * 64,
        _raf()[:200],
        _raf(section_magic=b"JUNK"),
        _raf(distortion=None, vignetting=None),
    ],
    ids=["empty", "short", "cut off", "not a tiff", "no tables"],
)
def test_a_file_without_usable_tables_gives_nothing(tmp_path: Path, content: bytes) -> None:
    assert embedded_lens.profile_for(_write(tmp_path, content), "x") is None


def test_a_missing_file_gives_nothing(tmp_path: Path) -> None:
    assert embedded_lens.profile_for(tmp_path / "gone.RAF", "x") is None


@pytest.mark.parametrize(
    ("distortion", "vignetting", "has_distortion", "has_vignetting"),
    [
        ((0.0, *RADII, 0.0, 0.0, 0.0), (0.0, *RADII, *LIGHT), False, True),
        ((0.0, *RADII, 1.0, 2.0, 80.0), (0.0, *RADII, *LIGHT), False, True),
        ((0.0, 0.75, 0.5, 1.0, *STRETCH), (0.0, *RADII, *LIGHT), False, True),
        ((0.0, *RADII, *STRETCH), (0.0, *RADII, 100.0, 100.0, 100.0), True, False),
        ((0.0, *RADII, *STRETCH), (0.0, *RADII, 98.0, 95.0, 2.0), True, False),
        ((0.0, *RADII, *STRETCH), (0.0, *RADII, 98.0, 95.0), True, False),
    ],
    ids=["no distortion", "wild distortion", "radii out of order", "no falloff", "wild falloff", "uneven"],
)
def test_a_table_that_cannot_be_believed_is_left_out(
    tmp_path: Path,
    distortion: tuple[float, ...],
    vignetting: tuple[float, ...],
    has_distortion: bool,
    has_vignetting: bool,
) -> None:
    profile = embedded_lens.profile_for(_write(tmp_path, _raf(distortion, vignetting)), "x")

    assert profile is not None
    assert (profile.distortion is not None) == has_distortion
    assert (profile.vignetting is not None) == has_vignetting


# -- what the tables mean -----------------------------------------------------


def _camera_profile() -> lens.LensProfile:
    return lens.camera_profile("x", (RADII, STRETCH), (RADII, LIGHT), 1.0)


def test_the_distortion_table_says_how_far_out_each_radius_landed() -> None:
    profile = _camera_profile()
    radius = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5], dtype=np.float32)

    factor = lens._distortion_factor(profile, radius, 1.0)

    # None at the centre, linear in the square of the radius up to the first
    # entry, the entries themselves, and the last one held past the end.
    expected = 1.0 + np.array([0.0, 0.25, 1.0, 2.25, 4.0, 4.0]) / 100.0
    assert factor == pytest.approx(expected, abs=1e-6)


def test_half_the_distortion_amount_halves_the_stretch() -> None:
    radius = np.array([1.0], dtype=np.float32)
    assert lens._distortion_factor(_camera_profile(), radius, 0.5) == pytest.approx([1.02])


def test_the_vignetting_table_says_how_much_light_is_left() -> None:
    profile = _camera_profile()
    radius = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)

    gain = lens._vignetting_gain(profile, radius, 1.0)

    assert gain is not None
    assert gain == pytest.approx(100.0 / np.array([100.0, 98.0, 90.0, 90.0]), rel=1e-6)


def test_table_radii_are_fractions_of_the_frame() -> None:
    """With a unit of 1, the frame's corner is radius 1 whatever its size."""
    profile = lens.camera_profile("x", (RADII, STRETCH), None, 1.0)
    for size in ((6000, 4000), (1500, 1000), (4000, 6000)):
        plan = lens._plan(profile, LensSettings(enabled=True), size)
        corner = np.array([np.hypot(*size) / 2.0], dtype=np.float32)
        # Before the zoom that fills the frame, the corner comes from 4%
        # further out, which is what the table says at radius 1.
        unzoomed = replace(plan, scale=1.0)
        green = lens._sources(profile, unzoomed, corner, np.zeros(1, np.float32))[1]
        assert float(green[0][0]) == pytest.approx(corner[0] * 1.04, rel=1e-5)
        assert plan.scale > 1.0


# -- real photos --------------------------------------------------------------


def test_a_sample_raf_is_corrected_from_its_own_tables() -> None:
    path = SAMPLES / "DSCF3335.RAF"
    if not path.is_file():
        pytest.skip("DSCF3335.RAF is not in tests/samples")
    photo = decoder.decode_proxy(path)

    assert photo.lens is not None
    assert photo.lens.from_camera
    assert photo.lens.name.strip()
    assert photo.lens.distortion is not None and photo.lens.vignetting is not None
    assert photo.lens.tca is None


def test_a_sample_arw_still_uses_the_database() -> None:
    path = SAMPLES / "_DSC2828.ARW"
    if not path.is_file():
        pytest.skip("_DSC2828.ARW is not in tests/samples")
    photo = decoder.decode_proxy(path)

    assert photo.lens is not None
    assert not photo.lens.from_camera
