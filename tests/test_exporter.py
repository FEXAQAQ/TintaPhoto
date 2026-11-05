"""Tests for writing the exported file (PLAN.md section 2.6).

Most of these build their own small photo, so that they say something about
the format rather than about a camera. The ones that need a real file use
``tests/samples`` and skip when it is not there.

What matters most here is also the easiest to get wrong: the photo the user
shot must come out of an export exactly as it went in, so several tests
compare the original's checksum, size and modification time afterwards.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import piexif
import pytest
import tifffile
from PIL import Image

from core import exporter, metadata, pipeline, watermark
from core.edit_state import EditState
from core.exporter import ExportOptions
from core.geometry import Geometry, Rect
from core.metadata import PhotoMetadata
from core.watermark import WatermarkSpec

SAMPLES = Path(__file__).resolve().parent / "samples"

_EXIF_IFD = 34665

requires_raw = pytest.mark.skipif(
    not sorted(SAMPLES.glob("*.ARW")), reason="no sample RAW file available"
)


def _fingerprint(path: Path) -> tuple[str, int, int]:
    """Return what must not change about an original photo."""
    status = path.stat()
    return (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        status.st_size,
        status.st_mtime_ns,
    )


def _write_jpeg(
    path: Path,
    size: tuple[int, int] = (64, 48),
    with_gps: bool = False,
) -> Path:
    """Write a small JPEG with the details a camera would record."""
    pixels = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    pixels[:, :, 0] = np.linspace(0, 255, size[0], dtype=np.uint8)
    pixels[:, :, 1] = 120
    sections: dict[str, dict] = {
        "0th": {
            piexif.ImageIFD.Make: "TINTA",
            piexif.ImageIFD.Model: "TP-1",
        },
        "Exif": {
            piexif.ExifIFD.ColorSpace: 1,
            piexif.ExifIFD.DateTimeOriginal: "2026:10:20 09:30:00",
            piexif.ExifIFD.FNumber: (56, 10),
            piexif.ExifIFD.ExposureTime: (1, 250),
            piexif.ExifIFD.ISOSpeedRatings: 400,
            piexif.ExifIFD.FocalLength: (350, 10),
            piexif.ExifIFD.LensModel: "E 28-200mm F2.8-5.6",
        },
    }
    if with_gps:
        sections["GPS"] = {
            piexif.GPSIFD.GPSLatitudeRef: "N",
            piexif.GPSIFD.GPSLatitude: ((25, 1), (2, 1), (0, 1)),
        }
    Image.fromarray(pixels).save(path, "JPEG", quality=95, exif=piexif.dump(sections))
    return path


def _info(**values: object) -> PhotoMetadata:
    """Return photo details with only the fields a test cares about."""
    defaults: dict[str, object] = {
        "path": Path("photo.arw"),
        "file_size": 1,
        "file_format": "ARW",
        "camera": "TINTA TP-1",
        "lens": "E 28-200mm F2.8-5.6",
        "aperture": 5.6,
        "shutter_seconds": 1 / 250,
        "iso": 400,
        "focal_length_mm": 35.0,
        "captured_at": datetime(2026, 10, 20, 9, 30, 0),
    }
    defaults.update(values)
    return PhotoMetadata(**defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# What can be asked for
# --------------------------------------------------------------------------


def test_options_are_held_inside_their_ranges() -> None:
    options = ExportOptions(
        format="webp", quality=500, bits=12, long_edge=1
    ).sane()
    assert options.format == exporter.FORMAT_JPEG
    assert options.quality == exporter.QUALITY_MAX
    assert options.bits == pipeline.EIGHT_BIT
    assert options.long_edge == exporter.LONG_EDGE_MIN


def test_a_jpeg_is_always_eight_bits_whatever_the_depth_says() -> None:
    assert ExportOptions(format="jpg", bits=16).depth == pipeline.EIGHT_BIT
    assert ExportOptions(format="png", bits=16).depth == pipeline.SIXTEEN_BIT


def test_the_original_size_survives_being_made_sane() -> None:
    assert ExportOptions(long_edge=None).sane().long_edge is None


def test_the_offered_name_is_the_original_plus_edit() -> None:
    assert exporter.default_name(Path("D:/photos/_DSC1234.ARW")) == "_DSC1234_edit.jpg"
    assert (
        exporter.default_name(Path("a.jpg"), "_web", exporter.FORMAT_TIFF) == "a_web.tif"
    )


def test_an_existing_name_is_never_overwritten(tmp_path: Path) -> None:
    target = tmp_path / "photo_edit.jpg"
    assert exporter.unique_path(target) == target
    target.write_bytes(b"first")
    assert exporter.unique_path(target).name == "photo_edit_2.jpg"
    (tmp_path / "photo_edit_2.jpg").write_bytes(b"second")
    assert exporter.unique_path(target).name == "photo_edit_3.jpg"


# --------------------------------------------------------------------------
# Copying an untouched JPEG (X-08)
# --------------------------------------------------------------------------


def test_an_untouched_jpeg_asked_for_as_a_jpeg_is_copied(tmp_path: Path) -> None:
    source = _write_jpeg(tmp_path / "photo.jpg")
    assert exporter.can_copy_original(source, EditState(), None, ExportOptions())


@pytest.mark.parametrize(
    "state, geometry, options",
    [
        (EditState(exposure=0.5), None, ExportOptions()),
        (EditState(temperature=10.0), None, ExportOptions()),
        (EditState(), Geometry(crop=Rect(0.1, 0.1, 0.5, 0.5)), ExportOptions()),
        (EditState(), Geometry(quarter_turns=1), ExportOptions()),
        (EditState(), None, ExportOptions(format=exporter.FORMAT_PNG)),
        (EditState(), None, ExportOptions(long_edge=800)),
    ],
)
def test_anything_that_changes_the_pixels_stops_the_copy(
    tmp_path: Path, state: EditState, geometry: Geometry | None, options: ExportOptions
) -> None:
    source = _write_jpeg(tmp_path / "photo.jpg")
    assert not exporter.can_copy_original(source, state, geometry, options)


def test_a_raw_file_is_never_copied(tmp_path: Path) -> None:
    source = tmp_path / "photo.arw"
    source.write_bytes(b"not really a raw file")
    assert not exporter.can_copy_original(source, EditState(), None, ExportOptions())


def test_a_png_is_never_copied(tmp_path: Path) -> None:
    """Copying spares a JPEG a second round of compression; a PNG is written
    again without losing anything, so it takes the ordinary route."""
    source = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), (120, 120, 120)).save(source, "PNG")
    assert not exporter.can_copy_original(source, EditState(), None, ExportOptions())


def test_the_copy_of_an_untouched_jpeg_is_the_original_byte_for_byte(
    tmp_path: Path,
) -> None:
    source = _write_jpeg(tmp_path / "photo.jpg")
    before = _fingerprint(source)

    result = exporter.export(source, tmp_path / "out" / "photo_edit.jpg")

    assert result.copied
    assert result.path.read_bytes() == source.read_bytes()
    assert _fingerprint(source) == before


def test_removing_the_location_leaves_the_picture_alone(tmp_path: Path) -> None:
    source = _write_jpeg(tmp_path / "photo.jpg", with_gps=True)
    before = _fingerprint(source)

    result = exporter.export(
        source, tmp_path / "photo_edit.jpg", options=ExportOptions(remove_gps=True)
    )

    assert result.copied
    assert not piexif.load(str(result.path))["GPS"]
    assert piexif.load(str(result.path))["0th"][piexif.ImageIFD.Make] == b"TINTA"
    assert np.array_equal(np.asarray(Image.open(result.path)), np.asarray(Image.open(source)))
    assert _fingerprint(source) == before


def test_a_copy_of_a_photo_without_a_location_is_not_rewritten(tmp_path: Path) -> None:
    """Nothing to remove means the file is not touched at all (X-08)."""
    source = _write_jpeg(tmp_path / "photo.jpg")
    result = exporter.export(
        source, tmp_path / "photo_edit.jpg", options=ExportOptions(remove_gps=True)
    )
    assert result.path.read_bytes() == source.read_bytes()


def test_the_details_can_be_left_out_of_the_copy(tmp_path: Path) -> None:
    source = _write_jpeg(tmp_path / "photo.jpg")
    result = exporter.export(
        source,
        tmp_path / "photo_edit.jpg",
        options=ExportOptions(keep_metadata=False),
    )
    assert not piexif.load(str(result.path))["0th"]


# --------------------------------------------------------------------------
# Writing each format
# --------------------------------------------------------------------------


def test_an_untouched_png_exports_with_its_pixels_unchanged(tmp_path: Path) -> None:
    """An 8 bit PNG is a finished photo, so passthrough carries it through."""
    pixels = np.random.default_rng(3).integers(0, 256, (12, 16, 3), dtype=np.uint8)
    source = tmp_path / "photo.png"
    Image.fromarray(pixels).save(source, "PNG")
    before = _fingerprint(source)

    result = exporter.export(
        source, tmp_path / "out.png", options=ExportOptions(format=exporter.FORMAT_PNG)
    )

    assert not result.copied
    with Image.open(result.path) as written:
        assert np.array_equal(np.asarray(written.convert("RGB")), pixels)
    assert _fingerprint(source) == before


def test_a_sixteen_bit_png_exports_through_the_pipeline(tmp_path: Path) -> None:
    """It keeps no 8 bit copy to pass through, so the picture is developed."""
    value = 32768
    source = tmp_path / "deep.png"
    ok, buffer = cv2.imencode(".png", np.full((8, 8, 3), value, dtype=np.uint16))
    assert ok
    buffer.tofile(source)

    result = exporter.export(
        source,
        tmp_path / "out.png",
        options=ExportOptions(format=exporter.FORMAT_PNG, bits=pipeline.SIXTEEN_BIT),
    )

    written = cv2.imdecode(np.fromfile(result.path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert written.dtype == np.uint16
    assert abs(int(written[0, 0, 0]) - value) <= 64


def _exported(tmp_path: Path, options: ExportOptions, **kwargs: object) -> Path:
    """Export a small JPEG through the pipeline and return the file."""
    source = _write_jpeg(tmp_path / "photo.jpg")
    before = _fingerprint(source)
    result = exporter.export(
        source,
        tmp_path / "out" / "photo_edit",
        state=EditState(exposure=0.2),
        options=options,
        **kwargs,  # type: ignore[arg-type]
    )
    assert _fingerprint(source) == before, "the original must never be written to"
    assert not result.copied
    return result.path


def test_a_developed_jpeg_carries_the_srgb_profile_and_the_details(
    tmp_path: Path,
) -> None:
    path = _exported(tmp_path, ExportOptions(format=exporter.FORMAT_JPEG, quality=85))

    assert path.suffix == ".jpg"
    with Image.open(path) as opened:
        assert opened.size == (64, 48)
        assert opened.info["icc_profile"]
        details = opened.getexif()
        assert details[piexif.ImageIFD.Make] == "TINTA"
        assert details.get_ifd(_EXIF_IFD)[piexif.ExifIFD.ISOSpeedRatings] == 400


@pytest.mark.parametrize("bits", [8, 16])
def test_a_png_is_written_at_the_depth_asked_for(tmp_path: Path, bits: int) -> None:
    path = _exported(tmp_path, ExportOptions(format=exporter.FORMAT_PNG, bits=bits))

    pixels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert pixels.dtype == (np.uint8 if bits == 8 else np.uint16)
    assert pixels.shape == (48, 64, 3)
    with Image.open(path) as opened:
        assert len(opened.info["icc_profile"]) > 0
        assert opened.getexif().get_ifd(_EXIF_IFD)[piexif.ExifIFD.ISOSpeedRatings] == 400


@pytest.mark.parametrize("bits", [8, 16])
def test_a_tiff_is_written_at_the_depth_asked_for(tmp_path: Path, bits: int) -> None:
    path = _exported(tmp_path, ExportOptions(format=exporter.FORMAT_TIFF, bits=bits))

    assert path.suffix == ".tif"
    with tifffile.TiffFile(path) as opened:
        page = opened.pages[0]
        pixels = page.asarray()
        assert pixels.dtype == (np.uint8 if bits == 8 else np.uint16)
        assert pixels.shape == (48, 64, 3)
        assert len(page.tags[exporter._TAG_ICC_PROFILE].value) > 0
        assert page.tags[_EXIF_IFD].value["ISOSpeedRatings"] == 400
        assert page.tags["Orientation"].value == 1
        assert "Software" not in page.tags


def test_a_tiff_still_opens_as_a_picture_after_its_details_are_added(
    tmp_path: Path,
) -> None:
    """The directory is rebuilt at the end of the file, so the pixels must
    still be found through it."""
    source = _write_jpeg(tmp_path / "photo.jpg")
    path = exporter.export(
        source, tmp_path / "out", options=ExportOptions(format=exporter.FORMAT_TIFF)
    ).path

    with Image.open(path) as opened:
        assert opened.size == (64, 48)
    assert cv2.imread(str(path), cv2.IMREAD_UNCHANGED).shape == (48, 64, 3)


def test_the_file_is_scaled_to_the_long_edge_asked_for(tmp_path: Path) -> None:
    source = _write_jpeg(tmp_path / "photo.jpg", size=(400, 300))
    path = exporter.export(
        source, tmp_path / "out", options=ExportOptions(long_edge=200)
    ).path
    with Image.open(path) as opened:
        assert opened.size == (200, 150)


def test_a_photo_is_never_enlarged(tmp_path: Path) -> None:
    source = _write_jpeg(tmp_path / "photo.jpg", size=(64, 48))
    path = exporter.export(
        source,
        tmp_path / "out",
        state=EditState(exposure=0.2),
        options=ExportOptions(long_edge=4000),
    ).path
    with Image.open(path) as opened:
        assert opened.size == (64, 48)


def test_the_details_can_be_left_out_of_a_developed_file(tmp_path: Path) -> None:
    path = _exported(tmp_path, ExportOptions(keep_metadata=False))
    with Image.open(path) as opened:
        assert piexif.ImageIFD.Make not in opened.getexif()


def test_nothing_is_left_behind_when_writing_fails(tmp_path: Path) -> None:
    source = _write_jpeg(tmp_path / "photo.jpg")
    target = tmp_path / "out" / "photo_edit.jpg"
    target.parent.mkdir()

    def explode(*_args: object, **_kwargs: object) -> None:
        raise OSError("the disk is full")

    original = exporter._write_jpeg
    exporter._write_jpeg = explode  # type: ignore[assignment]
    try:
        with pytest.raises(exporter.ExportError):
            exporter.export(source, target, state=EditState(exposure=0.2))
    finally:
        exporter._write_jpeg = original  # type: ignore[assignment]

    assert list(target.parent.iterdir()) == []


def test_the_progress_is_reported_from_start_to_finish(tmp_path: Path) -> None:
    source = _write_jpeg(tmp_path / "photo.jpg")
    seen: list[float] = []
    exporter.export(
        source, tmp_path / "out", state=EditState(exposure=0.2), progress=seen.append
    )
    assert seen[0] > 0.0
    assert seen[-1] == 1.0
    assert seen == sorted(seen)
    assert len(seen) >= 4


def test_a_copy_reports_progress_too(tmp_path: Path) -> None:
    source = _write_jpeg(tmp_path / "photo.jpg")
    seen: list[float] = []
    exporter.export(source, tmp_path / "out", progress=seen.append)
    assert seen[-1] == 1.0


def test_a_file_that_cannot_be_read_becomes_an_export_error(tmp_path: Path) -> None:
    source = tmp_path / "broken.jpg"
    source.write_bytes(b"not a photo")
    with pytest.raises(exporter.ExportError):
        exporter.export(source, tmp_path / "out", state=EditState(exposure=0.2))


# --------------------------------------------------------------------------
# The details themselves (X-06)
# --------------------------------------------------------------------------


def test_the_six_details_plan_asks_for_are_all_written(tmp_path: Path) -> None:
    block = exporter.exif_block(_info(), (1200, 800))
    path = tmp_path / "with_details.jpg"
    Image.new("RGB", (4, 4)).save(path, "JPEG", exif=b"Exif\x00\x00" + block)

    with Image.open(path) as opened:
        main = opened.getexif()
        details = main.get_ifd(_EXIF_IFD)
    assert main[piexif.ImageIFD.Make] == "TINTA"
    assert main[piexif.ImageIFD.Model] == "TP-1"
    assert details[piexif.ExifIFD.DateTimeOriginal] == "2026:10:20 09:30:00"
    assert details[piexif.ExifIFD.LensModel] == "E 28-200mm F2.8-5.6"
    assert round(float(details[piexif.ExifIFD.FNumber]), 1) == 5.6
    assert round(1 / float(details[piexif.ExifIFD.ExposureTime])) == 250
    assert details[piexif.ExifIFD.ISOSpeedRatings] == 400


def test_the_exported_file_says_it_is_upright_and_srgb(tmp_path: Path) -> None:
    block = exporter.exif_block(_info(), (1200, 800))
    path = tmp_path / "upright.jpg"
    Image.new("RGB", (4, 4)).save(path, "JPEG", exif=b"Exif\x00\x00" + block)

    with Image.open(path) as opened:
        main = opened.getexif()
        details = main.get_ifd(_EXIF_IFD)
    assert main[piexif.ImageIFD.Orientation] == 1
    assert details[piexif.ExifIFD.ColorSpace] == 1
    assert details[piexif.ExifIFD.PixelXDimension] == 1200
    assert details[piexif.ExifIFD.PixelYDimension] == 800


def test_a_photo_with_no_details_still_gets_a_readable_block() -> None:
    block = exporter.exif_block(None, (10, 20))
    assert block.startswith(b"II*\x00")


def test_a_long_exposure_is_written_as_a_fraction_of_tenths() -> None:
    assert struct.unpack("<II", exporter._rational(20.0)) == (200, 10)
    assert struct.unpack("<II", exporter._rational(1 / 250)) == (1, 250)
    assert struct.unpack("<II", exporter._rational(0.0)) == (0, 10)


def test_details_that_are_missing_are_simply_not_written(tmp_path: Path) -> None:
    sparse = _info(lens=None, iso=None, aperture=None, captured_at=None, camera=None)
    block = exporter.exif_block(sparse, (8, 8))
    path = tmp_path / "sparse.jpg"
    Image.new("RGB", (4, 4)).save(path, "JPEG", exif=b"Exif\x00\x00" + block)

    with Image.open(path) as opened:
        main = opened.getexif()
        assert piexif.ImageIFD.Make not in main
        assert piexif.ExifIFD.LensModel not in main.get_ifd(_EXIF_IFD)


# --------------------------------------------------------------------------
# The chunks of a 16 bit PNG (PLAN.md 6 step 12)
# --------------------------------------------------------------------------


def _png_chunks(data: bytes) -> list[str]:
    """Return the names of the chunks of *data*, in order."""
    names: list[str] = []
    at = 8
    while at < len(data):
        length = struct.unpack_from(">I", data, at)[0]
        name = data[at + 4 : at + 8]
        payload = data[at + 8 : at + 8 + length]
        expected = struct.unpack_from(">I", data, at + 8 + length)[0]
        assert zlib.crc32(name + payload) & 0xFFFFFFFF == expected, name
        names.append(name.decode("ascii"))
        at += 12 + length
    return names


def test_the_profile_and_the_details_go_in_before_the_pixels() -> None:
    written, buffer = cv2.imencode(".png", np.zeros((4, 4, 3), dtype=np.uint16))
    assert written
    data = exporter.with_png_chunks(bytes(buffer), b"a profile", b"II*\x00")

    names = _png_chunks(data)
    assert names[0] == "IHDR"
    assert names.index("iCCP") < names.index("IDAT")
    assert names.index("eXIf") < names.index("IDAT")


def test_the_profile_is_stored_compressed_under_its_usual_name() -> None:
    written, buffer = cv2.imencode(".png", np.zeros((4, 4, 3), dtype=np.uint16))
    assert written
    profile = b"a profile" * 20
    data = exporter.with_png_chunks(bytes(buffer), profile, None)

    at = data.index(b"iCCP")
    length = struct.unpack_from(">I", data, at - 4)[0]
    payload = data[at + 4 : at + 4 + length]
    name, rest = payload.split(b"\x00", 1)
    assert name == b"ICC Profile"
    assert zlib.decompress(rest[1:]) == profile


def test_something_that_is_not_a_png_is_refused() -> None:
    with pytest.raises(exporter.ExportError):
        exporter.with_png_chunks(b"not a png", b"", None)


# --------------------------------------------------------------------------
# The size shown in the dialog (PLAN.md 3.11)
# --------------------------------------------------------------------------


def test_the_size_of_a_tiff_is_counted_rather_than_guessed() -> None:
    options = ExportOptions(format=exporter.FORMAT_TIFF, bits=16)
    estimate = exporter.estimate_size(np.zeros((8, 8, 3), np.uint8), 1_000_000, options)
    assert estimate == exporter.TIFF_OVERHEAD_BYTES + 1_000_000 * 3 * 2


def test_the_estimate_grows_with_the_number_of_pixels() -> None:
    preview = (np.random.default_rng(0).random((120, 160, 3)) * 255).astype(np.uint8)
    pixels = 120 * 160
    small = exporter.estimate_size(preview, pixels, ExportOptions())
    large = exporter.estimate_size(preview, pixels * 4, ExportOptions())
    assert small > 0
    assert large == pytest.approx(small * 4, rel=0.01)


def test_a_lower_quality_is_estimated_smaller() -> None:
    preview = (np.random.default_rng(1).random((120, 160, 3)) * 255).astype(np.uint8)
    best = exporter.estimate_size(preview, 10_000, ExportOptions(quality=95))
    worst = exporter.estimate_size(preview, 10_000, ExportOptions(quality=20))
    assert worst < best


def test_a_sixteen_bit_png_is_estimated_larger_than_an_eight_bit_one() -> None:
    preview = (np.random.default_rng(2).random((60, 80, 3)) * 255).astype(np.uint8)
    eight = exporter.estimate_size(
        preview, 10_000, ExportOptions(format=exporter.FORMAT_PNG, bits=8)
    )
    sixteen = exporter.estimate_size(
        preview, 10_000, ExportOptions(format=exporter.FORMAT_PNG, bits=16)
    )
    assert sixteen > eight


def test_an_empty_preview_is_estimated_as_nothing() -> None:
    assert exporter.estimate_size(np.zeros((0, 0, 3), np.uint8), 10, ExportOptions()) == 0


def test_the_output_size_follows_the_crop_and_the_long_edge() -> None:
    whole = exporter.output_size((6000, 4000), None, ExportOptions())
    assert whole == (6000, 4000)

    cropped = exporter.output_size(
        (6000, 4000), Geometry(crop=Rect(0.0, 0.0, 0.5, 0.5)), ExportOptions()
    )
    assert cropped == (3000, 2000)

    scaled = exporter.output_size((6000, 4000), None, ExportOptions(long_edge=1200))
    assert scaled == (1200, 800)


def test_turning_the_photo_turns_the_output_size() -> None:
    turned = exporter.output_size(
        (6000, 4000), Geometry(quarter_turns=1), ExportOptions()
    )
    assert turned == (4000, 6000)


# --------------------------------------------------------------------------
# A real photo (PLAN.md section 9, stage 6)
# --------------------------------------------------------------------------


@requires_raw
def test_a_real_raw_file_exports_with_its_details_and_is_left_alone(
    tmp_path: Path,
) -> None:
    source = sorted(SAMPLES.glob("*.ARW"))[0]
    before = _fingerprint(source)
    info = metadata.read(source)

    result = exporter.export(
        source,
        tmp_path / exporter.default_name(source),
        options=ExportOptions(long_edge=800),
    )

    assert _fingerprint(source) == before
    with Image.open(result.path) as opened:
        assert max(opened.size) == 800
        details = opened.getexif().get_ifd(_EXIF_IFD)
    assert details[piexif.ExifIFD.ISOSpeedRatings] == info.iso
    assert opened.info is not None


# -- a photo that was already decoded (X-09) ------------------------------


def test_a_decoded_photo_exports_like_the_file_it_came_from(tmp_path: Path) -> None:
    from core import decoder

    source = _write_jpeg(tmp_path / "photo.jpg")
    state = EditState(exposure=0.2)
    # Sixteen bits, which are not dithered, so the two can be compared exactly.
    options = ExportOptions(format=exporter.FORMAT_PNG, bits=16)

    whole = exporter.export(source, tmp_path / "a" / "photo", state=state, options=options)
    decoded = exporter.export_decoded(
        decoder.decode(source), source, tmp_path / "b" / "photo", state, None, options
    )

    with Image.open(whole.path) as first, Image.open(decoded.path) as second:
        assert np.array_equal(np.asarray(first), np.asarray(second))


def test_an_untouched_jpeg_is_still_copied_when_it_was_decoded(tmp_path: Path) -> None:
    from core import decoder

    source = _write_jpeg(tmp_path / "photo.jpg")

    result = exporter.export_decoded(
        decoder.decode(source), source, tmp_path / "out" / "photo_edit.jpg", EditState()
    )

    assert result.copied
    assert result.path.read_bytes() == source.read_bytes()


def test_a_photo_to_develop_without_its_pixels_is_an_export_error(tmp_path: Path) -> None:
    source = _write_jpeg(tmp_path / "photo.jpg")

    with pytest.raises(exporter.ExportError):
        exporter.export_decoded(
            None, source, tmp_path / "out" / "photo_edit.jpg", EditState(exposure=0.5)
        )

# --------------------------------------------------------------------------
# The watermark (X-10, PLAN.md 2.9)
# --------------------------------------------------------------------------

_MARK = WatermarkSpec(text="© 2026", size=10.0, opacity=100)


def _outside(frame: tuple[int, int], spec: WatermarkSpec = _MARK) -> np.ndarray:
    """Return a mask of the pixels the watermark does not cover on *frame*."""
    sprite = watermark.rasterise(spec, frame, watermark.load_fonts())
    assert sprite is not None
    x, y = (round(value) for value in watermark.layout(spec, frame, sprite.size))
    mask = np.ones((frame[1], frame[0]), dtype=bool)
    mask[y : y + sprite.size[1], x : x + sprite.size[0]] = False
    return mask


def test_a_watermark_stops_the_copy(tmp_path: Path) -> None:
    source = _write_jpeg(tmp_path / "photo.jpg")
    assert not exporter.can_copy_original(
        source, EditState(), None, ExportOptions(watermark=_MARK).sane()
    )


def test_a_watermark_with_no_text_is_no_watermark(tmp_path: Path) -> None:
    source = _write_jpeg(tmp_path / "photo.jpg")
    options = ExportOptions(watermark=WatermarkSpec(text="  ")).sane()
    assert options.watermark is None
    assert exporter.can_copy_original(source, EditState(), None, options)


def test_an_untouched_jpeg_with_a_watermark_is_written_again(tmp_path: Path) -> None:
    source = _write_jpeg(tmp_path / "photo.jpg", size=(400, 300))
    before = _fingerprint(source)
    result = exporter.export(
        source, tmp_path / "photo_edit.jpg", options=ExportOptions(watermark=_MARK)
    )
    assert not result.copied
    assert result.path.read_bytes() != source.read_bytes()
    assert _fingerprint(source) == before


def test_the_watermark_touches_only_the_pixels_under_it(tmp_path: Path) -> None:
    """An untouched 8 bit PNG passes through, so every other pixel is the
    original's own, dither or not."""
    pixels = np.random.default_rng(5).integers(0, 256, (300, 400, 3), dtype=np.uint8)
    source = tmp_path / "photo.png"
    Image.fromarray(pixels).save(source, "PNG")

    result = exporter.export(
        source,
        tmp_path / "out.png",
        options=ExportOptions(format=exporter.FORMAT_PNG, watermark=_MARK),
    )

    with Image.open(result.path) as opened:
        written = np.asarray(opened.convert("RGB"))
    outside = _outside((400, 300))
    np.testing.assert_array_equal(written[outside], pixels[outside])
    assert not np.array_equal(written[~outside], pixels[~outside])


def test_a_sixteen_bit_file_gets_the_watermark_at_sixteen_bits(tmp_path: Path) -> None:
    source = tmp_path / "flat.png"
    Image.new("RGB", (400, 300), (0, 0, 0)).save(source, "PNG")
    result = exporter.export(
        source,
        tmp_path / "out.png",
        options=ExportOptions(
            format=exporter.FORMAT_PNG, bits=pipeline.SIXTEEN_BIT, watermark=_MARK
        ),
    )
    written = cv2.imdecode(np.fromfile(result.path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert written.dtype == np.uint16
    assert int(written.max()) == 65535
    assert int(written[_outside((400, 300))].max()) == 0


def test_the_watermark_is_drawn_at_the_size_written(tmp_path: Path) -> None:
    source = tmp_path / "flat.png"
    Image.new("RGB", (800, 600), (0, 0, 0)).save(source, "PNG")
    result = exporter.export(
        source,
        tmp_path / "out.png",
        options=ExportOptions(format=exporter.FORMAT_PNG, long_edge=400, watermark=_MARK),
    )
    with Image.open(result.path) as opened:
        written = np.asarray(opened.convert("L"))
    assert written.shape == (300, 400)
    rows = np.nonzero(written.max(axis=1) > 0)[0]
    # A line 10 % of the short edge of 300: the ink is well under 30 px tall.
    assert 10 <= rows[-1] - rows[0] + 1 <= 30


def test_finished_pixels_such_as_a_collage_get_the_watermark(tmp_path: Path) -> None:
    canvas = np.zeros((300, 400, 3), dtype=np.uint8)
    result = exporter.export_pixels(
        canvas,
        tmp_path / "collage.png",
        ExportOptions(format=exporter.FORMAT_PNG, watermark=_MARK),
    )
    with Image.open(result.path) as opened:
        written = np.asarray(opened.convert("RGB"))
    assert int(written.max()) == 255
    assert int(written[_outside((400, 300))].max()) == 0
