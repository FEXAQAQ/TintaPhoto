"""Tests for the photo information shown in the info tab (PLAN.md 3.9)."""

from __future__ import annotations

import struct
from datetime import datetime
from pathlib import Path

import piexif
import pytest
import rawpy
from PIL import Image

from core import metadata

SAMPLES = Path(__file__).resolve().parent / "samples"


def _sample_raws() -> list[Path]:
    return sorted({*SAMPLES.glob("*.ARW"), *SAMPLES.glob("*.arw")})


requires_raw = pytest.mark.skipif(not _sample_raws(), reason="no sample RAW file available")


def _sample_rafs() -> list[Path]:
    return sorted({*SAMPLES.glob("*.RAF"), *SAMPLES.glob("*.raf")})


requires_raf = pytest.mark.skipif(
    not _sample_rafs(), reason="no sample RAF file available"
)


class _PreviewOnly:
    """Stands in for LibRaw, which is only asked for the preview here."""

    def __init__(self, thumb: object) -> None:
        self._thumb = thumb

    def extract_thumb(self) -> object:
        if isinstance(self._thumb, Exception):
            raise self._thumb
        return self._thumb


def _write_jpeg(path: Path, **tags: object) -> Path:
    """Write a JPEG carrying a full set of shooting information."""
    main = {
        piexif.ImageIFD.Make: tags.get("make", "TINTA"),
        piexif.ImageIFD.Model: tags.get("model", "TP-1"),
    }
    exif = {
        piexif.ExifIFD.FNumber: (28, 10),
        piexif.ExifIFD.ExposureTime: (1, 250),
        piexif.ExifIFD.ISOSpeedRatings: 100,
        piexif.ExifIFD.FocalLength: (35, 1),
        piexif.ExifIFD.ExposureBiasValue: (-3, 10),
        piexif.ExifIFD.MeteringMode: 5,
        piexif.ExifIFD.Flash: 16,
        piexif.ExifIFD.WhiteBalance: 0,
        piexif.ExifIFD.DateTimeOriginal: "2026:10:12 14:03:22",
        piexif.ExifIFD.LensModel: "FE 35mm F1.8",
    }
    Image.new("RGB", (64, 48), (120, 120, 120)).save(
        path, "JPEG", exif=piexif.dump({"0th": main, "Exif": exif})
    )
    return path


# -- JPEG -----------------------------------------------------------------


def test_a_jpeg_reports_its_shooting_information(tmp_path: Path) -> None:
    info = metadata.read(_write_jpeg(tmp_path / "photo.jpg"))
    assert info.camera == "TINTA TP-1"
    assert info.lens == "FE 35mm F1.8"
    assert info.aperture == pytest.approx(2.8)
    assert info.shutter_seconds == pytest.approx(1 / 250)
    assert info.iso == 100
    assert info.focal_length_mm == pytest.approx(35.0)
    assert info.exposure_compensation == pytest.approx(-0.3)
    assert info.metering == "pattern"
    assert info.white_balance == "auto"
    assert info.flash_fired is False
    assert info.captured_at == datetime(2026, 10, 12, 14, 3, 22)


def test_text_padded_with_nul_characters_is_cut_at_the_first_one(tmp_path: Path) -> None:
    """A JPEG that fills its 63 character lens field with NULs."""
    padding = "\x00" * 39
    path = tmp_path / "padded.jpg"
    main = {
        piexif.ImageIFD.Make: "TINTA" + padding,
        piexif.ImageIFD.Model: "TP-5" + padding,
    }
    exif = {
        piexif.ExifIFD.LensModel: "15-45mm F3.5-5.6" + padding,
        piexif.ExifIFD.DateTimeOriginal: "2026:10:24 23:28:37" + padding,
    }
    Image.new("RGB", (8, 8)).save(path, "JPEG", exif=piexif.dump({"0th": main, "Exif": exif}))
    info = metadata.read(path)
    assert info.lens == "15-45mm F3.5-5.6"
    assert info.camera == "TINTA TP-5"
    assert info.captured_at == datetime(2026, 10, 24, 23, 28, 37)


@pytest.mark.parametrize("value", ["\x00" * 8, "  \x00 ", "", 42, None])
def test_text_that_is_only_padding_counts_as_missing(value: object) -> None:
    assert metadata._as_text(value) is None


def test_a_jpeg_reports_its_file_facts(tmp_path: Path) -> None:
    path = _write_jpeg(tmp_path / "photo.jpg")
    info = metadata.read(path)
    assert info.file_format == "JPG"
    assert info.file_size == path.stat().st_size
    assert (info.width, info.height) == (64, 48)
    assert info.megapixels == pytest.approx(64 * 48 / 1e6)


def test_a_png_reports_its_file_facts(tmp_path: Path) -> None:
    """A PNG goes down the same path as a JPEG, eXIf chunk and all."""
    path = tmp_path / "photo.png"
    Image.new("RGB", (64, 48), (120, 120, 120)).save(
        path, "PNG", exif=piexif.dump({"Exif": {piexif.ExifIFD.ISOSpeedRatings: 200}})
    )
    info = metadata.read(path)
    assert info.file_format == "PNG"
    assert info.file_size == path.stat().st_size
    assert (info.width, info.height) == (64, 48)
    assert info.iso == 200


def test_a_fired_flash_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "flash.jpg"
    Image.new("RGB", (8, 8)).save(
        path, "JPEG", exif=piexif.dump({"Exif": {piexif.ExifIFD.Flash: 1}})
    )
    assert metadata.read(path).flash_fired is True


def test_a_photo_without_tags_still_reports_its_file(tmp_path: Path) -> None:
    path = tmp_path / "bare.jpg"
    Image.new("RGB", (10, 20)).save(path, "JPEG")
    info = metadata.read(path)
    assert info.camera is None
    assert info.iso is None
    assert (info.width, info.height) == (10, 20)
    assert info.file_format == "JPG"


def test_an_unreadable_file_does_not_raise(tmp_path: Path) -> None:
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not a photo at all")
    info = metadata.read(path)
    assert info.file_format == "JPG"
    assert info.width is None
    assert info.megapixels is None


def test_a_missing_file_does_not_raise(tmp_path: Path) -> None:
    info = metadata.read(tmp_path / "nothing.ARW")
    assert info.file_size == 0
    assert info.camera is None


# -- the camera name ------------------------------------------------------


@pytest.mark.parametrize(
    ("make", "model", "expected"),
    [
        ("TINTA", "TP-1", "TINTA TP-1"),
        ("EXAMPLE CORPORATION", "EXAMPLE Z 6", "EXAMPLE Z 6"),
        ("Example", "Example Cam R6", "Example Cam R6"),
        ("", "Z-5", "Z-5"),
        ("TINTA", "", "TINTA"),
    ],
)
def test_the_maker_is_not_repeated_in_the_camera_name(
    make: str, model: str, expected: str
) -> None:
    assert metadata._camera_name(make, model) == expected


def test_a_camera_name_needs_at_least_one_of_the_two() -> None:
    assert metadata._camera_name(None, None) is None


# -- the TIFF reader ------------------------------------------------------


def test_the_tiff_reader_refuses_data_that_is_not_tiff() -> None:
    assert metadata._read_tiff_tags(b"this is not a TIFF header at all") == {}
    assert metadata._read_tiff_tags(b"") == {}


def test_the_tiff_reader_survives_a_truncated_header() -> None:
    assert metadata._read_tiff_tags(b"II\x2a\x00\x08\x00\x00\x00\x05\x00") == {}


def _tiff_with_model(order: str, prefix: bytes, model: bytes = b"TP-1\x00") -> bytes:
    """Build the smallest valid TIFF header holding one model tag.

    Layout: the 8 byte header, then one directory entry, then the end of the
    directory, and finally the text, which is too long to sit in the entry.
    """
    text_offset = 8 + 2 + 12 + 4
    header = prefix + struct.pack(order + "HI", 42, 8)
    entry = struct.pack(order + "HHII", metadata._TAG_MODEL, 2, len(model), text_offset)
    return header + struct.pack(order + "H", 1) + entry + struct.pack(order + "I", 0) + model


@pytest.mark.parametrize(("order", "prefix"), [("<", b"II"), (">", b"MM")])
def test_the_tiff_reader_handles_both_byte_orders(order: str, prefix: bytes) -> None:
    """A TIFF may be little or big endian; both must give the same answer."""
    tags = metadata._read_tiff_tags(_tiff_with_model(order, prefix))
    assert tags.get(metadata._TAG_MODEL) == "TP-1"


def test_the_tiff_reader_ignores_a_value_that_points_outside_the_file() -> None:
    """A damaged file must give up on the tag, not read past the end."""
    data = bytearray(_tiff_with_model("<", b"II"))
    data[8 + 2 + 8 : 8 + 2 + 12] = struct.pack("<I", 10_000_000)
    assert metadata._read_tiff_tags(bytes(data)) == {}


# -- RAW (needs a sample photo) -------------------------------------------


@requires_raw
def test_a_raw_file_reports_what_libraw_knows() -> None:
    info = metadata.read(_sample_raws()[0])
    assert info.aperture is not None and info.aperture > 0
    assert info.shutter_seconds is not None and info.shutter_seconds > 0
    assert info.iso is not None and info.iso > 0
    assert info.focal_length_mm is not None and info.focal_length_mm > 0
    assert info.lens
    assert info.captured_at is not None


@requires_raw
def test_a_raw_file_reports_the_camera_name_libraw_does_not_give() -> None:
    """The camera name is the first row of the info tab, so it must be there.

    What the name is depends on whose camera took the sample, so what is
    checked is that there is one: LibRaw does not give it, and it was
    missing from the info tab until the TIFF tags were read as well.
    """
    info = metadata.read(_sample_raws()[0])
    assert info.camera is not None
    assert len(info.camera.strip()) > 2


@requires_raw
def test_a_raw_file_reports_its_size_in_pixels() -> None:
    info = metadata.read(_sample_raws()[0])
    assert info.width is not None and info.width > 1000
    assert info.height is not None and info.height > 1000
    assert info.megapixels is not None and info.megapixels > 10


@requires_raw
def test_a_raw_file_reports_its_file_facts() -> None:
    path = _sample_raws()[0]
    info = metadata.read(path)
    assert info.file_format == "ARW"
    assert info.file_size == path.stat().st_size


@requires_raw
def test_a_raw_file_reports_the_camera_settings_libraw_omits() -> None:
    info = metadata.read(_sample_raws()[0])
    assert info.white_balance in {"auto", "manual", None}
    assert info.metering in set(metadata._METERING_MODES.values()) | {None}
    assert info.flash_fired in {True, False, None}


def test_a_portrait_jpeg_reports_the_size_it_is_shown_at(tmp_path: Path) -> None:
    """The decoder stands the photo up, so the size must say so too."""
    path = tmp_path / "portrait.jpg"
    Image.new("RGB", (64, 48)).save(
        path, "JPEG", exif=piexif.dump({"0th": {piexif.ImageIFD.Orientation: 8}})
    )
    info = metadata.read(path)
    assert (info.width, info.height) == (48, 64)


def test_a_landscape_jpeg_keeps_its_size(tmp_path: Path) -> None:
    path = tmp_path / "landscape.jpg"
    Image.new("RGB", (64, 48)).save(
        path, "JPEG", exif=piexif.dump({"0th": {piexif.ImageIFD.Orientation: 1}})
    )
    info = metadata.read(path)
    assert (info.width, info.height) == (64, 48)


# -- a RAW whose header is not a TIFF -------------------------------------


def test_the_preview_gives_the_tags_a_non_tiff_raw_keeps_nowhere_else(
    tmp_path: Path,
) -> None:
    """A RAF is not a TIFF, so its tags come from the preview's own EXIF."""
    jpeg = _write_jpeg(tmp_path / "preview.jpg", make="TINTA", model="TP-2")
    thumb = rawpy.Thumbnail(format=rawpy.ThumbFormat.JPEG, data=jpeg.read_bytes())

    tags = metadata._preview_tags(_PreviewOnly(thumb))

    assert tags[metadata._TAG_MAKE] == "TINTA"
    assert tags[metadata._TAG_MODEL] == "TP-2"
    # The sub-IFD has to be merged in as well, or the camera's own settings
    # would be missing even though the preview carries them.
    assert tags[metadata._TAG_WHITE_BALANCE] == 0
    assert tags[metadata._TAG_METERING_MODE] == 5


def test_a_preview_that_is_not_a_jpeg_is_left_alone(tmp_path: Path) -> None:
    thumb = rawpy.Thumbnail(format=rawpy.ThumbFormat.BITMAP, data=b"not a jpeg")
    assert metadata._preview_tags(_PreviewOnly(thumb)) == {}


def test_a_raw_file_without_a_preview_reports_no_tags() -> None:
    """A file LibRaw has no thumbnail for must not raise on the way out."""
    missing = rawpy.LibRawNoThumbnailError("no thumbnail")
    assert metadata._preview_tags(_PreviewOnly(missing)) == {}


def test_an_unreadable_preview_reports_no_tags() -> None:
    thumb = rawpy.Thumbnail(format=rawpy.ThumbFormat.JPEG, data=b"truncated")
    assert metadata._preview_tags(_PreviewOnly(thumb)) == {}


@requires_raw
def test_a_tiff_raw_does_not_go_looking_in_its_preview() -> None:
    """An ARW has its own tags, so the preview must not be touched at all."""
    path = _sample_raws()[0]
    with path.open("rb") as handle:
        tags = metadata._read_tiff_tags(handle.read(metadata._HEADER_BYTES))
    assert tags, "the sample ARW should have TIFF tags of its own"

    _, preview = metadata._raw_metadata(path, tags)
    assert preview == {}


@requires_raf
def test_a_raf_reports_the_camera_name_and_the_settings() -> None:
    """The fields that were empty for a RAF until the preview was read."""
    info = metadata.read(_sample_rafs()[0])
    assert info.camera is not None and len(info.camera.strip()) > 2
    assert info.white_balance in {"auto", "manual"}
    assert info.metering in set(metadata._METERING_MODES.values())
    assert info.flash_fired in {True, False}


@requires_raf
def test_a_raf_still_reports_what_libraw_gives() -> None:
    """Reading the preview must not cost the RAF what it already had."""
    info = metadata.read(_sample_rafs()[0])
    assert info.lens
    assert info.aperture is not None and info.aperture > 0
    assert info.shutter_seconds is not None and info.shutter_seconds > 0
    assert info.iso is not None and info.iso > 0
    assert info.focal_length_mm is not None and info.focal_length_mm > 0
    assert info.captured_at is not None
    assert info.width is not None and info.width > 1000
    assert info.height is not None and info.height > 1000
    assert info.file_format == "RAF"
