"""Tests for reading photos (PLAN.md section 2.1 and section 6 step 1).

The JPEG and PNG tests build their own files, so they need no sample photos.
The RAW tests use the files in ``tests/samples`` and skip when they are not
there, because large RAW files are not kept in the repository.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import piexif
import pytest
from PIL import Image

from core import color_management as cm
from core import decoder
from core.decoder import ColorSpace, DecodeError

SAMPLES = Path(__file__).resolve().parent / "samples"


def _sample_raws() -> list[Path]:
    return sorted({*SAMPLES.glob("*.ARW"), *SAMPLES.glob("*.arw")})


requires_raw = pytest.mark.skipif(not _sample_raws(), reason="no sample RAW file available")


def _portrait_raws() -> list[Path]:
    """Return the sample photos the camera recorded as rotated."""
    import rawpy

    portrait = []
    for path in _sample_raws():
        try:
            with rawpy.imread(str(path)) as raw:
                if decoder.libraw_flip_to_orientation(int(raw.sizes.flip)) in (5, 6, 7, 8):
                    portrait.append(path)
        except Exception:  # noqa: BLE001 - a sample that cannot be read is skipped
            continue
    return portrait


requires_portrait_raw = pytest.mark.skipif(
    not _portrait_raws(), reason="no portrait sample RAW file available"
)


def _sample_jpegs() -> list[Path]:
    return sorted({*SAMPLES.glob("*.JPG"), *SAMPLES.glob("*.jpg")})


requires_jpeg = pytest.mark.skipif(
    not _sample_jpegs(), reason="no sample JPEG available"
)


# -- file kinds -----------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "supported", "raw"),
    [
        ("DSC01234.ARW", True, True),
        ("dsc01234.arw", True, True),
        ("photo.CR3", True, True),
        ("photo.dng", True, True),
        ("photo.JPG", True, False),
        ("photo.jpeg", True, False),
        ("photo.png", True, False),
        ("photo.PNG", True, False),
        ("photo.bmp", False, False),
        ("photo.txt", False, False),
    ],
)
def test_supported_file_kinds(name: str, supported: bool, raw: bool) -> None:
    path = Path(name)
    assert decoder.is_supported(path) is supported
    assert decoder.is_raw(path) is raw


def test_only_a_png_counts_as_a_png() -> None:
    assert decoder.is_png(Path("photo.PNG"))
    assert not decoder.is_png(Path("photo.jpg"))
    assert not decoder.is_png(Path("photo.ARW"))


# -- orientation ----------------------------------------------------------


def _corner_image() -> np.ndarray:
    """A 2 x 3 image whose every pixel is different, so turns are visible."""
    return np.arange(6, dtype=np.float32).reshape(2, 3, 1).repeat(3, axis=2)


def test_orientation_one_changes_nothing() -> None:
    image = _corner_image()
    assert np.array_equal(decoder.apply_orientation(image, 1), image)


def test_orientation_three_turns_the_image_upside_down() -> None:
    image = _corner_image()
    turned = decoder.apply_orientation(image, 3)
    assert np.array_equal(turned, image[::-1, ::-1])


@pytest.mark.parametrize("orientation", [5, 6, 7, 8])
def test_quarter_turns_swap_width_and_height(orientation: int) -> None:
    image = _corner_image()
    turned = decoder.apply_orientation(image, orientation)
    assert turned.shape[:2] == image.shape[1::-1]


def test_orientation_six_puts_the_top_left_pixel_on_the_top_right() -> None:
    """Orientation 6 means the camera was rotated 90 degrees clockwise."""
    image = _corner_image()
    turned = decoder.apply_orientation(image, 6)
    assert turned[0, -1, 0] == image[0, 0, 0]
    assert turned[0, 0, 0] == image[-1, 0, 0]


def test_orientation_eight_puts_the_top_left_pixel_on_the_bottom_left() -> None:
    image = _corner_image()
    turned = decoder.apply_orientation(image, 8)
    assert turned[-1, 0, 0] == image[0, 0, 0]
    assert turned[0, 0, 0] == image[0, -1, 0]


@pytest.mark.parametrize("orientation", [1, 2, 3, 4, 5, 6, 7, 8])
def test_every_orientation_keeps_all_pixels(orientation: int) -> None:
    image = _corner_image()
    turned = decoder.apply_orientation(image, orientation)
    assert sorted(turned[..., 0].flatten()) == sorted(image[..., 0].flatten())


def test_an_unknown_orientation_is_left_alone() -> None:
    image = _corner_image()
    assert np.array_equal(decoder.apply_orientation(image, 0), image)


@pytest.mark.parametrize(
    ("flip", "orientation"), [(0, 1), (3, 3), (5, 8), (6, 6), (99, 1)]
)
def test_libraw_flip_maps_to_an_exif_orientation(flip: int, orientation: int) -> None:
    assert decoder.libraw_flip_to_orientation(flip) == orientation


# -- the camera's own crop ------------------------------------------------


class _Sizes:
    """The parts of LibRaw's ``raw.sizes`` the crop reads."""

    def __init__(self, **values: int) -> None:
        defaults = {
            "width": 6024, "height": 4024, "left_margin": 0, "top_margin": 0,
            "crop_left_margin": 12, "crop_top_margin": 12,
            "crop_width": 6000, "crop_height": 4000,
        }
        defaults.update(values)
        self.__dict__.update(defaults)


def test_the_camera_frame_is_cut_out_of_the_sensor() -> None:
    """A 24 megapixel ARW opens at 6000 by 4000, as the camera made it."""
    crop = decoder._default_crop(_Sizes(), 1)
    assert crop is not None
    assert (crop.rows, crop.columns) == (slice(12, 4012), slice(12, 6012))
    assert decoder.sensor_frame_size(_Sizes()) == (6000, 4000)


def test_at_half_size_the_crop_is_halved_but_remembers_the_full_frame() -> None:
    crop = decoder._default_crop(_Sizes(), 2)
    assert crop is not None
    assert (crop.rows, crop.columns) == (slice(6, 2006), slice(6, 3006))
    assert (crop.full_width, crop.full_height) == (6000, 4000)


def test_the_masked_margin_libraw_already_dropped_is_taken_off() -> None:
    sizes = _Sizes(width=6264, height=4176, top_margin=6, crop_left_margin=6,
                   crop_top_margin=18, crop_width=6240, crop_height=4155)
    crop = decoder._default_crop(sizes, 1)
    assert crop is not None
    assert (crop.rows, crop.columns) == (slice(12, 4167), slice(6, 6246))


@pytest.mark.parametrize(
    "values",
    [
        {"crop_width": 0, "crop_height": 0},  # the file names no frame
        {"crop_width": 6024, "crop_height": 4024, "crop_left_margin": 0,
         "crop_top_margin": 0},  # the frame is the whole of it
        {"crop_left_margin": 40},  # the frame would run off the edge
        {"top_margin": 20},  # the frame would start above the image
    ],
)
def test_no_crop_is_taken_when_the_file_gives_none_that_fits(values: dict[str, int]) -> None:
    sizes = _Sizes(**values)
    assert decoder._default_crop(sizes, 1) is None
    assert decoder.sensor_frame_size(sizes) == (sizes.width, sizes.height)


@requires_raw
def test_a_raw_file_opens_at_the_size_its_metadata_reports() -> None:
    """The info panel, the export size and the pixels must all agree."""
    import rawpy

    path = _sample_raws()[0]
    photo = decoder.decode(path)
    with rawpy.imread(str(path)) as raw:
        width, height = decoder.sensor_frame_size(raw.sizes)
        sideways = decoder.libraw_flip_to_orientation(int(raw.sizes.flip)) in (5, 6, 7, 8)
    expected = (width, height) if sideways else (height, width)
    assert photo.image.shape[:2] == expected
    assert photo.clipped.shape == photo.image.shape[:2]
    proxy = decoder.decode_proxy(path)
    assert proxy.shrunk_from == (photo.image.shape[1], photo.image.shape[0])


# -- proxy ----------------------------------------------------------------


def test_proxy_scales_the_long_edge_down() -> None:
    image = np.zeros((1000, 2000, 3), dtype=np.float32)
    proxy = decoder.make_proxy(image, long_edge=500)
    assert proxy.shape[:2] == (250, 500)
    assert proxy.dtype == np.float32


def test_proxy_scales_a_portrait_photo_by_its_height() -> None:
    image = np.zeros((2000, 1000, 3), dtype=np.float32)
    assert decoder.make_proxy(image, long_edge=500).shape[:2] == (500, 250)


def test_a_photo_smaller_than_the_proxy_is_left_alone() -> None:
    image = np.zeros((100, 200, 3), dtype=np.float32)
    assert decoder.make_proxy(image, long_edge=500) is image


def test_the_proxy_keeps_the_average_level() -> None:
    image = np.full((600, 800, 3), 0.42, dtype=np.float32)
    proxy = decoder.make_proxy(image, long_edge=200)
    assert proxy.mean() == pytest.approx(0.42, abs=1e-5)


# -- JPEG -----------------------------------------------------------------


def _write_jpeg(
    path: Path,
    colour: tuple[int, int, int] = (128, 128, 128),
    size: tuple[int, int] = (64, 48),
    orientation: int | None = None,
    adobe_rgb: bool = False,
) -> Path:
    """Write a small JPEG, optionally tagged with orientation or Adobe RGB."""
    image = Image.new("RGB", size, colour)
    # Only the sections that carry a value are written: piexif turns an empty
    # section into a pointer to an empty IFD, which Pillow then warns about.
    main: dict[int, object] = {}
    exif_ifd: dict[int, object] = {}
    interop: dict[int, object] = {}
    if orientation is not None:
        main[piexif.ImageIFD.Orientation] = orientation
    if adobe_rgb:
        exif_ifd[piexif.ExifIFD.ColorSpace] = 65535
        interop[piexif.InteropIFD.InteroperabilityIndex] = "R03"
    else:
        exif_ifd[piexif.ExifIFD.ColorSpace] = 1
    sections = {"0th": main, "Exif": exif_ifd}
    if interop:
        sections["Interop"] = interop
    image.save(path, "JPEG", quality=100, exif=piexif.dump(sections))
    return path


def test_a_plain_jpeg_is_read_as_srgb(tmp_path: Path) -> None:
    photo = decoder.decode(_write_jpeg(tmp_path / "plain.jpg"))
    assert not photo.is_raw
    assert photo.color_space is ColorSpace.SRGB
    assert photo.image.dtype == np.float32
    assert photo.size == (64, 48)


def test_a_jpeg_tagged_adobe_rgb_is_read_as_adobe_rgb(tmp_path: Path) -> None:
    photo = decoder.decode(_write_jpeg(tmp_path / "adobe.jpg", adobe_rgb=True))
    assert photo.color_space is ColorSpace.ADOBE_RGB


def test_a_grey_jpeg_stays_neutral_in_the_working_space(tmp_path: Path) -> None:
    photo = decoder.decode(_write_jpeg(tmp_path / "grey.jpg", colour=(128, 128, 128)))
    pixel = photo.image[0, 0]
    assert pixel.max() - pixel.min() < 1e-3, pixel


def test_a_grey_adobe_rgb_jpeg_also_stays_neutral(tmp_path: Path) -> None:
    photo = decoder.decode(
        _write_jpeg(tmp_path / "grey_adobe.jpg", colour=(128, 128, 128), adobe_rgb=True)
    )
    pixel = photo.image[0, 0]
    assert pixel.max() - pixel.min() < 1e-3, pixel


def test_the_two_colour_spaces_decode_a_mid_grey_differently(tmp_path: Path) -> None:
    """Adobe RGB has a different tone curve, so the same byte is not the same light."""
    srgb = decoder.decode(_write_jpeg(tmp_path / "s.jpg", colour=(128, 128, 128)))
    adobe = decoder.decode(
        _write_jpeg(tmp_path / "a.jpg", colour=(128, 128, 128), adobe_rgb=True)
    )
    assert srgb.image[0, 0, 0] != pytest.approx(adobe.image[0, 0, 0], abs=1e-4)


def test_a_jpeg_decodes_to_the_expected_linear_value(tmp_path: Path) -> None:
    """Mid grey 128 is about 0.2159 in linear light."""
    photo = decoder.decode(_write_jpeg(tmp_path / "grey.jpg", colour=(128, 128, 128)))
    back = cm.apply_matrix(photo.image, cm.PROPHOTO_TO_SRGB)[0, 0]
    assert back[0] == pytest.approx(cm.srgb_to_linear(np.float32([128 / 255]))[0], abs=2e-3)


def test_a_jpeg_is_turned_upright(tmp_path: Path) -> None:
    """A photo tagged as rotated must come back with width and height swapped."""
    path = _write_jpeg(tmp_path / "portrait.jpg", size=(64, 48), orientation=6)
    assert decoder.decode(path).size == (48, 64)


def test_an_upright_jpeg_keeps_its_size(tmp_path: Path) -> None:
    path = _write_jpeg(tmp_path / "landscape.jpg", size=(64, 48), orientation=1)
    assert decoder.decode(path).size == (64, 48)


def test_shrinking_a_mask_keeps_the_area_it_covers() -> None:
    """A block of clipped pixels must survive being scaled down.

    OpenCV drops a trailing axis of one, which once turned the shrunk mask
    into a column of the wrong shape and quietly lost every clipped pixel.
    """
    mask = np.zeros((400, 600), dtype=bool)
    mask[100:300, 200:500] = True  # a quarter of the frame
    shrunk = decoder._shrink_mask(mask, long_edge=120)
    assert shrunk.shape == (80, 120)
    assert shrunk.mean() == pytest.approx(mask.mean(), abs=0.02)


def test_shrinking_a_mask_that_is_already_small_changes_nothing() -> None:
    mask = np.zeros((40, 60), dtype=bool)
    mask[10:20, 10:20] = True
    assert np.array_equal(decoder._shrink_mask(mask, long_edge=120), mask)


def test_an_empty_mask_stays_empty_when_shrunk() -> None:
    assert not decoder._shrink_mask(np.zeros((400, 600), dtype=bool), 120).any()


def test_a_white_jpeg_is_marked_as_clipped(tmp_path: Path) -> None:
    photo = decoder.decode(_write_jpeg(tmp_path / "white.jpg", colour=(255, 255, 255)))
    assert photo.clipped.all()


def test_a_mid_grey_jpeg_is_not_marked_as_clipped(tmp_path: Path) -> None:
    photo = decoder.decode(_write_jpeg(tmp_path / "grey.jpg", colour=(128, 128, 128)))
    assert not photo.clipped.any()


# -- PNG ------------------------------------------------------------------


def _write_png(
    path: Path,
    colour: tuple[int, int, int] = (128, 128, 128),
    size: tuple[int, int] = (64, 48),
    orientation: int | None = None,
    adobe_rgb: bool = False,
    profile: bytes | None = None,
) -> Path:
    """Write a small 8 bit PNG, optionally tagged or with a profile.

    A PNG carries the same Exif block as a JPEG, in its eXIf chunk, so the
    tags are built the same way: only the sections that carry a value.
    """
    image = Image.new("RGB", size, colour)
    main: dict[int, object] = {}
    exif_ifd: dict[int, object] = {}
    interop: dict[int, object] = {}
    if orientation is not None:
        main[piexif.ImageIFD.Orientation] = orientation
    if adobe_rgb:
        exif_ifd[piexif.ExifIFD.ColorSpace] = 65535
        interop[piexif.InteropIFD.InteroperabilityIndex] = "R03"
    sections = {name: part for name, part in (("0th", main), ("Exif", exif_ifd)) if part}
    if interop:
        sections["Interop"] = interop
    extra: dict[str, object] = {}
    if sections:
        extra["exif"] = piexif.dump(sections)
    if profile is not None:
        extra["icc_profile"] = profile
    image.save(path, "PNG", **extra)
    return path


def _write_png16(path: Path, value: int, size: tuple[int, int] = (8, 8)) -> Path:
    """Write a 16 bit grey PNG at *value*, which Pillow cannot write itself."""
    pixels = np.full((size[1], size[0], 3), value, dtype=np.uint16)
    ok, buffer = cv2.imencode(".png", pixels)
    assert ok
    buffer.tofile(path)
    return path


def _write_rgba_png(path: Path, colour: tuple[int, int, int], alpha: int) -> Path:
    """Write an 8 x 8 PNG of one colour at one alpha."""
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    pixels[..., :3] = colour
    pixels[..., 3] = alpha
    Image.fromarray(pixels, "RGBA").save(path)
    return path


def test_a_plain_png_is_read_as_srgb(tmp_path: Path) -> None:
    photo = decoder.decode(_write_png(tmp_path / "plain.png"))
    assert not photo.is_raw
    assert photo.color_space is ColorSpace.SRGB
    assert photo.image.dtype == np.float32
    assert photo.size == (64, 48)


def test_a_grey_png_stays_neutral_in_the_working_space(tmp_path: Path) -> None:
    pixel = decoder.decode(_write_png(tmp_path / "grey.png")).image[0, 0]
    assert pixel.max() - pixel.min() < 1e-3, pixel


def test_a_png_decodes_to_the_expected_linear_value(tmp_path: Path) -> None:
    """Mid grey 128 is about 0.2159 in linear light, as it is in a JPEG."""
    photo = decoder.decode(_write_png(tmp_path / "grey.png"))
    back = cm.apply_matrix(photo.image, cm.PROPHOTO_TO_SRGB)[0, 0]
    assert back[0] == pytest.approx(cm.srgb_to_linear(np.float32([128 / 255]))[0], abs=2e-3)


def test_a_png_and_a_jpeg_of_the_same_grey_agree(tmp_path: Path) -> None:
    """The two formats are the same photo to the pipeline, so they must agree."""
    png = decoder.decode(_write_png(tmp_path / "grey.png"))
    jpeg = decoder.decode(_write_jpeg(tmp_path / "grey.jpg"))
    assert png.image[0, 0] == pytest.approx(jpeg.image[0, 0], abs=2e-3)


def test_a_png_tagged_adobe_rgb_is_read_as_adobe_rgb(tmp_path: Path) -> None:
    photo = decoder.decode(_write_png(tmp_path / "adobe.png", adobe_rgb=True))
    assert photo.color_space is ColorSpace.ADOBE_RGB


def test_a_pngs_embedded_profile_is_kept_for_the_export(tmp_path: Path) -> None:
    path = _write_png(tmp_path / "tagged.png", profile=cm.srgb_profile_bytes())
    photo = decoder.decode(path)
    assert photo.color_space is ColorSpace.SRGB
    assert photo.icc_profile


def test_an_eight_bit_png_keeps_its_own_pixels(tmp_path: Path) -> None:
    """They are what passthrough shows, so they must arrive untouched."""
    path = _write_png(tmp_path / "plain.png", colour=(10, 120, 240), size=(8, 6))
    photo = decoder.decode(path)
    assert photo.encoded is not None
    assert photo.encoded.dtype == np.uint8
    assert photo.encoded.shape == (6, 8, 3)
    assert photo.encoded[0, 0].tolist() == [10, 120, 240]


def test_a_sixteen_bit_png_is_read_at_sixteen_bits(tmp_path: Path) -> None:
    """A step that does not exist in 8 bit has to survive the way in."""
    low = decoder.decode(_write_png16(tmp_path / "low.png", 32768))
    high = decoder.decode(_write_png16(tmp_path / "high.png", 32768 + 64))
    assert high.image[0, 0, 0] > low.image[0, 0, 0]


def test_a_sixteen_bit_png_keeps_no_eight_bit_copy(tmp_path: Path) -> None:
    """Passthrough would have to throw away half the file, so it is off."""
    photo = decoder.decode(_write_png16(tmp_path / "deep.png", 32768))
    assert photo.encoded is None


def test_a_sixteen_bit_png_agrees_with_the_eight_bit_grey_it_matches(
    tmp_path: Path,
) -> None:
    deep = decoder.decode(_write_png16(tmp_path / "deep.png", 128 * 257))
    shallow = decoder.decode(_write_png(tmp_path / "grey.png"))
    assert deep.image[0, 0] == pytest.approx(shallow.image[0, 0], abs=1e-4)


def test_a_fully_transparent_png_is_composited_onto_white(tmp_path: Path) -> None:
    path = _write_rgba_png(tmp_path / "clear.png", (255, 0, 0), alpha=0)
    pixel = decoder.decode(path).image[0, 0]
    assert pixel.min() == pytest.approx(1.0, abs=1e-3), pixel


def test_a_half_transparent_png_lands_between_its_colour_and_white(
    tmp_path: Path,
) -> None:
    """Half an alpha of red over white is a pink: neither the red nor white."""
    path = _write_rgba_png(tmp_path / "half.png", (255, 0, 0), alpha=128)
    mixed = cm.apply_matrix(decoder.decode(path).image, cm.PROPHOTO_TO_SRGB)[0, 0]
    assert mixed[0] == pytest.approx(1.0, abs=1e-3)
    assert 0.05 < mixed[1] < 0.9
    assert mixed[1] == pytest.approx(mixed[2], abs=1e-3)


def test_a_transparent_sixteen_bit_png_is_also_composited_onto_white(
    tmp_path: Path,
) -> None:
    """The deep path has its own compositing, so it is checked on its own."""
    path = tmp_path / "deep_clear.png"
    pixels = np.zeros((8, 8, 4), dtype=np.uint16)
    pixels[..., 2] = 65535  # a red that is not there at all, in BGRA order
    ok, buffer = cv2.imencode(".png", pixels)
    assert ok
    buffer.tofile(path)
    photo = decoder.decode(path)
    assert photo.encoded is None
    assert photo.image[0, 0].min() == pytest.approx(1.0, abs=1e-3), photo.image[0, 0]


def test_a_greyscale_png_is_read_as_three_channels(tmp_path: Path) -> None:
    path = tmp_path / "grey_mode.png"
    Image.fromarray(np.full((8, 8), 128, dtype=np.uint8), "L").save(path)
    photo = decoder.decode(path)
    assert photo.image.shape == (8, 8, 3)
    pixel = photo.image[0, 0]
    assert pixel.max() - pixel.min() < 1e-3, pixel


def test_a_palette_png_is_read_as_colour(tmp_path: Path) -> None:
    path = tmp_path / "palette.png"
    Image.new("RGB", (8, 8), (128, 128, 128)).convert("P").save(path)
    assert decoder.decode(path).image.shape == (8, 8, 3)


def test_a_png_is_turned_upright(tmp_path: Path) -> None:
    """A PNG carries its orientation in the eXIf chunk, and it counts."""
    path = _write_png(tmp_path / "portrait.png", orientation=6)
    assert decoder.decode(path).size == (48, 64)


def test_a_white_png_is_marked_as_clipped(tmp_path: Path) -> None:
    photo = decoder.decode(_write_png(tmp_path / "white.png", colour=(255, 255, 255)))
    assert photo.clipped.all()


def test_a_png_proxy_is_shrunk_to_the_long_edge(tmp_path: Path) -> None:
    path = _write_png(tmp_path / "big.png", size=(400, 300))
    photo = decoder.decode_proxy(path, long_edge=100)
    assert max(photo.size) == 100
    assert photo.full_size == (400, 300)


# -- failures (F-06) ------------------------------------------------------


def test_an_unsupported_extension_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "photo.bmp"
    path.write_bytes(b"not a photo")
    with pytest.raises(DecodeError):
        decoder.decode(path)


def test_a_damaged_png_raises_a_decode_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n this is not a real photo")
    with pytest.raises(DecodeError):
        decoder.decode(path)


def test_a_damaged_jpeg_raises_a_decode_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0 this is not a real photo")
    with pytest.raises(DecodeError):
        decoder.decode(path)


def test_a_damaged_raw_raises_a_decode_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.ARW"
    path.write_bytes(b"\x00" * 2048)
    with pytest.raises(DecodeError):
        decoder.decode(path)


def test_a_missing_file_raises_a_decode_error(tmp_path: Path) -> None:
    with pytest.raises(DecodeError):
        decoder.decode(tmp_path / "nothing.jpg")


def test_an_empty_file_raises_a_decode_error(tmp_path: Path) -> None:
    path = tmp_path / "empty.jpg"
    path.touch()
    with pytest.raises(DecodeError):
        decoder.decode(path)


# -- RAW (needs a sample photo) -------------------------------------------


@requires_raw
def test_a_raw_file_decodes_to_linear_camera_rgb() -> None:
    photo = decoder.decode(_sample_raws()[0])
    assert photo.is_raw
    assert photo.image.dtype == np.float32
    assert photo.camera_to_xyz is not None
    assert photo.camera_whitebalance is not None
    assert photo.image.min() >= 0.0


@requires_raw
def test_the_white_balance_is_normalised_on_green() -> None:
    photo = decoder.decode(_sample_raws()[0])
    assert photo.camera_whitebalance is not None
    assert photo.camera_whitebalance[1] == pytest.approx(1.0)


@requires_raw
def test_the_camera_matrix_sends_camera_neutral_to_d65() -> None:
    photo = decoder.decode(_sample_raws()[0])
    assert photo.camera_to_xyz is not None
    assert np.allclose(photo.camera_to_xyz @ np.ones(3), cm.WHITE_D65, atol=1e-3)


@requires_raw
def test_a_raw_file_has_an_embedded_preview() -> None:
    preview = decoder.embedded_preview(_sample_raws()[0])
    assert preview is not None
    assert preview.dtype == np.uint8
    assert preview.shape[2] == 3


@requires_raw
def test_the_proxy_of_a_raw_file_has_the_documented_long_edge() -> None:
    sample = _sample_raws()[0]
    proxy = decoder.decode_proxy(sample)
    full = decoder.decode(sample)
    assert max(proxy.image.shape[:2]) == decoder.PROXY_LONG_EDGE
    assert proxy.is_raw and proxy.camera_whitebalance == full.camera_whitebalance
    assert proxy.camera_to_xyz is not None
    # The proxy keeps the aspect ratio of the full size photo.
    assert proxy.image.shape[1] / proxy.image.shape[0] == pytest.approx(
        full.image.shape[1] / full.image.shape[0], abs=1e-3
    )


@requires_raw
def test_a_raw_too_small_to_halve_matches_a_full_decode_shrunk_afterwards() -> None:
    """Shrinking before the conversion to float must not change the result."""
    sample = _sample_raws()[0]
    full = decoder.decode(sample)
    long_edge = max(full.size) * 3 // 4  # half size would be too small
    proxy = decoder.decode_proxy(sample, long_edge)
    shrunk = decoder.make_proxy(full.image, long_edge)
    assert proxy.image.shape == shrunk.shape
    assert np.allclose(proxy.image, shrunk, atol=2e-3)


@requires_raw
def test_a_raw_proxy_read_at_half_size_looks_like_the_full_decode() -> None:
    """The proxy is read at half size, which is fast and gives the same picture."""
    sample = _sample_raws()[0]
    full = decoder.decode(sample)
    proxy = decoder.decode_proxy(sample)
    shrunk = decoder.make_proxy(full.image)
    assert proxy.image.shape == shrunk.shape
    # It still knows how large the photo really is (export size, crop).
    assert proxy.full_size == full.size
    # The same colours overall, and the same pixels give or take the
    # sharpness a half size read trades for its speed.
    means = proxy.image.reshape(-1, 3).mean(axis=0)
    assert np.allclose(means, shrunk.reshape(-1, 3).mean(axis=0), rtol=0.01)
    assert float(np.median(np.abs(proxy.image - shrunk))) < 2e-3


def test_a_half_size_clip_mask_marks_a_pixel_if_any_of_its_sites_was() -> None:
    mask = np.zeros((5, 7), dtype=np.bool_)
    mask[0, 1] = True  # one site of the first block
    mask[3, 3] = mask[2, 2] = True  # two of another
    mask[4, 6] = True  # the odd last row and column are left out
    halved = decoder._halve_mask(mask)
    assert halved.shape == (2, 3)
    assert halved.tolist() == [[True, False, False], [False, True, False]]


def test_eight_bit_photos_decode_to_the_same_values_by_table(tmp_path: Path) -> None:
    """The lookup table gives exactly what decoding pixel by pixel gave."""
    pixels = np.arange(256, dtype=np.uint8).repeat(3).reshape(16, 16, 3)
    for space in (ColorSpace.SRGB, ColorSpace.ADOBE_RGB):
        expected = decoder._to_linear(pixels.astype(np.float32) / np.float32(255), space)
        assert np.array_equal(decoder._linear_table(space)[pixels], expected)


def test_a_large_jpeg_proxy_is_decoded_small_and_knows_its_full_size(tmp_path: Path) -> None:
    path = _write_jpeg(tmp_path / "large.jpg", (90, 140, 200), size=(1200, 900))
    proxy = decoder.decode_proxy(path, long_edge=300)
    assert proxy.size == (300, 225)
    assert proxy.full_size == (1200, 900)
    assert proxy.encoded is not None and proxy.encoded.shape[:2] == (225, 300)
    # A flat colour stays that colour however it was scaled.
    assert np.abs(proxy.encoded.astype(int) - (90, 140, 200)).max() <= 2


def test_a_sideways_jpeg_proxy_reports_its_upright_full_size(tmp_path: Path) -> None:
    path = _write_jpeg(tmp_path / "turned.jpg", size=(1200, 900), orientation=6)
    proxy = decoder.decode_proxy(path, long_edge=300)
    assert proxy.size == (225, 300)
    assert proxy.full_size == (900, 1200)


def test_the_proxy_of_a_small_jpeg_is_the_photo_itself(tmp_path: Path) -> None:
    photo = decoder.decode_proxy(_write_jpeg(tmp_path / "small.jpg", size=(64, 48)))
    assert photo.size == (64, 48)
    assert photo.encoded is not None


def test_the_proxy_of_a_large_jpeg_is_shrunk(tmp_path: Path) -> None:
    photo = decoder.decode_proxy(
        _write_jpeg(tmp_path / "large.jpg", size=(1200, 900)), long_edge=600
    )
    assert photo.size == (600, 450)
    assert photo.encoded is not None
    assert photo.encoded.shape[:2] == (450, 600)


def test_decode_proxy_reports_a_damaged_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.ARW"
    path.write_bytes(bytes(2048))
    with pytest.raises(DecodeError):
        decoder.decode_proxy(path)


@requires_portrait_raw
def test_a_portrait_raw_photo_is_turned_upright() -> None:
    """PLAN.md F-04: a photo taken sideways is shown the right way up.

    LibRaw writes the requested rotation back into sizes.flip, so reading it
    after processing reports every photo as upright; this is the check that
    the camera's own value is the one used.
    """
    photo = decoder.decode_proxy(_portrait_raws()[0])
    width, height = photo.size
    assert height > width, f"came out {width}x{height}"


@requires_portrait_raw
def test_the_embedded_preview_is_upright_too() -> None:
    """The quick preview must not appear on its side before the real decode."""
    for path in _portrait_raws():
        preview = decoder.embedded_preview(path)
        assert preview is not None
        assert preview.shape[0] > preview.shape[1], path.name


def test_a_preview_already_upright_is_left_alone() -> None:
    """Some bodies store a rotated photo's preview already turned (see _orient_preview)."""
    upright = np.zeros((160, 120, 3), dtype=np.uint8)  # portrait
    result = decoder._orient_preview(upright, orientation=8, sensor_landscape=True)
    assert result.shape == upright.shape


def test_a_preview_still_on_its_side_is_turned() -> None:
    sideways = np.zeros((120, 160, 3), dtype=np.uint8)  # landscape, like the sensor
    result = decoder._orient_preview(sideways, orientation=8, sensor_landscape=True)
    assert result.shape[:2] == (160, 120)


def test_an_upright_photo_never_has_its_preview_turned() -> None:
    preview = np.zeros((120, 160, 3), dtype=np.uint8)
    assert decoder._orient_preview(preview, 1, True).shape == preview.shape
    assert decoder._orient_preview(preview, 3, True).shape == preview.shape


@requires_portrait_raw
def test_the_preview_and_the_decode_agree_on_the_shape() -> None:
    """The provisional picture and the real one must not swap proportions."""
    path = _portrait_raws()[0]
    preview = decoder.embedded_preview(path)
    photo = decoder.decode_proxy(path)
    assert preview is not None
    preview_ratio = preview.shape[1] / preview.shape[0]
    photo_ratio = photo.image.shape[1] / photo.image.shape[0]
    assert preview_ratio == pytest.approx(photo_ratio, abs=0.02)


@requires_raw
def test_every_sample_agrees_between_its_preview_and_its_decode() -> None:
    """Whatever a camera does with its preview, the two must match."""
    for path in _sample_raws():
        preview = decoder.embedded_preview(path)
        photo = decoder.decode_proxy(path)
        if preview is None:
            continue
        preview_landscape = preview.shape[1] >= preview.shape[0]
        photo_landscape = photo.image.shape[1] >= photo.image.shape[0]
        assert preview_landscape == photo_landscape, path.name


#: How much of the frame a photo with a blown sky reports as clipped. It is
#: a property of the sample set rather than of the code, so falling short of
#: it means a sample is missing, not that the mask is wrong.
BLOWN_SKY_COVERAGE = 0.001


@requires_raw
def test_a_blown_sky_is_found_in_the_sensor_data() -> None:
    """At least one sample must report clipped pixels at proxy size.

    Looking for clipping in the processed image finds nothing: LibRaw blends
    the highlights and white balances, so a fully saturated sky comes out well
    below full scale. If every sample reports nothing, the mask is broken
    again and the highlight correction never runs - that is the regression
    this holds, and it is checked whatever the samples happen to be.

    Whether any sample is a real blown sky is a separate matter: without one
    the mask is only being exercised on a few hundred specular pixels, so the
    stricter half says so rather than failing.
    """
    coverage = [float(decoder.decode_proxy(path).clipped.mean()) for path in _sample_raws()]
    assert max(coverage) > 0.0, "the clipping mask is empty for every sample"
    if max(coverage) < BLOWN_SKY_COVERAGE:
        pytest.skip(
            f"no sample has a blown sky: the most any of them clips is "
            f"{max(coverage) * 100:.4f}% of the frame, where a blown sky gives "
            f"over {BLOWN_SKY_COVERAGE * 100:.1f}%"
        )


@requires_raw
def test_the_clipping_mask_survives_the_shrink_to_proxy_size() -> None:
    """The proxy's mask must cover about as much as the full size one."""
    for path in _sample_raws():
        full = decoder._read_raw(path).clipped
        if full.mean() < 0.002:
            continue
        proxy = decoder.decode_proxy(path).clipped
        assert proxy.mean() == pytest.approx(full.mean(), abs=0.003), path.name
        return
    pytest.skip("no sample has enough clipped highlights to compare")


@requires_raw
def test_every_sample_keeps_its_pixel_count_through_orientation() -> None:
    """Turning a photo upright must not crop or stretch it."""
    for path in _sample_raws():
        photo = decoder.decode_proxy(path)
        width, height = photo.size
        assert max(width, height) == decoder.PROXY_LONG_EDGE, path.name


@requires_jpeg
def test_a_camera_jpeg_is_read_with_the_right_colour_space() -> None:
    """A real camera file, not one the test wrote itself.

    The sample is tagged Adobe RGB the way a camera does it, through the
    EXIF ColorSpace and InteropIndex tags with no embedded profile, which is
    the fallback path of PLAN.md section 6 step 1.
    """
    for path in _sample_jpegs():
        photo = decoder.decode_proxy(path)
        assert photo.color_space in (ColorSpace.SRGB, ColorSpace.ADOBE_RGB), path.name
        assert photo.encoded is not None
        assert not photo.is_raw


@requires_jpeg
def test_a_camera_jpeg_is_turned_upright() -> None:
    """The decoded size must match what the EXIF orientation asks for."""
    from PIL import Image as PillowImage

    for path in _sample_jpegs():
        with PillowImage.open(path) as image:
            stored = image.size
            orientation = int(image.getexif().get(274, 1) or 1)
        photo = decoder.decode(path)
        expected = (stored[1], stored[0]) if orientation in (5, 6, 7, 8) else stored
        assert photo.size == expected, path.name


@requires_jpeg
def test_a_camera_jpeg_keeps_neutral_grey_neutral() -> None:
    """Whatever its colour space, grey must come out grey."""
    for path in _sample_jpegs():
        photo = decoder.decode_proxy(path)
        assert photo.encoded is not None
        exactly_grey = np.all(photo.encoded == photo.encoded[..., :1], axis=2)
        if not exactly_grey.any():
            continue
        pixels = cm.apply_matrix(photo.image[exactly_grey], cm.PROPHOTO_TO_SRGB)
        spread = float(np.mean(pixels.max(axis=1) - pixels.min(axis=1)))
        assert spread < 2e-3, f"{path.name}: {spread}"


def test_a_jpeg_has_no_embedded_preview(tmp_path: Path) -> None:
    assert decoder.embedded_preview(_write_jpeg(tmp_path / "plain.jpg")) is None


# -- film grain seed (PLAN.md 6 step 11.5) ---------------------------------


@requires_jpeg
def test_a_decoded_photo_and_its_proxy_carry_the_grain_seed_of_their_name() -> None:
    path = _sample_jpegs()[0]
    seed = decoder.grain_seed_for(path)

    assert decoder.decode(path).grain_seed == seed
    assert decoder.decode_proxy(path).grain_seed == seed
    assert decoder.resize_photo(decoder.decode_proxy(path), 600).grain_seed == seed


@requires_raw
def test_the_embedded_preview_comes_with_the_size_the_decode_will_report() -> None:
    """So the zoom is measured against the photo from the first moment, not
    against the much smaller preview (PLAN.md 3.3, 3.7)."""
    for path in _sample_raws()[:3]:
        found = decoder.embedded_preview_and_size(path)
        assert found is not None
        _preview, size = found
        assert size == decoder.decode(path).full_size, path.name
