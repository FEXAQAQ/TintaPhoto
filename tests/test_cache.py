"""Tests for the thumbnail cache and thumbnail making (PLAN.md section 4)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import piexif
import pytest
from PIL import Image

from core import thumbnails
from core.cache import DEFAULT_CACHE_LIMIT_BYTES, ThumbnailCache
from core.decoder import DecodeError

SAMPLES = Path(__file__).resolve().parent / "samples"


def _sample_raws() -> list[Path]:
    return sorted({*SAMPLES.glob("*.ARW"), *SAMPLES.glob("*.arw")})


requires_raw = pytest.mark.skipif(not _sample_raws(), reason="no sample RAW file available")


def _write_jpeg(
    path: Path, size: tuple[int, int] = (1200, 800), orientation: int | None = None
) -> Path:
    image = Image.new("RGB", size, (90, 140, 200))
    if orientation is None:
        image.save(path, "JPEG")
    else:
        exif = piexif.dump({"0th": {piexif.ImageIFD.Orientation: orientation}})
        image.save(path, "JPEG", exif=exif)
    return path


def _write_png(path: Path, size: tuple[int, int] = (1200, 800), alpha: int | None = None) -> Path:
    if alpha is None:
        Image.new("RGB", size, (90, 140, 200)).save(path, "PNG")
        return path
    Image.new("RGBA", size, (90, 140, 200, alpha)).save(path, "PNG")
    return path


def _thumbnail(colour: tuple[int, int, int] = (10, 20, 30)) -> np.ndarray:
    return np.full((120, 160, 3), colour, dtype=np.uint8)


# -- making thumbnails ----------------------------------------------------


def test_a_jpeg_thumbnail_fits_in_the_requested_box(tmp_path: Path) -> None:
    thumbnail = thumbnails.make(_write_jpeg(tmp_path / "photo.jpg"), long_edge=256)
    assert max(thumbnail.shape[:2]) == 256
    assert thumbnail.dtype == np.uint8
    assert thumbnail.shape[2] == 3


def test_a_png_thumbnail_fits_in_the_requested_box(tmp_path: Path) -> None:
    thumbnail = thumbnails.make(_write_png(tmp_path / "photo.png"), long_edge=256)
    assert max(thumbnail.shape[:2]) == 256
    assert thumbnail.dtype == np.uint8
    assert thumbnail.shape[2] == 3


def test_a_transparent_png_thumbnail_is_flattened_onto_white(tmp_path: Path) -> None:
    """The decoder flattens onto white, so the filmstrip must agree with it."""
    path = _write_png(tmp_path / "clear.png", size=(200, 200), alpha=0)
    thumbnail = thumbnails.make(path, long_edge=64)
    assert thumbnail[0, 0].tolist() == [255, 255, 255]


def test_a_small_photo_is_not_enlarged(tmp_path: Path) -> None:
    thumbnail = thumbnails.make(_write_jpeg(tmp_path / "small.jpg", size=(100, 60)))
    assert thumbnail.shape[:2] == (60, 100)


def test_a_thumbnail_keeps_the_aspect_ratio(tmp_path: Path) -> None:
    thumbnail = thumbnails.make(_write_jpeg(tmp_path / "wide.jpg", size=(1600, 400)), 256)
    assert thumbnail.shape[1] / thumbnail.shape[0] == pytest.approx(4.0, abs=0.05)


def test_a_rotated_photo_gives_an_upright_thumbnail(tmp_path: Path) -> None:
    path = _write_jpeg(tmp_path / "portrait.jpg", size=(1200, 800), orientation=6)
    thumbnail = thumbnails.make(path, long_edge=256)
    assert thumbnail.shape[0] > thumbnail.shape[1]


def test_a_damaged_photo_reports_that_it_has_no_thumbnail(tmp_path: Path) -> None:
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"\xff\xd8 not really a photo")
    with pytest.raises(DecodeError):
        thumbnails.make(path)


def test_a_thumbnail_survives_a_trip_through_the_cache_format() -> None:
    original = _thumbnail((200, 100, 50))
    restored = thumbnails.decode(thumbnails.encode(original) or b"")
    assert restored is not None
    assert restored.shape == original.shape
    assert np.allclose(restored.astype(int), original.astype(int), atol=4)


def test_damaged_cache_data_decodes_to_nothing() -> None:
    assert thumbnails.decode(b"this is not a jpeg") is None


@requires_raw
def test_a_raw_thumbnail_comes_from_the_embedded_preview() -> None:
    """It must be quick, which means not demosaicing the whole photo."""
    start = time.perf_counter()
    thumbnail = thumbnails.make(_sample_raws()[0], long_edge=256)
    elapsed = time.perf_counter() - start
    assert max(thumbnail.shape[:2]) == 256
    assert elapsed < 0.5, f"took {elapsed:.2f} s, so it was not the embedded preview"


# -- the cache ------------------------------------------------------------


def test_a_stored_thumbnail_comes_back(tmp_path: Path) -> None:
    photo = _write_jpeg(tmp_path / "photo.jpg")
    cache = ThumbnailCache(tmp_path / "cache")
    assert cache.get(photo) is None
    assert cache.put(photo, _thumbnail())
    restored = cache.get(photo)
    assert restored is not None
    assert restored.shape == (120, 160, 3)


def test_editing_the_photo_invalidates_its_thumbnail(tmp_path: Path) -> None:
    """The key covers size and modification time, so a changed file misses."""
    photo = _write_jpeg(tmp_path / "photo.jpg")
    cache = ThumbnailCache(tmp_path / "cache")
    cache.put(photo, _thumbnail())
    assert cache.get(photo) is not None

    time.sleep(0.01)
    _write_jpeg(photo, size=(800, 600))
    assert cache.get(photo) is None


def test_two_photos_do_not_share_an_entry(tmp_path: Path) -> None:
    first = _write_jpeg(tmp_path / "one.jpg")
    second = _write_jpeg(tmp_path / "two.jpg", size=(900, 600))
    cache = ThumbnailCache(tmp_path / "cache")
    cache.put(first, _thumbnail((255, 0, 0)))
    cache.put(second, _thumbnail((0, 0, 255)))
    assert cache.get(first)[0, 0, 0] > cache.get(second)[0, 0, 0]


def test_a_missing_photo_has_no_key(tmp_path: Path) -> None:
    assert ThumbnailCache.key(tmp_path / "nothing.jpg") is None


def test_the_key_changes_when_the_thumbnail_format_does(tmp_path: Path) -> None:
    """A fix to how thumbnails are made must retire the cached ones."""
    from core import cache as cache_module

    photo = _write_jpeg(tmp_path / "photo.jpg")
    before = ThumbnailCache.key(photo)
    original = cache_module._THUMBNAIL_VERSION
    try:
        cache_module._THUMBNAIL_VERSION = original + 1
        assert ThumbnailCache.key(photo) != before
    finally:
        cache_module._THUMBNAIL_VERSION = original


def test_the_cache_reports_what_it_is_using(tmp_path: Path) -> None:
    photo = _write_jpeg(tmp_path / "photo.jpg")
    cache = ThumbnailCache(tmp_path / "cache")
    assert cache.usage_bytes() == 0
    cache.put(photo, _thumbnail())
    assert cache.usage_bytes() > 0


def test_clearing_the_cache_empties_it(tmp_path: Path) -> None:
    photo = _write_jpeg(tmp_path / "photo.jpg")
    cache = ThumbnailCache(tmp_path / "cache")
    cache.put(photo, _thumbnail())
    cache.clear()
    assert cache.usage_bytes() == 0
    assert cache.get(photo) is None


def test_writing_does_not_measure_the_whole_cache_every_time(tmp_path: Path) -> None:
    """Measuring is O(everything), so it must not happen per thumbnail.

    Doing it on every write made opening a folder of three hundred photos
    stall the interface for half a second.
    """
    cache = ThumbnailCache(tmp_path / "cache")
    measured = 0
    original = ThumbnailCache._entries

    def counting(self):  # noqa: ANN001, ANN202 - a test stand-in
        nonlocal measured
        measured += 1
        return original(self)

    ThumbnailCache._entries = counting  # type: ignore[method-assign]
    try:
        for index in range(30):
            cache.put(_write_jpeg(tmp_path / f"p{index}.jpg", size=(120, 90)), _thumbnail())
    finally:
        ThumbnailCache._entries = original  # type: ignore[method-assign]
    assert measured == 0, f"the cache measured itself {measured} times in 30 writes"


def test_the_cache_is_trimmed_when_it_passes_its_limit(tmp_path: Path) -> None:
    cache = ThumbnailCache(tmp_path / "cache", limit_bytes=8 * 1024)
    photos = [_write_jpeg(tmp_path / f"photo{index}.jpg", size=(200, 150)) for index in range(12)]
    for index, photo in enumerate(photos):
        cache.put(photo, _thumbnail((index * 20, 100, 100)))
        time.sleep(0.002)  # so that the use times are distinguishable
    cache.trim()
    assert cache.usage_bytes() <= cache.limit_bytes


def test_trimming_keeps_the_most_recently_used(tmp_path: Path) -> None:
    cache = ThumbnailCache(tmp_path / "cache", limit_bytes=6 * 1024)
    photos = [_write_jpeg(tmp_path / f"p{index}.jpg", size=(200, 150)) for index in range(10)]
    for photo in photos:
        cache.put(photo, _thumbnail())
        time.sleep(0.002)
    cache.trim()
    # The ones written last are the ones still there.
    assert cache.get(photos[-1]) is not None
    assert cache.get(photos[0]) is None


def test_lowering_the_limit_trims_at_once(tmp_path: Path) -> None:
    cache = ThumbnailCache(tmp_path / "cache", limit_bytes=DEFAULT_CACHE_LIMIT_BYTES)
    for index in range(6):
        cache.put(_write_jpeg(tmp_path / f"p{index}.jpg", size=(200, 150)), _thumbnail())
        time.sleep(0.002)
    before = cache.usage_bytes()
    cache.set_limit(2 * 1024)
    assert cache.usage_bytes() < before
    assert cache.usage_bytes() <= 2 * 1024


def test_a_cache_folder_that_cannot_be_made_is_not_fatal(tmp_path: Path) -> None:
    """A file where the folder should be makes the cache unusable, not broken."""
    blocked = tmp_path / "blocked"
    blocked.write_text("not a folder", encoding="utf-8")
    cache = ThumbnailCache(blocked)
    photo = _write_jpeg(tmp_path / "photo.jpg")
    assert cache.put(photo, _thumbnail()) is False
    assert cache.get(photo) is None
    assert cache.usage_bytes() == 0
