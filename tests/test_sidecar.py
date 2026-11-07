"""Tests for the parameter file (PLAN.md section 7.1, E-01 to E-03).

The one that matters most is at the end: whatever happens, the photo itself
must come out byte for byte the way it went in (PLAN.md section 10, rule 8).
"""

from __future__ import annotations

import codecs
import hashlib
import json
from pathlib import Path

import pytest

from core import sidecar
from core.edit_state import Edit, EditState
from core.geometry import Geometry, Rect

APPLICATION = sidecar.Application("TintaPhoto", "0.1.0")

#: A RAW photo rests at the white balance the camera recorded, not at zero,
#: which is what makes the reference state worth passing around.
RAW_REFERENCE = EditState.for_photo(True, temperature=5600.0, tint=12.0)
JPEG_REFERENCE = EditState.for_photo(False)


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    """Return a file standing in for a photo, with recognisable contents."""
    path = tmp_path / "DSC01234.ARW"
    path.write_bytes(b"not really a raw file, but a file all the same")
    return path


#: Where the parameter files go, kept apart from the photo's folder.
STORE_NAME = "store"


def _store(photo: Path) -> Path:
    """Return the store of the test *photo* belongs to."""
    return photo.parent / STORE_NAME


def _save(
    photo: Path,
    state: EditState,
    reference: EditState,
    geometry: Geometry | None = None,
):
    """Store one edit of *photo*, the way the controller does."""
    return sidecar.save(
        photo,
        Edit(state, geometry) if geometry is not None else Edit(state),
        Edit(reference),
        APPLICATION,
        _store(photo),
    )


def _read(photo: Path, reference: EditState):
    """Read *photo*'s parameters from the test store."""
    return sidecar.read(photo, reference, _store(photo))


def _file(photo: Path) -> Path:
    """Return the file *photo*'s parameters are written to."""
    return sidecar.stored_path(photo, _store(photo))


def _write_old_file(photo: Path, document: dict) -> Path:
    """Leave a parameter file beside *photo*, the way 0.34 and earlier did."""
    beside = sidecar.legacy_path(photo)
    beside.write_text(json.dumps(document), encoding="utf-8")
    return beside


# -- where the file goes ---------------------------------------------------


def test_nothing_is_written_into_the_photo_folder(photo: Path) -> None:
    """PLAN.md 7.1: the folder holds the photos and nothing else."""
    before = sorted(path.name for path in photo.parent.iterdir())

    _save(photo, RAW_REFERENCE.with_value("exposure", 1.0), RAW_REFERENCE)
    _save(photo, RAW_REFERENCE.with_value("contrast", 30.0), RAW_REFERENCE)
    sidecar.update_adjustments(photo, {"exposure": 0.2}, APPLICATION, _store(photo))

    after = sorted(path.name for path in photo.parent.iterdir() if path.name != STORE_NAME)
    assert after == before
    assert _file(photo).is_file()


def test_the_store_is_created_when_it_is_first_needed(photo: Path) -> None:
    assert not _store(photo).exists()

    result = _save(photo, RAW_REFERENCE.with_value("exposure", 1.0), RAW_REFERENCE)

    assert result.path == _file(photo)
    assert result.path.parent == _store(photo)


def test_a_raw_and_a_jpeg_of_the_same_shot_have_separate_files(tmp_path: Path) -> None:
    """F-05: RAW+JPEG shooting must not make two photos share one file."""
    raw = tmp_path / "DSC01234.ARW"
    jpeg = tmp_path / "DSC01234.JPG"
    raw.write_bytes(b"raw")
    jpeg.write_bytes(b"jpeg")

    _save(raw, RAW_REFERENCE.with_value("exposure", 1.0), RAW_REFERENCE)
    _save(jpeg, JPEG_REFERENCE.with_value("contrast", 40.0), JPEG_REFERENCE)

    assert _file(raw) != _file(jpeg)
    assert _file(raw).is_file()
    assert _file(jpeg).is_file()
    assert _read(raw, RAW_REFERENCE).adjustments.exposure == 1.0
    assert _read(jpeg, JPEG_REFERENCE).adjustments.contrast == 40.0
    assert _read(jpeg, JPEG_REFERENCE).adjustments.exposure == 0.0


def test_two_photos_of_the_same_name_stay_apart(tmp_path: Path) -> None:
    first = tmp_path / "january" / "DSC01234.ARW"
    second = tmp_path / "february" / "DSC01234.ARW"
    for path in (first, second):
        path.parent.mkdir()
        path.write_bytes(b"photo")
    store = tmp_path / "store"

    assert sidecar.stored_path(first, store) != sidecar.stored_path(second, store)


def test_a_path_differing_only_in_case_is_the_same_photo(photo: Path) -> None:
    """Windows reaches one file by paths that differ only in case."""
    other = photo.with_name(photo.name.lower())

    assert sidecar.store_key(photo) == sidecar.store_key(other)


# -- a round trip ----------------------------------------------------------


def test_every_adjustment_survives_a_round_trip(photo: Path) -> None:
    state = EditState(
        temperature=4800.0,
        tint=-20.0,
        exposure=0.35,
        contrast=10.0,
        highlights=-25.0,
        shadows=20.0,
        vibrance=15.0,
        saturation=-5.0,
        is_raw=True,
    )
    _save(photo, state, RAW_REFERENCE)

    restored = _read(photo, RAW_REFERENCE)
    assert restored is not None
    assert restored.adjustments == state


def test_reading_a_photo_without_a_file_returns_nothing(photo: Path) -> None:
    assert _read(photo, RAW_REFERENCE) is None
    assert not sidecar.is_edited(photo, _store(photo))


def test_the_file_holds_the_documented_shape(photo: Path) -> None:
    """PLAN.md 7.1 fixes the blocks a parameter file is made of."""
    _save(photo, RAW_REFERENCE.with_value("exposure", 0.3), RAW_REFERENCE)

    with _file(photo).open(encoding="utf-8") as handle:
        document = json.load(handle)

    assert document["version"] == sidecar.SIDECAR_VERSION
    assert document["source"]["file"] == photo.name
    assert document["source"]["size_bytes"] == photo.stat().st_size
    assert document["adjustments"]["exposure"] == 0.3
    assert document["geometry"] == Geometry().to_dict()
    assert document["app"] == {"name": "TintaPhoto", "version": "0.1.0"}
    assert "updated_at" in document


@pytest.fixture
def store(photo: Path) -> Path:
    """Return the store of *photo*, made ready for files written by hand."""
    path = _store(photo)
    path.mkdir()
    return path


def test_a_missing_field_falls_back_to_where_the_photo_rests(
    photo: Path, store: Path
) -> None:
    """PLAN.md 7.1: 欄位缺漏用預設值補上."""
    _file(photo).write_text(
        json.dumps({"version": 1, "adjustments": {"exposure": 0.5}}), encoding="utf-8"
    )

    restored = _read(photo, RAW_REFERENCE)
    assert restored is not None
    assert restored.adjustments.exposure == 0.5
    # Not zero: a RAW photo's white balance rests at the camera's own value.
    assert restored.adjustments.temperature == RAW_REFERENCE.temperature
    assert restored.adjustments.tint == RAW_REFERENCE.tint


def test_a_value_outside_its_range_is_brought_back_in(photo: Path, store: Path) -> None:
    _file(photo).write_text(
        json.dumps({"adjustments": {"exposure": 99.0, "contrast": -4000.0}}),
        encoding="utf-8",
    )

    restored = _read(photo, RAW_REFERENCE)
    assert restored is not None
    assert restored.adjustments.exposure == 5.0
    assert restored.adjustments.contrast == -100.0


def test_nonsense_values_are_ignored(photo: Path, store: Path) -> None:
    _file(photo).write_text(
        json.dumps({"adjustments": {"exposure": "quite a lot", "contrast": None}}),
        encoding="utf-8",
    )

    restored = _read(photo, JPEG_REFERENCE)
    assert restored is not None
    assert restored.adjustments == JPEG_REFERENCE


def test_the_geometry_survives_a_round_trip(photo: Path) -> None:
    """PLAN.md 7.1: the crop and rotation are stored with the adjustments."""
    frame = Geometry(
        quarter_turns=1,
        angle=-1.2,
        crop=Rect(0.05, 0.08, 0.9, 0.84),
        aspect="3:2",
        orientation="portrait",
    )

    _save(photo, RAW_REFERENCE.with_value("exposure", 0.2), RAW_REFERENCE, geometry=frame)

    restored = _read(photo, RAW_REFERENCE)
    assert restored is not None
    assert restored.geometry == frame
    assert restored.adjustments.exposure == 0.2


def test_unknown_fields_are_preserved(photo: Path, store: Path) -> None:
    """PLAN.md 7.1: 未知欄位保留不刪除."""
    _file(photo).write_text(
        json.dumps(
            {
                "version": 1,
                "adjustments": {"exposure": 0.2},
                "future_field": {"something": "a later version wrote this"},
            }
        ),
        encoding="utf-8",
    )

    restored = _read(photo, RAW_REFERENCE)
    assert restored is not None
    assert restored.extra == {"future_field": {"something": "a later version wrote this"}}

    sidecar.save(
        photo,
        restored.edit.with_adjustments(restored.adjustments.with_value("exposure", 0.4)),
        Edit(RAW_REFERENCE),
        APPLICATION,
        store,
        extra=restored.extra,
    )
    again = _read(photo, RAW_REFERENCE)
    assert again is not None
    assert again.extra == restored.extra
    assert again.adjustments.exposure == 0.4


# -- deleting --------------------------------------------------------------


def test_a_photo_back_at_its_defaults_has_its_file_deleted(photo: Path) -> None:
    """PLAN.md 7.1: 所有參數都是預設值且沒有裁切時，刪除參數檔."""
    _save(photo, RAW_REFERENCE.with_value("exposure", 1.0), RAW_REFERENCE)
    assert _file(photo).is_file()

    result = _save(photo, RAW_REFERENCE, RAW_REFERENCE)

    assert result.deleted
    assert result.path is None
    assert not _file(photo).exists()
    assert not sidecar.is_edited(photo, _store(photo))


def test_an_untouched_photo_is_never_given_a_file(photo: Path) -> None:
    result = _save(photo, RAW_REFERENCE, RAW_REFERENCE)

    assert not result.deleted
    assert list(photo.parent.rglob("*.json")) == []


def test_a_photo_with_a_crop_keeps_its_file_even_at_default_adjustments(
    photo: Path,
) -> None:
    """A crop is worth storing on its own, with nothing else adjusted."""
    cropped = Geometry(crop=Rect(0.1, 0.1, 0.8, 0.8))

    result = _save(photo, RAW_REFERENCE, RAW_REFERENCE, geometry=cropped)

    assert result.path == _file(photo)
    assert _read(photo, RAW_REFERENCE).geometry.crop == cropped.crop


def test_a_photo_whose_crop_is_undone_loses_its_file(photo: Path) -> None:
    _save(photo, RAW_REFERENCE, RAW_REFERENCE, geometry=Geometry(angle=3.0))
    assert _file(photo).is_file()

    result = _save(photo, RAW_REFERENCE, RAW_REFERENCE, geometry=Geometry())

    assert result.deleted
    assert not _file(photo).exists()


# -- files beside the photo, from 0.34 and earlier (PLAN.md 7.1) ----------


def test_an_old_file_beside_the_photo_is_still_read(photo: Path) -> None:
    _write_old_file(photo, {"version": 1, "adjustments": {"exposure": 0.8}})

    restored = _read(photo, RAW_REFERENCE)

    assert restored is not None
    assert restored.adjustments.exposure == 0.8
    assert sidecar.is_edited(photo, _store(photo))


def test_the_store_wins_over_an_old_file_and_the_old_file_is_left_alone(
    photo: Path,
) -> None:
    beside = _write_old_file(photo, {"version": 1, "adjustments": {"exposure": 0.8}})
    untouched = beside.read_bytes()

    result = _save(photo, RAW_REFERENCE.with_value("exposure", 0.2), RAW_REFERENCE)

    assert result.path == _file(photo)
    assert _read(photo, RAW_REFERENCE).adjustments.exposure == 0.2
    assert beside.read_bytes() == untouched


def test_pasting_onto_a_photo_with_an_old_file_keeps_the_rest_of_it(photo: Path) -> None:
    """The old file is where a pasted change starts from; it is not rewritten."""
    beside = _write_old_file(
        photo, {"version": 1, "adjustments": {"exposure": 0.8, "contrast": 25.0}}
    )
    untouched = beside.read_bytes()

    sidecar.update_adjustments(photo, {"exposure": 0.3}, APPLICATION, _store(photo))

    restored = _read(photo, RAW_REFERENCE)
    assert restored.adjustments.exposure == 0.3
    assert restored.adjustments.contrast == 25.0
    assert beside.read_bytes() == untouched


def test_resetting_a_photo_with_an_old_file_does_not_bring_it_back(photo: Path) -> None:
    """The old file must not quietly return once the store's copy is gone."""
    beside = _write_old_file(photo, {"version": 1, "adjustments": {"exposure": 0.8}})
    untouched = beside.read_bytes()
    _save(photo, RAW_REFERENCE.with_value("exposure", 0.2), RAW_REFERENCE)

    result = _save(photo, RAW_REFERENCE, RAW_REFERENCE)

    assert result.deleted
    assert result.path is None
    assert _read(photo, RAW_REFERENCE) is None
    assert not sidecar.is_edited(photo, _store(photo))
    assert beside.read_bytes() == untouched


def test_resetting_straight_from_an_old_file_hides_it(photo: Path) -> None:
    """Opened from the old file and reset before anything else was saved."""
    _write_old_file(photo, {"version": 1, "adjustments": {"exposure": 0.8}})

    assert sidecar.delete(photo, _store(photo))
    assert _read(photo, RAW_REFERENCE) is None
    assert not sidecar.delete(photo, _store(photo))


def test_adjusting_again_after_a_reset_is_stored_as_usual(photo: Path) -> None:
    _write_old_file(photo, {"version": 1, "adjustments": {"exposure": 0.8}})
    _save(photo, RAW_REFERENCE, RAW_REFERENCE)

    _save(photo, RAW_REFERENCE.with_value("exposure", 0.4), RAW_REFERENCE)

    assert _read(photo, RAW_REFERENCE).adjustments.exposure == 0.4
    _save(photo, RAW_REFERENCE, RAW_REFERENCE)
    assert _read(photo, RAW_REFERENCE) is None


def test_a_photo_without_an_old_file_leaves_no_mark_when_reset(photo: Path) -> None:
    _save(photo, RAW_REFERENCE.with_value("exposure", 0.4), RAW_REFERENCE)

    _save(photo, RAW_REFERENCE, RAW_REFERENCE)

    assert list(_store(photo).iterdir()) == []


# -- the store of earlier versions -----------------------------------------


def test_the_earlier_store_is_moved_into_the_new_one(tmp_path: Path, photo: Path) -> None:
    """0.34 and earlier stored photos of unwritable folders somewhere else."""
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    old = earlier / sidecar.stored_path(photo, earlier).name
    old.write_text(json.dumps({"adjustments": {"exposure": 0.6}}), encoding="utf-8")

    assert sidecar.adopt(earlier, _store(photo)) == 1

    assert not old.exists()
    assert _read(photo, RAW_REFERENCE).adjustments.exposure == 0.6


def test_moving_the_earlier_store_keeps_what_the_new_one_has(
    tmp_path: Path, photo: Path
) -> None:
    _save(photo, RAW_REFERENCE.with_value("exposure", 0.2), RAW_REFERENCE)
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    old = earlier / _file(photo).name
    old.write_text(json.dumps({"adjustments": {"exposure": 0.6}}), encoding="utf-8")

    assert sidecar.adopt(earlier, _store(photo)) == 0

    assert old.exists()
    assert _read(photo, RAW_REFERENCE).adjustments.exposure == 0.2


def test_there_being_no_earlier_store_is_fine(tmp_path: Path) -> None:
    assert sidecar.adopt(tmp_path / "missing", tmp_path / "store") == 0
    assert not (tmp_path / "store").exists()


def test_a_store_that_cannot_be_written_to_reports_a_failure(
    photo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _deny_writes_in(_store(photo), monkeypatch)

    result = _save(photo, RAW_REFERENCE.with_value("exposure", 0.8), RAW_REFERENCE)

    assert result.failed
    assert result.path is None


# -- robustness ------------------------------------------------------------


def test_a_damaged_file_is_treated_as_absent(photo: Path, store: Path) -> None:
    _file(photo).write_text("{ this is not json", encoding="utf-8")

    assert _read(photo, RAW_REFERENCE) is None


def test_a_file_holding_something_other_than_an_object_is_ignored(
    photo: Path, store: Path
) -> None:
    _file(photo).write_text("[1, 2, 3]", encoding="utf-8")

    assert _read(photo, RAW_REFERENCE) is None


def test_no_temporary_file_is_left_behind(photo: Path) -> None:
    _save(photo, RAW_REFERENCE.with_value("exposure", 0.5), RAW_REFERENCE)

    assert list(photo.parent.glob("*.tmp")) == []


def test_a_file_with_non_ascii_in_its_path_works(tmp_path: Path) -> None:
    """Paths like D:\\Photos\\2026-10 京都 have to work (CLAUDE.md section 7)."""
    folder = tmp_path / "2026-10 京都"
    folder.mkdir()
    photo = folder / "DSC01234.ARW"
    photo.write_bytes(b"photo")

    _save(photo, RAW_REFERENCE.with_value("vibrance", 30.0), RAW_REFERENCE)

    restored = _read(photo, RAW_REFERENCE)
    assert restored is not None
    assert restored.adjustments.vibrance == 30.0


# -- which photos are edited (PLAN.md 3.1) ---------------------------------


def test_the_edited_photos_of_a_folder_are_the_ones_with_a_file(tmp_path: Path) -> None:
    photos = []
    for index in range(3):
        path = tmp_path / f"DSC0000{index}.ARW"
        path.write_bytes(b"photo")
        photos.append(path)
    _save(photos[1], RAW_REFERENCE.with_value("exposure", 0.4), RAW_REFERENCE)
    _write_old_file(photos[2], {"adjustments": {"exposure": 0.1}})

    assert sidecar.edited_photos(photos, _store(photos[0])) == {photos[1], photos[2]}


# -- the photo itself ------------------------------------------------------


def test_the_photo_is_never_written_to(photo: Path) -> None:
    """PLAN.md section 10, rule 8: never write to the user's photo."""
    before = photo.read_bytes()
    digest = hashlib.sha256(before).hexdigest()
    stat_before = photo.stat()

    _save(photo, RAW_REFERENCE.with_value("exposure", 1.0), RAW_REFERENCE)
    _save(photo, RAW_REFERENCE.with_value("contrast", 50.0), RAW_REFERENCE)
    _read(photo, RAW_REFERENCE)
    _save(photo, RAW_REFERENCE, RAW_REFERENCE)
    sidecar.delete(photo, _store(photo))

    stat_after = photo.stat()
    assert photo.read_bytes() == before
    assert hashlib.sha256(photo.read_bytes()).hexdigest() == digest
    assert stat_after.st_size == stat_before.st_size
    assert stat_after.st_mtime_ns == stat_before.st_mtime_ns


def _deny_writes_in(folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every attempt to create a file in *folder* fail.

    Windows reports a write protected card or a folder without permission as
    a ``PermissionError`` from the open itself, which is what this imitates;
    making a temporary folder genuinely unwritable needs the file system's
    own access control and would not run the same way everywhere.
    """
    real_open = Path.open

    def guarded(self: Path, mode: str = "r", *args: object, **kwargs: object):
        if self.parent == folder and any(flag in mode for flag in "wxa+"):
            raise PermissionError(13, "Access is denied")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)


def test_a_byte_order_mark_does_not_lose_the_adjustments(photo: Path) -> None:
    """A parameter file another program re-saved must still be read.

    Found while testing the packaged build: a file saved as "UTF-8 with
    BOM", which Windows editors offer, was read as damaged and silently
    ignored.
    """
    result = _save(photo, EditState(is_raw=True, exposure=0.75), RAW_REFERENCE)
    assert result.path is not None
    result.path.write_bytes(codecs.BOM_UTF8 + result.path.read_bytes())

    read_back = _read(photo, RAW_REFERENCE)

    assert read_back is not None
    assert read_back.adjustments.exposure == pytest.approx(0.75)


# -- the film filter -------------------------------------------------------


def test_the_filter_survives_a_round_trip(photo: Path) -> None:
    state = RAW_REFERENCE.with_filter("rich-negative").with_value("filter_strength", 65.0)
    _save(photo, state, RAW_REFERENCE)

    restored = _read(photo, RAW_REFERENCE)
    assert restored is not None
    assert restored.adjustments.filter == "rich-negative"
    assert restored.adjustments.filter_strength == 65.0


def test_a_photo_wearing_only_a_filter_still_gets_a_file(photo: Path) -> None:
    """A filter is a change like any other, so it is worth storing."""
    result = _save(photo, RAW_REFERENCE.with_filter("warm-negative"), RAW_REFERENCE)

    assert result.path is not None
    assert _file(photo).exists()


def test_moving_only_the_strength_leaves_no_file(photo: Path) -> None:
    """With no filter chosen the strength changes nothing, as with the midpoint."""
    result = _save(photo, RAW_REFERENCE.with_value("filter_strength", 20.0), RAW_REFERENCE)

    assert result.path is None
    assert not _file(photo).exists()


def test_a_filter_name_that_is_not_a_name_is_ignored(photo: Path) -> None:
    _save(photo, RAW_REFERENCE, RAW_REFERENCE, geometry=Geometry(quarter_turns=1))
    document = json.loads(_file(photo).read_text(encoding="utf-8"))
    document["adjustments"]["filter"] = 17
    _file(photo).write_text(json.dumps(document), encoding="utf-8")

    restored = _read(photo, RAW_REFERENCE)
    assert restored is not None
    assert restored.adjustments.filter == ""


def test_pasting_the_filter_into_a_closed_photo(photo: Path) -> None:
    """P-02 carries the filter to photos that are not open, as a name."""
    sidecar.update_adjustments(
        photo, {"filter": "cool-slide", "filter_strength": 80.0}, APPLICATION, _store(photo)
    )

    restored = _read(photo, RAW_REFERENCE)
    assert restored is not None
    assert restored.adjustments.filter == "cool-slide"
    assert restored.adjustments.filter_strength == 80.0
