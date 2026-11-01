"""Tests for saved sets of adjustments (PLAN.md 7.2, P-04 to P-06)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import presets
from core.edit_state import EditState
from core.presets import Preset

RAW_REST = EditState.for_photo(True, temperature=5600.0, tint=10.0)
JPEG_REST = EditState.for_photo(False)


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    return tmp_path / "presets"


# -- what a preset records -------------------------------------------------


def test_only_what_was_changed_is_recorded() -> None:
    """PLAN.md 7.2: 預設集只記錄有變動的參數."""
    state = RAW_REST.with_value("vibrance", 25.0).with_value("contrast", 10.0)

    preset = presets.create("Punchy", state, RAW_REST)

    assert set(preset.adjustments) == {"vibrance", "contrast"}
    assert preset.adjustments["vibrance"] == 25.0
    assert preset.white_balance is None


def test_a_preset_of_nothing_knows_it_is_empty() -> None:
    assert presets.create("Nothing", RAW_REST, RAW_REST).is_empty


def test_the_white_balance_unit_is_recorded_with_it() -> None:
    from_raw = presets.create("Warm", RAW_REST.with_value("temperature", 7000.0), RAW_REST)
    from_jpeg = presets.create("Warm", JPEG_REST.with_value("temperature", 30.0), JPEG_REST)

    assert from_raw.white_balance == presets.WHITE_BALANCE_KELVIN
    assert from_jpeg.white_balance == presets.WHITE_BALANCE_RELATIVE


def test_a_name_is_tidied_up() -> None:
    assert presets.create("  warm   tone  ", RAW_REST, RAW_REST).name == "warm tone"
    assert len(presets.create("x" * 200, RAW_REST, RAW_REST).name) == presets.NAME_MAX_LENGTH


def test_the_portrait_is_never_recorded() -> None:
    """2026-11-01: 預設集不保存人像（美膚、美白）."""
    state = RAW_REST.with_value("skin", 40.0).with_value("whiten", 20.0).with_value("grain", 30.0)

    preset = presets.create("Portrait", state, RAW_REST)

    assert set(preset.adjustments) == {"grain"}


def test_only_portrait_changes_make_an_empty_preset() -> None:
    state = JPEG_REST.with_value("skin", 40.0)
    assert presets.create("Skin", state, JPEG_REST).is_empty


# -- applying one ----------------------------------------------------------


def test_what_the_preset_does_not_mention_is_left_alone() -> None:
    """PLAN.md 7.2: 沒記錄的參數維持照片原本的值."""
    preset = presets.create("Punchy", RAW_REST.with_value("vibrance", 25.0), RAW_REST)
    photo = RAW_REST.with_value("exposure", 0.8).with_value("contrast", -20.0)

    applied = preset.apply_to(photo)

    assert applied.vibrance == 25.0
    assert applied.exposure == 0.8
    assert applied.contrast == -20.0


def test_an_older_preset_with_the_portrait_in_it_leaves_it_alone() -> None:
    """A preset file saved before the portrait was left out still sets the rest."""
    preset = Preset(
        id="old", name="Old", adjustments={"skin": 40.0, "whiten": 20.0, "vibrance": 25.0}
    )
    photo = JPEG_REST.with_value("skin", 10.0)

    applied = preset.apply_to(photo)

    assert applied.skin == 10.0
    assert applied.whiten == JPEG_REST.whiten
    assert applied.vibrance == 25.0


def test_a_kelvin_preset_skips_the_white_balance_of_a_jpeg() -> None:
    """PLAN.md 7.2: 從 RAW 建立的 K 值預設集套用到 JPG 時，略過色溫與色調."""
    preset = presets.create(
        "Warm",
        RAW_REST.with_value("temperature", 7000.0).with_value("vibrance", 20.0),
        RAW_REST,
    )
    jpeg = JPEG_REST.with_value("temperature", 15.0)

    applied = preset.apply_to(jpeg)

    assert applied.temperature == 15.0, "the JPEG's own white balance was overwritten"
    assert applied.vibrance == 20.0
    assert not preset.suits(jpeg)


def test_a_relative_preset_skips_the_white_balance_of_a_raw_photo() -> None:
    preset = presets.create(
        "Cool", JPEG_REST.with_value("tint", -20.0).with_value("saturation", 10.0), JPEG_REST
    )

    applied = preset.apply_to(RAW_REST)

    assert applied.tint == RAW_REST.tint
    assert applied.saturation == 10.0


def test_a_preset_without_white_balance_fits_both_kinds() -> None:
    preset = presets.create("Punchy", RAW_REST.with_value("vibrance", 25.0), RAW_REST)

    assert preset.suits(RAW_REST)
    assert preset.suits(JPEG_REST)
    assert preset.apply_to(JPEG_REST).vibrance == 25.0


def test_applying_keeps_the_value_inside_the_slider() -> None:
    preset = Preset(id="x", name="Odd", adjustments={"exposure": 99.0})

    assert preset.apply_to(RAW_REST).exposure == 5.0


def test_an_unknown_adjustment_in_a_preset_is_ignored() -> None:
    preset = Preset(id="x", name="Future", adjustments={"clarity": 40.0})

    assert preset.apply_to(RAW_REST) == RAW_REST


# -- the folder ------------------------------------------------------------


def test_a_preset_survives_being_saved_and_read_back(folder: Path) -> None:
    """Stage 5 acceptance: 儲存預設集後重啟程式，預設集還在，套用結果相同."""
    state = RAW_REST.with_value("temperature", 7000.0).with_value("vibrance", 25.0)
    saved = presets.save(presets.create("Golden hour", state, RAW_REST), folder)
    assert saved is not None

    restored = presets.load_all(folder)

    assert len(restored) == 1
    assert restored[0].name == "Golden hour"
    assert restored[0].id == saved.id
    assert restored[0].adjustments == saved.adjustments
    assert restored[0].apply_to(RAW_REST) == saved.apply_to(RAW_REST)


def test_the_file_holds_the_documented_shape(folder: Path) -> None:
    saved = presets.save(
        presets.create("Warm", RAW_REST.with_value("temperature", 7000.0), RAW_REST), folder
    )
    assert saved is not None and saved.path is not None

    with saved.path.open(encoding="utf-8") as handle:
        document = json.load(handle)

    assert document["version"] == presets.PRESET_VERSION
    assert document["id"] == saved.id
    assert saved.path.stem == saved.id, "the file is named after the UUID"
    assert document["name"] == "Warm"
    assert document["adjustments"] == {"temperature": 7000.0}
    assert document["white_balance"] == presets.WHITE_BALANCE_KELVIN
    assert "created_at" in document


def test_two_presets_may_share_a_name(folder: Path) -> None:
    first = presets.save(presets.create("Warm", RAW_REST.with_value("exposure", 0.5), RAW_REST), folder)
    second = presets.save(presets.create("Warm", RAW_REST.with_value("exposure", 1.0), RAW_REST), folder)

    assert first is not None and second is not None and first.id != second.id
    assert len(presets.load_all(folder)) == 2


def test_presets_come_back_in_name_order(folder: Path) -> None:
    for name in ("zebra", "Banana", "mango"):
        presets.save(presets.create(name, RAW_REST.with_value("exposure", 0.5), RAW_REST), folder)

    assert [item.name for item in presets.load_all(folder)] == ["Banana", "mango", "zebra"]


def test_renaming_keeps_the_settings_and_the_file(folder: Path) -> None:
    """P-06: rename."""
    saved = presets.save(presets.create("Draft", RAW_REST.with_value("vibrance", 30.0), RAW_REST), folder)
    assert saved is not None

    renamed = presets.rename(saved, "Final", folder)

    assert renamed is not None
    assert renamed.id == saved.id
    assert renamed.adjustments == saved.adjustments
    assert [item.name for item in presets.load_all(folder)] == ["Final"]


def test_renaming_to_nothing_is_refused(folder: Path) -> None:
    saved = presets.save(presets.create("Draft", RAW_REST.with_value("vibrance", 30.0), RAW_REST), folder)
    assert saved is not None

    assert presets.rename(saved, "   ", folder) is None
    assert [item.name for item in presets.load_all(folder)] == ["Draft"]


def test_deleting_removes_it(folder: Path) -> None:
    """P-06: delete."""
    saved = presets.save(presets.create("Draft", RAW_REST.with_value("vibrance", 30.0), RAW_REST), folder)
    assert saved is not None

    assert presets.delete(saved, folder)
    assert presets.load_all(folder) == []
    assert not presets.delete(saved, folder)


def test_a_missing_folder_simply_has_no_presets(tmp_path: Path) -> None:
    assert presets.load_all(tmp_path / "not there") == []


def test_a_damaged_preset_is_left_out_without_losing_the_others(folder: Path) -> None:
    presets.save(presets.create("Good", RAW_REST.with_value("vibrance", 30.0), RAW_REST), folder)
    (folder / "broken.json").write_text("{ not json", encoding="utf-8")
    (folder / "empty.json").write_text("{}", encoding="utf-8")

    loaded = presets.load_all(folder)

    assert [item.name for item in loaded] == ["Good"]


def test_no_temporary_file_is_left_behind(folder: Path) -> None:
    presets.save(presets.create("Good", RAW_REST.with_value("vibrance", 30.0), RAW_REST), folder)

    assert list(folder.glob("*.tmp")) == []


def test_a_preset_file_without_a_unit_is_read_sensibly(folder: Path) -> None:
    """A file from a version that did not record which unit it used."""
    folder.mkdir(parents=True)
    (folder / "old.json").write_text(
        json.dumps({"id": "old", "name": "Kelvin", "adjustments": {"temperature": 7200.0}}),
        encoding="utf-8",
    )
    (folder / "older.json").write_text(
        json.dumps({"id": "older", "name": "Nudge", "adjustments": {"temperature": 20.0}}),
        encoding="utf-8",
    )

    by_name = {item.name: item for item in presets.load_all(folder)}

    assert by_name["Kelvin"].white_balance == presets.WHITE_BALANCE_KELVIN
    assert by_name["Nudge"].white_balance == presets.WHITE_BALANCE_RELATIVE


def test_a_byte_order_mark_does_not_lose_a_preset(tmp_path: Path) -> None:
    """A preset file another program re-saved must still be read."""
    import codecs

    folder = tmp_path / "presets"
    saved = presets.save(
        presets.create("Warm", RAW_REST.with_value("exposure", 0.5), RAW_REST), folder
    )
    assert saved is not None and saved.path is not None
    saved.path.write_bytes(codecs.BOM_UTF8 + saved.path.read_bytes())

    assert [preset.name for preset in presets.load_all(folder)] == ["Warm"]


# -- the film filter -------------------------------------------------------


def test_a_preset_records_the_filter_it_was_saved_with() -> None:
    state = RAW_REST.with_filter("warm-negative").with_value("filter_strength", 70.0)

    preset = presets.create("Film", state, RAW_REST)

    assert preset.filter == "warm-negative"
    assert preset.adjustments["filter_strength"] == 70.0


def test_a_preset_that_changed_no_filter_does_not_mention_one() -> None:
    preset = presets.create("Punchy", RAW_REST.with_value("vibrance", 20.0), RAW_REST)

    assert preset.filter is None


def test_a_preset_without_a_filter_leaves_the_one_on_the_photo() -> None:
    """PLAN.md 7.2: what a preset does not mention, the photo keeps."""
    preset = presets.create("Punchy", RAW_REST.with_value("vibrance", 20.0), RAW_REST)

    changed = preset.apply_to(RAW_REST.with_filter("cool-slide"))

    assert changed.filter == "cool-slide"
    assert changed.vibrance == 20.0


def test_a_preset_can_remember_taking_the_filter_off() -> None:
    wearing = RAW_REST.with_filter("cool-slide")

    preset = presets.create("Plain", wearing.with_filter(""), wearing)

    assert preset.filter == ""
    assert preset.apply_to(RAW_REST.with_filter("warm-negative")).filter == ""


def test_the_filter_survives_a_preset_file(folder: Path) -> None:
    state = RAW_REST.with_filter("rich-negative").with_value("filter_strength", 40.0)
    saved = presets.save(presets.create("Film", state, RAW_REST), folder)
    assert saved is not None

    read = presets.load_all(folder)

    assert [preset.filter for preset in read] == ["rich-negative"]
    assert read[0].adjustments["filter_strength"] == 40.0


def test_a_preset_naming_only_a_filter_is_not_empty() -> None:
    preset = presets.create("Film", RAW_REST.with_filter("warm-negative"), RAW_REST)

    assert not preset.is_empty
