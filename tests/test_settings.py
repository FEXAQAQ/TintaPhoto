"""Tests for reading and writing the settings file (PLAN.md 7.3)."""

from __future__ import annotations

import json
from pathlib import Path

from app.i18n import DEFAULT_LANGUAGE, LANGUAGE_EN
from app.settings import (
    HISTOGRAM_LUMA,
    HISTOGRAM_RGB,
    MAX_RECENT_FOLDERS,
    SETTINGS_VERSION,
    STARTUP_LAST_FOLDER,
    STARTUP_WELCOME,
    Settings,
    WindowState,
)


def test_defaults_are_used_when_the_file_is_missing(tmp_path: Path) -> None:
    settings = Settings.load(tmp_path / "settings.json")
    assert settings.language == DEFAULT_LANGUAGE
    assert settings.startup_action == STARTUP_WELCOME
    assert settings.window.width is None
    assert settings.recent_folders == []


def test_save_then_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    settings = Settings(
        language=LANGUAGE_EN,
        startup_action=STARTUP_LAST_FOLDER,
        window=WindowState(x=10, y=20, width=1600, height=1000, maximized=True),
        inspector_collapsed=True,
        filmstrip_collapsed=True,
        last_folder="D:\\Photos\\2026-10 \u4eac\u90fd",
    )
    assert settings.save(path)

    loaded = Settings.load(path)
    assert loaded.language == LANGUAGE_EN
    assert loaded.startup_action == STARTUP_LAST_FOLDER
    assert loaded.window == WindowState(x=10, y=20, width=1600, height=1000, maximized=True)
    assert loaded.inspector_collapsed
    assert loaded.filmstrip_collapsed
    assert loaded.last_folder == "D:\\Photos\\2026-10 \u4eac\u90fd"


def test_saved_file_is_utf8_json_with_a_version(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    Settings(last_folder="D:\\\u7167\u7247").save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == SETTINGS_VERSION
    assert data["last_folder"] == "D:\\\u7167\u7247"


def test_no_temporary_file_is_left_behind(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    Settings().save(path)
    assert [item.name for item in tmp_path.iterdir()] == ["settings.json"]


def test_damaged_file_falls_back_to_defaults(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert Settings.load(path).language == DEFAULT_LANGUAGE


def test_non_object_file_falls_back_to_defaults(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert Settings.load(path).language == DEFAULT_LANGUAGE


def test_unknown_language_falls_back_to_the_default(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"language": "klingon"}), encoding="utf-8")
    assert Settings.load(path).language == DEFAULT_LANGUAGE


def test_bad_geometry_values_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"window": {"x": "left", "y": None, "width": [], "height": "tall"}}),
        encoding="utf-8",
    )
    window = Settings.load(path).window
    assert window == WindowState()


def test_unknown_fields_survive_a_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"language": LANGUAGE_EN, "future_option": {"keep": True}}),
        encoding="utf-8",
    )
    Settings.load(path).save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["future_option"] == {"keep": True}


def test_remember_folder_moves_it_to_the_front_without_duplicates() -> None:
    settings = Settings()
    settings.remember_folder("A")
    settings.remember_folder("B")
    settings.remember_folder("A")
    assert settings.recent_folders == ["A", "B"]
    assert settings.last_folder == "A"


def test_recent_folders_are_capped() -> None:
    settings = Settings()
    for index in range(MAX_RECENT_FOLDERS + 5):
        settings.remember_folder(f"folder-{index}")
    assert len(settings.recent_folders) == MAX_RECENT_FOLDERS


def test_a_recent_folder_remembers_its_photo_count_and_day() -> None:
    from datetime import date

    from app.settings import RecentFolder

    settings = Settings()
    settings.remember_folder("A", 148, today=date(2026, 10, 30))

    assert settings.recent_details["A"] == RecentFolder(count=148, opened="2026-10-30")


def test_forgetting_a_folder_takes_it_and_its_details_off_the_list() -> None:
    settings = Settings()
    settings.remember_folder("A", 3)
    settings.remember_folder("B", 5)

    settings.forget_folder("A")

    assert settings.recent_folders == ["B"]
    assert "A" not in settings.recent_details
    assert settings.recent_details["B"].count == 5
    assert settings.last_folder == "B"


def test_forgetting_the_last_folder_stops_it_opening_at_start_up() -> None:
    settings = Settings()
    settings.remember_folder("A")
    settings.remember_folder("B")

    settings.forget_folder("B")

    assert settings.recent_folders == ["A"]
    assert settings.last_folder is None


def test_forgetting_a_folder_not_on_the_list_changes_nothing() -> None:
    settings = Settings()
    settings.remember_folder("A", 3)

    settings.forget_folder("Z")

    assert settings.recent_folders == ["A"]
    assert settings.last_folder == "A"


def test_clearing_the_recent_list_empties_it(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    settings = Settings()
    settings.remember_folder("A", 3)
    settings.remember_folder("B", 5)

    settings.clear_recent()
    settings.save(path)
    loaded = Settings.load(path)

    assert loaded.recent_folders == []
    assert loaded.recent_details == {}
    assert loaded.last_folder is None


def test_the_details_survive_a_save_and_a_load(tmp_path: Path) -> None:
    from datetime import date

    path = tmp_path / "settings.json"
    settings = Settings()
    settings.remember_folder("D:/Photos/京都", 62, today=date(2026, 10, 12))
    settings.save(path)

    loaded = Settings.load(path)

    assert loaded.recent_details["D:/Photos/京都"].count == 62
    assert loaded.recent_details["D:/Photos/京都"].day == date(2026, 10, 12)


def test_a_folder_from_an_older_version_simply_has_no_details(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"recent_folders": ["A"]}), encoding="utf-8")

    loaded = Settings.load(path)

    assert loaded.recent_folders == ["A"]
    assert loaded.recent_details == {}


def test_nonsense_details_are_dropped_rather_than_trusted(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "recent_folders": ["A", "B", "C"],
                "recent_details": {
                    "A": {"count": -3, "opened": "2026-10-30"},
                    "B": {"count": 5, "opened": "yesterday"},
                    "C": "lots",
                    "gone": {"count": 1, "opened": "2026-10-30"},
                },
            }
        ),
        encoding="utf-8",
    )

    assert Settings.load(path).recent_details == {}


def test_details_go_when_their_folder_falls_off_the_list() -> None:
    settings = Settings()
    settings.remember_folder("first", 1)
    for index in range(MAX_RECENT_FOLDERS):
        settings.remember_folder(f"folder-{index}", 2)

    assert "first" not in settings.recent_details
    assert len(settings.recent_details) == MAX_RECENT_FOLDERS


# -- a file another program has saved (found while packaging) ------------


def test_a_byte_order_mark_does_not_throw_the_settings_away(tmp_path: Path) -> None:
    """Windows editors offer "UTF-8 with BOM"; the file is still ours."""
    settings = Settings(language="en", preview_background="mid")
    path = tmp_path / "settings.json"
    assert settings.save(path)
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())

    read_back = Settings.load(path)

    assert read_back.language == "en"
    assert read_back.preview_background == "mid"
    assert read_back == settings


def test_the_histogram_mode_survives_a_save_and_a_load(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    assert Settings(histogram_mode=HISTOGRAM_LUMA).save(path)
    assert Settings.load(path).histogram_mode == HISTOGRAM_LUMA


def test_an_unknown_histogram_mode_falls_back_to_three_channels() -> None:
    assert Settings.from_dict({"histogram_mode": "waveform"}).histogram_mode == HISTOGRAM_RGB
