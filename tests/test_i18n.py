"""Tests for the interface text lookup (PLAN.md 7.4)."""

from __future__ import annotations

import json

from app import i18n
from app.i18n import AVAILABLE_LANGUAGES, LANGUAGE_EN, LANGUAGE_ZH_TW, Translator, tr
from app.paths import resources_dir


def _load(language: str) -> dict[str, str]:
    path = resources_dir() / "i18n" / f"{language}.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_every_language_file_exists_and_is_an_object() -> None:
    for language in AVAILABLE_LANGUAGES:
        assert isinstance(_load(language), dict)


def test_language_files_have_identical_keys() -> None:
    chinese = set(_load(LANGUAGE_ZH_TW))
    english = set(_load(LANGUAGE_EN))
    assert chinese == english, {
        "missing_in_en": sorted(chinese - english),
        "missing_in_zh_TW": sorted(english - chinese),
    }


def test_no_empty_strings() -> None:
    for language in AVAILABLE_LANGUAGES:
        for key, value in _load(language).items():
            assert value.strip(), f"{language}: {key} is empty"


def test_translator_returns_the_language_text() -> None:
    translator = Translator(LANGUAGE_ZH_TW)
    assert translator.tr("panel.tab.adjust") == _load(LANGUAGE_ZH_TW)["panel.tab.adjust"]
    translator.set_language(LANGUAGE_EN)
    assert translator.tr("panel.tab.adjust") == _load(LANGUAGE_EN)["panel.tab.adjust"]


def test_unknown_language_falls_back_to_the_default() -> None:
    translator = Translator("klingon")
    assert translator.language == i18n.DEFAULT_LANGUAGE


def test_missing_key_returns_the_key_itself() -> None:
    translator = Translator(LANGUAGE_ZH_TW)
    assert translator.tr("no.such.key") == "no.such.key"


def test_placeholders_are_substituted() -> None:
    translator = Translator(LANGUAGE_EN)
    text = translator.tr("tooltip.with_shortcut", label="Crop", shortcut="R")
    assert "Crop" in text and "R" in text


def test_bad_placeholders_do_not_raise() -> None:
    translator = Translator(LANGUAGE_EN)
    assert translator.tr("tooltip.with_shortcut")


def test_module_level_tr_uses_the_initialised_language() -> None:
    i18n.init_translator(LANGUAGE_EN)
    try:
        assert tr("toolbar.info") == _load(LANGUAGE_EN)["toolbar.info"]
        assert i18n.current_language() == LANGUAGE_EN
    finally:
        i18n.init_translator(i18n.DEFAULT_LANGUAGE)


def test_a_byte_order_mark_in_a_language_file_is_read_anyway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A translation saved as "UTF-8 with BOM" must not blank the interface."""
    import codecs

    import app.i18n as module

    folder = tmp_path / "i18n"
    folder.mkdir()
    (folder / "en.json").write_bytes(
        codecs.BOM_UTF8 + b'{"app.name": "TintaPhoto"}'
    )
    monkeypatch.setattr(module, "resources_dir", lambda: tmp_path)

    assert module.Translator("en").tr("app.name") == "TintaPhoto"
