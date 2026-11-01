"""Interface text lookup (PLAN.md 7.4).

All interface strings are stored in ``resources/i18n/<language>.json`` and
retrieved with :func:`tr`. No interface string may be written literally in the
code. When a key is missing from the active language the English text is used,
and when it is missing there too the key itself is returned so that the gap is
visible rather than silent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

from app.paths import resources_dir

logger = logging.getLogger(__name__)

LANGUAGE_ZH_TW: Final[str] = "zh_TW"
LANGUAGE_EN: Final[str] = "en"

#: Languages offered in the settings dialog, in display order.
AVAILABLE_LANGUAGES: Final[tuple[str, ...]] = (LANGUAGE_ZH_TW, LANGUAGE_EN)

DEFAULT_LANGUAGE: Final[str] = LANGUAGE_ZH_TW
FALLBACK_LANGUAGE: Final[str] = LANGUAGE_EN


class Translator:
    """Holds the strings of one language plus the English fallback."""

    def __init__(self, language: str = DEFAULT_LANGUAGE) -> None:
        self._language = DEFAULT_LANGUAGE
        self._strings: dict[str, str] = {}
        self._fallback: dict[str, str] = {}
        self.set_language(language)

    @property
    def language(self) -> str:
        """Return the active language code."""
        return self._language

    def set_language(self, language: str) -> None:
        """Load *language*, falling back to the default when it is unknown."""
        if language not in AVAILABLE_LANGUAGES:
            logger.warning("Unknown interface language %r, using %r", language, DEFAULT_LANGUAGE)
            language = DEFAULT_LANGUAGE
        self._language = language
        self._fallback = _load_language(FALLBACK_LANGUAGE)
        self._strings = (
            self._fallback if language == FALLBACK_LANGUAGE else _load_language(language)
        )

    def tr(self, key: str, **params: object) -> str:
        """Return the text for *key*, substituting ``{name}`` placeholders."""
        text = self._strings.get(key) or self._fallback.get(key)
        if text is None:
            logger.warning("Missing interface string for key %r", key)
            return key
        if not params:
            return text
        try:
            return text.format(**params)
        except (KeyError, IndexError, ValueError):
            logger.warning("Bad placeholders in interface string %r", key)
            return text


def _language_file(language: str) -> Path:
    return resources_dir() / "i18n" / f"{language}.json"


def _load_language(language: str) -> dict[str, str]:
    """Read one language file; an unreadable file yields no strings."""
    path = _language_file(language)
    try:
        # utf-8-sig, so that a byte order mark in a translation file does
        # not leave the interface without any text at all.
        with path.open(encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        logger.exception("Could not read interface strings from %s", path)
        return {}
    if not isinstance(data, dict):
        logger.error("Interface strings in %s are not an object", path)
        return {}
    return {str(key): str(value) for key, value in data.items()}


_translator = Translator()


def init_translator(language: str) -> None:
    """Set the language used by the module level :func:`tr`."""
    _translator.set_language(language)


def current_language() -> str:
    """Return the language code currently used by :func:`tr`."""
    return _translator.language


def tr(key: str, **params: object) -> str:
    """Return the interface text registered under *key*."""
    return _translator.tr(key, **params)
