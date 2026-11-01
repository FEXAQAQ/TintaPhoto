"""Saved sets of adjustments (PLAN.md 7.2, P-04 to P-06).

A preset is one small JSON file in the application's own folder, named after
a UUID so that two presets can share a name, with the name itself inside.
Two things about the format are worth knowing:

* **Only what was changed is recorded.** A preset made from a photo with the
  vibrance raised holds the vibrance and nothing else, so applying it leaves
  the exposure of whatever photo it lands on alone. That is what makes a
  preset something to combine rather than a complete look.
* **The portrait is left out.** Skin and whitening depend on the faces in
  one photo, not on a look that carries from photo to photo, so a preset
  never records them and never sets them, even from an older preset file
  that has them (2026-11-01, PLAN.md 7.2).
* **White balance carries its unit.** For a RAW photo the temperature is in
  kelvin and means the same thing on any RAW photo; for a JPEG it is a nudge
  from wherever the camera left it, and the two numbers have nothing to do
  with each other. A preset therefore records which it is, and the pair is
  skipped when it lands on the other kind (PLAN.md 7.2).

Nothing here knows where the presets folder is: the caller passes it in, as
everything in ``core`` does.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from core.edit_state import NUMERIC_NAMES, PORTRAIT_NAMES, EditState, as_float

logger = logging.getLogger(__name__)

#: Format version of a preset file (PLAN.md section 7).
PRESET_VERSION: Final[int] = 1

_SUFFIX: Final[str] = ".json"
_TEMP_SUFFIX: Final[str] = ".tmp"

#: What the temperature and tint of a preset mean.
WHITE_BALANCE_KELVIN: Final[str] = "kelvin"
WHITE_BALANCE_RELATIVE: Final[str] = "relative"

#: The two adjustments whose meaning depends on the kind of photo.
WHITE_BALANCE_NAMES: Final[tuple[str, ...]] = ("temperature", "tint")

#: The adjustments a preset never records or sets: the portrait sliders.
LEFT_OUT_NAMES: Final[frozenset[str]] = frozenset(PORTRAIT_NAMES)

#: A name may not be empty and may not run on forever.
NAME_MAX_LENGTH: Final[int] = 60


@dataclass(frozen=True)
class Preset:
    """One saved set of adjustments."""

    id: str
    name: str
    #: Only the adjustments that were changed, by their name in EditState.
    adjustments: dict[str, float] = field(default_factory=dict)
    #: Which unit the temperature and tint are in, or ``None`` when neither
    #: of them is part of this preset.
    white_balance: str | None = None
    #: The film look this preset puts on, or ``None`` when it does not
    #: mention one. An empty string is a preset that takes the filter off,
    #: which is a thing it can be asked to remember.
    filter: str | None = None
    created_at: str = ""
    #: Where it was read from, for renaming and deleting.
    path: Path | None = None

    @property
    def is_empty(self) -> bool:
        """Return whether this preset would change nothing."""
        return not self.adjustments and self.filter is None

    def suits(self, state: EditState) -> bool:
        """Return whether the white balance of this preset fits *state*.

        A preset with no white balance in it fits anything.
        """
        if self.white_balance is None:
            return True
        wanted = WHITE_BALANCE_KELVIN if state.is_raw else WHITE_BALANCE_RELATIVE
        return self.white_balance == wanted

    def apply_to(self, state: EditState) -> EditState:
        """Return *state* with this preset's adjustments set on it (P-05).

        Anything the preset does not mention keeps the value the photo
        already has, and the white balance is left alone when it came from
        the other kind of photo (PLAN.md 7.2).
        """
        keeps_white_balance = self.suits(state)
        changed = state
        for name, value in self.adjustments.items():
            if name in WHITE_BALANCE_NAMES and not keeps_white_balance:
                continue
            if name not in NUMERIC_NAMES or name in LEFT_OUT_NAMES:
                continue
            changed = changed.with_value(name, value)
        if self.filter is not None:
            changed = changed.with_filter(self.filter)
        return changed

    def to_dict(self) -> dict[str, Any]:
        """Return the preset as it is stored (PLAN.md 7.2)."""
        document: dict[str, Any] = {
            "version": PRESET_VERSION,
            "id": self.id,
            "name": self.name,
            "adjustments": dict(self.adjustments),
            "created_at": self.created_at,
        }
        if self.white_balance is not None:
            document["white_balance"] = self.white_balance
        if self.filter is not None:
            document["filter"] = self.filter
        return document

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: Path | None = None) -> "Preset | None":
        """Build a preset from stored data, or ``None`` if it makes no sense."""
        if not isinstance(data, dict):
            return None
        identifier = str(data.get("id") or "").strip()
        name = str(data.get("name") or "").strip()
        if not identifier or not name:
            return None
        stored = data.get("adjustments")
        stored = stored if isinstance(stored, dict) else {}
        adjustments: dict[str, float] = {}
        for key, value in stored.items():
            if key not in NUMERIC_NAMES:
                continue
            number = as_float(value)
            if number is not None:
                adjustments[key] = number
        unit = data.get("white_balance")
        if unit not in (WHITE_BALANCE_KELVIN, WHITE_BALANCE_RELATIVE):
            unit = None
        if unit is None and any(key in adjustments for key in WHITE_BALANCE_NAMES):
            # A file from a version that did not record the unit: a kelvin
            # value is far outside the relative range, which says which it is.
            unit = (
                WHITE_BALANCE_KELVIN
                if abs(adjustments.get("temperature", 0.0)) > 1000.0
                else WHITE_BALANCE_RELATIVE
            )
        chosen = data.get("filter")
        return cls(
            id=identifier,
            name=name[:NAME_MAX_LENGTH],
            adjustments=adjustments,
            white_balance=unit,
            filter=chosen if isinstance(chosen, str) else None,
            created_at=str(data.get("created_at") or ""),
            path=path,
        )


def create(name: str, state: EditState, reference: EditState) -> Preset:
    """Return a new preset holding what *state* changed from *reference* (P-04).

    Only the differences are recorded, which is what PLAN.md 7.2 asks for:
    a preset that only raises the vibrance should not also pin the exposure
    of the photo it was made from onto every photo it is applied to.
    """
    adjustments = {
        field_name: float(getattr(state, field_name))
        for field_name in NUMERIC_NAMES
        if field_name not in LEFT_OUT_NAMES
        and getattr(state, field_name) != getattr(reference, field_name)
    }
    unit: str | None = None
    if any(key in adjustments for key in WHITE_BALANCE_NAMES):
        unit = WHITE_BALANCE_KELVIN if state.is_raw else WHITE_BALANCE_RELATIVE
    return Preset(
        id=str(uuid.uuid4()),
        name=clean_name(name),
        adjustments=adjustments,
        white_balance=unit,
        filter=state.filter if state.filter != reference.filter else None,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )


def clean_name(name: str) -> str:
    """Return *name* with the spaces trimmed and a sensible length."""
    return " ".join(str(name).split())[:NAME_MAX_LENGTH]


# -------------------------------------------------------------------------
# The folder
# -------------------------------------------------------------------------


def preset_path(directory: Path, identifier: str) -> Path:
    """Return the file a preset is stored in."""
    return directory / f"{identifier}{_SUFFIX}"


def load_all(directory: Path) -> list[Preset]:
    """Return every preset in *directory*, in the order they are shown.

    A file that cannot be read is logged and left out rather than stopping
    the rest from loading.
    """
    try:
        entries = sorted(directory.glob(f"*{_SUFFIX}"))
    except OSError:
        logger.info("The presets folder %s could not be listed", directory)
        return []

    presets: list[Preset] = []
    for path in entries:
        try:
            # utf-8-sig, so that a byte order mark left by whatever edited
            # the file does not lose the preset.
            with path.open(encoding="utf-8-sig") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            logger.exception("Could not read the preset %s", path)
            continue
        preset = Preset.from_dict(data, path)
        if preset is None:
            logger.error("The preset %s does not hold a preset", path)
            continue
        presets.append(preset)
    return sorted(presets, key=lambda item: (item.name.casefold(), item.id))


def save(preset: Preset, directory: Path) -> Preset | None:
    """Store *preset*, returning it with its path, or ``None`` if it failed."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("The presets folder %s cannot be used", directory)
        return None

    target = preset_path(directory, preset.id)
    temporary = target.with_name(target.name + _TEMP_SUFFIX)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(preset.to_dict(), handle, ensure_ascii=False, indent=2)
        os.replace(temporary, target)
    except OSError:
        logger.exception("Could not write the preset %s", target)
        _remove(temporary)
        return None
    return Preset(
        id=preset.id,
        name=preset.name,
        adjustments=dict(preset.adjustments),
        white_balance=preset.white_balance,
        created_at=preset.created_at,
        path=target,
    )


def rename(preset: Preset, name: str, directory: Path) -> Preset | None:
    """Give *preset* a new name (P-06); the file keeps its UUID."""
    cleaned = clean_name(name)
    if not cleaned:
        return None
    return save(
        Preset(
            id=preset.id,
            name=cleaned,
            adjustments=dict(preset.adjustments),
            white_balance=preset.white_balance,
            created_at=preset.created_at,
        ),
        directory,
    )


def delete(preset: Preset, directory: Path) -> bool:
    """Remove *preset* (P-06); return whether there was a file to remove."""
    return _remove(preset.path or preset_path(directory, preset.id))


def _remove(path: Path) -> bool:
    """Delete *path* if it is there; return whether it was."""
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError:
        logger.exception("Could not delete %s", path)
        return False
    return True


__all__ = [
    "NAME_MAX_LENGTH",
    "PRESET_VERSION",
    "WHITE_BALANCE_KELVIN",
    "WHITE_BALANCE_NAMES",
    "WHITE_BALANCE_RELATIVE",
    "Preset",
    "clean_name",
    "create",
    "delete",
    "load_all",
    "preset_path",
    "rename",
    "save",
]
