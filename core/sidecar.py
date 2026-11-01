"""Reading and writing the parameter file beside a photo (PLAN.md 7.1).

Editing is non-destructive: the photo itself is never touched, and everything
the user changes lives in a small JSON file named after it. The name keeps the
original extension, ``DSC01234.ARW.json``, because a camera shooting RAW+JPEG
produces two photos with the same stem that must not share one file (F-05).

Three things here are worth knowing about:

* A write goes to a temporary file and is then renamed, so an interrupted
  write cannot leave a half finished parameter file behind.
* When the folder cannot be written to - a write protected card, a read only
  folder, a network drive without permission - the file is stored in an
  alternative place instead, named after a hash of the photo's full path.
  Reading looks beside the photo first and only then there.
* When nothing is adjusted any more, the file is deleted rather than left
  behind holding nothing but defaults.

Nothing in this module ever opens a photo for writing. The only thing it asks
the photo for is its size and modification time, which go into the file so
that a parameter file can be recognised as belonging to another version of
the photo later on.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from core.decoder import is_raw
from core.edit_state import Edit, EditState
from core.geometry import Geometry
from core.lens import LensSettings

logger = logging.getLogger(__name__)

#: Format version of the parameter file (PLAN.md section 7).
SIDECAR_VERSION: Final[int] = 1

#: Appended to the photo's whole file name, extension included.
SIDECAR_SUFFIX: Final[str] = ".json"

#: Written while saving and renamed over the real file when complete.
_TEMP_SUFFIX: Final[str] = ".tmp"

@dataclass(frozen=True)
class Application:
    """Which program wrote a parameter file, for the ``app`` block.

    Passed in rather than read from the application layer, because nothing in
    ``core`` may know about the layers above it (PLAN.md section 5).
    """

    name: str
    version: str


@dataclass(frozen=True)
class Sidecar:
    """What one parameter file holds.

    ``extra`` is carried through untouched: a file written by a newer version
    of the program may hold fields this one has never heard of, and saving a
    photo must not throw away what it does not understand (PLAN.md section
    7.1, "未知欄位保留不刪除").
    """

    edit: Edit
    extra: dict[str, Any] = field(default_factory=dict)
    #: Where it was read from, or ``None`` for one that has never been saved.
    path: Path | None = None

    @property
    def adjustments(self) -> EditState:
        """Return the stored adjustments."""
        return self.edit.adjustments

    @property
    def geometry(self) -> Geometry:
        """Return the stored geometry."""
        return self.edit.geometry


@dataclass(frozen=True)
class SaveResult:
    """What became of a save.

    ``path`` is where the file ended up, or ``None`` when there is no longer a
    file at all. ``fallback`` says that the folder of the photo could not be
    written to and the alternative store was used, which is the one thing the
    user has to be told about.
    """

    path: Path | None = None
    fallback: bool = False
    deleted: bool = False
    failed: bool = False


# -------------------------------------------------------------------------
# Where a parameter file lives
# -------------------------------------------------------------------------


def sidecar_path(photo: Path) -> Path:
    """Return the parameter file that belongs beside *photo*.

    The whole file name is kept, extension included, so that
    ``DSC01234.ARW`` and ``DSC01234.JPG`` do not overwrite each other (F-05).
    """
    return photo.with_name(photo.name + SIDECAR_SUFFIX)


def fallback_key(photo: Path) -> str:
    """Return the file name *photo* is stored under in the alternative store.

    A hash of the full path, so that two photos of the same name in different
    folders stay apart. The path is resolved and case folded first, because on
    Windows the same photo can be reached by paths that differ only in case.
    """
    try:
        resolved = str(photo.resolve())
    except OSError:
        resolved = str(photo.absolute())
    identity = os.path.normcase(resolved)
    return hashlib.sha256(identity.encode("utf-8", "replace")).hexdigest()


def fallback_path(photo: Path, fallback_dir: Path) -> Path:
    """Return where *photo*'s parameters go when its folder is unwritable."""
    return fallback_dir / f"{fallback_key(photo)}{SIDECAR_SUFFIX}"


def existing_path(photo: Path, fallback_dir: Path | None = None) -> Path | None:
    """Return the parameter file of *photo*, beside it or in the store.

    Beside the photo wins: a folder that was unwritable once may be writable
    now, and the file the user can see and copy is the one that counts.
    """
    beside = sidecar_path(photo)
    if beside.is_file():
        return beside
    if fallback_dir is not None:
        stored = fallback_path(photo, fallback_dir)
        if stored.is_file():
            return stored
    return None


def is_edited(photo: Path, fallback_dir: Path | None = None) -> bool:
    """Return whether *photo* has been adjusted (for the filmstrip mark).

    Only the presence of the file is checked, never its contents: a parameter
    file exists exactly when something is not at its default, because saving
    an untouched photo deletes it. Opening a folder of three hundred photos
    therefore costs three hundred cheap questions to the file system rather
    than three hundred JSON files to parse.
    """
    return existing_path(photo, fallback_dir) is not None


def edited_photos(photos: list[Path], fallback_dir: Path | None = None) -> set[Path]:
    """Return which of *photos* have a parameter file."""
    return {photo for photo in photos if is_edited(photo, fallback_dir)}


# -------------------------------------------------------------------------
# Reading
# -------------------------------------------------------------------------


def read(
    photo: Path, reference: EditState, fallback_dir: Path | None = None
) -> Sidecar | None:
    """Return the stored parameters of *photo*, or ``None`` if there are none.

    *reference* is the state the photo opens in - for a RAW photo that is the
    white balance the camera recorded - and fills in any field the file does
    not have (PLAN.md section 7.1, "欄位缺漏用預設值補上"). A file that cannot
    be read or makes no sense is logged and treated as absent, so that one bad
    parameter file cannot stop a photo from opening.
    """
    found = read_document(photo, fallback_dir)
    if found is None:
        return None
    path, data = found
    return _from_dict(data, reference, path)


def read_document(
    photo: Path, fallback_dir: Path | None = None
) -> tuple[Path, dict[str, Any]] | None:
    """Return the parameter file of *photo* as it is written, and where it is.

    Used where the file has to be changed rather than understood, which is
    what pasting onto a photo that is not open does.
    """
    path = existing_path(photo, fallback_dir)
    if path is None:
        return None
    try:
        # Read as utf-8-sig: the file is written without a byte order mark,
        # but another program that has edited it may have added one, and a
        # mark must not cost the user their adjustments.
        with path.open(encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        logger.exception("Could not read the parameter file %s", path)
        return None
    if not isinstance(data, dict):
        logger.error("The parameter file %s does not hold an object", path)
        return None
    return path, data


def _from_dict(data: dict[str, Any], reference: EditState, path: Path) -> Sidecar:
    """Build a sidecar from stored data, keeping anything unrecognised."""
    adjustments = data.get("adjustments")
    adjustments = adjustments if isinstance(adjustments, dict) else {}
    return Sidecar(
        edit=Edit(
            adjustments=EditState.from_dict(adjustments, reference),
            geometry=Geometry.from_dict(data.get("geometry")),
        ),
        extra={key: value for key, value in data.items() if key not in _KNOWN_KEYS},
        path=path,
    )


#: Top level fields this version writes itself; everything else is preserved.
_KNOWN_KEYS: Final[frozenset[str]] = frozenset(
    {"version", "source", "adjustments", "geometry", "app", "updated_at"}
)


# -------------------------------------------------------------------------
# Writing
# -------------------------------------------------------------------------


def save(
    photo: Path,
    edit: Edit,
    reference: Edit,
    application: Application,
    fallback_dir: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> SaveResult:
    """Store the edit of *photo*, or delete the file if there is nothing in it.

    Writing goes beside the photo, and only to the alternative store when that
    fails. A photo whose adjustments are all back at *reference* and which has
    no crop has nothing worth storing, so its parameter file is removed
    instead of being left holding defaults (PLAN.md section 7.1).
    """
    if edit.is_untouched(reference):
        removed = delete(photo, fallback_dir)
        return SaveResult(deleted=removed)

    document = _to_dict(photo, edit, application, extra or {})
    beside = sidecar_path(photo)
    if _write_atomically(beside, document):
        if fallback_dir is not None:
            # A folder that has become writable again takes over, so that the
            # photo and its parameters travel together from now on.
            _remove(fallback_path(photo, fallback_dir))
        return SaveResult(path=beside)

    if fallback_dir is None:
        logger.error("The parameters of %s could not be stored", photo)
        return SaveResult(failed=True)

    stored = fallback_path(photo, fallback_dir)
    try:
        fallback_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("The alternative parameter store %s cannot be used", fallback_dir)
        return SaveResult(failed=True)
    if _write_atomically(stored, document):
        return SaveResult(path=stored, fallback=True)
    logger.error("The parameters of %s could not be stored", photo)
    return SaveResult(failed=True)


def update_adjustments(
    photo: Path,
    values: dict[str, float | str],
    application: Application,
    fallback_dir: Path | None = None,
) -> SaveResult:
    """Set some of the adjustments of *photo* in its parameter file (P-02).

    For a photo that is open the controller has its whole state and writes
    the file outright; this is for the others, where only the adjustments
    being pasted are known. Everything else in the file - the crop, the
    adjustments not being pasted, the fields a later version wrote - is left
    exactly as it was. An adjustment that is left out here keeps whatever the
    photo had, which for a white balance that cannot be carried across is
    what has to happen (PLAN.md 7.2).

    One of the values is a name rather than a number - the film filter - so
    what is handed in is written as it comes rather than read as a number.
    """
    if not values:
        return SaveResult()

    found = read_document(photo, fallback_dir)
    document: dict[str, Any] = dict(found[1]) if found is not None else {}
    stored = document.get("adjustments")
    adjustments = dict(stored) if isinstance(stored, dict) else {}
    adjustments.update(
        {
            name: value if isinstance(value, str) else float(value)
            for name, value in values.items()
        }
    )

    geometry = document.get("geometry")
    if not isinstance(geometry, dict):
        # A photo with no parameter file yet still has the lens correction it
        # opens with, and writing the file must not switch that off; one with
        # an older file keeps what that file says (PLAN.md 7.1).
        geometry = Geometry(lens=LensSettings.for_photo(is_raw(photo))).to_dict()
    document.update(
        {
            "version": SIDECAR_VERSION,
            "source": _source(photo),
            "adjustments": adjustments,
            "geometry": geometry,
            "app": {"name": application.name, "version": application.version},
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )

    beside = sidecar_path(photo)
    if _write_atomically(beside, document):
        if fallback_dir is not None:
            _remove(fallback_path(photo, fallback_dir))
        return SaveResult(path=beside)
    if fallback_dir is None:
        logger.error("The parameters of %s could not be stored", photo)
        return SaveResult(failed=True)
    stored_path = fallback_path(photo, fallback_dir)
    try:
        fallback_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("The alternative parameter store %s cannot be used", fallback_dir)
        return SaveResult(failed=True)
    if _write_atomically(stored_path, document):
        return SaveResult(path=stored_path, fallback=True)
    return SaveResult(failed=True)


def delete(photo: Path, fallback_dir: Path | None = None) -> bool:
    """Remove the parameter file of *photo* from both places.

    Returns whether anything was there to remove. Both are cleared, or an old
    copy in the alternative store would come back the next time the photo is
    opened.
    """
    removed = _remove(sidecar_path(photo))
    if fallback_dir is not None:
        removed = _remove(fallback_path(photo, fallback_dir)) or removed
    return removed


def _to_dict(
    photo: Path,
    edit: Edit,
    application: Application,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Build the document of PLAN.md section 7.1."""
    document: dict[str, Any] = dict(extra)
    document.update(
        {
            "version": SIDECAR_VERSION,
            "source": _source(photo),
            "adjustments": edit.adjustments.to_dict(),
            "geometry": edit.geometry.to_dict(),
            "app": {"name": application.name, "version": application.version},
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    return document


def _source(photo: Path) -> dict[str, Any]:
    """Describe the photo the parameters belong to.

    The photo is only ever asked for its size and modification time; it is
    never opened here (PLAN.md section 10, rule 8).
    """
    source: dict[str, Any] = {"file": photo.name}
    try:
        stat = photo.stat()
    except OSError:
        return source
    source["size_bytes"] = stat.st_size
    source["modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
    return source


def _write_atomically(target: Path, document: dict[str, Any]) -> bool:
    """Write *document* to *target* via a temporary file; return whether it worked."""
    temp = target.with_name(target.name + _TEMP_SUFFIX)
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2)
        os.replace(temp, target)
    except OSError as error:
        logger.info("Could not write %s: %s", target, error)
        _remove(temp)
        return False
    return True


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
    "Application",
    "SIDECAR_VERSION",
    "SaveResult",
    "Sidecar",
    "delete",
    "edited_photos",
    "existing_path",
    "read_document",
    "fallback_path",
    "is_edited",
    "read",
    "save",
    "sidecar_path",
    "update_adjustments",
]
