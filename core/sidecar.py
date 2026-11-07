"""Reading and writing the parameter file of a photo (PLAN.md 7.1).

Editing is non-destructive: the photo itself is never touched, and everything
the user changes lives in a small JSON file kept in the program's own store,
never in the photo's folder. The file is named after a hash of the photo's
full path, so two photos of the same name in different folders stay apart, and
``DSC01234.ARW`` and ``DSC01234.JPG`` of a RAW+JPEG shot each have their own
(F-05).

Four things here are worth knowing about:

* A write goes to a temporary file and is then renamed, so an interrupted
  write cannot leave a half finished parameter file behind.
* When nothing is adjusted any more, the file is deleted rather than left
  behind holding nothing but defaults.
* Versions before 0.35.0 wrote the file beside the photo, as
  ``DSC01234.ARW.json``. Those are still read when the store has nothing for
  the photo, but never written, moved or deleted: the store is what counts
  from then on.
* Because an old file beside the photo would come back once the store's copy
  is deleted, a photo reset to its defaults while one is there gets a small
  "cleared" mark in the store that hides it.

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

#: Marks a photo reset to its defaults whose old file beside it must not count.
_CLEARED_SUFFIX: Final[str] = ".cleared"

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
    file at all.
    """

    path: Path | None = None
    deleted: bool = False
    failed: bool = False


# -------------------------------------------------------------------------
# Where a parameter file lives
# -------------------------------------------------------------------------


def legacy_path(photo: Path) -> Path:
    """Return where versions before 0.35.0 kept *photo*'s parameter file.

    That was beside the photo, with the whole file name kept, extension
    included. It is only ever read now, never written.
    """
    return photo.with_name(photo.name + SIDECAR_SUFFIX)


def store_key(photo: Path) -> str:
    """Return the name *photo* is stored under in the store.

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


def stored_path(photo: Path, store: Path) -> Path:
    """Return where *photo*'s parameter file goes in *store*."""
    return store / f"{store_key(photo)}{SIDECAR_SUFFIX}"


def _cleared_path(photo: Path, store: Path) -> Path:
    """Return the mark that hides an old file beside *photo*."""
    return store / f"{store_key(photo)}{_CLEARED_SUFFIX}"


def existing_path(photo: Path, store: Path) -> Path | None:
    """Return the parameter file of *photo*, or ``None`` if it has none.

    The store wins. An old file beside the photo counts only when the store
    knows nothing about the photo - neither a file nor a mark saying it was
    reset since.
    """
    stored = stored_path(photo, store)
    if stored.is_file():
        return stored
    if _cleared_path(photo, store).is_file():
        return None
    beside = legacy_path(photo)
    if beside.is_file():
        return beside
    return None


def is_edited(photo: Path, store: Path) -> bool:
    """Return whether *photo* has been adjusted (for the filmstrip mark).

    Only the presence of the file is checked, never its contents: a parameter
    file exists exactly when something is not at its default, because saving
    an untouched photo deletes it. Opening a folder of three hundred photos
    therefore costs a few hundred cheap questions to the file system rather
    than three hundred JSON files to parse.
    """
    return existing_path(photo, store) is not None


def edited_photos(photos: list[Path], store: Path) -> set[Path]:
    """Return which of *photos* have a parameter file."""
    return {photo for photo in photos if is_edited(photo, store)}


def adopt(old_store: Path, store: Path) -> int:
    """Move the files of an earlier store into *store*; return how many moved.

    Before 0.35.0 the program only used a store - in another folder - for
    photos whose own folder could not be written to. Those files are the
    program's own, so they are moved rather than copied. A file *store*
    already has is newer and is kept; the old one is then left where it is.
    """
    try:
        found = sorted(old_store.glob(f"*{SIDECAR_SUFFIX}"))
    except OSError:
        logger.exception("Could not look in the earlier store %s", old_store)
        return 0
    if not found:
        return 0
    try:
        store.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("The parameter store %s cannot be used", store)
        return 0
    moved = 0
    for path in found:
        target = store / path.name
        if target.exists():
            continue
        try:
            os.replace(path, target)
        except OSError:
            logger.exception("Could not move %s into the store", path)
            continue
        moved += 1
    return moved


# -------------------------------------------------------------------------
# Reading
# -------------------------------------------------------------------------


def read(photo: Path, reference: EditState, store: Path) -> Sidecar | None:
    """Return the stored parameters of *photo*, or ``None`` if there are none.

    *reference* is the state the photo opens in - for a RAW photo that is the
    white balance the camera recorded - and fills in any field the file does
    not have (PLAN.md section 7.1, "欄位缺漏用預設值補上"). A file that cannot
    be read or makes no sense is logged and treated as absent, so that one bad
    parameter file cannot stop a photo from opening.
    """
    found = read_document(photo, store)
    if found is None:
        return None
    path, data = found
    return _from_dict(data, reference, path)


def read_document(photo: Path, store: Path) -> tuple[Path, dict[str, Any]] | None:
    """Return the parameter file of *photo* as it is written, and where it is.

    Used where the file has to be changed rather than understood, which is
    what pasting onto a photo that is not open does.
    """
    path = existing_path(photo, store)
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
    store: Path,
    extra: dict[str, Any] | None = None,
) -> SaveResult:
    """Store the edit of *photo*, or delete the file if there is nothing in it.

    A photo whose adjustments are all back at *reference* and which has no
    crop has nothing worth storing, so its parameter file is removed instead
    of being left holding defaults (PLAN.md section 7.1).
    """
    if edit.is_untouched(reference):
        return SaveResult(deleted=delete(photo, store))
    return _store(photo, _to_dict(photo, edit, application, extra or {}), store)


def update_adjustments(
    photo: Path,
    values: dict[str, float | str],
    application: Application,
    store: Path,
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

    found = read_document(photo, store)
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
    return _store(photo, document, store)


def delete(photo: Path, store: Path) -> bool:
    """Remove the parameters of *photo*; return whether it had any.

    An old file beside the photo is left exactly where it is - it is not the
    program's to delete any more - and hidden by a mark in the store instead,
    or it would come back the next time the photo is opened.
    """
    removed = _remove(stored_path(photo, store))
    cleared = _cleared_path(photo, store)
    if legacy_path(photo).is_file() and not cleared.is_file():
        try:
            store.mkdir(parents=True, exist_ok=True)
            cleared.touch()
        except OSError:
            logger.exception("Could not mark %s as reset", photo)
        else:
            removed = True
    return removed


def _store(photo: Path, document: dict[str, Any], store: Path) -> SaveResult:
    """Write *document* as *photo*'s parameter file in *store*."""
    try:
        store.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("The parameter store %s cannot be used", store)
        return SaveResult(failed=True)
    target = stored_path(photo, store)
    if not _write_atomically(target, document):
        logger.error("The parameters of %s could not be stored", photo)
        return SaveResult(failed=True)
    # The file in the store wins from now on, so a mark hiding an old file
    # beside the photo is no longer needed.
    _remove(_cleared_path(photo, store))
    return SaveResult(path=target)


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
    "adopt",
    "delete",
    "edited_photos",
    "existing_path",
    "is_edited",
    "legacy_path",
    "read",
    "read_document",
    "save",
    "store_key",
    "stored_path",
    "update_adjustments",
]
