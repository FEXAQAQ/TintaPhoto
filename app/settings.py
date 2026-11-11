"""Application settings stored in ``%APPDATA%\\TintaPhoto\\settings.json``.

PLAN.md section 7.3: interface language, window position and size, the last
opened folder and whether each panel is collapsed. Unknown fields read from
disk are preserved so that a file written by a newer version is not damaged.
Writing goes to a temporary file first and is then renamed, so an interrupted
write cannot leave a corrupt settings file (PLAN.md 7.1).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Final

from app.i18n import AVAILABLE_LANGUAGES, DEFAULT_LANGUAGE
from app.paths import ensure_dir, settings_file
from core import collage, exporter
from core import split as splitting
from core.cache import CACHE_LIMITS_BYTES, DEFAULT_CACHE_LIMIT_BYTES
from core.exporter import ExportOptions
from core.watermark import WatermarkSpec


logger = logging.getLogger(__name__)

SETTINGS_VERSION: Final[int] = 1

STARTUP_WELCOME: Final[str] = "welcome"
STARTUP_LAST_FOLDER: Final[str] = "last_folder"
STARTUP_ACTIONS: Final[tuple[str, ...]] = (STARTUP_WELCOME, STARTUP_LAST_FOLDER)

#: How clipping is marked on the photo (PLAN.md 3.10): red and blue, or one
#: colourless mark for those who would rather not have hues on the photo.
CLIPPING_COLOURED: Final[str] = "coloured"
CLIPPING_MONO: Final[str] = "mono"
CLIPPING_MARKS: Final[tuple[str, ...]] = (CLIPPING_COLOURED, CLIPPING_MONO)

#: The background behind the photo (PLAN.md 3.10).
BACKGROUND_BLACK: Final[str] = "black"
BACKGROUND_DARK: Final[str] = "dark"
BACKGROUND_MID: Final[str] = "mid"
BACKGROUNDS: Final[tuple[str, ...]] = (BACKGROUND_BLACK, BACKGROUND_DARK, BACKGROUND_MID)

#: How a RAW photo's white balance is shown (PLAN.md 3.10).
TEMPERATURE_KELVIN: Final[str] = "kelvin"
TEMPERATURE_RELATIVE: Final[str] = "relative"
TEMPERATURE_UNITS: Final[tuple[str, ...]] = (TEMPERATURE_KELVIN, TEMPERATURE_RELATIVE)

#: How the histogram is drawn (A-12): the three channels over one another,
#: or brightness alone. Clicking the histogram swaps them.
HISTOGRAM_RGB: Final[str] = "rgb"
HISTOGRAM_LUMA: Final[str] = "luma"
HISTOGRAM_MODES: Final[tuple[str, ...]] = (HISTOGRAM_RGB, HISTOGRAM_LUMA)

MAX_RECENT_FOLDERS: Final[int] = 10


@dataclass
class ExportSettings:
    """What the export dialog was last set to (PLAN.md 3.11, 7.3).

    The dialog opens on these again, so that exporting a folder of photos
    one after another does not mean choosing the same things each time.
    """

    format: str = exporter.FORMAT_JPEG
    quality: int = exporter.QUALITY_DEFAULT
    bits: int = 8
    #: The long edge asked for, or ``None`` for the photo's own size (X-03).
    long_edge: int | None = None
    #: Where files go, or ``None`` for the folder the photo is in (X-05).
    folder: str | None = None
    #: What is added to the original name (X-05).
    suffix: str = exporter.DEFAULT_SUFFIX
    keep_metadata: bool = True
    remove_gps: bool = True
    #: Open the folder when the export is done (PLAN.md 3.10).
    reveal_after: bool = False

    def remember(self, options: ExportOptions) -> None:
        """Take on what was just exported with."""
        options = options.sane()
        self.format = options.format
        self.quality = options.quality
        self.bits = options.bits
        self.long_edge = options.long_edge
        self.keep_metadata = options.keep_metadata
        self.remove_gps = options.remove_gps

    def to_dict(self) -> dict[str, Any]:
        """Return the choices as a JSON compatible dictionary."""
        return {
            "format": self.format,
            "quality": self.quality,
            "bits": self.bits,
            "long_edge": self.long_edge,
            "folder": self.folder,
            "suffix": self.suffix,
            "keep_metadata": self.keep_metadata,
            "remove_gps": self.remove_gps,
            "reveal_after": self.reveal_after,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExportSettings":
        """Build the choices from stored data, ignoring anything unusable."""
        if not isinstance(data, dict):
            data = {}
        settings = cls()
        stored = ExportOptions(
            format=str(data.get("format", settings.format)),
            quality=_as_int(data.get("quality"), settings.quality),
            bits=_as_int(data.get("bits"), settings.bits),
            long_edge=_as_optional_int(data.get("long_edge")),
            keep_metadata=bool(data.get("keep_metadata", True)),
            remove_gps=bool(data.get("remove_gps", True)),
        ).sane()
        settings.remember(stored)
        folder = data.get("folder")
        settings.folder = str(folder) if folder else None
        suffix = data.get("suffix")
        settings.suffix = str(suffix) if isinstance(suffix, str) else settings.suffix
        settings.reveal_after = bool(data.get("reveal_after", False))
        return settings


@dataclass
class WatermarkSettings:
    """The watermark and whether exports carry it (PLAN.md 2.9, 7.3).

    One for every photo: it belongs to where a photo is going, not to the
    photo, so it lives here rather than in any parameter file (W-11).
    """

    #: Lay it over exported photos (W-09); off until the user turns it on.
    enabled: bool = False
    spec: WatermarkSpec = field(default_factory=WatermarkSpec)

    @property
    def active(self) -> WatermarkSpec | None:
        """Return the watermark exports carry now, or ``None`` for none."""
        if not self.enabled or self.spec.is_empty:
            return None
        return self.spec

    def to_dict(self) -> dict[str, Any]:
        """Return the watermark as a JSON compatible dictionary."""
        return {"enabled": self.enabled, **self.spec.to_dict()}

    @classmethod
    def from_dict(cls, data: Any) -> "WatermarkSettings":
        """Build the watermark from stored data; a file without one has it off."""
        if not isinstance(data, dict):
            return cls()
        return cls(enabled=data.get("enabled") is True, spec=WatermarkSpec.from_dict(data))


@dataclass
class CollageSettings:
    """How the last collage was laid out (PLAN.md 2.8, 7.3).

    Only the layout, and whether its photos were also written one by one for
    a carousel: which photos were in it and where they sat belong to that one
    collage and are not kept.
    """

    platform: str = collage.PLATFORM_DEFAULT
    canvas: str = collage.CANVAS_DEFAULT
    split: str = collage.SPLIT_SIDE_BY_SIDE
    fit: str = collage.FIT_COVER
    gap: int = collage.GAP_MIN
    background: str = collage.BACKGROUND_WHITE
    linked: bool = False
    count: int = collage.COUNT_MIN
    seams: tuple[float, ...] = (collage.SEAM_DEFAULT,)
    #: Write each photo on a canvas of its own beside the collage (K-13).
    carousel: bool = False

    @property
    def layout(self) -> collage.Layout:
        """Return these choices as the layout they describe."""
        return collage.Layout(
            platform=self.platform,
            canvas=self.canvas,
            split=self.split,
            fit=self.fit,
            gap=self.gap,
            background=self.background,
            linked=self.linked,
            count=self.count,
            seams=self.seams,
        ).sane()

    def remember(self, layout: collage.Layout) -> None:
        """Take on *layout*."""
        layout = layout.sane()
        self.platform = layout.platform
        self.canvas = layout.canvas
        self.split = layout.split
        self.fit = layout.fit
        self.gap = layout.gap
        self.background = layout.background
        self.linked = layout.linked
        self.count = layout.count
        self.seams = layout.seams

    def to_dict(self) -> dict[str, Any]:
        """Return the choices as a JSON compatible dictionary."""
        return {
            "platform": self.platform,
            "canvas": self.canvas,
            "split": self.split,
            "fit": self.fit,
            "gap": self.gap,
            "background": self.background,
            "linked": self.linked,
            "count": self.count,
            "seams": list(self.seams),
            "carousel": self.carousel,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "CollageSettings":
        """Build the choices from stored data, ignoring anything unusable."""
        if not isinstance(data, dict):
            data = {}
        settings = cls()
        seams = data.get("seams")
        if not isinstance(seams, list):
            # Before 0.36 a collage was always two photos with one seam.
            seams = [data.get("seam", collage.SEAM_DEFAULT)]
        canvas = data.get("canvas")
        if canvas is None:
            # Before 0.36 the canvas was 16:9 lying down or 4:5 standing up.
            portrait = data.get("orientation") == "portrait"
            canvas = collage.CANVAS_4X5 if portrait else collage.CANVAS_16X9
        settings.remember(
            collage.Layout(
                platform=str(data.get("platform", settings.platform)),
                canvas=str(canvas),
                split=str(data.get("split", settings.split)),
                fit=str(data.get("fit", settings.fit)),
                gap=_as_int(data.get("gap"), settings.gap),
                background=str(data.get("background", settings.background)),
                linked=data.get("linked") is True,
                count=_as_int(data.get("count"), settings.count),
                seams=tuple(seams),
            )
        )
        settings.carousel = data.get("carousel") is True
        return settings


@dataclass
class SplitSettings:
    """How the photo was last cut into pieces (PLAN.md 2.12, 7.3)."""

    split: str = splitting.SPLIT_SIDE_BY_SIDE
    count: int = splitting.COUNT_DEFAULT
    #: The count chosen side by side or stacked, which a grid's fixed four
    #: leaves alone, so that leaving the grid goes back to it.
    line_count: int = splitting.COUNT_DEFAULT

    @property
    def spec(self) -> splitting.SplitSpec:
        """Return these choices as the cut they describe."""
        return splitting.SplitSpec(self.split, self.count).sane()

    def remember(self, spec: splitting.SplitSpec) -> None:
        """Take on *spec*."""
        spec = spec.sane()
        self.split = spec.split
        self.count = spec.count
        if spec.split != splitting.SPLIT_GRID:
            self.line_count = spec.count

    def to_dict(self) -> dict[str, Any]:
        """Return the choices as a JSON compatible dictionary."""
        return {"split": self.split, "count": self.line_count}

    @classmethod
    def from_dict(cls, data: Any) -> "SplitSettings":
        """Build the choices from stored data, ignoring anything unusable."""
        if not isinstance(data, dict):
            data = {}
        settings = cls()
        stored = _as_int(data.get("count"), settings.count)
        line = splitting.SplitSpec(splitting.SPLIT_SIDE_BY_SIDE, stored).sane()
        settings.line_count = line.count
        chosen = str(data.get("split", settings.split))
        settings.remember(splitting.SplitSpec(chosen, line.count))
        return settings


@dataclass(frozen=True)
class RecentFolder:
    """What the welcome screen says about a folder opened lately (PLAN.md 3.7)."""

    #: How many photos it held when it was last opened.
    count: int
    #: The day it was last opened, as an ISO date (YYYY-MM-DD).
    opened: str

    def to_dict(self) -> dict[str, Any]:
        """Return the entry as it is stored."""
        return {"count": self.count, "opened": self.opened}

    @classmethod
    def from_dict(cls, data: Any) -> "RecentFolder | None":
        """Build an entry from stored data, or ``None`` if it makes no sense."""
        if not isinstance(data, dict):
            return None
        count = _as_optional_int(data.get("count"))
        opened = data.get("opened")
        if count is None or count < 0 or not isinstance(opened, str):
            return None
        try:
            date.fromisoformat(opened)
        except ValueError:
            return None
        return cls(count=count, opened=opened)

    @property
    def day(self) -> date:
        """Return the day it was last opened."""
        return date.fromisoformat(self.opened)


@dataclass
class WindowState:
    """Remembered window geometry (PLAN.md U-04).

    Every field may be ``None``, meaning "never saved yet"; the window then
    applies the default size from :mod:`ui.theme`. Keeping pixel defaults out
    of this layer leaves ``ui/theme.py`` the single source of interface sizes.
    """

    x: int | None = None
    y: int | None = None
    width: int | None = None
    height: int | None = None
    maximized: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the geometry as a JSON compatible dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "maximized": self.maximized,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WindowState":
        """Build a window state from stored data, ignoring bad values."""
        return cls(
            x=_as_optional_int(data.get("x")),
            y=_as_optional_int(data.get("y")),
            width=_as_optional_int(data.get("width")),
            height=_as_optional_int(data.get("height")),
            maximized=bool(data.get("maximized", False)),
        )


@dataclass
class Settings:
    """All persisted application settings."""

    language: str = DEFAULT_LANGUAGE
    startup_action: str = STARTUP_WELCOME
    window: WindowState = field(default_factory=WindowState)
    inspector_collapsed: bool = False
    filmstrip_collapsed: bool = False
    #: Decode the photos either side of the current one in advance.
    preload_neighbours: bool = True
    #: Correct the preview for the screen's own profile (PLAN.md 6 step 13).
    use_display_profile: bool = True
    clipping_marks: str = CLIPPING_COLOURED
    preview_background: str = BACKGROUND_DARK
    #: Whether a RAW photo's white balance is shown in kelvin or as a nudge
    #: either way from what the camera recorded.
    temperature_unit: str = TEMPERATURE_KELVIN
    #: The histogram in three channels or as brightness alone (A-12).
    histogram_mode: str = HISTOGRAM_RGB
    #: How much room the thumbnail cache may take (PLAN.md 3.10).
    cache_limit_bytes: int = DEFAULT_CACHE_LIMIT_BYTES
    last_folder: str | None = None
    #: Where the open dialogs start: the folder of whatever was opened last,
    #: a whole folder or photos opened by themselves (F-01).
    browse_folder: str | None = None
    recent_folders: list[str] = field(default_factory=list)
    #: How many photos each recent folder held and when it was last opened,
    #: by its path. A folder remembered by an older version has no entry.
    recent_details: dict[str, RecentFolder] = field(default_factory=dict)
    #: The export dialog's last choices (PLAN.md 7.3).
    export: ExportSettings = field(default_factory=ExportSettings)
    #: The collage's last layout (PLAN.md 2.8).
    collage: CollageSettings = field(default_factory=CollageSettings)
    #: The watermark on exported photos (PLAN.md 2.9).
    watermark: WatermarkSettings = field(default_factory=WatermarkSettings)
    #: How the photo is cut into pieces (PLAN.md 2.12).
    split: SplitSettings = field(default_factory=SplitSettings)
    #: Fields written by other versions, kept so they survive a round trip.
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- conversion ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the settings as a JSON compatible dictionary."""
        data: dict[str, Any] = dict(self.extra)
        data.update(
            {
                "version": SETTINGS_VERSION,
                "language": self.language,
                "startup_action": self.startup_action,
                "window": self.window.to_dict(),
                "panels": {
                    "inspector_collapsed": self.inspector_collapsed,
                    "filmstrip_collapsed": self.filmstrip_collapsed,
                },
                "preload_neighbours": self.preload_neighbours,
                "use_display_profile": self.use_display_profile,
                "clipping_marks": self.clipping_marks,
                "preview_background": self.preview_background,
                "temperature_unit": self.temperature_unit,
                "histogram_mode": self.histogram_mode,
                "cache_limit_bytes": self.cache_limit_bytes,
                "last_folder": self.last_folder,
                "browse_folder": self.browse_folder,
                "recent_folders": self.recent_folders,
                "recent_details": {
                    folder: detail.to_dict()
                    for folder, detail in self.recent_details.items()
                    if folder in self.recent_folders
                },
                "export": self.export.to_dict(),
                "collage": self.collage.to_dict(),
                "watermark": self.watermark.to_dict(),
                "split": self.split.to_dict(),
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        """Build settings from stored data; missing fields use the defaults."""
        known = {
            "version",
            "language",
            "startup_action",
            "window",
            "panels",
            "preload_neighbours",
            # Retired in 0.35.0, when parameter files stopped going beside the
            # photos; listed so that an old settings file drops it.
            "warn_read_only",
            "use_display_profile",
            "clipping_marks",
            "preview_background",
            "temperature_unit",
            "histogram_mode",
            "cache_limit_bytes",
            "last_folder",
            "browse_folder",
            "recent_folders",
            "recent_details",
            "export",
            "collage",
            "watermark",
            "split",
        }
        panels = data.get("panels")
        panels = panels if isinstance(panels, dict) else {}
        window = data.get("window")
        window = window if isinstance(window, dict) else {}
        recent = data.get("recent_folders")
        recent = [str(item) for item in recent] if isinstance(recent, list) else []
        recent = recent[:MAX_RECENT_FOLDERS]
        stored_details = data.get("recent_details")
        stored_details = stored_details if isinstance(stored_details, dict) else {}
        details = {
            folder: detail
            for folder in recent
            if (detail := RecentFolder.from_dict(stored_details.get(folder))) is not None
        }
        language = str(data.get("language", DEFAULT_LANGUAGE))
        startup = str(data.get("startup_action", STARTUP_WELCOME))
        last_folder = data.get("last_folder")
        # A file from before photos could be opened by themselves has only
        # the last folder, which is where the dialogs started then.
        browse_folder = data.get("browse_folder", last_folder)

        return cls(
            language=language if language in AVAILABLE_LANGUAGES else DEFAULT_LANGUAGE,
            startup_action=startup if startup in STARTUP_ACTIONS else STARTUP_WELCOME,
            window=WindowState.from_dict(window),
            inspector_collapsed=bool(panels.get("inspector_collapsed", False)),
            filmstrip_collapsed=bool(panels.get("filmstrip_collapsed", False)),
            preload_neighbours=bool(data.get("preload_neighbours", True)),
            use_display_profile=bool(data.get("use_display_profile", True)),
            clipping_marks=_one_of(
                data.get("clipping_marks"), CLIPPING_MARKS, CLIPPING_COLOURED
            ),
            preview_background=_one_of(
                data.get("preview_background"), BACKGROUNDS, BACKGROUND_DARK
            ),
            temperature_unit=_one_of(
                data.get("temperature_unit"), TEMPERATURE_UNITS, TEMPERATURE_KELVIN
            ),
            histogram_mode=_one_of(
                data.get("histogram_mode"), HISTOGRAM_MODES, HISTOGRAM_RGB
            ),
            cache_limit_bytes=_one_of(
                _as_optional_int(data.get("cache_limit_bytes")),
                CACHE_LIMITS_BYTES,
                DEFAULT_CACHE_LIMIT_BYTES,
            ),
            last_folder=str(last_folder) if last_folder else None,
            browse_folder=str(browse_folder) if browse_folder else None,
            recent_folders=recent,
            recent_details=details,
            export=ExportSettings.from_dict(data.get("export", {})),
            collage=CollageSettings.from_dict(data.get("collage")),
            watermark=WatermarkSettings.from_dict(data.get("watermark")),
            split=SplitSettings.from_dict(data.get("split")),
            extra={key: value for key, value in data.items() if key not in known},
        )

    # -- persistence -----------------------------------------------------

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        """Read the settings file; a missing or damaged file yields defaults."""
        target = path or settings_file()
        try:
            # utf-8-sig: the file is written without a byte order mark, but
            # an editor that has saved it as "UTF-8 with BOM" would
            # otherwise silently throw every setting away.
            with target.open(encoding="utf-8-sig") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return cls()
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            logger.exception("Could not read settings from %s, using defaults", target)
            return cls()
        if not isinstance(data, dict):
            logger.error("Settings in %s are not an object, using defaults", target)
            return cls()
        return cls.from_dict(data)

    def save(self, path: Path | None = None) -> bool:
        """Write the settings atomically; return whether the write succeeded."""
        target = path or settings_file()
        if not ensure_dir(target.parent):
            logger.error("Could not create settings directory %s", target.parent)
            return False
        temp = target.with_name(target.name + ".tmp")
        try:
            with temp.open("w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
            os.replace(temp, target)
        except OSError:
            logger.exception("Could not write settings to %s", target)
            temp.unlink(missing_ok=True)
            return False
        return True

    # -- helpers ---------------------------------------------------------

    def remember_folder(
        self, folder: str, count: int | None = None, today: date | None = None
    ) -> None:
        """Record *folder* as the most recently opened one.

        *count* is how many photos it held; with it, the welcome screen can
        say so and when it was opened (PLAN.md 3.7).
        """
        self.last_folder = folder
        self.browse_folder = folder
        remaining = [item for item in self.recent_folders if item != folder]
        self.recent_folders = [folder, *remaining][:MAX_RECENT_FOLDERS]
        if count is not None:
            opened = (today or date.today()).isoformat()
            self.recent_details[folder] = RecentFolder(count=count, opened=opened)
        self.recent_details = {
            item: detail
            for item, detail in self.recent_details.items()
            if item in self.recent_folders
        }

    def remember_browse_folder(self, folder: str) -> None:
        """Start the open dialogs in *folder* next time.

        Photos opened by themselves leave only this behind: their folder is
        not put on the recent list, nor opened again at start-up (F-01).
        """
        self.browse_folder = folder

    def forget_folder(self, folder: str) -> None:
        """Take *folder* off the recent list; the folder itself is not touched.

        It is no longer the last folder either, so it is not opened again at
        start-up behind the user's back (PLAN.md 3.7).
        """
        self.recent_folders = [item for item in self.recent_folders if item != folder]
        self.recent_details.pop(folder, None)
        if self.last_folder == folder:
            self.last_folder = None
        if self.browse_folder == folder:
            self.browse_folder = None

    def clear_recent(self) -> None:
        """Empty the recent list, the last folder and where dialogs start (PLAN.md 3.1)."""
        self.recent_folders = []
        self.recent_details = {}
        self.last_folder = None
        self.browse_folder = None


def _one_of(value: Any, allowed: tuple[Any, ...], default: Any) -> Any:
    """Return *value* when it is one of *allowed*, and *default* otherwise."""
    return value if value in allowed else default


def _as_int(value: Any, default: int) -> int:
    """Return *value* as an int, or *default* when it is missing or invalid."""
    number = _as_optional_int(value)
    return default if number is None else number


def _as_optional_int(value: Any) -> int | None:

    """Return *value* as an int, or ``None`` when it is missing or invalid."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
