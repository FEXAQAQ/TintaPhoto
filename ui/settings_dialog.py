"""The settings dialog (PLAN.md 3.10).

Categories down the left, their options on the right, one option to a row:
its name on the left with a line of explanation under it where one helps,
and its control on the right. There is no OK and no Cancel, because every
change takes effect as it is made; the window saves them when the dialog is
closed.

The dialog holds no state of its own. It writes into the
:class:`app.settings.Settings` it was handed and says that something
changed, and the window makes itself match.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app import settings as settings_module
from app.i18n import AVAILABLE_LANGUAGES, tr
from app.settings import Settings
from core import exporter
from core.cache import CACHE_LIMITS_BYTES
from ui import theme
from ui.shortcuts import SHORTCUTS
from ui.widgets.buttons import TextButton
from ui.widgets.combo import ComboBox
from ui.widgets.dialog import Dialog
from ui.widgets.focus import ring_on_keyboard_only
from ui.widgets.region import styled_background
from ui.widgets.segmented import Segmented
from ui.widgets.switch import Switch
from ui.widgets.text import TranslatedLabel, elide_middle, format_number

#: A megabyte and a gigabyte, for the cache sizes.
_BYTES_PER_MB = 1024 * 1024
_BYTES_PER_GB = 1024 * _BYTES_PER_MB

#: The format and quality combinations offered in one dropdown (PLAN.md
#: 3.10). Anything else the export dialog is set to is added to the list
#: when the settings are opened, so that it can always show what is current.
_EXPORT_CHOICES: tuple[tuple[str, int, int], ...] = (
    (exporter.FORMAT_JPEG, 100, 8),
    (exporter.FORMAT_JPEG, 90, 8),
    (exporter.FORMAT_JPEG, 80, 8),
    (exporter.FORMAT_JPEG, 70, 8),
    (exporter.FORMAT_PNG, 0, 8),
    (exporter.FORMAT_PNG, 0, 16),
    (exporter.FORMAT_TIFF, 0, 8),
    (exporter.FORMAT_TIFF, 0, 16),
)

#: Where exported files go; the second one carries a folder with it.
_LOCATION_BESIDE = "beside"
_LOCATION_CHOSEN = "chosen"


class SettingsRow(QWidget):
    """One option: its name, a line of explanation, and its control."""

    def __init__(
        self,
        label_key: str,
        control: QWidget,
        note: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.control = control
        self._label_key = label_key

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, theme.SPACE_SM, 0, theme.SPACE_SM)
        layout.setSpacing(theme.SPACE_MD)

        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(0)
        self.label = QLabel(tr(label_key), self)
        self.label.setWordWrap(True)
        text.addWidget(self.label)
        self.note = QLabel(note or "", self)
        self.note.setProperty("role", "caption")
        self.note.setWordWrap(True)
        self.note.setVisible(bool(note))
        text.addWidget(self.note)
        layout.addLayout(text, 1)
        # Level with the name rather than with the middle of the row, so
        # that a row with a line of explanation under it does not push
        # its control half a line down.
        layout.addWidget(
            control, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop
        )

    def set_note(self, note: str) -> None:
        """Show a line under the name, or nothing when it is empty."""
        self.note.setText(note)
        self.note.setVisible(bool(note))

    def retranslate(self) -> None:
        """Read the option's name again after a language change (U-02).

        The line of explanation under it is set by the dialog, because the
        three rows that have one say something only the dialog knows.
        """
        self.label.setText(tr(self._label_key))


class SettingsDialog(Dialog):
    """Everything in PLAN.md 3.10, in five categories."""

    #: Something was changed; the window reads the settings again.
    changed = Signal()

    def __init__(
        self,
        settings: Settings,
        controller: object | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._controller = controller

        self.setWindowTitle(tr("settings.title"))
        self.setModal(True)
        self.setFixedSize(theme.DIALOG_WIDTH_SETTINGS, theme.DIALOG_HEIGHT_SETTINGS)

        self.categories = QListWidget(self)
        self.categories.setFixedWidth(theme.SETTINGS_CATEGORY_WIDTH)
        self.categories.setFrameShape(QListWidget.Shape.NoFrame)
        styled_background(self.categories, "SettingsCategories")
        self.pages = QStackedWidget(self)

        self._category_keys = (
            "settings.category.general",
            "settings.category.preview",
            "settings.category.export",
            "settings.category.performance",
            "settings.category.shortcuts",
        )
        for label_key, build in zip(
            self._category_keys,
            (
                self._build_general,
                self._build_preview,
                self._build_export,
                self._build_performance,
                self._build_shortcuts,
            ),
            strict=True,
        ):
            QListWidgetItem(tr(label_key), self.categories)
            self.pages.addWidget(_scrolled(build()))
        self.categories.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.categories.setCurrentRow(0)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.categories)
        layout.addWidget(self.pages, 1)

    # -- building the pages ----------------------------------------------

    def _build_general(self) -> QWidget:
        """Language and what to do at start-up."""
        page, rows = _page()

        self.language_box = _combo(
            self._language_options(), self._settings.language, self._on_language
        )
        rows.addWidget(SettingsRow("settings.language", self.language_box))

        self.startup_box = _combo(
            self._startup_options(), self._settings.startup_action, self._on_startup
        )
        rows.addWidget(SettingsRow("settings.startup", self.startup_box))

        rows.addStretch(1)
        return page

    def _build_preview(self) -> QWidget:
        """The screen profile, the clipping marks, the background, the unit."""
        page, rows = _page()

        self.profile_switch = _switch(
            self._settings.use_display_profile, self._on_display_profile
        )
        self.profile_row = SettingsRow(
            "settings.display_profile", self.profile_switch, self._profile_note()
        )
        rows.addWidget(self.profile_row)

        self.clipping_choice = _segmented(
            [
                (settings_module.CLIPPING_COLOURED, "settings.clipping.coloured"),
                (settings_module.CLIPPING_MONO, "settings.clipping.mono"),
            ],
            self._settings.clipping_marks,
            self._on_clipping,
        )
        rows.addWidget(SettingsRow("settings.clipping", self.clipping_choice))

        self.background_choice = _segmented(
            [
                (settings_module.BACKGROUND_BLACK, "settings.background.black"),
                (settings_module.BACKGROUND_DARK, "settings.background.dark"),
                (settings_module.BACKGROUND_MID, "settings.background.mid"),
            ],
            self._settings.preview_background,
            self._on_background,
        )
        rows.addWidget(SettingsRow("settings.background", self.background_choice))

        self.temperature_choice = _segmented(
            [
                (settings_module.TEMPERATURE_KELVIN, "settings.temperature.kelvin"),
                (settings_module.TEMPERATURE_RELATIVE, "settings.temperature.relative"),
            ],
            self._settings.temperature_unit,
            self._on_temperature_unit,
        )
        rows.addWidget(SettingsRow("settings.temperature", self.temperature_choice))
        rows.addStretch(1)
        return page

    def _build_export(self) -> QWidget:
        """The four export settings of PLAN.md 3.10."""
        page, rows = _page()
        export = self._settings.export

        self.format_box = _combo(
            self._export_options(),
            _export_key(export.format, export.quality, export.bits),
            self._on_export_format,
        )
        rows.addWidget(SettingsRow("settings.export_format", self.format_box))

        self.location_box = _combo(
            self._location_options(),
            _LOCATION_CHOSEN if export.folder else _LOCATION_BESIDE,
            self._on_location,
        )
        self.location_row = SettingsRow(
            "settings.location", self.location_box, self._location_note()
        )
        rows.addWidget(self.location_row)

        self.suffix_field = QLineEdit(export.suffix, self)
        ring_on_keyboard_only(self.suffix_field)
        self.suffix_field.setFixedWidth(theme.SETTINGS_CONTROL_WIDTH)
        self.suffix_field.setFixedHeight(theme.BUTTON_HEIGHT)
        self.suffix_field.textEdited.connect(self._on_suffix)
        rows.addWidget(SettingsRow("settings.suffix", self.suffix_field))

        self.reveal_switch = _switch(export.reveal_after, self._on_reveal)
        rows.addWidget(SettingsRow("settings.reveal", self.reveal_switch))
        rows.addStretch(1)
        return page

    def _build_performance(self) -> QWidget:
        """The thumbnail cache and whether neighbours are decoded ahead."""
        page, rows = _page()

        self.cache_box = _combo(
            self._cache_options(), str(self._settings.cache_limit_bytes), self._on_cache_limit
        )
        self.cache_row = SettingsRow(
            "settings.cache_limit", self.cache_box, self._usage_note()
        )
        rows.addWidget(self.cache_row)

        self.clear_button = TextButton("settings.clear_cache", self)
        self.clear_button.clicked.connect(self._on_clear_cache)
        rows.addWidget(SettingsRow("settings.clear_cache_row", self.clear_button))

        self.preload_switch = _switch(self._settings.preload_neighbours, self._on_preload)
        rows.addWidget(SettingsRow("settings.preload", self.preload_switch))
        rows.addStretch(1)
        return page

    def _build_shortcuts(self) -> QWidget:
        """Section 8 as a read only list, with a box to search it."""
        page, rows = _page()

        self.search_field = QLineEdit(self)
        ring_on_keyboard_only(self.search_field)
        self.search_field.setPlaceholderText(tr("settings.search"))
        self.search_field.setFixedHeight(theme.BUTTON_HEIGHT)
        self.search_field.textChanged.connect(self._on_search)
        rows.addWidget(self.search_field)
        rows.addSpacing(theme.SPACE_SM)

        self._shortcut_rows: list[tuple[QWidget, str]] = []
        for entry in SHORTCUTS:
            row = QWidget(page)
            line = QHBoxLayout(row)
            line.setContentsMargins(0, theme.SPACE_XS, 0, theme.SPACE_XS)
            line.setSpacing(theme.SPACE_MD)
            name = TranslatedLabel(entry.label_key, row)
            name.setProperty("role", "secondary")
            keys = TranslatedLabel(entry.keys_key, row)
            keys.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            line.addWidget(name, 1)
            line.addWidget(keys)
            rows.addWidget(row)
            self._shortcut_rows.append(
                (row, f"{tr(entry.label_key)} {tr(entry.keys_key)}".lower())
            )
        rows.addStretch(1)
        return page

    # -- what the dropdowns offer ----------------------------------------

    def _language_options(self) -> list[tuple[str, str]]:
        """Return the languages, each in its own words (PLAN.md 7.4)."""
        return [(code, tr(f"settings.language.{code}")) for code in AVAILABLE_LANGUAGES]

    def _startup_options(self) -> list[tuple[str, str]]:
        """Return what the program can do when it starts (PLAN.md 3.10)."""
        return [
            (settings_module.STARTUP_WELCOME, tr("settings.startup.welcome")),
            (settings_module.STARTUP_LAST_FOLDER, tr("settings.startup.last_folder")),
        ]

    def _export_options(self) -> list[tuple[str, str]]:
        """Return the format and quality combinations, as they read."""
        return [
            (_export_key(*choice), _export_text(*choice))
            for choice in self._export_choices()
        ]

    def _location_options(self) -> list[tuple[str, str]]:
        """Return the two places an export can go (PLAN.md 3.10)."""
        return [
            (_LOCATION_BESIDE, tr("settings.location.beside")),
            (_LOCATION_CHOSEN, tr("settings.location.chosen")),
        ]

    def _cache_options(self) -> list[tuple[str, str]]:
        """Return the cache sizes, as they read."""
        return [(str(limit), _size_text(limit)) for limit in CACHE_LIMITS_BYTES]

    # -- reading it all again --------------------------------------------

    def retranslate(self) -> None:
        """Read the whole dialog again after a language change (U-02).

        The language is changed in this very dialog, so it is the first
        thing that has to follow: the rows and controls read themselves
        again, and what is built here - the categories, the dropdowns and
        the lines of explanation - is written again from the settings.
        """
        self.setWindowTitle(tr("settings.title"))
        for index, label_key in enumerate(self._category_keys):
            self.categories.item(index).setText(tr(label_key))

        _relabel(self.language_box, self._language_options())
        _relabel(self.startup_box, self._startup_options())
        _relabel(self.format_box, self._export_options())
        _relabel(self.location_box, self._location_options())
        _relabel(self.cache_box, self._cache_options())

        self.profile_row.set_note(self._profile_note())
        self.location_row.set_note(self._location_note())
        self.cache_row.set_note(self._usage_note())

        self.search_field.setPlaceholderText(tr("settings.search"))
        self._shortcut_rows = [
            (row, f"{tr(entry.label_key)} {tr(entry.keys_key)}".lower())
            for (row, _), entry in zip(self._shortcut_rows, SHORTCUTS, strict=True)
        ]
        # What was typed in the search box still has to narrow the list,
        # and it now has to match the names in the new language.
        self._on_search(self.search_field.text())

    # -- what the rows say -----------------------------------------------

    def _profile_note(self) -> str:
        """Return the name of the screen's profile, or that there is none."""
        from ui import native

        profile = native.display_profile()
        if not profile:
            return tr("settings.display_profile.none")
        from io import BytesIO

        from PIL import ImageCms

        try:
            name = ImageCms.getProfileDescription(
                ImageCms.ImageCmsProfile(BytesIO(profile))
            ).strip()
        except Exception:  # noqa: BLE001 - a profile that cannot be read has no name
            return tr("settings.display_profile.none")
        return tr("settings.display_profile.found", name=name)

    def _location_note(self) -> str:
        """Return the chosen export folder, shortened if it is long."""
        folder = self._settings.export.folder
        if not folder:
            return ""
        return elide_middle(folder, self.location_box, theme.DIALOG_WIDTH_SETTINGS // 2)

    def _usage_note(self) -> str:
        """Return how much room the cached thumbnails take (PLAN.md 3.10)."""
        usage = getattr(self._controller, "thumbnail_cache_usage", None)
        if usage is None:
            return ""
        return tr("settings.cache_usage", size=_size_text(usage()))

    def _export_choices(self) -> list[tuple[str, int, int]]:
        """Return the offered combinations, including the one in use."""
        export = self._settings.export
        current = (export.format, export.quality, export.bits)
        choices = list(_EXPORT_CHOICES)
        if _export_key(*current) not in {_export_key(*choice) for choice in choices}:
            choices.insert(0, current)
        return choices

    # -- reacting ---------------------------------------------------------

    def _announce(self) -> None:
        self.changed.emit()

    def _on_language(self, code: str) -> None:
        self._settings.language = code
        self._announce()

    def _on_startup(self, action: str) -> None:
        self._settings.startup_action = action
        self._announce()

    def _on_display_profile(self, on: bool) -> None:
        self._settings.use_display_profile = on
        self._announce()

    def _on_clipping(self, choice: str) -> None:
        self._settings.clipping_marks = choice
        self._announce()

    def _on_background(self, choice: str) -> None:
        self._settings.preview_background = choice
        self._announce()

    def _on_temperature_unit(self, choice: str) -> None:
        self._settings.temperature_unit = choice
        self._announce()

    def _on_export_format(self, key: str) -> None:
        name, quality, bits = key.split(":")
        export = self._settings.export
        export.format = name
        export.bits = int(bits)
        if int(quality):
            export.quality = int(quality)
        self._announce()

    def _on_location(self, choice: str) -> None:
        export = self._settings.export
        if choice == _LOCATION_BESIDE:
            export.folder = None
        else:
            chosen = QFileDialog.getExistingDirectory(
                self, tr("export.choose_folder"), export.folder or str(Path.home())
            )
            if not chosen:
                # Nothing was picked, so the dropdown goes back to what the
                # setting still says.
                self.location_box.setCurrentIndex(
                    self.location_box.findData(_LOCATION_BESIDE)
                )
                return
            export.folder = chosen
        self.location_row.set_note(self._location_note())
        self._announce()

    def _on_suffix(self, text: str) -> None:
        self._settings.export.suffix = text
        self._announce()

    def _on_reveal(self, on: bool) -> None:
        self._settings.export.reveal_after = on
        self._announce()

    def _on_cache_limit(self, value: str) -> None:
        self._settings.cache_limit_bytes = int(value)
        self._announce()
        self.cache_row.set_note(self._usage_note())

    def _on_clear_cache(self) -> None:
        clear = getattr(self._controller, "clear_thumbnail_cache", None)
        if clear is not None:
            clear()
        self.cache_row.set_note(self._usage_note())

    def _on_preload(self, on: bool) -> None:
        self._settings.preload_neighbours = on
        self._announce()

    def _on_search(self, text: str) -> None:
        """Show only the shortcuts whose name or keys match *text*."""
        wanted = text.strip().lower()
        for row, haystack in self._shortcut_rows:
            row.setVisible(wanted in haystack)


# --------------------------------------------------------------------------
# Small builders, so that every row of a kind is built exactly alike
# --------------------------------------------------------------------------


def _page() -> tuple[QWidget, QVBoxLayout]:
    """Return a page and the layout its rows go into."""
    page = QWidget()
    styled_background(page, "SettingsPage")
    rows = QVBoxLayout(page)
    rows.setContentsMargins(
        theme.PANEL_PADDING, theme.PANEL_PADDING, theme.PANEL_PADDING, theme.PANEL_PADDING
    )
    rows.setSpacing(0)
    return page, rows


def _scrolled(page: QWidget) -> QScrollArea:
    """Return *page* in a scroll area, for the categories that are long."""
    area = QScrollArea()
    styled_background(area, "SettingsPage")
    area.setWidget(page)
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    return area


def _combo(
    options: list[tuple[str, str]], current: str, on_change: Callable[[str], None]
) -> ComboBox:
    """Return a dropdown of *options*, each a value and the text for it."""
    box = ComboBox()
    box.setFixedWidth(theme.SETTINGS_CONTROL_WIDTH)
    for value, text in options:
        box.addItem(text, value)
    box.setCurrentIndex(max(0, box.findData(current)))
    box.currentIndexChanged.connect(lambda index: on_change(box.itemData(index)))
    return box


def _relabel(box: ComboBox, options: list[tuple[str, str]]) -> None:
    """Write a dropdown's items again, leaving the one chosen alone.

    The values never change with the language, so each item is found by the
    value it carries rather than by where it sits in the list.
    """
    for value, text in options:
        index = box.findData(value)
        if index >= 0:
            box.setItemText(index, text)


def _segmented(
    options: list[tuple[str, str]], current: str, on_change: Callable[[str], None]
) -> Segmented:
    """Return a segmented control of *options*, each a value and a text key."""
    control = Segmented(options)
    control.setFixedWidth(theme.SETTINGS_CONTROL_WIDTH)
    control.set_value(current)
    control.chosen.connect(on_change)
    return control


def _switch(on: bool, on_change: Callable[[bool], None]) -> Switch:
    """Return a switch in the state *on*."""
    control = Switch()
    control.setChecked(on)
    control.toggled.connect(on_change)
    return control


def _export_key(name: str, quality: int, bits: int) -> str:
    """Return the value a format and quality combination is stored under."""
    return f"{name}:{quality if name == exporter.FORMAT_JPEG else 0}:{bits}"


def _export_text(name: str, quality: int, bits: int) -> str:
    """Return how a format and quality combination reads (PLAN.md 3.10)."""
    if name == exporter.FORMAT_JPEG:
        return tr("settings.export_jpeg", quality=quality)
    return tr("settings.export_deep", format=name.upper(), bits=bits)


def _size_text(size_bytes: int) -> str:
    """Return a size in the larger of megabytes and gigabytes."""
    if size_bytes >= _BYTES_PER_GB:
        return tr("settings.gigabytes", size=format_number(size_bytes / _BYTES_PER_GB, 1))
    return tr("settings.megabytes", size=format_number(size_bytes / _BYTES_PER_MB, 0))
