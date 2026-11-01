"""The export dialog (PLAN.md 3.11).

Six rows, each with its label in a column of its own so that every control
starts at the same place: the format, the quality or the bit depth, the size,
the name, the folder and what to do with the details of the shot - or, for a
collage, whether its photos are also written one by one for a carousel.
Along the bottom, roughly how large the file will be, and the two buttons.

The dialog only decides; it exports nothing and develops nothing. Anything it
cannot work out for itself - how many pixels will come out, how large the
file will be, whether the original will simply be copied - it asks for by
saying that its settings changed, and is told.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from app.i18n import tr
from app.settings import ExportSettings
from core import exporter
from core.exporter import ExportOptions
from ui import theme
from ui.widgets.buttons import PrimaryButton, TextButton
from ui.widgets.dialog import Dialog
from ui.widgets.focus import ring_on_keyboard_only
from ui.widgets.segmented import Segmented
from ui.widgets.slider_row import SliderRow, SliderSpec
from ui.widgets.switch import Switch
from ui.widgets.text import elide_middle, format_number


#: A megabyte, for the estimate at the bottom of the dialog.
_BYTES_PER_MB = 1024 * 1024

_SIZE_ORIGINAL = "original"
_SIZE_CUSTOM = "custom"


def quality_spec() -> SliderSpec:
    """Return the quality slider: the same control as every other slider."""
    return SliderSpec(
        name="quality",
        label_key="export.quality",
        minimum=float(exporter.QUALITY_MIN),
        maximum=float(exporter.QUALITY_MAX),
        default=float(exporter.QUALITY_DEFAULT),
        signed=False,
        show_name=False,
        fill_from=float(exporter.QUALITY_MIN),
    )


class ExportDialog(Dialog):
    """Asks how the photo should be written (X-01 to X-08)."""

    #: The settings changed; the window answers with the size, the estimate
    #: and whether the original would be copied.
    options_changed = Signal(object)

    def __init__(
        self,
        source: Path,
        settings: ExportSettings,
        parent: QWidget | None = None,
        *,
        collage_name: str | None = None,
        collage_size: tuple[int, int] | None = None,
        carousel: bool = False,
        batch: int | None = None,
    ) -> None:
        """Ask about exporting *source*, or a collage beside it.

        With *collage_name* the dialog writes a collage (PLAN.md 3.13): its
        size is the canvas and cannot be chosen, and nothing in it is one
        shot, so there are no details of one to keep or leave out. *source*
        is then the first photo, whose folder the collage is offered in, and
        *carousel* whether its photos are offered one by one as well (K-13).

        With *batch*, the number of photos ticked in the filmstrip, the
        dialog writes all of them (X-09): each keeps its own name and only
        what is added to it is asked for, the size of each is its own, and
        *source* is the first of them, whose folder is offered.
        """
        super().__init__(parent)
        self._source = source
        self._batch = batch
        self._collage = collage_name is not None
        self._stem = collage_name if collage_name is not None else source.stem
        self._settings = ExportSettings(**vars(settings))
        self._folder = Path(self._settings.folder) if self._settings.folder else source.parent

        self.setWindowTitle(tr("export.title"))
        self.setModal(True)
        self.setFixedWidth(theme.DIALOG_WIDTH_EXPORT)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            theme.PANEL_PADDING, theme.PANEL_PADDING, theme.PANEL_PADDING, theme.PANEL_PADDING
        )
        layout.setSpacing(theme.SPACE_MD)

        title = QLabel(tr("export.title"), self)
        title.setProperty("role", "title")
        layout.addWidget(title)
        if self._collage:
            subtitle = tr("collage.title")
        elif batch is not None:
            subtitle = tr("export.batch_photos", count=batch)
        else:
            subtitle = source.name
        name = QLabel(subtitle, self)
        name.setProperty("role", "secondary")
        layout.addWidget(name)
        layout.addSpacing(theme.SPACE_XS)

        self._build_format(layout)
        self._build_quality(layout)
        self._build_size(layout)
        self._build_name(layout)
        self._build_location(layout)
        self._build_metadata(layout)
        self._build_carousel(layout)
        layout.addStretch(1)
        self._build_footer(layout)

        if collage_size is not None:
            self._fix_size(*collage_size)
        self.carousel_switch.setChecked(carousel)
        self._show_row(self._carousel_row, self._collage)
        self._show_settings()
        if batch is not None:
            # Nobody is asked about several photos, so nothing will hide it.
            self.set_copying(False)
        # The name is the one thing most likely to be changed, and starting
        # there keeps the focus ring off a segment that is already marked as
        # the current one (PLAN.md 3.6).
        self.name_field.setFocus()

    # -- building --------------------------------------------------------

    def _row(self, layout: QVBoxLayout, label_key: str | None) -> QHBoxLayout:
        """Add a row whose label sits in the fixed column on the left."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACE_SM)
        label = QLabel(tr(label_key) if label_key else "", self)
        label.setProperty("role", "secondary")
        label.setFixedWidth(theme.DIALOG_LABEL_WIDTH)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(label)
        layout.addLayout(row)
        return row

    def _build_format(self, layout: QVBoxLayout) -> None:
        """The format, as a segmented control (X-02)."""
        self.format_choice = Segmented(
            [
                (exporter.FORMAT_JPEG, "export.format.jpg"),
                (exporter.FORMAT_PNG, "export.format.png"),
                (exporter.FORMAT_TIFF, "export.format.tiff"),
            ],
            self,
        )
        self.format_choice.chosen.connect(self._on_format)
        self._row(layout, "export.format").addWidget(self.format_choice, 1)

    def _build_quality(self, layout: QVBoxLayout) -> None:
        """The JPEG quality, or the bit depth of a PNG or TIFF (X-02)."""
        self._quality_row = self._row(layout, "export.quality")
        self.quality_slider = SliderRow(quality_spec(), self)
        self.quality_slider.value_changed.connect(lambda *_: self._changed())
        self._quality_row.addWidget(self.quality_slider, 1)

        self._depth_row = self._row(layout, "export.depth")
        self.depth_choice = Segmented(
            [("8", "export.depth.8"), ("16", "export.depth.16")], self
        )
        self.depth_choice.chosen.connect(lambda *_: self._changed())
        self._depth_row.addWidget(self.depth_choice, 1)
        self._depth_label = self._label_of(self._depth_row)
        self._quality_label = self._label_of(self._quality_row)

    def _build_size(self, layout: QVBoxLayout) -> None:
        """The size, with the pixels it comes to shown beside it (X-03)."""
        row = self._row(layout, "export.size")
        self.size_choice = Segmented(
            [(_SIZE_ORIGINAL, "export.size.original"), (_SIZE_CUSTOM, "export.size.custom")],
            self,
        )
        self.size_choice.chosen.connect(lambda *_: self._on_size())
        # Two thirds of the row for two buttons, so that a segment here is
        # the same width as one in the three part control above it
        # (PLAN.md 3.5: elements of one kind look exactly alike).
        row.addWidget(self.size_choice, 2)
        self.pixels_label = QLabel("", self)
        self.pixels_label.setProperty("role", "secondary")
        self.pixels_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        row.addWidget(self.pixels_label, 1)

        self._long_edge_row = self._row(layout, None)
        self.long_edge_field = QLineEdit(self)
        ring_on_keyboard_only(self.long_edge_field)
        self.long_edge_field.setFixedHeight(theme.BUTTON_HEIGHT)
        self.long_edge_field.setFixedWidth(theme.DIALOG_LABEL_WIDTH)
        self.long_edge_field.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.long_edge_field.editingFinished.connect(self._changed)
        self._long_edge_row.addWidget(self.long_edge_field)
        unit = QLabel(tr("export.pixels_unit"), self)
        unit.setProperty("role", "secondary")
        self._long_edge_row.addWidget(unit)
        self._long_edge_row.addStretch(1)

    def _fix_size(self, width: int, height: int) -> None:
        """Show a size that cannot be changed, and leave out the shot's details."""
        self.size_choice.set_value(_SIZE_ORIGINAL)
        self.size_choice.hide()
        self.pixels_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.set_output_size(width, height)
        for row in self._metadata_rows:
            self._show_row(row, False)

    def _build_name(self, layout: QVBoxLayout) -> None:
        """The name, with the extension shown but not editable (X-05)."""
        row = self._row(layout, "export.name")
        if self._batch is not None:
            # Each photo keeps its own name; the field holds what follows it.
            stem = QLabel(tr("export.batch_stem"), self)
            stem.setProperty("role", "secondary")
            row.addWidget(stem)
        self.name_field = QLineEdit(self)
        ring_on_keyboard_only(self.name_field)
        self.name_field.setFixedHeight(theme.BUTTON_HEIGHT)
        self.name_field.textEdited.connect(lambda _text: self._changed())
        row.addWidget(self.name_field, 1)
        self.extension_label = QLabel("", self)
        self.extension_label.setProperty("role", "secondary")
        row.addWidget(self.extension_label)

    def _build_location(self, layout: QVBoxLayout) -> None:
        """Where the file goes, and a way to change it (X-05)."""
        row = self._row(layout, "export.location")
        self.folder_label = QLabel("", self)
        self.folder_label.setProperty("role", "secondary")
        row.addWidget(self.folder_label, 1)
        self.change_button = TextButton("export.change", self)
        self.change_button.clicked.connect(self._choose_folder)
        row.addWidget(self.change_button)

    def _build_metadata(self, layout: QVBoxLayout) -> None:
        """The two switches: keep the details, leave the location out (X-06)."""
        row = self._row(layout, "export.metadata")
        keep = QLabel(tr("export.keep_metadata"), self)
        row.addWidget(keep, 1)
        self.keep_metadata_switch = Switch(self)
        self.keep_metadata_switch.toggled.connect(lambda _on: self._changed())
        row.addWidget(self.keep_metadata_switch)

        gps_row = self._row(layout, None)
        gps = QLabel(tr("export.remove_gps"), self)
        gps_row.addWidget(gps, 1)
        self.remove_gps_switch = Switch(self)
        self.remove_gps_switch.toggled.connect(lambda _on: self._changed())
        gps_row.addWidget(self.remove_gps_switch)
        self._metadata_rows = (row, gps_row)

    def _build_carousel(self, layout: QVBoxLayout) -> None:
        """A collage's photos, one by one beside it, for a carousel (K-13)."""
        self._carousel_row = self._row(layout, "export.carousel")
        singles = QLabel(tr("export.carousel_singles"), self)
        self._carousel_row.addWidget(singles, 1)
        self.carousel_switch = Switch(self)
        self.carousel_switch.toggled.connect(lambda _on: self._changed())
        self._carousel_row.addWidget(self.carousel_switch)

    @staticmethod
    def _show_row(row: QHBoxLayout, visible: bool) -> None:
        """Show or hide every widget of *row*, its label included."""
        for index in range(row.count()):
            widget = row.itemAt(index).widget()
            if widget is not None:
                widget.setVisible(visible)

    def _build_footer(self, layout: QVBoxLayout) -> None:
        """The estimate on the left, cancel and export on the right."""
        self.note_label = QLabel("", self)
        self.note_label.setProperty("role", "caption")
        self.note_label.setWordWrap(True)
        layout.addWidget(self.note_label)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACE_SM)
        self.estimate_label = QLabel(tr("export.estimate_unknown"), self)
        self.estimate_label.setProperty("role", "secondary")
        row.addWidget(self.estimate_label)
        row.addStretch(1)
        self.cancel_button = TextButton("panel.action.cancel", self)
        self.cancel_button.clicked.connect(self.reject)
        row.addWidget(self.cancel_button)
        self.export_button = PrimaryButton("export.action", "shortcut.export", self)
        self.export_button.setDefault(True)
        self.export_button.clicked.connect(self.accept)
        row.addWidget(self.export_button)
        layout.addLayout(row)

    @staticmethod
    def _label_of(row: QHBoxLayout) -> QWidget:
        """Return the label a row was built with."""
        item = row.itemAt(0).widget()
        assert item is not None
        return item

    # -- what the dialog holds -------------------------------------------

    @property
    def options(self) -> ExportOptions:
        """Return what has been chosen, ready for the exporter."""
        custom = self.size_choice.value == _SIZE_CUSTOM and not self._collage
        return ExportOptions(
            format=self.format_choice.value,
            quality=round(self.quality_slider.value),
            bits=int(self.depth_choice.value),
            long_edge=self._long_edge() if custom else None,
            keep_metadata=self.keep_metadata_switch.isChecked(),
            remove_gps=self.remove_gps_switch.isChecked(),
        ).sane()

    @property
    def carousel(self) -> bool:
        """Return whether a collage's photos are written one by one as well."""
        return self._collage and self.carousel_switch.isChecked()

    @property
    def folder(self) -> Path:
        """Return the folder the file will be written to."""
        return self._folder

    @property
    def batch(self) -> int | None:
        """Return how many photos the dialog writes, or ``None`` for one."""
        return self._batch

    @property
    def suffix(self) -> str:
        """Return what each photo's name is followed by, for several (X-09)."""
        return self.name_field.text().strip()

    @property
    def target(self) -> Path:
        """Return the path the file will be written to (X-05)."""
        stem = self.name_field.text().strip() or self._stem
        return self._folder / f"{stem}{self.options.extension}"

    @property
    def chosen_settings(self) -> ExportSettings:
        """Return these choices, to be remembered for next time (PLAN.md 3.11)."""
        settings = ExportSettings(**vars(self._settings))
        if self._collage:
            # Only what a collage shares with a photo: the format and its
            # quality or depth. Where a photo goes, what it is named and
            # what size it comes out are the photo's own business.
            options = self.options
            settings.format = options.format
            settings.quality = options.quality
            settings.bits = options.bits
            return settings
        settings.remember(self.options)
        settings.folder = (
            None if self._folder == self._source.parent else str(self._folder)
        )
        if self._batch is not None:
            settings.suffix = self.suffix
            return settings
        stem = self.name_field.text().strip()
        if stem.startswith(self._source.stem):
            settings.suffix = stem[len(self._source.stem) :]
        return settings

    def _long_edge(self) -> int:
        """Return the long edge typed in, or the last sensible value."""
        text = self.long_edge_field.text().strip()
        try:
            return int(text)
        except ValueError:
            return self._settings.long_edge or exporter.LONG_EDGE_MAX

    # -- being told things -----------------------------------------------

    def set_output_size(self, width: int, height: int) -> None:
        """Show how many pixels the exported file will be (PLAN.md 3.11)."""
        self.pixels_label.setText(tr("export.pixels", width=width, height=height))

    def set_estimate(self, size: int) -> None:
        """Show roughly how large the file will be (PLAN.md 3.11)."""
        if size <= 0:
            self.estimate_label.setText(tr("export.estimate_unknown"))
            return
        self.estimate_label.setText(
            tr("export.estimate", size=format_number(size / _BYTES_PER_MB, 1, False))
        )

    def set_copying(self, copying: bool) -> None:
        """Say when the original will be copied rather than written again.

        The quality is not used then, and the file comes out exactly as it
        went in (X-08), which the user would otherwise have no way to know.
        """
        self.note_label.setText(tr("export.copy_note") if copying else "")
        self.note_label.setVisible(copying)

    # -- reacting --------------------------------------------------------

    def _show_settings(self) -> None:
        """Put the remembered choices on screen (PLAN.md 3.11)."""
        settings = self._settings
        self.format_choice.set_value(settings.format)
        self.quality_slider.set_value(float(settings.quality))
        self.depth_choice.set_value(str(settings.bits))
        self.size_choice.set_value(
            _SIZE_CUSTOM if settings.long_edge else _SIZE_ORIGINAL
        )
        self.long_edge_field.setText(str(settings.long_edge or ""))
        if self._collage:
            self.name_field.setText(self._stem)
        elif self._batch is not None:
            self.name_field.setText(settings.suffix)
        else:
            self.name_field.setText(f"{self._source.stem}{settings.suffix}")
        self.keep_metadata_switch.setChecked(settings.keep_metadata)
        self.remove_gps_switch.setChecked(settings.remove_gps)
        self._refresh()

    def _on_format(self, _value: str) -> None:
        """A JPEG has a quality, the others have a depth (PLAN.md 3.11)."""
        self._refresh()

    def _on_size(self) -> None:
        """The long edge is only asked for when a custom size is chosen."""
        self._refresh()
        if self.size_choice.value == _SIZE_CUSTOM:
            self.long_edge_field.setFocus()
            self.long_edge_field.selectAll()

    def _refresh(self) -> None:
        """Show the controls that belong to what is chosen, and ask again."""
        jpeg = self.format_choice.value == exporter.FORMAT_JPEG
        self.quality_slider.setVisible(jpeg)
        self._quality_label.setVisible(jpeg)
        self.depth_choice.setVisible(not jpeg)
        self._depth_label.setVisible(not jpeg)

        custom = self.size_choice.value == _SIZE_CUSTOM and not self._collage
        self._show_row(self._long_edge_row, custom)
        if self._batch is not None:
            # Each photo comes out at its own size, which has no one figure.
            self.pixels_label.setText("" if custom else tr("export.size.each"))

        self.extension_label.setText(self.options.extension)
        self._show_folder()
        self._changed()

    def _show_folder(self) -> None:
        """Show the folder, shortened in the middle if it is long."""
        self.folder_label.setText(
            elide_middle(str(self._folder), self.folder_label, self._path_width())
        )
        self.folder_label.setToolTip(str(self._folder))

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Measure the folder row again once the dialog has its real sizes."""
        super().showEvent(event)
        self._show_folder()


    def _changed(self) -> None:
        """Say that the settings changed, so the figures can be worked out.

        Several photos have no estimate: it would take developing every one
        of them, so the count stands in its place (PLAN.md 3.11).
        """
        self.extension_label.setText(self.options.extension)
        if self._batch is not None:
            self.estimate_label.setText(tr("export.batch_total", count=self._batch))
            return
        self.estimate_label.setText(tr("export.estimate_unknown"))
        self.options_changed.emit(self.options)

    def _path_width(self) -> int:
        """Return how much room the folder path has, in pixels.

        The dialog is a fixed width, so this is known before it is shown:
        what is left of the row once the label column, the button and the
        gaps either side of it are taken out (PLAN.md 3.11).
        """
        return (
            theme.DIALOG_WIDTH_EXPORT
            - 2 * theme.PANEL_PADDING
            - theme.DIALOG_LABEL_WIDTH
            - 2 * theme.SPACE_SM
            - self.change_button.sizeHint().width()
        )

    def _choose_folder(self) -> None:

        """Ask where the file should go (X-05)."""
        chosen = QFileDialog.getExistingDirectory(
            self, tr("export.choose_folder"), str(self._folder)
        )
        if not chosen:
            return
        self._folder = Path(chosen)
        self._refresh()
