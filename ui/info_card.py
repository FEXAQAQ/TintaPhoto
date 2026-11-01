"""The info card laid over the photo (PLAN.md section 3.9).

The toolbar shows only the file name; everything else about the photo is on
this card, out of the way until it is wanted: the info button in the
toolbar, or I, brings it up over the photo's top left corner and takes it
away again, as does Esc. It used to be a tab of the right hand panel, but
it is looked at rather than worked in, so it no longer takes a place beside
the tabs that change the photo.

Two groups, capture and file, under headers that only name them. A row
whose value the photo does not carry is hidden rather than filled with a
dash.
"""

from __future__ import annotations

import subprocess
from typing import Final

from PySide6.QtCore import QEasingCurve, QPoint, Qt, QVariantAnimation
from PySide6.QtWidgets import (
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from app.i18n import tr
from core.metadata import PhotoMetadata
from ui import native, theme
from ui.widgets.group import ControlGroup
from ui.widgets.region import styled_background
from ui.widgets.text import (
    elide_middle,
    format_number,
    use_tabular_figures,
)

#: Width the folder path is shortened to; the rest of the row holds the label.
_PATH_WIDTH: Final[int] = 160
_SEPARATOR: Final[str] = " "  # an em space, wide enough to group values
_BYTES_PER_MB: Final[float] = 1024 * 1024


class InfoRow(QWidget):
    """One line: its name on the left, its value on the right."""

    def __init__(self, label_key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(theme.SLIDER_ROW_HEIGHT)
        self._label_key = label_key

        self._label = QLabel(tr(label_key), self)
        self._label.setProperty("role", "secondary")

        self.value = QLabel("", self)
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        use_tabular_figures(self.value)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_SM)
        layout.addWidget(self._label, 0)
        layout.addStretch(1)
        layout.addWidget(self.value, 0)

    def retranslate(self) -> None:
        """Read the row's name again after a language change (U-02)."""
        self._label.setText(tr(self._label_key))

    def set_value(self, text: str | None) -> None:
        """Show *text*, or hide the whole row when there is nothing to show."""
        self.value.setText(text or "")
        self.setVisible(bool(text))


class InfoCard(QWidget):
    """Everything known about the current photo, on a card over it."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        styled_background(self, "InfoCard")
        self.setFixedWidth(theme.PANEL_WIDTH)
        #: Whether the card has been asked to be up; it may still be fading in.
        self._wanted = False
        self._animation: QVariantAnimation | None = None
        # Switched off at rest, so a card that is up is drawn exactly as it
        # would be with no animation at all.
        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setEnabled(False)
        self.setGraphicsEffect(self._opacity)
        self._metadata: PhotoMetadata | None = None
        #: Whether the photo has a parameter file, so that the row saying so
        #: can be written again in another language.
        self._edited = False

        self._rows: dict[str, InfoRow] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, theme.SPACE_XS, 0, theme.SPACE_SM)
        layout.setSpacing(0)


        layout.addSpacing(theme.SPACE_SM)
        self._capture = self._add_group(layout, "info.group.capture", (
            "info.camera",
            "info.lens",
            "info.exposure",
            "info.focal_length",
            "info.exposure_compensation",
            "info.camera_settings",
            "info.captured_at",
        ), first=True)
        self._file = self._add_group(layout, "info.group.file", (
            "info.dimensions",
            "info.file",
            "info.location",
            "info.edit_state",
        ))

        self._location_row = self._rows["info.location"]
        self._location_row.value.setCursor(Qt.CursorShape.PointingHandCursor)
        self._location_row.value.mouseReleaseEvent = self._open_location  # type: ignore[method-assign]

        self.set_metadata(None)

    def _add_group(
        self,
        layout: QVBoxLayout,
        title_key: str,
        keys: tuple[str, ...],
        first: bool = False,
    ) -> ControlGroup:
        """Add a titled group of rows and return it, so it can be hidden."""
        group = ControlGroup(title_key, first=first, parent=self, foldable=False)
        for key in keys:
            row = InfoRow(key)
            self._rows[key] = row
            group.add_row(row)
        layout.addWidget(group)
        return group

    # -- contents --------------------------------------------------------

    # -- coming and going ----------------------------------------------------

    @property
    def wanted(self) -> bool:
        """Return whether the card has been asked to be up (it may be fading in)."""
        return self._wanted

    @property
    def moving(self) -> bool:
        """Return whether the card is still fading in."""
        return self._animation is not None

    @staticmethod
    def home() -> QPoint:
        """Return where the card rests: the photo's top left corner."""
        return QPoint(theme.CANVAS_MARGIN, theme.CANVAS_MARGIN)

    def set_shown(self, shown: bool) -> None:
        """Bring the card in, or take it away (PLAN.md 3.6, 3.9).

        The same way as the application menu beside the info button: it
        fades in where it rests, and it is gone at once when taken away.
        """
        if shown == self._wanted and shown == self.isVisible():
            return
        self._wanted = shown
        self._stop()
        if not shown or not native.animations_enabled():
            # Taken away, or Windows is set to show no animation: there, or not.
            self._settle(shown)
            return
        self.adjustSize()
        self.move(self.home())
        self._opacity.setEnabled(True)
        self._opacity.setOpacity(0.0)
        self.show()
        self.raise_()
        animation = QVariantAnimation(self)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setDuration(theme.INFO_CARD_FADE_MS)
        # Even, like the menu's own fade, so the two look the same.
        animation.setEasingCurve(QEasingCurve.Type.Linear)
        animation.valueChanged.connect(lambda value: self._opacity.setOpacity(float(value)))
        animation.finished.connect(lambda: self._settle(True))
        animation.start()
        self._animation = animation

    def _stop(self) -> None:
        animation = self._animation
        self._animation = None
        if animation is not None:
            animation.stop()
            animation.deleteLater()

    def _settle(self, shown: bool) -> None:
        """End the fade, the card wholly there or gone."""
        self._stop()
        self._opacity.setOpacity(1.0)
        self._opacity.setEnabled(False)
        self.adjustSize()
        self.move(self.home())
        self.setVisible(shown)
        if shown:
            self.raise_()

    def groups(self) -> tuple[ControlGroup, ...]:
        """Return the two groups, in display order."""
        return (self._capture, self._file)

    def set_metadata(self, info: PhotoMetadata | None) -> None:
        """Show everything known about a photo, or the empty state."""
        self._metadata = info
        self._capture.setVisible(info is not None)
        self._file.setVisible(info is not None)
        if info is None:
            self.adjustSize()
            return

        self._rows["info.camera"].set_value(info.camera)
        self._rows["info.lens"].set_value(info.lens)
        self._rows["info.exposure"].set_value(_exposure(info))
        self._rows["info.focal_length"].set_value(
            tr("info.millimetres", value=format_number(info.focal_length_mm))
            if info.focal_length_mm
            else None
        )
        self._rows["info.exposure_compensation"].set_value(
            tr("info.exposure_value", value=format_number(info.exposure_compensation, 1, True))
            if info.exposure_compensation is not None
            else None
        )
        self._rows["info.camera_settings"].set_value(_camera_settings(info))
        self._rows["info.captured_at"].set_value(
            info.captured_at.strftime("%Y-%m-%d") + _SEPARATOR + info.captured_at.strftime("%H:%M")
            if info.captured_at
            else None
        )

        self._rows["info.dimensions"].set_value(_dimensions(info))
        self._rows["info.file"].set_value(
            f"{format_number(info.file_size / _BYTES_PER_MB, 1)} MB{_SEPARATOR}{info.file_format}"
            if info.file_size
            else info.file_format
        )
        folder = str(info.path.parent)
        self._location_row.set_value(elide_middle(folder, self._location_row.value, _PATH_WIDTH))
        self._location_row.value.setToolTip(folder)
        self.adjustSize()

    def set_edited(self, edited: bool) -> None:
        """Say whether the photo has a parameter file (PLAN.md 3.9)."""
        self._edited = edited
        self._rows["info.edit_state"].set_value(
            tr("info.edited") if edited else tr("info.not_edited")
        )

    def retranslate(self) -> None:
        """Write the values again after a language change (U-02).

        The units, the shutter speed and the words for the white balance and
        the metering are all interface text, so the whole set of values is
        written again from the photo's own metadata rather than only the
        names of the rows.
        """
        self.set_metadata(self._metadata)
        self.set_edited(self._edited)

    def _open_location(self, event) -> None:
        """Show the photo in File Explorer (PLAN.md section 3.9)."""
        if self._metadata is None:
            return
        path = self._metadata.path
        try:
            subprocess.Popen(["explorer", "/select,", str(path)])  # noqa: S603, S607
        except OSError:
            pass


def _exposure(info: PhotoMetadata) -> str | None:
    """Join aperture, shutter and ISO into the one row PLAN.md asks for."""
    parts: list[str] = []
    if info.aperture:
        parts.append(tr("info.aperture", value=format_number(info.aperture, 1)))
    if info.shutter_seconds:
        parts.append(_shutter(info.shutter_seconds))
    if info.iso:
        parts.append(tr("info.iso", value=info.iso))
    return _SEPARATOR.join(parts) if parts else None


def _shutter(seconds: float) -> str:
    """Format a shutter speed the way a camera shows it."""
    if seconds < 1.0:
        return tr("info.shutter_fraction", value=round(1.0 / seconds))
    return tr("info.shutter_seconds", value=format_number(seconds, 0 if seconds >= 10 else 1))


def _camera_settings(info: PhotoMetadata) -> str | None:
    """Join white balance, metering and flash into one row."""
    parts: list[str] = []
    if info.white_balance:
        parts.append(tr(f"info.white_balance.{info.white_balance}"))
    if info.metering:
        parts.append(tr(f"info.metering.{info.metering}"))
    if info.flash_fired is not None:
        parts.append(tr("info.flash.fired" if info.flash_fired else "info.flash.off"))
    return "、".join(parts) if parts else None


def _dimensions(info: PhotoMetadata) -> str | None:
    """Format the pixel size and the megapixel count."""
    if info.width is None or info.height is None:
        return None
    megapixels = info.megapixels or 0.0
    return (
        f"{info.width} × {info.height}"
        f"{_SEPARATOR}{tr('info.megapixels', value=format_number(megapixels, 1))}"
    )
