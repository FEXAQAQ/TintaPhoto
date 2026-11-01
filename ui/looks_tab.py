"""The filter tab of the right hand panel: the film looks.

A short list of the film looks that ship with the program and one slider
saying how much of the chosen one to use. Each row shows, at its left, the
photo now open wearing that look, so that the looks can be told apart by eye
rather than by name; the controller develops those small pictures in the
background and they arrive through :meth:`LooksTab.set_thumbnails`. A look
goes on the photo only when it is clicked. Resting the pointer on one does
nothing: a preview that came and went as the pointer crossed the list made
the photo flicker, and the user asked for it gone.

The looks sit in categories, each under a header that names it, in the order
``core.filters.CATEGORIES`` gives; a category with no look installed is not
shown at all. "None", the photo wearing no look, stands above them, and the
strength, which belongs to whichever look is on, is pinned below the list.

The saved presets were in this list for a while; they are slider values
rather than a look, so they now sit at the top of the adjust tab.

A look is one of a set with one taken, so each row is a bordered choice
button, and the current one is marked in the accent colour, the one thing
PLAN.md 3.4 keeps the accent for. Reading the looks from disk is
:mod:`core.filters`; putting one on a photo is the controller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QImage,
    QPainter,
    QPainterPath,
    QPixmap,
    QTransform,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.i18n import tr
from core import edit_state
from core.edit_state import EditState
from core.filters import CATEGORIES, DEFAULT_CATEGORY, NO_FILTER, Look
from ui import theme
from ui.widgets.buttons import ROLE_CHOICE
from ui.widgets.group import ControlGroup, Divider
from ui.widgets.region import styled_background
from ui.widgets.scroll import scrolled
from ui.widgets.slider_row import SliderRow, SliderSpec
from ui.widgets.text import empty_caption

#: The strength slider: a quality that runs up from nothing rather than out
#: from a middle, so it reads without a sign and its track fills from the
#: low end, as the vignette's midpoint does.
STRENGTH_SPEC = SliderSpec(
    name="filter_strength",
    label_key="filters.strength",
    minimum=edit_state.STRENGTH_MIN,
    maximum=edit_state.STRENGTH_MAX,
    default=edit_state.STRENGTH_DEFAULT,
    signed=False,
    fill_from=edit_state.STRENGTH_MIN,
)


def look_name(identifier: str) -> str:
    """Return what to call the look *identifier* on screen.

    The looks are files in a folder, so there can be one the translations
    have never heard of; rather than show a missing key, it is called by its
    own name. Nothing here is written literally: the names of the looks that
    ship are under ``filters.name.`` in both languages (PLAN.md 7.4).
    """
    if not identifier:
        return tr("filters.none")
    key = f"filters.name.{identifier}"
    text = tr(key)
    return identifier.upper() if text == key else text


def _standing_alone(row: QWidget, parent: QWidget) -> QWidget:
    """Return *row* in the padding a row inside a group is given.

    A group hands its controls the panel's side padding, because the line
    above its header has to run the whole width of the panel (PLAN.md 3.1).
    A row standing outside every group has to be given that padding here, or
    it reaches both edges while the rows under a header do not. The space
    below it is carried by the holder, so that the gap goes away whenever
    the row itself does.
    """
    holder = QWidget(parent)
    layout = QHBoxLayout(holder)
    layout.setContentsMargins(
        theme.PANEL_PADDING, 0, theme.PANEL_PADDING, theme.SPACE_SM
    )
    layout.setSpacing(0)
    layout.addWidget(row)
    return holder


def thumbnail_pixmap(image: NDArray[np.uint8] | None, ratio: float) -> QPixmap:
    """Return the picture a row shows at its left.

    The square photo with rounded corners, followed by the gap before the
    name: the gap is part of the picture because a button lays its icon and
    its text side by side with a spacing of its own that the stylesheet
    cannot set. Until the photo arrives the square is a plain patch of the
    track colour, so the rows do not shift when it does.
    """
    side = theme.FILTER_THUMBNAIL_SIZE
    canvas = QPixmap(
        round((side + theme.FILTER_THUMBNAIL_GAP) * ratio), round(side * ratio)
    )
    canvas.setDevicePixelRatio(ratio)
    canvas.fill(Qt.GlobalColor.transparent)
    outline = QPainterPath()
    radius = theme.RADIUS_FILTER_THUMB
    outline.addRoundedRect(QRectF(0, 0, side, side), radius, radius)
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    if image is None:
        painter.fillPath(outline, QColor(theme.TRACK))
    else:
        height, width = image.shape[:2]
        picture = QImage(image.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        pixels = round(side * ratio)
        scaled = QPixmap.fromImage(picture).scaled(
            pixels,
            pixels,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        # Filled through a brush rather than drawn inside a clip, because a
        # clip is not antialiased and the corners would come out jagged.
        brush = QBrush(scaled)
        brush.setTransform(QTransform.fromScale(1 / ratio, 1 / ratio))
        painter.fillPath(outline, brush)
    painter.end()
    return canvas


class FilterRow(QPushButton):
    """One film look: the photo wearing it, its name, click to put it on."""

    def __init__(self, identifier: str, parent: QWidget | None = None) -> None:
        super().__init__(look_name(identifier), parent)
        self.identifier = identifier
        self.setProperty("role", ROLE_CHOICE)
        self.setProperty("thumbnail", True)
        self.setCheckable(True)
        self.setFixedHeight(theme.FILTER_ROW_HEIGHT)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setIconSize(
            QSize(
                theme.FILTER_THUMBNAIL_SIZE + theme.FILTER_THUMBNAIL_GAP,
                theme.FILTER_THUMBNAIL_SIZE,
            )
        )
        self._has_thumbnail = False
        self.set_thumbnail(None)

    def set_thumbnail(self, image: NDArray[np.uint8] | None) -> None:
        """Show *image*, the photo wearing this look, or the empty patch for ``None``."""
        self._has_thumbnail = image is not None
        self.setIcon(QIcon(thumbnail_pixmap(image, self.devicePixelRatioF())))

    def has_thumbnail(self) -> bool:
        """Return whether the photo is shown yet, rather than the empty patch."""
        return self._has_thumbnail

    def retranslate(self) -> None:
        """Read the look's name again after a language change (U-02)."""
        self.setText(look_name(self.identifier))


class LooksTab(QWidget):
    """The list of film looks and the strength of the one in use."""

    #: A look to put on, by its identifier, or "" for none.
    chosen = Signal(str)
    #: A new strength, and whether the slider is still being dragged.
    strength_changed = Signal(float, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: dict[str, FilterRow] = {}
        self._chosen = NO_FILTER
        #: The latest thumbnails, kept so a rebuilt list shows them at once.
        self._thumbnails: Mapping[str, NDArray[np.uint8]] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)


        self._body = QWidget(self)
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        layout.addWidget(self._body, 1)

        self._list = QWidget(self._body)
        list_layout = QVBoxLayout(self._list)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(0)
        list_layout.addSpacing(theme.SPACE_SM)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        # "None" is not a film look and belongs under no category header: it
        # is the row that says the photo is wearing nothing at all, so it
        # stands above the categories rather than inside one of them.
        self._none_row = FilterRow(NO_FILTER, self._list)
        self._none_row.clicked.connect(lambda _checked=False: self._take(NO_FILTER))
        self._group.addButton(self._none_row)
        self._none_holder = _standing_alone(self._none_row, self._list)
        list_layout.addWidget(self._none_holder)

        # One group per category. Each carries the line above its header as
        # well, so that it reads as a section starting rather than as a label
        # dropped over the list. The first also holds the caption that says
        # nothing is installed, because with no looks it is the one shown.
        self._categories: dict[str, ControlGroup] = {}
        for category in CATEGORIES:
            group = ControlGroup(f"filters.group.{category}", parent=self._list)
            self._categories[category] = group
            list_layout.addWidget(group)
        self._none_yet = empty_caption(
            "filters.empty", self._categories[DEFAULT_CATEGORY], padded=False
        )
        self._categories[DEFAULT_CATEGORY].add_row(self._none_yet)
        self._shown: tuple[ControlGroup, ...] = ()
        list_layout.addStretch(1)

        self._scroll = scrolled(self._list, "TabPage", self._body)
        body_layout.addWidget(self._scroll, 1)

        # The strength is the one control here that is not a choice: it says
        # how much of whichever look is on, from whichever category, so it
        # stands outside them all. It is pinned below the list, where it
        # stays in reach however far the list is scrolled.
        body_layout.addWidget(Divider(self._body))
        pinned = QWidget(self._body)
        styled_background(pinned, "TabPage")
        pinned_layout = QVBoxLayout(pinned)
        pinned_layout.setContentsMargins(
            theme.PANEL_PADDING, theme.SPACE_SM, theme.PANEL_PADDING, theme.SPACE_SM
        )
        self._strength = SliderRow(STRENGTH_SPEC)
        self._strength.value_changed.connect(self._on_strength_changed)
        pinned_layout.addWidget(self._strength)
        body_layout.addWidget(pinned)

        self.set_filters(())
        self.set_photo(None)

    # -- contents --------------------------------------------------------

    def set_filters(self, looks: Sequence[Look]) -> None:
        """Show the looks that are installed, with "none" above them."""
        for row in self._rows.values():
            if row is self._none_row:
                continue
            self._group.removeButton(row)
            for group in self._categories.values():
                if group.isAncestorOf(row):
                    group.remove_row(row)
            row.deleteLater()
        self._rows = {}

        held = {look.category for look in looks}
        shown = [
            group
            for category, group in self._categories.items()
            if category in held or (not looks and category == DEFAULT_CATEGORY)
        ]
        for group in self._categories.values():
            group.setVisible(group in shown)
        self._shown = tuple(shown)

        # With nothing installed the tab says so: there is no look to take
        # off, so even "none" would be a choice of one.
        self._none_yet.setVisible(not looks)
        self._none_holder.setVisible(bool(looks))
        if not looks:
            self._show_chosen()
            return
        self._rows[NO_FILTER] = self._none_row
        placed = dict.fromkeys(self._categories, 0)
        for look in looks:
            group = self._categories[look.category]
            index = placed[look.category]
            placed[look.category] += 1
            row = FilterRow(look.identifier, group)
            row.clicked.connect(
                lambda _checked=False, name=look.identifier: self._take(name)
            )
            self._group.addButton(row)
            # Always above the caption that says the folder held nothing,
            # however often the list is rebuilt.
            group.insert_row(index, row)
            self._rows[look.identifier] = row
        self.set_thumbnails(self._thumbnails)
        self._show_chosen()

    def set_photo(self, state: EditState | None) -> None:
        """Show the filter of the photo now open, or the empty state."""
        self._body.setVisible(state is not None)
        if state is None:
            return
        self.set_state(state)

    def set_state(self, state: EditState) -> None:
        """Follow *state* without announcing it, as undo and presets do."""
        self._chosen = state.filter
        self._show_chosen()
        self._strength.blockSignals(True)
        self._strength.set_value(state.filter_strength)
        self._strength.blockSignals(False)

    def set_thumbnails(self, thumbnails: Mapping[str, NDArray[np.uint8]]) -> None:
        """Show the photo wearing each look; a look missing from *thumbnails* waits empty."""
        self._thumbnails = thumbnails
        for identifier, row in self._rows.items():
            row.set_thumbnail(thumbnails.get(identifier))

    def rows(self) -> dict[str, FilterRow]:
        """Return the rows by identifier, for the window and the tests."""
        return dict(self._rows)

    def strength_row(self) -> SliderRow:
        """Return the strength slider."""
        return self._strength

    def scroll_area(self) -> QScrollArea:
        """Return the scroll area the list sits in; the strength is outside it."""
        return self._scroll

    def groups(self) -> tuple[ControlGroup, ...]:
        """Return the category groups being shown, in display order."""
        return self._shown

    # -- reacting --------------------------------------------------------

    def _show_chosen(self) -> None:
        """Mark the look in use, and dim the strength when none is."""
        for identifier, row in self._rows.items():
            row.setChecked(identifier == self._chosen)
        # With no look on, its strength changes nothing, so the slider says
        # so rather than inviting a move that would do nothing (PLAN.md 3.4:
        # every control has a disabled state).
        self._strength.setEnabled(bool(self._chosen))

    def _take(self, identifier: str) -> None:
        """Put the look that was clicked on, and say so."""
        if identifier == self._chosen:
            self._show_chosen()
            return
        self._chosen = identifier
        self._show_chosen()
        self.chosen.emit(identifier)

    def _on_strength_changed(self, _name: str, value: float, dragging: bool) -> None:
        self._set_titles_dimmed(dragging)
        self.strength_changed.emit(value, dragging)

    def _set_titles_dimmed(self, dimmed: bool) -> None:
        """Fade the group titles while the slider is moving (PLAN.md 3.6)."""
        for group in self.groups():
            group.header.set_faded(dimmed)
