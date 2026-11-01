"""The 300 px right hand panel (PLAN.md 3.1).

The histogram sits at the top, below it the tabs - adjust, filters,
retouch, crop - and each tab has its own fixed action row along the bottom. What is known
about the photo is not a tab: it is a card over the photo (PLAN.md 3.9).

The filter tab holds the film looks; it sits beside the adjustments because
it is the same kind of decision. The saved presets are slider values, so
they are the first group of the adjust tab rather than a tab of their own.
Portrait and lens correction are on the retouch tab: neither is wanted on
every photo, and the adjust tab is the one every photo goes through.
Ctrl+N is the Nth tab.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import Final

from PySide6.QtCore import Property, QSize, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.i18n import tr
from core.edit_state import EditState
from core.filters import Look
from core.geometry import Geometry
from ui import icons, theme
from ui.adjust_tab import AdjustTab
from ui.crop_tab import CropTab
from ui.histogram_widget import HistogramWidget
from ui.retouch_tab import RetouchTab
from ui.looks_tab import LooksTab
from ui.widgets.buttons import ROLE_TAB, TextButton, tooltip
from ui.widgets.region import styled_background
from ui.widgets.scroll import scrolled


class InspectorTab(IntEnum):
    """The tabs of the right hand panel, in display order."""

    ADJUST = 0
    LOOKS = 1
    RETOUCH = 2
    CROP = 3


#: Icon, label and shortcut of each tab. The shortcut is only ever shown:
#: the window is what registers it (PLAN.md section 8).
_TAB_SPECS: Final[tuple[tuple[InspectorTab, str, str, str], ...]] = (
    (InspectorTab.ADJUST, icons.ICON_SLIDERS, "panel.tab.adjust", "shortcut.tab_adjust"),
    (InspectorTab.LOOKS, icons.ICON_FILTERS, "panel.tab.looks", "shortcut.tab_looks"),
    (InspectorTab.RETOUCH, icons.ICON_RETOUCH, "panel.tab.retouch", "shortcut.tab_retouch"),
    (InspectorTab.CROP, icons.ICON_CROP, "panel.tab.crop", "shortcut.tab_crop"),
)

#: The tabs laid out in groups with a line between them; those pad
#: themselves (see ``_build_pages``).
_GROUPED_TABS: Final[frozenset[InspectorTab]] = frozenset(
    {InspectorTab.ADJUST, InspectorTab.RETOUCH, InspectorTab.CROP}
)

#: Buttons of each tab's bottom action row (PLAN.md 3.1).
_ACTION_SPECS: Final[dict[InspectorTab, tuple[str, ...]]] = {
    InspectorTab.ADJUST: ("panel.action.copy", "panel.action.paste", "panel.action.reset"),
    InspectorTab.LOOKS: (),
    InspectorTab.RETOUCH: (),
    InspectorTab.CROP: ("panel.action.reset", "panel.action.cancel", "panel.action.done"),
}


class Inspector(QWidget):
    """The right hand panel: histogram, tab strip, tab page, action row."""

    tab_changed = Signal(InspectorTab)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        styled_background(self, "Inspector")
        self.setFixedWidth(theme.PANEL_WIDTH)

        self._tab_buttons: dict[InspectorTab, QPushButton] = {}
        self._action_buttons: dict[InspectorTab, tuple[TextButton, ...]] = {}
        self._scrolls: dict[InspectorTab, QScrollArea] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_histogram())
        layout.addWidget(self._build_tab_bar())
        self._pages = self._build_pages()
        layout.addWidget(self._pages, 1)
        self._action_rows = self._build_action_rows()
        layout.addWidget(self._action_rows)

        self.set_current_tab(InspectorTab.ADJUST)

    # -- construction ----------------------------------------------------

    def _build_histogram(self) -> QWidget:
        """Build the fixed histogram area at the top of the panel."""
        self.histogram = HistogramWidget(self)
        return self.histogram

    def _build_tab_bar(self) -> QWidget:
        """Build the icon and text tab strip."""
        bar = QWidget(self)
        styled_background(bar, "TabBar")
        bar.setFixedHeight(theme.TAB_BAR_HEIGHT)

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tab_group = QButtonGroup(self)
        self._tab_group.setExclusive(True)
        for tab, icon_name, label_key, shortcut_key in _TAB_SPECS:
            button = QPushButton(tr(label_key), bar)
            button.setProperty("role", ROLE_TAB)
            # The name is on the button; what the tooltip adds is the
            # shortcut, which is otherwise nowhere on screen (PLAN.md 3.8).
            button.setToolTip(tooltip(label_key, shortcut_key))
            button.setIcon(icons.load_icon(icon_name))
            button.setIconSize(QSize(theme.ICON_SIZE, theme.ICON_SIZE))
            button.setCheckable(True)
            button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
            button.clicked.connect(lambda _checked, t=tab: self.set_current_tab(t))
            self._tab_group.addButton(button, int(tab))
            self._tab_buttons[tab] = button
            layout.addWidget(button, 1)
        return bar

    def _build_pages(self) -> QStackedWidget:
        """Build the page of each tab, the long ones in a scroll area."""
        pages = QStackedWidget(self)
        self.adjust_tab = AdjustTab(pages)
        self.looks_tab = LooksTab(pages)
        self.retouch_tab = RetouchTab(pages)
        self.crop_tab = CropTab(pages)
        contents: dict[InspectorTab, QWidget] = {
            InspectorTab.ADJUST: self.adjust_tab,
            InspectorTab.LOOKS: self.looks_tab,
            InspectorTab.RETOUCH: self.retouch_tab,
            InspectorTab.CROP: self.crop_tab,
        }
        for tab, _icon_name, _label_key, _shortcut_key in _TAB_SPECS:
            page = QWidget(pages)
            styled_background(page, "TabPage")
            layout = QVBoxLayout(page)
            # Every tab has groups and pads itself: the lines between the
            # groups run the whole width of the panel, which a page margin
            # would cut.
            layout.setContentsMargins(0, 0, 0, 0)
            if tab in _GROUPED_TABS:
                # With every group open these are taller than the panel, so
                # the page scrolls rather than having its last rows cut off.
                self._scrolls[tab] = scrolled(contents[tab], "TabPage", page)
                layout.addWidget(self._scrolls[tab])
            else:
                # The filter tab scrolls its list itself, because its
                # strength slider stays pinned below the list.
                self._scrolls[tab] = self.looks_tab.scroll_area()
                layout.addWidget(contents[tab])
            pages.addWidget(page)
        return pages

    def set_adjustments(self, state: EditState, shot: tuple[float, float]) -> None:
        """Set the sliders up for a newly opened photo."""
        self.adjust_tab.set_photo(state, shot)
        self.looks_tab.set_photo(state)
        self.retouch_tab.set_photo(state)
        copy, _paste, reset = self.action_buttons(InspectorTab.ADJUST)
        copy.setEnabled(True)
        reset.setEnabled(True)
        # Paste waits until there is something on the clipboard (P-02).

    def set_geometry(self, geometry: Geometry | None) -> None:
        """Set the crop tab, and the retouch tab's lens group, up for the photo now open.

        The lens correction is part of the geometry (PLAN.md 2.2), so it is
        shown from here too: undo, reset and opening a photo all announce it
        this way.
        """
        self.crop_tab.set_photo(geometry)
        if geometry is not None:
            self.retouch_tab.set_lens(geometry.lens)
        for button in self.action_buttons(InspectorTab.CROP):
            button.setEnabled(geometry is not None)

    def set_filters(self, looks: Sequence[Look]) -> None:
        """Give the filter tab the film looks that are installed."""
        self.looks_tab.set_filters(looks)

    def _build_action_rows(self) -> QStackedWidget:
        """Build the bottom action row of each tab."""
        rows = QStackedWidget(self)
        styled_background(rows, "PanelActionRow")
        rows.setFixedHeight(theme.PANEL_ACTION_ROW_HEIGHT)
        for tab, _icon_name, _label_key, _shortcut_key in _TAB_SPECS:
            row = QWidget(rows)
            styled_background(row, "PanelActionRow")
            layout = QHBoxLayout(row)
            layout.setContentsMargins(
                theme.PANEL_PADDING, theme.SPACE_SM, theme.PANEL_PADDING, theme.SPACE_SM
            )
            layout.setSpacing(theme.SPACE_SM)
            buttons: list[TextButton] = []
            for label_key in _ACTION_SPECS[tab]:
                button = TextButton(label_key, row)
                # Stage 0 has nothing to act on yet.
                button.setEnabled(False)
                layout.addWidget(button, 1)
                buttons.append(button)
            if not buttons:
                layout.addStretch(1)
            self._action_buttons[tab] = tuple(buttons)
            rows.addWidget(row)
        return rows

    def retranslate(self) -> None:
        """Read the tab strip again after a language change (U-02).

        The action rows and the pages hold widgets that read themselves
        again; only these buttons are built here.
        """
        for tab, _icon_name, label_key, shortcut_key in _TAB_SPECS:
            button = self._tab_buttons[tab]
            button.setText(tr(label_key))
            button.setToolTip(tooltip(label_key, shortcut_key))

    # -- state -----------------------------------------------------------

    @property
    def current_tab(self) -> InspectorTab:
        """Return the tab currently shown."""
        return InspectorTab(self._pages.currentIndex())

    def set_current_tab(self, tab: InspectorTab) -> None:
        """Show *tab* and mark its button as current."""
        self._tab_buttons[tab].setChecked(True)
        self._pages.setCurrentIndex(int(tab))
        self._action_rows.setCurrentIndex(int(tab))
        self.tab_changed.emit(tab)

    def tab_buttons(self) -> dict[InspectorTab, QPushButton]:
        """Return the tab strip's buttons, for the window and the tests."""
        return dict(self._tab_buttons)

    def action_buttons(self, tab: InspectorTab) -> Sequence[TextButton]:
        """Return the action row buttons of *tab*, in display order."""
        return self._action_buttons[tab]

    def scroll_area(self, tab: InspectorTab) -> QScrollArea:
        """Return the scroll area of *tab*."""
        return self._scrolls[tab]

    # -- collapsing ------------------------------------------------------

    def _get_panel_width(self) -> int:
        return self.maximumWidth()

    def _set_panel_width(self, width: int) -> None:
        self.setFixedWidth(max(0, width))

    #: Animated by the window when the panel is collapsed (PLAN.md 3.6).
    panel_width = Property(int, _get_panel_width, _set_panel_width)
