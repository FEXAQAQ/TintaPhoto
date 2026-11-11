"""The right hand panel while the photo is being cut into pieces (PLAN.md 3.17).

Takes the inspector's place, at the same width and in the same style, as the
watermark's panel does: a bar naming the mode, one group for how the photo
is cut and into how many pieces, with the size each piece comes out at, and
an action row along the bottom.

The panel only says what was chosen: the window keeps it in the settings,
and the photo's overlay draws the cuts.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.i18n import tr
from core import split
from core.split import SplitSpec
from ui import theme
from ui.widgets.buttons import TextButton, tooltip
from ui.widgets.group import ControlGroup
from ui.widgets.region import styled_background
from ui.widgets.scroll import scrolled
from ui.widgets.segmented import Segmented
from ui.widgets.text import TranslatedLabel, use_tabular_figures


class SplitPanel(QWidget):
    """How the photo is cut, in the inspector's place."""

    #: The cut the controls now describe.
    spec_changed = Signal(object)
    #: Back to editing.
    leave_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        styled_background(self, "SplitPanel")
        self.setFixedWidth(theme.PANEL_WIDTH)
        self._spec = SplitSpec()
        #: The count side by side or stacked, which a grid's fixed four
        #: leaves alone, so that leaving the grid goes back to it.
        self._line_count = self._spec.count
        #: The size of the finished photo, or ``None`` while there is none.
        self._photo_size: tuple[int, int] | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_title())

        content = QWidget(self)
        body = QVBoxLayout(content)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_cut_group(content))
        body.addStretch(1)
        outer.addWidget(scrolled(content, "TabPage", self), 1)
        outer.addWidget(self._build_action_row())

        self.set_spec(self._spec)

    # -- building --------------------------------------------------------

    def _build_title(self) -> QWidget:
        """The bar along the top, where the tab strip is when editing."""
        bar = QWidget(self)
        styled_background(bar, "TabBar")
        bar.setFixedHeight(theme.TAB_BAR_HEIGHT)
        row = QHBoxLayout(bar)
        row.setContentsMargins(theme.PANEL_PADDING, 0, theme.PANEL_PADDING, 0)
        row.addWidget(TranslatedLabel("split.title", bar))
        row.addStretch(1)
        return bar

    def _labelled(self, group: ControlGroup, label_key: str, control: QWidget) -> list[QWidget]:
        """Add *control* with its name in the fixed column on the left."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACE_SM)
        label = TranslatedLabel(label_key, group)
        label.setProperty("role", "secondary")
        label.setFixedWidth(theme.DIALOG_LABEL_WIDTH)
        row.addWidget(label)
        row.addWidget(control, 1)
        group.add_layout(row)
        return [label, control]

    def _build_cut_group(self, parent: QWidget) -> ControlGroup:
        """The way the photo is cut, into how many, and the size of each piece."""
        group = ControlGroup("split.group.cut", parent=parent)
        self.split_choice = Segmented(
            [
                (split.SPLIT_SIDE_BY_SIDE, "split.split.side_by_side"),
                (split.SPLIT_STACKED, "split.split.stacked"),
                (split.SPLIT_GRID, "split.split.grid"),
            ],
            group,
        )
        self.split_choice.chosen.connect(lambda _value: self._changed())
        self._labelled(group, "split.split", self.split_choice)

        self.count_choice = Segmented(
            [(str(count), f"split.count.{count}") for count in split.COUNTS], group
        )
        self.count_choice.chosen.connect(lambda _value: self._changed())
        self._count_row = self._labelled(group, "split.count", self.count_choice)

        self.size_label = QLabel("", group)
        self.size_label.setProperty("role", "caption")
        self.size_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        use_tabular_figures(self.size_label)
        group.add_row(self.size_label)

        hint = TranslatedLabel("split.hint", group)
        hint.setProperty("role", "caption")
        hint.setWordWrap(True)
        group.add_row(hint)
        return group

    def _build_action_row(self) -> QWidget:
        """The row along the bottom, as on every tab of the inspector."""
        row = QWidget(self)
        styled_background(row, "PanelActionRow")
        row.setFixedHeight(theme.PANEL_ACTION_ROW_HEIGHT)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(
            theme.PANEL_PADDING, theme.SPACE_SM, theme.PANEL_PADDING, theme.SPACE_SM
        )
        layout.setSpacing(theme.SPACE_SM)
        self.leave_button = TextButton("split.leave", row)
        self.leave_button.setToolTip(tooltip("split.leave", "shortcut.split"))
        self.leave_button.clicked.connect(self.leave_requested.emit)
        layout.addWidget(self.leave_button, 1)
        return row

    # -- being told things -----------------------------------------------

    @property
    def spec(self) -> SplitSpec:
        """Return the cut the controls describe."""
        return self._spec

    def set_spec(self, spec: SplitSpec, line_count: int | None = None) -> None:
        """Show *spec* without announcing it.

        *line_count* is the count to go back to when a grid is left; left
        out, the count shown before is kept.
        """
        spec = spec.sane()
        self._spec = spec
        grid = spec.split == split.SPLIT_GRID
        if line_count is not None:
            self._line_count = SplitSpec(split.SPLIT_STACKED, line_count).sane().count
        if not grid:
            self._line_count = spec.count
        self.split_choice.set_value(spec.split)
        self.count_choice.set_value(str(spec.count))
        # A grid is always four: the count shows it but cannot be changed.
        for widget in self._count_row:
            widget.setEnabled(not grid)
        self._show_size()

    def set_photo_size(self, size: tuple[int, int] | None) -> None:
        """Say how large each piece of a photo of *size* comes out."""
        self._photo_size = size
        self._show_size()

    def retranslate(self) -> None:
        """Read what is not a translated label again (U-02)."""
        self.leave_button.setToolTip(tooltip("split.leave", "shortcut.split"))
        self._show_size()

    # -- reacting --------------------------------------------------------

    def _show_size(self) -> None:
        size = self._photo_size
        if size is None or not split.can_cut(size, self._spec):
            self.size_label.setText("")
            return
        width, height = split.piece_size(size, self._spec)
        self.size_label.setText(
            tr(
                "split.size",
                size=tr("export.pixels", width=width, height=height),
                count=self._spec.pieces,
            )
        )

    def _changed(self) -> None:
        """Take on what the controls say and announce it.

        Leaving the grid goes back to the count chosen before it.
        """
        chosen = self.split_choice.value
        if chosen == split.SPLIT_GRID:
            count = split.GRID_COUNT
        elif self._spec.split == split.SPLIT_GRID:
            count = self._line_count
        else:
            count = int(self.count_choice.value)
        self.set_spec(SplitSpec(chosen, count))
        self.spec_changed.emit(self._spec)
