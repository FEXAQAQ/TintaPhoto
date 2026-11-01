"""A collapsible group of controls in the right hand panel (PLAN.md 3.1).

The header is the whole width of the panel and reads as the start of a
section: a line above it, an arrow that says whether the group is open, and
the group's name heavier and brighter than the names of the controls beneath
it. Without that the groups are only three faint lines between eight
sliders, and one group looks like the next.
"""

from __future__ import annotations

from typing import Final

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QSize, Qt, Signal
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from app.i18n import tr
from ui import icons, native, theme
from ui.widgets.region import set_state, styled_background

#: A group's header; the stylesheet gives it its size and its three states.
ROLE_GROUP = "group"

#: Qt's own "no maximum", which the body goes back to once it has finished
#: folding; leaving it at the height it was animated to would stop it
#: growing again when a row inside it changes.
_NO_MAXIMUM: Final[int] = 16777215


class Divider(QWidget):
    """A one pixel line across the panel, between two groups."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        styled_background(self, "Divider")
        self.setFixedHeight(theme.BORDER_WIDTH)


class GroupHeader(QPushButton):
    """The title row of a group, which opens and closes it when pressed."""

    def __init__(self, label_key: str, parent: QWidget | None = None) -> None:
        super().__init__(tr(label_key), parent)
        self._label_key = label_key
        self.setProperty("role", ROLE_GROUP)
        self.setCheckable(True)
        self.setChecked(True)
        self.setFixedHeight(theme.GROUP_HEADER_HEIGHT)
        self.setIconSize(QSize(theme.ICON_SIZE_SMALL, theme.ICON_SIZE_SMALL))
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.toggled.connect(self._show_whether_open)
        self._show_whether_open(True)
        #: A button that sits at the right end of the header, such as the
        #: white balance eyedropper (A-24).
        self._action: QWidget | None = None
        #: Whether the header only names its group and does not fold it.
        self._fixed = False

    def set_action(self, button: QWidget) -> None:
        """Put *button* at the right end of the header, inside its band.

        It is a child of the header rather than its neighbour, so the band
        still runs the whole width and lights up as one; a press on the
        button is the button's and does not fold the group.
        """
        button.setParent(self)
        self._action = button
        self._place_action()
        button.show()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Keep the button at the right end as the panel changes width."""
        super().resizeEvent(event)
        self._place_action()

    def _place_action(self) -> None:
        """Line the button's icon up with the right edge of the controls."""
        if self._action is None:
            return
        inset = (theme.ICON_BUTTON_SIZE - theme.ICON_SIZE) // 2
        size = self._action.size()
        self._action.move(
            self.width() - theme.PANEL_PADDING + inset - size.width(),
            (self.height() - size.height()) // 2,
        )

    def _show_whether_open(self, open_: bool) -> None:
        """Point the arrow down while the group is open, right while closed."""
        self.setIcon(
            icons.load_icon(
                icons.ICON_CHEVRON_DOWN if open_ else icons.ICON_CHEVRON_RIGHT,
                size=theme.ICON_SIZE_SMALL,
            )
        )
        self.setToolTip(tr("group.collapse" if open_ else "group.expand"))

    def retranslate(self) -> None:
        """Read the name and the tooltip again after a language change."""
        self.setText(tr(self._label_key))
        if not self._fixed:
            self._show_whether_open(self.isChecked())

    def set_faded(self, faded: bool) -> None:
        """Fade the title while a slider is being dragged (PLAN.md 3.6)."""
        set_state(self, "faded", faded)

    def make_fixed(self) -> None:
        """Let the header only name its group: no arrow, nothing to press.

        For a group that is read rather than worked in, such as the info
        card's (PLAN.md 3.9), where folding half of a card away saves nothing.
        """
        self.setCheckable(False)
        self.setIcon(icons.load_icon(icons.ICON_CHEVRON_DOWN, size=0))
        self.setIconSize(QSize(0, 0))
        self.setToolTip("")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._fixed = True


class ControlGroup(QWidget):
    """A header with its controls under it, which the header folds away."""

    #: Whether the group is open now.
    toggled = Signal(bool)

    def __init__(
        self,
        label_key: str,
        first: bool = False,
        parent: QWidget | None = None,
        foldable: bool = True,
    ) -> None:
        super().__init__(parent)
        self.header = GroupHeader(label_key, self)
        if not foldable:
            self.header.make_fixed()
        self.header.toggled.connect(self._on_toggled)
        self._fold_animation: QPropertyAnimation | None = None

        self._body = QWidget(self)
        self._body_layout = QVBoxLayout(self._body)
        # The controls keep the panel's own side padding; the header and the
        # line above it run the whole width, so that a section reads as a
        # band across the panel rather than as an indented block. The space
        # below the last control is the 16 px PLAN.md 3.5 puts between one
        # group and the next.
        self._body_layout.setContentsMargins(
            theme.PANEL_PADDING, theme.SPACE_XS, theme.PANEL_PADDING, theme.GROUP_GAP
        )
        # A little air between the rows: eight sliders at 28 px with
        # nothing between them read as one block of lines.
        self._body_layout.setSpacing(theme.SPACE_XS)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        # The first group needs no line: the tab bar's own underline is
        # already the edge above it.
        if not first:
            layout.addWidget(Divider(self))
        layout.addWidget(self.header)
        layout.addWidget(self._body)

    def add_row(self, widget: QWidget) -> None:
        """Put one control at the bottom of the group."""
        widget.setParent(self._body)
        self._body_layout.addWidget(widget)

    def insert_row(self, index: int, widget: QWidget) -> None:
        """Put one control at *index* among those already in the group."""
        widget.setParent(self._body)
        self._body_layout.insertWidget(index, widget)

    def remove_row(self, widget: QWidget) -> None:
        """Take one control out again, for a group built from a folder."""
        self._body_layout.removeWidget(widget)
        widget.setParent(None)

    def add_layout(self, layout) -> None:
        """Put a whole row or grid of controls at the bottom of the group."""
        self._body_layout.addLayout(layout)

    @property
    def is_open(self) -> bool:
        """Return whether the controls are showing.

        A group whose header only names it is always open.
        """
        return self.header.isChecked() or not self.header.isCheckable()

    @property
    def folding(self) -> bool:
        """Return whether the group is still opening or closing."""
        return self._fold_animation is not None

    def _on_toggled(self, open_: bool) -> None:
        """Fold the group over the time PLAN.md 3.6 gives a panel."""
        self._stop_folding()
        if not native.animations_enabled():
            # Windows is set to show no animation, so the group simply is
            # open or closed (PLAN.md 3.6).
            self._body.setMaximumHeight(_NO_MAXIMUM)
            self._body.setVisible(open_)
            self.toggled.emit(open_)
            return

        if open_:
            self._body.setMaximumHeight(0)
            self._body.setVisible(True)
        animation = QPropertyAnimation(self._body, b"maximumHeight", self)
        animation.setDuration(theme.DURATION_PANEL_MS)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.setEndValue(self._body.sizeHint().height() if open_ else 0)
        animation.finished.connect(lambda: self._finish_folding(open_))
        animation.start()
        self._fold_animation = animation
        self.toggled.emit(open_)

    def _finish_folding(self, open_: bool) -> None:
        """Leave the body shown or hidden, and free to size itself again."""
        self._stop_folding()
        self._body.setVisible(open_)
        self._body.setMaximumHeight(_NO_MAXIMUM)

    def _stop_folding(self) -> None:
        """Stop a fold that is still running, and let it go.

        Held and dropped here rather than handed to Qt to delete, because a
        reference left pointing at a deleted animation is what once broke
        the zoom (see ``PhotoCanvas._stop_animation``).
        """
        animation = self._fold_animation
        self._fold_animation = None
        if animation is not None:
            animation.stop()
            animation.deleteLater()
