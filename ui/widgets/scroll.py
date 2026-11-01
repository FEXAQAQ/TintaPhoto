"""The scroll area of a right hand panel tab (PLAN.md 3.1 and 3.3).

With every group open the controls of a tab are taller than the panel, and
without a scroll area Qt cuts the last rows off rather than shrinking rows
that are a fixed height: they are neither visible nor reachable. PLAN.md 3.3
already has the wheel scrolling over the panel, and 3.4 has the slider
ignoring it so that scrolling cannot nudge a photo.

The panel is a fixed 300 px wide, so a bar that came and went would move
every control eight pixels sideways each time a group was folded away. The
gutter is reserved at all times instead, and only the handle in it comes and
goes.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea, QScrollBar, QWidget

from ui.widgets.region import set_state, styled_background


def scrolled(content: QWidget, region: str, parent: QWidget | None = None) -> QScrollArea:
    """Return *content* in a panel scroll area, painted as *region*."""
    area = QScrollArea(parent)
    styled_background(area, region)
    area.setWidget(content)
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    # Always on, so that the width of the controls never depends on whether
    # the tab happens to be taller than the panel at this moment.
    area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    # Tab moves between the controls, not onto the area holding them; Qt
    # scrolls a control into view by itself once the focus reaches it.
    area.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    bar = area.verticalScrollBar()
    bar.rangeChanged.connect(lambda _low, _high: _mark_whether_empty(bar))
    _mark_whether_empty(bar)
    return area


def _mark_whether_empty(bar: QScrollBar) -> None:
    """Let the stylesheet hide the handle while there is nothing to scroll.

    Qt leaves an empty scroll bar enabled and draws its handle down the
    whole track, which says there is more to see when there is not. Folding
    a group changes the range on every frame of the animation, so the
    restyle is skipped unless the state really changed.
    """
    empty = bar.minimum() == bar.maximum()
    if bar.property("empty") != ("true" if empty else "false"):
        set_state(bar, "empty", empty)
