"""Text helpers shared by the panels (PLAN.md section 3.5).

Numbers are set in tabular figures so that a value does not jiggle while it
changes, negatives use a real minus sign rather than a hyphen, and long paths
are shortened in the middle rather than cut off at the end.
"""

from __future__ import annotations

from typing import Final

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import QLabel, QWidget

from app.i18n import tr
from ui import theme

#: The proper minus sign, which is as wide as a digit (PLAN.md 3.5).
MINUS_SIGN: Final[str] = "−"
_ELLIPSIS: Final[str] = "…"
_TABULAR_FIGURES: Final[str] = "tnum"
#: The fewest characters a shortened path is left with.
_SHORTEST: Final[int] = 4


class TranslatedLabel(QLabel):
    """A label that reads its own text again after a language change (U-02).

    Used wherever a label shows one interface string: it remembers the key it
    was built from, so the walk of :mod:`ui.retranslate` reaches it without
    the panel around it having to know about it. A label whose text is built
    from a photo's own values is set by its panel instead.
    """

    def __init__(self, key: str, parent: QWidget | None = None, **params: object) -> None:
        super().__init__(tr(key, **params), parent)
        self._key = key
        self._params = params

    def retranslate(self) -> None:
        """Read the text again in the language now in use."""
        self.setText(tr(self._key, **self._params))


def empty_caption(key: str, parent: QWidget, *, padded: bool = True) -> TranslatedLabel:
    """Return the caption a panel shows while there is nothing to show.

    Every such line is said the same way, in the same place (PLAN.md 3.7). The three that hold groups carry their own padding,
    because the lines between the groups have to run the whole width of the
    panel, so the padding belongs to the label rather than to the tab.
    """
    label = TranslatedLabel(key, parent)
    label.setProperty("role", "caption")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setWordWrap(True)
    if padded:
        label.setContentsMargins(
            theme.PANEL_PADDING,
            theme.PANEL_PADDING,
            theme.PANEL_PADDING,
            theme.PANEL_PADDING,
        )
    return label


def use_tabular_figures(widget: QWidget) -> None:
    """Ask *widget* to use tabular figures, where the font offers them."""
    font = widget.font()
    try:
        font.setFeature(QFont.Tag(_TABULAR_FIGURES), 1)
    except (AttributeError, TypeError):
        # Older Qt builds have no font feature API; the fallback font is
        # already tabular in its digits, so nothing is lost.
        return
    widget.setFont(font)


def format_number(value: float, decimals: int = 0, signed: bool = False) -> str:
    """Format a number with a real minus sign, optionally always signed."""
    text = f"{value:+.{decimals}f}" if signed else f"{value:.{decimals}f}"
    return text.replace("-", MINUS_SIGN)


def elide_middle(text: str, widget: QWidget, width: int) -> str:
    """Shorten *text* in the middle so that it fits in *width* pixels.

    As much of the path as will fit is kept, so that a folder two levels
    down is still recognisable; only the middle gives way.
    """
    metrics = QFontMetrics(widget.font())
    if metrics.horizontalAdvance(text) <= width:
        return text

    full = max(1, metrics.horizontalAdvance(text))
    keep = max(_SHORTEST, min(len(text), int(len(text) * width / full)))
    while keep > _SHORTEST and metrics.horizontalAdvance(_middle(text, keep)) > width:
        keep -= 1
    while keep < len(text) and metrics.horizontalAdvance(_middle(text, keep + 1)) <= width:
        keep += 1
    return _middle(text, keep)


def _middle(text: str, keep: int) -> str:
    """Return *text* with its middle replaced by an ellipsis, *keep* long."""
    half = max(1, keep // 2)
    return f"{text[:keep - half]}{_ELLIPSIS}{text[-half:]}"
