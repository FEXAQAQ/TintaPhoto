"""A read-only view for the Markdown documents the dialogs show.

The about dialog and the development log both put a document on screen, so
they show it through the same widget: one place decides how a document is
escaped, coloured and framed, and the two dialogs cannot drift apart
(PLAN.md 3.5).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QTextBrowser, QWidget

from ui import theme


def as_markup(text: str) -> str:
    """Return *text* with anything that looks like a tag made harmless.

    Qt reads inline HTML inside Markdown, so a copyright line that still
    holds a ``<NAME>`` placeholder would be taken for an unclosed tag and
    swallow the rest of the document.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class MarkdownView(QTextBrowser):
    """Shows a Markdown document, in the theme's colours, at its top."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setOpenExternalLinks(False)
        self.setOpenLinks(False)
        self.setFrameShape(QTextBrowser.Shape.NoFrame)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def set_document_text(self, text: str) -> None:
        """Show *text*, which is read as Markdown and never as HTML."""
        self.setMarkdown(as_markup(text))
        self._recolour(theme.TEXT_PRIMARY)

    def _recolour(self, colour: str) -> None:
        """Set the colour of every character of the document.

        Qt sets no colour of its own when it reads Markdown, and the
        document inside the widget does not inherit the one from the style
        sheet, so it is set here - from the theme, like every other colour
        (PLAN.md 3.5).
        """
        cursor = self.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        style = QTextCharFormat()
        style.setForeground(QColor(colour))
        cursor.mergeCharFormat(style)
        # Back to the beginning, or the document is shown scrolled to its end.
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        self.setTextCursor(cursor)
        self.verticalScrollBar().setValue(0)
