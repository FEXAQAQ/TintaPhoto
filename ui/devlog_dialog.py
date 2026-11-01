"""The development log dialog.

Shows ``docs/DEVLOG.md``, the short day by day record of what was built, as
a menu command beside the about dialog. The log is kept as one document so
that what the dialog shows and what the repository holds cannot drift
apart; it is written in Chinese, while the dialog's own controls are
translated.
"""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.i18n import tr
from app.paths import devlog_file
from ui import theme
from ui.widgets.buttons import PrimaryButton
from ui.widgets.dialog import Dialog
from ui.widgets.markdown_view import MarkdownView


#: Where the days begin. What stands above it in the file is written for
#: whoever opens the repository - the title, and where the other documents
#: are - and the dialog says that much itself.
_FIRST_DAY = "\n## "


def devlog_text() -> str:
    """Return the development log, from its first day onwards.

    Never raises: a missing document leaves the dialog with a line saying
    so, which is better than refusing to open.
    """
    try:
        text = devlog_file().read_text(encoding="utf-8")
    except OSError:
        return tr("devlog.missing")
    _, separator, days = text.partition(_FIRST_DAY)
    return separator.lstrip("\n") + days if separator else text


class DevlogDialog(Dialog):
    """What has been built, newest first."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("menu.devlog"))
        self.setModal(True)
        self.setFixedWidth(theme.DIALOG_WIDTH_SETTINGS)
        self.setMinimumHeight(theme.DIALOG_HEIGHT_SETTINGS)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            theme.PANEL_PADDING, theme.PANEL_PADDING, theme.PANEL_PADDING, theme.PANEL_PADDING
        )
        layout.setSpacing(theme.SPACE_MD)

        title = QLabel(tr("menu.devlog"), self)
        title.setProperty("role", "title")
        layout.addWidget(title)

        self.summary = QLabel(tr("devlog.summary"), self)
        self.summary.setProperty("role", "secondary")
        layout.addWidget(self.summary)

        self.document = MarkdownView(self)
        self.document.set_document_text(devlog_text())
        layout.addWidget(self.document, 1)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(theme.SPACE_SM)
        buttons.addStretch(1)
        self.close_button = PrimaryButton("devlog.close", parent=self)
        self.close_button.setDefault(True)
        self.close_button.clicked.connect(self.accept)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)
