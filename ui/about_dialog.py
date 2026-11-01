"""The about dialog (PLAN.md 13.5).

Shows the version, this project's own licence and the third party licences,
which is what the plan asks the program to be able to show. The text itself
is ``resources/about.md``, kept as one document so that what the dialog
shows and what the release ships cannot drift apart; it is a legal notice
and stays in English, while the dialog's own controls are translated.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.i18n import tr
from app.paths import APP_NAME, APP_VERSION, distribution_dir, resources_dir
from ui import native, theme
from ui.widgets.buttons import PrimaryButton, TextButton
from ui.widgets.dialog import Dialog
from ui.widgets.markdown_view import MarkdownView

#: What the version line in the document is written as, so that the version
#: has one definition (app/paths.py) rather than two.
_VERSION_PLACEHOLDER = "{version}"


def about_text() -> str:
    """Return the about document, with the version filled in.

    Never raises: a missing document leaves the dialog with the name and
    the version, which is better than refusing to open.
    """
    path = resources_dir() / "about.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return f"# {APP_NAME}\n\nVersion {APP_VERSION}\n"
    return text.replace(_VERSION_PLACEHOLDER, APP_VERSION)


class AboutDialog(Dialog):
    """Version, licence and the third party notices."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("menu.about"))
        self.setModal(True)
        self.setFixedWidth(theme.DIALOG_WIDTH_SETTINGS)
        self.setMinimumHeight(theme.DIALOG_HEIGHT_SETTINGS)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            theme.PANEL_PADDING, theme.PANEL_PADDING, theme.PANEL_PADDING, theme.PANEL_PADDING
        )
        layout.setSpacing(theme.SPACE_MD)

        title = QLabel(APP_NAME, self)
        title.setProperty("role", "title")
        layout.addWidget(title)

        self.summary = QLabel(
            tr("about.summary", version=APP_VERSION, licence=tr("about.licence")), self
        )
        self.summary.setProperty("role", "secondary")
        layout.addWidget(self.summary)

        self.document = MarkdownView(self)
        self.document.set_document_text(about_text())
        layout.addWidget(self.document, 1)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(theme.SPACE_SM)
        self.licences_button = TextButton("about.show_licences", self)
        self.licences_button.clicked.connect(self._show_licences)
        buttons.addWidget(self.licences_button)
        buttons.addStretch(1)
        self.close_button = PrimaryButton("about.close", parent=self)
        self.close_button.setDefault(True)
        self.close_button.clicked.connect(self.accept)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

    def _show_licences(self) -> None:
        """Open the folder holding the full licence texts (PLAN.md 13.5)."""
        folder = self.licences_folder()
        if folder.is_dir():
            native.reveal(folder / "README.md")

    @staticmethod
    def licences_folder() -> Path:
        """Return where the licence texts are, beside the program."""
        return distribution_dir() / "licenses"

