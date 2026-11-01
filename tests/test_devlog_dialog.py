"""Tests for the development log dialog and the document it shows."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6.QtWidgets")

from app.i18n import tr  # noqa: E402
from app.paths import devlog_file  # noqa: E402
from ui.devlog_dialog import DevlogDialog, devlog_text  # noqa: E402
from ui.widgets.markdown_view import as_markup  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def dialog(qt_app: object) -> DevlogDialog:
    return DevlogDialog()


def test_the_log_is_the_one_in_the_repository() -> None:
    assert devlog_file() == PROJECT_ROOT / "docs" / "DEVLOG.md"
    assert devlog_file().is_file()


def test_it_shows_what_the_log_says(dialog: DevlogDialog) -> None:
    shown = dialog.document.toPlainText()
    newest = devlog_file().read_text(encoding="utf-8").split("## ")[1].splitlines()[0]
    assert newest in shown


def test_it_leaves_out_what_the_file_says_to_the_repository(dialog: DevlogDialog) -> None:
    """The file opens with a title and pointers to the other documents; the
    dialog has its own title, so it starts at the first day."""
    preamble = devlog_file().read_text(encoding="utf-8").split("\n## ")[0]
    assert devlog_text().startswith("## ")
    assert preamble not in devlog_text()
    newest = devlog_text().splitlines()[0].removeprefix("## ")
    assert dialog.document.toPlainText().lstrip().startswith(newest)


def test_the_newest_day_is_at_the_top(dialog: DevlogDialog) -> None:
    """The log is read from the top, so it opens at the top as well."""
    assert dialog.document.verticalScrollBar().value() == 0


def test_a_missing_log_still_leaves_a_dialog(
    qt_app: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ui.devlog_dialog as module

    monkeypatch.setattr(module, "devlog_file", lambda: Path("nowhere at all"))
    assert devlog_text() == tr("devlog.missing")
    assert tr("devlog.missing") in DevlogDialog().document.toPlainText()


def test_a_document_cannot_smuggle_in_markup() -> None:
    """Qt reads inline HTML inside Markdown; the log is text, not markup."""
    assert as_markup("<b>& <br>") == "&lt;b&gt;&amp; &lt;br&gt;"


def test_the_menu_offers_it_and_the_window_can_open_it(qt_app: object) -> None:
    """The log is reached from the application menu, beside about."""
    from app.settings import Settings
    from ui.main_window import MainWindow

    window = MainWindow(Settings())
    try:
        menu = window.toolbar.menu
        action = menu.devlog_action
        assert action.isEnabled()
        assert action.text() == tr("menu.devlog")
        assert menu.actions().index(action) < menu.actions().index(menu.about_action)
        assert callable(window.open_devlog)
    finally:
        window.deleteLater()


def test_the_release_carries_the_log() -> None:
    """The dialog reads docs/DEVLOG.md, so the package has to hold it."""
    spec = (PROJECT_ROOT / "tintaphoto.spec").read_text(encoding="utf-8")
    assert '(str(PROJECT / "docs" / "DEVLOG.md"), "docs")' in spec
