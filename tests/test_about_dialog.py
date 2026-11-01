"""Tests for the about dialog and the licence files it points at (13.5, 13.6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.paths import APP_VERSION

pytest.importorskip("PySide6.QtWidgets")

from app.i18n import tr  # noqa: E402
from ui.about_dialog import AboutDialog, about_text  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def dialog(qt_app: object) -> AboutDialog:
    return AboutDialog()


def test_it_shows_the_version_the_program_reports(dialog: AboutDialog) -> None:
    assert APP_VERSION in dialog.summary.text()
    assert tr("about.licence") in dialog.summary.text()


def test_the_version_is_written_in_one_place_only(dialog: AboutDialog) -> None:
    """The dialog says the version; the document does not repeat it."""
    source = (PROJECT_ROOT / "resources" / "about.md").read_text(encoding="utf-8")
    assert APP_VERSION not in source
    assert APP_VERSION not in dialog.document.toPlainText()
    assert APP_VERSION in dialog.summary.text()


def test_it_shows_this_project_s_licence_and_the_third_party_one(
    dialog: AboutDialog,
) -> None:
    """PLAN.md 13.6: the licence information can be seen from the program."""
    shown = dialog.document.toPlainText()
    assert "MIT" in shown
    for component in ("PySide6", "LibRaw", "NumPy", "OpenCV", "Pillow"):
        assert component in shown, component


def test_it_carries_the_trademark_statement(dialog: AboutDialog) -> None:
    shown = dialog.document.toPlainText()
    assert "not affiliated" in shown


def test_a_missing_document_still_leaves_a_dialog(
    qt_app: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ui.about_dialog as module

    monkeypatch.setattr(module, "resources_dir", lambda: Path("nowhere at all"))
    assert APP_VERSION in about_text()


def test_the_licences_folder_is_where_the_dialog_says(dialog: AboutDialog) -> None:
    folder = dialog.licences_folder()
    assert folder.is_dir()
    assert (folder / "README.md").is_file()


@pytest.mark.parametrize(
    "name",
    [
        "LGPL-3.0.txt",
        "GPL-3.0.txt",
        "LGPL-2.1.txt",
        "Apache-2.0.txt",
        "MIT-CMU.txt",
        "BSD-3-Clause.txt",
        "MIT.txt",
        "ISC.txt",
    ],
)
def test_every_licence_text_a_component_needs_is_present(name: str) -> None:
    """PLAN.md 13.6: the release carries the full text of each licence."""
    path = PROJECT_ROOT / "licenses" / name
    assert path.is_file(), name
    assert len(path.read_text(encoding="utf-8")) > 500, name


def test_the_notices_name_every_component_that_is_shipped() -> None:
    notices = (PROJECT_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for component in ("PySide6", "rawpy", "LibRaw", "NumPy", "OpenCV", "Pillow", "tifffile", "piexif"):
        assert component in notices, component
    assert "GPL v2 with a packaging exception" in notices


def test_the_notices_have_a_version_for_every_runtime_component() -> None:
    """A dash would mean the audit was never run (PLAN.md 13.5)."""
    notices = (PROJECT_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    runtime = notices.split("## Runtime components")[1].split("##")[0]
    rows = [line for line in runtime.splitlines() if line.startswith("| ") and "---" not in line]
    for row in rows[1:]:
        assert "| — |" not in row, row
