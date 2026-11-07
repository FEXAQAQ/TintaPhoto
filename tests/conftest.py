"""Shared test fixtures.

Interface tests run on Qt's offscreen platform so that they need no display
and can run unattended.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# As main.py does, before numpy is loaded: the tests run the same threads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


@pytest.fixture(scope="session")
def qt_app() -> Iterator[object]:
    """Return the single QApplication shared by the interface tests."""
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from ui import theme

    application.setStyleSheet(theme.build_qss())
    yield application


@pytest.fixture(autouse=True)
def scratch_profile(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point %APPDATA% and %LOCALAPPDATA% at a throwaway folder for every test.

    Every photo's parameter file is kept under %APPDATA% (PLAN.md 7.1), so a
    test that opens and adjusts a photo must not be able to write into the
    person's own application data. ``app/paths.py`` reads both variables on
    every call, which is what makes this enough.
    """
    scratch = tmp_path_factory.mktemp("profile")
    monkeypatch.setenv("APPDATA", str(scratch / "roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(scratch / "local"))
