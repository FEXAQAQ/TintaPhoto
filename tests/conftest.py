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
