"""TintaPhoto entry point.

Loads the settings, registers the bundled font, applies the generated dark
theme and shows the main window. Any unexpected error during start-up is
logged and reported in one plain sentence rather than as a traceback.
"""

from __future__ import annotations

import logging
import sys

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication, QMessageBox

from app.i18n import init_translator, tr
from app.paths import APP_NAME, APP_VERSION, resources_dir, setup_logging
from app.settings import Settings
from ui import icons, theme
from ui.main_window import MainWindow

logger = logging.getLogger(__name__)


def _register_bundled_fonts() -> None:
    """Register any font files shipped in ``resources/fonts``.

    The interface falls back to the system UI font when no font file is
    present, so a missing font never prevents start-up (PLAN.md 3.5).
    """
    fonts_dir = resources_dir() / "fonts"
    if not fonts_dir.is_dir():
        return
    for path in sorted(fonts_dir.glob("*.tt[fc]")):
        if QFontDatabase.addApplicationFont(str(path)) == -1:
            logger.warning("Could not register the font %s", path.name)


def main() -> int:
    """Run the application and return its exit code."""
    setup_logging()
    logger.info("Starting %s %s", APP_NAME, APP_VERSION)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(APP_NAME)

    settings = Settings.load()
    init_translator(settings.language)
    _register_bundled_fonts()
    app.setWindowIcon(icons.app_icon())
    app.setStyleSheet(theme.build_qss())


    try:
        window = MainWindow(settings)
    except Exception:
        logger.exception("The main window could not be created")
        QMessageBox.critical(None, tr("app.name"), tr("error.startup_failed"))
        return 1

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
