"""Render the application icon from ``resources/icons/app.svg``.

    .venv\\Scripts\\python.exe tools\\make_app_icon.py

Writes ``resources/app.ico`` with every size Windows asks for: the taskbar
and the title bar use the small ones, Explorer and the installer the large
ones. It is a build step rather than something the program does, so it is
run by hand whenever the drawing changes and the result is committed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile  # noqa: E402

from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtSvg import QSvgRenderer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from PIL import Image  # noqa: E402

SOURCE = PROJECT_ROOT / "resources" / "icons" / "app.svg"
TARGET = PROJECT_ROOT / "resources" / "app.ico"

#: What Windows asks for, smallest first.
SIZES: tuple[int, ...] = (16, 24, 32, 48, 64, 128, 256)


def render(svg: bytes, size: int, scratch: Path) -> Image.Image:
    """Return the drawing at *size* pixels square, as a Pillow image."""
    picture = QImage(size, size, QImage.Format.Format_ARGB32)
    picture.fill(Qt.GlobalColor.transparent)
    painter = QPainter(picture)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    QSvgRenderer(svg).render(painter, QRectF(0, 0, size, size))
    painter.end()

    # Through a file rather than a buffer: Qt and Pillow each read what the
    # other wrote, and neither has to know about the other's memory.
    step = scratch / f"{size}.png"
    picture.save(str(step), "PNG")
    with Image.open(step) as drawn:
        return drawn.convert("RGBA")


def main() -> int:
    QApplication.instance() or QApplication([])
    svg = SOURCE.read_bytes()
    with tempfile.TemporaryDirectory() as folder:
        scratch = Path(folder)
        frames = [render(svg, size, scratch) for size in SIZES]
        # Pillow writes every size given into the one file.
        frames[-1].save(TARGET, format="ICO", sizes=[(size, size) for size in SIZES])
    print(f"wrote {TARGET} with {', '.join(str(size) for size in SIZES)} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
