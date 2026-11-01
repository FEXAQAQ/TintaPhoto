"""Build everything that gets handed to someone else, in one go.

    .venv\\Scripts\\python.exe tools\\build_release.py

Four steps, in the only order that works:

1. render ``resources/app.ico`` from the drawing
2. PyInstaller, which fills ``dist/TintaPhoto``
3. Inno Setup, which packs that folder into the installer
4. a zip of the same folder, for anyone who would rather not install

Steps 3 and 4 only wrap what step 2 produced, so running them without it
is what leaves yesterday's program inside today's installer. That is the
whole reason this script exists.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.paths import APP_NAME, APP_VERSION  # noqa: E402

DIST = PROJECT_ROOT / "dist"
BUILT_FOLDER = DIST / APP_NAME
SPEC = PROJECT_ROOT / "tintaphoto.spec"
INSTALLER_SCRIPT = PROJECT_ROOT / "tintaphoto.iss"
ICON_TOOL = PROJECT_ROOT / "tools" / "make_app_icon.py"

INSTALLER = DIST / f"{APP_NAME}-{APP_VERSION}-setup.exe"
ZIP = DIST / f"{APP_NAME}-{APP_VERSION}-windows-x64.zip"

#: Where Inno Setup puts its compiler, per user first since that is where
#: `winget install JRSoftware.InnoSetup` leaves it.
COMPILER_PLACES: tuple[Path, ...] = (
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe",
    Path("C:/Program Files (x86)/Inno Setup 6/ISCC.exe"),
    Path("C:/Program Files/Inno Setup 6/ISCC.exe"),
)


def installer_compiler() -> Path | None:
    """Return Inno Setup's compiler, or ``None`` if it is not installed."""
    for place in COMPILER_PLACES:
        if place.is_file():
            return place
    return None


def installer_version() -> str | None:
    """Return the version the installer script declares."""
    found = re.search(
        r'#define\s+AppVersion\s+"([^"]+)"', INSTALLER_SCRIPT.read_text(encoding="utf-8")
    )
    return found.group(1) if found else None


def run(step: str, command: list[str]) -> float:
    """Run one step, timing it; raise if it fails."""
    print(f"\n== {step} ==")
    started = time.perf_counter()
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"{step} failed with exit code {result.returncode}")
    return time.perf_counter() - started


def make_zip() -> float:
    """Pack the built folder, keeping its name as the folder inside."""
    started = time.perf_counter()
    ZIP.unlink(missing_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(BUILT_FOLDER.rglob("*")):
            if path.is_file():
                archive.write(path, Path(APP_NAME) / path.relative_to(BUILT_FOLDER))
    return time.perf_counter() - started


def describe(path: Path) -> str:
    """Return a file's size and when it was written."""
    if not path.is_file():
        return f"{path.name}: not built"
    status = path.stat()
    when = time.strftime("%H:%M:%S", time.localtime(status.st_mtime))
    size = status.st_size / 1024 / 1024
    return f"{path.name:<36s}{size:5.0f} MB   written {when}"


def main() -> int:
    declared = installer_version()
    if declared != APP_VERSION:
        raise SystemExit(
            f"the installer script says version {declared} but the program says "
            f"{APP_VERSION}; change tintaphoto.iss to match app/paths.py"
        )

    timings: list[tuple[str, float]] = []
    timings.append(("the icon", run("Rendering the icon", [sys.executable, str(ICON_TOOL)])))
    timings.append(
        (
            "the program",
            run(
                "Building the program",
                [sys.executable, "-m", "PyInstaller", str(SPEC), "--noconfirm", "--log-level", "WARN"],
            ),
        )
    )

    compiler = installer_compiler()
    if compiler is None:
        print(
            "\nInno Setup is not installed, so there is no installer this time.\n"
            "  winget install JRSoftware.InnoSetup"
        )
    else:
        timings.append(
            ("the installer", run("Building the installer", [str(compiler), str(INSTALLER_SCRIPT)]))
        )

    print("\n== Packing the zip ==")
    timings.append(("the zip", make_zip()))

    print("\nDone.")
    for what, seconds in timings:
        print(f"  {what:<16s}{seconds:6.1f} s")
    print()
    print(" ", describe(INSTALLER))
    print(" ", describe(ZIP))
    print(f"\nBoth are in {DIST}, which git ignores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
