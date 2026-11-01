# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build of TintaPhoto (PLAN.md stage 7, 13.2).

    pyinstaller tintaphoto.spec

A folder build rather than one file: PLAN.md 13.2 uses Qt under the LGPL v3,
which asks for the Qt libraries to stay separate files the user can replace.

Everything Qt ships that this program never touches is left out - WebEngine
alone is well over a hundred megabytes - and so is the FFmpeg library that
comes with OpenCV, whose video module is never used. The licence files are
copied next to the executable at the end, so that the release folder is the
release zip (PLAN.md 13.5).
"""

from pathlib import Path
import shutil

PROJECT = Path(SPECPATH)
APP_NAME = "TintaPhoto"

#: What the program reads at runtime, placed where ``app.paths`` looks.
datas = [
    (str(PROJECT / "resources"), "resources"),
    # The development log the menu command shows (app/paths.py).
    (str(PROJECT / "docs" / "DEVLOG.md"), "docs"),
]

#: The icon Windows shows for the executable, drawn in
#: resources/icons/app.svg and rendered by tools/make_app_icon.py.
ICON = str(PROJECT / "resources" / "app.ico")

#: Python modules that are never imported. Qt's own libraries are filtered
#: out below as well, since the hooks collect them whatever is imported.
excludes = [
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQuickWidgets",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.Qt3DCore",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtSql",
    "PySide6.QtTest",
    "PySide6.QtDesigner",
    "PySide6.QtHelp",
    "PySide6.QtUiTools",
    "PySide6.QtBluetooth",
    "PySide6.QtNfc",
    "PySide6.QtSerialPort",
    "PySide6.QtWebSockets",
    "PySide6.QtWebChannel",
    "PySide6.QtPositioning",
    "PySide6.QtRemoteObjects",
    "PySide6.QtSensors",
    "PySide6.QtSpatialAudio",
    "PySide6.QtTextToSpeech",
    "PySide6.QtScxml",
    "PySide6.QtStateMachine",
    # Development only, never part of a release.
    "pytest",
    "_pytest",
    "pip",
    "setuptools",
    "tkinter",
    "unittest",
]

#: Anything whose file name contains one of these is dropped from the build.
#: The Qt modules this program uses are Core, Gui, Widgets and Svg.
UNWANTED = (
    "qt6webengine",
    "qt6pdf",
    "qt6quick",
    "qt6qml",
    "qt6multimedia",
    "qt63d",
    "qt6charts",
    "qt6datavisualization",
    "qt6sql",
    "qt6test",
    "qt6designer",
    "qt6help",
    "qt6uitools",
    "qt6bluetooth",
    "qt6nfc",
    "qt6serialport",
    "qt6websockets",
    "qt6webchannel",
    "qt6positioning",
    "qt6remoteobjects",
    "qt6sensors",
    "qt6spatialaudio",
    "qt6texttospeech",
    "qt6scxml",
    "qt6statemachine",
    "qtwebengine",
    "opengl32sw",
    # OpenCV's video module is never used, and its FFmpeg library is LGPL
    # (PLAN.md 13.2).
    "opencv_videoio_ffmpeg",
)


def wanted(entry):
    """Return whether a collected file belongs in the release."""
    name = Path(entry[0]).name.lower()
    return not any(unwanted in name for unwanted in UNWANTED)


a = Analysis(
    [str(PROJECT / "main.py")],
    pathex=[str(PROJECT)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    excludes=excludes,
    noarchive=False,
)
a.binaries = [entry for entry in a.binaries if wanted(entry)]
a.datas = [entry for entry in a.datas if wanted(entry)]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon=ICON,
)

collected = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)

# The release folder is what gets zipped, so the licences travel with it and
# can be read without starting the program (PLAN.md 13.5, 13.6).
release = Path(DISTPATH) / APP_NAME
for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
    shutil.copy2(PROJECT / name, release / name)
shutil.copytree(PROJECT / "licenses", release / "licenses", dirs_exist_ok=True)
