# Building TintaPhoto

How to run TintaPhoto from source, test it and package it for Windows.
What the program does and how to use it is in the [README](../README.md);
requirements, interface specification and the image processing pipeline are
in [PLAN.md](PLAN.md), and progress in [PROGRESS.md](PROGRESS.md).

## Requirements

* Windows 10 or 11, 64 bit
* Python 3.12

## Running from source

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python main.py
```

In Command Prompt the activation script is `.venv\Scripts\activate.bat`. Once
the environment exists, it can also be skipped entirely:

```
.venv\Scripts\python.exe main.py
```

Running plain `python main.py` without the virtual environment picks up
whichever Python is first on `PATH`, which does not have the dependencies
installed and fails with `ModuleNotFoundError: No module named 'PySide6'`.

Or double-click `run.bat`, which is that last command and nothing else: it
works from whatever folder it is called in, and if the program stops with an
error it keeps the window open and says where the log is.

## Tests

```powershell
pytest
```

Test photos belong in `tests/samples/`. Large RAW files are not committed to
the repository; see `.gitignore`. Which photos the suite looks for, and what
skips without them, is listed in
[tests/samples/README.md](../tests/samples/README.md).

## The application icon

A drop of ink carrying the whole range of colour, drawn in
`resources/icons/app.svg`. The interface itself uses no gradients
(PLAN.md 3.4); the icon is the one piece of artwork that does. After
changing the drawing, render the file Windows uses:

```powershell
.venv\Scripts\python.exe tools\make_app_icon.py
```

That writes `resources/app.ico` with every size from 16 to 256 px. The
window, the executable and the installer all take their icon from it.

## Packaging

```powershell
.venv\Scripts\python.exe -m PyInstaller tintaphoto.spec --noconfirm
```

The build lands in `dist\TintaPhoto` as a folder, not a single file, because
Qt is used under the LGPL v3 and its libraries stay separate files that can be
replaced. `LICENSE`, `THIRD_PARTY_NOTICES.md` and `licenses` are copied in
beside the executable, so the folder is the release: zip it as it is.

The Qt modules the program never touches - WebEngine, PDF, Quick, Multimedia
and the rest - and the FFmpeg library that comes with OpenCV are left out of
the build, which takes it from about 400 MB to 210 MB.

## Giving it to someone else

Either an installer or a zip; both come out of `dist`. One command builds
everything, in the only order that works - the icon, then the program, then
the installer and the zip around it:

```powershell
.venv\Scripts\python.exe tools\build_release.py
```

`build.bat` is the same thing to double-click. Either way it takes about a
minute and a half and ends by printing what it made. By hand it is these, in
this order; the second and third only wrap what the
first produced, so running them alone is what puts yesterday's program
inside today's installer:

```powershell
.venv\Scripts\python.exe -m PyInstaller tintaphoto.spec --noconfirm
"$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe" tintaphoto.iss
Compress-Archive -Path dist\TintaPhoto -DestinationPath dist\TintaPhoto-<version>-windows-x64.zip -Force
```

The installer is `dist\TintaPhoto-<version>-setup.exe`, about 55 MB. It installs for
one user and asks for no administrator rights, puts the program in the
user's own folder with a Start menu entry and an optional desktop icon, and
leaves an uninstaller that removes the program and keeps the user's
settings and presets. Inno Setup is only needed to build it
(`winget install JRSoftware.InnoSetup`); nothing of it ships with the
program.

The version in those file names is `APP_VERSION` in `app\paths.py`, the
one place it is written; see CLAUDE.md section 11 for when it goes up and
`CHANGELOG.md` for what changed.

The zip is the same folder, for anyone who would rather not install
anything. It comes to about 81 MB, which is too large for most mail, so it goes
through a file link. The other person unpacks it anywhere - including a path
with spaces or Chinese in it - and runs `TintaPhoto.exe`. Nothing is
installed: no Python, no runtime, nothing in the registry.

Because neither the installer nor the program is signed, Windows SmartScreen
says "Windows protected your PC" the first time. **More info** and then
**Run anyway** starts it. Signing needs a code signing certificate, which costs money every
year; until then that warning is part of sharing it.

The program keeps its settings and presets in `%APPDATA%\TintaPhoto` and its
thumbnail cache in `%LOCALAPPDATA%\TintaPhoto`. To remove it, delete the
folder, and those two if they are not wanted either. Photos are only ever
read.
