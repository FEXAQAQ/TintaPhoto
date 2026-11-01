# Third party notices

TintaPhoto is distributed under the MIT licence (see `LICENSE`) and uses the
open source components below. None of them is under the GPL, so the project
itself stays MIT licensed (PLAN.md 13.2). Full licence texts are in the
`licenses` folder.

Versions are those checked with `pip-licenses --format=markdown` on
2026-10-20; this file is updated whenever a dependency is added or upgraded.

## Runtime components

| Component | Version | Licence | Project address |
| --- | --- | --- | --- |
| PySide6 (Qt for Python) | 6.12.0 | LGPL v3 | https://wiki.qt.io/Qt_for_Python |
| Qt 6 (through PySide6) | 6.12.0 | LGPL v3 | https://www.qt.io |
| rawpy | 0.27.1 | MIT | https://pypi.org/project/rawpy/ |
| LibRaw (bundled with rawpy) | 0.22.1 | LGPL 2.1 | https://www.libraw.org |
| NumPy | 2.5.4 | BSD 3-Clause | https://numpy.org |
| OpenCV (`opencv-python-headless`) | 5.0.0.93 | Apache 2.0 | https://opencv.org |
| Pillow | 13.0.0 | MIT-CMU (HPND) | https://python-pillow.org |
| tifffile | 2026.10.15 | BSD 3-Clause | https://github.com/cgohlke/tifffile |
| piexif | 1.1.3 | MIT | https://github.com/hMatoba/Piexif |
| shiboken6 (with PySide6) | 6.12.0 | LGPL v3 | https://wiki.qt.io/Qt_for_Python |

## Development only, not distributed

| Component | Licence | Note |
| --- | --- | --- |
| pytest | MIT | test runner |
| pip-licenses | MIT | licence audit |
| PyInstaller | GPL v2 with a packaging exception | the exception allows the packaged application to keep any licence |
| pyinstaller-hooks-contrib | Apache 2.0 and GPL v2 | part of PyInstaller; used only while packaging |
| Inno Setup | its own permissive licence | builds the installer; none of it is included in what is installed |

The Qt modules that are not used - QtWebEngine, QtPdf, QtQuick, QtMultimedia
and the rest - are left out of the packaged build, so they are neither
distributed nor covered by anything above.

## Qt source code offer

TintaPhoto uses Qt through PySide6 under the terms of the LGPL v3. Qt is used
unmodified and the packaged build keeps the Qt libraries as separate files that
the user can replace. The corresponding source code for Qt and PySide6 is
available from https://download.qt.io and
https://code.qt.io/cgit/pyside/pyside-setup.git/.

## Icons and fonts

* The interface icons in `resources/icons` are drawn for this project and are
  covered by the project's MIT licence. They follow the visual conventions of
  the Lucide icon set (24 px grid, 1.5 px stroke) but contain no Lucide files.
* Latin text and numbers are set in **Inter 4.1**, Copyright (c) 2016 The Inter
  Project Authors, used under the SIL Open Font License 1.1. The two static
  font files in `resources/fonts` (`Inter-Regular.ttf`, `Inter-SemiBold.ttf`)
  were taken unchanged from the official release at
  https://github.com/rsms/inter/releases/tag/v4.1 on 2026-10-21; the full
  licence text, with its copyright line, is in `licenses/OFL-1.1.txt`. The
  font is redistributed, not sold, and carries no reserved font name, so the
  OFL's conditions are met by shipping that text alongside it. Chinese text
  uses Microsoft JhengHei, which is part of Windows and is not redistributed
  here.
* The sRGB ICC profile embedded in exported files is generated at runtime by
  Pillow's `ImageCms`, so no third party profile is shipped.

## Lens database

* Lens correction (PLAN.md 2.2, A-26 to A-28) uses the **Lensfun** lens
  database, (c) the Lensfun project and its contributors, used under the
  Creative Commons Attribution-ShareAlike 3.0 licence (CC BY-SA 3.0). The 56
  XML files in `resources/lensfun` were taken unchanged from
  https://github.com/lensfun/lensfun/tree/master/data/db at commit
  `bc5615d6278cb5fb75dd53e74015367453d5591d` (2026-10-28), downloaded on
  2026-10-30; the full licence text is in `licenses/CC-BY-SA-3.0.txt`. The
  database is shipped as it is and stays under CC BY-SA 3.0; it is data that
  the program reads, not part of the program's own code, which stays MIT.
  Anyone may take the database out, change it and pass it on under the same
  licence.
* Only the data is used. The Lensfun library (LGPL 3) and its tools (GPL 3)
  are not included, and no Lensfun code is copied: the distortion, chromatic
  aberration and vignetting models in `core/lens.py` are implemented from the
  formulas Lensfun publishes in its documentation.

## Face detection model

* Skin enhancement (PLAN.md 2.2, A-29) finds faces with **YuNet**, the face
  detector of the OpenCV model zoo, Copyright (c) 2020 Shiqi Yu, used under
  the MIT licence. The model file `resources/models/face_detection_yunet_2023mar.onnx`
  was taken unchanged from
  https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
  (last changed in commit `f12e12798e8314f7c074a6656816c048dcc95b7a`,
  2023-06-06), downloaded on 2026-10-31, SHA-256
  `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`; its
  licence, with the copyright line, is in `licenses/MIT-YuNet.txt`.
* The model is run by OpenCV's own `FaceDetectorYN`, which is part of the
  OpenCV package already listed above; no other package is added. Which pixels
  are skin, and how the skin is smoothed and brightened, is implemented in
  `core/faces.py` and `core/pipeline.py` from published principles.

## Code provenance

No code is copied from any GPL licensed photo editor. The image processing is implemented from published principles. If a
snippet from an MIT or BSD licensed project is ever adopted, its copyright
notice is kept in the source file and recorded in this file.
