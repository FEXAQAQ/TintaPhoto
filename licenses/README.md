# Third party licence texts

This folder carries the full licence text of every component distributed with
TintaPhoto (PLAN.md 13.2 and 13.5). The packaged release zip must contain this
folder together with `LICENSE` and `THIRD_PARTY_NOTICES.md`.

| File | Applies to |
| --- | --- |
| `LGPL-3.0.txt` | PySide6 and Qt, used under the LGPL v3 |
| `GPL-3.0.txt` | referred to by the LGPL v3, which is written as an extension of it |
| `LGPL-2.1.txt` | LibRaw, which rawpy bundles |
| `Apache-2.0.txt` | OpenCV |
| `MIT-CMU.txt` | Pillow |
| `BSD-3-Clause.txt` | NumPy, tifffile |
| `MIT.txt` | rawpy, piexif |
| `ISC.txt` | the Lucide icon set, whose visual conventions the drawn icons follow |
| `OFL-1.1.txt` | the Inter font in `resources/fonts` |
| `CC-BY-SA-3.0.txt` | the Lensfun lens database in `resources/lensfun` |
| `MIT-YuNet.txt` | the YuNet face detection model in `resources/models` |

Each text was downloaded verbatim from its official source rather than
retyped, so that no transcription error can creep into a legal document.
`OFL-1.1.txt` is the `LICENSE.txt` of the Inter release itself, so it carries
the font's own copyright line as the licence requires. `CC-BY-SA-3.0.txt` is
the `data/COPYING.CC_BY-SA_3.0` of the Lensfun repository, the copy the
database itself points to. `MIT-YuNet.txt` is the `LICENSE` of the model's
folder in the OpenCV model zoo, which carries the model's copyright line.

Nothing is outstanding.
