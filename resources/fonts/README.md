# Fonts

Latin text and numbers use **Inter** (SIL Open Font License 1.1); Chinese text
uses Microsoft JhengHei, which ships with Windows and is therefore not included
here (PLAN.md 13.2).

Two static files are bundled, taken unchanged from the official Inter 4.1
release (https://github.com/rsms/inter/releases/tag/v4.1), from `extras/ttf`
in its zip:

```
resources/fonts/Inter-Regular.ttf     weight 400
resources/fonts/Inter-SemiBold.ttf    weight 600
```

Those are the two weights `ui/theme.py` asks for; both register under the one
family name `Inter`, which is first in the font stack, so the interface picks
them up with no further wiring. `main.py` registers every `.ttf`/`.ttc` in
this folder at start-up and carries on without them if they are missing, in
which case the interface falls back to Segoe UI.

The licence is recorded in `THIRD_PARTY_NOTICES.md` and its full text is in
`licenses/OFL-1.1.txt`. Adding a weight means adding its file here; nothing
else has to change.
