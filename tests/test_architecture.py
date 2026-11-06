"""Guards for the rules in PLAN.md sections 5, 3.5 and 7.4.

These checks are cheap and catch the mistakes that are easiest to make later:
an interface import leaking into ``core``, a literal colour or pixel value
outside ``ui/theme.py``, or an interface string key that has no translation.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_IN_CORE = ("PySide6", "shiboken6")

#: The controller layer runs the background threads and talks to the interface
#: through signals, so it uses Qt's core classes. What it must not do is build
#: or touch an interface itself.
WIDGET_MODULES = ("PySide6.QtWidgets", "PySide6.QtGui", "PySide6.QtSvg")

_HEX_COLOUR = re.compile(r"#[0-9A-Fa-f]{3,8}\b")
#: The newest release heading in CHANGELOG.md, as in "## 0.1.0 - 2026-10-24".
_CHANGELOG_VERSION = re.compile(r"^## (\d+\.\d+\.\d+)", re.MULTILINE)
#: The newest day in docs/DEVLOG.md, as in "## 2026-10-24".
_DEVLOG_DAY = re.compile(r"^## (\d{4}-\d{2}-\d{2})$", re.MULTILINE)


def _python_files(package: str) -> list[Path]:
    return sorted((PROJECT_ROOT / package).rglob("*.py"))


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    return modules


def _imported_module_paths(path: Path) -> set[str]:
    """Return the full dotted names of everything a file imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def test_core_does_not_import_any_interface_package() -> None:
    for path in _python_files("core"):
        modules = _imported_modules(path)
        for forbidden in FORBIDDEN_IN_CORE:
            assert forbidden not in modules, f"{path.name} imports {forbidden}"


def test_core_does_not_import_the_interface_or_controller_layer() -> None:
    for path in _python_files("core"):
        modules = _imported_modules(path)
        assert "ui" not in modules, f"{path.name} imports ui"
        assert "app" not in modules, f"{path.name} imports app"


def test_controller_layer_does_not_import_the_interface_layer() -> None:
    for path in _python_files("app"):
        assert "ui" not in _imported_modules(path), f"{path.name} imports ui"


def test_controller_layer_does_not_build_an_interface() -> None:
    """Widgets belong to ui/; the controller only sends it signals."""
    for path in _python_files("app"):
        imported = _imported_module_paths(path)
        for forbidden in WIDGET_MODULES:
            assert forbidden not in imported, f"{path.name} imports {forbidden}"


def test_only_theme_contains_literal_colours() -> None:
    offenders: list[str] = []
    for package in ("app", "core", "ui"):
        for path in _python_files(package):
            if path.name == "theme.py":
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if _HEX_COLOUR.search(line):
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{number}")
    assert not offenders, f"literal colours outside ui/theme.py: {offenders}"


def _translation_keys() -> set[str]:
    path = PROJECT_ROOT / "resources" / "i18n" / "zh_TW.json"
    with path.open(encoding="utf-8") as handle:
        return set(json.load(handle))


def _tr_keys_used() -> dict[str, str]:
    """Return every literal key passed to ``tr()``, mapped to its file."""
    used: dict[str, str] = {}
    for package in ("app", "core", "ui"):
        for path in _python_files(package):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.id if isinstance(node.func, ast.Name) else None
                if name != "tr" or not node.args:
                    continue
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    used[first.value] = str(path.relative_to(PROJECT_ROOT))
    return used


def test_every_tr_key_used_in_the_code_has_a_translation() -> None:
    available = _translation_keys()
    missing = {key: where for key, where in _tr_keys_used().items() if key not in available}
    assert not missing, f"interface strings without a translation: {missing}"


def _all_source_text() -> str:
    """Return every line of Python in the project, as one piece of text."""
    paths = [PROJECT_ROOT / "main.py"]
    for package in ("app", "core", "ui", "tools"):
        paths += _python_files(package)
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def test_no_translation_is_left_behind_unused() -> None:
    """A string nothing asks for is dead weight in both language files.

    Some keys are put together as they are needed - ``f"info.metering.{mode}"``
    - so a key also counts as asked for when the part before its last dot
    appears in the code.
    """
    source = _all_source_text()
    orphans = []
    for key in sorted(_translation_keys()):
        if key in source:
            continue
        if any(
            f"{key.rsplit('.', cut)[0]}." in source
            for cut in range(1, key.count(".") + 1)
        ):
            continue
        orphans.append(key)
    assert not orphans, f"translations nothing asks for: {orphans}"


def test_main_entry_point_exists() -> None:
    assert (PROJECT_ROOT / "main.py").is_file()


def test_the_matrix_library_is_held_to_one_thread_before_numpy_loads() -> None:
    """Called from several worker threads at once, a matrix library that
    starts threads of its own crawls and can crash the program (PLAN.md 5).
    The setting only counts if it is made before anything loads numpy."""
    tree = ast.parse((PROJECT_ROOT / "main.py").read_text(encoding="utf-8"))
    setting = None
    first_import = None
    for index, node in enumerate(tree.body):
        if setting is None and "OPENBLAS_NUM_THREADS" in ast.unparse(node):
            setting = index
        if isinstance(node, ast.ImportFrom) and node.module not in ("__future__", "os"):
            first_import = index if first_import is None else first_import
        if isinstance(node, ast.Import) and any(a.name != "os" for a in node.names):
            first_import = index if first_import is None else first_import
    assert setting is not None
    assert first_import is not None and setting < first_import


# -- the packaged release (PLAN.md 13.2, 13.6) ---------------------------


def _spec_text() -> str:
    return (PROJECT_ROOT / "tintaphoto.spec").read_text(encoding="utf-8")


def test_the_build_is_a_folder_rather_than_one_file() -> None:
    """Qt under the LGPL v3 stays replaceable, so COLLECT and not onefile."""
    spec = _spec_text()
    assert "COLLECT(" in spec
    assert "exclude_binaries=True" in spec


def test_the_build_leaves_out_what_is_never_used() -> None:
    """PLAN.md 13.2: no FFmpeg, and none of the Qt modules nothing imports."""
    spec = _spec_text().lower()
    for unwanted in ("opencv_videoio_ffmpeg", "qtwebengine", "qt6pdf", "qt6quick"):
        assert unwanted in spec, unwanted


def test_the_release_folder_carries_the_licences() -> None:
    """PLAN.md 13.6: the zip holds LICENSE, the notices and the texts."""
    spec = _spec_text()
    assert "LICENSE" in spec
    assert "THIRD_PARTY_NOTICES.md" in spec
    assert 'copytree(PROJECT / "licenses"' in spec


def test_the_resources_the_program_reads_are_bundled() -> None:
    assert '(str(PROJECT / "resources"), "resources")' in _spec_text()


def test_the_installer_and_the_program_agree_on_the_version() -> None:
    """Two files name the version; a release with them out of step would
    install 0.1.0 over 0.2.0 and leave Windows none the wiser."""
    import re

    from app.paths import APP_VERSION

    script = (PROJECT_ROOT / "tintaphoto.iss").read_text(encoding="utf-8")
    declared = re.search(r'#define\s+AppVersion\s+"([^"]+)"', script)
    assert declared is not None, "the installer script has no AppVersion"
    assert declared.group(1) == APP_VERSION


def test_the_changelog_names_the_version_the_program_reports() -> None:
    """CLAUDE.md 11: a version that goes up brings a changelog entry with it.

    Writing one without the other is the easy mistake, so the newest heading
    in CHANGELOG.md has to be the version the program itself reports.
    """
    from app.paths import APP_VERSION

    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    newest = _CHANGELOG_VERSION.search(changelog)
    assert newest is not None, "CHANGELOG.md names no version"
    assert newest.group(1) == APP_VERSION


def test_the_packaging_metadata_does_not_repeat_the_version() -> None:
    """CLAUDE.md 11: app/paths.py is the one place the version is written,
    so the packaging metadata reads it instead of keeping its own copy."""
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in pyproject
    assert 'version = { attr = "app.paths.APP_VERSION" }' in pyproject
    assert re.search(r'^version = "', pyproject, re.MULTILINE) is None


def test_the_development_log_starts_with_a_day() -> None:
    """CLAUDE.md 10: every piece of work leaves a line in docs/DEVLOG.md.

    The log is read top down and a menu entry is meant to show it later, so
    the newest day has to come first and the days have to run backwards.
    """
    devlog = (PROJECT_ROOT / "docs" / "DEVLOG.md").read_text(encoding="utf-8")
    days = _DEVLOG_DAY.findall(devlog)
    assert days, "docs/DEVLOG.md has no dated entry"
    assert days == sorted(days, reverse=True), "the days are out of order"


def test_the_release_is_built_by_one_command() -> None:
    """Running only half of it is what puts yesterday's program in today's
    installer, so the steps live together in one script."""
    script = (PROJECT_ROOT / "tools" / "build_release.py").read_text(encoding="utf-8")
    for step in ("make_app_icon", "PyInstaller", "tintaphoto.iss", "make_zip"):
        assert step in script, step


def test_the_lens_database_brings_its_licence_with_it() -> None:
    """PLAN.md 13.2: the Lensfun database is CC BY-SA 3.0 and ships that text."""
    database = sorted((PROJECT_ROOT / "resources" / "lensfun").glob("*.xml"))
    assert database, "lens correction needs resources/lensfun"

    licence = PROJECT_ROOT / "licenses" / "CC-BY-SA-3.0.txt"
    assert licence.is_file(), "the lens database needs licenses/CC-BY-SA-3.0.txt"
    assert "Attribution-ShareAlike 3.0" in licence.read_text(encoding="utf-8")

    notices = (PROJECT_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "Lensfun" in notices
    assert "CC BY-SA 3.0" in notices


def test_a_bundled_font_brings_its_licence_with_it() -> None:
    """PLAN.md 13.2: anything distributed with the program ships its licence.

    The font is the one piece of artwork in the release that someone else
    owns, so the check is that its text and its notice are both there.
    """
    fonts = sorted((PROJECT_ROOT / "resources" / "fonts").glob("*.tt[fc]"))
    if not fonts:
        return  # the program runs on the system font without it

    licence = PROJECT_ROOT / "licenses" / "OFL-1.1.txt"
    assert licence.is_file(), "a bundled font needs licenses/OFL-1.1.txt"
    text = licence.read_text(encoding="utf-8")
    assert "SIL OPEN FONT LICENSE" in text.upper()
    assert "Copyright" in text  # the licence requires the font's own line

    notices = (PROJECT_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "Inter" in notices
    assert "OFL" in notices


def test_the_face_detection_model_brings_its_licence_with_it() -> None:
    """PLAN.md 13.2: the YuNet model is MIT and ships its copyright line."""
    model = PROJECT_ROOT / "resources" / "models" / "face_detection_yunet_2023mar.onnx"
    assert model.is_file(), "skin enhancement needs the YuNet model"

    licence = PROJECT_ROOT / "licenses" / "MIT-YuNet.txt"
    assert licence.is_file(), "the face detection model needs licenses/MIT-YuNet.txt"
    assert "Shiqi Yu" in licence.read_text(encoding="utf-8")

    notices = (PROJECT_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "YuNet" in notices
