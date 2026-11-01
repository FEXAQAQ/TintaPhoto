"""Tests for the design tokens and the generated stylesheet (PLAN.md 3.5)."""

from __future__ import annotations

import re

from ui import theme

_HEX_COLOUR = re.compile(r"^#[0-9A-F]{6}$")

_COLOUR_TOKENS = (
    "BG_CANVAS",
    "BG_APP",
    "BG_PANEL",
    "BG_RAISED",
    "BG_INSET",
    "BORDER",
    "TRACK",
    "TRACK_HOVER",
    "TRACK_PRESSED",
    "TEXT_PRIMARY",
    "TEXT_SECONDARY",
    "TEXT_TERTIARY",
    "ACCENT",
    "DANGER",
    "WHITE",
    "PREVIEW_BG_BLACK",
    "PREVIEW_BG_DARK",
    "PREVIEW_BG_MID",
)

#: Sizes that must sit on the 4 px grid (PLAN.md 3.5).
_GRID_TOKENS = (
    "SPACE_XS",
    "SPACE_SM",
    "SPACE_MD",
    "SPACE_LG",
    "SPACE_XL",
    "SPACE_XXL",
    "TOOLBAR_HEIGHT",
    "PANEL_WIDTH",
    "PANEL_PADDING",
    "FILMSTRIP_HEIGHT",
    "FILMSTRIP_TAB_WIDTH",
    "FILMSTRIP_TAB_HEIGHT",
    "CANVAS_MARGIN",
    "ICON_BUTTON_SIZE",
    "SLIDER_ROW_HEIGHT",
    "GROUP_GAP",
    "HISTOGRAM_HEIGHT",
    "TAB_BAR_HEIGHT",
    "PANEL_ACTION_ROW_HEIGHT",
    "BUTTON_HEIGHT",
    "SCROLLBAR_WIDTH",
    "WELCOME_CONTENT_WIDTH",
    "WELCOME_BUTTON_HEIGHT",
    "WELCOME_HERO_HEIGHT",
    "WELCOME_DROP_HEIGHT",
    "WELCOME_ROW_HEIGHT",
    "WELCOME_ROW_GAP",
    "WELCOME_GROUP_GAP",
    "WELCOME_SECTION_GAP",
    "DROP_HINT_ICON_SIZE",
    "NAME_DIALOG_MIN_WIDTH",
    "OPENING_FALLING_DROP_HEIGHT",
    "OPENING_LETTER_RISE",
    "OPENING_ITEM_RISE",
    "VIEW_BAR_HEIGHT",
    "CHIP_HEIGHT",
    "WINDOW_MIN_WIDTH",
    "WINDOW_MIN_HEIGHT",
    "WINDOW_DEFAULT_WIDTH",
    "WINDOW_DEFAULT_HEIGHT",
    "DIALOG_WIDTH_SETTINGS",
    "DIALOG_WIDTH_EXPORT",
)


def test_colours_are_uppercase_six_digit_hex() -> None:
    for name in _COLOUR_TOKENS:
        value = getattr(theme, name)
        assert _HEX_COLOUR.match(value), f"{name} = {value!r}"


def test_sizes_sit_on_the_four_pixel_grid() -> None:
    for name in _GRID_TOKENS:
        value = getattr(theme, name)
        assert value % theme.UNIT == 0, f"{name} = {value} is not a multiple of {theme.UNIT}"


def test_icon_size_fits_inside_its_button() -> None:
    assert theme.ICON_SIZE < theme.ICON_BUTTON_SIZE


def test_font_sizes_follow_the_documented_hierarchy() -> None:
    assert (
        theme.FONT_SIZE_CAPTION
        < theme.FONT_SIZE_LABEL
        < theme.FONT_SIZE_BODY
        < theme.FONT_SIZE_TITLE
        < theme.FONT_SIZE_WORDMARK
    )


def test_font_stack_prefers_the_bundled_face() -> None:
    assert theme.FONT_STACK_UI[0] == theme.FONT_FAMILY_BUNDLED
    assert theme.FONT_STACK_UI[-1] == "sans-serif"
    assert theme.font_stack_css().startswith(f'"{theme.FONT_FAMILY_BUNDLED}"')


def test_durations_are_short_and_positive() -> None:
    for duration in (
        theme.DURATION_HOVER_MS,
        theme.DURATION_PANEL_MS,
        theme.DURATION_PHOTO_FADE_MS,
        theme.DURATION_ZOOM_MS,
    ):
        assert 0 < duration <= 300


def test_qss_contains_no_unresolved_placeholder() -> None:
    qss = theme.build_qss()
    assert "{" in qss  # the stylesheet does contain rule blocks
    assert "None" not in qss
    # An f-string placeholder that was not replaced would leave a bare name
    # between single braces, which is never valid in a Qt stylesheet.
    assert not re.search(r"(?<!\{)\{[A-Z_]+\}(?!\})", qss)


def test_qss_uses_the_token_colours() -> None:
    qss = theme.build_qss()
    for name in ("BG_APP", "BG_PANEL", "BG_CANVAS", "TEXT_PRIMARY", "ACCENT"):
        assert getattr(theme, name) in qss, name


def test_qss_defines_the_four_region_selectors() -> None:
    qss = theme.build_qss()
    for selector in ("#Toolbar", "#PreviewArea", "#Inspector", "#Filmstrip"):
        assert selector in qss, selector


def test_qss_defines_hover_pressed_and_disabled_states() -> None:
    qss = theme.build_qss()
    for state in (":hover", ":pressed", ":disabled", ":focus"):
        assert state in qss, state


def test_the_scroll_bar_handle_brightens_as_the_pointer_takes_hold() -> None:
    """PLAN.md 3.4 wants three states, and each has to be a step up.

    The handle rests at TRACK, which is already where the buttons land when
    they are pressed, so it cannot share their ladder: its hover used to be
    BG_RAISED, which made it darker under the pointer than at rest.
    """
    assert theme.TRACK < theme.TRACK_HOVER < theme.TRACK_PRESSED  # #RRGGBB greys
    qss = theme.build_qss()
    for orientation in ("vertical", "horizontal"):
        hover = qss.split(f"QScrollBar::handle:{orientation}:hover {{")[1].split("}")[0]
        pressed = qss.split(f"QScrollBar::handle:{orientation}:pressed {{")[1].split("}")[0]
        assert f"background-color: {theme.TRACK_HOVER}" in hover, orientation
        assert f"background-color: {theme.TRACK_PRESSED}" in pressed, orientation


def test_the_inset_is_darker_than_the_panel_it_sits_in() -> None:
    """Layers are told apart by shades of grey (PLAN.md 3.5)."""
    assert theme.BG_INSET < theme.BG_PANEL  # both are #RRGGBB greys
    assert theme.BG_INSET > theme.BG_CANVAS


def test_fully_rounded_shapes_are_half_their_own_height() -> None:
    assert theme.RADIUS_VIEW_BAR * 2 == theme.VIEW_BAR_HEIGHT
    assert theme.RADIUS_CHIP * 2 == theme.CHIP_HEIGHT


def test_the_welcome_button_is_the_only_one_that_is_taller_than_a_panel_button() -> None:
    assert theme.WELCOME_BUTTON_HEIGHT > theme.BUTTON_HEIGHT


def test_the_welcome_screen_has_two_steps_of_gap_rather_than_one() -> None:
    """One gap repeated between every block left the welcome screen as five
    evenly spaced rows: the name, the line that says what the program is
    for, the one action, the recent folders and the drop zone all read as
    peers (PLAN.md 3.7 does not make them peers)."""
    assert theme.SPACE_MD < theme.WELCOME_GROUP_GAP < theme.WELCOME_SECTION_GAP


def test_the_tagline_is_set_above_the_folder_names_listed_below_it() -> None:
    """It was the smallest text on the screen but one, so the eye went to
    the list of recent folders rather than to the sentence that says what
    the program is for."""
    assert theme.FONT_SIZE_TAGLINE > theme.FONT_SIZE_RECENT_NAME > theme.FONT_SIZE_CAPTION


def test_the_ink_colours_are_the_ones_the_icon_is_painted_with() -> None:
    """The opening draws the icon's drop; it must be the icon's drop."""
    import re

    from app.paths import resources_dir

    artwork = (resources_dir() / "icons" / "app.svg").read_text(encoding="utf-8")
    stops = re.findall(r'offset="([0-9.]+)" stop-color="(#[0-9A-Fa-f]{6})"', artwork)
    assert [float(offset) for offset, _ in stops] == list(theme.INK_STOPS)
    assert [colour.upper() for _, colour in stops] == [c.upper() for c in theme.INK_COLOURS]


def test_qss_styles_the_new_regions_and_the_welcome_button() -> None:
    qss = theme.build_qss()
    for selector in ("#FilmstripPosition", 'QPushButton[role="hero"]'):
        assert selector in qss, selector


def test_the_drop_zone_is_drawn_from_the_tokens_not_from_a_colour_of_its_own() -> None:
    """PLAN.md 3.7's frame moved out of the stylesheet so that it can reach
    the accent colour over 80 ms; its colours still come from here."""
    assert theme.DROP_ZONE_DASH == (4.0, 2.0)
    # What it is drawn from, resting and lit; the drawing itself is checked
    # on the widget, in tests/test_welcome_view.py.
    assert theme.TRACK != theme.ACCENT


def test_the_welcome_button_is_not_filled() -> None:
    """PLAN.md 3.5 keeps the one filled button for exporting."""
    qss = theme.build_qss()
    rule = qss.split('QPushButton[role="hero"] {')[1].split("}")[0]
    assert "background-color: transparent" in rule


def test_a_corner_is_slower_the_larger_the_thing_it_belongs_to() -> None:
    """Three steps of corner, so the interface is not one box of equal
    rectangles: a control, a card inside a panel, and the panel itself."""
    assert theme.RADIUS_THUMB <= theme.RADIUS_CONTROL
    assert theme.RADIUS_CONTROL < theme.RADIUS_CARD < theme.RADIUS_PANEL


def test_an_outlined_button_brightens_its_edge_under_the_pointer() -> None:
    """The welcome screen's button and the panel buttons answered the
    pointer differently; PLAN.md 3.4 wants one kind to behave as one.

    They reach the same edge by different means now: the panel button is
    swapped by the stylesheet, while the welcome screen's takes the
    documented 80 ms to get there and so is painted by the widget. What has
    to match is where they arrive.
    """
    qss = theme.build_qss()
    # Anchored at the start of a line, or the rule for the buttons of the
    # floating bar is found first.
    plain = qss.split(chr(10) + "QPushButton:hover {")[1].split("}")[0]
    assert f"border-color: {theme.TEXT_TERTIARY}" in plain
    # The hero button's hover is no longer in the stylesheet at all.
    assert 'QPushButton[role="hero"]:hover' not in qss


def test_the_interface_wears_no_colour_of_its_own() -> None:
    """The accent is the white of the headings: what marks the current
    state is brightness, so the only colour on screen is the photograph's."""
    assert theme.ACCENT == theme.TEXT_PRIMARY
    # The marks drawn on the photo itself keep their meaning (PLAN.md 2.2).
    assert theme.CLIP_SHADOW != theme.ACCENT
    assert theme.DANGER != theme.ACCENT


def test_the_filled_button_can_show_a_focus_ring_on_its_light_face() -> None:
    qss = theme.build_qss()
    rule = qss.split('QPushButton[role="primary"]:focus {')[1].split("}")[0]
    assert theme.BG_APP in rule


def _relative_luminance(colour: str) -> float:
    """Return the relative luminance of ``#rrggbb``, as WCAG defines it."""
    channels = []
    for index in (1, 3, 5):
        value = int(colour[index : index + 2], 16) / 255
        channels.append(
            value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
        )
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast(colour: str, background: str) -> float:
    """Return the contrast ratio between two ``#rrggbb`` colours."""
    first = _relative_luminance(colour)
    second = _relative_luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def test_every_level_of_text_can_be_read_on_every_background() -> None:
    """The interface is read rather than looked at (PLAN.md 3.4).

    The hints and group titles used to sit at 3.3:1 against the panel, under
    the 4.5:1 that text this small wants, and they were reported as hard to
    read. Every level now clears it on every background it is put on - the
    panel being the hardest, since it is the lightest of the three.
    """
    backgrounds = (theme.BG_CANVAS, theme.BG_PANEL, theme.BG_RAISED)
    for level in (theme.TEXT_PRIMARY, theme.TEXT_SECONDARY, theme.TEXT_TERTIARY):
        for background in backgrounds:
            ratio = contrast(level, background)
            assert ratio >= 4.5, f"{level} on {background} is only {ratio:.2f}:1"


def test_the_levels_of_text_stay_far_enough_apart_to_be_told_apart() -> None:
    """Brightness is what ranks one piece of text above another, since the
    interface wears no colour of its own; the steps have to be seen."""
    primary = _relative_luminance(theme.TEXT_PRIMARY)
    secondary = _relative_luminance(theme.TEXT_SECONDARY)
    tertiary = _relative_luminance(theme.TEXT_TERTIARY)

    assert primary > secondary > tertiary
    assert primary / secondary >= 1.5, "the primary text no longer stands out"
    assert secondary / tertiary >= 1.5, "the two quiet levels have run together"
