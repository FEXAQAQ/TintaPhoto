"""Design tokens and dark theme stylesheet generation.

Every colour, font size, spacing value, corner radius and animation duration
used by the interface is defined here (PLAN.md section 3.5). No other module
may contain a literal colour code, pixel value or font size; import the
constants from this module instead.

This module deliberately does not import PySide6 so that the tokens and the
generated stylesheet can be unit tested without a display.
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------
# Colour  (PLAN.md 3.5 - neutral greys only, so the interface never biases
# the user's judgement of the photo's colour)
# --------------------------------------------------------------------------

BG_CANVAS: Final[str] = "#101010"       # photo preview background
BG_APP: Final[str] = "#141414"          # window base, toolbar
BG_PANEL: Final[str] = "#1A1A1A"        # side panel
BG_RAISED: Final[str] = "#262626"       # hover, selected button, dropdown
#: One step below the panel, for an area set into it rather than sitting on
#: it: the histogram's plot. PLAN.md 3.5 separates layers by shades of grey,
#: which is what this is; the alternative would be the translucent black
#: layer that 3.5 rules out.
BG_INSET: Final[str] = "#121212"
BORDER: Final[str] = "#222222"          # 1 px separator between regions
TRACK: Final[str] = "#2E2E2E"           # slider track, scroll bar handle
#: A scroll bar handle answering the pointer (PLAN.md 3.4 asks every
#: clickable element for a hover and a pressed state). It cannot borrow the
#: ladder the buttons climb - transparent, BG_RAISED, TRACK - because it
#: already rests at the top of it, so these carry it two even steps further
#: up instead. They used to be BG_RAISED, which made the handle darker
#: under the pointer than at rest.
TRACK_HOVER: Final[str] = "#3A3A3A"
TRACK_PRESSED: Final[str] = "#464646"
TEXT_PRIMARY: Final[str] = "#EDEDED"    # primary text, numeric values
# Both of these were raised once the interface was read rather than looked
# at: at 11 px the old tertiary fell below the 4.5:1 that small text wants,
# worst of all in the panels, whose background is lighter than the canvas
# the welcome screen sits on. Every step now clears it on all three
# backgrounds, and the steps between them are more even than they were.
TEXT_SECONDARY: Final[str] = "#B4B4B4"  # labels, secondary text, icons
TEXT_TERTIARY: Final[str] = "#8E8E8E"   # group titles, hints, disabled
#: What marks the current state: the same white as a group's heading, so
#: that the interface carries no colour of its own at all. The colour in
#: front of the user is the photograph's.
ACCENT: Final[str] = "#EDEDED"
DANGER: Final[str] = "#E5534B"          # errors, highlight clipping

#: Blue over a blocked shadow, which PLAN.md 2.2 asks for by name. It used
#: to be the accent colour, and a mark on the photo has to go on saying what
#: it says whatever the interface is wearing.
CLIP_SHADOW: Final[str] = "#8AA4FF"

#: The four colours of the drop of ink in the application icon
#: (resources/icons/app.svg), in the order of its gradient. The interface
#: uses none of them; only the icon's artwork and the welcome screen's
#: opening, which draws that artwork, do (PLAN.md 3.4).
INK_COLOURS: Final[tuple[str, ...]] = ("#FFD166", "#FF7A5C", "#C45BE0", "#4D9BFF")
#: Where each sits along the drop's gradient.
INK_STOPS: Final[tuple[float, ...]] = (0.0, 0.3, 0.6, 1.0)
WHITE: Final[str] = "#FFFFFF"           # primary button hover, crop outline
SCRIM: Final[str] = "#000000"           # dims the window behind a dialog

#: The dots at the ends of the white balance tracks, which say which way each
#: one runs (PLAN.md 3.6). Muted, so that they mark a direction without
#: competing with the photo.
MARK_BLUE: Final[str] = "#6E8BD6"
MARK_YELLOW: Final[str] = "#D6C06E"
MARK_GREEN: Final[str] = "#7FB37F"
MARK_MAGENTA: Final[str] = "#C47FB3"

#: The three channels of the histogram, drawn over one another (A-12). Light
#: enough to stay readable where all three overlap.
#: Clipping marked without hues, for anyone who would rather not have
#: colour on the photo (PLAN.md 3.10): each mark is the opposite of what it
#: sits on, so that it still shows.
MARK_CLIPPED_HIGH: Final[str] = "#000000"
MARK_CLIPPED_LOW: Final[str] = "#FFFFFF"

HISTOGRAM_RED: Final[str] = "#FF5A5A"

HISTOGRAM_GREEN: Final[str] = "#5AFF8C"
HISTOGRAM_BLUE: Final[str] = "#5A9CFF"
#: How solid each channel is, out of 255, so that overlaps read as a blend.
HISTOGRAM_ALPHA: Final[int] = 110
#: The histogram as brightness alone: one grey curve, the secondary text's
#: grey, more solid than a channel because there is nothing to blend with.
HISTOGRAM_LUMA: Final[str] = "#B4B4B4"
HISTOGRAM_LUMA_ALPHA: Final[int] = 160

#: Preview background choices offered in the settings dialog (PLAN.md 3.10).
PREVIEW_BG_BLACK: Final[str] = "#000000"
PREVIEW_BG_DARK: Final[str] = BG_CANVAS
PREVIEW_BG_MID: Final[str] = "#3A3A3A"

# --------------------------------------------------------------------------
# Typography  (PLAN.md 3.5)
# --------------------------------------------------------------------------

#: Font family name of the bundled Latin face, used when registering a font file.
FONT_FAMILY_BUNDLED: Final[str] = "Inter"

#: Latin and numeric font stack. Bundled Inter is preferred; the remaining
#: entries are fallbacks so the interface stays correct before Inter is added.
FONT_STACK_UI: Final[tuple[str, ...]] = (
    FONT_FAMILY_BUNDLED,
    "Segoe UI",
    "Microsoft JhengHei UI",
    "sans-serif",
)

FONT_SIZE_TITLE: Final[int] = 15
FONT_SIZE_BODY: Final[int] = 13
FONT_SIZE_LABEL: Final[int] = 12
FONT_SIZE_CAPTION: Final[int] = 11
#: The name on the welcome screen, drawn letter by letter by its opening.
FONT_SIZE_WORDMARK: Final[int] = 46
#: The line under it, and a recent folder's name.
FONT_SIZE_TAGLINE: Final[int] = 16
FONT_SIZE_RECENT_NAME: Final[int] = 14

FONT_WEIGHT_TITLE: Final[int] = 600
FONT_WEIGHT_NORMAL: Final[int] = 400

#: Letter spacing of the application wordmark on the welcome screen, in px:
#: a touch tight, as a name set this large wants.
WORDMARK_LETTER_SPACING: Final[float] = -0.5

# --------------------------------------------------------------------------
# Spacing and size  (PLAN.md 3.5 - base unit 4 px, every value a multiple)
# --------------------------------------------------------------------------

UNIT: Final[int] = 4

SPACE_XS: Final[int] = 4
SPACE_SM: Final[int] = 8
SPACE_MD: Final[int] = 12
SPACE_LG: Final[int] = 16
SPACE_XL: Final[int] = 24
SPACE_XXL: Final[int] = 32

TOOLBAR_HEIGHT: Final[int] = 44
PANEL_WIDTH: Final[int] = 300
PANEL_PADDING: Final[int] = 16
# PLAN.md 3.1 gives the filmstrip a height of 110 px, but 3.5 requires every
# size to be a multiple of 4. 112 px is the nearest value that satisfies both.
FILMSTRIP_HEIGHT: Final[int] = 112
#: The bookmark tab standing up from the filmstrip's top left edge, which
#: folds it away and brings it back (PLAN.md 3.1).
FILMSTRIP_TAB_WIDTH: Final[int] = 48
FILMSTRIP_TAB_HEIGHT: Final[int] = 16
CANVAS_MARGIN: Final[int] = 24

ICON_BUTTON_SIZE: Final[int] = 28
ICON_SIZE: Final[int] = 18
ICON_SIZE_SMALL: Final[int] = 12
#: A smaller icon button, for a control beside what it works on (the
#: histogram's arrows), with the small icon.
SMALL_ICON_BUTTON_SIZE: Final[int] = 20
ICON_STROKE_WIDTH: Final[float] = 1.5

#: The mark in the corner of an edited thumbnail (PLAN.md 3.1): a solid patch
#: the size of a small icon plus its padding, so the glyph reads over any
#: photo without a shadow or a translucent layer, which 3.4 rules out.
THUMBNAIL_BADGE_SIZE: Final[int] = 16
#: The box in a thumbnail's top left corner while photos are being chosen
#: for export (PLAN.md 3.1, X-09): the same size as the edited mark, so the
#: two corners of a thumbnail carry marks of one size.
CHECK_BOX_SIZE: Final[int] = THUMBNAIL_BADGE_SIZE

SLIDER_ROW_HEIGHT: Final[int] = 28
#: The line under a two-way slider that says what each end means (PLAN.md 3.16).
SLIDER_ENDS_HEIGHT: Final[int] = 16
GROUP_GAP: Final[int] = 16
#: The title row of a collapsible group (PLAN.md 3.1): a little taller than a
#: slider row, so that it reads as the start of a section and gives the arrow
#: and the name room to be pressed.
GROUP_HEADER_HEIGHT: Final[int] = 32
SLIDER_TRACK_HEIGHT: Final[int] = 2
SLIDER_HANDLE_SIZE: Final[int] = 10
SLIDER_HANDLE_SIZE_HOVER: Final[int] = 12

#: The crop box (PLAN.md 3.6): a 1 px white outline with L shaped corners,
#: and everything outside it darkened.
CROP_LINE_WIDTH: Final[int] = 1
CROP_HANDLE_LENGTH: Final[int] = 20
CROP_HANDLE_WIDTH: Final[int] = 3
#: How near the pointer has to be to take hold of an edge or a corner.
CROP_GRAB_RADIUS: Final[int] = 12
#: How far the photo outside the crop box is darkened: the opacity of the
#: shade laid over it, so 0.60 is the 60% PLAN.md 3.6 asks for.
CROP_OUTSIDE_DIM: Final[float] = 0.60
#: The spot removal brush (PLAN.md 3.6, A-32): white outlines over the photo,
#: like the crop box's, each with a dark edge either side so that it shows
#: on a white wall as well as on a dark sky. The selected stroke is drawn
#: heavier, the way the current state is marked everywhere else. What has
#: been mended is not outlined until it is selected.
SPOT_LINE_WIDTH: Final[int] = 1
SPOT_SELECTED_WIDTH: Final[int] = 2
SPOT_HALO: Final[str] = "#000000"
SPOT_HALO_ALPHA: Final[int] = 128
SPOT_HALO_WIDTH: Final[int] = 1
#: How near the pointer has to be to take hold of a spot too small to aim at.
SPOT_GRAB_RADIUS: Final[int] = 8
#: The light wash over what the brush is painting, out of 255: enough to see
#: where it has been, thin enough to see what is under it.
SPOT_PAINT_ALPHA: Final[int] = 112
#: How far the pointer moves, in pixels, before the paint takes another
#: point; and how far a press may wander and still count as a click.
SPOT_PAINT_STEP: Final[int] = 2
SPOT_CLICK_SLOP: Final[int] = 4
#: The portrait brush (PLAN.md 3.15): the skin the sliders work on is shown
#: as a rose wash over the photo, a mark that says something about the
#: photo as the clipping marks do, so it has a colour of its own. What the
#: brush is painting is washed white when it adds and dark when it takes
#: away; the brush itself is the spot brush's circle.
SKIN_RANGE: Final[str] = "#FF6B8B"
SKIN_RANGE_ALPHA: Final[int] = 120
SKIN_REMOVE_WASH: Final[str] = "#000000"
SKIN_REMOVE_ALPHA: Final[int] = 112
#: Half the length of the + or − drawn in the brush's middle.
SKIN_BRUSH_MARK: Final[int] = 4
#: The liquify brush (PLAN.md 3.16): the spot brush's circle with a dot in
#: its middle, this many pixels across, where the brush pushes from.
LIQUIFY_BRUSH_DOT: Final[int] = 4
#: The face's points while they are placed by hand (PLAN.md 3.16, L-12):
#: each mark is this many pixels across, filled in when it is the one in
#: hand, and the pointer takes hold of one this near it.
FACE_HANDLE_SIZE: Final[int] = 12
FACE_HANDLE_GRAB: Final[int] = 12
#: The dot in the middle of the chin's ring, across, and the dark edge
#: either side of each point's name, so it reads over light skin and dark
#: hair alike.
FACE_HANDLE_DOT: Final[int] = 4
FACE_LABEL_HALO: Final[int] = 2
#: The arrows that say which way a face slider moves the face, and how long
#: its outline stays on the photo after the pointer leaves the slider.
FACE_ARROW_LENGTH: Final[int] = 16
FACE_REACH_LINGER_MS: Final[int] = 800
#: How near the pointer has to be to take hold of the collage's seam, either
#: side of it (PLAN.md 3.13).
COLLAGE_SEAM_GRAB: Final[int] = 8
#: How see-through the second photo is while it is aligned over the first:
#: half, so that both can be read at once (PLAN.md 3.13).
COLLAGE_ALIGN_OPACITY: Final[float] = 0.50
#: The watermark over the photo (PLAN.md 3.14): a dashed outline round it
#: while the pointer is over it or it is dragged, held SPACE_XS clear of the
#: text, and 1 px accent guides where it has snapped, stopping SPACE_SM short
#: of it so they never cross the text.
WATERMARK_OUTLINE_WIDTH: Final[int] = 1
WATERMARK_OUTLINE_GAP: Final[int] = 4
WATERMARK_GUIDE_WIDTH: Final[int] = 1
WATERMARK_GUIDE_GAP: Final[int] = 8
#: The position picker (PLAN.md 3.14): three by three squares.
ANCHOR_CELL: Final[int] = 20
ANCHOR_GAP: Final[int] = 4

HISTOGRAM_HEIGHT: Final[int] = 160
#: The band above the plot that holds its name and the two triangles.
HISTOGRAM_TITLE_HEIGHT: Final[int] = 28
#: How far the arrows and the triangles sit from the histogram's sides.
HISTOGRAM_EDGE: Final[int] = SPACE_SM
TAB_BAR_HEIGHT: Final[int] = 36
TAB_UNDERLINE_HEIGHT: Final[int] = 2
PANEL_ACTION_ROW_HEIGHT: Final[int] = 44
BUTTON_HEIGHT: Final[int] = 28
#: A row of the looks tab: the photo as it would leave it in a square at the
#: left of the row, and the row tall enough to hold it with the same 4 px
#: all round (PLAN.md 3.1). A thumbnail the height of an ordinary button
#: would be 20 px of photo, too small to tell one look from another.
FILTER_THUMBNAIL_SIZE: Final[int] = 40
FILTER_ROW_HEIGHT: Final[int] = 48
#: The space between the thumbnail and the look's name.
FILTER_THUMBNAIL_GAP: Final[int] = 8
BORDER_WIDTH: Final[int] = 1
FOCUS_RING_WIDTH: Final[int] = 1
SCROLLBAR_WIDTH: Final[int] = 8

#: The welcome screen (PLAN.md 3.7). One centred column, as wide as the
#: two buttons side by side; the drop, the name and the tagline are drawn across
#: the whole screen above it, so that the opening's ripples are not cut off
#: at the column's edge.
WELCOME_CONTENT_WIDTH: Final[int] = 340
#: The room the drop, the name and the tagline take above the button, and
#: where in it each sits: the drop's middle, then the baselines.
WELCOME_HERO_HEIGHT: Final[int] = 180
WELCOME_DROP_HEIGHT: Final[int] = 60
WELCOME_DROP_CENTRE: Final[int] = 30
WELCOME_WORDMARK_BASELINE: Final[int] = 134
WELCOME_TAGLINE_BASELINE: Final[int] = 174

WELCOME_RECENT_COUNT: Final[int] = 5
#: A recent folder: its name over its path, the photo count over the day it
#: was opened, in a row tall enough for the two lines with 4 px between rows.
WELCOME_ROW_HEIGHT: Final[int] = 44
WELCOME_ROW_GAP: Final[int] = 4

#: Between the pieces of one thought: the tagline and the button that
#: answers it.
WELCOME_GROUP_GAP: Final[int] = 32
#: Between one section of the screen and the next: the button and the list.
WELCOME_SECTION_GAP: Final[int] = 44

#: The two buttons of the welcome screen, "open file" and "open folder", side
#: by side across the column. Taller than a button in a panel, because on an
#: otherwise empty screen they are the things to reach for; each has its
#: icon and name centred, and its shortcut only in its tooltip.
WELCOME_BUTTON_HEIGHT: Final[int] = 48
WELCOME_BUTTON_GAP: Final[int] = SPACE_MD
WELCOME_BUTTON_WIDTH: Final[int] = (WELCOME_CONTENT_WIDTH - WELCOME_BUTTON_GAP) // 2

#: The line under the welcome screen's list saying photos can be dropped
#: anywhere on the window: a small dashed square stands for the drop, the
#: same dashes the preview area shows while something is dragged over it.
DROP_HINT_ICON_SIZE: Final[int] = 16
#: The rhythm of those dashes, in pixels on and pixels off.
DROP_ZONE_DASH: Final[tuple[float, float]] = (4.0, 2.0)

#: The dialog that asks for a preset's name.
NAME_DIALOG_MIN_WIDTH: Final[int] = 400

#: The floating bar under the photo (PLAN.md 3.3): one button tall, with the
#: space above and below it that the buttons in a panel have.
VIEW_BAR_HEIGHT: Final[int] = 36

#: A small label on a background of its own, such as the position in the
#: filmstrip.
CHIP_HEIGHT: Final[int] = 20

WINDOW_MIN_WIDTH: Final[int] = 1024
WINDOW_MIN_HEIGHT: Final[int] = 640
WINDOW_DEFAULT_WIDTH: Final[int] = 1440
WINDOW_DEFAULT_HEIGHT: Final[int] = 900

DIALOG_WIDTH_SETTINGS: Final[int] = 600
DIALOG_HEIGHT_SETTINGS: Final[int] = 560
#: Width of the category list down the left of the settings dialog (3.10).
SETTINGS_CATEGORY_WIDTH: Final[int] = 140
#: Width of the control at the right of a settings row.
SETTINGS_CONTROL_WIDTH: Final[int] = 200

DIALOG_WIDTH_EXPORT: Final[int] = 440
#: Width of the label column of a dialog row, so that every row lines up
#: (PLAN.md 3.11).
DIALOG_LABEL_WIDTH: Final[int] = 72

#: The switch of a setting that is either on or off (PLAN.md 3.10, 3.11).
SWITCH_WIDTH: Final[int] = 36
SWITCH_HEIGHT: Final[int] = 20
SWITCH_KNOB_SIZE: Final[int] = 12

#: How wide the message under the photo may grow before it wraps (PLAN.md 3.6).
TOAST_MAX_WIDTH: Final[int] = 480


# --------------------------------------------------------------------------
# Corner radius  (PLAN.md 3.5)
# --------------------------------------------------------------------------

#: A corner is slower the larger the thing it belongs to: a control is
#: something to press, a card is something to read, and a panel or a menu is
#: a surface. Keeping the three apart is what stops the interface reading as
#: a box of identical rectangles.
RADIUS_CONTROL: Final[int] = 6   # buttons, input fields
RADIUS_CARD: Final[int] = 8      # a recess or a patch inside a panel
RADIUS_PANEL: Final[int] = 12    # panels, dialogs, menus
RADIUS_THUMB: Final[int] = 6     # filmstrip thumbnails
#: A filter thumbnail sits 4 px inside a button whose corners are
#: RADIUS_CONTROL; its own corners are rounded less, so the two curves run
#: alongside each other instead of the inner one bulging towards the outer.
RADIUS_FILTER_THUMB: Final[int] = 4
#: The tick box on a thumbnail. RADIUS_THUMB on a box this small comes out
#: nearly round and reads as a dot, not as something to tick.
RADIUS_CHECK_BOX: Final[int] = 4

#: Fully rounded ends, for something that floats over the photo rather than
#: sitting in a panel: half its own height, so the sides are semicircles.
RADIUS_VIEW_BAR: Final[int] = VIEW_BAR_HEIGHT // 2
RADIUS_CHIP: Final[int] = CHIP_HEIGHT // 2

# --------------------------------------------------------------------------
# Motion  (PLAN.md 3.6 - short and restrained, always ease-out)
# --------------------------------------------------------------------------

DURATION_HOVER_MS: Final[int] = 80
DURATION_PANEL_MS: Final[int] = 160
DURATION_PHOTO_FADE_MS: Final[int] = 120
DURATION_ZOOM_MS: Final[int] = 160
#: One turn of the wheel over the filmstrip, glided rather than jumped.
DURATION_SCROLL_MS: Final[int] = 160
DURATION_TOAST_MS: Final[int] = 2000

# The welcome screen's opening, played once when the program starts: the one
# animation longer than the set above (PLAN.md 3.6, 3.7). A drop of ink falls
# and lands, rings spread from where it landed in the four colours of the
# drop, the drop grows to its size, the name comes up a letter at a time in
# those colours and settles to white, the tagline is laid down from left to
# right like a brush stroke, and the rest rises into place one after the
# other. The stages overlap, so each says when it begins as well as how long
# it runs; the lengths are the ones the mock-up was watched at (2026-10-31).
#: The drop falls from the top of the screen, speeding up.
OPENING_FALL_MS: Final[int] = 380
#: The rings, one after another from the moment it lands.
OPENING_RIPPLE_GAP_MS: Final[int] = 110
OPENING_RIPPLE_MS: Final[int] = 1100
#: The drop grows from the size it fell at to its own, overshooting a little.
OPENING_GROW_MS: Final[int] = 520
#: The name, a letter at a time; each comes up in an ink colour and settles
#: to white a little after it has risen.
OPENING_LETTERS_AT_MS: Final[int] = 620
OPENING_LETTER_GAP_MS: Final[int] = 38
OPENING_LETTER_MS: Final[int] = 420
OPENING_LETTER_SETTLE_AFTER_MS: Final[int] = 180
#: The tagline, laid down from left to right.
OPENING_TAGLINE_AT_MS: Final[int] = 1150
OPENING_TAGLINE_MS: Final[int] = 600
#: The button, the list's title and then each folder, and last the hint at
#: the bottom; each rises into place over the same length.
OPENING_BUTTON_AT_MS: Final[int] = 1500
OPENING_LIST_AT_MS: Final[int] = 1600
OPENING_ROWS_AT_MS: Final[int] = 1650
OPENING_ROW_GAP_MS: Final[int] = 70
OPENING_HINT_AT_MS: Final[int] = 1950
OPENING_ITEM_MS: Final[int] = 400
#: Sizes the opening moves things by: the drop while it falls, how far the
#: rings spread (wider than tall, as rings on water are seen from above at
#: an angle), how far the letters and the rest rise.
OPENING_FALLING_DROP_HEIGHT: Final[int] = 24
OPENING_RIPPLE_RADIUS: Final[int] = 300
OPENING_RIPPLE_FLATTEN: Final[float] = 0.34
OPENING_RIPPLE_WIDTH: Final[float] = 1.6
OPENING_LETTER_RISE: Final[int] = 16
OPENING_ITEM_RISE: Final[int] = 12
#: The info card (PLAN.md 3.9) comes in the way the application menu beside
#: it does: it fades in where it rests, evenly, over as long as Qt fades a
#: menu in, and it is gone at once when taken away.
INFO_CARD_FADE_MS: Final[int] = 150
#: Leaving the welcome screen for the first photo, and a camera's preview
#: giving way to the photo as this program develops it: one picture taking
#: the place of another, cross-faded rather than cut (PLAN.md 3.6).
DURATION_PAGE_FADE_MS: Final[int] = 200
#: The "developing" sign on the photo (PLAN.md 3.7): shown only once a photo
#: has kept the user waiting this long, so a quick one shows nothing; it
#: comes up over a fade, breathes once a second while it waits, and goes
#: over DURATION_PAGE_FADE_MS as the photo it was waiting for arrives.
DEVELOPING_DELAY_MS: Final[int] = 300
DEVELOPING_FADE_MS: Final[int] = 120
DEVELOPING_BREATH_MS: Final[int] = 1000
#: How faint the drop gets at the bottom of its breath.
DEVELOPING_BREATH_LOW: Final[float] = 0.55
DEVELOPING_DROP_HEIGHT: Final[int] = 16
#: A message with something to click on stays longer than the two seconds of
#: PLAN.md 3.6: the export's "show in Explorer" link has to be reachable.
DURATION_TOAST_ACTION_MS: Final[int] = 6000
#: The message under the photo rises this far into place while it fades in,
#: and sinks back while it fades out (PLAN.md 3.6).
TOAST_SHOW_MS: Final[int] = 160
TOAST_HIDE_MS: Final[int] = 120
TOAST_RISE: Final[int] = 8

#: Exporting (PLAN.md 3.6, 3.11). The dim behind a dialog fades rather than
#: cuts, and the dialog drops into place while it fades in, then lifts away
#: while it fades out; going is quicker than coming.
SCRIM_FADE_MS: Final[int] = 120
DIALOG_SHOW_MS: Final[int] = 160
DIALOG_HIDE_MS: Final[int] = 120
DIALOG_DROP: Final[int] = 12
DIALOG_LIFT: Final[int] = 8
#: The moment the export is asked for, the photo shrinks to a thumbnail and
#: flies along a curve into the export button: it rises this far above the
#: photo on the way, ends this tall, and fades over the last share of the
#: flight. The button flashes as it lands. 400 ms was too quick to follow
#: (2026-11-01): the photo was gone before the eye found it.
EXPORT_FLY_MS: Final[int] = 700
EXPORT_FLY_ARC: Final[int] = 40
EXPORT_FLY_END_HEIGHT: Final[int] = 20
EXPORT_FLY_FADE_SHARE: Final[float] = 0.15
EXPORT_PULSE_MS: Final[int] = 80
EXPORT_PULSE_STRENGTH: Final[float] = 0.6
#: While the export runs the button fills from the left. The fill catches up
#: with each step the export reports, and between steps creeps on towards
#: this share of what is left, slowing as it goes, so it never stands still
#: and never runs past the next step (each step is a good deal more than
#: this share away).
EXPORT_FILL_CATCH_MS: Final[int] = 80
EXPORT_FILL_CREEP_MS: Final[int] = 1500
EXPORT_FILL_CREEP_SHARE: Final[float] = 0.3
#: How long the button says the export is done, and the cross-fade back.
EXPORT_DONE_HOLD_MS: Final[int] = 1200
EXPORT_DONE_FADE_MS: Final[int] = 160
#: The size of the tick drawn before "exported".
EXPORT_TICK_SIZE: Final[int] = 12

#: Opacity the floating view bar fades to after the pointer leaves (PLAN.md 3.3).
VIEW_BAR_IDLE_OPACITY: Final[float] = 0.40
#: How far the window is dimmed behind a dialog (PLAN.md 3.11).
SCRIM_OPACITY: Final[float] = 0.60



def font_stack_css() -> str:
    """Return the UI font stack as a CSS ``font-family`` value."""
    return ", ".join(
        name if name == "sans-serif" else f'"{name}"' for name in FONT_STACK_UI
    )


def build_qss() -> str:
    """Build the dark theme stylesheet from the tokens in this module.

    The stylesheet is generated at runtime rather than stored as a static
    ``.qss`` file so that every colour and size has exactly one definition
    (PLAN.md section 3.5).
    """
    font_family = font_stack_css()
    return f"""
/* Generated from ui/theme.py - do not edit by hand. */

/* Plain widgets stay transparent so that a container never paints over the
   region behind it; each region below sets its own background. */
QWidget {{
    background-color: transparent;
    color: {TEXT_PRIMARY};
    font-family: {font_family};
    font-size: {FONT_SIZE_BODY}px;
    font-weight: {FONT_WEIGHT_NORMAL};
}}

QMainWindow, QDialog {{
    background-color: {BG_APP};
}}

QToolTip {{
    background-color: {BG_RAISED};
    color: {TEXT_PRIMARY};
    border: {BORDER_WIDTH}px solid {BORDER};
    border-radius: {RADIUS_CONTROL}px;
    padding: {SPACE_XS}px {SPACE_SM}px;
    font-size: {FONT_SIZE_CAPTION}px;
}}

/* ---------------------------------------------------------------- regions */

#Toolbar {{
    background-color: {BG_APP};
    border-bottom: {BORDER_WIDTH}px solid {BORDER};
}}

#PreviewArea, #PreviewCanvas, #WelcomeView, #PhotoPage {{
    background-color: {BG_CANVAS};
}}

/* The three backgrounds the settings offer behind the photo (PLAN.md 3.10). */
#PreviewArea[background="black"], #PreviewCanvas[background="black"] {{
    background-color: {PREVIEW_BG_BLACK};
}}

#PreviewArea[background="dark"], #PreviewCanvas[background="dark"] {{
    background-color: {PREVIEW_BG_DARK};
}}

#PreviewArea[background="mid"], #PreviewCanvas[background="mid"] {{
    background-color: {PREVIEW_BG_MID};
}}


/* While a photo is being dragged onto the window (PLAN.md 3.7). The page on
   top of the preview covers its edges, so the outline is drawn on whichever
   page is showing as well as on the area itself. */
#PreviewArea[dropTarget="true"], #PhotoPage[dropTarget="true"] {{
    border: {TAB_UNDERLINE_HEIGHT}px dashed {TEXT_TERTIARY};
}}

/* The info card over the photo (PLAN.md 3.9): solid, framed by a line, no
   shadow and nothing seen through it (3.4). */
#InfoCard {{
    background-color: {BG_PANEL};
    border: {BORDER_WIDTH}px solid {TRACK};
    border-radius: {RADIUS_PANEL}px;
}}

/* The welcome screen (PLAN.md 3.7) draws its name, tagline, recent folders
   and drop hint itself, in `ui/welcome_view.py`, so that its opening can move
   them a letter or a row at a time. */

/* The short message under the photo (PLAN.md 3.6). */
#Toast {{
    background-color: {BG_RAISED};
    border: {BORDER_WIDTH}px solid {BORDER};
    border-radius: {RADIUS_CONTROL}px;
}}

#Toast QLabel {{
    color: {TEXT_PRIMARY};
    font-size: {FONT_SIZE_LABEL}px;
}}

/* The floating bar under the photo (PLAN.md 3.3). Fully rounded ends, so
   that it reads as floating over the photo rather than as a piece of a
   panel; one flat colour, because 3.5 rules out a frosted one. */
#ViewBar {{
    background-color: {BG_PANEL};
    border: {BORDER_WIDTH}px solid {BORDER};
    border-radius: {RADIUS_VIEW_BAR}px;
    /* Less the border, which Qt adds to a stylesheet height, so that the bar
       is exactly twice its corner radius tall and the ends are semicircles. */
    min-height: {VIEW_BAR_HEIGHT - 2 * BORDER_WIDTH}px;
}}

#ViewBar QPushButton {{
    border: none;
    padding: 0 {SPACE_SM}px;
    min-height: {SLIDER_ROW_HEIGHT}px;
    color: {TEXT_SECONDARY};
}}

#ViewBar QPushButton:hover {{
    background-color: {BG_RAISED};
    color: {TEXT_PRIMARY};
}}

/* The before and after button is held down rather than clicked (PLAN.md 3.3),
   so its pressed state is the one the user looks at the longest. */
#ViewBar QPushButton:pressed {{
    background-color: {TRACK};
    color: {TEXT_PRIMARY};
}}

#ViewBar QPushButton:disabled {{
    background-color: transparent;
    color: {TEXT_TERTIARY};
}}

#ViewBar QLabel {{
    color: {TEXT_SECONDARY};
    font-size: {FONT_SIZE_LABEL}px;
}}

/* Shown in the top left corner of the preview while the photo as it was
   found is on screen (PLAN.md 3.3). */
#BeforeLabel {{
    background-color: {BG_PANEL};
    border: {BORDER_WIDTH}px solid {BORDER};
    border-radius: {RADIUS_CONTROL}px;
    color: {TEXT_PRIMARY};
    font-size: {FONT_SIZE_LABEL}px;
    padding: {SPACE_XS}px {SPACE_SM}px;
}}

#Inspector, #CollagePanel, #WatermarkPanel, #PortraitPanel, #LiquifyPanel {{
    background-color: {BG_PANEL};
    border-left: {BORDER_WIDTH}px solid {BORDER};
}}

#Filmstrip {{
    background-color: {BG_PANEL};
    border-top: {BORDER_WIDTH}px solid {BORDER};
}}

/* The bookmark on the filmstrip's top left edge (PLAN.md 3.1): the strip's
   own colour and line, open at the bottom where it joins the strip. */
QPushButton#FilmstripTab {{
    background-color: {BG_PANEL};
    border: {BORDER_WIDTH}px solid {BORDER};
    border-bottom: none;
    border-top-left-radius: {RADIUS_CONTROL}px;
    border-top-right-radius: {RADIUS_CONTROL}px;
    padding: 0;
    min-width: {FILMSTRIP_TAB_WIDTH - 2 * BORDER_WIDTH}px;
    max-width: {FILMSTRIP_TAB_WIDTH - 2 * BORDER_WIDTH}px;
    min-height: {FILMSTRIP_TAB_HEIGHT - BORDER_WIDTH}px;
    max-height: {FILMSTRIP_TAB_HEIGHT - BORDER_WIDTH}px;
}}

QPushButton#FilmstripTab:hover {{
    background-color: {BG_RAISED};
}}

QPushButton#FilmstripTab:pressed {{
    background-color: {TRACK};
}}

QPushButton#FilmstripTab:disabled {{
    background-color: {BG_PANEL};
}}

#Histogram {{
    background-color: {BG_PANEL};
    border-bottom: {BORDER_WIDTH}px solid {BORDER};
}}

/* Which photo of the folder is open, at the right hand end of the filmstrip
   (PLAN.md 3.1). On a patch of its own, so that it is not read as a label
   belonging to the thumbnail beside it. */
#FilmstripPosition {{
    background-color: {BG_APP};
    border-radius: {RADIUS_CHIP}px;
    color: {TEXT_SECONDARY};
    font-size: {FONT_SIZE_CAPTION}px;
    min-height: {CHIP_HEIGHT}px;
    padding: 0 {SPACE_SM}px;
}}

#PanelActionRow, #TabBar {{
    background-color: {BG_PANEL};
}}

/* The settings dialog: categories down the left, their page on the right
   (PLAN.md 3.10). */
#SettingsCategories {{
    background-color: {BG_PANEL};
    border-right: {BORDER_WIDTH}px solid {BORDER};
    padding: {SPACE_SM}px;
    outline: none;
}}

#SettingsCategories::item {{
    color: {TEXT_SECONDARY};
    border-radius: {RADIUS_CONTROL}px;
    padding: {SPACE_SM}px;
}}

#SettingsCategories::item:hover {{
    background-color: {BG_RAISED};
    color: {TEXT_PRIMARY};
}}

#SettingsCategories::item:selected {{
    background-color: {TRACK};
    color: {TEXT_PRIMARY};
}}

#SettingsPage {{
    background-color: {BG_APP};
}}


#PanelActionRow {{
    border-top: {BORDER_WIDTH}px solid {BORDER};
}}

/* The tab bar is separated from the page by the tabs' own underline, so it
   carries no border of its own; a border here would clip that underline. */

#TabPage {{
    background-color: {BG_PANEL};
}}

/* ---------------------------------------------------------------- labels */

QLabel {{
    background-color: transparent;
    color: {TEXT_PRIMARY};
}}

QLabel[role="title"] {{
    font-size: {FONT_SIZE_TITLE}px;
    font-weight: {FONT_WEIGHT_TITLE};
}}

QLabel[role="secondary"] {{
    color: {TEXT_SECONDARY};
    font-size: {FONT_SIZE_LABEL}px;
}}

QLabel[role="caption"] {{
    color: {TEXT_TERTIARY};
    font-size: {FONT_SIZE_CAPTION}px;
}}

/* The name beside a control that is disabled - a switch, a slider - goes
   the same grey as the control (PLAN.md 3.8). */
QLabel:disabled,
QLabel[role="secondary"]:disabled {{
    color: {TEXT_TERTIARY};
}}

/* A button on a group's header sits on the header's own band, which is
   already BG_RAISED while the pointer is over either, so it lights a step
   further than an icon button elsewhere. */
QPushButton[role="group"] > QPushButton[role="icon"]:hover {{
    background-color: {TRACK_HOVER};
}}

QPushButton[role="group"] > QPushButton[role="icon"]:pressed {{
    background-color: {TRACK_PRESSED};
}}

/* The same for the remove button on a recent folder of the welcome screen:
   the row under it is BG_RAISED whenever the button can be seen. */
QPushButton[role="recent_row"] > QPushButton[role="icon"]:hover {{
    background-color: {TRACK_HOVER};
}}

QPushButton[role="recent_row"] > QPushButton[role="icon"]:pressed {{
    background-color: {TRACK_PRESSED};
}}

/* A group title while a slider of that group is moving (PLAN.md 3.6). */
QLabel[role="faded"] {{
    color: {BG_RAISED};
    font-size: {FONT_SIZE_CAPTION}px;
}}

/* The one line on the welcome screen that says what the program is for
   (PLAN.md 3.7). It is the second thing read after the name, so it is set
   above the folder names listed below it rather than under them: at the
   label size it was the smallest text on the screen but one, and the eye
   went to the list instead. */
/* --------------------------------------------------------------- buttons */

/* A stylesheet height is the height of the contents and Qt adds the border
   to it, so every rule below takes the border off the token it is built
   from; otherwise a 28 px button comes out 30 and sits off the 4 px grid of
   PLAN.md 3.5. A rule that turns the border off asks for the token itself. */
QPushButton {{
    background-color: transparent;
    color: {TEXT_PRIMARY};
    border: {BORDER_WIDTH}px solid {BORDER};
    border-radius: {RADIUS_CONTROL}px;
    padding: 0 {SPACE_MD}px;
    min-height: {BUTTON_HEIGHT - 2 * BORDER_WIDTH}px;
    font-size: {FONT_SIZE_BODY}px;
}}

/* An outlined button brightens its edge as well as its face, the way the
   welcome screen's button does; the two were answering the pointer
   differently. */
QPushButton:hover {{
    background-color: {BG_RAISED};
    border-color: {TEXT_TERTIARY};
}}

QPushButton:pressed {{
    background-color: {TRACK};
}}

QPushButton:disabled {{
    color: {TEXT_TERTIARY};
    background-color: transparent;
}}

QPushButton:focus {{
    border: {FOCUS_RING_WIDTH}px solid {ACCENT};
}}

/* The single filled primary button in the whole interface (PLAN.md 3.5). */
QPushButton[role="primary"] {{
    background-color: {TEXT_PRIMARY};
    color: {BG_APP};
    /* A border of its own colour, so that the focus ring can take its place
       without the button growing by two pixels. */
    border: {BORDER_WIDTH}px solid {TEXT_PRIMARY};
    font-weight: {FONT_WEIGHT_TITLE};
    padding: 0 {SPACE_LG}px;
    min-height: {BUTTON_HEIGHT - 2 * BORDER_WIDTH}px;
}}

QPushButton[role="primary"]:hover {{
    background-color: {WHITE};
    border-color: {WHITE};
}}

QPushButton[role="primary"]:pressed {{
    background-color: {TEXT_SECONDARY};
    border-color: {TEXT_SECONDARY};
}}

QPushButton[role="primary"]:disabled {{
    background-color: {BG_RAISED};
    border-color: {BG_RAISED};
    color: {TEXT_TERTIARY};
}}

/* On a light face the ring is the dark of the window; the accent ring of
   every other button would be invisible here. */
QPushButton[role="primary"]:focus {{
    border: {FOCUS_RING_WIDTH}px solid {BG_APP};
}}

/* The one button of the welcome screen. Outlined rather than filled: the
   filled button of PLAN.md 3.5 is the export button, and a solid block the
   width of the content is a heavier thing than this screen needs. */
QPushButton[role="hero"] {{
    background-color: transparent;
    color: {TEXT_PRIMARY};
    border: {BORDER_WIDTH}px solid {TRACK};
    border-radius: {RADIUS_CONTROL}px;
    padding: 0 {SPACE_LG}px;
    font-size: {FONT_SIZE_BODY}px;
    /* A stylesheet height is the height of the contents, and Qt adds the
       border to it, so the border is taken off here to land on the size the
       token asks for. */
    min-height: {WELCOME_BUTTON_HEIGHT - 2 * BORDER_WIDTH}px;
}}

QPushButton[role="hero"]:pressed {{
    background-color: {TRACK};
}}

QPushButton[role="hero"]:disabled {{
    background-color: transparent;
    color: {TEXT_TERTIARY};
    border-color: {BORDER};
}}

QPushButton[role="hero"]:focus {{
    border: {FOCUS_RING_WIDTH}px solid {ACCENT};
}}

/* One of a set of choices, such as a crop ratio (PLAN.md 3.2). The chosen
   one is the current state, which is what the accent colour is for. */
QPushButton[role="choice"] {{
    padding: 0 {SPACE_XS}px;
    font-size: {FONT_SIZE_LABEL}px;
    color: {TEXT_SECONDARY};
}}

QPushButton[role="choice"]:hover {{
    background-color: {BG_RAISED};
    color: {TEXT_PRIMARY};
}}

QPushButton[role="choice"]:pressed {{
    background-color: {TRACK};
}}

QPushButton[role="choice"]:checked {{
    color: {ACCENT};
    border-color: {ACCENT};
}}

QPushButton[role="choice"]:disabled {{
    color: {TEXT_TERTIARY};
    border-color: {BORDER};
}}

/* A row of the looks tab: the photo as it would leave it on the left, the name
   after it, both read from the left edge (PLAN.md 3.1). The thumbnail is the
   button's icon, so it takes the button's states with it. */
/* A saved preset in the adjust tab's list: a name, read from the left edge,
   the text 8 px in from the outside of the border. */
QPushButton[role="choice"][listed="true"] {{
    text-align: left;
    padding: 0 {SPACE_SM - BORDER_WIDTH}px;
}}

QPushButton[role="choice"][thumbnail="true"] {{
    text-align: left;
    padding: 0 {(FILTER_ROW_HEIGHT - FILTER_THUMBNAIL_SIZE) // 2 - BORDER_WIDTH}px;
    min-height: {FILTER_ROW_HEIGHT - 2 * BORDER_WIDTH}px;
}}

/* Borderless icon button, 28 x 28 px with an 18 px icon. */
QPushButton[role="icon"] {{
    background-color: transparent;
    border: {BORDER_WIDTH}px solid transparent;
    border-radius: {RADIUS_CONTROL}px;
    padding: 0;
    min-width: {ICON_BUTTON_SIZE - 2 * BORDER_WIDTH}px;
    max-width: {ICON_BUTTON_SIZE - 2 * BORDER_WIDTH}px;
    min-height: {ICON_BUTTON_SIZE - 2 * BORDER_WIDTH}px;
    max-height: {ICON_BUTTON_SIZE - 2 * BORDER_WIDTH}px;
}}

QPushButton[role="icon"]:hover {{
    background-color: {BG_RAISED};
}}

QPushButton[role="icon"]:pressed {{
    background-color: {TRACK};
}}

/* The smaller icon button (SMALL_ICON_BUTTON_SIZE, with the small icon). */
QPushButton[role="icon_small"] {{
    background-color: transparent;
    border: {BORDER_WIDTH}px solid transparent;
    border-radius: {RADIUS_CONTROL}px;
    padding: 0;
    min-width: {SMALL_ICON_BUTTON_SIZE - 2 * BORDER_WIDTH}px;
    max-width: {SMALL_ICON_BUTTON_SIZE - 2 * BORDER_WIDTH}px;
    min-height: {SMALL_ICON_BUTTON_SIZE - 2 * BORDER_WIDTH}px;
    max-height: {SMALL_ICON_BUTTON_SIZE - 2 * BORDER_WIDTH}px;
}}

QPushButton[role="icon_small"]:hover {{
    background-color: {BG_RAISED};
}}

QPushButton[role="icon_small"]:pressed {{
    background-color: {TRACK};
}}

/* A mode that is on, such as the collage (PLAN.md 3.13): the accent marks
   the current state, as it does on a chosen segment. */
QPushButton[role="icon"]:checked {{
    background-color: {BG_RAISED};
    border-color: {ACCENT};
}}

QPushButton[role="icon"]:disabled {{
    background-color: transparent;
}}

QPushButton[role="icon"]:focus {{
    border: {FOCUS_RING_WIDTH}px solid {ACCENT};
}}

/* Panel tab: icon + text, current tab marked by a white underline. The
   underline is part of the button's height, so the content height is the bar
   height minus the underline; otherwise the button overflows the bar and the
   underline is clipped away. */
QPushButton[role="tab"] {{
    background-color: transparent;
    border: none;
    border-bottom: {TAB_UNDERLINE_HEIGHT}px solid {BORDER};
    border-radius: 0;
    color: {TEXT_SECONDARY};
    font-size: {FONT_SIZE_LABEL}px;
    min-height: {TAB_BAR_HEIGHT - TAB_UNDERLINE_HEIGHT}px;
    /* No side padding. Five tabs share 300 px, which is 60 px each, and
       measured with Inter the longest English label - Presets, not the new
       Filters - wants 65 px with 4 px either side and 57 px without. The
       tabs are equally wide and their contents centred, so the padding was
       only ever buying air between words that the equal widths already
       give. */
    padding: 0;
}}

QPushButton[role="tab"]:hover {{
    color: {TEXT_PRIMARY};
    background-color: {BG_RAISED};
}}

QPushButton[role="tab"]:pressed {{
    background-color: {TRACK};
}}

QPushButton[role="tab"]:checked {{
    color: {TEXT_PRIMARY};
    border-bottom: {TAB_UNDERLINE_HEIGHT}px solid {TEXT_PRIMARY};
    font-weight: {FONT_WEIGHT_TITLE};
}}

QPushButton[role="tab"]:disabled {{
    color: {TEXT_TERTIARY};
}}

/* The title row of a collapsible group in the right hand panel (PLAN.md
   3.1). Heavier and brighter than the names of the sliders below it: a
   group title one step fainter than its own contents reads as a remark
   rather than as a heading. The whole width is pressable, and the line that
   separates one group from the next is drawn above it. */
QPushButton[role="group"] {{
    background-color: transparent;
    border: {BORDER_WIDTH}px solid transparent;
    border-radius: 0;
    color: {TEXT_PRIMARY};
    font-size: {FONT_SIZE_LABEL}px;
    font-weight: {FONT_WEIGHT_TITLE};
    min-height: {GROUP_HEADER_HEIGHT - 2 * BORDER_WIDTH}px;
    padding: 0 {PANEL_PADDING}px;
    text-align: left;
}}

QPushButton[role="group"]:hover {{
    background-color: {BG_RAISED};
}}

QPushButton[role="group"]:pressed {{
    background-color: {TRACK};
}}

QPushButton[role="group"]:disabled {{
    background-color: transparent;
    color: {TEXT_TERTIARY};
}}

QPushButton[role="group"]:focus {{
    border: {FOCUS_RING_WIDTH}px solid {ACCENT};
}}

/* A group title while a slider of that group is moving (PLAN.md 3.6). */
QPushButton[role="group"][faded="true"] {{
    color: {BG_RAISED};
}}

/* The line between two groups. */
#Divider {{
    background-color: {BORDER};
}}

/* Borderless text button, used for the recent folder list. */
QPushButton[role="link"] {{
    background-color: transparent;
    border: none;
    border-radius: {RADIUS_CONTROL}px;
    /* Asked of the palette rather than named here, so that the 80 ms hover
       transition can set it a step at a time (ui/widgets/hover.py); the
       resting value is {TEXT_SECONDARY}. */
    color: palette(buttontext);
    font-size: {FONT_SIZE_BODY}px;
    min-height: {BUTTON_HEIGHT}px;
    padding: 0 {SPACE_SM}px;
    text-align: left;
}}

QPushButton[role="link"]:pressed {{
    background-color: {TRACK};
}}

QPushButton[role="link"]:disabled {{
    color: {TEXT_TERTIARY};
}}

/* --------------------------------------------------------- input fields */

QLineEdit {{
    background-color: {BG_RAISED};
    color: {TEXT_PRIMARY};
    border: {BORDER_WIDTH}px solid {BORDER};
    border-radius: {RADIUS_CONTROL}px;
    padding: 0 {SPACE_XS}px;
    selection-background-color: {ACCENT};
    selection-color: {BG_APP};
}}

/* The ring is hung on a property rather than on ``:focus``, which matches
   however the focus arrived; PLAN.md 3.6 shows it for the keyboard and not
   while the mouse is in use (see ui/widgets/focus.py). */
QLineEdit[keyboardFocus="true"] {{
    border: {FOCUS_RING_WIDTH}px solid {ACCENT};
}}

/* A document shown in the interface: the about text (PLAN.md 13.5). */
QTextBrowser {{
    background-color: transparent;
    color: {TEXT_PRIMARY};
    border: none;
    selection-background-color: {ACCENT};
    selection-color: {BG_APP};
}}

/* ------------------------------------------------------------- dropdowns */


QComboBox {{
    background-color: {BG_RAISED};
    color: {TEXT_PRIMARY};
    border: {BORDER_WIDTH}px solid {BORDER};
    border-radius: {RADIUS_CONTROL}px;
    padding: 0 {SPACE_SM}px;
    min-height: {BUTTON_HEIGHT - 2 * BORDER_WIDTH}px;
    font-size: {FONT_SIZE_BODY}px;
}}

QComboBox:hover {{
    border-color: {TRACK};
}}

QComboBox[keyboardFocus="true"] {{
    border: {FOCUS_RING_WIDTH}px solid {ACCENT};
}}

QComboBox:disabled {{
    color: {TEXT_TERTIARY};
}}

/* The arrow is drawn by the widget itself (ui/widgets/combo.py), since no
   picture files are shipped for the interface; the button only has to
   leave room for it. */
QComboBox::drop-down {{
    border: none;
    width: {SPACE_LG}px;
}}

QComboBox::down-arrow {{
    image: none;
}}

QComboBox QAbstractItemView {{
    background-color: {BG_RAISED};
    border: {BORDER_WIDTH}px solid {BORDER};
    border-radius: {RADIUS_CONTROL}px;
    padding: {SPACE_XS}px;
    outline: none;
    selection-background-color: {TRACK};
    selection-color: {TEXT_PRIMARY};
}}

/* ----------------------------------------------------------------- menus */


QMenu {{
    background-color: {BG_RAISED};
    border: {BORDER_WIDTH}px solid {BORDER};
    border-radius: {RADIUS_PANEL}px;
    padding: {SPACE_XS}px;
}}

QMenu::item {{
    background-color: transparent;
    color: {TEXT_PRIMARY};
    border-radius: {RADIUS_CONTROL}px;
    padding: {SPACE_SM}px {SPACE_MD}px;
    font-size: {FONT_SIZE_BODY}px;
}}

QMenu::item:selected {{
    background-color: {TRACK};
}}

QMenu::item:disabled {{
    color: {TEXT_TERTIARY};
}}

QMenu::separator {{
    height: {BORDER_WIDTH}px;
    background-color: {BORDER};
    margin: {SPACE_XS}px {SPACE_SM}px;
}}

/* ---------------------------------------------------------- scroll areas */

QScrollArea {{
    background-color: transparent;
    border: none;
}}

QScrollBar:vertical {{
    background-color: transparent;
    width: {SCROLLBAR_WIDTH}px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background-color: {TRACK};
    border-radius: {UNIT}px;
    min-height: {SPACE_XXL}px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {TRACK_HOVER};
}}

QScrollBar::handle:vertical:pressed {{
    background-color: {TRACK_PRESSED};
}}

/* The right hand panel keeps its scroll bar gutter at all times, so that
   folding a group away cannot move every control sideways. Qt leaves an
   empty bar enabled and draws its handle down the whole track, which says
   there is more to see when there is not, so the handle is not drawn
   (see ui/widgets/scroll.py). This outranks the two rules above it, so the
   pointer cannot light up a gutter with nothing behind it either. */

QScrollBar[empty="true"]::handle:vertical {{
    background-color: transparent;
}}

QScrollBar:horizontal {{
    background-color: transparent;
    height: {SCROLLBAR_WIDTH}px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background-color: {TRACK};
    border-radius: {UNIT}px;
    min-width: {SPACE_XXL}px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {TRACK_HOVER};
}}

QScrollBar::handle:horizontal:pressed {{
    background-color: {TRACK_PRESSED};
}}

QScrollBar::add-line, QScrollBar::sub-line,
QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
    border: none;
    height: 0;
    width: 0;
}}
"""
