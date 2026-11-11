"""Every keyboard shortcut, in one table (PLAN.md section 8).

The window registers the ones it can from :data:`REGISTERED`, and the
settings dialog shows the whole of :data:`SHORTCUTS` as a read only list,
which is why the table also holds the ones no shortcut object can express:
a key held down, a double click, or an arrow key whose meaning depends on
what has the focus.

Nothing here is text for the screen: a row carries the keys of its label and
of its combination, and :func:`app.i18n.tr` turns them into words.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class Shortcut:
    """One row of the shortcut list."""

    #: What it does, as a text key.
    label_key: str
    #: The combination, as a text key. Those under ``shortcut.`` are also
    #: what the window hands to Qt; those under ``shortcut.keys.`` are only
    #: ever shown, because Qt cannot express them.
    keys_key: str


#: Section 8 in full, in the order it is written there.
SHORTCUTS: Final[tuple[Shortcut, ...]] = (
    Shortcut("menu.open_file", "shortcut.open_file"),
    Shortcut("menu.open_folder", "shortcut.open_folder"),
    Shortcut("shortcut.name.recent_remove", "shortcut.recent_remove"),
    Shortcut("toolbar.undo", "shortcut.undo"),
    Shortcut("toolbar.redo", "shortcut.keys.redo"),
    Shortcut("panel.action.copy", "shortcut.copy_settings"),
    Shortcut("panel.action.paste", "shortcut.paste_settings"),
    Shortcut("panel.action.reset", "shortcut.reset"),
    Shortcut("adjust.pick_white_balance", "shortcut.white_balance_pick"),
    Shortcut("adjust.auto_light", "shortcut.auto_light"),
    Shortcut("adjust.spot_tool", "shortcut.spot_tool"),
    Shortcut("shortcut.name.brush_size", "shortcut.keys.spot_size"),
    Shortcut("shortcut.name.spot_delete", "shortcut.spot_delete"),
    Shortcut("panel.action.save_preset", "shortcut.new_preset"),
    Shortcut("toolbar.export", "shortcut.export"),
    Shortcut("shortcut.name.check", "shortcut.keys.check"),
    Shortcut("toolbar.collage", "shortcut.collage"),
    Shortcut("toolbar.split", "shortcut.split"),
    Shortcut("toolbar.watermark", "shortcut.watermark"),
    Shortcut("toolbar.portrait", "shortcut.portrait"),
    Shortcut("portrait.add", "shortcut.portrait_add"),
    Shortcut("portrait.remove", "shortcut.portrait_remove"),
    Shortcut("portrait.show", "shortcut.portrait_show"),
    Shortcut("toolbar.liquify", "shortcut.liquify"),
    Shortcut("shortcut.name.liquify_reverse", "shortcut.keys.liquify_reverse"),
    Shortcut("shortcut.name.liquify_edit_faces", "shortcut.keys.liquify_edit_faces"),
    Shortcut("menu.settings", "shortcut.settings"),
    Shortcut("shortcut.name.crop", "shortcut.crop"),
    Shortcut("shortcut.name.crop_flip", "shortcut.crop_flip"),
    Shortcut("shortcut.name.crop_grid", "shortcut.crop_grid"),
    Shortcut("shortcut.name.crop_apply", "shortcut.keys.crop_apply"),
    Shortcut("shortcut.name.turn", "shortcut.keys.turn"),
    Shortcut("shortcut.name.mirror", "shortcut.keys.mirror"),
    Shortcut("shortcut.name.compare", "shortcut.keys.compare"),
    Shortcut("shortcut.name.zoom_toggle", "shortcut.zoom_toggle"),
    Shortcut("shortcut.name.zoom_steps", "shortcut.keys.zoom_steps"),
    Shortcut("shortcut.name.pan", "shortcut.keys.pan"),
    Shortcut("shortcut.name.photo", "shortcut.keys.photo"),
    Shortcut("shortcut.name.fullscreen", "shortcut.fullscreen"),
    Shortcut("shortcut.name.panels_all", "shortcut.toggle_panels"),
    Shortcut("shortcut.name.panels_each", "shortcut.keys.panels_each"),
    Shortcut("shortcut.name.tabs", "shortcut.keys.tabs"),
    Shortcut("toolbar.info", "shortcut.info"),
    Shortcut("shortcut.name.clipping", "shortcut.clipping"),
    Shortcut("histogram.switch_mode", "shortcut.histogram_mode"),
    Shortcut("shortcut.name.slider_reset", "shortcut.keys.slider_reset"),
    Shortcut("shortcut.name.slider_fine", "shortcut.keys.slider_fine"),
)

#: The combinations the window asks Qt to listen for, in the order they are
#: registered. Several rows above share a handler or have no single
#: combination, so this is not simply the table.
REGISTERED: Final[tuple[str, ...]] = (
    "shortcut.open_file",
    "shortcut.open_folder",
    "shortcut.undo",
    "shortcut.redo",
    "shortcut.redo_alternative",
    "shortcut.copy_settings",
    "shortcut.paste_settings",
    "shortcut.reset",
    "shortcut.white_balance_pick",
    "shortcut.auto_light",
    "shortcut.spot_tool",
    "shortcut.spot_smaller",
    "shortcut.spot_larger",
    "shortcut.new_preset",
    "shortcut.export",
    "shortcut.check_all",
    "shortcut.check_none",
    "shortcut.collage",
    "shortcut.split",
    "shortcut.watermark",
    "shortcut.portrait",
    "shortcut.portrait_add",
    "shortcut.portrait_remove",
    "shortcut.portrait_show",
    "shortcut.liquify",
    "shortcut.settings",
    "shortcut.crop",
    "shortcut.crop_flip",
    "shortcut.crop_grid",
    "shortcut.crop_apply",
    "shortcut.crop_cancel",
    "shortcut.turn_left",
    "shortcut.turn_right",
    "shortcut.mirror_horizontal",
    "shortcut.mirror_vertical",
    "shortcut.zoom_in",
    "shortcut.zoom_in_alternative",
    "shortcut.zoom_out",
    "shortcut.zoom_fit",
    "shortcut.fullscreen",
    # Tab is not here: Qt's focus travel would take it before a shortcut
    # could, so the window intercepts it in focusNextPrevChild() instead.
    "shortcut.toggle_inspector",
    "shortcut.toggle_filmstrip",
    "shortcut.tab_adjust",
    "shortcut.tab_looks",
    "shortcut.tab_retouch",
    "shortcut.tab_crop",
    "shortcut.info",
    "shortcut.clipping",
    "shortcut.histogram_mode",
)
