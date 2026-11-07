"""The main window and its four regions (PLAN.md section 3.1).

Toolbar across the top, preview area in the centre, inspector panel on the
right, filmstrip along the bottom. The window owns the controller and wires
its signals to the regions; it never decodes or processes a photo itself.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Final

from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QPropertyAnimation,
    QRectF,
    Qt,
    QTimer,
)
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from app.batch_export import BatchExporter, BatchExportResult
from app.collage_controller import CollageController
from app.controller import Controller, Preview
from app.i18n import current_language, init_translator, tr
from app.settings import (
    CLIPPING_COLOURED,
    STARTUP_LAST_FOLDER,
    TEMPERATURE_KELVIN,
    Settings,
)

from core import collage as collage_module
from core import geometry
from core.edit_state import EditState
from core.geometry import Geometry
from core.lens import LensSettings
from core.metadata import PhotoMetadata
from ui import native, theme
from ui.filmstrip import Filmstrip, FilmstripTab
from ui.histogram_widget import clipping_overlay
from ui.about_dialog import AboutDialog
from ui.collage_panel import CollagePanel
from ui.liquify_panel import LiquifyPanel
from ui.portrait_panel import BRUSH_ADD, BRUSH_REMOVE, PortraitPanel
from ui.watermark_panel import WatermarkPanel
from ui.collage_view import CollageCanvas
from ui.devlog_dialog import DevlogDialog
from ui.export_dialog import ExportDialog
from ui.presets import NameDialog
from ui.info_card import InfoCard
from ui.inspector import Inspector, InspectorTab
from ui.preview_view import PreviewArea
from ui.retranslate import retranslate_tree
from ui.settings_dialog import SettingsDialog
from ui.shortcuts import REGISTERED
from ui.toolbar import Toolbar
from ui.widgets.dialog import Dialog
from ui.widgets.flight import FlyingPhoto
from ui.widgets.region import Scrim
from ui.widgets.toast import Toast

logger = logging.getLogger(__name__)

#: How much one press of [ or ] changes the spot removal brush (A-32): a
#: quarter, so that eight presses run from a speck of dust to a blemish.
SPOT_SIZE_STEP: Final[float] = 1.25
#: How much of the view the faces fill when their points are placed by hand
#: (L-12): near enough to tell the points apart, with room around them.
FACE_EDIT_SHARE: Final[float] = 0.6


class MainWindow(QMainWindow):
    """The application window."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self._panel_animations: dict[str, QPropertyAnimation] = {}
        # Whether each panel is collapsed. Kept separately from the widget's
        # visibility, which is ambiguous while the collapse animation runs.
        self._collapsed: dict[str, bool] = {}

        self.setWindowTitle(tr("app.name"))
        self.setMinimumSize(theme.WINDOW_MIN_WIDTH, theme.WINDOW_MIN_HEIGHT)
        self.setAcceptDrops(True)

        self.controller = Controller(settings, self)
        self.collage = CollageController(self.controller, settings, self)
        #: Writes the photos ticked in the filmstrip (PLAN.md 2.6, X-09).
        self.batch = BatchExporter(self.controller, self)
        #: The photos about to be written together, while their dialog
        #: closes and the first of them flies into the export button.
        self._batch_sending: list[Path] | None = None
        #: How many photos the running export of several writes.
        self._batch_total = 0
        #: Whether the collage has the middle of the window (PLAN.md 3.13).
        self._collage_mode = False
        #: Whether the watermark is being placed (PLAN.md 3.14).
        self._watermark_mode = False
        #: Whether the portrait is open (PLAN.md 3.15).
        self._portrait_mode = False
        #: Whether liquify is open (PLAN.md 3.16).
        self._liquify_mode = False
        #: How the photo was shown before it was zoomed onto the faces to
        #: place their points (L-12), and which photo that was; the view
        #: goes back to it when placing ends.
        self._view_before_faces: tuple[Path, tuple[bool, float, QPointF]] | None = None
        #: Whether the faces are still to be zoomed onto, because placing
        #: began before they were found.
        self._faces_to_zoom = False
        #: What the watermark on screen was last drawn for, so that a new
        #: preview of the same photo does not draw it again.
        self._watermark_drawn: tuple[object, tuple[int, int]] | None = None
        self._developed = None
        #: Where the pixels on screen belong: the size of the whole photo and
        #: the part of it they cover, so that redrawing them with the clipping
        #: marks on puts them back in the same place (PLAN.md 3.3).
        self._developed_frame: tuple[int, int] | None = None
        self._developed_region: tuple[float, float, float, float] | None = None
        #: The photo on screen, so that a preview of a different one can be
        #: told from the next preview of this one (PLAN.md 3.6).
        self._shown: Path | None = None
        #: Whether what is on screen is a RAW file's own preview, which the
        #: developed photo cross-fades over rather than replacing (3.7).
        self._shown_provisional = False
        #: The photo the developed picture last arrived for, and the timer
        #: that brings up the "developing" sign when one keeps the user
        #: waiting longer than DEVELOPING_DELAY_MS (PLAN.md 3.7).
        self._developed_path: Path | None = None
        self._developing_timer = QTimer(self)
        self._developing_timer.setSingleShot(True)
        self._developing_timer.setInterval(theme.DEVELOPING_DELAY_MS)
        self._developing_timer.timeout.connect(self._show_developing)
        self._screen = None
        #: The export dialog while it is open, so that what it asks about
        #: can be answered (PLAN.md 3.11).
        self._dialog: ExportDialog | None = None
        #: Whether each panel was folded away before going full screen, so
        #: that leaving it puts them back the way they were.
        self._panels_before_full_screen: tuple[bool, bool] = (False, False)
        #: What the user has asked for, which is not always what is on
        #: screen: before a photo is open there is nothing for the panels to
        #: show, so they stay out of the way whatever this says. It is what
        #: gets remembered between runs (U-04).
        self._panels_wanted: dict[str, bool] = {
            "inspector": settings.inspector_collapsed,
            "filmstrip": settings.filmstrip_collapsed,
        }

        self.toolbar = Toolbar(self)
        self.preview = PreviewArea(self)
        #: What is known about the photo, on a card over it (PLAN.md 3.9).
        self.info_card = InfoCard(self.preview.canvas)
        self.info_card.hide()
        self.inspector = Inspector(self)
        self.filmstrip = Filmstrip(self)
        self.toast = Toast(self)
        self.scrim = Scrim(self)
        #: The photo flying into the export button (PLAN.md 3.6).
        self.flight = FlyingPhoto(self)

        central = QWidget(self)
        central.setObjectName("Central")
        middle = QWidget(central)
        middle_layout = QHBoxLayout(middle)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(0)
        middle_layout.addWidget(self.preview, 1)
        middle_layout.addWidget(self.inspector)
        self.collage_canvas = CollageCanvas(self.collage, middle)
        self.collage_panel = CollagePanel(middle)
        self.collage_panel.set_layout(self.collage.layout)
        middle_layout.addWidget(self.collage_canvas, 1)
        middle_layout.addWidget(self.collage_panel)
        self.collage_canvas.hide()
        self.collage_panel.hide()
        #: In the inspector's place while the watermark is open (PLAN.md 3.14).
        self.watermark_panel = WatermarkPanel(middle)
        middle_layout.addWidget(self.watermark_panel)
        self.watermark_panel.hide()
        #: In the inspector's place while the portrait is open (PLAN.md 3.15).
        self.portrait_panel = PortraitPanel(middle)
        middle_layout.addWidget(self.portrait_panel)
        self.portrait_panel.hide()
        #: In the inspector's place while liquify is open (PLAN.md 3.16).
        self.liquify_panel = LiquifyPanel(middle)
        middle_layout.addWidget(self.liquify_panel)
        self.liquify_panel.hide()

        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(middle, 1)
        layout.addWidget(self.filmstrip)
        self.setCentralWidget(central)
        # Made after everything else in the window, so that it is drawn over
        # the photo it stands on (PLAN.md 3.1).
        self.filmstrip_tab = FilmstripTab(self.filmstrip, central)
        self.filmstrip_tab.hide()

        self._connect()
        self._register_shortcuts()
        # Tab is caught in eventFilter, which only an application wide filter
        # sees before the focus widget turns the key into focus traversal.
        application = QApplication.instance()
        if application is not None:
            application.installEventFilter(self)
        self._restore_state()
        self.apply_settings()
        self.preview.setFocus()

    # -- wiring ----------------------------------------------------------

    def _connect(self) -> None:
        """Join the controller to the regions that show what it produces."""
        self.controller.folder_opened.connect(self._on_folder_opened)
        self.controller.selection_changed.connect(self._on_selection_changed)
        self.controller.preview_ready.connect(self._on_preview_ready)
        self.controller.metadata_ready.connect(self._on_metadata_ready)
        self.controller.thumbnail_ready.connect(self.filmstrip.set_thumbnail)
        self.controller.adjustments_ready.connect(self.inspector.set_adjustments)
        self.controller.adjustments_ready.connect(self._on_photo_announced)
        self.controller.state_restored.connect(self.inspector.adjust_tab.set_state)
        self.controller.state_restored.connect(self.portrait_panel.set_state)
        self.controller.state_restored.connect(self.inspector.looks_tab.set_state)
        self.controller.geometry_changed.connect(self._on_geometry_changed)
        self.controller.history_changed.connect(self._on_history_changed)
        self.controller.edited_photos_ready.connect(self._on_edited_photos)
        self.controller.edited_changed.connect(self._on_edited_changed)
        self.controller.presets_changed.connect(self.inspector.adjust_tab.set_presets)
        self.controller.clipboard_changed.connect(self._on_clipboard_changed)
        self.controller.announcement.connect(self.toast.show_message)
        self.controller.error.connect(self.toast.show_message)
        self.controller.error.connect(lambda _message: self._stop_developing())
        self.controller.export_progress.connect(self._on_export_progress)
        self.controller.export_finished.connect(self._on_export_finished)
        self.controller.export_failed.connect(self._on_export_failed)
        self.batch.progress.connect(self._on_export_progress)
        self.batch.advanced.connect(self._on_batch_advanced)
        self.batch.finished.connect(self._on_batch_finished)
        self.filmstrip.choosing_changed.connect(lambda _choosing: self._show_export_count())
        self.filmstrip.checked_changed.connect(lambda _photos: self._show_export_count())

        self.inspector.adjust_tab.value_set.connect(self.controller.set_adjustment)
        self.inspector.retouch_tab.lens_set.connect(self._on_lens_set)
        self.controller.lens_ready.connect(self.inspector.retouch_tab.set_lens_profile)
        self.inspector.histogram.clipping_toggled.connect(self._on_clipping_toggled)
        self.inspector.histogram.set_mode(self._settings.histogram_mode)
        self.preview.canvas.resized.connect(self._on_preview_resized)
        self.preview.canvas.view_changed.connect(self.controller.set_view)
        # Reset goes through the controller, which owns the state the photo was
        # found in and makes the reset one step of history (E-05).
        reset_button = self.inspector.action_buttons(InspectorTab.ADJUST)[-1]
        reset_button.clicked.connect(self.controller.reset)
        self.inspector.adjust_tab.pick_button.toggled.connect(self.preview.canvas.set_picking)
        self.inspector.adjust_tab.auto_button.clicked.connect(self.auto_light)
        self.preview.canvas.white_balance_picked.connect(self._on_white_balance_picked)
        self.controller.white_balance_refused.connect(self._on_white_balance_refused)

        retouch = self.inspector.retouch_tab
        spots = self.preview.canvas.spots
        retouch.spot_button.toggled.connect(self._on_spot_tool_toggled)
        retouch.spot_size_set.connect(self._on_spot_size_set)
        retouch.spots_cleared.connect(self.controller.clear_spots)
        spots.spot_painted.connect(self._on_spot_painted)
        spots.spot_moved.connect(self._on_spot_moved)
        spots.selection_changed.connect(self._on_spot_selected)
        self.preview.canvas.spot_deleted.connect(self._on_spot_deleted)

        self.toolbar.undo_button.clicked.connect(self.controller.undo)
        self.toolbar.redo_button.clicked.connect(self.controller.redo)
        self.toolbar.export_button.clicked.connect(self.export)
        self.preview.view_bar.comparing.connect(self.controller.set_comparing)

        self._connect_crop()
        self._connect_presets()
        self._connect_filters()
        self._connect_collage()
        self._connect_watermark()
        self._connect_portrait()
        self._connect_liquify()

    def _connect_liquify(self) -> None:
        """Join the liquify panel and brush to the controller (PLAN.md 3.16)."""
        panel = self.liquify_panel
        panel.reset_requested.connect(self.controller.reset_liquify)
        panel.leave_requested.connect(self.leave_liquify)
        panel.face_value_set.connect(self.controller.set_face_shape_value)
        panel.editing_toggled.connect(self.set_editing_faces)
        panel.face_marks_reset_requested.connect(self.controller.reset_face_marks)
        self.preview.canvas.faces.handle_moved.connect(self._on_face_handle_moved)
        panel.tool_changed.connect(lambda _tool: self._show_liquify_brush())
        panel.brush_size_set.connect(lambda _radius, _dragging: self._show_liquify_brush())
        overlay = self.preview.canvas.liquify
        overlay.stroke_changed.connect(self._on_liquify_stroke_changed)
        overlay.stroke_painted.connect(self._on_liquify_painted)
        self.toolbar.liquify_button.clicked.connect(self.toggle_liquify)

    def _connect_portrait(self) -> None:
        """Join the portrait's panel and brush to the controller (PLAN.md 3.15)."""
        panel = self.portrait_panel
        panel.value_set.connect(self.controller.set_adjustment)
        panel.auto_set.connect(self.controller.set_skin_auto)
        panel.reset_requested.connect(self.controller.reset_skin)
        panel.leave_requested.connect(self.leave_portrait)
        panel.brush_changed.connect(self._show_portrait_brush)
        panel.brush_size_set.connect(lambda _radius, _dragging: self._show_portrait_brush())
        panel.showing_changed.connect(self._on_portrait_showing)
        self.controller.adjustments_ready.connect(
            lambda state, _shot: self.portrait_panel.set_state(state)
        )
        self.controller.faces_ready.connect(self._on_faces_ready)
        self.controller.skin_shown.connect(self._on_skin_shown)
        self.preview.canvas.skin.stroke_painted.connect(self._on_skin_painted)
        self.toolbar.portrait_button.clicked.connect(self.toggle_portrait)

    def _connect_watermark(self) -> None:
        """Join the watermark's panel and overlay to the settings (PLAN.md 3.14)."""
        panel = self.watermark_panel
        panel.spec_changed.connect(self._on_watermark_spec)
        panel.enabled_changed.connect(self._on_watermark_enabled)
        panel.leave_requested.connect(self.leave_watermark)
        self.preview.canvas.watermark.moved.connect(self._on_watermark_moved)
        self.toolbar.watermark_button.clicked.connect(self.toggle_watermark)

    def _connect_collage(self) -> None:
        """Join the collage's panel and canvas to its controller (PLAN.md 3.13)."""
        collage = self.collage
        panel = self.collage_panel
        panel.layout_changed.connect(collage.set_layout)
        panel.choose_requested.connect(self.choose_collage_photo)
        panel.leave_requested.connect(self.leave_collage)
        self.collage_canvas.choose_requested.connect(self.choose_collage_photo)
        panel.zoom_changed.connect(collage.set_zoom)
        panel.alignment_scale_changed.connect(collage.set_alignment_scale)
        panel.aligning_toggled.connect(self.collage_canvas.set_aligning)
        panel.aligning_slot_changed.connect(collage.set_aligning_slot)
        collage.layout_changed.connect(self._on_collage_layout_changed)
        collage.placement_changed.connect(self._on_collage_placement_changed)
        collage.photo_changed.connect(self._on_collage_photo_changed)
        collage.error.connect(self.toast.show_message)
        collage.export_progress.connect(self._on_export_progress)
        collage.export_finished.connect(self._on_export_finished)
        collage.export_failed.connect(self._on_export_failed)
        self.toolbar.collage_button.clicked.connect(self.toggle_collage)
        self.toolbar.info_button.clicked.connect(
            lambda: self.set_info_shown(self.toolbar.info_button.isChecked())
        )

    def _connect_filters(self) -> None:
        """Join the filter tab to the controller."""
        tab = self.inspector.looks_tab
        tab.chosen.connect(self._on_filter_chosen)
        tab.strength_changed.connect(self._on_filter_strength)
        self.controller.look_thumbnails_ready.connect(tab.set_thumbnails)
        self.inspector.set_filters(self.controller.filters)

    def _on_filter_chosen(self, identifier: str) -> None:
        """Put the chosen film look on the photo."""
        if self.controller.current is None:
            return
        self.controller.set_adjustments(self.controller.state.with_filter(identifier))

    def _on_filter_strength(self, value: float, dragging: bool) -> None:
        """Follow the strength slider, which drags like any other."""
        if self.controller.current is None:
            return
        self.controller.set_adjustments(
            self.controller.state.with_value("filter_strength", value), dragging
        )

    def _connect_presets(self) -> None:
        """Join the saved presets and the clipboard to the controller (P-01 to P-06)."""
        copy, paste, _reset = self.inspector.action_buttons(InspectorTab.ADJUST)
        copy.clicked.connect(self.copy_adjustments)
        paste.clicked.connect(self.paste_adjustments)
        paste.setEnabled(False)

        tab = self.inspector.adjust_tab
        tab.preset_applied.connect(self.controller.apply_preset)
        tab.preset_renamed.connect(self.controller.rename_preset)
        tab.preset_deleted.connect(self.controller.delete_preset)
        tab.save_preset_button.clicked.connect(self.save_preset)
        self.filmstrip.selection_changed.connect(self._on_filmstrip_selection)
        self.filmstrip_tab.clicked.connect(self.toggle_filmstrip)
        self.controller.load_presets()

    def _connect_crop(self) -> None:
        """Join the crop tab and the crop box to the controller (PLAN.md 3.2)."""
        crop_tab = self.inspector.crop_tab
        crop_tab.aspect_chosen.connect(self._on_aspect_chosen)
        crop_tab.orientation_flipped.connect(self._on_orientation_flipped)
        crop_tab.angle_changed.connect(self._on_angle_changed)
        crop_tab.turn_requested.connect(self._on_turn_requested)
        crop_tab.mirror_requested.connect(self._on_mirror_requested)
        crop_tab.straighten_toggled.connect(self._on_straighten_toggled)

        box = self.preview.canvas.crop
        box.crop_changed.connect(self._on_crop_dragged)
        box.angle_changed.connect(self._on_angle_changed)
        box.straighten_finished.connect(lambda: crop_tab.set_straightening(False))

        self.inspector.tab_changed.connect(self._on_tab_changed)
        reset, cancel, done = self.inspector.action_buttons(InspectorTab.CROP)
        reset.clicked.connect(self.controller.reset_crop)
        cancel.clicked.connect(self.cancel_crop)
        done.clicked.connect(self.finish_crop)

        self.filmstrip.photo_selected.connect(self._on_thumbnail_clicked)
        self.preview.canvas.previous_requested.connect(self.controller.select_previous)
        self.preview.canvas.next_requested.connect(self.controller.select_next)

        menu = self.toolbar.menu
        menu.open_file_action.setEnabled(True)
        menu.open_file_action.triggered.connect(self.open_file_dialog)
        menu.open_folder_action.setEnabled(True)
        menu.open_folder_action.triggered.connect(self.open_folder_dialog)
        menu.folder_chosen.connect(self._open_path)
        menu.recent_cleared.connect(self._clear_recent_folders)
        menu.settings_action.setEnabled(True)
        menu.settings_action.triggered.connect(self.open_settings)
        menu.devlog_action.setEnabled(True)
        menu.devlog_action.triggered.connect(self.open_devlog)
        menu.about_action.setEnabled(True)
        menu.about_action.triggered.connect(self.open_about)


        self.preview.welcome.open_file_requested.connect(self.open_file_dialog)
        self.preview.welcome.open_folder_requested.connect(self.open_folder_dialog)
        self.preview.welcome.folder_requested.connect(lambda path: self._open_path(Path(path)))
        self.preview.welcome.folder_removed.connect(self._forget_folder)
        self.preview.welcome.set_opening_enabled(True)

    # -- opening photos --------------------------------------------------

    def open_file_dialog(self) -> None:
        """Ask for one or more photos and open just those (F-01)."""
        patterns = " ".join(f"*{suffix}" for suffix in sorted(_supported_suffixes()))
        chosen, _filter = QFileDialog.getOpenFileNames(
            self,
            tr("dialog.open_file"),
            self._settings.browse_folder or "",
            f"{tr('dialog.photo_filter')} ({patterns});;{tr('dialog.all_files')} (*)",
        )
        if chosen:
            self.controller.open_paths([Path(path) for path in chosen])

    def open_folder_dialog(self) -> None:
        """Ask for a folder and open every photo in it (F-02)."""
        chosen = QFileDialog.getExistingDirectory(
            self, tr("dialog.open_folder"), self._settings.browse_folder or ""
        )
        if chosen:
            self._open_path(Path(chosen))

    def _open_path(self, path: Path) -> None:
        self.controller.open(path)

    def _show_recent_folders(self) -> None:
        """Show the recent folders in the menu and on the welcome screen."""
        recent = self._settings.recent_folders
        self.toolbar.menu.set_recent_folders(recent)
        self.preview.welcome.set_recent_folders(recent, self._settings.recent_details)

    def _forget_folder(self, folder: str) -> None:
        """Take *folder* off the recent list; the folder itself stays as it is."""
        self._settings.forget_folder(folder)
        self._settings.save()
        self._show_recent_folders()

    def _clear_recent_folders(self) -> None:
        """Empty the recent list; no folder or photo is touched."""
        self._settings.clear_recent()
        self._settings.save()
        self._show_recent_folders()

    def _on_folder_opened(self, photos: list[Path]) -> None:
        """Fill the filmstrip and start making its thumbnails."""
        # Another folder starts in the editor, where the panels are set up
        # for what was opened.
        self.leave_watermark()
        self.leave_portrait()
        self.leave_liquify()
        self.filmstrip.set_photos(photos)
        self.controller.request_thumbnails(photos)
        self._show_recent_folders()
        # There is something to show now, so the panels come back.
        self._apply_panel_visibility()


    def _on_selection_changed(self, index: int, total: int) -> None:
        self.filmstrip.set_current(index, total)
        # A selected spot belongs to the photo it was selected on.
        self.preview.canvas.spots.select(-1)
        # So does the skin shown in the portrait; the next photo's follows.
        self.preview.canvas.skin.set_skin(None)
        # A photo moved to starts the wait over: the sign comes up only if
        # this one is still being developed DEVELOPING_DELAY_MS from now.
        self.preview.set_developing(False)
        self._developing_timer.start()
        self._enable_photo_modes()
        if self._collage_mode:
            # The collage keeps the toolbar until it is left.
            return
        current = self.controller.current
        if current is not None and not self._watermark_mode:
            # The watermark keeps its title while photos are stepped through.
            self.toolbar.set_title(current.parent.name, current.name)
        self._enable_export()

    # -- exporting (PLAN.md 2.6, 3.11) -----------------------------------

    def export(self) -> None:
        """Ask how to export the photo, and then do it in the background.

        The dialog closes the moment it is accepted and the progress appears
        on the toolbar button, so that nothing covers the photo (X-07).
        While photos are ticked in the filmstrip it writes those (X-09), and
        while they are being written, pressing it again stops them.
        """
        if self.batch.exporting:
            self._stop_batch()
            return
        if self._collage_mode:
            self._export_collage()
            return
        if self.filmstrip.choosing:
            self._export_batch()
            return
        photo = self.controller.current
        if photo is None or self.controller.exporting:
            return

        dialog = ExportDialog(
            photo, self._settings.export, self, watermark=self._settings.watermark
        )
        dialog.options_changed.connect(self._on_export_options)
        self.controller.export_estimate_ready.connect(dialog.set_estimate)
        self._dialog = dialog
        # The dialog settled on its first settings while it was being built,
        # before there was anyone to tell, so it is answered once here.
        self._on_export_options(dialog.options)
        try:
            accepted = self._run_dialog(dialog, send_off=True)
        finally:
            self.controller.export_estimate_ready.disconnect(dialog.set_estimate)
            self._dialog = None
        if not accepted:
            return

        self._remember_export(dialog)
        button = self.toolbar.export_button
        button.start(self.flight.remaining_ms())
        if self.controller.export(dialog.target, dialog.options):
            button.setEnabled(False)
        else:
            self._call_off_export()

    def _remember_export(self, dialog: ExportDialog) -> None:
        """Keep what an accepted export dialog chose, for next time (PLAN.md 3.11).

        Its watermark switch is the panel's switch too (W-09), unless there
        was no text to switch on.
        """
        self._settings.export = dialog.chosen_settings
        if not self._settings.watermark.spec.is_empty:
            self._settings.watermark.enabled = dialog.watermark_enabled
            self.watermark_panel.set_state(
                dialog.watermark_enabled, self._settings.watermark.spec
            )

    def _on_export_options(self, options: object) -> None:
        """Tell the dialog what its settings would come to (PLAN.md 3.11)."""
        dialog = self._dialog
        if dialog is None:
            return
        if self._collage_mode:
            dialog.set_output_size(*self.collage.output_size())
            dialog.set_copying(False)
            self.collage.request_export_estimate(options)
            return
        dialog.set_output_size(*self.controller.export_output_size(options))
        dialog.set_copying(self.controller.export_will_copy(options))
        self.controller.request_export_estimate(options)

    def _on_export_progress(self, fraction: float) -> None:
        """Show how far the export has got, on the button itself (X-07)."""
        self.toolbar.export_button.set_progress(fraction)

    def _on_export_finished(self, result: object) -> None:
        """Say where the file went, with an offer to go and look at it."""
        self.toolbar.export_button.finish()
        self._enable_export()
        key = "export.done_copied" if result.copied else "export.done"
        message = tr(key, name=result.path.name)
        singles = getattr(result, "singles", ())
        if singles:
            message = tr("export.done_carousel", name=result.path.name, count=len(singles))
        self.toast.show_message(
            message,
            tr("export.reveal"),
            lambda path=result.path: native.reveal(path),
        )

    def _on_export_failed(self, message: str) -> None:
        """Say in one sentence that the file was not written."""
        self._end_export()
        self.toast.show_message(message)

    # -- exporting several photos (PLAN.md 2.6 X-09, 3.11) ---------------

    def _show_export_count(self) -> None:
        """Name the export button by how many photos are ticked, while choosing."""
        strip = self.filmstrip
        self.toolbar.set_export_count(len(strip.checked_photos()) if strip.choosing else None)
        self._enable_export()

    def check_all_photos(self) -> None:
        """Tick every photo, while photos are being chosen (Ctrl+A)."""
        self.filmstrip.check_all()

    def check_no_photos(self) -> None:
        """Untick every photo, while photos are being chosen (Ctrl+D)."""
        self.filmstrip.check_none()

    def _export_batch(self) -> None:
        """Ask how to write the ticked photos, then write them one by one."""
        paths = self.filmstrip.checked_photos()
        if not paths or self.controller.exporting or self.collage.exporting:
            return
        dialog = ExportDialog(
            paths[0],
            self._settings.export,
            self,
            batch=len(paths),
            watermark=self._settings.watermark,
        )
        self._batch_sending = paths
        try:
            accepted = self._run_dialog(dialog, send_off=True)
        finally:
            self._batch_sending = None
        if not accepted:
            return

        self._remember_export(dialog)
        # The photos are on their way, so what was ticked has done its job.
        self.filmstrip.stop_choosing()
        self._batch_total = len(paths)
        button = self.toolbar.export_button
        self._say_batch_photo(1)
        button.set_stoppable(True)
        self.toolbar.refit_export()
        button.start(self.flight.remaining_ms())
        if self.batch.export(paths, dialog.folder, dialog.suffix, dialog.options):
            self._enable_export()
        else:
            self._call_off_export()

    def _say_batch_photo(self, index: int, words: str | None = None) -> None:
        """Say on the button which photo is being written, or *words*."""
        total = self._batch_total
        widest = tr("toolbar.exporting_count", index=total, total=total)
        if words is None:
            words = tr("toolbar.exporting_count", index=index, total=total)
        self.toolbar.export_button.set_working_words(words, widest)

    def _on_batch_advanced(self, index: int, total: int) -> None:
        """Count the photos on the button as they are written."""
        self._batch_total = total
        if not self.batch.stopping:
            self._say_batch_photo(index)

    def _stop_batch(self) -> None:
        """Stop the photos being written once the one in hand is done."""
        self.batch.stop()
        button = self.toolbar.export_button
        button.set_stoppable(False)
        self._say_batch_photo(0, tr("toolbar.stopping"))
        self._enable_export()

    def _on_batch_finished(self, result: BatchExportResult) -> None:
        """Say how many photos were written, with an offer to go and look."""
        written = len(result.written)
        reveal = None
        if written:
            reveal = lambda path=result.written[0]: native.reveal(path)  # noqa: E731
        if result.stopped:
            self._end_export()
            message = tr("export.batch_stopped", written=written, total=result.total)
        elif not written:
            self._end_export()
            self.toast.show_message(tr("export.batch_failed"))
            return
        else:
            self.toolbar.export_button.finish()
            self._enable_export()
            if result.failed:
                message = tr("export.batch_partial", written=written, failed=len(result.failed))
            else:
                message = tr("export.batch_done", count=written)
        if reveal is None:
            self.toast.show_message(message)
        else:
            self.toast.show_message(message, tr("export.reveal"), reveal)

    def _end_export(self) -> None:
        """Put the export button back the way it rests."""
        self.flight.stop()
        self.toolbar.export_button.set_progress(None)
        self._enable_export()

    def _call_off_export(self) -> None:
        """Take back the start of an export that did not begin after all."""
        self.flight.stop()
        self.toolbar.export_button.rest()
        self._enable_export()

    def _run_dialog(self, dialog: Dialog, send_off: bool = False) -> bool:
        """Show *dialog* over the dimmed window; return whether it was accepted.

        The dim fades out as the dialog starts to fade, rather than after it
        (PLAN.md 3.6). With *send_off*, accepting it also sends the photo
        flying into the export button at that same moment.
        """

        def closing(accepted: bool) -> None:
            self.scrim.uncover()
            if accepted and send_off:
                self._send_off_photo()

        dialog.closing.connect(closing)
        self.scrim.cover(self)
        try:
            return dialog.exec() == Dialog.DialogCode.Accepted
        finally:
            self.scrim.uncover()

    def _send_off_photo(self) -> None:
        """Fly what is being exported into the export button (PLAN.md 3.6).

        The collage flies as one picture, also when it writes its photos one
        by one as well. Several photos send the first of them: the photo on
        screen when it is one of them, otherwise its thumbnail.
        """
        sending = self._batch_sending
        if sending is not None and self.controller.current not in sending:
            self._send_off_thumbnail(sending[0])
            return
        if self._collage_mode:
            source: QWidget = self.collage_canvas
            area = self.collage_canvas.canvas_rect()
        else:
            source = self.preview.canvas
            area = self.preview.canvas.photo_rect()
        area = area.intersected(QRectF(source.rect()))
        if area.isEmpty() or not source.isVisible():
            return
        picture = source.grab(area.toAlignedRect())
        start = QRectF(area.toAlignedRect()).translated(
            QPointF(source.mapTo(self, QPoint(0, 0)))
        )
        button = self.toolbar.export_button
        target = QPointF(button.mapTo(self, button.rect().center()))
        self.flight.launch(picture, start, target)

    def _send_off_thumbnail(self, path: Path) -> None:
        """Fly the thumbnail of *path* into the export button, if it can be seen."""
        thumbnail = self.filmstrip.thumbnail(path)
        if (
            thumbnail is None
            or thumbnail.picture is None
            or thumbnail.visibleRegion().isEmpty()
        ):
            return
        start = QRectF(thumbnail.rect()).translated(
            QPointF(thumbnail.mapTo(self, QPoint(0, 0)))
        )
        button = self.toolbar.export_button
        target = QPointF(button.mapTo(self, button.rect().center()))
        self.flight.launch(thumbnail.picture, start, target)

    def _enable_export(self) -> None:
        """Let the export button follow whether there is anything to export.

        While several photos are being written it stays pressable, which is
        how they are stopped, until it has been pressed (X-09).
        """
        busy = self.controller.exporting or self.collage.exporting
        if self._collage_mode:
            ready = self.collage.complete
        elif self.filmstrip.choosing:
            ready = bool(self.filmstrip.checked_photos())
        else:
            ready = self.controller.current is not None
        enabled = ready and not busy
        if self.batch.exporting:
            enabled = not self.batch.stopping
        self.toolbar.export_button.setEnabled(enabled)
        self.collage_canvas.setEnabled(not self.collage.exporting)
        self.collage_panel.setEnabled(not self.collage.exporting)

    # -- the collage (PLAN.md 2.8, 3.13) ---------------------------------

    # -- the portrait (PLAN.md 2.10, 3.15) --------------------------------

    def toggle_portrait(self) -> None:
        """Open the portrait, or go back to editing if it is open (P)."""
        if self._portrait_mode:
            self.leave_portrait()
        else:
            self.enter_portrait()

    def enter_portrait(self) -> None:
        """Put the portrait's panel in the inspector's place, its brush on the photo.

        The tools that take the click on the photo are put down and a crop in
        progress is finished first, as for the watermark. Undo and redo go on
        working: what is done here is done to the photo. The watermark or the
        collage, if open, is left first, so the modes switch in one click.
        """
        if self._portrait_mode or self.controller.current is None:
            self.toolbar.portrait_button.setChecked(self._portrait_mode)
            return
        self.leave_watermark()
        self.leave_collage()
        self.leave_liquify()
        if self.controller.cropping:
            self.finish_crop()
        self.set_spot_tool(False)
        self.set_picking_white_balance(False)
        self._portrait_mode = True
        self.toolbar.portrait_button.setChecked(True)
        panel = self.portrait_panel
        panel.set_state(self.controller.state)
        panel.set_skin(self.controller.skin)
        panel.set_faces(self._faces_count())
        self.inspector.hide()
        panel.show()
        self.preview.canvas.skin.set_open(True)
        self._show_portrait_brush()
        self._on_portrait_showing(panel.showing)
        self.controller.request_skin_shown()
        self.preview.canvas.setFocus()

    def leave_portrait(self) -> None:
        """Go back to editing; the skin range stays with each photo."""
        if not self._portrait_mode:
            return
        self._portrait_mode = False
        self.portrait_panel.set_brush(None)
        canvas = self.preview.canvas
        canvas.skin.set_open(False)
        canvas.skin.set_skin(None)
        canvas.update()
        self.toolbar.portrait_button.setChecked(False)
        self.portrait_panel.hide()
        if not self.is_inspector_collapsed():
            self.inspector.show()
        self.preview.setFocus()

    @property
    def in_portrait(self) -> bool:
        """Return whether the portrait is open."""
        return self._portrait_mode

    def toggle_portrait_brush(self, brush: str) -> None:
        """Take the brush up for *brush*, or put it down if it already does that (B, E)."""
        panel = self.portrait_panel
        panel.set_brush(None if panel.brush == brush else brush)

    def toggle_portrait_showing(self) -> None:
        """Show the skin over the photo, or stop showing it (M)."""
        self.portrait_panel.set_showing(not self.portrait_panel.showing)

    def _faces_count(self) -> int | None:
        """Return how many faces the photo now open holds, or ``None`` while looking."""
        found = self.controller.faces_of_current()
        return None if found is None else len(found)

    def _show_portrait_brush(self) -> None:
        """Give the photo's brush what the panel says, at the size it is on screen."""
        panel = self.portrait_panel
        brush = panel.brush
        canvas = self.preview.canvas
        has_photo = self.controller.current is not None
        canvas.skin.set_brush(
            self._portrait_mode and brush is not None and has_photo,
            brush != BRUSH_REMOVE,
            self.controller.shown_radius(panel.brush_size) if has_photo else 0.0,
        )
        canvas.refresh_cursor()
        canvas.update()

    def _on_portrait_showing(self, showing: bool) -> None:
        self.preview.canvas.skin.set_showing(showing)
        self.preview.canvas.update()

    def _on_faces_ready(self, count: int | None) -> None:
        """Say how many faces were found, and show the skin found beside them."""
        self.portrait_panel.set_faces(count)
        self.liquify_panel.set_faces(count)
        self._show_face_guides()
        if self._portrait_mode and count:
            self.controller.request_skin_shown()

    def _on_skin_shown(self, skin: object) -> None:
        if not self._portrait_mode:
            return
        self.preview.canvas.skin.set_skin(skin)  # type: ignore[arg-type]
        self.preview.canvas.update()

    def _on_skin_painted(self, points: list[tuple[float, float]]) -> None:
        """Add to the skin or take from it, as the brush says."""
        panel = self.portrait_panel
        brush = panel.brush
        if brush is None:
            return
        self.controller.add_skin_stroke(
            points, panel.brush_size, adding=brush == BRUSH_ADD, snap=panel.snap
        )

    # -- liquify (PLAN.md 2.11, 3.16) ------------------------------------

    def toggle_liquify(self) -> None:
        """Open liquify, or go back to editing if it is open (L)."""
        if self._liquify_mode:
            self.leave_liquify()
        else:
            self.enter_liquify()

    def enter_liquify(self) -> None:
        """Put the liquify panel in the inspector's place, its brush on the photo.

        As for the portrait: the tools that take the click on the photo are
        put down, a crop in progress is finished, and another mode, if open,
        is left first (L-01). Undo and redo go on working.
        """
        if self._liquify_mode or self.controller.current is None:
            self.toolbar.liquify_button.setChecked(self._liquify_mode)
            return
        self.leave_portrait()
        self.leave_watermark()
        self.leave_collage()
        if self.controller.cropping:
            self.finish_crop()
        self.set_spot_tool(False)
        self.set_picking_white_balance(False)
        self._liquify_mode = True
        self.toolbar.liquify_button.setChecked(True)
        panel = self.liquify_panel
        panel.set_strokes(len(self.controller.liquify_strokes))
        panel.set_face_shape(self.controller.face_shape)
        panel.set_faces(self._faces_count())
        self.inspector.hide()
        panel.show()
        self.preview.canvas.liquify.set_open(True)
        self._show_liquify_brush()
        self.preview.canvas.setFocus()

    def leave_liquify(self) -> None:
        """Go back to editing; the strokes stay with each photo."""
        if not self._liquify_mode:
            return
        self.set_editing_faces(False)
        self._liquify_mode = False
        canvas = self.preview.canvas
        canvas.liquify.set_open(False)
        canvas.refresh_cursor()
        canvas.update()
        self.toolbar.liquify_button.setChecked(False)
        self.liquify_panel.hide()
        if not self.is_inspector_collapsed():
            self.inspector.show()
        self.preview.setFocus()

    @property
    def in_liquify(self) -> bool:
        """Return whether liquify is open."""
        return self._liquify_mode

    def _show_liquify_brush(self) -> None:
        """Give the photo's brush what the panel says, at the size it is on screen."""
        panel = self.liquify_panel
        canvas = self.preview.canvas
        has_photo = self.controller.current is not None
        canvas.liquify.set_brush(
            panel.tool, self.controller.shown_radius(panel.brush_size) if has_photo else 0.0
        )
        canvas.refresh_cursor()
        canvas.update()

    def set_editing_faces(self, editing: bool) -> None:
        """Start or stop placing the face's points by hand (L-12).

        While they are placed the liquify brush is put down and the photo is
        shown without liquify; letting go of the button shapes it again.
        """
        editing = editing and self._liquify_mode
        canvas = self.preview.canvas
        was_editing = canvas.faces.active
        self.liquify_panel.set_editing(editing)
        self.controller.set_editing_faces(editing)
        canvas.faces.set_open(editing)
        canvas.liquify.set_open(self._liquify_mode and not editing)
        current = self.controller.current
        if editing and not was_editing and current is not None:
            # The faces are brought up close, and the view is put back after.
            self._view_before_faces = (current, canvas.view_state())
            self._faces_to_zoom = True
        self._show_face_guides()
        if not editing and was_editing:
            self._faces_to_zoom = False
            before = self._view_before_faces
            self._view_before_faces = None
            if before is not None and before[0] == current:
                canvas.restore_view(before[1])
        canvas.refresh_cursor()
        canvas.update()

    @property
    def editing_faces(self) -> bool:
        """Return whether the face's points are being placed by hand."""
        return self.preview.canvas.faces.active

    def _show_face_guides(self) -> None:
        """Give the photo the face's points as they are now, while they are placed."""
        canvas = self.preview.canvas
        if not canvas.faces.active:
            return
        guides = self.controller.face_guides_on_screen()
        canvas.faces.set_guides(guides)
        if self._faces_to_zoom and guides:
            self._faces_to_zoom = False
            xs = [x for guide in guides for x, _ in guide.handles]
            ys = [y for guide in guides for _, y in guide.handles]
            canvas.zoom_onto((min(xs), min(ys), max(xs), max(ys)), FACE_EDIT_SHARE)
        canvas.update()

    def _on_face_handle_moved(
        self, face: int, handle: str, x: float, y: float, dragging: bool
    ) -> None:
        """Put a face's point where it was dragged; letting go is one step."""
        self.controller.move_face_handle(face, handle, (x, y), dragging)

    def _on_liquify_stroke_changed(self, points: list[tuple[float, float]]) -> None:
        """Show the stroke being painted as a draft (PLAN.md 3.16)."""
        panel = self.liquify_panel
        self.controller.draft_liquify(points, panel.tool, panel.brush_size, panel.strength)

    def _on_liquify_painted(self, points: list[tuple[float, float]]) -> None:
        """Reshape the photo along the stroke just painted."""
        panel = self.liquify_panel
        self.controller.add_liquify_stroke(points, panel.tool, panel.brush_size, panel.strength)

    # -- the watermark (PLAN.md 2.9, 3.14) -------------------------------

    def toggle_watermark(self) -> None:
        """Open the watermark, or go back to editing if it is open (T)."""
        if self._watermark_mode:
            self.leave_watermark()
        else:
            self.enter_watermark()

    def enter_watermark(self) -> None:
        """Place the watermark over the photo, its panel in the inspector's place.

        The photo on screen is the photo as it will be exported - adjusted
        and cropped - so a crop in progress is finished first, and the tools
        that take the click on the photo are put down. The portrait or the
        collage, if open, is left first, so the modes switch in one click.
        """
        if self._watermark_mode or self.controller.current is None:
            self.toolbar.watermark_button.setChecked(self._watermark_mode)
            return
        self.leave_portrait()
        self.leave_liquify()
        self.leave_collage()
        if self.controller.cropping:
            self.finish_crop()
        self.set_spot_tool(False)
        self.set_picking_white_balance(False)
        self.controller.set_comparing(False)
        self._watermark_mode = True
        self.toolbar.watermark_button.setChecked(True)
        self.toolbar.set_title(None, tr("watermark.title"))
        self.toolbar.undo_button.setEnabled(False)
        self.toolbar.redo_button.setEnabled(False)
        self.preview.view_bar.compare_button.setEnabled(False)
        stored = self._settings.watermark
        self.watermark_panel.set_state(stored.enabled, stored.spec)
        self.inspector.hide()
        self.watermark_panel.show()
        self.preview.canvas.watermark.set_active(True)
        self._watermark_drawn = None
        self._draw_watermark()
        self.preview.canvas.setFocus()

    def leave_watermark(self) -> None:
        """Go back to editing; the watermark stays set for the next export."""
        if not self._watermark_mode:
            return
        self._watermark_mode = False
        canvas = self.preview.canvas
        canvas.watermark.set_active(False)
        canvas.update()
        self.toolbar.watermark_button.setChecked(False)
        self.preview.view_bar.compare_button.setEnabled(True)
        self.watermark_panel.hide()
        if not self.is_inspector_collapsed():
            self.inspector.show()
        current = self.controller.current
        if current is not None:
            self.toolbar.set_title(current.parent.name, current.name)
        self._on_history_changed(self.controller.can_undo, self.controller.can_redo)
        self.preview.setFocus()

    @property
    def in_watermark(self) -> bool:
        """Return whether the watermark is being placed."""
        return self._watermark_mode

    def _draw_watermark(self) -> None:
        """Draw the watermark for the photo on screen, unless it already is."""
        if not self._watermark_mode:
            return
        canvas = self.preview.canvas
        spec = self._settings.watermark.spec
        frame = canvas.frame_size
        if self._watermark_drawn == (spec, frame):
            return
        self._watermark_drawn = (spec, frame)
        picture = self.controller.watermark_picture(spec, frame)
        if picture is None:
            canvas.watermark.set_picture(spec, None, 1.0)
            self.watermark_panel.set_missing_font(False)
        else:
            canvas.watermark.set_picture(spec, picture.rgba, picture.scale)
            self.watermark_panel.set_missing_font(picture.missing_glyphs)
        canvas.update()

    def _on_watermark_spec(self, spec: object) -> None:
        """Take on what the panel says the watermark is now.

        Text typed into an empty watermark turns it on for exports, since a
        watermark being written is one that is wanted (W-09).
        """
        stored = self._settings.watermark
        if stored.spec.is_empty and not spec.is_empty and not stored.enabled:
            stored.enabled = True
            self.watermark_panel.set_state(True, spec)
        stored.spec = spec
        self._draw_watermark()

    def _on_watermark_enabled(self, enabled: bool) -> None:
        """Remember whether exported photos carry the watermark (W-09)."""
        self._settings.watermark.enabled = enabled

    def _on_watermark_moved(self, spec: object, settled: bool) -> None:
        """Take on where the watermark was dragged; the picker follows."""
        self._settings.watermark.spec = spec
        self._watermark_drawn = (spec, self.preview.canvas.frame_size)
        self.watermark_panel.set_spec(spec)
        if settled:
            self.preview.canvas.update()

    def toggle_collage(self) -> None:
        """Open the collage, or go back to editing if it is open."""
        if self._collage_mode:
            self.leave_collage()
        else:
            self.enter_collage()

    def enter_collage(self) -> None:
        """Lay two to four photos out on one canvas, in place of the preview.

        Photos chosen together in the filmstrip go straight in. Photos
        already in the collage are developed again, since they may have been
        edited since they were put there. The portrait or the watermark, if
        open, is left first, so the modes switch in one click.
        """
        if self._collage_mode:
            return
        self.leave_portrait()
        self.leave_liquify()
        self.leave_watermark()
        if self.controller.cropping:
            self.finish_crop()
        self.set_spot_tool(False)
        self.controller.flush()
        self.collage.refresh()
        chosen = self.filmstrip.selected_photos()
        if len(chosen) >= collage_module.COUNT_MIN:
            self.collage.fill(chosen)
        self._collage_mode = True
        # The collage exports itself, not photos ticked in the strip.
        self.filmstrip.set_choosing_allowed(False)
        # The card is about the photo being edited, which the collage covers.
        self.set_info_shown(False)
        self.toolbar.info_button.setEnabled(False)
        self._enable_photo_modes()
        self._show_toolbar_for_welcome()
        self.toolbar.collage_button.setChecked(True)
        self.preview.hide()
        self.inspector.hide()
        self.collage_canvas.show()
        self.collage_panel.show()
        self.collage_panel.set_paths(self.collage.paths)
        self.toolbar.set_title(None, tr("collage.title"))
        self.toolbar.undo_button.setEnabled(False)
        self.toolbar.redo_button.setEnabled(False)
        self._enable_export()
        self.collage_canvas.setFocus()

    def leave_collage(self) -> None:
        """Go back to the photo being edited; the collage is kept for later."""
        if not self._collage_mode:
            return
        self._collage_mode = False
        self.filmstrip.set_choosing_allowed(True)
        self._show_toolbar_for_welcome()
        self.toolbar.info_button.setEnabled(True)
        self._enable_photo_modes()
        self.toolbar.collage_button.setChecked(False)
        self.collage_canvas.hide()
        self.collage_panel.hide()
        self.preview.show()
        self.inspector.show()
        current = self.controller.current
        if current is not None:
            self.toolbar.set_title(current.parent.name, current.name)
        else:
            self.toolbar.set_title(None, None)
        self._on_history_changed(self.controller.can_undo, self.controller.can_redo)
        self._enable_export()
        self.preview.setFocus()

    @property
    def in_collage(self) -> bool:
        """Return whether the collage is on screen."""
        return self._collage_mode

    def _enable_photo_modes(self) -> None:
        """Let the portrait and the watermark be opened while there is a photo for them.

        The collage can be opened from the welcome screen, before any photo
        is; once one is, either button leaves the collage for it.
        """
        has_photo = self.controller.current is not None
        self.toolbar.portrait_button.setEnabled(has_photo)
        self.toolbar.liquify_button.setEnabled(has_photo)
        self.toolbar.watermark_button.setEnabled(has_photo)

    def choose_collage_photo(self, slot: int) -> None:
        """Ask for the photo to put in one half of the collage."""
        patterns = " ".join(f"*{suffix}" for suffix in sorted(_supported_suffixes()))
        folder = self.collage.default_folder()
        start = str(folder) if folder is not None else (self._settings.browse_folder or "")
        chosen, _filter = QFileDialog.getOpenFileName(
            self,
            tr("collage.choose_title"),
            start,
            f"{tr('dialog.photo_filter')} ({patterns});;{tr('dialog.all_files')} (*)",
        )
        if chosen:
            self.collage.load(slot, Path(chosen))

    def _on_collage_layout_changed(self, layout: object) -> None:
        """Show the new layout; a different count shows other photos and readiness."""
        self.collage_panel.set_layout(layout)
        self.collage_panel.set_paths(self.collage.paths)
        if self._collage_mode:
            self._enable_export()

    def _on_collage_placement_changed(self) -> None:
        """Let the zoom and alignment sliders follow what the canvas did."""
        collage = self.collage
        aligning = collage.aligning_slot
        self.collage_panel.show_placements(
            tuple(collage.placement(slot).zoom for slot in range(collage_module.SLOTS)),
            collage.alignment(aligning).scale,
            aligning,
        )

    def _on_collage_photo_changed(self, _slot: int, _photo: object) -> None:
        self.collage_panel.set_paths(self.collage.paths)
        if self._collage_mode:
            self._enable_export()

    def _export_collage(self) -> None:
        """Ask how to write the collage, then write it in the background."""
        if not self.collage.complete or self.collage.exporting:
            return
        first = next(path for path in self.collage.paths if path is not None)
        dialog = ExportDialog(
            first,
            self._settings.export,
            self,
            collage_name=self.collage.default_name(),
            collage_size=self.collage.output_size(),
            carousel=self._settings.collage.carousel,
            watermark=self._settings.watermark,
        )
        dialog.options_changed.connect(self._on_export_options)
        self.collage.export_estimate_ready.connect(dialog.set_estimate)
        self._dialog = dialog
        self._on_export_options(dialog.options)
        try:
            accepted = self._run_dialog(dialog, send_off=True)
        finally:
            self.collage.export_estimate_ready.disconnect(dialog.set_estimate)
            self._dialog = None
        if not accepted:
            return
        self._remember_export(dialog)
        self._settings.collage.carousel = dialog.carousel
        self.toolbar.export_button.start(self.flight.remaining_ms())
        if self.collage.export(dialog.target, dialog.options, dialog.carousel):
            self._enable_export()
        else:
            self._call_off_export()


    def _on_preview_ready(self, preview: Preview) -> None:
        """Show a developed photo, with the clipping marked if asked for."""
        if preview.backdrop:
            # Not what is on screen but what lies under it, for panning to
            # uncover. It is the whole photo, so the histogram can describe
            # the edit from it while the preview is zoomed in (PLAN.md 3.3).
            self.preview.set_backdrop(self._marked(preview.image), preview.frame)
            if not preview.before:
                self.inspector.histogram.set_photo(preview.image)
            return
        self._developed = preview.image
        self._developed_frame = preview.frame
        self._developed_region = preview.region
        self.preview.set_showing_before(preview.before)
        # Only a photo the user has moved to fades in (PLAN.md 3.6). The
        # previews that follow a slider are the same photo again, and
        # fading each one would smear the adjustment.
        moved_to = not preview.before and preview.path != self._shown
        # The photo as developed here taking over from the camera's own
        # preview of it: cross-faded, because the two differ in colour and a
        # swap reads as a flicker (PLAN.md 3.7).
        blend = (
            not moved_to
            and self._shown_provisional
            and not preview.provisional
            and not preview.before
            and preview.region is None
        )
        self._shown = preview.path
        if not preview.before:
            self._shown_provisional = preview.provisional
        self.preview.show_photo(
            self._marked(preview.image),
            fade=moved_to,
            frame=preview.frame,
            region=preview.region,
            blend=blend,
        )
        # Another photo, or another crop, may be another shape: the watermark
        # is drawn again for it (PLAN.md 3.14).
        self._draw_watermark()
        if not preview.provisional and preview.path == self.controller.current:
            self._developed_path = preview.path
            self._stop_developing()
        if preview.region is not None:
            # Only what is on screen was developed, so its counts are not the
            # photo's own; the histogram keeps the ones it has (PLAN.md 3.3).
            return
        if preview.before:
            # The histogram keeps describing the edit while the photo as it
            # was found is glanced at, so that it stays a steady reference.
            return
        if not preview.draft:
            # A draft is developed small, so its counts would not be the
            # photo's own; the full size one follows in a moment.
            self.inspector.histogram.set_photo(preview.image)

    def _marked(self, image) -> object:
        """Return the photo with its clipped pixels marked, if that is on."""
        highlights, shadows = self.inspector.histogram.showing_clipping
        return clipping_overlay(
            image,
            highlights,
            shadows,
            coloured=self._settings.clipping_marks == CLIPPING_COLOURED,
        )

    def _on_clipping_toggled(self, _highlights: bool, _shadows: bool) -> None:
        """Redraw the photo with or without the clipping marks (A-13)."""
        if self._developed is not None:
            self.preview.show_photo(
                self._marked(self._developed),
                frame=self._developed_frame,
                region=self._developed_region,
            )

    def _show_developing(self) -> None:
        """Bring up the sign, if the photo on screen is still not developed."""
        current = self.controller.current
        if current is not None and current != self._developed_path and not self._collage_mode:
            self.preview.set_developing(True)

    def _stop_developing(self) -> None:
        """The photo has arrived, or will not: no more waiting to be said."""
        self._developing_timer.stop()
        self.preview.set_developing(False)

    def _on_history_changed(self, can_undo: bool, can_redo: bool) -> None:
        """Enable undo and redo according to the current photo's history."""
        if self._collage_mode or self._watermark_mode:
            # Neither the collage nor the watermark is in the photo's history.
            return
        self.toolbar.undo_button.setEnabled(can_undo)
        self.toolbar.redo_button.setEnabled(can_redo)

    def _on_preview_resized(self, long_edge: int) -> None:
        """Develop the preview at the size it is actually shown."""
        self.controller.set_preview_long_edge(long_edge)

    # -- cropping (PLAN.md 3.2) ------------------------------------------

    def _on_photo_announced(self, _state: EditState, _shot: tuple[float, float]) -> None:
        """Put the controller back into crop mode for a photo that just arrived.

        Moving to another photo leaves crop mode behind, and a photo may
        arrive long after the crop tab was opened; either way the tab on
        screen is what says whether the whole photo belongs there
        (PLAN.md 3.2).

        This waits for the photo rather than following every geometry
        change, because a geometry announced while the crop tab is still on
        screen is also how leaving it with Esc puts the crop back: taken for
        a photo arriving, that would put the controller straight back into
        crop mode and develop the whole photo twice over for nothing.
        """
        if self.inspector.current_tab is InspectorTab.CROP:
            self._set_cropping(True)

    def _on_geometry_changed(self, geometry_now: Geometry) -> None:
        """Show a geometry the controller settled on, without sending it back."""
        self.inspector.set_geometry(geometry_now)
        box = self.preview.canvas.crop
        box.set_source(self.controller.photo_size)
        box.set_geometry(geometry_now)
        self._show_spots()
        if self._portrait_mode:
            self.portrait_panel.set_skin(geometry_now.skin)
            self._show_portrait_brush()
            self.controller.request_skin_shown()
        if self._liquify_mode:
            self.liquify_panel.set_strokes(len(geometry_now.liquify))
            self.liquify_panel.set_face_shape(geometry_now.face_shape)
            self._show_liquify_brush()
            self._show_face_guides()
        self.preview.canvas.update()

    def _on_lens_set(self, settings: LensSettings, dragging: bool) -> None:
        """Apply a new lens correction (PLAN.md 2.2).

        It is part of the geometry, so it goes the way a crop does: the crop
        tab and the crop box keep a copy of the geometry, and a copy without
        the new correction would put the old one back with the next drag.
        """
        self._change_geometry(replace(self.controller.geometry, lens=settings), dragging)

    def _change_geometry(self, geometry_now: Geometry, dragging: bool = False) -> None:
        """Apply a new geometry and show it everywhere at once."""
        self.controller.set_geometry(geometry_now, dragging)
        self.inspector.crop_tab.set_geometry(geometry_now)
        box = self.preview.canvas.crop
        box.set_source(self.controller.photo_size)
        box.set_geometry(geometry_now)
        self._show_spots()
        self.preview.canvas.update()

    def _on_aspect_chosen(self, aspect: str) -> None:
        self._change_geometry(
            geometry.with_aspect(self.controller.geometry, self.controller.photo_size, aspect)
        )

    def _on_orientation_flipped(self) -> None:
        self._change_geometry(
            geometry.with_orientation_flipped(
                self.controller.geometry, self.controller.photo_size
            )
        )

    def _on_angle_changed(self, angle: float, dragging: bool) -> None:
        self._change_geometry(
            geometry.with_angle(self.controller.geometry, self.controller.photo_size, angle),
            dragging,
        )

    def _on_turn_requested(self, quarter_turns: int) -> None:
        self._change_geometry(
            geometry.with_quarter_turn(self.controller.geometry, quarter_turns)
        )

    def _on_mirror_requested(self, vertical: bool) -> None:
        self._change_geometry(geometry.with_mirror(self.controller.geometry, vertical))

    def _on_crop_dragged(self, rect: object, dragging: bool) -> None:
        self._change_geometry(
            geometry.with_crop(self.controller.geometry, self.controller.photo_size, rect),
            dragging,
        )

    def _on_straighten_toggled(self, armed: bool) -> None:
        self.preview.canvas.crop.arm_straighten(armed)

    def _on_tab_changed(self, tab: InspectorTab) -> None:
        """Enter or leave the crop tab; leaving applies the crop.

        The eyedropper belongs to the adjust tab, so leaving it puts the
        eyedropper down as well (A-24).
        """
        if tab is not InspectorTab.ADJUST:
            self.set_picking_white_balance(False)
        if tab is not InspectorTab.RETOUCH:
            self.set_spot_tool(False)
        self._set_cropping(tab is InspectorTab.CROP)
        self._update_look_thumbnails()

    def _update_look_thumbnails(self) -> None:
        """Ask for the filter tab's thumbnails only while it can be seen.

        They are the photo developed once per look, which is wasted work
        behind another tab or a folded panel; the size goes in screen pixels,
        so they stay sharp on a high density screen.
        """
        showing = (
            self.inspector.current_tab is InspectorTab.LOOKS
            and not self.is_inspector_collapsed()
        )
        side = round(theme.FILTER_THUMBNAIL_SIZE * self.devicePixelRatioF())
        self.controller.set_look_thumbnail_side(side if showing else 0)

    # -- the white balance eyedropper (A-24) ------------------------------

    def auto_light(self) -> None:
        """Set the light sliders from the photo (Ctrl+U), where they can be seen."""
        if self.controller.current is None:
            return
        self._show_tab(InspectorTab.ADJUST)
        self.controller.auto_light()

    def toggle_white_balance_pick(self) -> None:
        """Take up the eyedropper, or put it down again (W).

        Like W in most photo editors it brings up the adjust tab first, which is where
        the sliders it moves are.
        """
        if self.controller.current is None:
            return
        if self.preview.canvas.picking:
            self.set_picking_white_balance(False)
            return
        if self.inspector.current_tab is not InspectorTab.ADJUST:
            self._show_tab(InspectorTab.ADJUST)
        self.set_picking_white_balance(True)

    def set_picking_white_balance(self, picking: bool) -> None:
        """Take up or put down the eyedropper; the button shows which."""
        self.inspector.adjust_tab.pick_button.setChecked(picking)

    def _on_white_balance_picked(self, across: float, down: float) -> None:
        """Read the spot that was clicked, and put the eyedropper down."""
        self.set_picking_white_balance(False)
        self.controller.pick_white_balance((across, down))

    def _on_white_balance_refused(self) -> None:
        """Hand the eyedropper back: the message asks for another spot."""
        if self.inspector.current_tab is InspectorTab.ADJUST and not self._collage_mode:
            self.set_picking_white_balance(True)

    # -- spot removal (PLAN.md 2.2, A-32) --------------------------------

    @property
    def spot_tool_on(self) -> bool:
        """Return whether clicks on the photo mend spots."""
        return self.preview.canvas.spots.active

    def toggle_spot_tool(self) -> None:
        """Take up the spot removal tool, or put it down again (Q).

        Like the eyedropper's W it brings up the tab it belongs to first.
        """
        if self.controller.current is None or self._collage_mode:
            return
        if self.spot_tool_on:
            self.set_spot_tool(False)
            return
        if self.inspector.current_tab is not InspectorTab.RETOUCH:
            self._show_tab(InspectorTab.RETOUCH)
        self.set_spot_tool(True)

    def set_spot_tool(self, on: bool) -> None:
        """Take up or put down the spot removal tool; the button shows which."""
        self.inspector.retouch_tab.spot_button.setChecked(on)

    def _on_spot_tool_toggled(self, on: bool) -> None:
        self.preview.canvas.set_spot_tool(on)
        self._show_spots()

    def _show_spots(self) -> None:
        """Draw the photo's spots and the brush as they are now."""
        overlay = self.preview.canvas.spots
        has_photo = self.controller.current is not None
        overlay.set_spots(self.controller.spots_on_screen() if has_photo else [])
        overlay.set_brush(self.controller.shown_radius(self.inspector.retouch_tab.spot_size))
        self.inspector.retouch_tab.set_spot_count(
            len(self.controller.geometry.spots) if has_photo else 0
        )
        self.preview.canvas.update()

    def resize_spot_brush(self, larger: bool) -> None:
        """Make the brush, or the selected spot, a step larger or smaller ([ and ]).

        In the portrait it is the portrait's brush (PLAN.md 3.15), and in
        liquify the liquify brush (3.16).
        """
        if self._portrait_mode:
            panel = self.portrait_panel
            factor = SPOT_SIZE_STEP if larger else 1.0 / SPOT_SIZE_STEP
            panel.set_brush_size(panel.brush_size * factor)
            self._show_portrait_brush()
            return
        if self._liquify_mode:
            liquify_panel = self.liquify_panel
            factor = SPOT_SIZE_STEP if larger else 1.0 / SPOT_SIZE_STEP
            liquify_panel.set_brush_size(liquify_panel.brush_size * factor)
            self._show_liquify_brush()
            return
        if not self.spot_tool_on:
            return
        retouch = self.inspector.retouch_tab
        factor = SPOT_SIZE_STEP if larger else 1.0 / SPOT_SIZE_STEP
        retouch.set_spot_size(retouch.spot_size * factor)
        self._on_spot_size_set(retouch.spot_size, False)

    def _on_spot_size_set(self, radius: float, dragging: bool) -> None:
        """Size the brush, and the selected spot with it."""
        selected = self.preview.canvas.spots.selected
        if selected >= 0:
            self.controller.resize_spot(selected, radius, dragging)
        self._show_spots()

    def _on_spot_painted(self, points: list[tuple[float, float]]) -> None:
        """Mend what the brush painted; the program chooses where from."""
        self.controller.add_spot(points, self.inspector.retouch_tab.spot_size)

    def _on_spot_moved(
        self, index: int, across: float, down: float, source: bool, dragging: bool
    ) -> None:
        self.controller.move_spot(index, (across, down), source, dragging)

    def _on_spot_selected(self, index: int) -> None:
        """Show the selected spot's size on the slider, so it can be changed."""
        spots = self.controller.geometry.spots
        if 0 <= index < len(spots):
            self.inspector.retouch_tab.set_spot_size(spots[index].radius)
            self._show_spots()

    def _on_spot_deleted(self, index: int) -> None:
        self.preview.canvas.spots.select(-1)
        self.controller.remove_spot(index)

    def _set_cropping(self, cropping: bool) -> None:
        """Show or hide the crop box and tell the controller."""
        has_photo = self.controller.current is not None
        self.controller.set_cropping(cropping and has_photo)
        box = self.preview.canvas.crop
        box.set_source(self.controller.photo_size)
        box.set_geometry(self.controller.geometry)
        box.set_active(cropping and has_photo)
        if not cropping:
            self.inspector.crop_tab.set_straightening(False)
            box.arm_straighten(False)
        self.preview.canvas.update()

    def toggle_crop(self) -> None:
        """Enter the crop tab, or leave it again (R)."""
        if self.controller.current is None:
            return
        if self.inspector.current_tab is InspectorTab.CROP:
            self.finish_crop()
        else:
            self.inspector.set_current_tab(InspectorTab.CROP)

    def finish_crop(self) -> None:
        """Apply the crop and go back to the adjust tab (Enter)."""
        if self.inspector.current_tab is not InspectorTab.CROP:
            return
        self.inspector.set_current_tab(InspectorTab.ADJUST)

    def cancel_crop(self) -> None:
        """Put the geometry back and leave the crop tab (Esc)."""
        if self.inspector.current_tab is not InspectorTab.CROP:
            return
        self.controller.cancel_crop()
        self.inspector.set_current_tab(InspectorTab.ADJUST)

    def flip_crop_orientation(self) -> None:
        """Turn the chosen ratio on its side (X)."""
        if self.inspector.current_tab is InspectorTab.CROP:
            self.inspector.crop_tab.flip_orientation()

    def toggle_crop_grid(self) -> None:
        """Keep the thirds grid on screen, or let it fade again (O)."""
        if self.inspector.current_tab is not InspectorTab.CROP:
            return
        box = self.preview.canvas.crop
        box.set_grid_always(not box.grid_visible)
        self.preview.canvas.update()

    def turn_left(self) -> None:
        """Turn the photo a quarter turn to the left (Ctrl+[)."""
        if self.controller.current is not None:
            self._on_turn_requested(-1)

    def turn_right(self) -> None:
        """Turn the photo a quarter turn to the right (Ctrl+])."""
        if self.controller.current is not None:
            self._on_turn_requested(1)

    def mirror_horizontally(self) -> None:
        """Flip the photo left to right (Shift+H, C-08)."""
        if self.controller.current is not None:
            self._on_mirror_requested(False)

    def mirror_vertically(self) -> None:
        """Flip the photo top to bottom (Shift+V, C-08)."""
        if self.controller.current is not None:
            self._on_mirror_requested(True)

    # -- copying, pasting and presets (P-01 to P-06) ---------------------

    def copy_adjustments(self) -> None:
        """Take the colour settings of the photo being shown (P-01)."""
        self.controller.copy_adjustments()

    def paste_adjustments(self) -> None:
        """Put them on every chosen photo (P-02)."""
        self.controller.paste_adjustments(self.filmstrip.selected_photos())

    def save_preset(self) -> None:
        """Ask for a name and save the current settings under it (P-04)."""
        if self.controller.current is None:
            return
        name = NameDialog.ask("presets.save_title", parent=self)
        if not name:
            return
        saved = self.controller.save_preset(name)
        self._show_tab(InspectorTab.ADJUST)
        if saved is not None:
            QTimer.singleShot(0, lambda: self._reveal_preset(saved.id))

    def _reveal_preset(self, preset_id: str) -> None:
        """Scroll the adjust tab to a preset just saved, once it is laid out."""
        row = self.inspector.adjust_tab.preset_row(preset_id)
        if row is not None:
            self.inspector.scroll_area(InspectorTab.ADJUST).ensureWidgetVisible(row)

    def _on_clipboard_changed(self, has_settings: bool) -> None:
        """Let the paste button follow whether there is anything to paste."""
        _copy, paste, _reset = self.inspector.action_buttons(InspectorTab.ADJUST)
        paste.setEnabled(has_settings and self.controller.current is not None)

    def _on_filmstrip_selection(self, photos: list) -> None:
        """Say on the paste button how many photos it would reach (P-02)."""
        _copy, paste, _reset = self.inspector.action_buttons(InspectorTab.ADJUST)
        if len(photos) > 1 and self.controller.has_copied_adjustments:
            paste.setToolTip(tr("toast.settings_pasted", count=len(photos)))
        else:
            paste.setToolTip("")

    def _on_metadata_ready(self, info: PhotoMetadata) -> None:
        self.info_card.set_metadata(info)
        self.info_card.set_edited(self.controller.is_edited(info.path))

    def _on_edited_photos(self, edited: set) -> None:
        """Mark the edited photos of the folder, in the strip and the info tab."""
        self.filmstrip.set_edited_photos(edited)
        current = self.controller.current
        if current is not None:
            self.info_card.set_edited(current in edited)

    def _on_edited_changed(self, path: Path, edited: bool) -> None:
        """Follow one photo's parameter file being written or deleted."""
        self.filmstrip.set_edited(path, edited)
        if path == self.controller.current:
            self.info_card.set_edited(edited)

    def _on_thumbnail_clicked(self, path: Path) -> None:
        if self._collage_mode:
            # The filmstrip feeds the collage while it is open (PLAN.md 3.13).
            self.collage.fill([path])
            return
        photos = self.controller.photos
        if path in photos:
            self.controller.select(photos.index(path))

    # -- dropping files on the window (PLAN.md 3.7) ----------------------

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Accept photos and folders dragged onto the window.

        A drag that started in this window - a thumbnail on its way to the
        collage - is not something to open. Neither is anything dropped on
        the collage outside its two halves, which take drops themselves.
        """
        if event.source() is not None or self._collage_mode:
            event.ignore()
            return
        if _dropped_paths(event):
            self._set_drop_target(True)
            event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Stop showing the drop outline."""
        self._set_drop_target(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Open what was dropped: a folder whole, or just the photos dropped."""
        self._set_drop_target(False)
        paths = _dropped_paths(event)
        if paths:
            self.controller.open_paths(paths)
            event.acceptProposedAction()

    def _set_drop_target(self, active: bool) -> None:
        """Show or hide the marks that say a drag would land here."""
        self.preview.set_drop_target(active)

    # -- window state ----------------------------------------------------

    def _restore_state(self) -> None:
        """Apply the remembered geometry and collapsed state."""
        window = self._settings.window
        width = window.width or theme.WINDOW_DEFAULT_WIDTH
        height = window.height or theme.WINDOW_DEFAULT_HEIGHT
        self.resize(
            max(theme.WINDOW_MIN_WIDTH, width),
            max(theme.WINDOW_MIN_HEIGHT, height),
        )
        if window.x is not None and window.y is not None:
            self.move(window.x, window.y)
        if window.maximized:
            self.showMaximized()

        self._apply_panel_visibility(animate=False)

        self._show_recent_folders()
        self._open_last_folder()

    def _open_last_folder(self) -> None:
        """Open the folder from last time, if that is what was asked for.

        PLAN.md 3.10 offers this instead of the welcome screen. A folder
        that has since been moved or unplugged simply leaves the welcome
        screen up, which is what the user would otherwise have seen.
        """
        if self._settings.startup_action != STARTUP_LAST_FOLDER:
            return
        last = self._settings.last_folder
        if not last:
            return
        folder = Path(last)
        if not folder.is_dir():
            logger.info("The folder from last time is no longer there: %s", folder)
            return
        self.controller.open_folder(folder)

    def _store_state(self) -> None:
        """Copy the current geometry and collapsed state into the settings."""
        window = self._settings.window
        window.maximized = self.isMaximized()
        if self.isMaximized() or self.isMinimized():
            # Remember only the size the window returns to. normalGeometry()
            # excludes the window frame, so its position cannot be mixed with
            # the frame position stored below.
            normal = self.normalGeometry()
            if normal.isValid():
                window.width = normal.width()
                window.height = normal.height()
        else:
            # pos() and move() both work in frame coordinates, size() and
            # resize() both in client coordinates. Saving and restoring with
            # these matching pairs is what keeps the window from creeping down
            # the screen by the height of the title bar on every restart.
            position = self.pos()
            window.x = position.x()
            window.y = position.y()
            window.width = self.width()
            window.height = self.height()
        # What the user asked for, not what the welcome screen is doing to
        # the panels at this moment.
        self._settings.inspector_collapsed = self._panels_wanted["inspector"]
        self._settings.filmstrip_collapsed = self._panels_wanted["filmstrip"]
        self._settings.histogram_mode = self.inspector.histogram.mode

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Apply the dark native title bar once the window has a handle."""
        super().showEvent(event)
        native.apply_dark_title_bar(int(self.winId()))
        self._update_display_profile()

    def moveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Read the screen's profile again after a move to another one."""
        super().moveEvent(event)
        screen = self.screen()
        if screen is not self._screen:
            self._screen = screen
            self._update_display_profile()
            # Another screen may have another pixel density.
            self._update_look_thumbnails()

    def _update_display_profile(self) -> None:
        """Tell the controller which screen the photo is being shown on.

        Turned off in the settings, the preview is shown as plain sRGB
        instead (PLAN.md 6 step 13).
        """
        profile = native.display_profile() if self._settings.use_display_profile else None
        self.controller.set_display_profile(profile)
        # The collage's photos were developed for the screen they were on.
        self.collage.refresh()

    # -- the settings dialog (PLAN.md 3.10) ------------------------------

    def open_settings(self) -> None:
        """Show the settings, which take effect as they are changed."""
        dialog = SettingsDialog(self._settings, self.controller, self)
        dialog.changed.connect(self.apply_settings)
        try:
            self._run_dialog(dialog)
        finally:
            self._settings.save()

    def open_about(self) -> None:
        """Show the version and the licences (PLAN.md 13.5)."""
        self._run_dialog(AboutDialog(self))

    def open_devlog(self) -> None:
        """Show the development log, newest day first."""
        self._run_dialog(DevlogDialog(self))

    def apply_settings(self) -> None:

        """Make the window match the settings as they now stand."""
        self._apply_language()
        self.preview.set_background(self._settings.preview_background)
        self.inspector.adjust_tab.set_temperature_unit(
            self._settings.temperature_unit == TEMPERATURE_KELVIN
        )
        self.controller.set_thumbnail_cache_limit(self._settings.cache_limit_bytes)
        self._update_display_profile()
        if self._developed is not None:
            self.preview.show_photo(
                self._marked(self._developed),
                frame=self._developed_frame,
                region=self._developed_region,
            )


    def _apply_language(self) -> None:
        """Put the interface into the language the settings now name (U-02).

        The window is not rebuilt: every widget that holds text reads it
        again, so the photo on screen, the history and where the user had
        got to all stay as they are. A dialog that is open is a child of the
        window, so it changes with it.
        """
        language = self._settings.language
        if language == current_language():
            return
        init_translator(language)
        retranslate_tree(self)

    def retranslate(self) -> None:
        """Read the window's own title again after a language change."""
        self.setWindowTitle(tr("app.name"))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Keep the message centred."""
        super().resizeEvent(event)
        self.toast.parent_resized()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Persist the window state and stop background work before closing."""
        self._store_state()
        if not self._settings.save():
            logger.warning("The window state could not be saved")
        self.controller.shutdown()
        self.collage.shutdown()
        self.batch.shutdown()
        super().closeEvent(event)

    # -- collapsing panels -----------------------------------------------

    def shortcut_handlers(self) -> dict[str, Callable[[], None]]:
        """Return what each combination of PLAN.md section 8 does.

        The rest of the section - a key held down, a double click, an arrow
        key whose meaning depends on what has the focus - is handled where it
        belongs, by the preview and by the sliders, so it is not here.
        """
        canvas = self.preview.canvas
        return {
            "shortcut.toggle_inspector": self.toggle_inspector,
            "shortcut.toggle_filmstrip": self.toggle_filmstrip,
            "shortcut.open_file": self.open_file_dialog,
            "shortcut.open_folder": self.open_folder_dialog,
            "shortcut.clipping": self.inspector.histogram.toggle_clipping,
            "shortcut.histogram_mode": self.inspector.histogram.toggle_mode,
            "shortcut.undo": self.controller.undo,
            "shortcut.redo": self.controller.redo,
            "shortcut.redo_alternative": self.controller.redo,
            "shortcut.reset": self.controller.reset,
            "shortcut.white_balance_pick": self.toggle_white_balance_pick,
            "shortcut.spot_tool": self.toggle_spot_tool,
            "shortcut.spot_smaller": lambda: self.resize_spot_brush(False),
            "shortcut.spot_larger": lambda: self.resize_spot_brush(True),
            "shortcut.auto_light": self.auto_light,
            "shortcut.crop": self.toggle_crop,
            "shortcut.crop_flip": self.flip_crop_orientation,
            "shortcut.crop_grid": self.toggle_crop_grid,
            "shortcut.crop_apply": self.finish_crop,
            "shortcut.crop_cancel": self.on_escape,
            "shortcut.turn_left": self.turn_left,
            "shortcut.turn_right": self.turn_right,
            "shortcut.mirror_horizontal": self.mirror_horizontally,
            "shortcut.mirror_vertical": self.mirror_vertically,
            "shortcut.copy_settings": self.copy_adjustments,
            "shortcut.paste_settings": self.paste_adjustments,
            "shortcut.new_preset": self.save_preset,
            "shortcut.export": self.export,
            "shortcut.check_all": self.check_all_photos,
            "shortcut.check_none": self.check_no_photos,
            "shortcut.collage": self.toggle_collage,
            "shortcut.watermark": self.toggle_watermark,
            "shortcut.portrait": self.toggle_portrait,
            "shortcut.portrait_add": lambda: self.toggle_portrait_brush(BRUSH_ADD),
            "shortcut.portrait_remove": lambda: self.toggle_portrait_brush(BRUSH_REMOVE),
            "shortcut.portrait_show": self.toggle_portrait_showing,
            "shortcut.liquify": self.toggle_liquify,
            "shortcut.settings": self.open_settings,
            "shortcut.fullscreen": self.toggle_full_screen,
            "shortcut.zoom_in": canvas.zoom_in,
            "shortcut.zoom_in_alternative": canvas.zoom_in,
            "shortcut.zoom_out": canvas.zoom_out,
            "shortcut.zoom_fit": canvas.fit_to_window,
            "shortcut.tab_adjust": lambda: self._show_tab(InspectorTab.ADJUST),
            "shortcut.tab_looks": lambda: self._show_tab(InspectorTab.LOOKS),
            "shortcut.tab_retouch": lambda: self._show_tab(InspectorTab.RETOUCH),
            "shortcut.tab_crop": lambda: self._show_tab(InspectorTab.CROP),
            "shortcut.info": self.toggle_info,
        }

    def _register_shortcuts(self) -> None:
        """Ask Qt to listen for every combination :mod:`ui.shortcuts` lists.

        That module holds the list, which is also what the settings dialog
        shows, so what is registered and what is displayed cannot drift
        apart; this says only what each one does. A combination listed with
        nothing to do raises here rather than silently doing nothing.
        """
        handlers = self.shortcut_handlers()
        for key in REGISTERED:
            shortcut = QShortcut(QKeySequence(tr(key)), self)
            shortcut.activated.connect(
                lambda handler=handlers[key], name=key: self._on_shortcut(name, handler)
            )

    def _on_shortcut(self, key: str, handler: Callable[[], None]) -> None:
        """Run a shortcut, unless it acts on what the collage or the watermark covers."""
        if self._collage_mode and key not in _COLLAGE_SHORTCUTS:
            return
        if self._watermark_mode and key not in _WATERMARK_SHORTCUTS:
            return
        if self._portrait_mode and key in _NOT_IN_PORTRAIT:
            return
        if not self._portrait_mode and key in _PORTRAIT_ONLY:
            return
        if self._liquify_mode and key in _NOT_IN_PORTRAIT:
            return
        handler()

    def _show_tab(self, tab: InspectorTab) -> None:
        """Show one of the tabs, opening the panel if it is collapsed.

        Asking for a tab is asking for the panel, so it is recorded as what
        the user wants rather than only opened: left out, the next folder
        opened would fold the panel away again and the key that collapses it
        would do nothing the first time it was pressed. The welcome screen
        keeps the whole window either way (PLAN.md 3.7).
        """
        if self.is_inspector_collapsed():
            self._panels_wanted["inspector"] = False
            self._apply_panel_visibility()
        self.inspector.set_current_tab(tab)

    def on_escape(self) -> None:
        """Take Esc to mean the nearest thing to get out of.

        Cropping first, since the crop box is what the key is in front of,
        and the eyedropper, which is the same kind of mode; then choosing
        photos to export, the filmstrip's mode; then the info card, which is
        over the photo; then full screen, the only other thing Esc leaves
        (PLAN.md section 8).
        """
        if self._watermark_mode:
            self.leave_watermark()
        elif self._portrait_mode and self.portrait_panel.brush is not None:
            # The brush is put down first, then the portrait is left.
            self.portrait_panel.set_brush(None)
        elif self._portrait_mode:
            self.leave_portrait()
        elif self._liquify_mode and self.editing_faces:
            # Placing the points ends first, then liquify is left.
            self.set_editing_faces(False)
        elif self._liquify_mode:
            self.leave_liquify()
        elif self._collage_mode and self.collage_canvas.aligning:
            # Esc ends the nearer thing first: aligning, then the collage.
            self.collage_panel.set_aligning(False)
        elif self._collage_mode:
            self.leave_collage()
        elif self.controller.cropping:
            self.cancel_crop()
        elif self.preview.canvas.picking:
            self.set_picking_white_balance(False)
        elif self.spot_tool_on and self.preview.canvas.spots.selected >= 0:
            # A selected spot is let go first, then the tool.
            self.preview.canvas.spots.select(-1)
            self.preview.canvas.update()
        elif self.spot_tool_on:
            self.set_spot_tool(False)
        elif self.filmstrip.choosing:
            self.filmstrip.stop_choosing()
        elif self.info_shown:
            self.set_info_shown(False)
        elif self.isFullScreen():
            self.toggle_full_screen()

    # -- the info card (PLAN.md 3.9) ----------------------------------------

    @property
    def info_shown(self) -> bool:
        """Return whether the info card is up over the photo, or on its way up."""
        return self.info_card.wanted

    def toggle_info(self) -> None:
        """Bring the info card up, or take it away (I, or the info button)."""
        self.set_info_shown(not self.info_shown)

    def set_info_shown(self, shown: bool) -> None:
        """Show the card over the photo's top left corner, or hide it."""
        shown = shown and self.controller.current is not None and not self._collage_mode
        self.info_card.set_shown(shown)
        self.toolbar.info_button.setChecked(shown)

    def toggle_full_screen(self) -> None:
        """Show the photo on the whole screen, or go back (PLAN.md section 8).

        Full screen means the photo and nothing else, so the panels fold
        away with it and come back as they were.
        """
        if self.isFullScreen():
            self.showNormal()
            self.set_inspector_collapsed(self._panels_before_full_screen[0])
            self.set_filmstrip_collapsed(
                self._panels_before_full_screen[1], hidden=not self.controller.photos
            )
            return
        self._panels_before_full_screen = (
            self.is_inspector_collapsed(),
            self.is_filmstrip_collapsed(),
        )
        self.set_inspector_collapsed(True, animate=False)
        # Not even the tab: full screen is the photo and nothing else.
        self.set_filmstrip_collapsed(True, animate=False, hidden=True)
        self.showFullScreen()
        self.preview.setFocus()

    def _compare_key(self) -> int:
        """Return the key that shows the photo as it was found while held.

        Held down rather than pressed (PLAN.md section 8), which a QShortcut
        cannot express: it fires on the press and knows nothing of the
        release. The key itself still comes from the shortcut table.
        """
        sequence = QKeySequence(tr("shortcut.compare"))
        if sequence.count() != 1:
            return 0
        return int(sequence[0].key())

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Start showing the photo as it was found while the key is held (E-06)."""
        if (
            event.key() == self._compare_key()
            and not event.isAutoRepeat()
            and not self._collage_mode
            and not self._watermark_mode
        ):
            self.controller.set_comparing(True)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Go back to the edited photo when the key is let go (E-06)."""
        if event.key() == self._compare_key() and not event.isAutoRepeat():
            self.controller.set_comparing(False)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt naming
        """Take Tab for the panels, wherever the focus sits in this window.

        Tab hides and shows every panel, as photo editors commonly do (PLAN.md section 8).
        Qt turns the key into focus traversal inside the focus widget's own
        ``event()``, before any ``keyPressEvent`` of ours could see it, so it
        is caught here on its way there - the application hands every event
        to its filters first.

        Overriding ``focusNextPrevChild`` looked like the same thing and was
        not: Qt calls it whenever it has the focus to hand on, and a stacked
        widget hands it on every time it hides the page the focus sits on.
        Clicking a control in the panel and then switching tabs therefore
        read as Tab, collapsing both panels and letting them straight back -
        the whole window appeared to zoom once per tab switch.
        """
        if event.type() != QEvent.Type.KeyPress:
            return False
        if not isinstance(watched, QWidget) or watched.window() is not self:
            return False
        if event.key() not in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
            return False
        if self._watermark_mode or self._portrait_mode or self._liquify_mode:
            # None of the watermark's, the portrait's or the liquify panel
            # is the inspector the key folds away, so Tab moves between its
            # controls as usual.
            return False
        self.toggle_focus_mode()
        return True

    def is_inspector_collapsed(self) -> bool:
        """Return whether the right hand panel is collapsed."""
        return self._collapsed.get(self.inspector.objectName(), False)

    def is_filmstrip_collapsed(self) -> bool:
        """Return whether the filmstrip is collapsed."""
        return self._collapsed.get(self.filmstrip.objectName(), False)

    def toggle_focus_mode(self) -> None:
        """Hide every panel, or restore both of them."""
        collapse = not (
            self._panels_wanted["inspector"] and self._panels_wanted["filmstrip"]
        )
        self._panels_wanted["inspector"] = collapse
        self._panels_wanted["filmstrip"] = collapse
        self._apply_panel_visibility()

    def toggle_inspector(self) -> None:
        """Collapse or expand the right hand panel."""
        self._panels_wanted["inspector"] = not self._panels_wanted["inspector"]
        self._apply_panel_visibility()

    def toggle_filmstrip(self) -> None:
        """Collapse or expand the filmstrip."""
        self._panels_wanted["filmstrip"] = not self._panels_wanted["filmstrip"]
        self._apply_panel_visibility()

    def _apply_panel_visibility(self, animate: bool = True) -> None:
        """Show the panels the user wants, once there is a photo to show.

        The welcome screen is the whole window: an empty panel and an empty
        filmstrip beside it would say nothing (PLAN.md 3.7). They appear
        with the first photo, folded the way the user last left them.
        """
        nothing_open = not self.controller.photos
        self._show_toolbar_for_welcome()
        self.set_inspector_collapsed(
            nothing_open or self._panels_wanted["inspector"], animate
        )
        self.set_filmstrip_collapsed(
            nothing_open or self._panels_wanted["filmstrip"],
            animate,
            hidden=nothing_open or self.isFullScreen(),
        )


    def _show_toolbar_for_welcome(self) -> None:
        """Keep only the menu in the toolbar while the welcome screen is up.

        The collage covers the welcome screen when it is opened from there
        (C), and needs its own button to be left again.
        """
        self.toolbar.set_welcome(not self.controller.photos and not self._collage_mode)

    def set_inspector_collapsed(self, collapsed: bool, animate: bool = True) -> None:
        """Set whether the right hand panel is collapsed."""
        self._set_panel_collapsed(
            self.inspector, "panel_width", theme.PANEL_WIDTH, collapsed, animate
        )
        self._update_look_thumbnails()

    def set_filmstrip_collapsed(
        self, collapsed: bool, animate: bool = True, hidden: bool = False
    ) -> None:
        """Set whether the filmstrip is collapsed.

        Folded away, the strip leaves its bookmark tab at the bottom of the
        window to bring it back (PLAN.md 3.1). *hidden* takes the tab away
        too, for when there is no folder open or the photo has the whole
        screen.
        """
        self.filmstrip_tab.set_collapsed(collapsed)
        self.filmstrip_tab.setVisible(not hidden)
        self.filmstrip_tab.follow()
        self._set_panel_collapsed(
            self.filmstrip, "panel_height", theme.FILMSTRIP_HEIGHT, collapsed, animate
        )

    def _set_panel_collapsed(
        self,
        panel: QWidget,
        property_name: str,
        expanded_size: int,
        collapsed: bool,
        animate: bool,
    ) -> None:
        """Collapse or expand *panel* by animating one of its size properties."""
        key = panel.objectName()
        if key in self._collapsed and self._collapsed[key] == collapsed:
            return
        self._collapsed[key] = collapsed

        running = self._panel_animations.pop(key, None)
        if running is not None:
            running.stop()
            running.deleteLater()

        target = 0 if collapsed else expanded_size

        if not animate or not native.animations_enabled():
            panel.setProperty(property_name, target)
            panel.setVisible(not collapsed)
            return

        if not collapsed:
            panel.setProperty(property_name, 0)
            panel.setVisible(True)

        animation = QPropertyAnimation(panel, property_name.encode(), self)
        animation.setDuration(theme.DURATION_PANEL_MS)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.setEndValue(target)
        if collapsed:
            animation.finished.connect(lambda: panel.setVisible(False))
        animation.start()
        self._panel_animations[key] = animation


#: What the keyboard still does while the collage is open: everything else
#: acts on the photo being edited, which the collage covers (PLAN.md 3.13).
#: The portrait, liquify and watermark keys leave the collage for that mode.
_COLLAGE_SHORTCUTS: frozenset[str] = frozenset(
    {
        "shortcut.toggle_filmstrip",
        "shortcut.open_file",
        "shortcut.open_folder",
        "shortcut.export",
        "shortcut.collage",
        "shortcut.watermark",
        "shortcut.portrait",
        "shortcut.liquify",
        "shortcut.settings",
        "shortcut.fullscreen",
        "shortcut.crop_cancel",
    }
)

#: What does nothing while the portrait is open (PLAN.md 3.15), or liquify
#: (3.16): the keys that work on the editing panel it covers, or that take
#: the click on the photo away from its brush. Everything else - history,
#: stepping through the photos, zooming, exporting - goes on as in the
#: editor.
_NOT_IN_PORTRAIT: frozenset[str] = frozenset(
    {
        "shortcut.white_balance_pick",
        "shortcut.auto_light",
        "shortcut.spot_tool",
        "shortcut.crop",
        "shortcut.crop_flip",
        "shortcut.crop_grid",
        "shortcut.crop_apply",
        "shortcut.turn_left",
        "shortcut.turn_right",
        "shortcut.mirror_horizontal",
        "shortcut.mirror_vertical",
        "shortcut.reset",
        "shortcut.copy_settings",
        "shortcut.paste_settings",
        "shortcut.new_preset",
        "shortcut.toggle_inspector",
        "shortcut.tab_adjust",
        "shortcut.tab_looks",
        "shortcut.tab_retouch",
        "shortcut.tab_crop",
        "shortcut.clipping",
        "shortcut.histogram_mode",
    }
)

#: The portrait's own keys, which mean nothing anywhere else.
_PORTRAIT_ONLY: frozenset[str] = frozenset(
    {"shortcut.portrait_add", "shortcut.portrait_remove", "shortcut.portrait_show"}
)

#: What still works while the watermark is being placed (PLAN.md 3.14):
#: leaving it or switching to another mode, stepping through the photos it
#: is checked against, looking closer, and exporting. The rest acts on the
#: editing panel it covers.
_WATERMARK_SHORTCUTS: frozenset[str] = frozenset(
    {
        "shortcut.watermark",
        "shortcut.portrait",
        "shortcut.liquify",
        "shortcut.collage",
        "shortcut.crop_cancel",
        "shortcut.toggle_filmstrip",
        "shortcut.open_file",
        "shortcut.open_folder",
        "shortcut.export",
        "shortcut.settings",
        "shortcut.info",
        "shortcut.zoom_in",
        "shortcut.zoom_in_alternative",
        "shortcut.zoom_out",
        "shortcut.zoom_fit",
    }
)


def _supported_suffixes() -> set[str]:
    """Return the file extensions the open dialog should offer."""
    from core.decoder import SUPPORTED_EXTENSIONS

    return set(SUPPORTED_EXTENSIONS)


def _dropped_paths(event) -> list[Path]:
    """Return the local paths carried by a drag, photos and folders only."""
    data = event.mimeData()
    if not data.hasUrls():
        return []
    paths: list[Path] = []
    for url in data.urls():
        if not url.isLocalFile():
            continue
        path = Path(url.toLocalFile())
        if path.is_dir() or path.suffix.lower() in _supported_suffixes():
            paths.append(path)
    return paths
