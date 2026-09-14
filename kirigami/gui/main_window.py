"""Kirigami main window: episode queue, cameras, timeline, segments, export."""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import uuid

import numpy as np
from yaml import YAMLError
from PySide6.QtCore import QObject, Qt, QThread, QTimer, QElapsedTimer, QSize, Signal
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME
from ..annotation import (
    Annotation,
    add_mark,
    add_preset,
    annotation_from_episode,
    load_annotation,
    load_subtask_presets,
    mark_exported,
    nearest_mark,
    remove_mark,
    save_annotation,
    save_subtask_presets,
    validate_subtask_name,
)
from ..exporter import default_output_root, export_annotation
from ..loader import EpisodeRef, LoadedEpisode, discover_many, load_episode
from ..robot3d import Robot3DConfig, Robot3DPlayer
from ..video import VideoBank, bgr_to_rgb
from ..source_guard import SourceChangedError
from ..trimming import trim_source
from .widgets import CameraLabel, JointPlot, TimelineBar
from .robot_view import Robot3DPanel
from .theme import apply_theme
from .preset_dialog import PresetDialog


class ExportWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str, float)

    def __init__(self, episode: LoadedEpisode, annotation: Annotation, out_root: Path):
        super().__init__()
        self.episode = episode
        self.annotation = deepcopy(annotation)
        self.out_root = out_root

    def run(self) -> None:
        try:
            written = export_annotation(
                self.episode,
                self.annotation,
                self.out_root,
                progress=lambda msg, frac: self.progress.emit(msg, frac),
            )
            self.finished.emit(written)
        except Exception as exc:
            self.failed.emit(str(exc))


class TrimWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str, float)

    def __init__(self, episode, annotation):
        super().__init__()
        self.episode = episode
        self.annotation = deepcopy(annotation)

    def run(self):
        try:
            result = trim_source(self.episode, self.annotation, *self.annotation.keep_range,
                                 progress=lambda message, fraction: self.progress.emit(message, fraction))
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(
        self,
        initial_dirs: list[Path] | None = None,
        out_root: Path | None = None,
        presets_path: Path | None = None,
        robot_config: Robot3DConfig | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(APP_NAME)
        self.resize(1600, 980)
        apply_theme(self)

        self.presets_path = presets_path or (Path(__file__).resolve().parents[2] / "subtasks.yaml")
        self.presets = load_subtask_presets(self.presets_path)
        self._preset_dialog: PresetDialog | None = None
        self.out_root_override = Path(out_root) if out_root else None
        self.robot = Robot3DPlayer(robot_config)

        self.refs: list[EpisodeRef] = []
        self.episode: LoadedEpisode | None = None
        self.annotation: Annotation | None = None
        self.videos: VideoBank | None = None
        self.frame = 0
        self._export_thread: QThread | None = None
        self._export_worker: ExportWorker | None = None
        self._trim_thread: QThread | None = None
        self._trim_worker: TrimWorker | None = None
        self._trim_result = None
        self._trim_error = ""

        self._build_ui()
        self._bind_shortcuts()

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._on_tick)
        self._play_clock = QElapsedTimer()
        self._play_start_frame = 0
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(300)
        self._save_timer.timeout.connect(self._persist)

        if initial_dirs:
            self._add_dirs(initial_dirs)
        else:
            self._refresh_robot_view()

    @staticmethod
    def _label(text: str, role: str = "muted") -> QLabel:
        label = QLabel(text)
        label.setProperty("role", role)
        return label

    def _card(self, title: str, badge: str = ""):
        card = QFrame()
        card.setProperty("role", "card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(9)
        header = QHBoxLayout()
        header.addWidget(self._label(title, "heading"))
        header.addStretch()
        if badge:
            header.addWidget(self._label(badge, "eyebrow"))
        layout.addLayout(header)
        return card, layout, header

    def _button(self, text, callback, variant="") -> QPushButton:
        button = QPushButton(text)
        if variant:
            button.setProperty("variant", variant)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(callback)
        return button

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("Workspace")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        topbar = QFrame()
        topbar.setObjectName("Topbar")
        top = QHBoxLayout(topbar)
        top.setContentsMargins(24, 15, 24, 15)
        top.setSpacing(13)
        logo = self._label("K", "logo")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setFixedSize(36, 36)
        top.addWidget(logo)
        top.addWidget(self._label("Kirigami", "brand"))
        top.addSpacing(13)
        title = QVBoxLayout()
        title.setSpacing(2)
        title.addWidget(self._label("EPISODE STUDIO", "eyebrow"))
        self.session_title = self._label("Your annotation workspace")
        title.addWidget(self.session_title)
        top.addLayout(title, 1)
        self.save_status = self._label("LOCAL WORKSPACE", "badge")
        top.addWidget(self.save_status)
        top.addSpacing(10)
        top.addWidget(self._button("Open folder…", self._on_open_folder))
        outer.addWidget(topbar)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(18, 18, 18, 8)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        body_layout.addWidget(splitter)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_center())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([220, 910, 390])
        outer.addWidget(body, 1)
        self.statusBar().showMessage("Open a task folder to start annotating")

    def _build_left(self) -> QWidget:
        box = QFrame()
        box.setObjectName("Sidebar")
        box.setMinimumWidth(190)
        box.setMaximumWidth(290)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(16, 20, 16, 18)
        layout.setSpacing(12)
        layout.addWidget(self._label("LIBRARY", "eyebrow"))
        heading = QHBoxLayout()
        heading.addWidget(self._label("Episodes", "heading"))
        heading.addStretch()
        self.queue_count = self._label("0", "eyebrow")
        heading.addWidget(self.queue_count)
        layout.addLayout(heading)
        layout.addWidget(self._button("+  Add episodes", self._on_open_folder, "primary"))
        self.list_widget = QListWidget()
        self.list_widget.setSpacing(2)
        self.list_widget.setWordWrap(True)
        self.list_widget.currentRowChanged.connect(self._on_select_episode)
        layout.addWidget(self.list_widget, 1)
        self.queue_summary = self._label("No episodes loaded")
        self.queue_summary.setWordWrap(True)
        layout.addWidget(self.queue_summary)
        nav = QHBoxLayout()
        nav.addWidget(self._button("‹  Previous", lambda: self._move_selection(-1)))
        nav.addWidget(self._button("Next  ›", lambda: self._move_selection(1)))
        layout.addLayout(nav)
        clear = self._button("Clear library", self._clear_queue, "quiet")
        layout.addWidget(clear)
        layout.addSpacing(18)
        layout.addWidget(self._label("EXPORT DESTINATION", "eyebrow"))
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Next to the source task")
        self.out_edit.setToolTip("Leave empty to export into <task>_sliced next to the task folder")
        if self.out_root_override:
            self.out_edit.setText(str(self.out_root_override))
        layout.addWidget(self.out_edit)
        layout.addWidget(self._button("Choose folder…", self._browse_out_root))
        return box

    def _build_center(self) -> QWidget:
        box = QWidget()
        box.setMinimumWidth(550)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        views_heading = QHBoxLayout()
        views_heading.addWidget(self._label("Synchronized views", "heading"))
        views_heading.addStretch()
        views_heading.addWidget(self._label("3 CAMERAS  /  2 ARMS", "eyebrow"))
        layout.addLayout(views_heading)
        grid = QGridLayout()
        grid.setSpacing(12)
        self.cam_labels = {}
        for key, title, badge, row, col in (
            ("base_0", "Overview", "CAM 01", 0, 0),
            ("left_wrist_0", "Left wrist", "CAM 02", 1, 0),
            ("right_wrist_0", "Right wrist", "CAM 03", 1, 1),
        ):
            card, card_layout, _ = self._card(title, badge)
            label = CameraLabel(title)
            self.cam_labels[key] = label
            card_layout.addWidget(label, 1)
            grid.addWidget(card, row, col)
        robot_card, robot_layout, robot_header = self._card("Robot 3D")
        self.material_combo = QComboBox()
        self.material_combo.setObjectName("MaterialSelect")
        self.material_combo.addItems(["White model", "Arm colors"])
        self.material_combo.setFixedWidth(124)
        self.material_combo.setFixedHeight(24)
        self.material_combo.currentIndexChanged.connect(self._on_material_changed)
        robot_header.addWidget(self.material_combo)
        self.robot_panel = Robot3DPanel()
        self.robot_panel.orbitRequested.connect(self._orbit_robot)
        self.robot_panel.zoomRequested.connect(self._zoom_robot)
        self.robot_panel.expandRequested.connect(self._expand_robot_view)
        self.robot_panel.set_player(self.robot)
        self.robot_panel.set_status(self.robot.status)
        robot_layout.addWidget(self.robot_panel, 1)
        robot_footer = QHBoxLayout()
        robot_footer.setSpacing(10)
        robot_footer.addWidget(self._label("● L", "leftLegend"))
        robot_footer.addWidget(self._label("● R", "rightLegend"))
        robot_footer.addStretch()
        robot_footer.addWidget(self._label("Drag to orbit · Scroll to zoom"))
        expand = self._button("Expand", self._expand_robot_view, "quiet")
        expand.setObjectName("ExpandView")
        expand.setFixedHeight(18)
        robot_footer.addWidget(expand)
        reset = self._button("Reset", self._reset_robot_view, "quiet")
        reset.setObjectName("ResetView")
        reset.setFixedHeight(18)
        robot_footer.addWidget(reset)
        robot_layout.addLayout(robot_footer)
        grid.addWidget(robot_card, 0, 1)
        for i in range(2):
            grid.setColumnStretch(i, 1)
            grid.setRowStretch(i, 1)
        layout.addLayout(grid, 6)

        timeline_card, timeline_layout, timeline_header = self._card("Timeline")
        self.time_label = self._label("00:00.00 / 00:00.00")
        timeline_header.addWidget(self.time_label)
        self.timeline = TimelineBar()
        self.timeline.frameClicked.connect(self._seek)
        self.timeline.markToggleRequested.connect(self._toggle_mark)
        timeline_layout.addWidget(self.timeline)
        transport = QHBoxLayout()
        transport.setSpacing(8)
        self.play_btn = self._button("Play", self._toggle_play, "primary")
        self.play_btn.setMinimumWidth(72)
        transport.addWidget(self.play_btn)
        transport.addWidget(self._button("‹", lambda: self._nudge(-1)))
        transport.addWidget(self._button("›", lambda: self._nudge(1)))
        transport.addSpacing(4)
        transport.addWidget(self._label("FRAME", "eyebrow"))
        self.frame_spin = QSpinBox()
        self.frame_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.frame_spin.setKeyboardTracking(False)
        self.frame_spin.setMaximum(0)
        self.frame_spin.setMinimumWidth(78)
        self.frame_spin.valueChanged.connect(self._on_spin)
        transport.addWidget(self.frame_spin)
        transport.addStretch()
        transport.addWidget(self._button("+ Split  M", self._toggle_mark_at_playhead))
        transport.addWidget(self._button("Remove", self._delete_nearest_mark, "quiet"))
        timeline_layout.addLayout(transport)
        trim_row = QHBoxLayout()
        trim_row.setSpacing(8)
        self.trim_in_btn = self._button("Keep in  I", lambda: self._set_trim_boundary(True))
        self.trim_in_btn.setToolTip("Keep the current frame and everything after it")
        self.trim_out_btn = self._button("Keep out  O", lambda: self._set_trim_boundary(False))
        self.trim_out_btn.setToolTip("Keep the current frame and everything before it")
        self.trim_range_label = self._label("Mark the first and last frames to keep")
        self.trim_range_label.setWordWrap(True)
        self.trim_reset_btn = self._button("Reset trim", self._reset_trim_range, "quiet")
        self.trim_apply_btn = self._button("Trim source…", self._on_trim_source, "trim")
        for button in (self.trim_in_btn, self.trim_out_btn, self.trim_reset_btn, self.trim_apply_btn):
            button.setEnabled(False)
        trim_row.addWidget(self.trim_in_btn)
        trim_row.addWidget(self.trim_out_btn)
        trim_row.addWidget(self.trim_range_label, 1)
        trim_row.addWidget(self.trim_reset_btn)
        trim_row.addWidget(self.trim_apply_btn)
        timeline_layout.addLayout(trim_row)
        layout.addWidget(timeline_card)
        joint_card, joint_layout, joint_header = self._card("Joint signals")
        joint_header.addWidget(self._label("● LEFT", "leftLegend"))
        joint_header.addWidget(self._label("● RIGHT", "rightLegend"))
        self.joint_plot = JointPlot()
        self.joint_plot.frameClicked.connect(self._seek)
        joint_layout.addWidget(self.joint_plot, 1)
        layout.addWidget(joint_card, 1)
        return box

    def _build_right(self) -> QWidget:
        box, layout, header = self._card("Segments")
        box.setMinimumWidth(360)
        self.segment_count = self._label("0 SEGMENTS", "eyebrow")
        header.addWidget(self.segment_count)
        description = self._label("Split the episode, name each subtask,\nthen export the segments you need.")
        description.setWordWrap(True)
        layout.addWidget(description)
        layout.addSpacing(3)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["#", "In", "Out", "Length", "Subtask", "Export"])
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        header = self.table.horizontalHeader()
        header.setMinimumSectionSize(24)
        for col in (0, 1, 2, 3, 5):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(43)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.cellClicked.connect(self._on_segment_clicked)
        layout.addWidget(self.table, 1)
        self.segment_summary = self._label("No segments yet", "badge")
        layout.addWidget(self.segment_summary)
        layout.addSpacing(12)
        preset_header = QHBoxLayout()
        preset_header.addWidget(self._label("SUBTASK PRESETS", "eyebrow"))
        preset_header.addStretch()
        self.manage_presets_btn = self._button("Manage…", self._on_manage_presets, "quiet")
        self.manage_presets_btn.setToolTip("Add, rename or delete reusable subtask names")
        preset_header.addWidget(self.manage_presets_btn)
        layout.addLayout(preset_header)
        preset_row = QHBoxLayout()
        self.preset_edit = QLineEdit()
        self.preset_edit.setPlaceholderText("New subtask name")
        self.preset_edit.returnPressed.connect(self._on_add_preset)
        preset_row.addWidget(self.preset_edit, 1)
        preset_row.addWidget(self._button("Add", self._on_add_preset))
        layout.addLayout(preset_row)
        layout.addSpacing(12)
        self.confirm_btn = self._button("Export segments  →", self._on_confirm, "primary")
        self.confirm_btn.setMinimumHeight(44)
        layout.addWidget(self.confirm_btn)
        hint = self._label("Space  Play / pause\n← →  Step frame     M  Add split\nDelete  Remove nearest split")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(self._button("Load robot model…", self._on_load_urdf, "quiet"))
        return box

    def _bind_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._toggle_play)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self._nudge(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self._nudge(1))
        QShortcut(QKeySequence("Shift+Left"), self, lambda: self._nudge_seconds(-1))
        QShortcut(QKeySequence("Shift+Right"), self, lambda: self._nudge_seconds(1))
        QShortcut(QKeySequence(Qt.Key.Key_M), self, self._toggle_mark_at_playhead)
        QShortcut(QKeySequence(Qt.Key.Key_I), self, lambda: self._set_trim_boundary(True))
        QShortcut(QKeySequence(Qt.Key.Key_O), self, lambda: self._set_trim_boundary(False))
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, self._delete_nearest_mark)
        QShortcut(QKeySequence(Qt.Key.Key_Backspace), self, self._delete_nearest_mark)

        open_act = QAction("Open folder", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._on_open_folder)
        self.addAction(open_act)

    # --- queue ----------------------------------------------------------------

    def _on_open_folder(self) -> None:
        if self._trim_thread is not None:
            return
        path = QFileDialog.getExistingDirectory(self, "Select task or episode folder")
        if path:
            self._add_dirs([Path(path)])

    def _add_dirs(self, dirs: list[Path]) -> None:
        if self._trim_thread is not None:
            return
        new_refs = discover_many(dirs)
        existing = {r.path.resolve() for r in self.refs}
        added = 0
        for ref in new_refs:
            if ref.path.resolve() in existing:
                continue
            self.refs.append(ref)
            existing.add(ref.path.resolve())
            added += 1
        self._refresh_list()
        if added and self.list_widget.currentRow() < 0:
            self.list_widget.setCurrentRow(0)
        self.statusBar().showMessage(f"Loaded {len(self.refs)} episodes (+{added})")

    def _clear_queue(self) -> None:
        if self._trim_thread is not None:
            return
        if not self._persist():
            return
        self.refs.clear()
        self.list_widget.clear()
        self._unload()
        self._refresh_list()

    def _refresh_list(self) -> None:
        current = self.list_widget.currentRow()
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        completed = 0
        for ref in self.refs:
            if ref.is_exported:
                flag = "●"
                completed += 1
            elif ref.has_annotation:
                flag = "◐"
            else:
                flag = "○"
            item = QListWidgetItem(
                f"{flag}  Episode {ref.episode_id}\n     {ref.n_frames:,} frames  ·  {ref.duration_s:.1f}s"
            )
            item.setToolTip(f"{ref.task_name}\n{ref.path}")
            item.setSizeHint(QSize(0, 70))
            item.setData(Qt.ItemDataRole.UserRole, str(ref.path))
            self.list_widget.addItem(item)
        if 0 <= current < self.list_widget.count():
            self.list_widget.setCurrentRow(current)
        self.list_widget.blockSignals(False)
        self.queue_count.setText(str(len(self.refs)))
        self.queue_summary.setText(f"{completed} of {len(self.refs)} exported" if self.refs else "No episodes loaded")

    def _move_selection(self, delta: int) -> None:
        if self._trim_thread is not None:
            return
        if not self.refs:
            return
        row = max(0, min(self.list_widget.currentRow() + delta, len(self.refs) - 1))
        self.list_widget.setCurrentRow(row)

    def _on_select_episode(self, row: int) -> None:
        if row < 0 or row >= len(self.refs):
            return
        self._load_ref(self.refs[row])

    # --- load / display -------------------------------------------------------

    def _unload(self) -> None:
        self.timer.stop()
        self.play_btn.setText("Play")
        if self.videos is not None:
            self.videos.close()
            self.videos = None
        self.episode = None
        self.annotation = None
        self._refresh_trim_controls()
        self.frame = 0
        for lab in self.cam_labels.values():
            lab.set_frame(None)
        self.table.setRowCount(0)
        self.timeline.set_state(0, 0, None)
        self.joint_plot.set_data(np.zeros((0, 7)), np.zeros((0, 7)), 0, [])
        self.robot_panel.set_status(self.robot.status)
        self.time_label.setText("0.00s / 0.00s")
        self.session_title.setText("Your annotation workspace")
        self.save_status.setText("LOCAL WORKSPACE")
        self._update_segment_summary()
        self.robot.reset_trail()
        if self.robot.available:
            idle = np.array([0.0, 0.55, 0.65, 0.0, -0.4, 0.0, 0.035])
            self.robot.update(idle, idle)
        self._refresh_robot_view()

    def _load_ref(self, ref: EpisodeRef) -> None:
        if self._trim_thread is not None:
            return
        if not self._persist():
            return
        self._unload()
        try:
            episode = load_episode(ref.path)
            existing = load_annotation(ref.path)
            videos = VideoBank(episode.videos) if episode.videos else None
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self.episode = episode
        for index, queued in enumerate(self.refs):
            if queued.path == ref.path:
                self.refs[index] = episode.ref
        self._refresh_list()
        self.videos = videos
        self.annotation = annotation_from_episode(episode, existing)
        self.session_title.setText(f"{ref.task_name}  /  Episode {ref.episode_id}")
        self.save_status.setText("EXPORTED" if self.annotation.exported_at else "READY TO ANNOTATE")
        self.timer.setInterval(max(1, round(1000 / episode.fps)))
        self.frame = 0
        self.frame_spin.blockSignals(True)
        self.frame_spin.setMaximum(max(episode.n_frames - 1, 0))
        self.frame_spin.setValue(0)
        self.frame_spin.blockSignals(False)
        self.robot.reset_trail()
        if self.robot.available:
            self.robot.bake_trails(episode.left_joint_aligned, episode.right_joint_aligned)
        self._rebuild_table()
        self._refresh_views()
        if not self.out_edit.text().strip():
            self.out_edit.setPlaceholderText(str(default_output_root(ref.path.parent)))
        self.statusBar().showMessage(str(ref.path))

    def _seek(self, frame: int, *, from_playback: bool = False) -> None:
        if self.episode is None or self._trim_thread is not None:
            return
        self.frame = int(np.clip(frame, 0, max(self.episode.n_frames - 1, 0)))
        if self.timer.isActive() and not from_playback:
            self._play_start_frame = self.frame
            self._play_clock.start()
        if self.frame_spin.value() != self.frame:
            self.frame_spin.blockSignals(True)
            self.frame_spin.setValue(self.frame)
            self.frame_spin.blockSignals(False)
        self._refresh_views()

    def _on_spin(self, value: int) -> None:
        if self.episode is None:
            return
        if value != self.frame:
            self._seek(value)

    def _nudge(self, delta: int) -> None:
        self._seek(self.frame + delta)

    def _nudge_seconds(self, seconds: float) -> None:
        if self.episode is None:
            return
        self._seek(self.frame + int(round(seconds * self.episode.fps)))

    def _toggle_play(self) -> None:
        if self.episode is None or self._trim_thread is not None:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            if self.frame >= self.episode.n_frames - 1:
                self._seek(0)
            self._play_start_frame = self.frame
            self._play_clock.start()
            self.timer.start()
            self.play_btn.setText("Pause")

    def _on_tick(self) -> None:
        if self.episode is None:
            return
        if not self.episode.n_frames:
            self.timer.stop()
            return
        target_ts = int(self.episode.base_ts[self._play_start_frame]) + self._play_clock.elapsed()
        nxt = int(np.searchsorted(self.episode.base_ts, target_ts, side="right") - 1)
        if nxt >= self.episode.n_frames - 1:
            self.timer.stop()
            self.play_btn.setText("Play")
        if nxt != self.frame:
            self._seek(nxt, from_playback=True)

    def _refresh_views(self) -> None:
        ep = self.episode
        if ep is None or self.annotation is None:
            return
        if self.videos is not None:
            frames = self.videos.read(self.frame)
            for key, lab in self.cam_labels.items():
                bgr = frames.get(key)
                lab.set_frame(bgr_to_rgb(bgr) if bgr is not None else None)
        self.timeline.set_state(ep.n_frames, self.frame, self.annotation)
        self._refresh_trim_controls()
        self.joint_plot.set_data(ep.left_joint_aligned, ep.right_joint_aligned, self.frame, self.annotation.marks)
        t = ep.time_s(self.frame)
        def clock_text(seconds: float) -> str:
            return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"
        self.time_label.setText(f"{clock_text(t)}  /  {clock_text(ep.duration_s)}")
        if self.robot.available and self.frame < ep.n_frames:
            self.robot.update(ep.left_joint_aligned[self.frame], ep.right_joint_aligned[self.frame])
            self._refresh_robot_view()
        else:
            self.robot_panel.set_status(self.robot.status)

    def _refresh_robot_view(self) -> None:
        self.robot_panel.set_player(self.robot)
        detail = getattr(self, "_robot_detail", None)
        if detail is not None:
            detail.panel.set_player(self.robot)

    def _expand_robot_view(self) -> None:
        from PySide6.QtWidgets import QDialog

        detail = getattr(self, "_robot_detail", None)
        if detail is not None:
            detail.showNormal()
            detail.raise_()
            detail.activateWindow()
            return

        class DetailDialog(QDialog):
            def closeEvent(dialog, event):
                dialog.panel.cleanup()
                super().closeEvent(event)

        detail = DetailDialog(self)
        self._robot_detail = detail
        detail.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        detail.destroyed.connect(lambda: setattr(self, "_robot_detail", None))
        detail.setWindowTitle("Kirigami · Robot 3D")
        detail.resize(1200, 850)
        layout = QVBoxLayout(detail)
        header = QHBoxLayout()
        header.addWidget(self._label("Robot 3D", "sectionTitle"))
        header.addStretch()
        material = QComboBox()
        material.addItems(["White model", "Arm colors"])
        material.setCurrentIndex(self.material_combo.currentIndex())
        material.currentIndexChanged.connect(self.material_combo.setCurrentIndex)
        self.material_combo.currentIndexChanged.connect(material.setCurrentIndex)
        header.addWidget(material)
        header.addWidget(self._button("Reset view", self._reset_robot_view, "quiet"))
        layout.addLayout(header)
        detail.panel = Robot3DPanel(detail)
        detail.panel.set_player(self.robot)
        detail.panel.orbitRequested.connect(self._orbit_robot)
        detail.panel.zoomRequested.connect(self._zoom_robot)
        layout.addWidget(detail.panel, 1)
        layout.addWidget(self._label("● L / teal     ● R / amber       Drag to orbit · Scroll to zoom"))
        detail.show()

    def _on_material_changed(self, index: int) -> None:
        self.robot.material_mode = "white" if index == 0 else "color"
        self._refresh_robot_view()

    def _orbit_robot(self, dx: float, dy: float) -> None:
        self.robot.orbit(dx, dy)
        self._refresh_robot_view()

    def _zoom_robot(self, steps: float) -> None:
        self.robot.zoom(steps)
        self._refresh_robot_view()

    def _reset_robot_view(self) -> None:
        self.robot.reset_view()
        self._refresh_robot_view()

    def _update_segment_summary(self) -> None:
        segments = self.annotation.segments if self.annotation else []
        ready = len(self.annotation.exportable()) if self.annotation else 0
        self.segment_count.setText(f"{len(segments)} SEGMENTS")
        self.segment_summary.setText(f"{ready} of {len(segments)} segments ready to export" if segments else "No segments yet")

    # --- marks / table --------------------------------------------------------

    def _persist(self) -> bool:
        self._save_timer.stop()
        if self._trim_thread is not None:
            return False
        if self.episode is None or self.annotation is None:
            return True
        try:
            save_annotation(self.episode.ref.path, self.annotation)
        except SourceChangedError as exc:
            # Never write frame indices from an old window into an already trimmed source.
            ref = self.episode.ref
            recovery = ref.path.parent / ".kirigami-backups" / "stale-labels" / uuid.uuid4().hex
            try:
                saved = save_annotation(recovery, self.annotation)
            except (OSError, RuntimeError) as recovery_error:
                QMessageBox.critical(self, "Source changed", f"{exc}\nCould not preserve pending labels: {recovery_error}")
                return False
            self.annotation = None
            self._load_ref(ref)
            QMessageBox.warning(self, "Source reloaded", f"{exc}\nThe updated episode has been reloaded.\nYour previous labels were preserved at:\n{saved}")
            return False
        except (OSError, RuntimeError) as exc:
            self.save_status.setText("SAVE FAILED")
            QMessageBox.critical(self, "Save failed", str(exc))
            return False
        self._refresh_list()
        self.save_status.setText("EXPORTED" if self.annotation.exported_at else "ALL CHANGES SAVED")
        self._update_segment_summary()
        return True

    def _toggle_mark(self, frame: int) -> None:
        if self.annotation is None or self._trim_thread is not None:
            return
        hit = nearest_mark(self.annotation, frame, max_dist=0)
        if hit is not None:
            remove_mark(self.annotation, hit)
        else:
            add_mark(self.annotation, frame)
        self._after_mark_change()

    def _toggle_mark_at_playhead(self) -> None:
        if self._trim_thread is not None:
            return
        if self.annotation is not None and add_mark(self.annotation, self.frame):
            self._after_mark_change()

    def _delete_nearest_mark(self) -> None:
        if self.annotation is None or self._trim_thread is not None:
            return
        hit = nearest_mark(self.annotation, self.frame, max_dist=10**9)
        if hit is None:
            return
        remove_mark(self.annotation, hit)
        self._after_mark_change()

    def _after_mark_change(self) -> None:
        self._rebuild_table()
        self._persist()
        self._refresh_views()

    def _rebuild_table(self) -> None:
        self._update_segment_summary()
        # Old editors may lose focus while rows are replaced. Their row indices
        # no longer identify the same segment after a split or episode change.
        for row in range(self.table.rowCount()):
            old = self.table.cellWidget(row, 4)
            if old is not None:
                old.blockSignals(True)
                old.lineEdit().blockSignals(True)
        if self.annotation is None:
            self.table.setRowCount(0)
            return
        self.table.blockSignals(True)
        segs = self.annotation.segments
        self.table.setRowCount(len(segs))
        for i, seg in enumerate(segs):
            vals = [
                str(i),
                str(seg.start_frame),
                str(seg.end_frame),
                self._seg_time_text(seg.start_frame, seg.end_frame),
            ]
            for col, text in enumerate(vals):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(i, col, item)
            combo = QComboBox()
            combo.setEditable(True)
            combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            combo.addItem("")
            for name in self.presets:
                combo.addItem(name)
            if seg.subtask and combo.findText(seg.subtask) < 0:
                combo.addItem(seg.subtask)
            combo.blockSignals(True)
            combo.setCurrentText(seg.subtask)
            combo.blockSignals(False)
            combo.setProperty("committedSubtask", seg.subtask)
            combo.currentTextChanged.connect(lambda text, row=i: self._on_subtask_changed(row, text))
            combo.lineEdit().editingFinished.connect(lambda row=i, editor=combo: self._commit_subtask(row, editor))
            self.table.setCellWidget(i, 4, combo)
            check = QCheckBox()
            check.setChecked(not seg.discard)
            check.stateChanged.connect(lambda state, row=i: self._on_export_toggled(row, state))
            self.table.setCellWidget(i, 5, check)
        self.table.blockSignals(False)

    def _seg_time_text(self, start: int, end: int) -> str:
        if self.episode is None:
            return ""
        t0, t1 = self.episode.frame_range_timestamps(start, end)
        dt = (t1 - t0) / 1000.0
        return f"{max(dt, 0.0):.1f}s"

    def _on_subtask_changed(self, row: int, text: str) -> None:
        if self.annotation is None or row >= len(self.annotation.segments):
            return
        if self.annotation.segments[row].subtask != text.strip():
            self.annotation.segments[row].subtask = text.strip()
            self.annotation.invalidate_export()
            self.save_status.setText("SAVING…")
            self._update_segment_summary()
            self._save_timer.start()
            self.timeline.update()

    def _commit_subtask(self, row: int, editor: QComboBox) -> None:
        if (self.annotation is None or row >= len(self.annotation.segments)
                or self.table.cellWidget(row, 4) is not editor):
            return
        name = self.annotation.segments[row].subtask
        # An unchanged label can refer to a renamed/deleted preset. Merely
        # focusing that row must not recreate the old name in the shared list.
        if name != editor.property("committedSubtask") and name and name not in self.presets:
            try:
                validate_subtask_name(name)
                presets = add_preset(self.presets, name)
                self._save_presets(presets)
            except (ValueError, OSError, YAMLError) as exc:
                self.statusBar().showMessage(str(exc))
                self._persist()
                return
        editor.setProperty("committedSubtask", name)
        self._persist()

    def _on_export_toggled(self, row: int, state: int) -> None:
        if self.annotation is None or row >= len(self.annotation.segments):
            return
        discard = state == Qt.CheckState.Unchecked.value or state == 0
        if self.annotation.segments[row].discard != discard:
            self.annotation.segments[row].discard = discard
            self.annotation.invalidate_export()
        self._persist()
        self.timeline.update()

    def _on_segment_clicked(self, row: int, col: int) -> None:
        if self.annotation is None or row >= len(self.annotation.segments):
            return
        self._seek(self.annotation.segments[row].start_frame)

    def _on_add_preset(self) -> None:
        name = self.preset_edit.text().strip()
        if not name:
            return
        try:
            validate_subtask_name(name)
            presets = add_preset(self.presets, name)
            self._save_presets(presets)
        except (ValueError, OSError, YAMLError) as exc:
            QMessageBox.warning(self, "Cannot add subtask", str(exc))
            return
        self.preset_edit.clear()

    def _refresh_preset_choices(self) -> None:
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, 4)
            if combo is None:
                continue
            text = combo.currentText()
            cursor = combo.lineEdit().cursorPosition()
            blocked = combo.blockSignals(True)
            edit_blocked = combo.lineEdit().blockSignals(True)
            combo.clear()
            combo.addItems(["", *self.presets])
            if text and combo.findText(text) < 0:
                # Keep this row's existing label visible, even if it is no
                # longer a reusable preset. Other rows won't offer this name.
                combo.addItem(text)
            combo.setCurrentText(text)
            combo.lineEdit().setCursorPosition(cursor)
            combo.lineEdit().blockSignals(edit_blocked)
            combo.blockSignals(blocked)

    def _save_presets(self, names: list[str]) -> None:
        if load_subtask_presets(self.presets_path) != self.presets:
            raise ValueError("The preset file changed in another window. Close and reopen Manage to reload it.")
        save_subtask_presets(self.presets_path, names)
        self.presets = list(names)
        self._refresh_preset_choices()

    def _on_manage_presets(self) -> None:
        if self._preset_dialog is not None:
            self._preset_dialog.raise_()
            self._preset_dialog.activateWindow()
            return
        try:
            self.presets = load_subtask_presets(self.presets_path)
        except (ValueError, OSError, YAMLError) as exc:
            QMessageBox.warning(self, "Cannot load presets", str(exc))
            return
        self._refresh_preset_choices()
        self.timer.stop()
        self.play_btn.setText("Play")
        dialog = PresetDialog(self.presets, self.presets_path, self._save_presets, self)
        self._preset_dialog = dialog
        dialog.finished.connect(self._on_presets_closed)
        dialog.open()

    def _on_presets_closed(self, _result: int) -> None:
        dialog = self._preset_dialog
        self._preset_dialog = None
        if dialog is not None:
            dialog.deleteLater()

    # --- manual head/tail trim ------------------------------------------------

    def _refresh_trim_controls(self) -> None:
        enabled = self.episode is not None and self.annotation is not None and self.episode.n_frames > 0
        self.trim_in_btn.setEnabled(enabled)
        self.trim_out_btn.setEnabled(enabled)
        changed = False
        if enabled:
            start, end = self.annotation.keep_range
            n = self.episode.n_frames
            changed = start > 0 or end < n
            self.trim_range_label.setText(f"Keep {start:,}–{end - 1:,} · {end - start:,} frames\nRemove head {start:,} / tail {n - end:,}")
            self.trim_range_label.setToolTip("Frame numbers start at 0. Both displayed endpoints are retained.\nTrim source writes all synchronized streams back to the imported episode.")
        else:
            self.trim_range_label.setText("Mark the first and last frames to keep")
        self.trim_reset_btn.setEnabled(enabled and changed)
        self.trim_apply_btn.setEnabled(enabled and changed)

    def _set_trim_boundary(self, first: bool) -> None:
        if self.episode is None or self.annotation is None or self._trim_thread is not None or self._export_thread is not None:
            return
        if isinstance(QApplication.focusWidget(), (QLineEdit, QComboBox, QAbstractSpinBox)):
            return
        start, end = self.annotation.keep_range
        if first:
            start = self.frame
        else:
            end = self.frame + 1
        if start >= end:
            self.statusBar().showMessage("Keep in must be at or before Keep out. Reset trim to choose a new range.", 6000)
            return
        self.timer.stop()
        self.play_btn.setText("Play")
        self.annotation.trim_start, self.annotation.trim_end = start, end
        self._persist()
        self._refresh_views()

    def _reset_trim_range(self) -> None:
        if self.annotation is None or self._trim_thread is not None:
            return
        self.annotation.trim_start, self.annotation.trim_end = 0, None
        self._persist()
        self._refresh_views()

    def _on_trim_source(self) -> None:
        if self._trim_thread is not None or self._export_thread is not None or self.episode is None or self.annotation is None:
            return
        start, end = self.annotation.keep_range
        n = self.episode.n_frames
        if not 0 <= start < end <= n or (start == 0 and end == n):
            return
        self.timer.stop()
        self.play_btn.setText("Play")
        if not self._persist():
            return
        source = self.episode.ref.path
        answer = QMessageBox.question(self, "Trim imported source",
            f"Write the trimmed episode back to:\n{source}\n\n"
            f"Keep frames {start}–{end - 1} (both included): {end - start} / {n} frames.\n"
            f"Delete {start} leading and {n - end} trailing frames.\n\n"
            "Camera videos, observations, robot/action/event timestamps, and counts will be trimmed together. "
            "Absolute timestamps stay unchanged. Split labels will be rebased; existing exports will need re-exporting.\n\n"
            f"A complete original backup will be kept under:\n{source.parent / '.kirigami-backups' / source.name}\n\nApply this trim?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._save_timer.stop()
        if self.videos is not None:
            self.videos.close()
            self.videos = None
        self._trim_result, self._trim_error = None, ""
        self._trim_ref = self.episode.ref
        thread = QThread(self)
        worker = TrimWorker(self.episode, self.annotation)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_trim_progress)
        worker.finished.connect(self._on_trim_result)
        worker.failed.connect(self._on_trim_error)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._on_trim_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._trim_thread, self._trim_worker = thread, worker
        self.centralWidget().setEnabled(False)
        self.trim_progress = QProgressDialog("Preparing source trim…", None, 0, 100, self)
        self.trim_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.trim_progress.setMinimumDuration(0)
        self.trim_progress.setValue(0)
        thread.start()

    def _on_trim_progress(self, message: str, fraction: float) -> None:
        self.trim_progress.setLabelText(message)
        self.trim_progress.setValue(int(fraction * 100))

    def _on_trim_result(self, result) -> None:
        self._trim_result = result

    def _on_trim_error(self, message: str) -> None:
        self._trim_error = message

    def _on_trim_thread_finished(self) -> None:
        self._trim_thread, self._trim_worker = None, None
        self.trim_progress.reset()
        self.centralWidget().setEnabled(True)
        # The old annotation's frame numbers must never be saved after source replacement.
        self.annotation = None
        self._load_ref(self._trim_ref)
        if self._trim_error:
            QMessageBox.critical(self, "Source trim failed", self._trim_error)
        elif self._trim_result is not None:
            result = self._trim_result
            self.save_status.setText("SOURCE TRIMMED")
            text = f"Saved {result.n_frames:,} frames back to:\n{result.source}\n\nOriginal backup:\n{result.backup}"
            if result.warning:
                QMessageBox.warning(self, "Source trimmed", text + "\n\n" + result.warning)
            else:
                QMessageBox.information(self, "Source trimmed", text)

    # --- export ---------------------------------------------------------------

    def _output_root(self) -> Path | None:
        if self.out_edit.text().strip():
            return Path(self.out_edit.text().strip()).expanduser()
        if self.out_root_override is not None:
            return self.out_root_override
        if self.episode is None:
            return None
        return default_output_root(self.episode.ref.path.parent)

    def _on_confirm(self) -> None:
        if self._export_thread is not None or self._trim_thread is not None:
            return
        if self.episode is None or self.annotation is None:
            return
        missing = self.annotation.missing_names()
        if missing:
            QMessageBox.warning(self, "Missing subtask", "Every exported segment needs a subtask name.")
            return
        if not self.annotation.exportable():
            QMessageBox.warning(self, "Nothing to export", "No exportable segments.")
            return
        out_root = self._output_root()
        if out_root is None:
            return
        try:
            for seg in self.annotation.exportable():
                validate_subtask_name(seg.subtask)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid subtask", str(exc))
            return
        if not self._persist():
            return
        self.timer.stop()
        self.play_btn.setText("Play")
        self._active_export_root = out_root

        self.confirm_btn.setEnabled(False)
        self.list_widget.setEnabled(False)
        self.progress = QProgressDialog("Exporting…", None, 0, 100, self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setValue(0)

        thread = QThread(self)
        worker = ExportWorker(self.episode, self.annotation, out_root)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_export_progress)
        worker.finished.connect(self._on_export_done)
        worker.failed.connect(self._on_export_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._on_export_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._export_thread = thread
        self._export_worker = worker
        thread.start()

    def _on_export_thread_finished(self) -> None:
        self._export_thread = None
        self._export_worker = None

    def _on_export_progress(self, msg: str, frac: float) -> None:
        if hasattr(self, "progress"):
            self.progress.setLabelText(msg)
            self.progress.setValue(int(frac * 100))

    def _on_export_done(self, written) -> None:
        self.confirm_btn.setEnabled(True)
        self.list_widget.setEnabled(True)
        if hasattr(self, "progress"):
            self.progress.reset()
        if self.episode is not None and self.annotation is not None:
            out_root = self._active_export_root
            if out_root is not None:
                mark_exported(self.annotation, out_root)
                self._persist()
            self._refresh_list()
        QMessageBox.information(self, "Exported", f"Wrote {len(written)} episode(s).")
        self._advance_to_next_pending()

    def _on_export_failed(self, message: str) -> None:
        self.confirm_btn.setEnabled(True)
        self.list_widget.setEnabled(True)
        if hasattr(self, "progress"):
            self.progress.reset()
        QMessageBox.critical(self, "Export failed", message)

    def _advance_to_next_pending(self) -> None:
        cur = self.list_widget.currentRow()
        for i in range(cur + 1, len(self.refs)):
            if not self.refs[i].is_exported:
                self.list_widget.setCurrentRow(i)
                return
        if cur + 1 < len(self.refs):
            self.list_widget.setCurrentRow(cur + 1)

    def _browse_out_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output root")
        if path:
            self.out_edit.setText(path)

    def _on_load_urdf(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select URDF", filter="URDF (*.urdf)")
        if not path:
            return
        mesh = QFileDialog.getExistingDirectory(self, "Select mesh directory (optional)")
        err = self.robot.load(Path(path), Path(mesh) if mesh else None)
        self.robot_panel.set_status(err or self.robot.status)
        self._refresh_robot_view()
        self._refresh_views()

    def closeEvent(self, event) -> None:
        if self._trim_thread is not None:
            QMessageBox.information(self, "Source trim in progress", "Please wait for the source trim to finish before closing.")
            event.ignore()
            return
        if self._export_thread is not None:
            QMessageBox.information(self, "Export in progress", "Please wait for the export to finish before closing.")
            event.ignore()
            return
        if not self._persist():
            event.ignore()
            return
        self.timer.stop()
        if self.videos is not None:
            self.videos.close()
        detail = getattr(self, "_robot_detail", None)
        if detail is not None:
            detail.close()
        self.robot_panel.cleanup()
        super().closeEvent(event)
