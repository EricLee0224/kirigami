"""Kirigami main window: episode queue, cameras, timeline, segments, export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
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
)
from ..exporter import default_output_root, export_annotation
from ..loader import EpisodeRef, LoadedEpisode, discover_many, load_episode
from ..robot3d import Robot3DConfig, Robot3DPlayer
from ..video import VideoBank, bgr_to_rgb
from .widgets import CameraLabel, JointPlot, Robot3DPanel, TimelineBar


class ExportWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str, float)

    def __init__(self, episode: LoadedEpisode, annotation: Annotation, out_root: Path):
        super().__init__()
        self.episode = episode
        self.annotation = annotation
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
        self.resize(1480, 900)

        self.presets_path = presets_path or (Path(__file__).resolve().parents[2] / "subtasks.yaml")
        self.presets = load_subtask_presets(self.presets_path)
        self.out_root_override = Path(out_root) if out_root else None
        self.robot = Robot3DPlayer(robot_config)

        self.refs: list[EpisodeRef] = []
        self.episode: LoadedEpisode | None = None
        self.annotation: Annotation | None = None
        self.videos: VideoBank | None = None
        self.frame = 0
        self._export_thread: QThread | None = None
        self._export_worker: ExportWorker | None = None

        self._build_ui()
        self._bind_shortcuts()

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._on_tick)

        if initial_dirs:
            self._add_dirs(initial_dirs)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)

        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_center())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([280, 780, 420])

        self.statusBar().showMessage("Open a task folder (e.g. romoya-egg-stage2-0912_100)")

    def _build_left(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.addWidget(QLabel("Episodes"))
        btn_row = QHBoxLayout()
        open_btn = QPushButton("Open folder…")
        open_btn.clicked.connect(self._on_open_folder)
        btn_row.addWidget(open_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_queue)
        btn_row.addWidget(clear_btn)
        layout.addLayout(btn_row)

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self._on_select_episode)
        layout.addWidget(self.list_widget, 1)

        nav = QHBoxLayout()
        prev_btn = QPushButton("Prev")
        prev_btn.clicked.connect(lambda: self._move_selection(-1))
        next_btn = QPushButton("Next")
        next_btn.clicked.connect(lambda: self._move_selection(1))
        nav.addWidget(prev_btn)
        nav.addWidget(next_btn)
        layout.addLayout(nav)

        layout.addWidget(QLabel("Output root"))
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("default: <task>_sliced next to the task dir")
        if self.out_root_override:
            self.out_edit.setText(str(self.out_root_override))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_out_root)
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(browse)
        layout.addLayout(out_row)
        return box

    def _build_center(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        cams = QHBoxLayout()
        self.cam_labels = {
            "base_0": CameraLabel("base_0"),
            "left_wrist_0": CameraLabel("left_wrist_0"),
            "right_wrist_0": CameraLabel("right_wrist_0"),
        }
        for key in ("base_0", "left_wrist_0", "right_wrist_0"):
            wrap = QVBoxLayout()
            wrap.addWidget(QLabel(key))
            wrap.addWidget(self.cam_labels[key], 1)
            holder = QWidget()
            holder.setLayout(wrap)
            cams.addWidget(holder)
        layout.addLayout(cams, 3)

        transport = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        self.frame_spin = QSpinBox()
        self.frame_spin.setMaximum(0)
        self.frame_spin.valueChanged.connect(self._on_spin)
        self.time_label = QLabel("0.00s / 0.00s")
        mark_btn = QPushButton("Mark (M)")
        mark_btn.clicked.connect(self._toggle_mark_at_playhead)
        del_btn = QPushButton("Del mark")
        del_btn.clicked.connect(self._delete_nearest_mark)
        transport.addWidget(self.play_btn)
        transport.addWidget(QLabel("Frame"))
        transport.addWidget(self.frame_spin)
        transport.addWidget(self.time_label, 1)
        transport.addWidget(mark_btn)
        transport.addWidget(del_btn)
        layout.addLayout(transport)

        self.timeline = TimelineBar()
        self.timeline.frameClicked.connect(self._seek)
        self.timeline.markToggleRequested.connect(self._toggle_mark)
        layout.addWidget(self.timeline)

        self.joint_plot = JointPlot()
        self.joint_plot.frameClicked.connect(self._seek)
        layout.addWidget(self.joint_plot, 1)
        return box

    def _build_right(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.addWidget(QLabel("Robot 3D"))
        self.robot_panel = Robot3DPanel()
        self.robot_panel.set_status(self.robot.status)
        layout.addWidget(self.robot_panel, 2)

        urdf_row = QHBoxLayout()
        urdf_btn = QPushButton("Load URDF…")
        urdf_btn.clicked.connect(self._on_load_urdf)
        urdf_row.addWidget(urdf_btn)
        layout.addLayout(urdf_row)

        layout.addWidget(QLabel("Segments"))
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["#", "Start", "End", "Time", "Subtask", "Export"])
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.cellClicked.connect(self._on_segment_clicked)
        layout.addWidget(self.table, 2)

        preset_row = QHBoxLayout()
        self.preset_edit = QLineEdit()
        self.preset_edit.setPlaceholderText("new subtask name")
        add_preset_btn = QPushButton("Add name")
        add_preset_btn.clicked.connect(self._on_add_preset)
        preset_row.addWidget(self.preset_edit, 1)
        preset_row.addWidget(add_preset_btn)
        layout.addLayout(preset_row)

        self.confirm_btn = QPushButton("Confirm export")
        self.confirm_btn.setMinimumHeight(36)
        self.confirm_btn.clicked.connect(self._on_confirm)
        layout.addWidget(self.confirm_btn)

        hint = QLabel("Space play · ←/→ frame · Shift+arrow 1s · M mark · Del nearest mark")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888;")
        layout.addWidget(hint)
        return box

    def _bind_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._toggle_play)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self._nudge(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self._nudge(1))
        QShortcut(QKeySequence("Shift+Left"), self, lambda: self._nudge_seconds(-1))
        QShortcut(QKeySequence("Shift+Right"), self, lambda: self._nudge_seconds(1))
        QShortcut(QKeySequence(Qt.Key.Key_M), self, self._toggle_mark_at_playhead)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, self._delete_nearest_mark)
        QShortcut(QKeySequence(Qt.Key.Key_Backspace), self, self._delete_nearest_mark)

        open_act = QAction("Open folder", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._on_open_folder)
        self.addAction(open_act)

    # --- queue ----------------------------------------------------------------

    def _on_open_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select task or episode folder")
        if path:
            self._add_dirs([Path(path)])

    def _add_dirs(self, dirs: list[Path]) -> None:
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
        self.refs.clear()
        self.list_widget.clear()
        self._unload()

    def _refresh_list(self) -> None:
        current = self.list_widget.currentRow()
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for ref in self.refs:
            if ref.is_exported:
                flag = "●"
            elif ref.has_annotation:
                flag = "◐"
            else:
                flag = "○"
            item = QListWidgetItem(
                f"{flag} {ref.task_name}/{ref.episode_id}  {ref.n_frames}f  {ref.duration_s:.1f}s"
            )
            item.setData(Qt.ItemDataRole.UserRole, str(ref.path))
            self.list_widget.addItem(item)
        if 0 <= current < self.list_widget.count():
            self.list_widget.setCurrentRow(current)
        self.list_widget.blockSignals(False)

    def _move_selection(self, delta: int) -> None:
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
        self.frame = 0
        for lab in self.cam_labels.values():
            lab.set_frame(None)

    def _load_ref(self, ref: EpisodeRef) -> None:
        self._unload()
        try:
            episode = load_episode(ref.path)
            videos = VideoBank(episode.videos) if episode.videos else None
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self.episode = episode
        self.videos = videos
        existing = load_annotation(ref.path)
        self.annotation = annotation_from_episode(episode, existing)
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

    def _seek(self, frame: int) -> None:
        if self.episode is None:
            return
        self.frame = int(np.clip(frame, 0, max(self.episode.n_frames - 1, 0)))
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
        if self.episode is None:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            self.timer.start()
            self.play_btn.setText("Pause")

    def _on_tick(self) -> None:
        if self.episode is None:
            return
        nxt = self.frame + 1
        if nxt >= self.episode.n_frames:
            self.timer.stop()
            self.play_btn.setText("Play")
            return
        self._seek(nxt)

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
        self.joint_plot.set_data(ep.left_joint_aligned, ep.right_joint_aligned, self.frame, self.annotation.marks)
        t = ep.time_s(self.frame)
        self.time_label.setText(f"{t:.2f}s / {ep.duration_s:.2f}s   frame {self.frame}/{max(ep.n_frames - 1, 0)}")
        if self.robot.available and self.frame < ep.n_frames:
            self.robot.update(ep.left_joint_aligned[self.frame], ep.right_joint_aligned[self.frame])
            rgb = self.robot.render_rgb()
            self.robot_panel.set_trail_image(rgb, self.robot.status)
        else:
            self.robot_panel.set_status(self.robot.status)

    # --- marks / table --------------------------------------------------------

    def _persist(self) -> None:
        if self.episode is None or self.annotation is None:
            return
        save_annotation(self.episode.ref.path, self.annotation)
        self._refresh_list()

    def _toggle_mark(self, frame: int) -> None:
        if self.annotation is None:
            return
        hit = nearest_mark(self.annotation, frame, max_dist=max(4, int(self.annotation.n_frames / 200)))
        if hit is not None:
            remove_mark(self.annotation, hit)
        else:
            add_mark(self.annotation, frame)
        self._after_mark_change()

    def _toggle_mark_at_playhead(self) -> None:
        self._toggle_mark(self.frame)

    def _delete_nearest_mark(self) -> None:
        if self.annotation is None:
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
            combo.addItem("")
            for name in self.presets:
                combo.addItem(name)
            if seg.subtask and combo.findText(seg.subtask) < 0:
                combo.addItem(seg.subtask)
            combo.blockSignals(True)
            combo.setCurrentText(seg.subtask)
            combo.blockSignals(False)
            combo.currentTextChanged.connect(lambda text, row=i: self._on_subtask_changed(row, text))
            self.table.setCellWidget(i, 4, combo)
            check = QCheckBox()
            check.setChecked(not seg.discard)
            check.stateChanged.connect(lambda state, row=i: self._on_export_toggled(row, state))
            self.table.setCellWidget(i, 5, check)
        self.table.blockSignals(False)

    def _seg_time_text(self, start: int, end: int) -> str:
        if self.episode is None:
            return ""
        dt = self.episode.time_s(max(end - 1, start)) - self.episode.time_s(start)
        return f"{end - start}f / {max(dt, 0.0):.1f}s"

    def _on_subtask_changed(self, row: int, text: str) -> None:
        if self.annotation is None or row >= len(self.annotation.segments):
            return
        self.annotation.segments[row].subtask = text.strip()
        if text.strip():
            self.presets = add_preset(self.presets, text.strip())
            save_subtask_presets(self.presets_path, self.presets)
        self._persist()
        self._refresh_views()

    def _on_export_toggled(self, row: int, state: int) -> None:
        if self.annotation is None or row >= len(self.annotation.segments):
            return
        self.annotation.segments[row].discard = state == Qt.CheckState.Unchecked.value or state == 0
        self._persist()
        self._refresh_views()

    def _on_segment_clicked(self, row: int, col: int) -> None:
        if self.annotation is None or row >= len(self.annotation.segments):
            return
        self._seek(self.annotation.segments[row].start_frame)

    def _on_add_preset(self) -> None:
        name = self.preset_edit.text().strip()
        if not name:
            return
        self.presets = add_preset(self.presets, name)
        save_subtask_presets(self.presets_path, self.presets)
        self.preset_edit.clear()
        self._rebuild_table()

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
        save_annotation(self.episode.ref.path, self.annotation)

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
        thread.finished.connect(worker.deleteLater)
        self._export_thread = thread
        self._export_worker = worker
        thread.start()

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
            out_root = self._output_root()
            if out_root is not None:
                mark_exported(self.annotation, out_root)
                save_annotation(self.episode.ref.path, self.annotation)
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
        self._refresh_views()

    def closeEvent(self, event) -> None:
        if self.annotation is not None and self.episode is not None:
            save_annotation(self.episode.ref.path, self.annotation)
        if self.videos is not None:
            self.videos.close()
        super().closeEvent(event)
