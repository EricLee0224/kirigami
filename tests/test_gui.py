"""Desktop regressions. Run on the host's X11/Wayland session, not offscreen."""

import os
import tempfile
import time
import shutil
import threading
import unittest
from pathlib import Path
from dataclasses import asdict
from unittest.mock import patch

import yaml

from PySide6.QtCore import QCoreApplication, QEvent, QThread, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from kirigami.annotation import Annotation, Segment, load_subtask_presets, mark_exported
from kirigami.app import desktop_startup_error
from kirigami.gui.main_window import MainWindow
from kirigami.gui.task_progress import TaskProgressDialog
from kirigami.robot3d import Robot3DConfig
from kirigami.annotation import load_annotation
from kirigami.loader import load_episode
from kirigami.trimming import trim_source
from tests.test_core import _write_synthetic
from tests.test_trimming import contents
from tests.test_camera_sync import write_async_episode
from tests.test_infrared import write_infrared_episode


@unittest.skipUnless(os.environ.get("KIRIGAMI_DESKTOP_TESTS") == "1", "enable KIRIGAMI_DESKTOP_TESTS=1 in a desktop session")
class TestDesktopWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        error = desktop_startup_error()
        if error:
            raise RuntimeError(error)
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.presets = self.root / "subtasks.yaml"
        self.win = MainWindow(presets_path=self.presets, robot_config=Robot3DConfig(urdf=None))
        self.win.show()
        self.win.activateWindow()
        QTest.qWait(100)

    def tearDown(self):
        # Let Python export workers run while pumping the desktop event loop.
        deadline = time.monotonic() + 10
        with patch.object(QMessageBox, "information"), patch.object(QMessageBox, "critical"), patch.object(QMessageBox, "warning"):
            while (self.win._export_thread is not None or self.win._trim_thread is not None) and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.02)
            self.win.close()
        self.app.processEvents()
        self.temp.cleanup()

    def test_long_episode_can_have_nearby_marks(self):
        self.win.annotation = Annotation("synthetic", "0007", 18000, marks=[100], segments=[Segment(0, 100), Segment(100, 18000)])
        self.win.frame = 150
        self.win._toggle_mark_at_playhead()
        self.win._toggle_mark_at_playhead()
        self.assertEqual(self.win.annotation.marks, [100, 150])

    def test_presets_contain_committed_names_only(self):
        self.win.annotation = Annotation("synthetic", "0007", 20, segments=[Segment(0, 20)])
        self.win._rebuild_table()
        combo = self.win.table.cellWidget(0, 4)
        combo.setFocus()
        QTest.keyClicks(combo.lineEdit(), "pick")
        self.assertEqual(self.win.annotation.segments[0].subtask, "pick")
        self.assertEqual(load_subtask_presets(self.presets), [])
        QTest.keyClick(combo.lineEdit(), Qt.Key.Key_Tab)
        self.assertEqual(load_subtask_presets(self.presets), ["pick"])

    def _open_presets(self, names):
        self.presets.write_text(yaml.safe_dump({"project": "demo", "subtasks": names}), encoding="utf-8")
        QTest.mouseClick(self.win.manage_presets_btn, Qt.MouseButton.LeftButton)
        self.app.processEvents()
        dialog = self.win._preset_dialog
        self.assertIsNotNone(dialog)
        self.assertTrue(dialog.isVisible())
        # X11 activates a newly mapped dialog asynchronously; shortcuts are
        # routed by the active widget window, not the QTest event receiver.
        deadline = time.monotonic() + 2
        while not dialog.isActiveWindow() and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertTrue(dialog.isActiveWindow())
        return dialog

    def test_manage_presets_preserves_labels_and_does_not_resurrect_old_names(self):
        source = _write_synthetic(self.root)
        self.win._add_dirs([source])
        self.win.annotation.marks = [5, 10]
        self.win.annotation.segments = [Segment(0, 5, "pick"), Segment(5, 10), Segment(10, 20, "place")]
        mark_exported(self.win.annotation, self.root / "output")
        self.win._rebuild_table()
        self.win._persist()
        before = contents(source)
        labels = asdict(self.win.annotation)

        dialog = self._open_presets(["pick", "place", "reset"])
        dialog.list.setCurrentRow(0)
        dialog.name_edit.setText("grasp")
        QTest.mouseClick(dialog.rename_btn, Qt.MouseButton.LeftButton)
        self.assertEqual(load_subtask_presets(self.presets), ["grasp", "place", "reset"])
        dialog.list.setCurrentRow(1)
        QTest.mouseClick(dialog.delete_btn, Qt.MouseButton.LeftButton)
        self.assertEqual(load_subtask_presets(self.presets), ["grasp", "reset"])
        self.assertEqual(asdict(self.win.annotation), labels)
        self.assertEqual(contents(source), before)
        self.assertEqual(yaml.safe_load(self.presets.read_text())["project"], "demo")
        dialog.accept()
        self.app.processEvents()
        self.win.activateWindow()
        deadline = time.monotonic() + 2
        while not self.win.isActiveWindow() and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertTrue(self.win.isActiveWindow())

        # Old labels remain in their own rows, but aren't offered to other rows.
        self.assertEqual(self.win.table.cellWidget(0, 4).currentText(), "pick")
        empty = self.win.table.cellWidget(1, 4)
        self.assertEqual(empty.findText("pick"), -1)
        self.assertEqual(empty.findText("place"), -1)
        self.assertGreaterEqual(empty.findText("grasp"), 0)
        for row in (0, 2):
            combo = self.win.table.cellWidget(row, 4)
            combo.setFocus()
            self.app.processEvents()
            self.assertTrue(combo.lineEdit().hasFocus())
            QTest.keyClick(combo.lineEdit(), Qt.Key.Key_Tab)
        # Rebuilding the table and reopening the episode must also retain deletions.
        self.win._rebuild_table()
        self.win._load_ref(self.win.episode.ref)
        self.assertEqual(load_subtask_presets(self.presets), ["grasp", "reset"])
        self.assertEqual(asdict(self.win.annotation), labels)
        self.win._on_manage_presets()
        self.assertEqual(self.win._preset_dialog.names, ["grasp", "reset"])
        self.win._preset_dialog.accept()

    def test_manage_presets_batch_delete_all_and_add_again(self):
        self.win.annotation = Annotation("synthetic", "0007", 20, marks=[10],
                                         segments=[Segment(0, 10), Segment(10, 20)])
        self.win._rebuild_table()
        dialog = self._open_presets(["pick", "place", "reset"])
        dialog.list.selectAll()
        dialog.list.setFocus()
        QTest.keyClick(dialog.list, Qt.Key.Key_Delete)
        self.assertEqual(load_subtask_presets(self.presets), [])
        self.assertEqual(self.win.annotation.marks, [10])
        self.assertTrue(dialog.empty.isVisible())
        self.assertFalse(dialog.delete_btn.isEnabled())
        dialog.name_edit.setText("抓取物体")
        QTest.mouseClick(dialog.add_btn, Qt.MouseButton.LeftButton)
        self.assertEqual(load_subtask_presets(self.presets), ["抓取物体"])
        self.assertFalse(dialog.empty.isVisible())
        dialog.accept()
        self.app.processEvents()
        self.win.preset_edit.setText("place")
        QTest.keyClick(self.win.preset_edit, Qt.Key.Key_Return)
        self.assertEqual(load_subtask_presets(self.presets), ["抓取物体", "place"])

    def test_manage_presets_rejects_duplicates_invalid_names_and_failed_writes(self):
        dialog = self._open_presets(["pick", "place"])
        dialog.list.setCurrentRow(0)
        for name in ("place", "invalid/name"):
            dialog.name_edit.setText(name)
            QTest.mouseClick(dialog.rename_btn, Qt.MouseButton.LeftButton)
            self.assertEqual(load_subtask_presets(self.presets), ["pick", "place"])
            self.assertEqual(dialog.feedback.property("role"), "error")
        # A failed atomic replacement must leave both the file and GUI intact.
        before = self.presets.read_bytes()
        with patch("kirigami.annotation.os.replace", side_effect=OSError("Disk write failed")):
            QTest.mouseClick(dialog.delete_btn, Qt.MouseButton.LeftButton)
        self.assertEqual(self.presets.read_bytes(), before)
        self.assertEqual(self.win.presets, ["pick", "place"])
        self.assertEqual(dialog.names, ["pick", "place"])
        self.assertIn("Disk write failed", dialog.feedback.text())
        QTest.mouseClick(dialog.delete_btn, Qt.MouseButton.LeftButton)
        self.assertEqual(load_subtask_presets(self.presets), ["place"])
        dialog.accept()

    def test_manage_presets_detects_external_edits_and_reloads(self):
        dialog = self._open_presets(["pick", "place"])
        dialog.list.setCurrentRow(0)
        self.presets.write_text("subtasks: [external]\n", encoding="utf-8")
        QTest.mouseClick(dialog.delete_btn, Qt.MouseButton.LeftButton)
        self.assertEqual(load_subtask_presets(self.presets), ["external"])
        self.assertIn("another window", dialog.feedback.text())
        dialog.accept()
        self.app.processEvents()
        self.win._on_manage_presets()
        dialog = self.win._preset_dialog
        self.assertEqual(dialog.names, ["external"])
        dialog.list.setCurrentRow(0)
        QTest.mouseClick(dialog.delete_btn, Qt.MouseButton.LeftButton)
        self.assertEqual(load_subtask_presets(self.presets), [])
        dialog.accept()

    def test_load_play_annotate_export_reopen(self):
        source = _write_synthetic(self.root)
        self.win._add_dirs([source])
        self.win.out_edit.setText(str(self.root / "output"))
        self.win._toggle_play()
        QTest.qWait(180)
        self.assertGreater(self.win.frame, 0)
        self.win._toggle_play()
        self.win._seek(10)
        self.win._toggle_mark_at_playhead()
        for row, name in enumerate(["pick", "place"]):
            self.win.table.cellWidget(row, 4).setCurrentText(name)
        with patch.object(QMessageBox, "information"), patch.object(QMessageBox, "critical") as errors:
            self.win._on_confirm()
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                self.app.processEvents()
                if self.win._export_thread is None:
                    break
                time.sleep(0.025)
            self.assertIsNone(self.win._export_thread)
            self.assertFalse(errors.called)
        self.assertTrue(self.win.episode.ref.is_exported)
        self.assertEqual(len(list((self.root / "output").glob("*/*/slice_meta.json"))), 2)
        self.win._load_ref(self.win.episode.ref)
        self.assertEqual(self.win.annotation.marks, [10])
        self.win._seek(5)
        self.win._toggle_mark_at_playhead()
        self.assertFalse(self.win.episode.ref.is_exported)

    def _make_episode_library(self, count=30):
        template = _write_synthetic(self.root / "fixture")
        cameras = ["base_0_rgb.mp4", "left_wrist_0_rgb.mp4", "right_wrist_0_rgb.mp4"]
        sources = []
        for index in range(count):
            source = self.root / "episodes" / f"{index:04d}"
            shutil.copytree(template, source)
            # Distinct preview colors catch a stale camera image after switching.
            shutil.copyfile(template / "camera" / cameras[index % 3], source / "camera/base_0_rgb.mp4")
            sources.append(source)
        self.win._add_dirs([sources[0].parent])
        return sources

    def _click_episode(self, row):
        view = self.win.list_widget
        item = view.item(row)
        view.scrollToItem(item)
        QTest.qWait(25)
        scroll = view.verticalScrollBar().value()
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=view.visualItemRect(item).center())
        QTest.qWait(25)
        return scroll

    def _assert_episode_selected(self, source):
        view = self.win.list_widget
        self.assertEqual(self.win.episode.ref.path, source)
        self.assertEqual(view.currentItem().data(Qt.ItemDataRole.UserRole), str(source))
        self.assertEqual([item.data(Qt.ItemDataRole.UserRole) for item in view.selectedItems()], [str(source)])
        self.assertEqual(self.win.annotation.source_episode_id, source.name)
        self.assertEqual(self.win.annotation.source_path, str(source))
        self.assertEqual(self.win.session_title.text(), f"{source.parent.name}  /  Episode {source.name}")
        self.assertEqual(self.win.videos.paths["base_0"], source / "camera/base_0_rgb.mp4")

    def test_episode_mouse_clicks_preserve_scroll_selection_and_preview(self):
        sources = self._make_episode_library()
        view = self.win.list_widget
        # setCurrentRow alone misses the selection changes on mouse release.
        for row in [15, 25, 26, 3, 29, 0, 16, 5]:
            with self.subTest(row=row):
                scroll = self._click_episode(row)
                self._assert_episode_selected(sources[row])
                self.assertEqual(view.verticalScrollBar().value(), scroll)
                pixel = self.win.cam_labels["base_0"]._pixmap.toImage().pixelColor(32, 24)
                rgb = [pixel.red(), pixel.green(), pixel.blue()]
                self.assertEqual(rgb.index(max(rgb)), row % 3)
                self.win.table.cellWidget(0, 4).setCurrentText(f"task_{row}")
                QTest.qWait(350)
                self._assert_episode_selected(sources[row])
                self.assertEqual(view.verticalScrollBar().value(), scroll)
                self.assertEqual(load_annotation(sources[row]).segments[0].subtask, f"task_{row}")

    def test_adding_episodes_keeps_current_item_and_scroll(self):
        sources = self._make_episode_library()
        self._click_episode(25)
        view = self.win.list_widget
        current = view.currentItem()
        scroll = view.verticalScrollBar().value()
        # A second task can contain the same episode number: identity is its path.
        extra = self.root / "other_task" / sources[25].name
        shutil.copytree(sources[25], extra)
        self.win._add_dirs([extra])
        self.app.processEvents()
        self.assertIs(view.currentItem(), current)
        self.assertEqual(view.verticalScrollBar().value(), scroll)
        self._assert_episode_selected(sources[25])
        self._click_episode(30)
        self._assert_episode_selected(extra)

    def test_failed_save_restores_selection_to_loaded_episode(self):
        sources = self._make_episode_library(3)
        self.win.table.cellWidget(0, 4).setCurrentText("pending_label")
        with patch("kirigami.gui.main_window.save_annotation", side_effect=OSError("Disk write failed")), \
             patch.object(QMessageBox, "critical") as errors:
            self._click_episode(2)
            errors.assert_called_once()
        self._assert_episode_selected(sources[0])
        self.assertEqual(self.win.annotation.segments[0].subtask, "pending_label")

    def test_failed_load_clears_selection_and_old_preview(self):
        self._make_episode_library(3)
        with patch("kirigami.gui.main_window.load_episode", side_effect=RuntimeError("Cannot read episode")), \
             patch.object(QMessageBox, "critical") as errors:
            self._click_episode(2)
            errors.assert_called_once()
        self.assertIsNone(self.win.episode)
        self.assertEqual(self.win.list_widget.currentRow(), -1)
        self.assertEqual(self.win.list_widget.selectedItems(), [])
        for camera in self.win.cam_labels.values():
            self.assertIsNone(camera._pixmap)

    def _wait_for_trim(self):
        deadline = time.monotonic() + 20
        while self.win._trim_thread is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.02)
        self.assertIsNone(self.win._trim_thread)

    def _choose_trim(self):
        self.win._seek(5)
        QTest.mouseClick(self.win.trim_in_btn, Qt.MouseButton.LeftButton)
        self.win._seek(15)
        QTest.mouseClick(self.win.trim_out_btn, Qt.MouseButton.LeftButton)
        self.assertEqual(self.win.annotation.keep_range, (5, 16))

    def test_trim_source_applies_on_click_without_dialogs_or_backup(self):
        source = _write_synthetic(self.root)
        self.win._add_dirs([source])
        self._choose_trim()
        self.assertEqual(load_annotation(source).keep_range, (5, 16))
        with patch.object(QMessageBox, "question") as questions, \
             patch.object(QMessageBox, "information") as notifications, \
             patch.object(QMessageBox, "warning") as warnings, \
             patch.object(QMessageBox, "critical") as errors:
            QTest.mouseClick(self.win.trim_apply_btn, Qt.MouseButton.LeftButton)
            self.assertIsNotNone(self.win._trim_thread)
            self.assertFalse(self.win.centralWidget().isEnabled())
            self.win._clear_queue()
            self.assertEqual(len(self.win.refs), 1)
            self._wait_for_trim()
            self.assertFalse(errors.called, errors.call_args)
            questions.assert_not_called()
            notifications.assert_not_called()
            warnings.assert_not_called()
        self.assertEqual(self.win.episode.n_frames, 11)
        self.assertEqual(self.win.refs[0].n_frames, 11)
        self.assertEqual(self.win.frame_spin.maximum(), 10)
        self.assertEqual(self.win.frame, 0)
        self.assertEqual(self.win.annotation.keep_range, (0, 11))
        self.assertFalse(self.win.trim_apply_btn.isEnabled())
        self.assertTrue(self.win.centralWidget().isEnabled())
        self.assertFalse((source.parent / ".kirigami-backups").exists())
        self.assertFalse(list(source.parent.glob(".kirigami-trim-*")))
        self.assertEqual(self.win.save_status.text(), "SOURCE TRIMMED")
        self.assertIn("11 frames retained", self.win.statusBar().currentMessage())
        self.win._load_ref(self.win.refs[0])
        self.assertEqual(self.win.episode.n_frames, 11)
        self.win._seek(10)
        self.assertIsNotNone(self.win.cam_labels["base_0"]._pixmap)

    def test_trim_failure_reopens_original_with_selected_bounds(self):
        source = _write_synthetic(self.root)
        self.win._add_dirs([source])
        self._choose_trim()
        before = contents(source)
        with patch.object(QMessageBox, "critical") as errors, \
             patch("kirigami.gui.main_window.trim_source", side_effect=RuntimeError("Encoding failed")):
            self.win._on_trim_source()
            self._wait_for_trim()
            self.assertTrue(errors.called)
        self.assertEqual(contents(source), before)
        self.assertEqual(self.win.episode.n_frames, 20)
        self.assertEqual(self.win.annotation.keep_range, (5, 16))
        self.assertTrue(self.win.trim_apply_btn.isEnabled())
        self.assertIsNotNone(self.win.videos)

    def test_async_cameras_preview_and_trim_without_confirmation(self):
        source = write_async_episode(self.root)
        self.win._add_dirs([source])
        self.win._seek(10)
        self.assertEqual(self.win.videos.idx, {"base_0": 10, "left_wrist_0": 12, "right_wrist_0": 8})
        self._choose_trim()
        with patch.object(QMessageBox, "question") as questions, \
             patch.object(QMessageBox, "information") as notifications, \
             patch.object(QMessageBox, "critical") as errors:
            QTest.mouseClick(self.win.trim_apply_btn, Qt.MouseButton.LeftButton)
            self._wait_for_trim()
            errors.assert_not_called()
            questions.assert_not_called()
            notifications.assert_not_called()
        self.assertEqual({key: len(ts) for key, ts in self.win.episode.cam_ts.items()},
                         {"base_0": 11, "left_wrist_0": 12, "right_wrist_0": 9})
        self.assertFalse((source.parent / ".kirigami-backups").exists())
        self.win._seek(10)
        self.assertEqual(self.win.videos.idx["base_0"], 10)
        self.assertIsNotNone(self.win.cam_labels["right_wrist_0"]._pixmap)

    def test_trim_with_six_infrared_streams_reloads_rgb_preview(self):
        source, infrared = write_infrared_episode(self.root)
        self.win._add_dirs([source])
        self._choose_trim()
        with patch.object(QMessageBox, "question") as questions, \
             patch.object(QMessageBox, "critical") as errors:
            QTest.mouseClick(self.win.trim_apply_btn, Qt.MouseButton.LeftButton)
            self._wait_for_trim()
            errors.assert_not_called()
            questions.assert_not_called()
        self.assertEqual(self.win.episode.n_frames, 11)
        for name in infrared:
            self.assertTrue((source / "camera" / f"{name}.mkv").is_file())
            self.assertNotIn(f"{name}.mkv", self.win.episode.extra_videos)
        self.win._seek(self.win.episode.n_frames - 1)
        for label in self.win.cam_labels.values():
            self.assertIsNotNone(label._pixmap)

    def test_progress_updates_do_not_run_nested_events(self):
        progress = TaskProgressDialog("Test task", "Working", self.win)
        progress.show()
        QTest.qWait(30)
        called = []
        QTimer.singleShot(0, lambda: called.append(True))
        progress.setLabelText("Almost finished")
        progress.setValue(95)
        self.assertEqual(called, [])
        self.app.processEvents()
        self.assertEqual(called, [True])
        progress.finish()

    def test_trim_joins_worker_before_reloading_and_releasing_objects(self):
        source = _write_synthetic(self.root)
        self.win._add_dirs([source])
        self._choose_trim()
        cleanup_done = threading.Event()
        destroyed = threading.Event()

        def delayed_thread_cleanup():
            # QThread.finished can already be queued for the GUI while the
            # worker thread is still running its completion/destruction code.
            time.sleep(0.08)
            cleanup_done.set()

        reload_ref = self.win._load_ref
        def checked_reload(ref):
            self.assertTrue(cleanup_done.is_set())
            self.assertTrue(destroyed.is_set())
            reload_ref(ref)

        with patch.object(self.win, "_load_ref", side_effect=checked_reload) as reload, \
             patch.object(QMessageBox, "critical") as errors:
            self.win._on_trim_source()
            self.win._trim_worker.destroyed.connect(destroyed.set, Qt.ConnectionType.DirectConnection)
            self.win._trim_thread.finished.connect(delayed_thread_cleanup, Qt.ConnectionType.DirectConnection)
            self._wait_for_trim()
            errors.assert_not_called()
            reload.assert_called_once()
        self.assertIsNone(self.win.trim_progress)
        self.assertIsNone(self.win._trim_worker)

    def test_repeated_trims_release_dialogs_threads_and_workers(self):
        template = _write_synthetic(self.root / "fixture")
        episodes = self.root / "episodes"
        for index in range(6):
            shutil.copytree(template, episodes / f"{index:04d}")
        self.win._add_dirs([episodes])
        with patch.object(QMessageBox, "critical") as errors:
            for index in range(6):
                self.win.list_widget.setCurrentRow(index)
                self._choose_trim()
                self.win._on_trim_source()
                self._wait_for_trim()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                self.app.processEvents()
                self.assertIsNone(self.win.trim_progress)
                self.assertIsNone(self.win._trim_worker)
                self.assertEqual(self.win.findChildren(QThread), [])
                self.assertEqual(self.win.findChildren(QDialog), [])
                self.assertEqual(self.win.episode.n_frames, 11)
            errors.assert_not_called()

    def test_external_trim_recovers_stale_labels_and_reloads(self):
        source = _write_synthetic(self.root)
        self.win._add_dirs([source])
        self.win.annotation.segments[0].subtask = "pick"
        self.win._persist()
        self.win.annotation.segments[0].subtask = "unsaved"
        trim_source(load_episode(source), load_annotation(source), 5, 16)
        with patch.object(QMessageBox, "warning") as warning:
            self.assertFalse(self.win._persist())
            self.assertTrue(warning.called)
        self.assertEqual(self.win.episode.n_frames, 11)
        self.assertEqual(load_annotation(source).segments[0].subtask, "pick")
        recovered = list((source.parent / ".kirigami-backups/stale-labels").glob("*/annotations/slices.json"))
        self.assertEqual(len(recovered), 1)
        self.assertEqual(load_annotation(recovered[0].parent.parent).segments[0].subtask, "unsaved")
