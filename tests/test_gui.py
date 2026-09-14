"""Desktop regressions. Run on the host's X11/Wayland session, not offscreen."""

import os
import tempfile
import time
import unittest
from pathlib import Path
from dataclasses import asdict
from unittest.mock import patch

import yaml

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from kirigami.annotation import Annotation, Segment, load_subtask_presets, mark_exported
from kirigami.app import desktop_startup_error
from kirigami.gui.main_window import MainWindow
from kirigami.robot3d import Robot3DConfig
from kirigami.annotation import load_annotation
from kirigami.loader import load_episode
from kirigami.trimming import trim_source
from tests.test_core import _write_synthetic
from tests.test_trimming import contents


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

    def test_trim_source_review_cancel_apply_and_reopen(self):
        source = _write_synthetic(self.root)
        self.win._add_dirs([source])
        self._choose_trim()
        self.assertEqual(load_annotation(source).keep_range, (5, 16))
        before = contents(source)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Cancel):
            self.win._on_trim_source()
        self.assertEqual(contents(source), before)
        self.assertIsNone(self.win._trim_thread)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes), \
             patch.object(QMessageBox, "information"), patch.object(QMessageBox, "critical") as errors:
            self.win._on_trim_source()
            self.assertIsNotNone(self.win._trim_thread)
            self.assertFalse(self.win.centralWidget().isEnabled())
            self.win._clear_queue()
            self.assertEqual(len(self.win.refs), 1)
            self._wait_for_trim()
            self.assertFalse(errors.called, errors.call_args)
        self.assertEqual(self.win.episode.n_frames, 11)
        self.assertEqual(self.win.refs[0].n_frames, 11)
        self.assertEqual(self.win.frame_spin.maximum(), 10)
        self.assertEqual(self.win.frame, 0)
        self.assertEqual(self.win.annotation.keep_range, (0, 11))
        self.assertFalse(self.win.trim_apply_btn.isEnabled())
        self.assertTrue(self.win.centralWidget().isEnabled())
        self.assertEqual(contents(self.win._trim_result.backup), before)
        self.win._load_ref(self.win.refs[0])
        self.assertEqual(self.win.episode.n_frames, 11)
        self.win._seek(10)
        self.assertIsNotNone(self.win.cam_labels["base_0"]._pixmap)

    def test_trim_failure_reopens_original_with_selected_bounds(self):
        source = _write_synthetic(self.root)
        self.win._add_dirs([source])
        self._choose_trim()
        before = contents(source)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes), \
             patch.object(QMessageBox, "critical") as errors, \
             patch("kirigami.gui.main_window.trim_source", side_effect=RuntimeError("Encoding failed")):
            self.win._on_trim_source()
            self._wait_for_trim()
            self.assertTrue(errors.called)
        self.assertEqual(contents(source), before)
        self.assertEqual(self.win.episode.n_frames, 20)
        self.assertEqual(self.win.annotation.keep_range, (5, 16))
        self.assertTrue(self.win.trim_apply_btn.isEnabled())
        self.assertIsNotNone(self.win.videos)

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
