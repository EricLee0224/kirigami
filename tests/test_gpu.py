"""Run against a visible native desktop and its real OpenGL context."""

import os
import time
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from kirigami.app import desktop_startup_error
from kirigami.gui.main_window import MainWindow


def pixels(panel):
    image = panel.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
    return np.frombuffer(image.bits(), dtype=np.uint8).reshape(image.height(), image.width(), 4).copy()


@unittest.skipUnless(os.environ.get("KIRIGAMI_DESKTOP_TESTS") == "1", "enable KIRIGAMI_DESKTOP_TESTS=1 on the host desktop")
class TestGpuViewport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        error = desktop_startup_error()
        if error:
            raise RuntimeError(error)
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.win = MainWindow()
        self.win.show()
        QTest.qWait(150)

    def tearDown(self):
        self.win.close()
        self.app.processEvents()

    def test_native_scene_changes_without_cpu_rasterization_or_mesh_reupload(self):
        panel = self.win.robot_panel
        self.assertTrue(panel.isValid())
        self.assertTrue(panel.ready, panel.error)
        self.assertTrue(panel.renderer)
        before = pixels(panel)
        buffers = {key: gpu.vbo.bufferId() for key, (_, gpu) in panel._meshes.items()}
        self.assertGreater(self.win.robot.triangle_count, 250000)
        with patch.object(self.win.robot, "render_rgb", side_effect=AssertionError("CPU rasterizer used by GUI")):
            q = np.array([0.2, 0.8, 0.7, -0.4, 0, 0.5, 0.08])
            self.win.robot.update(q, q)
            self.win._refresh_robot_view()
            moved = pixels(panel)
            self.assertGreater(np.count_nonzero(before != moved), 1000)
            self.win._orbit_robot(60, 10)
            self.win._zoom_robot(1)
            self.win.material_combo.setCurrentIndex(1)
            colored = pixels(panel)
            self.assertGreater(np.count_nonzero(moved != colored), 1000)
        self.assertEqual(buffers, {key: gpu.vbo.bufferId() for key, (_, gpu) in panel._meshes.items()})
        panel.makeCurrent()
        self.assertEqual(panel.context().functions().glGetError(), 0)
        panel.doneCurrent()

    def test_expand_escape_reopen_and_reload(self):
        QTest.mouseDClick(self.win.robot_panel, Qt.MouseButton.LeftButton)
        QTest.qWait(150)
        detail = self.win._robot_detail
        self.assertTrue(detail.panel.ready, detail.panel.error)
        self.assertGreater(detail.panel.width(), self.win.robot_panel.width())
        self.win.material_combo.setCurrentIndex(1)
        self.assertIs(detail.panel.player, self.win.robot)
        QTest.keyClick(detail, Qt.Key.Key_Escape)
        deadline = time.monotonic() + 2
        while self.win._robot_detail is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertIsNone(self.win._robot_detail)
        self.win._expand_robot_view()
        QTest.qWait(150)
        self.assertTrue(self.win._robot_detail.panel.ready, self.win._robot_detail.panel.error)
        cfg = self.win.robot.config
        self.assertEqual(self.win.robot.load(cfg.urdf, cfg.mesh_dir), "")
        self.win._refresh_robot_view()
        QTest.qWait(100)
        for panel in (self.win.robot_panel, self.win._robot_detail.panel):
            self.assertTrue(panel.ready, panel.error)
            self.assertFalse(panel.error)
