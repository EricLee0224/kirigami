"""Protect CAD surface fidelity and gripper kinematics independently of a display."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from kirigami.robot3d import Robot3DPlayer
from kirigami.urdf_fk import UrdfArm
from kirigami.yam_ultra import YAM_ULTRA_V2_URDF


@unittest.skipUnless(YAM_ULTRA_V2_URDF.exists(), "YAM Ultra 2 URDF missing")
class TestOriginalSurfaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.player = Robot3DPlayer()
        if not cls.player.available:
            raise RuntimeError(cls.player.status)

    def test_all_source_triangles_and_fingers_are_preserved(self):
        meshes = self.player._left.arm.meshes
        self.assertEqual({vis.link for vis, _ in meshes},
                         {"base", "link1", "link2", "link3", "link4", "link5", "gripper", "tip_left", "tip_right"})
        for vis, geometry in meshes:
            with self.subTest(link=vis.link):
                source = trimesh.load(str(vis.mesh_path), force="mesh")
                np.testing.assert_array_equal(geometry.faces, source.faces)
                np.testing.assert_array_equal(geometry.vertices, source.vertices)
                self.assertEqual(len(geometry.vertex_data), len(source.faces) * 3)
                # Normal smoothing may reorder faces; vertex coordinates must remain unchanged.
                np.testing.assert_allclose(
                    np.sort(geometry.vertex_data[:, :3], axis=0),
                    np.sort(source.triangles.reshape(-1, 3), axis=0), atol=1e-7)
                normals = geometry.vertex_data[:, 3:]
                self.assertTrue(np.isfinite(normals).all())
                np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)

    def test_arms_share_geometry_and_fingers_follow_stroke(self):
        left, right = self.player._left.arm, self.player._right.arm
        for (_, a), (_, b) in zip(left.meshes, right.meshes):
            self.assertIs(a, b)
        q = np.zeros(7)
        self.player.update(q, q)
        closed = {name: left.link_xyz(name).copy() for name in ("tip_left", "tip_right", "gripper")}

        def finger_centers():
            centers = []
            for vis, geometry in left.meshes:
                if vis.link in {"tip_left", "tip_right"}:
                    transform = left.transforms[vis.link] @ vis.origin
                    centers.append(transform[:3, :3] @ geometry.vertices.mean(axis=0) + transform[:3, 3])
            return centers

        # CAD link origins are offset from the fingers, so compare actual mesh centers.
        a, b = finger_centers()
        closed_spacing = np.linalg.norm(a - b)
        q[6] = 0.08
        self.player.update(q, q)
        np.testing.assert_allclose(left.link_xyz("gripper"), closed["gripper"])
        for name in ("tip_left", "tip_right"):
            self.assertGreater(np.linalg.norm(left.link_xyz(name) - closed[name]), 0.01)
        a, b = finger_centers()
        self.assertGreater(np.linalg.norm(a - b), closed_spacing + 0.07)


class TestMeshLoading(unittest.TestCase):
    def test_visual_scale_and_missing_parts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mesh_path = root / "box.stl"
            trimesh.creation.box().export(mesh_path)
            urdf = root / "robot.urdf"
            urdf.write_text('<robot name="test"><link name="base"><visual>'
                            '<origin xyz="1 2 3"/><geometry><mesh filename="box.stl" scale="2 3 4"/>'
                            '</geometry></visual></link></robot>')
            arm = UrdfArm(urdf, root, (10, 0, 0))
            vertices, _, _ = arm.world_meshes()[0]
            np.testing.assert_allclose(vertices.min(axis=0), [10, 0.5, 1])
            np.testing.assert_allclose(vertices.max(axis=0), [12, 3.5, 5])
            mesh_path.unlink()
            with self.assertRaises(FileNotFoundError):
                UrdfArm(urdf, root, (0, 0, 0))
