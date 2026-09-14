#!/usr/bin/env python
from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kirigami.annotation import (
    add_mark,
    annotation_from_episode,
    load_annotation,
    load_subtask_presets,
    rebuild_segments,
    save_annotation,
    save_subtask_presets,
    set_marks,
)
from kirigami.exporter import (
    cut_video,
    default_output_root,
    export_annotation,
    remove_previous_exports,
    slice_camera_timestamps,
    slice_keyed_lists,
    slice_robot,
)
from kirigami.loader import discover_episodes, load_episode, nearest_indices
from kirigami.robot3d import Robot3DConfig, Robot3DPlayer
from kirigami.yam_ultra import YAM_ULTRA_V2_URDF, gripper_to_fingers, q7_to_cfg

REAL_EP = Path(os.environ.get("KIRIGAMI_TEST_EPISODE", "/data_storage/yzc/yh/data/romoya-demo/romoya-egg-stage2-0912_100/0000"))
REAL_TASK = REAL_EP.parent


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise unittest.SkipTest("ffmpeg not on PATH")
    return exe


def _write_tiny_mp4(path: Path, n_frames: int = 20, color: str = "red") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s=64x48:r=10:d=4",
        "-frames:v",
        str(n_frames),
        "-c:v",
        "libx265",
        "-pix_fmt",
        "yuv444p",
        "-tag:v",
        "hvc1",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)


def _write_synthetic(root: Path, n_cam: int = 20) -> Path:
    ep = root / "synth_task" / "0007"
    cam_dir = ep / "camera"
    cam_dir.mkdir(parents=True)
    (ep / "robot").mkdir()
    (ep / "action").mkdir()
    (ep / "event").mkdir()
    (ep / "manifests").mkdir()

    base = np.arange(n_cam, dtype=np.int64) * 33 + 1_000_000
    cam_ts = {
        "base_0": [int(v) for v in base],
        "left_wrist_0": [int(v) - 2 for v in base],
        "right_wrist_0": [int(v) + 2 for v in base],
    }
    with open(cam_dir / "timestamp.pkl", "wb") as f:
        pickle.dump(cam_ts, f)
    (cam_dir / "metadata.json").write_text(
        json.dumps(
            {
                "base_0_rgb": {
                    "frames": n_cam,
                    "encoder": "libx265",
                    "pixel_format": "yuv444p",
                    "quality": {"mode": "crf", "value": 14},
                    "video": "camera/base_0_rgb.mp4",
                }
            },
            indent=2,
        )
    )
    _write_tiny_mp4(cam_dir / "base_0_rgb.mp4", n_cam, "red")
    _write_tiny_mp4(cam_dir / "left_wrist_0_rgb.mp4", n_cam, "green")
    _write_tiny_mp4(cam_dir / "right_wrist_0_rgb.mp4", n_cam, "blue")
    _write_tiny_mp4(cam_dir / "base_0_rgb_remove_hand.mp4", n_cam, "yellow")

    n_robot = n_cam * 3
    robot_ts = [int(base[0] + i * 11) for i in range(n_robot)]
    robot = {
        "left": {
            "joint": [np.full(7, i * 0.01, dtype=np.float32) for i in range(n_robot)],
            "vel": [np.zeros(7, dtype=np.float32) for _ in range(n_robot)],
            "effort": [np.zeros(7, dtype=np.float32) for _ in range(n_robot)],
            "eef": [None] * n_robot,
            "timestamps": robot_ts,
        },
        "right": {
            "joint": [np.full(7, -i * 0.01, dtype=np.float32) for i in range(n_robot)],
            "vel": [np.zeros(7, dtype=np.float32) for _ in range(n_robot)],
            "effort": [np.zeros(7, dtype=np.float32) for _ in range(n_robot)],
            "eef": [None] * n_robot,
            "timestamps": robot_ts,
        },
    }
    with open(ep / "robot" / "robot_state_dict.pkl", "wb") as f:
        pickle.dump(robot, f)

    n_act = n_cam * 2
    act_ts = [int(base[0] + i * 16) for i in range(n_act)]
    action = {
        "actions": [np.zeros(14, dtype=np.float32) for _ in range(n_act)],
        "action_space": ["abs_qpos"] * n_act,
        "hz": [120.0] * n_act,
        "timestamps": act_ts,
        "metadata": [{}] * n_act,
    }
    with open(ep / "action" / "executed_action_dict.pkl", "wb") as f:
        pickle.dump(action, f)
    with open(ep / "event" / "event_dict.pkl", "wb") as f:
        pickle.dump(
            {
                "data": [True, True],
                "timestamps": [int(base[0]), int(base[-1] + 33)],
                "names": ["record_start", "record_stop"],
                "metadata": [{}, {}],
            },
            f,
        )
    (ep / "manifests" / "camera.json").write_text(
        json.dumps(
            {
                "rec": {
                    "counts": {
                        "action.executed": n_act,
                        "camera.base_0_rgb": n_cam,
                        "camera.left_wrist_0_rgb": n_cam,
                        "camera.right_wrist_0_rgb": n_cam,
                        "event.workflow": 2,
                        "items": 0,
                        "robot.robot_state": n_robot,
                    },
                    "metadata": {"task": {"id": "synth", "prompt": ["default"], "subtasks": []}},
                    "schema": "prometheus_raw_episode_v1",
                }
            },
            indent=2,
        )
    )
    return ep


class TestNearest(unittest.TestCase):
    def test_nearest_edges(self):
        src = np.array([10, 20, 30], dtype=np.int64)
        tgt = np.array([0, 21, 99], dtype=np.int64)
        idx = nearest_indices(src, tgt)
        self.assertEqual(list(idx), [0, 1, 2])


class TestAnnotation(unittest.TestCase):
    def test_rebuild_preserves_labels(self):
        segs = rebuild_segments([10, 20], 30)
        segs[0].subtask = "a"
        segs[1].discard = True
        segs2 = rebuild_segments([10, 20], 30, segs)
        self.assertEqual(segs2[0].subtask, "a")
        self.assertTrue(segs2[1].discard)
        self.assertEqual([(s.start_frame, s.end_frame) for s in segs2], [(0, 10), (10, 20), (20, 30)])

    def test_presets_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "subtasks.yaml"
            save_subtask_presets(path, ["pick", "place"])
            self.assertEqual(load_subtask_presets(path), ["pick", "place"])


class TestRealLoader(unittest.TestCase):
    @unittest.skipUnless(REAL_EP.exists(), "real episode missing")
    def test_discover_and_load(self):
        refs = discover_episodes(REAL_TASK)
        self.assertGreater(len(refs), 10)
        ep = load_episode(REAL_EP)
        self.assertGreater(ep.n_frames, 100)
        self.assertEqual(ep.state_aligned.shape[0], ep.n_frames)
        self.assertEqual(ep.state_aligned.shape[1], 14)
        self.assertEqual(ep.left_joint_aligned.shape[1], 7)
        t0, t1 = ep.frame_range_timestamps(10, 40)
        self.assertLess(t0, t1)
        robot = slice_robot(ep.robot_raw, t0, t1)
        self.assertGreater(len(robot["left"]["timestamps"]), 0)
        self.assertEqual(len(robot["left"]["joint"]), len(robot["left"]["timestamps"]))
        action = slice_keyed_lists(ep.action_raw, t0, t1)
        self.assertGreater(len(action["timestamps"]), 0)
        cam = slice_camera_timestamps(ep.cam_ts, 10, 40)
        self.assertEqual(len(cam["base_0"]), 30)


class TestExportSynthetic(unittest.TestCase):
    def test_cut_and_export(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ep_dir = _write_synthetic(root, n_cam=20)
            episode = load_episode(ep_dir)
            self.assertEqual(episode.n_frames, 20)
            self.assertIn("base_0_rgb_remove_hand.mp4", episode.extra_videos)

            out_vid = root / "cut.mp4"
            cut_video(episode.videos["base_0"], out_vid, 5, 12, episode.camera_meta.get("base_0_rgb"))
            import cv2

            cap = cv2.VideoCapture(str(out_vid))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.assertEqual(n, 7)

            ann = annotation_from_episode(episode)
            set_marks(ann, [5, 12])
            add_mark(ann, 5)  # duplicate ignored
            ann.segments[0].subtask = "reset"
            ann.segments[0].discard = True
            ann.segments[1].subtask = "pick_up_egg"
            ann.segments[2].subtask = "place_egg"
            save_annotation(ep_dir, ann)
            loaded = load_annotation(ep_dir)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.marks, [5, 12])

            out_root = default_output_root(ep_dir.parent)
            written = export_annotation(episode, ann, out_root)
            self.assertEqual(len(written), 2)
            self.assertTrue((written[0] / "camera" / "timestamp.pkl").exists())
            self.assertTrue((written[0] / "camera" / "base_0_rgb_remove_hand.mp4").exists())
            self.assertTrue((written[0] / "slice_meta.json").exists())
            with open(written[0] / "camera" / "timestamp.pkl", "rb") as f:
                ts = pickle.load(f)
            self.assertEqual(len(ts["base_0"]), 7)
            man = json.loads((written[0] / "manifests" / "camera.json").read_text())
            self.assertEqual(man["rec"]["metadata"]["task"]["subtasks"], ["pick_up_egg"])
            sliced = load_episode(written[0])
            self.assertEqual(sliced.n_frames, 7)

            n_removed = remove_previous_exports(out_root, ep_dir)
            self.assertEqual(n_removed, 2)
            self.assertFalse(written[0].exists())


class TestRobot3D(unittest.TestCase):
    def test_empty_config(self):
        player = Robot3DPlayer(Robot3DConfig(urdf=None))
        self.assertFalse(player.available)
        player.update(np.zeros(7), np.zeros(7))
        self.assertIsNone(player.render_rgb())

    def test_gripper_map(self):
        self.assertAlmostEqual(gripper_to_fingers(0.0), 0.0)
        self.assertLess(gripper_to_fingers(0.096), 0.0)
        cfg = q7_to_cfg([0, 0, 0, 0, 0, 0, 0.048])
        self.assertIn("joint7", cfg)
        self.assertEqual(cfg["joint7"], cfg["joint8"])

    @unittest.skipUnless(YAM_ULTRA_V2_URDF.exists(), "YAM Ultra 2 URDF missing")
    def test_load_yam_ultra(self):
        player = Robot3DPlayer()
        self.assertTrue(player.available, player.status)
        q = np.array([0.2, 0.8, 0.7, -0.4, 0.0, 0.5, 0.04], dtype=np.float32)
        player.update(q, q)
        left, right = player.ee_xyz()
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertGreater(abs(float(left[1] - right[1])), 0.3)
        rgb = player.render_rgb(320, 240)
        self.assertIsNotNone(rgb)
        self.assertEqual(rgb.shape, (240, 320, 3))


if __name__ == "__main__":
    unittest.main()
