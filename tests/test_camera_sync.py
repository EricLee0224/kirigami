"""Independent camera clocks, dropped frames and per-stream counts."""

import json
from pathlib import Path
import pickle
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from kirigami.annotation import annotation_from_episode, load_annotation, set_marks
from kirigami.camera import camera_frame_ranges
from kirigami.exporter import export_annotation
from kirigami.loader import load_episode
from kirigami.trimming import trim_source
from kirigami.video import VideoBank
from tests.test_core import _write_synthetic
from tests.test_trimming import contents


def frame_means(path):
    cap = cv2.VideoCapture(str(path))
    values = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        values.append(float(frame.mean()))
    cap.release()
    return values


def write_async_episode(root):
    source = _write_synthetic(root)
    base = np.arange(20, dtype=np.int64) * 33 + 1_000_000
    timestamps = {"base_0": base.tolist(),
                  "left_wrist_0": (np.arange(23) * 30 + base[0] - 17).tolist(),
                  "right_wrist_0": (np.delete(base, [3, 7, 13]) + 11).tolist()}
    with (source / "camera/timestamp.pkl").open("wb") as handle:
        pickle.dump(timestamps, handle)
    metadata, manifests = {}, {}
    counts = {f"camera.{key}_rgb": len(ts) for key, ts in timestamps.items()}
    for key, ts in timestamps.items():
        video = source / "camera" / f"{key}_rgb.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
        if not writer.isOpened():
            raise RuntimeError("Cannot create synthetic camera video")
        for index in range(len(ts)):
            writer.write(np.full((48, 64, 3), 20 + index * 8, dtype=np.uint8))
        writer.release()
        metadata[f"{key}_rgb"] = {"frames": len(ts), "encoder": "libx264", "pixel_format": "yuv420p"}
        manifests[key] = {"sensor_name": key, "camera_frames": {f"{key}_rgb": len(ts)},
                          "camera_timestamp_samples": len(ts), "complete_sensor_bundles": len(ts),
                          "counts": {**counts, "items": sum(counts.values())},
                          "metadata": {"task": {"prompt": ["demo"], "subtasks": []}}}
    extra = source / "camera/derived/left_wrist_0_rgb_masked.mp4"
    extra.parent.mkdir()
    shutil.copy2(source / "camera/left_wrist_0_rgb.mp4", extra)
    metadata[extra.stem] = dict(metadata["left_wrist_0_rgb"])
    (source / "camera/metadata.json").write_text(json.dumps(metadata))
    (source / "manifests/camera.json").write_text(json.dumps(manifests))
    preview = source / "robot/state_joint_vis.png"
    cv2.imwrite(str(preview), np.zeros((8, 8, 3), dtype=np.uint8))
    (source / "manifests/robot.json").write_text(json.dumps({"robot": {"plot_outputs": [str(preview)], "plot_error": "old plot error"}}))
    return source


class TestCameraSynchronization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.fixture_path = write_async_episode(Path(cls.fixture.name))

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "task/0007"
        shutil.copytree(self.fixture_path, self.source)
        self.episode = load_episode(self.source)
        self.ann = annotation_from_episode(self.episode)

    def _check_counts(self, path, expected):
        ep = load_episode(path)
        metadata = ep.camera_meta
        for key, ts in expected.items():
            np.testing.assert_array_equal(ep.cam_ts[key], ts)
            self.assertEqual(len(frame_means(ep.videos[key])), len(ts))
            self.assertEqual(metadata[f"{key}_rgb"]["frames"], len(ts))
        self.assertEqual(metadata["left_wrist_0_rgb_masked"]["frames"], len(expected["left_wrist_0"]))
        for key, block in ep.manifests["camera.json"].items():
            self.assertEqual(block["camera_timestamp_samples"], len(expected[key]))
            self.assertEqual(block["complete_sensor_bundles"], len(expected[key]))
            self.assertEqual(block["camera_frames"][f"{key}_rgb"], len(expected[key]))
            for camera, ts in expected.items():
                self.assertEqual(block["counts"][f"camera.{camera}_rgb"], len(ts))
            self.assertEqual(block["counts"]["items"], sum(map(len, expected.values())))

    def test_trim_and_slice_keep_real_frames_and_per_camera_counts(self):
        originals = {key: frame_means(path) for key, path in self.episode.videos.items()}
        # Overview [5, 16) covers these different ranges on the camera clocks.
        ranges = {"base_0": (5, 16), "left_wrist_0": (7, 19), "right_wrist_0": (4, 13)}
        expected = {key: ts[a:b] for key, ts in self.episode.cam_ts.items() for a, b in [ranges[key]]}
        trim_source(self.episode, self.ann, 5, 16)
        self._check_counts(self.source, expected)
        self.assertFalse((self.source / "robot/state_joint_vis.png").exists())
        robot_manifest = load_episode(self.source).manifests["robot.json"]["robot"]
        self.assertEqual(robot_manifest["plot_outputs"], [])
        self.assertNotIn("plot_error", robot_manifest)
        for key, (a, b) in ranges.items():
            np.testing.assert_allclose(frame_means(self.source / "camera" / f"{key}_rgb.mp4"), originals[key][a:b], atol=3)
        np.testing.assert_allclose(frame_means(self.source / "camera/derived/left_wrist_0_rgb_masked.mp4"),
                                   originals["left_wrist_0"][7:19], atol=3)
        self.assertFalse(list(self.source.parent.glob(".kirigami-trim-*")))
        self.assertFalse((self.source.parent / ".kirigami-backups").exists())

        trimmed = load_episode(self.source)
        ann = load_annotation(self.source)
        set_marks(ann, [5])
        ann.segments[0].subtask, ann.segments[1].subtask = "pick", "place"
        outputs = export_annotation(trimmed, ann, self.root / "out")
        combined = {key: [] for key in expected}
        for output, segment in zip(outputs, ann.segments):
            t0, t1 = trimmed.frame_range_timestamps(segment.start_frame, segment.end_frame)
            selected = {key: ts[(ts >= t0) & (ts < t1)] for key, ts in expected.items()}
            self._check_counts(output, selected)
            self.assertEqual(len(frame_means(output / "camera/left_wrist_0_rgb_masked.mp4")), len(selected["left_wrist_0"]))
            for key, ts in load_episode(output).cam_ts.items():
                combined[key].extend(ts.tolist())
        for key in expected:
            np.testing.assert_array_equal(combined[key], expected[key])

    def test_preview_uses_nearest_camera_timestamp(self):
        with VideoBank(self.episode.videos, self.episode.cam_ts) as bank:
            bank.read(10)
            self.assertEqual(bank.idx, {"base_0": 10, "left_wrist_0": 12, "right_wrist_0": 8})
            bank.read(19)
            self.assertEqual(bank.idx, {"base_0": 19, "left_wrist_0": 21, "right_wrist_0": 16})

    def test_equal_counts_with_clock_offsets_still_use_timestamps(self):
        clocks = {"base_0": [100, 200, 300, 400], "left_wrist_0": [150, 250, 350, 450],
                  "right_wrist_0": [95, 195, 295, 395]}
        self.assertEqual(camera_frame_ranges(clocks, 1, 3),
                         {"base_0": (1, 3), "left_wrist_0": (1, 3), "right_wrist_0": (2, 4)})

    def test_empty_camera_window_does_not_overwrite_source(self):
        before = contents(self.source)
        with self.assertRaisesRegex(ValueError, "right_wrist_0 has no frames"):
            trim_source(self.episode, self.ann, 7, 8)
        self.assertEqual(contents(self.source), before)

    def test_own_video_timestamp_mismatch_is_still_rejected(self):
        self.episode.cam_ts["left_wrist_0"] = self.episode.cam_ts["left_wrist_0"][:-1]
        self.ann.segments[0].subtask = "pick"
        with self.assertRaisesRegex(ValueError, "left_wrist_0.*23 frames.*22"):
            export_annotation(self.episode, self.ann, self.root / "out")
        self.assertFalse((self.root / "out").exists())
