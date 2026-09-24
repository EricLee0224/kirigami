"""Regression checks for data loss, invalid exports and annotation durability."""

import hashlib
import fcntl
import json
import shutil
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from kirigami.annotation import (
    add_mark, annotation_from_episode, load_annotation, mark_exported,
    save_annotation, set_marks,
)
from kirigami.exporter import cut_video, export_annotation
from kirigami.loader import load_episode
from tests.test_core import _write_synthetic, _write_tiny_mp4


def snapshot(paths):
    return {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for root in paths for p in root.rglob("*") if p.is_file()
    }


class TestExportSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.episode = load_episode(_write_synthetic(Path(cls.fixture.name)))

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "out"
        self.ann = annotation_from_episode(self.episode)
        set_marks(self.ann, [10])
        self.ann.segments[0].subtask = "pick"
        self.ann.segments[1].subtask = "place"

    def test_different_tasks_never_overwrite_same_episode_id(self):
        first = export_annotation(self.episode, self.ann, self.out)
        before = snapshot(first)
        other_path = self.root / "other_task" / self.episode.ref.episode_id
        shutil.copytree(self.episode.ref.path, other_path)
        other = load_episode(other_path)
        other_ann = annotation_from_episode(other, deepcopy(self.ann))
        second = export_annotation(other, other_ann, self.out)
        self.assertFalse(set(first) & set(second))
        self.assertEqual(snapshot(first), before)

    def test_failed_encoding_preserves_every_previous_segment(self):
        first = export_annotation(self.episode, self.ann, self.out)
        before = snapshot(first)
        from kirigami.exporter import export_segment

        def fail_second(episode, segment, *args, **kwargs):
            if segment.start_frame == 10:
                raise RuntimeError("encoder failed")
            return export_segment(episode, segment, *args, **kwargs)

        with patch("kirigami.exporter.export_segment", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                export_annotation(self.episode, self.ann, self.out)
        self.assertEqual(snapshot(first), before)
        self.assertFalse(list(self.out.glob(".kirigami-export-*")))

    def test_publication_failure_rolls_back_old_outputs(self):
        first = export_annotation(self.episode, self.ann, self.out)
        before = snapshot(first)
        original_rename = Path.rename

        def fail_second_install(source, target):
            if source.parent.name == "new" and source.name == "1":
                raise OSError("simulated rename failure")
            return original_rename(source, target)

        with patch.object(Path, "rename", fail_second_install):
            with self.assertRaisesRegex(OSError, "rename failure"):
                export_annotation(self.episode, self.ann, self.out)
        self.assertEqual(snapshot(first), before)

    def test_replacement_removes_old_labels_after_success(self):
        old = export_annotation(self.episode, self.ann, self.out)
        self.ann.segments[0].subtask = "renamed"
        new = export_annotation(self.episode, self.ann, self.out)
        self.assertFalse(old[0].exists())
        for path in new:
            self.assertEqual(load_episode(path).n_frames, 10)
            manifest = json.loads((path / "manifests/camera.json").read_text())
            self.assertNotIn(".kirigami-export-", json.dumps(manifest))

    def test_unsafe_names_are_rejected_before_changing_outputs(self):
        old = export_annotation(self.episode, self.ann, self.out)
        before = snapshot(old)
        for name in ("../escape", str(self.root / "absolute"), "a/b", "a\\b", ".", "..", "", "bad\x00name"):
            with self.subTest(name=name):
                self.ann.segments[0].subtask = name
                with self.assertRaises(ValueError):
                    export_annotation(self.episode, self.ann, self.out)
                self.assertEqual(snapshot(old), before)

    def test_symlink_output_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.out.mkdir()
        (self.out / "pick").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            export_annotation(self.episode, self.ann, self.out)
        self.assertFalse(list(outside.iterdir()))

    def test_short_video_cannot_replace_valid_output(self):
        old = export_annotation(self.episode, self.ann, self.out)
        before = snapshot(old)
        broken = deepcopy(self.episode)
        short = self.root / "short.mp4"
        _write_tiny_mp4(short, n_frames=3)
        broken.videos["left_wrist_0"] = short
        with self.assertRaisesRegex(ValueError, "video/timestamp count mismatch"):
            export_annotation(broken, self.ann, self.out)
        self.assertEqual(snapshot(old), before)

    def test_missing_camera_and_timestamp_mismatch_are_rejected(self):
        broken = deepcopy(self.episode)
        broken.videos.pop("left_wrist_0")
        with self.assertRaisesRegex(ValueError, "Missing required camera"):
            export_annotation(broken, self.ann, self.out)
        broken = deepcopy(self.episode)
        broken.cam_ts["right_wrist_0"] = broken.cam_ts["right_wrist_0"][:-1]
        with self.assertRaisesRegex(ValueError, "timestamp count mismatch"):
            export_annotation(broken, self.ann, self.out)

    def test_no_replace_refuses_existing_output(self):
        old = export_annotation(self.episode, self.ann, self.out)
        before = snapshot(old)
        with self.assertRaises(FileExistsError):
            export_annotation(self.episode, self.ann, self.out, replace_previous=False)
        self.assertEqual(snapshot(old), before)

    def test_concurrent_export_is_rejected(self):
        self.out.mkdir()
        with open(self.out / ".kirigami-export.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "Another export"):
                export_annotation(self.episode, self.ann, self.out)

    def test_trim_preserves_requested_first_and_last_frame(self):
        source = self.root / "ramp.mp4"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        for i in range(20):
            writer.write(np.full((48, 64, 3), 20 + i * 10, dtype=np.uint8))
        writer.release()
        output = self.root / "cut.mp4"
        cut_video(source, output, 5, 12, {"encoder": "libx264", "pixel_format": "yuv420p"})

        def means(path):
            cap = cv2.VideoCapture(str(path))
            result = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result.append(float(frame.mean()))
            cap.release()
            return result

        expected = means(source)[5:12]
        actual = means(output)
        self.assertEqual(len(actual), 7)
        np.testing.assert_allclose(actual, expected, atol=3)

    def test_edits_invalidate_export_status(self):
        source = self.root / "source"
        for edit in (
            lambda ann: add_mark(ann, 5),
            lambda ann: setattr(ann.segments[0], "subtask", "new_name"),
            lambda ann: setattr(ann.segments[0], "discard", True),
        ):
            ann = deepcopy(self.ann)
            mark_exported(ann, self.out)
            save_annotation(source, ann)
            edit(ann)
            save_annotation(source, ann)
            self.assertFalse(load_annotation(source).exported_at)

    def test_failed_annotation_replace_preserves_previous_json(self):
        source = self.root / "source"
        path = save_annotation(source, self.ann)
        before = path.read_bytes()
        add_mark(self.ann, 5)
        with patch("kirigami.annotation.os.replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                save_annotation(source, self.ann)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(len(list(path.parent.iterdir())), 1)
