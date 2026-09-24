"""Source trimming overwrites in place without keeping successful-trim backups."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from kirigami.annotation import annotation_from_episode, load_annotation, mark_exported, save_annotation, set_marks
from kirigami.exporter import export_annotation
from kirigami.loader import discover_episodes, load_episode
from kirigami.source_guard import SourceChangedError, episode_lock
from kirigami.trimming import _exchange_directories, _sync_dir, cut_video, trim_source
from kirigami.annotation import atomic_write_text
from tests.test_core import _write_synthetic


def contents(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


class TestSourceTrim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.fixture_path = _write_synthetic(Path(cls.fixture.name))

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "task" / "0007"
        shutil.copytree(self.fixture_path, self.source)
        self.episode = load_episode(self.source)
        self.ann = annotation_from_episode(self.episode)
        set_marks(self.ann, [3, 8, 17])
        for seg, name in zip(self.ann.segments, ["idle", "pick", "place", "idle"]):
            seg.subtask = name
        self.ann.segments[2].discard = True
        self.ann.trim_start, self.ann.trim_end = 5, 16
        mark_exported(self.ann, self.root / "exports")
        save_annotation(self.source, self.ann)

    def _assert_no_trim_artifacts(self):
        self.assertFalse((self.source.parent / ".kirigami-backups").exists())
        self.assertFalse(list(self.source.parent.glob(".kirigami-trim-*")))

    def test_trim_synchronizes_streams_without_backup_and_rebases_labels(self):
        (self.source / "README.txt").write_text("Keep task instructions unchanged\n")
        nested = self.source / "camera/remove_hand/base_0_rgb.mp4"
        nested.parent.mkdir()
        shutil.copy2(self.source / "camera/base_0_rgb_remove_hand.mp4", nested)
        observations = self.source / "observations"
        observations.mkdir()
        values = np.arange(60, dtype=np.float32).reshape(20, 3)
        np.save(observations / "state.npy", values)
        obs_ts = np.arange(1_000_000, 1_000_660, 20)
        np.savez(observations / "state.npz", timestamps=obs_ts, state=np.arange(len(obs_ts) * 2).reshape(-1, 2))
        robot = deepcopy(self.episode.robot_raw)
        robot["left"]["observation"] = {"state": np.arange(120).reshape(60, 2)}
        for key in robot["right"]:
            robot["right"][key] = robot["right"][key][::2]
        with (self.source / "robot/robot_state_dict.pkl").open("wb") as handle:
            pickle.dump(robot, handle)
        manifest_path = self.source / "manifests/camera.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["rec"]["robot_samples"] = {"left": 60, "right": 30}
        manifest_path.write_text(json.dumps(manifest))
        old_task = manifest["rec"]["metadata"]["task"]
        result = trim_source(self.episode, self.ann, 5, 16)
        self.assertEqual(result.source, self.source)
        self.assertEqual(result.n_frames, 11)
        self.assertEqual((result.removed_head, result.removed_tail), (5, 4))
        self.assertFalse(result.warning)
        self._assert_no_trim_artifacts()
        loaded = load_episode(self.source)
        self.assertEqual(loaded.n_frames, 11)
        t0, t1 = self.episode.frame_range_timestamps(5, 16)
        for key in loaded.cam_ts:
            ts = self.episode.cam_ts[key]
            np.testing.assert_array_equal(loaded.cam_ts[key], ts[(ts >= t0) & (ts < t1)])
        np.testing.assert_array_equal(loaded.left_joint_aligned, self.episode.left_joint_aligned[5:16])
        for key in ("left", "right"):
            expected = [ts for ts in robot[key]["timestamps"] if t0 <= ts < t1]
            self.assertEqual(loaded.robot_raw[key]["timestamps"], expected)
        self.assertTrue(all(t0 <= ts < t1 for ts in loaded.action_raw["timestamps"]))
        self.assertEqual(loaded.event_raw["timestamps"], [])
        np.testing.assert_array_equal(np.load(observations / "state.npy"), values[5:16])
        with np.load(observations / "state.npz") as obs:
            np.testing.assert_array_equal(obs["timestamps"], obs_ts[(obs_ts >= t0) & (obs_ts < t1)])
        for video in (self.source / "camera").rglob("*.mp4"):
            cap = cv2.VideoCapture(str(video))
            count = 0
            while cap.grab():
                count += 1
            cap.release()
            self.assertEqual(count, 11, video)
        self.assertTrue(nested.exists())
        self.assertEqual((self.source / "README.txt").read_text(), "Keep task instructions unchanged\n")
        new_manifest = json.loads(manifest_path.read_text())["rec"]
        self.assertEqual(new_manifest["metadata"]["task"], old_task)
        self.assertEqual(new_manifest["robot_samples"], {side: len(loaded.robot_raw[side]["timestamps"]) for side in ("left", "right")})
        self.assertEqual(loaded.camera_meta["base_0_rgb"]["frames"], 11)
        adjusted = load_annotation(self.source)
        self.assertEqual(adjusted.marks, [3])
        self.assertEqual([(s.start_frame, s.end_frame, s.subtask, s.discard) for s in adjusted.segments],
                         [(0, 3, "pick", False), (3, 11, "place", True)])
        self.assertEqual(adjusted.keep_range, (0, 11))
        self.assertFalse(adjusted.exported_at)
        history = json.loads((self.source / "annotations/trim_history.json").read_text())
        self.assertEqual((history[-1]["start_frame"], history[-1]["end_frame"]), (5, 16))
        self.assertNotIn("backup_path", history[-1])
        self.assertEqual([ref.path for ref in discover_episodes(self.source.parent)], [self.source])

    def test_invalid_and_empty_ranges_leave_original_untouched(self):
        before = contents(self.source)
        for start, end in ((0, 0), (-1, 5), (4, 3), (0, 21), (0, 20)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                trim_source(self.episode, self.ann, start, end)
            self.assertEqual(contents(self.source), before)

    def test_one_frame_is_retained_inclusively(self):
        trim_source(self.episode, self.ann, 7, 8)
        result = load_episode(self.source)
        self.assertEqual(result.n_frames, 1)
        self.assertEqual(result.base_ts[0], self.episode.base_ts[7])

    def test_encoding_failure_does_not_touch_source(self):
        before = contents(self.source)
        calls = []
        def fail_second(*args, **kwargs):
            calls.append(args)
            if len(calls) == 2:
                raise RuntimeError("Encoder stopped")
            return cut_video(*args, **kwargs)
        with patch("kirigami.trimming.cut_video", side_effect=fail_second), self.assertRaisesRegex(RuntimeError, "Encoder stopped"):
            trim_source(self.episode, self.ann, 5, 16)
        self.assertEqual(contents(self.source), before)
        self._assert_no_trim_artifacts()

    def test_atomic_exchange_failure_retains_source(self):
        before = contents(self.source)
        with patch("kirigami.trimming._exchange_directories", side_effect=OSError("Unsupported filesystem")):
            with self.assertRaisesRegex(OSError, "Unsupported filesystem"):
                trim_source(self.episode, self.ann, 5, 16)
        self.assertEqual(contents(self.source), before)
        self._assert_no_trim_artifacts()

    def test_interruption_immediately_after_exchange_cleans_up_old_data(self):
        def interrupted(source, prepared):
            _exchange_directories(source, prepared)
            raise KeyboardInterrupt()
        with patch("kirigami.trimming._exchange_directories", side_effect=interrupted), self.assertRaises(KeyboardInterrupt):
            trim_source(self.episode, self.ann, 5, 16)
        self.assertEqual(load_episode(self.source).n_frames, 11)
        self._assert_no_trim_artifacts()

    def test_interruption_before_exchange_preserves_source_and_removes_staging(self):
        before = contents(self.source)
        with patch("kirigami.trimming._exchange_directories", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            trim_source(self.episode, self.ann, 5, 16)
        self.assertEqual(contents(self.source), before)
        self._assert_no_trim_artifacts()

    def test_stale_windows_cannot_overwrite_or_export_changed_source(self):
        trim_source(self.episode, self.ann, 5, 16)
        before = contents(self.source)
        for action in (lambda: save_annotation(self.source, self.ann),
                       lambda: export_annotation(self.episode, self.ann, self.root / "exports"),
                       lambda: trim_source(self.episode, self.ann, 5, 16)):
            with self.assertRaises(SourceChangedError):
                action()
            self.assertEqual(contents(self.source), before)

    def test_source_lock_blocks_trimming_and_label_saves(self):
        before = contents(self.source)
        with episode_lock(self.source, exclusive=True):
            with self.assertRaisesRegex(RuntimeError, "another window"):
                trim_source(self.episode, self.ann, 5, 16)
            with self.assertRaisesRegex(RuntimeError, "another window"):
                save_annotation(self.source, self.ann)
        with episode_lock(self.source):
            with self.assertRaisesRegex(RuntimeError, "another window"):
                trim_source(self.episode, self.ann, 5, 16)
        self.assertEqual(contents(self.source), before)

    def test_external_source_edits_abort_before_exchange(self):
        before = contents(self.source)
        def progress(message, fraction):
            if fraction == 0.95:
                (self.source / "external-note.txt").write_text("External change")
        with self.assertRaisesRegex(RuntimeError, "changed while trimming"):
            trim_source(self.episode, self.ann, 5, 16, progress=progress)
        after = contents(self.source)
        after.pop("external-note.txt")
        self.assertEqual(after, before)

    def test_unsupported_observations_and_links_block_writeback(self):
        unknown = self.source / "observations.hdf5"
        unknown.write_bytes(b"unknown schema")
        before = contents(self.source)
        with self.assertRaisesRegex(ValueError, "Unsupported episode file"):
            trim_source(self.episode, self.ann, 5, 16)
        self.assertEqual(contents(self.source), before)
        unknown.unlink()
        (self.source / "link").symlink_to(self.fixture_path, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            trim_source(self.episode, self.ann, 5, 16)

    def test_repeated_trims_keep_history_without_accumulating_backups(self):
        trim_source(self.episode, self.ann, 5, 16)
        self._assert_no_trim_artifacts()
        second_episode = load_episode(self.source)
        second_ann = load_annotation(self.source)
        trim_source(second_episode, second_ann, 1, 7)
        self._assert_no_trim_artifacts()
        self.assertEqual(load_episode(self.source).n_frames, 6)
        history = json.loads((self.source / "annotations/trim_history.json").read_text())
        self.assertEqual(len(history), 2)
        self.assertEqual([(item["start_frame"], item["end_frame"]) for item in history], [(5, 16), (1, 7)])
        self.assertTrue(all("backup_path" not in item for item in history))

    def test_final_sync_failure_reports_applied_trim_and_cleans_temporary_data(self):
        calls = 0
        def fail_after_exchange(path):
            nonlocal calls
            if path == self.source.parent:
                calls += 1
                if calls == 2:
                    raise OSError("Directory sync error")
            return _sync_dir(path)
        with patch("kirigami.trimming._sync_dir", side_effect=fail_after_exchange):
            result = trim_source(self.episode, self.ann, 5, 16)
        self.assertIn("Trim was applied", result.warning)
        self.assertIn("Directory sync error", result.warning)
        self.assertEqual(load_episode(self.source).n_frames, 11)
        self._assert_no_trim_artifacts()

    def test_cleanup_failure_reports_leftover_temporary_data(self):
        before = contents(self.source)
        with patch("kirigami.trimming.shutil.rmtree", side_effect=OSError("Cleanup failed")):
            result = trim_source(self.episode, self.ann, 5, 16)
        self.assertEqual(load_episode(self.source).n_frames, 11)
        self.assertIn("Trim was applied", result.warning)
        self.assertIn("Cleanup failed", result.warning)
        temporary = list(self.source.parent.glob(".kirigami-trim-*"))
        self.assertEqual(len(temporary), 1)
        self.assertIn(str(temporary[0]), result.warning)
        self.assertEqual(contents(temporary[0] / "episode"), before)
        self.assertFalse((self.source.parent / ".kirigami-backups").exists())

    def test_staging_journal_failure_leaves_original_untouched(self):
        before = contents(self.source)
        def fail_journal(path, text):
            if path.name == "transaction.json":
                raise OSError("Journal disk error")
            return atomic_write_text(path, text)
        with patch("kirigami.trimming.atomic_write_text", side_effect=fail_journal), self.assertRaisesRegex(OSError, "Journal disk error"):
            trim_source(self.episode, self.ann, 5, 16)
        self.assertEqual(contents(self.source), before)
        self._assert_no_trim_artifacts()

    def test_mismatched_observation_arrays_are_rejected(self):
        obs = self.source / "observations"
        obs.mkdir()
        np.save(obs / "state.npy", np.zeros((19, 14)))
        before = contents(self.source)
        with self.assertRaisesRegex(ValueError, "Sample count mismatch"):
            trim_source(self.episode, self.ann, 5, 16)
        self.assertEqual(contents(self.source), before)
