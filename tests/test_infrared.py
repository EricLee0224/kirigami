"""IR files stay untouched during Trim and are omitted from training slices."""

import json
from pathlib import Path
import pickle
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from kirigami.annotation import annotation_from_episode, load_annotation, set_marks
from kirigami.camera import IR_TIMESTAMP_REL, camera_frame_ranges
from kirigami.exporter import cut_video, export_annotation, validate_camera_video
from kirigami.loader import load_episode
from kirigami.trimming import trim_source
from tests.test_camera_sync import write_async_episode
from tests.test_core import _ffmpeg
from tests.test_trimming import contents


def write_ir_video(path, frames, pixel_format="gray", timestamp_scale=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:]
    proc = subprocess.run([
        _ffmpeg(), "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", pixel_format,
        "-video_size", f"{width}x{height}", "-framerate", "30", "-i", "pipe:0",
        "-vf", f"setpts={timestamp_scale}*PTS", "-c:v", "ffv1", "-level", "3",
        "-pix_fmt", pixel_format, "-fps_mode", "passthrough", str(path),
    ], input=frames.tobytes(), capture_output=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode())


def read_ir_video(path):
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(path),
    ], capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    pixel_format = stream["pix_fmt"]
    dtype = {"gray": np.uint8, "gray16le": np.dtype("<u2")}[pixel_format]
    decoded = subprocess.run([
        _ffmpeg(), "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "rawvideo",
        "-pix_fmt", pixel_format, "-fps_mode", "passthrough", "pipe:1",
    ], capture_output=True, check=True)
    frames = np.frombuffer(decoded.stdout, dtype=dtype).reshape(-1, stream["height"], stream["width"])
    return stream, frames


def write_infrared_episode(root):
    source = write_async_episode(root)
    episode = load_episode(source)
    metadata, manifests = episode.camera_meta, episode.manifests["camera.json"]
    counts = {f"camera.{key}_rgb": len(ts) for key, ts in episode.cam_ts.items()}
    originals = {}
    rng = np.random.default_rng(73)
    for key, timestamps in episode.cam_ts.items():
        for side in ("left", "right"):
            name = f"{key}_ir_{side}"
            frames = rng.integers(0, 256, size=(len(timestamps), 48, 64), dtype=np.uint8)
            write_ir_video(source / "camera" / f"{name}.mkv", frames)
            originals[name] = frames
            metadata[name] = {
                "codec": "ffv1", "container": "matroska", "lossless": True,
                "pixel_format": "gray", "encoding": "mono8", "frames": len(timestamps),
                "width": 64, "height": 48, "video": f"camera/{name}.mkv",
                "timestamp_policy": "shared_by_complete_physical_camera_bundle",
                "timestamps": f"camera/timestamp.pkl::{key}",
            }
            manifests[key]["camera_frames"][name] = len(timestamps)
            counts[f"camera.{name}"] = len(timestamps)
    for block in manifests.values():
        block["counts"] = {**counts, "items": sum(counts.values())}
    (source / "camera/metadata.json").write_text(json.dumps(metadata))
    (source / "manifests/camera.json").write_text(json.dumps(manifests))
    return source, originals


class TestInfraredCameras(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.fixture_path, cls.originals = write_infrared_episode(Path(cls.fixture.name))

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
        self.ir_before = self._ir_snapshot()

    def _ir_snapshot(self):
        return {str(path.relative_to(self.source)): (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
                for path in self.source.rglob("*") if path.suffix.lower() == ".mkv"}

    def _assert_ir_retained(self):
        self.assertEqual(self._ir_snapshot(), self.ir_before)
        episode = load_episode(self.source)
        self.assertFalse(any(path.suffix.lower() == ".mkv" for path in episode.extra_videos.values()))
        with (self.source / IR_TIMESTAMP_REL).open("rb") as handle:
            ir_ts = pickle.load(handle)
        for key, original in self.episode.cam_ts.items():
            np.testing.assert_array_equal(ir_ts[key], original)
        for name, frames in self.originals.items():
            meta = episode.camera_meta[name]
            self.assertEqual(meta["frames"], len(frames))
            self.assertEqual(meta["kirigami_trim_policy"], "preserve_untrimmed")
            self.assertEqual(meta["timestamps"], self.episode.camera_meta[name]["timestamps"].replace(
                "camera/timestamp.pkl::", "camera/ir_timestamp.pkl::"))
            for block in episode.manifests["camera.json"].values():
                self.assertEqual(block["counts"][f"camera.{name}"], len(frames))
                if name in block["camera_frames"]:
                    self.assertEqual(block["camera_frames"][name], len(frames))
        for path in self.source.rglob("*.mkv"):
            self.assertEqual(path.stat().st_nlink, 1)

    def _assert_export_omits_ir(self, output):
        self.assertFalse(any(path.suffix.lower() == ".mkv" for path in output.rglob("*")))
        self.assertFalse((output / IR_TIMESTAMP_REL).exists())
        episode = load_episode(output)
        self.assertTrue(set(self.originals).isdisjoint(episode.camera_meta))
        for block in episode.manifests["camera.json"].values():
            self.assertTrue(set(self.originals).isdisjoint(block["camera_frames"]))
            self.assertTrue({f"camera.{name}" for name in self.originals}.isdisjoint(block["counts"]))
            self.assertEqual(block["counts"]["items"],
                             sum(v for k, v in block["counts"].items() if k != "items"))

    def test_trim_then_slice_preserves_ir_files_and_excludes_them_from_exports(self):
        ranges = camera_frame_ranges(self.episode.cam_ts, 5, 16)
        with patch("kirigami.trimming.cut_video", wraps=cut_video) as cuts, \
             patch("kirigami.exporter._probe_video_stream", side_effect=AssertionError("IR must not be opened")):
            trim_source(self.episode, self.ann, 5, 16)
        self.assertTrue(cuts.call_args_list)
        self.assertTrue(all(call.args[0].suffix == ".mp4" for call in cuts.call_args_list))
        self._assert_ir_retained()
        trimmed = load_episode(self.source)
        for key, (a, b) in ranges.items():
            np.testing.assert_array_equal(trimmed.cam_ts[key], self.episode.cam_ts[key][a:b])
        t0, t1 = self.episode.frame_range_timestamps(5, 16)
        for side, original in self.episode.robot_raw.items():
            ts = np.asarray(original["timestamps"])
            mask = (ts >= t0) & (ts < t1)
            # Includes every joint's gripper dimension, plus velocity/effort.
            for field in ("joint", "vel", "effort", "eef", "timestamps"):
                np.testing.assert_array_equal(trimmed.robot_raw[side][field], np.asarray(original[field])[mask])
        ann = load_annotation(self.source)
        set_marks(ann, [5])
        ann.segments[0].subtask, ann.segments[1].subtask = "pick", "place"
        outputs = export_annotation(trimmed, ann, self.root / "out")
        for output, segment in zip(outputs, ann.segments):
            self.assertEqual(load_episode(output).n_frames, segment.n_frames)
            self._assert_export_omits_ir(output)
        self.assertFalse(list(self.source.parent.glob(".kirigami-trim-*")))
        self.assertFalse((self.source.parent / ".kirigami-backups").exists())

    def test_repeated_trims_preserve_original_ir_clocks_and_files(self):
        trim_source(self.episode, self.ann, 5, 16)
        clocks = (self.source / IR_TIMESTAMP_REL).read_bytes()
        trim_source(load_episode(self.source), load_annotation(self.source), 1, 7)
        self.assertEqual(load_episode(self.source).n_frames, 6)
        self.assertEqual((self.source / IR_TIMESTAMP_REL).read_bytes(), clocks)
        self._assert_ir_retained()

    def test_invalid_and_unmatched_mkvs_do_not_block_trim_or_slice(self):
        for path in (self.source / "camera").rglob("*.mkv"):
            path.write_bytes(b"IR deliberately unreadable")
        nested = self.source / "camera/extra/unknown_camera.MKV"
        nested.parent.mkdir()
        nested.write_bytes(b"no camera clock or metadata")
        self.ir_before = self._ir_snapshot()
        trim_source(self.episode, self.ann, 5, 16)
        self._assert_ir_retained()
        ann = load_annotation(self.source)
        ann.segments[0].subtask = "pick"
        outputs = export_annotation(load_episode(self.source), ann, self.root / "out")
        self._assert_export_omits_ir(outputs[0])

    def test_failed_trim_keeps_ir_files_and_source_unchanged(self):
        before = contents(self.source)
        with patch("kirigami.trimming.cut_video", side_effect=RuntimeError("RGB encoder failed")), \
             self.assertRaisesRegex(RuntimeError, "RGB encoder failed"):
            trim_source(self.episode, self.ann, 5, 16)
        self.assertEqual(contents(self.source), before)
        self.assertEqual(self._ir_snapshot(), self.ir_before)
        self.assertTrue(all(path.stat().st_nlink == 1 for path in self.source.rglob("*.mkv")))
        self.assertFalse(list(self.source.parent.glob(".kirigami-trim-*")))

    def test_slice_without_trim_also_excludes_ir(self):
        self.ann.segments[0].subtask = "pick"
        outputs = export_annotation(self.episode, self.ann, self.root / "out")
        self._assert_export_omits_ir(outputs[0])
        self.assertEqual(self._ir_snapshot(), self.ir_before)

    def test_cut_without_metadata_preserves_8_and_16_bit_pixels(self):
        rng = np.random.default_rng(49)
        for pix, dtype, limit in [("gray", np.uint8, 256), ("gray16le", np.uint16, 65536)]:
            with self.subTest(pixel_format=pix):
                frames = rng.integers(0, limit, size=(13, 48, 64), dtype=dtype)
                source, output = self.root / f"{pix}.mkv", self.root / f"{pix}_cut.mkv"
                write_ir_video(source, frames, pix)
                cut_video(source, output, 3, 11)
                stream, decoded = read_ir_video(output)
                self.assertEqual(stream["codec_name"], "ffv1")
                self.assertEqual(stream["pix_fmt"], pix)
                np.testing.assert_array_equal(decoded, frames[3:11])

    def test_mkv_validation_counts_decoded_frames_despite_timestamp_gaps(self):
        path = self.root / "base_0_ir_left.mkv"
        frames = self.originals["base_0_ir_left"]
        write_ir_video(path, frames, timestamp_scale=2)
        cap = cv2.VideoCapture(str(path))
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.assertNotEqual(reported, len(frames))
        validate_camera_video(path, "base_0", len(frames))
        with self.assertRaisesRegex(ValueError, "video/timestamp count mismatch"):
            validate_camera_video(path, "base_0", len(frames) + 1)

    def test_other_mkv_codecs_are_not_silently_reencoded_as_ir(self):
        path = self.root / "base_0_ir_left.mkv"
        # The actual codec, not the extension or metadata, selects the encoder.
        shutil.copyfile(self.source / "camera/base_0_rgb.mp4", path)
        with self.assertRaisesRegex(ValueError, "Unsupported MKV codec"):
            cut_video(path, self.root / "out.mkv", 2, 5)
