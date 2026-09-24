"""Export half-open frame ranges as prometheus_raw_episode_v1 folders."""

from __future__ import annotations

import json
import hashlib
import fcntl
import pickle
import shutil
import subprocess
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .annotation import Annotation, Segment, atomic_write_text, validate_subtask_name
from .loader import CAM_KEYS, CAM_VIDEO_NAMES, LoadedEpisode
from .source_guard import assert_source_revision, episode_lock
from .camera import (PRESERVED_CAMERA_SUFFIXES, camera_frame_ranges, camera_key,
                     ignored_camera_streams, rewrite_sample_counts, validate_camera_timestamps)

ProgressCb = Callable[[str, float], None]


def default_output_root(task_dir: Path) -> Path:
    task_dir = Path(task_dir)
    return task_dir.parent / f"{task_dir.name}_sliced"


def _mask_in_range(timestamps, t0: int, t1: int) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=np.int64)
    if ts.size == 0:
        return np.zeros((0,), dtype=bool)
    return (ts >= t0) & (ts < t1)


def _take(seq, mask: np.ndarray):
    if seq is None:
        return []
    return [seq[i] for i in range(len(seq)) if i < len(mask) and mask[i]]


def slice_side_dict(side: dict, t0: int, t1: int) -> dict:
    ts = side.get("timestamps") or []
    mask = _mask_in_range(ts, t0, t1)
    out = {}
    for key, value in side.items():
        if isinstance(value, list) and len(value) == len(ts):
            out[key] = _take(value, mask)
        else:
            out[key] = deepcopy(value)
    return out


def slice_robot(robot: dict, t0: int, t1: int) -> dict:
    return {side: slice_side_dict(robot[side], t0, t1) for side in robot}


def slice_keyed_lists(data: dict, t0: int, t1: int, ts_key: str = "timestamps") -> dict:
    ts = data.get(ts_key) or []
    mask = _mask_in_range(ts, t0, t1)
    out = {}
    n = len(ts)
    for key, value in data.items():
        if isinstance(value, list) and len(value) == n:
            out[key] = _take(value, mask)
        else:
            out[key] = deepcopy(value)
    return out


def slice_camera_timestamps(cam_ts: dict[str, np.ndarray], start: int, end: int) -> dict[str, list[int]]:
    ranges = camera_frame_ranges(cam_ts, start, end)
    out: dict[str, list[int]] = {}
    for key, arr in cam_ts.items():
        a, b = ranges[key]
        sl = np.asarray(arr)[a:b]
        out[key] = [int(v) for v in sl]
    return out


def _ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH (activate the kirigami env)")
    return exe


def _probe_video_stream(path: Path) -> dict:
    exe = shutil.which("ffprobe")
    if not exe:
        raise RuntimeError("ffprobe not found on PATH (activate the kirigami env)")
    proc = subprocess.run([
        exe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,width,height", "-of", "json", str(path),
    ], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Cannot inspect camera video {path.name}: {proc.stderr.strip()}")
    streams = json.loads(proc.stdout).get("streams", [])
    if not streams or not streams[0].get("pix_fmt"):
        raise ValueError(f"Missing video stream or pixel format: {path}")
    return streams[0]


def cut_video(
    src: Path,
    dst: Path,
    start: int,
    end: int,
    camera_meta: dict | None = None,
) -> None:
    """Frame-accurate re-encode. end is exclusive."""
    if start < 0 or end <= start:
        raise ValueError(f"empty video range [{start}, {end})")
    dst.parent.mkdir(parents=True, exist_ok=True)
    meta = camera_meta or {}
    pix = meta.get("pixel_format") or "yuv444p"
    crf = 14
    quality = meta.get("quality") or {}
    if isinstance(quality, dict) and quality.get("value") is not None:
        crf = int(quality["value"])
    encoder = meta.get("encoder") or "libx265"
    if src.suffix.lower() == ".mkv":
        # Recorder IR streams use FFV1, and metadata stores "codec" rather
        # than "encoder". Probe the actual format, including when metadata is
        # absent; never fall back to RGB/HEVC or reduce sensor bit depth.
        stream = _probe_video_stream(src)
        if stream["codec_name"] != "ffv1":
            raise ValueError(f"Unsupported MKV codec in {src.name}: {stream['codec_name']}; expected lossless FFV1")
        encoder = "ffv1"
        pix = "+" + stream["pix_fmt"]  # Fail instead of silently converting.
    vf = f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS"
    cmd = [
        _ffmpeg_bin(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-vf",
        vf,
        "-an",
        "-c:v",
        encoder,
        "-pix_fmt",
        pix,
    ]
    if encoder == "ffv1":
        cmd.extend(["-level", "3", "-coder", "1", "-context", "1", "-g", "1", "-slicecrc", "1"])
    else:
        cmd.extend(["-crf", str(crf)])
    if dst.suffix.lower() == ".mp4" and encoder in {"libx265", "hevc_nvenc", "hevc_vaapi"}:
        cmd.extend(["-tag:v", "hvc1"])
    cmd.extend(["-fps_mode", "passthrough", str(dst)])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg failed for {src.name} [{start},{end}): {proc.stderr.strip()}")
    actual = _count_video_frames(dst)
    if actual != end - start:
        raise RuntimeError(f"Video frame mismatch for {src.name}: expected {end - start}, decoded {actual}")


def _count_video_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return -1
    try:
        # Decode every exported frame: container counts alone miss truncated data.
        n = 0
        while cap.grab():
            n += 1
        return n
    finally:
        cap.release()


def validate_camera_video(path: Path, key: str, n_timestamps: int) -> None:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera video: {path}")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    # Matroska's reported count can be estimated from duration/fps. Sensor
    # timestamps must be checked against the actual number of decoded frames.
    if n <= 0 or path.suffix.lower() == ".mkv":
        n = _count_video_frames(path)
    if n != n_timestamps:
        raise ValueError(f"Camera video/timestamp count mismatch: {key}: {path.name} has {n} frames, "
                         f"but its timestamp stream has {n_timestamps}")


def _update_task_fields(obj, subtask: str) -> None:
    if isinstance(obj, dict):
        if "task" in obj and isinstance(obj["task"], dict):
            obj["task"]["subtasks"] = [subtask]
            obj["task"]["prompt"] = [subtask]
        for value in obj.values():
            _update_task_fields(value, subtask)
    elif isinstance(obj, list):
        for item in obj:
            _update_task_fields(item, subtask)


def rewrite_manifests(
    manifests: dict[str, dict],
    subtask: str,
    n_cam: int,
    n_robot: int,
    n_action: int,
    n_event: int,
    dest_dir: Path,
    *,
    camera_counts: dict[str, int] | None = None,
    robot_counts: dict[str, int] | None = None,
    excluded_cameras: set[str] | frozenset[str] = frozenset(),
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, raw in manifests.items():
        doc = rewrite_sample_counts(raw,
            camera_counts if camera_counts is not None else {f"camera.{key}_rgb": n_cam for key in CAM_KEYS},
            robot_counts if robot_counts is not None else {"left": n_robot, "right": n_robot}, n_action, n_event, n_cam,
            excluded_cameras=excluded_cameras)
        _update_episode_dirs(doc, dest_dir)
        _update_task_fields(doc, subtask)
        out[name] = doc
    return out


def _update_episode_dirs(obj, dest_dir: Path) -> None:
    if isinstance(obj, dict):
        if "episode_dir" in obj:
            obj["episode_dir"] = str(dest_dir)
        for value in obj.values():
            _update_episode_dirs(value, dest_dir)
    elif isinstance(obj, list):
        for item in obj:
            _update_episode_dirs(item, dest_dir)


def rewrite_camera_meta(meta: dict, n_cam: int | dict[str, int], *,
                        preserved_streams: set[str] | frozenset[str] = frozenset()) -> dict:
    out = deepcopy(meta)
    for name, block in out.items():
        if name in preserved_streams:
            continue
        if isinstance(block, dict) and "frames" in block:
            if isinstance(n_cam, int):
                block["frames"] = n_cam
            else:
                try:
                    key = camera_key(name, n_cam)
                except ValueError:
                    continue
                block["frames"] = n_cam[key]
    return out


def write_slice_meta(
    dest: Path,
    episode: LoadedEpisode,
    segment: Segment,
    subtask: str,
    t0: int,
    t1: int,
) -> None:
    payload = {
        "schema": "kirigami_slice_meta_v1",
        "source_path": str(episode.ref.path.resolve()),
        "source_task": episode.ref.task_name,
        "source_episode_id": episode.ref.episode_id,
        "subtask": subtask,
        "start_frame": segment.start_frame,
        "end_frame": segment.end_frame,
        "n_frames": segment.n_frames,
        "t_start_unix_ms": t0,
        "t_end_unix_ms": t1,
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (dest / "slice_meta.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _dump_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _previous_exports(out_root: Path, source_path: Path) -> list[Path]:
    out_root = Path(out_root).resolve()
    source = str(Path(source_path).resolve())
    previous = []
    for meta in out_root.glob("*/*/slice_meta.json"):
        if meta.parent.is_symlink() or meta.parent.parent.is_symlink() or not meta.resolve().is_relative_to(out_root):
            continue
        try:
            data = json.loads(meta.read_text())
        except Exception:
            continue
        if data.get("source_path") == source:
            previous.append(meta.parent)
    return previous


def remove_previous_exports(out_root: Path, source_path: Path) -> int:
    """Delete only exports owned by this source; normal export uses staging below."""
    previous = _previous_exports(out_root, source_path)
    for path in previous:
        shutil.rmtree(path)
    return len(previous)


def _destination(out_root: Path, episode: LoadedEpisode, segment: Segment, index: int) -> Path:
    subtask = validate_subtask_name(segment.subtask)
    source_id = hashlib.sha256(str(episode.ref.path.resolve()).encode()).hexdigest()[:12]
    episode_id = validate_subtask_name(episode.ref.episode_id)
    dest = out_root / subtask / f"{episode_id}_{source_id}_{index:02d}"
    if dest.parent.is_symlink() or dest.is_symlink() or not dest.resolve().is_relative_to(out_root):
        raise ValueError(f"Output path escapes output root or uses a symlink: {dest}")
    return dest


def validate_episode_for_export(episode: LoadedEpisode) -> None:
    for key in CAM_KEYS:
        if key not in episode.videos:
            raise ValueError(f"Missing required camera video: {key}")
        ts = validate_camera_timestamps(key, episode.cam_ts.get(key, []))
        validate_camera_video(episode.videos[key], key, len(ts))
    for key, ts in episode.cam_ts.items():
        validate_camera_timestamps(key, ts)
    for name, data in [("left", episode.robot_raw.get("left", {})), ("right", episode.robot_raw.get("right", {})), ("action", episode.action_raw)]:
        ts = data.get("timestamps", [])
        if len(ts) == 0 or np.any(np.diff(np.asarray(ts, dtype=np.int64)) < 0):
            raise ValueError(f"Missing or unsorted {name} timestamps")
        for key, value in data.items():
            if isinstance(value, list) and len(value) != len(ts):
                raise ValueError(f"{name}.{key} has {len(value)} samples, expected {len(ts)}")


def export_segment(
    episode: LoadedEpisode,
    segment: Segment,
    dest: Path,
    progress: ProgressCb | None = None,
    *,
    manifest_dest: Path | None = None,
) -> Path:
    if segment.n_frames <= 0:
        raise ValueError("empty segment")
    subtask = validate_subtask_name(segment.subtask)
    if not 0 <= segment.start_frame < segment.end_frame <= episode.n_frames:
        raise ValueError("Segment is outside the source frame range")
    validate_episode_for_export(episode)

    t0, t1 = episode.frame_range_timestamps(segment.start_frame, segment.end_frame)
    start, end = segment.start_frame, segment.end_frame
    dest = Path(dest)
    # Never remove a caller's directory. Replacement is handled transactionally.
    dest.mkdir(parents=True)

    cam_ts = slice_camera_timestamps(episode.cam_ts, start, end)
    ranges = camera_frame_ranges(episode.cam_ts, start, end)
    camera_counts = {f"camera.{key}_rgb": len(ts) for key, ts in cam_ts.items()}
    n_cam = len(cam_ts.get("base_0", []))
    robot = slice_robot(episode.robot_raw, t0, t1)
    action = slice_keyed_lists(episode.action_raw, t0, t1)
    event = slice_keyed_lists(episode.event_raw, t0, t1) if episode.event_raw else {"data": [], "timestamps": [], "names": [], "metadata": []}

    n_robot = len((robot.get("left") or {}).get("timestamps") or [])
    n_action = len(action.get("timestamps") or [])
    n_event = len(event.get("timestamps") or [])
    if not n_robot or not len((robot.get("right") or {}).get("timestamps") or []) or not n_action:
        raise ValueError("Segment has no robot or action samples in its time range")

    jobs = []
    for key in CAM_KEYS:
        src = episode.videos.get(key)
        if src is None:
            continue
        name = CAM_VIDEO_NAMES[key]
        meta = episode.camera_meta.get(f"{key}_rgb") if isinstance(episode.camera_meta, dict) else None
        jobs.append((f"camera {name}", src, dest / "camera" / name, meta if isinstance(meta, dict) else None, ranges[key]))
    for name, src in episode.extra_videos.items():
        if src.suffix.lower() in PRESERVED_CAMERA_SUFFIXES:
            continue
        # keep extra videos next to the three main cams, not nested remove_hand/
        key = camera_key(name, ranges)
        validate_camera_video(src, key, len(episode.cam_ts[key]))
        meta = episode.camera_meta.get(src.stem)
        jobs.append((f"extra {name}", src, dest / "camera" / Path(name).name,
                     meta if isinstance(meta, dict) else None, ranges[key]))
        camera_counts[f"camera.{Path(name).stem}"] = len(cam_ts[key])

    total = max(len(jobs), 1)
    for i, (label, src, dst, meta, (a, b)) in enumerate(jobs):
        if progress:
            progress(f"{dest.name}: {label}", i / (total + 1))
        cut_video(src, dst, a, b, meta)

    if progress:
        progress(f"{dest.name}: write pkl/manifests", len(jobs) / (total + 1))

    _dump_pickle(dest / "camera" / "timestamp.pkl", cam_ts)
    _dump_pickle(dest / "robot" / "robot_state_dict.pkl", robot)
    _dump_pickle(dest / "action" / "executed_action_dict.pkl", action)
    _dump_pickle(dest / "event" / "event_dict.pkl", event)

    ignored = ignored_camera_streams(episode.ref.path / "camera", episode.camera_meta)
    meta = rewrite_camera_meta({name: block for name, block in episode.camera_meta.items() if name not in ignored},
                               {key: len(ts) for key, ts in cam_ts.items()})
    (dest / "camera").mkdir(parents=True, exist_ok=True)
    (dest / "camera" / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    mans = rewrite_manifests(episode.manifests, subtask, n_cam, n_robot, n_action, n_event, manifest_dest or dest,
        camera_counts=camera_counts, robot_counts={side: len(data.get("timestamps", [])) for side, data in robot.items()},
        excluded_cameras=ignored)
    man_dir = dest / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    if mans:
        for name, doc in mans.items():
            (man_dir / name).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    else:
        (man_dir / "camera.json").write_text("{}\n")

    write_slice_meta(dest, episode, segment, subtask, t0, t1)
    if progress:
        progress(f"{dest.name}: done", 1.0)
    return dest


def export_annotation(
    episode: LoadedEpisode,
    annotation: Annotation,
    out_root: Path,
    replace_previous: bool = True,
    progress: ProgressCb | None = None,
) -> list[Path]:
    with episode_lock(episode.ref.path):
        assert_source_revision(episode.ref.path, episode.source_revision)
        return _export_annotation_locked(episode, annotation, out_root, replace_previous, progress)


def _export_annotation_locked(episode, annotation, out_root, replace_previous, progress):
    segs = [(i, s) for i, s in enumerate(annotation.segments) if not s.discard and s.n_frames > 0]
    if not segs:
        raise ValueError("no exportable segments (need subtask names and Export checked)")
    out_root = Path(out_root).expanduser().resolve()
    validate_episode_for_export(episode)
    destinations = [_destination(out_root, episode, seg, i) for i, seg in segs]
    for _, seg in segs:
        if not 0 <= seg.start_frame < seg.end_frame <= episode.n_frames:
            raise ValueError("Segment is outside the source frame range")
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / ".kirigami-export.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another export is already writing to this output root") from exc
        previous = _previous_exports(out_root, episode.ref.path) if replace_previous else []
        for dest in destinations:
            if dest.exists() and dest not in previous:
                raise FileExistsError(f"Refusing to overwrite an existing directory: {dest}")
        staging = Path(tempfile.mkdtemp(prefix=".kirigami-export-", dir=out_root))
        backups: list[tuple[Path, Path]] = []
        installed: list[Path] = []
        preserve_staging = False
        try:
            staged = []
            for position, ((_, seg), dest) in enumerate(zip(segs, destinations)):
                def _cb(msg: str, frac: float, position=position) -> None:
                    if progress:
                        progress(msg, 0.95 * (position + frac) / len(segs))
                staged.append(export_segment(episode, seg, staging / "new" / str(position), progress=_cb, manifest_dest=dest))
            # Keep all previous data until every new segment has passed validation.
            (staging / "backup").mkdir()
            atomic_write_text(staging / "recovery.json", json.dumps({
                "source_path": str(episode.ref.path.resolve()),
                "previous": [{"destination": str(old), "backup": f"backup/{index}"} for index, old in enumerate(previous)],
                "new_destinations": [str(dest) for dest in destinations],
            }, indent=2) + "\n")
            for index, old in enumerate(previous):
                backup = staging / "backup" / str(index)
                old.rename(backup)
                backups.append((old, backup))
            for source, dest in zip(staged, destinations):
                if dest.parent.is_symlink() or not dest.resolve().is_relative_to(out_root):
                    raise ValueError(f"Output path changed during export: {dest}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    raise FileExistsError(f"Output appeared during export: {dest}")
                source.rename(dest)
                installed.append(dest)
        except BaseException:
            try:
                for dest in reversed(installed):
                    shutil.rmtree(dest)
                for old, backup in reversed(backups):
                    backup.rename(old)
            except OSError as rollback_error:
                preserve_staging = True
                raise RuntimeError(f"Could not restore previous exports; backups are preserved in {staging}") from rollback_error
            raise
        finally:
            if not preserve_staging:
                shutil.rmtree(staging, ignore_errors=True)
        if progress:
            progress("Export complete", 1.0)
        return destinations
