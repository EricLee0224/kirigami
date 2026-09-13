"""Export half-open frame ranges as prometheus_raw_episode_v1 folders."""

from __future__ import annotations

import json
import pickle
import shutil
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .annotation import Annotation, Segment
from .loader import CAM_KEYS, CAM_VIDEO_NAMES, LoadedEpisode

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
    out: dict[str, list[int]] = {}
    for key, arr in cam_ts.items():
        sl = np.asarray(arr)[start:end]
        out[key] = [int(v) for v in sl]
    return out


def _ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH (activate the kirigami env)")
    return exe


def cut_video(
    src: Path,
    dst: Path,
    start: int,
    end: int,
    camera_meta: dict | None = None,
) -> None:
    """Frame-accurate re-encode. end is exclusive."""
    if end <= start:
        raise ValueError(f"empty video range [{start}, {end})")
    dst.parent.mkdir(parents=True, exist_ok=True)
    meta = camera_meta or {}
    pix = meta.get("pixel_format") or "yuv444p"
    crf = 14
    quality = meta.get("quality") or {}
    if isinstance(quality, dict) and quality.get("value") is not None:
        crf = int(quality["value"])
    encoder = meta.get("encoder") or "libx265"
    vf = f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS"
    cmd = [
        _ffmpeg_bin(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vf",
        vf,
        "-an",
        "-c:v",
        encoder,
        "-pix_fmt",
        pix,
        "-crf",
        str(crf),
        "-tag:v",
        "hvc1",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg failed for {src.name} [{start},{end}): {proc.stderr.strip()}")


def _count_video_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return -1
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


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


def _set_if_present(obj: dict, key: str, value) -> None:
    if key in obj:
        obj[key] = value


def rewrite_manifests(
    manifests: dict[str, dict],
    subtask: str,
    n_cam: int,
    n_robot: int,
    n_action: int,
    n_event: int,
    dest_dir: Path,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, raw in manifests.items():
        doc = deepcopy(raw)
        _walk_counts(doc, n_cam=n_cam, n_robot=n_robot, n_action=n_action, n_event=n_event, dest_dir=dest_dir)
        _update_task_fields(doc, subtask)
        out[name] = doc
    return out


def _walk_counts(obj, n_cam: int, n_robot: int, n_action: int, n_event: int, dest_dir: Path) -> None:
    if isinstance(obj, dict):
        if "counts" in obj and isinstance(obj["counts"], dict):
            c = obj["counts"]
            if "action.executed" in c:
                c["action.executed"] = n_action
            if "robot.robot_state" in c:
                c["robot.robot_state"] = n_robot
            if "event.workflow" in c:
                c["event.workflow"] = n_event
            for cam in CAM_KEYS:
                key = f"camera.{cam}_rgb"
                if key in c:
                    c[key] = n_cam
            if "items" in c:
                c["items"] = n_cam * 3 + n_robot + n_action + n_event
        if "camera_frames" in obj and isinstance(obj["camera_frames"], dict):
            for k in list(obj["camera_frames"]):
                obj["camera_frames"][k] = n_cam
        for key, val in (
            ("camera_timestamp_samples", n_cam),
            ("complete_sensor_bundles", n_cam),
            ("executed_actions", n_action),
            ("event_samples", n_event),
        ):
            _set_if_present(obj, key, val)
        if "robot_samples" in obj and isinstance(obj["robot_samples"], dict):
            obj["robot_samples"] = {side: n_robot for side in obj["robot_samples"]}
        if "incomplete_robot_samples" in obj and isinstance(obj["incomplete_robot_samples"], dict):
            obj["incomplete_robot_samples"] = {side: n_robot for side in obj["incomplete_robot_samples"]}
        if "episode_dir" in obj:
            obj["episode_dir"] = str(dest_dir)
        for value in obj.values():
            _walk_counts(value, n_cam, n_robot, n_action, n_event, dest_dir)
    elif isinstance(obj, list):
        for item in obj:
            _walk_counts(item, n_cam, n_robot, n_action, n_event, dest_dir)


def rewrite_camera_meta(meta: dict, n_cam: int) -> dict:
    out = deepcopy(meta)
    for block in out.values():
        if isinstance(block, dict) and "frames" in block:
            block["frames"] = n_cam
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


def remove_previous_exports(out_root: Path, source_path: Path) -> int:
    """Delete sliced episodes whose slice_meta.source_path matches this source."""
    out_root = Path(out_root)
    if not out_root.exists():
        return 0
    source = str(Path(source_path).resolve())
    removed = 0
    for meta in out_root.glob("*/*/slice_meta.json"):
        try:
            data = json.loads(meta.read_text())
        except Exception:
            continue
        if data.get("source_path") == source:
            shutil.rmtree(meta.parent, ignore_errors=True)
            removed += 1
    return removed


def _next_index_name(subtask_dir: Path, src_ep: str, seg_idx: int) -> Path:
    return subtask_dir / f"{src_ep}_{seg_idx:02d}"


def export_segment(
    episode: LoadedEpisode,
    segment: Segment,
    dest: Path,
    progress: ProgressCb | None = None,
) -> Path:
    if segment.n_frames <= 0:
        raise ValueError("empty segment")
    subtask = segment.subtask.strip()
    if not subtask:
        raise ValueError("segment has no subtask name")

    t0, t1 = episode.frame_range_timestamps(segment.start_frame, segment.end_frame)
    start, end = segment.start_frame, segment.end_frame
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    cam_ts = slice_camera_timestamps(episode.cam_ts, start, end)
    n_cam = len(cam_ts.get("base_0", []))
    robot = slice_robot(episode.robot_raw, t0, t1)
    action = slice_keyed_lists(episode.action_raw, t0, t1)
    event = slice_keyed_lists(episode.event_raw, t0, t1) if episode.event_raw else {"data": [], "timestamps": [], "names": [], "metadata": []}

    n_robot = len((robot.get("left") or {}).get("timestamps") or [])
    n_action = len(action.get("timestamps") or [])
    n_event = len(event.get("timestamps") or [])

    jobs: list[tuple[str, Path, Path, dict | None]] = []
    for key in CAM_KEYS:
        src = episode.videos.get(key)
        if src is None:
            continue
        name = CAM_VIDEO_NAMES[key]
        meta = episode.camera_meta.get(f"{key}_rgb") if isinstance(episode.camera_meta, dict) else None
        jobs.append((f"camera {name}", src, dest / "camera" / name, meta if isinstance(meta, dict) else None))
    for name, src in episode.extra_videos.items():
        # keep extra videos next to the three main cams, not nested remove_hand/
        jobs.append((f"extra {name}", src, dest / "camera" / Path(name).name, None))

    total = max(len(jobs), 1)
    for i, (label, src, dst, meta) in enumerate(jobs):
        if progress:
            progress(f"{dest.name}: {label}", i / (total + 1))
        cut_video(src, dst, start, end, meta)

    if progress:
        progress(f"{dest.name}: write pkl/manifests", len(jobs) / (total + 1))

    _dump_pickle(dest / "camera" / "timestamp.pkl", cam_ts)
    _dump_pickle(dest / "robot" / "robot_state_dict.pkl", robot)
    _dump_pickle(dest / "action" / "executed_action_dict.pkl", action)
    _dump_pickle(dest / "event" / "event_dict.pkl", event)

    meta = rewrite_camera_meta(episode.camera_meta, n_cam)
    (dest / "camera").mkdir(parents=True, exist_ok=True)
    (dest / "camera" / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    mans = rewrite_manifests(episode.manifests, subtask, n_cam, n_robot, n_action, n_event, dest)
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
    segs = annotation.exportable()
    if not segs:
        raise ValueError("no exportable segments (need subtask names and Export checked)")
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if replace_previous:
        remove_previous_exports(out_root, episode.ref.path)

    written: list[Path] = []
    n = len(segs)
    for i, seg in enumerate(segs):
        sub_dir = out_root / seg.subtask.strip()
        dest = _next_index_name(sub_dir, episode.ref.episode_id, i)

        def _cb(msg: str, frac: float, i=i) -> None:
            if progress:
                progress(msg, (i + frac) / n)

        written.append(export_segment(episode, seg, dest, progress=_cb))
    return written
