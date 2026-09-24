"""Camera streams share a clock, not necessarily frame numbers or frame counts."""

from copy import deepcopy
from pathlib import Path

import numpy as np

# MKV/IR files are retained as raw attachments, outside the training timeline.
CAMERA_VIDEO_SUFFIXES = {".mp4"}
PRESERVED_CAMERA_SUFFIXES = {".mkv"}
IR_TIMESTAMP_REL = Path("camera/ir_timestamp.pkl")


def ignored_camera_streams(cam_dir: Path, metadata: dict) -> set[str]:
    """Identify ignored streams without opening or decoding their videos."""
    names = {path.stem for path in cam_dir.rglob("*")
             if path.is_file() and path.suffix.lower() in PRESERVED_CAMERA_SUFFIXES}
    names.update(name for name, block in metadata.items()
                 if isinstance(block, dict)
                 and Path(str(block.get("video", ""))).suffix.lower() in PRESERVED_CAMERA_SUFFIXES)
    return names


def frame_time_range(base_ts, start: int, end: int) -> tuple[int, int]:
    n = len(base_ts)
    if not n:
        return 0, 0
    start, end = int(np.clip(start, 0, n)), int(np.clip(end, 0, n))
    if start >= end:
        t = int(base_ts[min(start, n - 1)])
        return t, t
    t0 = int(base_ts[start])
    dt = int(base_ts[-1] - base_ts[-2]) if n >= 2 else 33
    t1 = int(base_ts[end]) if end < n else int(base_ts[-1]) + max(dt, 1)
    return t0, t1


def validate_camera_timestamps(key: str, timestamps) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=np.int64)
    if ts.ndim != 1 or not len(ts):
        raise ValueError(f"Missing or invalid camera timestamps: {key}")
    if np.any(np.diff(ts) <= 0):
        raise ValueError(f"Camera timestamps must be strictly increasing: {key}")
    return ts


def camera_frame_ranges(cam_ts, start: int, end: int) -> dict[str, tuple[int, int]]:
    base = validate_camera_timestamps("base_0", cam_ts.get("base_0", []))
    if not 0 <= start < end <= len(base):
        raise ValueError("Camera range is outside the overview frame range")
    t0, t1 = frame_time_range(base, start, end)
    ranges = {}
    for key, values in cam_ts.items():
        ts = validate_camera_timestamps(key, values)
        a, b = np.searchsorted(ts, [t0, t1], side="left")
        if a == b:
            raise ValueError(f"Camera {key} has no frames in the selected time range [{t0}, {t1})")
        ranges[key] = (int(a), int(b))
    return ranges


def camera_key(name: str | Path, keys) -> str:
    stem = Path(name).stem
    matches = [key for key in keys if stem == key or stem.startswith(key + "_")]
    if not matches:
        raise ValueError(f"Cannot associate video with camera timestamps: {name}")
    return max(matches, key=len)


def rewrite_sample_counts(doc, cameras: dict[str, int], robot: dict[str, int],
                          n_action: int, n_event: int, n_base: int, *,
                          excluded_cameras: set[str] | frozenset[str] = frozenset()):
    """Rewrite counts per physical camera, including per-recorder summaries."""
    if isinstance(doc, list):
        return [rewrite_sample_counts(item, cameras, robot, n_action, n_event, n_base,
                                      excluded_cameras=excluded_cameras) for item in doc]
    if not isinstance(doc, dict):
        return deepcopy(doc)
    out = {key: rewrite_sample_counts(value, cameras, robot, n_action, n_event, n_base,
                                     excluded_cameras=excluded_cameras)
           for key, value in doc.items()}
    camera_frames = out.get("camera_frames")
    if isinstance(camera_frames, dict):
        out["camera_frames"] = {key: cameras.get(f"camera.{key}", value) for key, value in camera_frames.items()
                                if key not in excluded_cameras}
        camera_frames = out["camera_frames"]
    camera_n = cameras.get(f"camera.{out.get('sensor_name')}_rgb")
    if camera_n is None and isinstance(camera_frames, dict) and len(camera_frames) == 1:
        camera_n = cameras.get(f"camera.{next(iter(camera_frames))}")
    if camera_n is None:
        camera_n = n_base
    if isinstance(out.get("counts"), dict):
        counts = out["counts"]
        for name in excluded_cameras:
            counts.pop(f"camera.{name}", None)
        values = {**cameras, "robot.robot_state": robot["left"], "action.executed": n_action, "event.workflow": n_event}
        for key in counts.keys() & values.keys():
            counts[key] = values[key]
        if "items" in counts:
            counts["items"] = sum(value for key, value in counts.items() if key != "items" and isinstance(value, (int, float)))
    for key, value in {"camera_timestamp_samples": camera_n, "complete_sensor_bundles": camera_n,
                       "executed_actions": n_action, "event_samples": n_event}.items():
        if key in out:
            out[key] = value
    for key in ("robot_samples", "incomplete_robot_samples"):
        if isinstance(out.get(key), dict):
            out[key] = {side: robot.get(side, value) for side, value in out[key].items()}
    return out
