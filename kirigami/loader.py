"""Discover and load Prometheus raw episodes (prometheus_raw_episode_v1)."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CAM_KEYS = ("base_0", "left_wrist_0", "right_wrist_0")
CAM_VIDEO_NAMES = {
    "base_0": "base_0_rgb.mp4",
    "left_wrist_0": "left_wrist_0_rgb.mp4",
    "right_wrist_0": "right_wrist_0_rgb.mp4",
}
ANNOTATION_REL = Path("annotations") / "slices.json"


def nearest_indices(src_ts: np.ndarray, tgt_ts: np.ndarray) -> np.ndarray:
    """For each tgt timestamp, index of the closest src timestamp (src sorted asc)."""
    src_ts = np.asarray(src_ts, dtype=np.int64)
    tgt_ts = np.asarray(tgt_ts, dtype=np.int64)
    if src_ts.size == 0:
        return np.zeros(len(tgt_ts), dtype=np.int64)
    pos = np.searchsorted(src_ts, tgt_ts)
    pos = np.clip(pos, 1, len(src_ts) - 1)
    prev = pos - 1
    choose_prev = np.abs(tgt_ts - src_ts[prev]) <= np.abs(src_ts[pos] - tgt_ts)
    idx = np.where(choose_prev, prev, pos)
    idx = np.where(tgt_ts <= src_ts[0], 0, idx)
    idx = np.where(tgt_ts >= src_ts[-1], len(src_ts) - 1, idx)
    return idx.astype(np.int64)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _stack_or_empty(seq, width: int) -> np.ndarray:
    if not seq:
        return np.zeros((0, width), dtype=np.float32)
    return np.asarray(np.stack(seq, axis=0), dtype=np.float32)


@dataclass
class EpisodeRef:
    path: Path
    task_name: str
    episode_id: str
    n_frames: int = 0
    duration_s: float = 0.0

    @property
    def annotation_path(self) -> Path:
        return self.path / ANNOTATION_REL

    @property
    def has_annotation(self) -> bool:
        return self.annotation_path.exists()

    @property
    def is_exported(self) -> bool:
        if not self.annotation_path.exists():
            return False
        try:
            data = json.loads(self.annotation_path.read_text())
        except Exception:
            return False
        return bool(data.get("exported_at"))


@dataclass
class LoadedEpisode:
    ref: EpisodeRef
    cam_ts: dict[str, np.ndarray]
    base_ts: np.ndarray
    robot_raw: dict
    action_raw: dict
    event_raw: dict
    camera_meta: dict
    manifests: dict[str, dict]
    videos: dict[str, Path]
    extra_videos: dict[str, Path]
    left_joint_aligned: np.ndarray
    right_joint_aligned: np.ndarray
    state_aligned: np.ndarray
    action_aligned: np.ndarray
    n_frames: int
    fps: float = 30.0

    @property
    def duration_s(self) -> float:
        if self.n_frames < 2:
            return self.n_frames / self.fps if self.n_frames else 0.0
        return float(self.base_ts[-1] - self.base_ts[0]) / 1000.0

    def time_s(self, frame: int) -> float:
        frame = int(np.clip(frame, 0, max(self.n_frames - 1, 0)))
        if self.n_frames == 0:
            return 0.0
        return float(self.base_ts[frame] - self.base_ts[0]) / 1000.0

    def frame_range_timestamps(self, start: int, end: int) -> tuple[int, int]:
        """Unix-ms [t_start, t_end) for a half-open frame range."""
        if self.n_frames == 0:
            return 0, 0
        start = int(np.clip(start, 0, self.n_frames))
        end = int(np.clip(end, 0, self.n_frames))
        if start >= end:
            t = int(self.base_ts[min(start, self.n_frames - 1)])
            return t, t
        t0 = int(self.base_ts[start])
        if end >= self.n_frames:
            dt = int(self.base_ts[-1] - self.base_ts[-2]) if self.n_frames >= 2 else 33
            t1 = int(self.base_ts[-1]) + max(dt, 1)
        else:
            t1 = int(self.base_ts[end])
        return t0, t1


def is_episode_dir(path: Path) -> bool:
    return path.is_dir() and (path / "camera" / "timestamp.pkl").exists()


def discover_episodes(task_dir: Path) -> list[EpisodeRef]:
    """List episode folders under a task directory (0000, 0001, ...)."""
    task_dir = Path(task_dir)
    if is_episode_dir(task_dir):
        n_frames, duration_s = _peek_timing(task_dir)
        return [
            EpisodeRef(
                path=task_dir,
                task_name=task_dir.parent.name,
                episode_id=task_dir.name,
                n_frames=n_frames,
                duration_s=duration_s,
            )
        ]
    refs: list[EpisodeRef] = []
    if not task_dir.is_dir():
        return refs
    for child in sorted(task_dir.iterdir()):
        if not is_episode_dir(child):
            continue
        n_frames, duration_s = _peek_timing(child)
        refs.append(
            EpisodeRef(
                path=child,
                task_name=task_dir.name,
                episode_id=child.name,
                n_frames=n_frames,
                duration_s=duration_s,
            )
        )
    return refs


def discover_many(paths: list[Path]) -> list[EpisodeRef]:
    seen: set[Path] = set()
    refs: list[EpisodeRef] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            continue
        for ref in discover_episodes(path):
            key = ref.path.resolve()
            if key in seen:
                continue
            seen.add(key)
            refs.append(ref)
    return refs


def _peek_timing(ep: Path) -> tuple[int, float]:
    meta = _read_json(ep / "camera" / "metadata.json")
    n = 0
    if isinstance(meta, dict):
        block = meta.get("base_0_rgb") or next(iter(meta.values()), None)
        if isinstance(block, dict):
            n = int(block.get("frames") or 0)
    if n <= 0:
        ts_path = ep / "camera" / "timestamp.pkl"
        try:
            with open(ts_path, "rb") as f:
                cam_ts = pickle.load(f)
            n = len(cam_ts.get("base_0", []))
        except Exception:
            n = 0
    duration = n / 30.0 if n else 0.0
    return n, duration


def _find_extra_videos(cam_dir: Path, main_videos: set[str]) -> dict[str, Path]:
    extra: dict[str, Path] = {}
    if not cam_dir.is_dir():
        return extra
    for path in sorted(cam_dir.rglob("*.mp4")):
        if path.name in main_videos:
            continue
        extra[path.name] = path
    return extra


def load_episode(path: Path) -> LoadedEpisode:
    ep = Path(path)
    if not is_episode_dir(ep):
        raise FileNotFoundError(f"not a prometheus episode: {ep}")

    with open(ep / "camera" / "timestamp.pkl", "rb") as f:
        cam_ts_raw = pickle.load(f)
    cam_ts = {k: np.asarray(cam_ts_raw[k], dtype=np.int64) for k in cam_ts_raw}
    if "base_0" not in cam_ts:
        raise KeyError(f"{ep}: camera/timestamp.pkl missing base_0")
    base_ts = cam_ts["base_0"]
    n_frames = int(base_ts.shape[0])

    robot_path = ep / "robot" / "robot_state_dict.pkl"
    action_path = ep / "action" / "executed_action_dict.pkl"
    event_path = ep / "event" / "event_dict.pkl"
    with open(robot_path, "rb") as f:
        robot = pickle.load(f)
    with open(action_path, "rb") as f:
        action = pickle.load(f)
    event = {}
    if event_path.exists():
        with open(event_path, "rb") as f:
            event = pickle.load(f)

    camera_meta = _read_json(ep / "camera" / "metadata.json")
    manifests: dict[str, dict] = {}
    man_dir = ep / "manifests"
    if man_dir.is_dir():
        for p in sorted(man_dir.glob("*.json")):
            manifests[p.name] = _read_json(p)

    videos: dict[str, Path] = {}
    main_names: set[str] = set()
    for key, name in CAM_VIDEO_NAMES.items():
        candidate = ep / "camera" / name
        main_names.add(name)
        if candidate.exists():
            videos[key] = candidate

    extra_videos = _find_extra_videos(ep / "camera", main_names)

    left = robot.get("left") or {}
    right = robot.get("right") or {}
    left_joint = _stack_or_empty(left.get("joint") or [], 7)
    right_joint = _stack_or_empty(right.get("joint") or [], 7)
    left_ts = np.asarray(left.get("timestamps") or [], dtype=np.int64)
    right_ts = np.asarray(right.get("timestamps") or [], dtype=np.int64)

    actions = action.get("actions") or []
    act = _stack_or_empty(actions, 14)
    act_ts = np.asarray(action.get("timestamps") or [], dtype=np.int64)

    if n_frames:
        li = nearest_indices(left_ts, base_ts) if left_joint.size else np.zeros(n_frames, dtype=np.int64)
        ri = nearest_indices(right_ts, base_ts) if right_joint.size else np.zeros(n_frames, dtype=np.int64)
        ai = nearest_indices(act_ts, base_ts) if act.size else np.zeros(n_frames, dtype=np.int64)
        left_aligned = left_joint[li] if left_joint.size else np.zeros((n_frames, 7), dtype=np.float32)
        right_aligned = right_joint[ri] if right_joint.size else np.zeros((n_frames, 7), dtype=np.float32)
        act_aligned = act[ai] if act.size else np.zeros((n_frames, 14), dtype=np.float32)
    else:
        left_aligned = np.zeros((0, 7), dtype=np.float32)
        right_aligned = np.zeros((0, 7), dtype=np.float32)
        act_aligned = np.zeros((0, 14), dtype=np.float32)

    state = np.concatenate([left_aligned, right_aligned], axis=1).astype(np.float32)
    fps = _estimate_fps(base_ts)

    task_name = ep.parent.name
    ref = EpisodeRef(
        path=ep,
        task_name=task_name,
        episode_id=ep.name,
        n_frames=n_frames,
        duration_s=float(base_ts[-1] - base_ts[0]) / 1000.0 if n_frames >= 2 else n_frames / fps,
    )
    return LoadedEpisode(
        ref=ref,
        cam_ts=cam_ts,
        base_ts=base_ts,
        robot_raw=robot,
        action_raw=action,
        event_raw=event,
        camera_meta=camera_meta,
        manifests=manifests,
        videos=videos,
        extra_videos=extra_videos,
        left_joint_aligned=left_aligned,
        right_joint_aligned=right_aligned,
        state_aligned=state,
        action_aligned=act_aligned,
        n_frames=n_frames,
        fps=fps,
    )


def _estimate_fps(base_ts: np.ndarray) -> float:
    if base_ts.size < 2:
        return 30.0
    dt = np.median(np.diff(base_ts.astype(np.float64)))
    if dt <= 0:
        return 30.0
    return float(1000.0 / dt)
