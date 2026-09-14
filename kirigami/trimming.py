"""Manual head/tail trimming, validated before an atomic source-directory exchange."""

from __future__ import annotations

import ctypes
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import shutil
import stat
import uuid

import numpy as np

from .annotation import Annotation, atomic_write_text, load_annotation, save_annotation, trim_annotation
from .exporter import ProgressCb, cut_video, rewrite_camera_meta, validate_episode_for_export
from .loader import LoadedEpisode, _load_episode
from .source_guard import assert_source_revision, episode_lock, source_revision


@dataclass
class TrimResult:
    source: Path
    backup: Path | None
    n_frames: int
    removed_head: int
    removed_tail: int
    warning: str = ""


def _inventory(root: Path) -> dict:
    entries = {}
    for path in [root, *sorted(root.rglob("*"))]:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) and not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"Source contains a symlink or special file: {path}")
        entries[str(path.relative_to(root))] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode)
    return entries


def _sequence(value) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim > 0
    return isinstance(value, (list, tuple))


def _select(value, mask):
    if isinstance(value, np.ndarray):
        return value[mask].copy()
    selected = [v for v, keep in zip(value, mask) if keep]
    return tuple(selected) if isinstance(value, tuple) else selected


def _slice_fields(data, mask: np.ndarray, label: str):
    """Timestamp-aligned tables may contain nested observation arrays."""
    if isinstance(data, dict):
        return {key: (deepcopy(value) if key in {"metadata", "config", "calibration"} and isinstance(value, dict)
                      else _slice_fields(value, mask, f"{label}.{key}")) for key, value in data.items()}
    if _sequence(data):
        if len(data) != len(mask):
            raise ValueError(f"Sample count mismatch: {label} has {len(data)}, expected {len(mask)}")
        return _select(data, mask)
    return deepcopy(data)


def _slice_timed(data: dict, t0: int, t1: int, label: str, *, required=False) -> dict:
    if not isinstance(data, dict) or "timestamps" not in data:
        raise ValueError(f"Missing timestamps in {label}")
    ts = np.asarray(data["timestamps"], dtype=np.int64)
    if ts.ndim != 1 or (len(ts) > 1 and np.any(np.diff(ts) < 0)):
        raise ValueError(f"Unsorted or invalid timestamps in {label}")
    mask = (ts >= t0) & (ts < t1)
    if required and not mask.any():
        raise ValueError(f"No {label} observations in the selected time range")
    return _slice_fields(data, mask, label)


def _read_data(path: Path):
    if path.suffix == ".json":
        return json.loads(path.read_text())
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return dict(archive)
    with path.open("rb") as handle:
        return pickle.load(handle)


def _write_data(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    elif path.suffix == ".npy":
        np.save(path, data, allow_pickle=False)
    elif path.suffix == ".npz":
        np.savez(path, **data)
    else:
        with path.open("wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _manifest_counts(doc, cameras: dict, robot: dict, n_action: int, n_event: int, n_frames: int):
    if isinstance(doc, list):
        return [_manifest_counts(item, cameras, robot, n_action, n_event, n_frames) for item in doc]
    if not isinstance(doc, dict):
        return deepcopy(doc)
    out = {key: _manifest_counts(value, cameras, robot, n_action, n_event, n_frames) for key, value in doc.items()}
    if isinstance(out.get("counts"), dict):
        counts = out["counts"]
        values = {**cameras, "robot.robot_state": robot["left"], "action.executed": n_action, "event.workflow": n_event}
        for key in counts.keys() & values.keys():
            counts[key] = values[key]
        if "items" in counts:
            counts["items"] = sum(value for key, value in counts.items() if key != "items" and isinstance(value, (int, float)))
    for key, value in {"camera_timestamp_samples": n_frames, "complete_sensor_bundles": n_frames,
                       "executed_actions": n_action, "event_samples": n_event}.items():
        if key in out:
            out[key] = value
    for key in ("robot_samples", "incomplete_robot_samples"):
        if isinstance(out.get(key), dict):
            out[key] = {side: robot.get(side, value) for side, value in out[key].items()}
    if isinstance(out.get("camera_frames"), dict):
        out["camera_frames"] = {key: n_frames for key in out["camera_frames"]}
    return out


def _prepare(episode: LoadedEpisode, ann: Annotation, start: int, end: int):
    source = episode.ref.path
    validate_episode_for_export(episode)
    if not 0 <= start < end <= episode.n_frames:
        raise ValueError("Keep range must contain at least one frame within the episode")
    if start == 0 and end == episode.n_frames:
        raise ValueError("Nothing to trim: the entire episode is selected")
    if ann.n_frames != episode.n_frames:
        raise ValueError("Annotation frame count differs from the source; reload the episode")
    t0, t1 = episode.frame_range_timestamps(start, end)
    changed = {}
    for key, ts in episode.cam_ts.items():
        if len(ts) != episode.n_frames or ts.ndim != 1 or np.any(np.diff(ts) <= 0):
            raise ValueError(f"Camera stream {key} is not frame-aligned with the overview camera")
    # Preserve the timestamp container types and absolute Unix timestamp values.
    camera_data = _read_data(source / "camera/timestamp.pkl")
    mask = np.zeros(episode.n_frames, dtype=bool)
    mask[start:end] = True
    changed[Path("camera/timestamp.pkl")] = {key: _select(values, mask) for key, values in camera_data.items()}
    robot = {side: _slice_timed(data, t0, t1, f"robot.{side}", required=True)
             for side, data in episode.robot_raw.items()}
    action = _slice_timed(episode.action_raw, t0, t1, "action", required=True)
    event = _slice_timed(episode.event_raw, t0, t1, "event") if episode.event_raw else {}
    changed[Path("robot/robot_state_dict.pkl")] = robot
    changed[Path("action/executed_action_dict.pkl")] = action
    if (source / "event/event_dict.pkl").exists():
        changed[Path("event/event_dict.pkl")] = event
    if (source / "camera/metadata.json").exists():
        changed[Path("camera/metadata.json")] = rewrite_camera_meta(episode.camera_meta, end - start)

    jobs, static = [], []
    cameras = {f"camera.{key}_rgb": end - start for key in episode.cam_ts}
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(source)
        if rel in changed or rel.as_posix() in {"annotations/slices.json", "annotations/trim_history.json"}:
            continue
        if rel.parts[0] == "camera" and path.suffix.lower() == ".mp4":
            keys = [key for key in episode.cam_ts if path.stem == key or path.stem.startswith(key + "_")]
            if not keys:
                raise ValueError(f"Cannot associate video with camera timestamps: {rel}")
            meta = episode.camera_meta.get(path.stem)
            jobs.append((rel, meta if isinstance(meta, dict) else None))
            cameras[f"camera.{path.stem}"] = end - start
        elif rel.parts[0] in {"observation", "observations"} and path.suffix in {".pkl", ".json", ".npy", ".npz"}:
            data = _read_data(path)
            if isinstance(data, dict) and "timestamps" in data:
                changed[rel] = _slice_timed(data, t0, t1, str(rel), required=True)
            else:
                changed[rel] = _slice_fields(data, mask, str(rel))
        elif rel.parent == Path("manifests") and path.suffix == ".json":
            continue  # Rewrite after all camera streams have been counted.
        elif rel.as_posix() == "slice_meta.json":
            meta = _read_data(path)
            if meta.get("schema") != "kirigami_slice_meta_v1":
                raise ValueError(f"Unsupported slice metadata: {rel}")
            offset = int(meta["start_frame"])
            meta.update(start_frame=offset + start, end_frame=offset + end,
                        n_frames=end - start, t_start_unix_ms=t0, t_end_unix_ms=t1)
            changed[rel] = meta
        elif path.suffix.lower() in {".md", ".txt", ".yaml", ".yml", ".log"} and rel.parts[0] not in {"observation", "observations", "camera", "robot", "action", "event", "annotations"}:
            static.append(rel)
        elif rel.parts[0] == "camera" and path.stem in {"calibration", "intrinsics", "extrinsics"} and path.suffix in {".json", ".yaml", ".yml"}:
            static.append(rel)
        else:
            raise ValueError(f"Unsupported episode file: {rel}. Its slicing rule is needed before overwriting the source.")
    robot_counts = {side: len(data["timestamps"]) for side, data in robot.items()}
    for name, raw in episode.manifests.items():
        changed[Path("manifests") / name] = _manifest_counts(raw, cameras, robot_counts,
            len(action["timestamps"]), len(event.get("timestamps", [])), end - start)
    return changed, jobs, static, t0, t1


def _exchange_directories(source: Path, prepared: Path) -> None:
    """Linux renameat2 exchanges complete directories, including nonempty ones."""
    libc = ctypes.CDLL(None, use_errno=True)
    exchange = getattr(libc, "renameat2", None)
    if exchange is None:
        raise RuntimeError("Atomic source trimming requires Linux renameat2 support")
    exchange.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    exchange.restype = ctypes.c_int
    if exchange(-100, os.fsencode(source), -100, os.fsencode(prepared), 2) != 0:
        code = ctypes.get_errno()
        raise OSError(code, f"Atomic source replacement failed: {os.strerror(code)}. Original episode retained.")


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sync_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for path in sorted([root, *(p for p in root.rglob("*") if p.is_dir())], key=lambda p: len(p.parts), reverse=True):
        _sync_dir(path)


def trim_source(episode: LoadedEpisode, annotation: Annotation, start: int, end: int,
                progress: ProgressCb | None = None) -> TrimResult:
    source = episode.ref.path.absolute()
    with episode_lock(source, exclusive=True):
        assert_source_revision(source, episode.source_revision)
        before = _inventory(source)
        fresh = _load_episode(source)
        on_disk = load_annotation(source)
        if on_disk and (on_disk.signature() != annotation.signature() or on_disk.keep_range != annotation.keep_range):
            raise ValueError("Source labels changed on disk; reload before trimming")
        changed, jobs, static, t0, t1 = _prepare(fresh, annotation, start, end)
        transaction_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12]
        backup_root = source.parent / ".kirigami-backups" / source.name
        for path in (backup_root.parent, backup_root):
            if path.is_symlink():
                raise ValueError(f"Backup directory must not be a symlink: {path}")
        transaction = backup_root / transaction_id
        transaction.mkdir(mode=0o700, parents=True)
        backup = transaction / "original"
        committed = False
        prepared_identity = None
        record = dict(schema="kirigami_trim_v1", id=transaction_id, source_path=str(source),
                      backup_path=str(backup), start_frame=start, end_frame=end,
                      old_n_frames=fresh.n_frames, n_frames=end - start,
                      t_start_unix_ms=t0, t_end_unix_ms=t1,
                      source_revision=fresh.source_revision)
        journal = transaction / "transaction.json"
        try:
            backup.mkdir()
            prepared_identity = (backup.stat().st_dev, backup.stat().st_ino)
            if backup.stat().st_dev != source.stat().st_dev:
                raise ValueError("Source and backup must be on the same filesystem for atomic replacement")
            atomic_write_text(journal, json.dumps({**record, "state": "preparing"}, indent=2) + "\n")
            for directory in (p for p in source.rglob("*") if p.is_dir()):
                (backup / directory.relative_to(source)).mkdir(parents=True, exist_ok=True)
            for rel in static:
                (backup / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / rel, backup / rel)
            for i, (rel, meta) in enumerate(jobs):
                if progress:
                    progress(f"Trim {rel}", 0.8 * i / max(len(jobs), 1))
                cut_video(source / rel, backup / rel, start, end, meta)
            for rel, data in changed.items():
                _write_data(backup / rel, data)
            adjusted = trim_annotation(annotation, start, end)
            adjusted.source_revision = source_revision(backup)
            save_annotation(backup, adjusted)
            history_path = source / "annotations/trim_history.json"
            history = json.loads(history_path.read_text()) if history_path.exists() else []
            if not isinstance(history, list):
                raise ValueError("Invalid existing trim history")
            atomic_write_text(backup / "annotations/trim_history.json", json.dumps([*history, record], indent=2) + "\n")
            if progress:
                progress("Validate retained observations and timestamps", 0.85)
            verified = _load_episode(backup)
            validate_episode_for_export(verified)
            if verified.n_frames != end - start or not np.array_equal(verified.base_ts, fresh.base_ts[start:end]):
                raise ValueError("Prepared source failed frame/timestamp validation")
            for path in [*backup.rglob("*"), backup]:
                original = source / path.relative_to(backup)
                if original.exists():
                    shutil.copystat(original, path)
            _sync_tree(backup)
            atomic_write_text(journal, json.dumps({**record, "state": "prepared"}, indent=2) + "\n")
            for path in (transaction, backup_root, backup_root.parent, source.parent):
                _sync_dir(path)
            if progress:
                progress("Write trimmed episode back to source", 0.95)
            if _inventory(source) != before:
                raise RuntimeError("Source files changed while trimming; original was not replaced")
            # This is the only operation that changes the source dataset.
            _exchange_directories(source, backup)
            committed = True
        finally:
            if not committed:
                # A signal can arrive after renameat2 succeeds but before the assignment.
                # Only discard staging when identities prove it still holds the NEW data.
                try:
                    identity = (source.stat().st_dev, source.stat().st_ino)
                    committed = prepared_identity is not None and identity == prepared_identity
                    if not committed and identity == before["."][:2]:
                        if prepared_identity is None or (backup.stat().st_dev, backup.stat().st_ino) == prepared_identity:
                            shutil.rmtree(transaction, ignore_errors=True)
                except OSError:
                    pass  # Ambiguous state: preserve both data and the recovery journal.
        warning = ""
        try:
            _sync_dir(source.parent)
            _sync_dir(transaction)
            atomic_write_text(journal, json.dumps({**record, "state": "committed"}, indent=2) + "\n")
            _sync_dir(transaction)
        except OSError as exc:
            warning = f"Trim was applied, but final durability/logging failed: {exc}. Original backup retained at {backup}"
        if progress:
            progress("Source trim complete", 1.0)
        return TrimResult(source, backup, end - start, start, fresh.n_frames - end, warning)
