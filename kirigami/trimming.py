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
import tempfile
import uuid

import numpy as np

from .annotation import Annotation, atomic_write_text, load_annotation, save_annotation, trim_annotation
from .exporter import ProgressCb, cut_video, rewrite_camera_meta, validate_camera_video, validate_episode_for_export
from .camera import (CAMERA_VIDEO_SUFFIXES, IR_TIMESTAMP_REL, PRESERVED_CAMERA_SUFFIXES,
                     camera_frame_ranges, camera_key, ignored_camera_streams, rewrite_sample_counts)
from .loader import LoadedEpisode, _load_episode
from .source_guard import assert_source_revision, episode_lock, source_revision

ROBOT_PREVIEW_NAMES = {"state_joint_vis.png", "state_eef_xyz_vis.png"}


@dataclass
class TrimResult:
    source: Path
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


def _clear_stale_plot_references(doc) -> None:
    if isinstance(doc, dict):
        if isinstance(doc.get("plot_outputs"), list):
            doc["plot_outputs"] = [name for name in doc["plot_outputs"] if Path(str(name)).name not in ROBOT_PREVIEW_NAMES]
            doc.pop("plot_error", None)
        for value in doc.values():
            _clear_stale_plot_references(value)
    elif isinstance(doc, list):
        for value in doc:
            _clear_stale_plot_references(value)


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
    ranges = camera_frame_ranges(episode.cam_ts, start, end)
    # Preserve the timestamp container types and absolute Unix timestamp values.
    camera_data = _read_data(source / "camera/timestamp.pkl")
    mask = np.zeros(episode.n_frames, dtype=bool)
    mask[start:end] = True
    changed[Path("camera/timestamp.pkl")] = {key: values[ranges[key][0]:ranges[key][1]] for key, values in camera_data.items()}
    robot = {side: _slice_timed(data, t0, t1, f"robot.{side}", required=True)
             for side, data in episode.robot_raw.items()}
    action = _slice_timed(episode.action_raw, t0, t1, "action", required=True)
    event = _slice_timed(episode.event_raw, t0, t1, "event") if episode.event_raw else {}
    changed[Path("robot/robot_state_dict.pkl")] = robot
    changed[Path("action/executed_action_dict.pkl")] = action
    if (source / "event/event_dict.pkl").exists():
        changed[Path("event/event_dict.pkl")] = event
    ignored = ignored_camera_streams(source / "camera", episode.camera_meta)
    if ignored and not (source / IR_TIMESTAMP_REL).exists():
        # Capture the pre-trim clocks once. Repeated trims keep these intact.
        changed[IR_TIMESTAMP_REL] = camera_data
    if (source / "camera/metadata.json").exists():
        meta = rewrite_camera_meta(episode.camera_meta, {key: b - a for key, (a, b) in ranges.items()},
                                   preserved_streams=ignored)
        for name in ignored:
            block = meta.get(name)
            if not isinstance(block, dict):
                continue
            block["kirigami_trim_policy"] = "preserve_untrimmed"
            timestamp_ref = block.get("timestamps")
            if isinstance(timestamp_ref, str) and timestamp_ref.startswith("camera/timestamp.pkl::"):
                block["timestamps"] = timestamp_ref.replace("camera/timestamp.pkl::", f"{IR_TIMESTAMP_REL.as_posix()}::", 1)
        changed[Path("camera/metadata.json")] = meta

    jobs, static, preserved = [], [], []
    cameras = {f"camera.{key}_rgb": b - a for key, (a, b) in ranges.items()}
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(source)
        if rel in changed or rel.as_posix() in {"annotations/slices.json", "annotations/trim_history.json"}:
            continue
        if rel.parent == Path("robot") and rel.name in ROBOT_PREVIEW_NAMES:
            continue  # Recorder-generated plots refer to the pre-trim timeline.
        if rel == IR_TIMESTAMP_REL or (rel.parts[0] == "camera" and path.suffix.lower() in PRESERVED_CAMERA_SUFFIXES):
            preserved.append(rel)
        elif rel.parts[0] == "camera" and path.suffix.lower() in CAMERA_VIDEO_SUFFIXES:
            key = camera_key(path, episode.cam_ts)
            validate_camera_video(path, key, len(episode.cam_ts[key]))
            meta = episode.camera_meta.get(path.stem)
            jobs.append((rel, meta if isinstance(meta, dict) else None, ranges[key]))
            cameras[f"camera.{path.stem}"] = ranges[key][1] - ranges[key][0]
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
        changed[Path("manifests") / name] = rewrite_sample_counts(raw, cameras, robot_counts,
            len(action["timestamps"]), len(event.get("timestamps", [])), end - start)
        _clear_stale_plot_references(changed[Path("manifests") / name])
    return changed, jobs, static, preserved, t0, t1


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


def _sync_tree(root: Path, preserved: set[Path] | frozenset[Path] = frozenset()) -> None:
    for path in root.rglob("*"):
        if path.is_file() and path.relative_to(root) not in preserved:
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
        changed, jobs, static, preserved, t0, t1 = _prepare(fresh, annotation, start, end)
        transaction_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12]
        transaction = Path(tempfile.mkdtemp(prefix=f".kirigami-trim-{source.name}-", dir=source.parent))
        prepared = transaction / "episode"
        committed = False
        prepared_identity = None
        warnings = []
        record = dict(schema="kirigami_trim_v1", id=transaction_id, source_path=str(source),
                      start_frame=start, end_frame=end,
                      old_n_frames=fresh.n_frames, n_frames=end - start,
                      t_start_unix_ms=t0, t_end_unix_ms=t1,
                      source_revision=fresh.source_revision)
        if preserved:
            record["preserved_camera_videos"] = [str(rel) for rel in preserved
                                                  if rel.suffix.lower() in PRESERVED_CAMERA_SUFFIXES]
        journal = transaction / "transaction.json"
        try:
            prepared.mkdir()
            prepared_identity = (prepared.stat().st_dev, prepared.stat().st_ino)
            if prepared.stat().st_dev != source.stat().st_dev:
                raise ValueError("Source and temporary directory must be on the same filesystem for atomic replacement")
            atomic_write_text(journal, json.dumps({**record, "state": "preparing"}, indent=2) + "\n")
            for directory in (p for p in source.rglob("*") if p.is_dir()):
                (prepared / directory.relative_to(source)).mkdir(parents=True, exist_ok=True)
            for rel in static:
                (prepared / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / rel, prepared / rel)
            for rel in preserved:
                # The staging directory is on the same filesystem. Reuse the
                # existing inode so multi-GB IR files incur no decode or copy.
                os.link(source / rel, prepared / rel)
            for i, (rel, meta, (a, b)) in enumerate(jobs):
                if progress:
                    progress(f"Trim {rel}", 0.8 * i / max(len(jobs), 1))
                cut_video(source / rel, prepared / rel, a, b, meta)
            for rel, data in changed.items():
                _write_data(prepared / rel, data)
            adjusted = trim_annotation(annotation, start, end)
            adjusted.source_revision = source_revision(prepared)
            save_annotation(prepared, adjusted)
            history_path = source / "annotations/trim_history.json"
            history = json.loads(history_path.read_text()) if history_path.exists() else []
            if not isinstance(history, list):
                raise ValueError("Invalid existing trim history")
            atomic_write_text(prepared / "annotations/trim_history.json", json.dumps([*history, record], indent=2) + "\n")
            if progress:
                progress("Validate retained observations and timestamps", 0.85)
            verified = _load_episode(prepared)
            validate_episode_for_export(verified)
            if verified.n_frames != end - start or not np.array_equal(verified.base_ts, fresh.base_ts[start:end]):
                raise ValueError("Prepared source failed frame/timestamp validation")
            for key, expected in changed[Path("camera/timestamp.pkl")].items():
                if not np.array_equal(verified.cam_ts[key], expected):
                    raise ValueError(f"Prepared camera timestamps failed validation: {key}")
            for path in [*prepared.rglob("*"), prepared]:
                if path.relative_to(prepared) in preserved:
                    continue
                original = source / path.relative_to(prepared)
                if original.exists():
                    shutil.copystat(original, path)
            _sync_tree(prepared, set(preserved))
            atomic_write_text(journal, json.dumps({**record, "state": "prepared"}, indent=2) + "\n")
            for path in (transaction, source.parent):
                _sync_dir(path)
            if progress:
                progress("Write trimmed episode back to source", 0.95)
            if _inventory(source) != before:
                raise RuntimeError("Source files changed while trimming; original was not replaced")
            # This is the only operation that changes the source dataset.
            _exchange_directories(source, prepared)
            committed = True
        finally:
            if not committed:
                # A signal can arrive after renameat2 succeeds but before the assignment.
                # Identify which side of the exchange we reached before cleanup.
                try:
                    identity = (source.stat().st_dev, source.stat().st_ino)
                    committed = prepared_identity is not None and identity == prepared_identity
                    if not committed and identity == before["."][:2]:
                        if prepared_identity is None or (prepared.stat().st_dev, prepared.stat().st_ino) == prepared_identity:
                            shutil.rmtree(transaction, ignore_errors=True)
                except OSError:
                    pass  # Ambiguous state: preserve both data and the recovery journal.
            if committed:
                try:
                    _sync_dir(source.parent)
                    _sync_dir(transaction)
                except OSError as exc:
                    warnings.append(f"Trim was applied, but directory synchronization failed: {exc}.")
                # After the exchange this temporary directory holds the old
                # episode. Remove it, including on a handled interruption;
                # successful trims never keep a source-data backup.
                try:
                    shutil.rmtree(transaction)
                except OSError as exc:
                    warnings.append(f"Trim was applied, but temporary data could not be removed at {transaction}: {exc}.")
                else:
                    try:
                        _sync_dir(source.parent)
                    except OSError as exc:
                        warnings.append(f"Trim was applied and temporary data removed, but directory synchronization failed: {exc}.")
        if progress:
            progress("Source trim complete", 1.0)
        return TrimResult(source, end - start, start, fresh.n_frames - end, "\n".join(warnings))
