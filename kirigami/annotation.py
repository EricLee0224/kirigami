"""Split marks, segments, and on-disk annotation sidecar."""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .loader import ANNOTATION_REL, LoadedEpisode
from .source_guard import assert_source_revision, episode_lock, source_revision

SCHEMA = "kirigami_annotation_v1"


@dataclass
class Segment:
    start_frame: int
    end_frame: int
    subtask: str = ""
    discard: bool = False

    @property
    def n_frames(self) -> int:
        return max(0, self.end_frame - self.start_frame)

    def as_dict(self) -> dict:
        return {
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "subtask": self.subtask,
            "discard": bool(self.discard),
        }


@dataclass
class Annotation:
    source_path: str
    source_episode_id: str
    n_frames: int
    marks: list[int] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    exported_at: str = ""
    output_root: str = ""
    schema: str = SCHEMA
    exported_signature: str = ""
    trim_start: int = 0
    trim_end: int | None = None
    source_revision: str = field(default="", repr=False)

    @property
    def keep_range(self) -> tuple[int, int]:
        return self.trim_start, self.n_frames if self.trim_end is None else self.trim_end

    def exportable(self) -> list[Segment]:
        return [s for s in self.segments if not s.discard and s.n_frames > 0 and s.subtask.strip()]

    def missing_names(self) -> list[Segment]:
        return [s for s in self.segments if not s.discard and s.n_frames > 0 and not s.subtask.strip()]

    def signature(self) -> str:
        payload = [self.source_path, self.n_frames, self.marks, [s.as_dict() for s in self.segments]]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def invalidate_export(self) -> None:
        self.exported_at = ""
        self.exported_signature = ""


def validate_subtask_name(name: str) -> str:
    name = name.strip()
    if not name or name in {".", ".."} or any(c in name for c in "/\\") or any(ord(c) < 32 for c in name):
        raise ValueError("Subtask must be a folder name, without /, \\, . or .. path components.")
    return name


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as f:
            temporary = Path(f.name)
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def normalize_marks(marks: list[int], n_frames: int) -> list[int]:
    uniq = sorted({int(m) for m in marks if 0 < int(m) < n_frames})
    return uniq


def bounds_from_marks(marks: list[int], n_frames: int) -> list[int]:
    return [0, *normalize_marks(marks, n_frames), n_frames]


def rebuild_segments(
    marks: list[int],
    n_frames: int,
    existing: list[Segment] | None = None,
) -> list[Segment]:
    """Rebuild [mark_i, mark_{i+1}) segments, preserving labels when ranges match."""
    bounds = bounds_from_marks(marks, n_frames)
    old = list(existing or [])
    by_range = {(s.start_frame, s.end_frame): s for s in old}
    by_start = {s.start_frame: s for s in old}
    segs: list[Segment] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        prev = by_range.get((a, b)) or by_start.get(a)
        if prev is not None:
            segs.append(Segment(start_frame=a, end_frame=b, subtask=prev.subtask, discard=prev.discard))
        else:
            segs.append(Segment(start_frame=a, end_frame=b))
    return segs


def annotation_from_episode(episode: LoadedEpisode, existing: Annotation | None = None) -> Annotation:
    if existing is None:
        marks: list[int] = []
        segs = rebuild_segments(marks, episode.n_frames)
        return Annotation(
            source_path=str(episode.ref.path.resolve()),
            source_episode_id=episode.ref.episode_id,
            n_frames=episode.n_frames,
            marks=marks,
            segments=segs,
            source_revision=episode.source_revision,
        )
    marks = normalize_marks(existing.marks, episode.n_frames)
    segs = rebuild_segments(marks, episode.n_frames, existing.segments)
    existing.n_frames = episode.n_frames
    existing.source_path = str(episode.ref.path.resolve())
    existing.source_episode_id = episode.ref.episode_id
    existing.marks = marks
    existing.segments = segs
    existing.source_revision = episode.source_revision
    start, end = existing.keep_range
    if not 0 <= start < end <= episode.n_frames:
        existing.trim_start, existing.trim_end = 0, None
    return existing


def trim_annotation(ann: Annotation, start: int, end: int) -> Annotation:
    """Clip existing labeled intervals and rebase to the retained first frame."""
    if not 0 <= start < end <= ann.n_frames:
        raise ValueError("Keep range must contain at least one frame within the episode")
    from copy import deepcopy
    out = deepcopy(ann)
    out.n_frames = end - start
    out.marks = [m - start for m in ann.marks if start < m < end]
    out.segments = [Segment(max(s.start_frame, start) - start, min(s.end_frame, end) - start,
                            s.subtask, s.discard)
                    for s in ann.segments if s.end_frame > start and s.start_frame < end]
    out.trim_start, out.trim_end = 0, None
    out.source_revision = ""
    out.invalidate_export()
    return out


def set_marks(ann: Annotation, marks: list[int]) -> Annotation:
    marks = normalize_marks(marks, ann.n_frames)
    if marks != ann.marks:
        ann.invalidate_export()
    ann.marks = marks
    ann.segments = rebuild_segments(ann.marks, ann.n_frames, ann.segments)
    return ann


def add_mark(ann: Annotation, frame: int) -> bool:
    frame = int(frame)
    if frame <= 0 or frame >= ann.n_frames:
        return False
    if frame in ann.marks:
        return False
    set_marks(ann, [*ann.marks, frame])
    return True


def remove_mark(ann: Annotation, frame: int) -> bool:
    frame = int(frame)
    if frame not in ann.marks:
        return False
    set_marks(ann, [m for m in ann.marks if m != frame])
    return True


def nearest_mark(ann: Annotation, frame: int, max_dist: int = 8) -> int | None:
    if not ann.marks:
        return None
    best = min(ann.marks, key=lambda m: abs(m - frame))
    if abs(best - frame) <= max_dist:
        return best
    return None


def load_annotation(ep_dir: Path) -> Annotation | None:
    path = Path(ep_dir) / ANNOTATION_REL
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    segs = [Segment(**s) for s in data.get("segments") or []]
    ann = Annotation(
        source_path=data.get("source_path", str(Path(ep_dir).resolve())),
        source_episode_id=data.get("source_episode_id", Path(ep_dir).name),
        n_frames=int(data.get("n_frames") or 0),
        marks=list(data.get("marks") or []),
        segments=segs,
        exported_at=data.get("exported_at", ""),
        output_root=data.get("output_root", ""),
        schema=data.get("schema", SCHEMA),
        exported_signature=data.get("exported_signature", ""),
        trim_start=int(data.get("trim_start", 0)),
        trim_end=int(data["trim_end"]) if data.get("trim_end") is not None else None,
        source_revision=source_revision(ep_dir),
    )
    # Legacy sidecars predate signatures; establish a baseline before editing.
    if ann.exported_at and not ann.exported_signature:
        ann.exported_signature = ann.signature()
    return ann


def save_annotation(ep_dir: Path, ann: Annotation) -> Path:
    with episode_lock(ep_dir):
        if source_revision(ep_dir):
            assert_source_revision(ep_dir, ann.source_revision)
        return _save_annotation(ep_dir, ann)


def _save_annotation(ep_dir: Path, ann: Annotation) -> Path:
    path = Path(ep_dir) / ANNOTATION_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    if ann.exported_at and ann.exported_signature != ann.signature():
        ann.invalidate_export()
    payload = {
        "schema": ann.schema,
        "source_path": ann.source_path,
        "source_episode_id": ann.source_episode_id,
        "n_frames": ann.n_frames,
        "marks": list(ann.marks),
        "segments": [s.as_dict() for s in ann.segments],
        "exported_at": ann.exported_at,
        "output_root": ann.output_root,
        "exported_signature": ann.exported_signature,
        "trim_start": ann.trim_start,
        "trim_end": ann.trim_end,
    }
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def mark_exported(ann: Annotation, output_root: Path) -> None:
    ann.exported_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ann.output_root = str(Path(output_root).resolve())
    ann.exported_signature = ann.signature()


def load_subtask_presets(path: Path) -> list[str]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    raw = data.get("subtasks") if isinstance(data, dict) else data
    if not raw:
        return []
    names: list[str] = []
    seen: set[str] = set()
    for item in raw:
        name = str(item).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def save_subtask_presets(path: Path, names: list[str]) -> None:
    path = Path(path)
    existing = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text()) or {}
        if isinstance(loaded, dict):
            existing = loaded
    existing["subtasks"] = names
    atomic_write_text(path, yaml.safe_dump(existing, sort_keys=False, allow_unicode=True))


def add_preset(names: list[str], new_name: str) -> list[str]:
    name = new_name.strip()
    if not name or name in names:
        return names
    return [*names, name]
