"""Coordinate source reads/writes across desktop windows on this host."""

from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import tempfile


class SourceChangedError(RuntimeError):
    pass


def source_revision(path: Path) -> str:
    timestamp = Path(path) / "camera/timestamp.pkl"
    if not timestamp.exists():
        return ""
    st = timestamp.stat()
    return f"{st.st_dev}:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}"


def assert_source_revision(path: Path, expected: str) -> None:
    if expected and source_revision(path) != expected:
        raise SourceChangedError("Source episode changed on disk. Reload it before saving or processing.")


@contextmanager
def episode_lock(path: Path, *, exclusive: bool = False):
    # Outside the episode, so a directory exchange cannot replace the lock inode.
    root = Path(tempfile.gettempdir()) / f"kirigami-locks-{os.getuid()}"
    root.mkdir(mode=0o700, exist_ok=True)
    key = hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()
    with open(root / f"{key}.lock", "a") as lock:
        try:
            mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(lock, mode | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This episode is being processed in another window. Wait for it to finish.") from exc
        yield
