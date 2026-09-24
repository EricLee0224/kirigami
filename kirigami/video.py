"""OpenCV video bank with reasonably robust HEVC seeking."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .camera import validate_camera_timestamps
from .loader import nearest_indices


class VideoBank:
    def __init__(self, videos: dict[str, Path], timestamps: dict[str, np.ndarray] | None = None):
        self.paths = {k: Path(p) for k, p in videos.items()}
        self.frame_indices = {}
        if timestamps is not None:
            base = validate_camera_timestamps("base_0", timestamps.get("base_0", []))
            self.frame_indices = {key: nearest_indices(validate_camera_timestamps(key, timestamps.get(key, [])), base)
                                  for key in self.paths}
        self.caps: dict[str, cv2.VideoCapture] = {}
        self.idx: dict[str, int] = {}
        self._last: dict[str, np.ndarray | None] = {}
        for key, path in self.paths.items():
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"cannot open video: {path}")
            self.caps[key] = cap
            self.idx[key] = -1
            self._last[key] = None

    def close(self) -> None:
        for cap in self.caps.values():
            cap.release()
        self.caps.clear()
        self.idx.clear()
        self._last.clear()

    def __enter__(self) -> "VideoBank":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def read(self, frame_idx: int) -> dict[str, np.ndarray | None]:
        return {key: self._read_one(key, int(self.frame_indices[key][frame_idx])
                                   if key in self.frame_indices and 0 <= frame_idx < len(self.frame_indices[key]) else frame_idx)
                for key in self.caps}

    def _read_one(self, key: str, target: int) -> np.ndarray | None:
        cap = self.caps[key]
        last = self.idx[key]
        if target < 0:
            return None
        if last == target and self._last[key] is not None:
            return self._last[key]
        frame = None
        if last >= 0 and target == last + 1:
            ok, frame = cap.read()
        elif last >= 0 and last < target <= last + 20:
            ok = False
            for i in range(target - last):
                ok, frame = cap.read()
                if not ok:
                    self.idx[key] = last + i
                    self._last[key] = None
                    return None
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(target))
            ok, frame = cap.read()
        if ok and frame is not None:
            self.idx[key] = target
            self._last[key] = frame
            return frame
        return None


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
