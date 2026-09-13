"""YAM Ultra 2 dual-arm layout used by Kirigami."""

from __future__ import annotations

import os
from pathlib import Path


def _discover_yam_ultra_v2() -> Path:
    env = os.environ.get("KIRIGAMI_YAM_ULTRA_DIR")
    if env:
        return Path(env).expanduser()
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "models" / "yam_ultra" / "v2",
        repo_root.parent / "i2rt" / "i2rt" / "robot_models" / "arm" / "yam_ultra" / "v2",
    ]
    for path in candidates:
        if (path / "yam_ultra.urdf").exists():
            return path
    return candidates[0]


I2RT_YAM_ULTRA_V2 = _discover_yam_ultra_v2()
YAM_ULTRA_V2_URDF = I2RT_YAM_ULTRA_V2 / "yam_ultra.urdf"
YAM_ULTRA_V2_MESH_DIR = I2RT_YAM_ULTRA_V2 / "assets"

# Two identical arms facing +X, bases separated along Y.
BASE_SEPARATION_M = 0.46
LEFT_BASE_XYZ = (0.0, BASE_SEPARATION_M / 2.0, 0.0)
RIGHT_BASE_XYZ = (0.0, -BASE_SEPARATION_M / 2.0, 0.0)

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
FINGER_JOINTS = ("joint7", "joint8")
LINK_FRAMES = ("base", "link1", "link2", "link3", "link4", "link5", "gripper")

# Recorded q[6] is gripper stroke in metres (see linear_4310 gripper_stroke).
GRIPPER_STROKE_M = 0.096
FINGER_TRAVEL_M = 0.04695


def gripper_to_fingers(g: float) -> float:
    """Map recorded gripper opening [0, stroke] onto URDF prismatic finger joints."""
    frac = float(max(0.0, min(g / GRIPPER_STROKE_M, 1.0)))
    return -FINGER_TRAVEL_M * frac


def q7_to_cfg(q: object) -> dict[str, float]:
    arr = [float(v) for v in list(q)]
    cfg = {name: arr[i] if i < len(arr) else 0.0 for i, name in enumerate(ARM_JOINTS)}
    g = arr[6] if len(arr) > 6 else 0.0
    finger = gripper_to_fingers(g)
    cfg["joint7"] = finger
    cfg["joint8"] = finger
    return cfg
