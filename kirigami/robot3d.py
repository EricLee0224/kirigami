"""Dual YAM Ultra 2 player: two parallel arms 46 cm apart."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from .urdf_fk import UrdfArm
from .yam_ultra import (
    ARM_JOINTS,
    LEFT_BASE_XYZ,
    LINK_FRAMES,
    RIGHT_BASE_XYZ,
    YAM_ULTRA_V2_MESH_DIR,
    YAM_ULTRA_V2_URDF,
    q7_to_cfg,
)


@dataclass
class JointMap:
    left: list[str] = field(default_factory=lambda: list(ARM_JOINTS))
    right: list[str] = field(default_factory=lambda: list(ARM_JOINTS))
    left_gripper: str | None = "gripper"
    right_gripper: str | None = "gripper"


@dataclass
class Robot3DConfig:
    urdf: Path | None = YAM_ULTRA_V2_URDF
    mesh_dir: Path | None = YAM_ULTRA_V2_MESH_DIR
    joint_map: JointMap = field(default_factory=JointMap)
    left_base: tuple[float, float, float] = LEFT_BASE_XYZ
    right_base: tuple[float, float, float] = RIGHT_BASE_XYZ

    @classmethod
    def from_yaml(cls, path: Path) -> "Robot3DConfig":
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text()) or {}
        root = Path(path).resolve().parent
        jm = data.get("joint_map") or {}
        sep = float(data.get("base_separation_m") or 0.46)

        def _resolve(value) -> Path | None:
            if not value:
                return None
            p = Path(value)
            if not p.is_absolute():
                p = root / p
            return p

        return cls(
            urdf=_resolve(data.get("urdf") or YAM_ULTRA_V2_URDF),
            mesh_dir=_resolve(data.get("mesh_dir") or YAM_ULTRA_V2_MESH_DIR),
            joint_map=JointMap(
                left=list(jm.get("left") or ARM_JOINTS),
                right=list(jm.get("right") or ARM_JOINTS),
                left_gripper=jm.get("left_gripper", "gripper"),
                right_gripper=jm.get("right_gripper", "gripper"),
            ),
            left_base=(0.0, sep / 2.0, 0.0),
            right_base=(0.0, -sep / 2.0, 0.0),
        )


class _OrbitCamera:
    def __init__(self) -> None:
        self.eye = np.array([0.90, 0.62, 0.52], dtype=np.float64)
        self.target = np.array([0.10, 0.0, 0.14], dtype=np.float64)
        self.up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.fov_deg = 48.0

    def project(self, pts: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """Perspective project Nx3 world points -> Nx2 pixels and depth."""
        f = self.target - self.eye
        f = f / (np.linalg.norm(f) + 1e-9)
        s = np.cross(f, self.up)
        s = s / (np.linalg.norm(s) + 1e-9)
        u = np.cross(s, f)
        rel = pts - self.eye
        cam = np.stack([rel @ s, rel @ u, rel @ f], axis=1)
        z = np.maximum(cam[:, 2], 1e-4)
        foc = 0.5 * height / np.tan(np.deg2rad(self.fov_deg) * 0.5)
        u_px = width * 0.5 + foc * cam[:, 0] / z
        v_px = height * 0.5 - foc * cam[:, 1] / z
        return np.stack([u_px, v_px], axis=1), z


class _ArmModel:
    def __init__(self, urdf_path: Path, mesh_dir: Path | None, base_xyz: tuple[float, float, float]):
        self.arm = UrdfArm(urdf_path, mesh_dir, base_xyz)

    def update(self, q7) -> None:
        self.arm.update(q7_to_cfg(q7))

    def frame_xyz(self, link: str) -> np.ndarray | None:
        return self.arm.link_xyz(link)

    def chain_xyz(self) -> np.ndarray:
        pts = []
        for name in LINK_FRAMES:
            xyz = self.frame_xyz(name)
            if xyz is not None:
                pts.append(xyz)
        if not pts:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack(pts, axis=0)

    def world_meshes(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return self.arm.world_meshes()


class Robot3DPlayer:
    def __init__(self, config: Robot3DConfig | None = None):
        self.config = config or Robot3DConfig()
        self.available = False
        self.status = "等待 URDF"
        self._left: _ArmModel | None = None
        self._right: _ArmModel | None = None
        self._cam = _OrbitCamera()
        self._left_trail = np.zeros((0, 3), dtype=np.float32)
        self._right_trail = np.zeros((0, 3), dtype=np.float32)
        if self.config.urdf is not None:
            err = self.load(self.config.urdf, self.config.mesh_dir, self.config.joint_map)
            if err:
                self.status = err

    def load(self, urdf_path: Path, mesh_dir: Path | None, joint_map: JointMap | None = None) -> str:
        urdf_path = Path(urdf_path)
        if not urdf_path.exists():
            self.available = False
            self.status = f"URDF 不存在: {urdf_path}"
            return self.status
        if joint_map is not None:
            self.config.joint_map = joint_map
        self.config.urdf = urdf_path
        self.config.mesh_dir = mesh_dir
        try:
            self._left = _ArmModel(urdf_path, mesh_dir, self.config.left_base)
            self._right = _ArmModel(urdf_path, mesh_dir, self.config.right_base)
        except Exception as exc:
            self.available = False
            self.status = f"URDF 加载失败: {exc}"
            return self.status
        self.available = True
        self.status = "YAM Ultra 2 x2  |  46cm"
        self.reset_trail()
        return ""

    def reset_trail(self) -> None:
        self._left_trail = np.zeros((0, 3), dtype=np.float32)
        self._right_trail = np.zeros((0, 3), dtype=np.float32)

    def bake_trails(self, q_left: np.ndarray, q_right: np.ndarray) -> None:
        if not self.available:
            return
        left, right = [], []
        n = min(len(q_left), len(q_right))
        step = max(1, n // 400)
        for i in range(0, n, step):
            self.update(q_left[i], q_right[i], record_trail=False)
            lx, rx = self.ee_xyz()
            if lx is not None:
                left.append(lx)
            if rx is not None:
                right.append(rx)
        self._left_trail = np.stack(left, axis=0) if left else np.zeros((0, 3), dtype=np.float32)
        self._right_trail = np.stack(right, axis=0) if right else np.zeros((0, 3), dtype=np.float32)

    def update(self, q_left: np.ndarray, q_right: np.ndarray, record_trail: bool = False) -> None:
        if not self.available or self._left is None or self._right is None:
            return
        self._left.update(q_left)
        self._right.update(q_right)
        if record_trail:
            lx, rx = self.ee_xyz()
            if lx is not None:
                self._left_trail = np.vstack([self._left_trail, lx.reshape(1, 3)]) if self._left_trail.size else lx.reshape(1, 3)
            if rx is not None:
                self._right_trail = np.vstack([self._right_trail, rx.reshape(1, 3)]) if self._right_trail.size else rx.reshape(1, 3)

    def ee_xyz(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        left = self._left.frame_xyz(self.config.joint_map.left_gripper or "gripper") if self._left else None
        right = self._right.frame_xyz(self.config.joint_map.right_gripper or "gripper") if self._right else None
        return left, right

    def trails(self) -> tuple[np.ndarray, np.ndarray]:
        return self._left_trail, self._right_trail

    def render_rgb(self, width: int = 480, height: int = 360) -> np.ndarray | None:
        if not self.available or self._left is None or self._right is None:
            return None
        img = np.full((height, width, 3), 22, dtype=np.uint8)
        self._draw_grid(img)
        tris: list[tuple[float, np.ndarray, tuple[int, int, int]]] = []
        for arm, tint in ((self._left, (0.55, 0.75, 1.0)), (self._right, (1.0, 0.72, 0.40))):
            for verts, faces, color in arm.world_meshes():
                shaded = np.clip(color * np.array(tint), 0, 255).astype(np.int32)
                proj, depth = self._cam.project(verts, width, height)
                v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
                normals = np.cross(v1 - v0, v2 - v0)
                view = self._cam.eye - v0
                shade = np.clip((normals * view).sum(axis=1), 0.0, None)
                shade = 0.35 + 0.65 * shade / (np.linalg.norm(normals, axis=1) * np.linalg.norm(view, axis=1) + 1e-9)
                for i, face in enumerate(faces):
                    pts = proj[face]
                    z = float(depth[face].mean())
                    if z <= 0 or not np.isfinite(pts).all():
                        continue
                    ip = np.round(pts).astype(np.int32)
                    lit = np.clip(shaded * float(shade[i]), 0, 255).astype(np.int32)
                    tris.append((z, ip, (int(lit[0]), int(lit[1]), int(lit[2]))))
        tris.sort(key=lambda item: item[0], reverse=True)
        for _, ip, color in tris:
            cv2.fillConvexPoly(img, ip, color, lineType=cv2.LINE_AA)

        for arm, color in ((self._left, (66, 165, 245)), (self._right, (255, 167, 38))):
            chain = arm.chain_xyz()
            if len(chain) >= 2:
                self._draw_polyline(img, chain, color, thickness=2)
        self._draw_polyline(img, self._left_trail, (80, 190, 255), thickness=1)
        self._draw_polyline(img, self._right_trail, (255, 190, 80), thickness=1)
        for xyz, color in zip(self.ee_xyz(), ((40, 120, 255), (255, 140, 30))):
            if xyz is None:
                continue
            uv, _ = self._cam.project(xyz.reshape(1, 3), width, height)
            cv2.circle(img, (int(uv[0, 0]), int(uv[0, 1])), 5, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            img,
            self.status,
            (10, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return img

    def render_trail_preview(self, width: int = 480, height: int = 360) -> np.ndarray | None:
        return self.render_rgb(width, height)

    def _draw_grid(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        xs = np.linspace(-0.35, 0.55, 10)
        ys = np.linspace(-0.40, 0.40, 9)
        color = (45, 45, 48)
        for x in xs:
            pts = np.array([[x, ys[0], 0.0], [x, ys[-1], 0.0]], dtype=np.float64)
            self._draw_polyline(img, pts, color, 1)
        for y in ys:
            pts = np.array([[xs[0], y, 0.0], [xs[-1], y, 0.0]], dtype=np.float64)
            self._draw_polyline(img, pts, color, 1)

    def _draw_polyline(self, img: np.ndarray, xyz: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
        if xyz is None or len(xyz) < 2:
            return
        uv, z = self._cam.project(np.asarray(xyz, dtype=np.float64), img.shape[1], img.shape[0])
        pts = []
        for (u, v), depth in zip(uv, z):
            if depth <= 0 or not np.isfinite(u) or not np.isfinite(v):
                if len(pts) >= 2:
                    cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)
                pts = []
                continue
            pts.append((int(u), int(v)))
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)

    def start_viser(self) -> str | None:
        return None
