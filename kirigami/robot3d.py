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
        self.fov_deg = 42.0

    def project(self, pts: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """Perspective project Nx3 world points -> Nx2 pixels and depth."""
        f = self.target - self.eye
        f = f / (np.linalg.norm(f) + 1e-9)
        s = np.cross(f, self.up)
        s = s / (np.linalg.norm(s) + 1e-9)
        u = np.cross(s, f)
        rel = pts - self.eye
        cam = np.stack([rel @ s, rel @ u, rel @ f], axis=1)
        depth = cam[:, 2]
        z = np.maximum(depth, 1e-4)
        foc = 0.5 * height / np.tan(np.deg2rad(self.fov_deg) * 0.5)
        u_px = width * 0.5 + foc * cam[:, 0] / z
        v_px = height * 0.5 - foc * cam[:, 1] / z
        return np.stack([u_px, v_px], axis=1), depth


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
        self.material_mode = "white"
        self._stage_key = None
        self._stage_image = None
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
        separation_cm = np.linalg.norm(np.asarray(self.config.left_base) - np.asarray(self.config.right_base)) * 100
        self.status = f"YAM Ultra 2 × 2 · {separation_cm:g} cm base spacing"
        self.reset_trail()
        idle = np.array([0.0, 0.55, 0.65, 0.0, -0.4, 0.0, 0.035])
        self.update(idle, idle)
        return ""

    def orbit(self, dx: float, dy: float) -> None:
        offset = self._cam.eye - self._cam.target
        radius = float(np.linalg.norm(offset))
        azimuth = np.arctan2(offset[1], offset[0]) - dx * 0.007
        elevation = np.clip(np.arcsin(offset[2] / radius) + dy * 0.006, 0.08, 1.42)
        self._cam.eye = self._cam.target + radius * np.array([
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ])

    def zoom(self, steps: float) -> None:
        offset = self._cam.eye - self._cam.target
        radius = float(np.linalg.norm(offset))
        new_radius = np.clip(radius * np.exp(-np.clip(steps, -20, 20) * 0.12), 0.35, 3.0)
        self._cam.eye = self._cam.target + offset * new_radius / radius

    def reset_view(self) -> None:
        self._cam = _OrbitCamera()

    def _studio_background(self, width: int, height: int) -> np.ndarray:
        key = (width, height, tuple(self._cam.eye), tuple(self._cam.target), self.config.left_base, self.config.right_base)
        if key != self._stage_key:
            top, bottom = np.array([231, 236, 242]), np.array([247, 249, 251])
            rows = np.linspace(top, bottom, height).astype(np.uint8)
            image = np.repeat(rows[:, None, :], width, axis=1)
            self._draw_grid(image)
            shadow = np.zeros((height, width), dtype=np.uint8)
            for base in (self.config.left_base, self.config.right_base):
                angles = np.linspace(0, 2 * np.pi, 48)
                ring = np.column_stack([base[0] + 0.12 * np.cos(angles), base[1] + 0.12 * np.sin(angles), np.full(48, base[2] + 0.001)])
                uv, depth = self._cam.project(ring, width, height)
                if (depth > 0).all():
                    cv2.fillConvexPoly(shadow, np.round(uv).astype(np.int32), 45, cv2.LINE_AA)
            shadow = cv2.GaussianBlur(shadow, (0, 0), max(2.0, height / 70))
            image = np.maximum(image.astype(np.int16) - shadow[:, :, None], 0).astype(np.uint8)
            self._stage_image, self._stage_key = image, key
        return self._stage_image.copy()

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

    @property
    def camera(self) -> _OrbitCamera:
        return self._cam

    def visual_instances(self):
        if self.available:
            for arm in (self._left, self._right):
                yield from arm.arm.visual_instances()

    @property
    def triangle_count(self) -> int:
        return sum(len(geometry.faces) for geometry, _, _ in self.visual_instances())

    def render_rgb(self, width: int = 480, height: int = 360) -> np.ndarray | None:
        """CPU diagnostic preview. The desktop viewport renders directly with OpenGL."""
        if not self.available or self._left is None or self._right is None:
            return None
        img = self._studio_background(width, height)
        tris: list[tuple[float, np.ndarray, tuple[int, int, int]]] = []
        key_light = np.array([0.35, -0.4, 0.85])
        key_light /= np.linalg.norm(key_light)
        fill_light = np.array([-0.5, 0.7, 0.5])
        fill_light /= np.linalg.norm(fill_light)
        for arm, tint in ((self._left, (0.55, 0.75, 1.0)), (self._right, (1.0, 0.72, 0.40))):
            for verts, faces, color in arm.world_meshes():
                shaded = np.array([244, 245, 247]) if self.material_mode == "white" else np.clip(color * np.array(tint), 0, 255)
                proj, depth = self._cam.project(verts, width, height)
                v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
                normals = np.cross(v1 - v0, v2 - v0)
                view = self._cam.eye - v0
                unit_normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)
                shade = 0.63 + 0.29 * np.maximum(unit_normals @ key_light, 0) + 0.14 * np.maximum(unit_normals @ fill_light, 0)
                # Vectorize projection and lighting; only rasterization needs a loop.
                points = proj[faces]
                depths = depth[faces]
                visible = (depths > 1e-4).all(axis=1) & np.isfinite(points).all(axis=(1, 2))
                visible &= (normals * view).sum(axis=1) > 0
                visible &= (points[:, :, 0].max(axis=1) >= 0) & (points[:, :, 0].min(axis=1) < width)
                visible &= (points[:, :, 1].max(axis=1) >= 0) & (points[:, :, 1].min(axis=1) < height)
                pixels = np.round(points[visible]).astype(np.int32)
                lighting = np.clip(shade[visible, None] * shaded, 0, 255).astype(np.uint8)
                for z, ip, lit in zip(depths[visible].mean(axis=1), pixels, lighting.tolist()):
                    tris.append((float(z), ip, tuple(lit)))
        tris.sort(key=lambda item: item[0], reverse=True)
        for _, ip, color in tris:
            cv2.fillConvexPoly(img, ip, color, lineType=cv2.LINE_AA)

        self._draw_polyline(img, self._left_trail, (70, 145, 130), thickness=1)
        self._draw_polyline(img, self._right_trail, (187, 148, 93), thickness=1)
        for xyz, color in zip(self.ee_xyz(), ((22, 132, 119), (189, 135, 71))):
            if xyz is None:
                continue
            uv, depth = self._cam.project(xyz.reshape(1, 3), width, height)
            if depth[0] > 0 and (np.abs(uv) < 100000).all():
                cv2.circle(img, (int(uv[0, 0]), int(uv[0, 1])), 3, color, -1, lineType=cv2.LINE_AA)
        for base, label, color in ((self.config.left_base, "L", (22, 132, 119)), (self.config.right_base, "R", (189, 135, 71))):
            point = np.asarray(base) + [-0.075, 0, 0.002]
            uv, depth = self._cam.project(point.reshape(1, 3), width, height)
            if depth[0] > 0 and (np.abs(uv) < 100000).all():
                x, y = np.round(uv[0]).astype(int)
                cv2.circle(img, (x, y), 7, color, -1, cv2.LINE_AA)
                cv2.putText(img, label, (x - 3, y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def render_trail_preview(self, width: int = 480, height: int = 360) -> np.ndarray | None:
        return self.render_rgb(width, height)

    def _draw_grid(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        xs = np.linspace(-0.35, 0.55, 10)
        ys = np.linspace(-0.40, 0.40, 9)
        color = (215, 223, 231)
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
