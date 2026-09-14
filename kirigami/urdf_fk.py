"""Minimal URDF kinematics + visual mesh loader for one YAM arm."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def origin_matrix(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_to_matrix(rpy)
    T[:3, 3] = xyz
    return T


def joint_motion(jtype: str, axis: np.ndarray, q: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return T
    axis = axis / n
    if jtype == "prismatic":
        T[:3, 3] = axis * float(q)
        return T
    # Rodrigues
    x, y, z = axis
    c, s = np.cos(q), np.sin(q)
    C = 1.0 - c
    T[:3, :3] = np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )
    return T


def _xyz(elem, attr: str, default="0 0 0") -> np.ndarray:
    raw = (elem.get(attr) if elem is not None else None) or default
    return np.fromstring(raw, sep=" ", dtype=np.float64)


def _parse_origin(node) -> np.ndarray:
    origin = node.find("origin") if node is not None else None
    if origin is None:
        return np.eye(4, dtype=np.float64)
    return origin_matrix(_xyz(origin, "xyz"), _xyz(origin, "rpy"))


@dataclass
class JointSpec:
    name: str
    jtype: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


@dataclass
class VisualSpec:
    link: str
    origin: np.ndarray
    mesh_path: Path
    color: np.ndarray
    scale: np.ndarray


@dataclass(eq=False)
class MeshGeometry:
    vertices: np.ndarray
    faces: np.ndarray
    # Interleaved triangle positions/normals, uploaded once per OpenGL context.
    vertex_data: np.ndarray


@lru_cache(maxsize=32)
def _load_geometry(path: Path, mtime_ns: int, size: int) -> MeshGeometry:
    """Share original surfaces across both arms; file changes invalidate the cache."""
    mesh = trimesh.load(str(path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"No triangle geometry in {path}")
    # Split normals at CAD creases without removing or changing any triangles.
    smooth = trimesh.graph.smooth_shade(mesh, angle=np.radians(35))
    data = np.concatenate(
        [smooth.vertices[smooth.faces], smooth.vertex_normals[smooth.faces]], axis=2,
    ).reshape(-1, 6).astype(np.float32)
    return MeshGeometry(np.asarray(mesh.vertices), np.asarray(mesh.faces), data)


class UrdfArm:
    def __init__(self, urdf_path: Path, mesh_dir: Path | None, base_xyz: tuple[float, float, float]):
        self.urdf_path = Path(urdf_path)
        self.mesh_dir = Path(mesh_dir) if mesh_dir else self.urdf_path.parent / "assets"
        self.base = np.eye(4, dtype=np.float64)
        self.base[:3, 3] = np.asarray(base_xyz, dtype=np.float64)
        self.joints: dict[str, JointSpec] = {}
        self.children: dict[str, list[str]] = {}
        self.visuals: list[VisualSpec] = []
        self.meshes: list[tuple[VisualSpec, MeshGeometry]] = []
        self.transforms: dict[str, np.ndarray] = {}
        self._parse()
        self._load_meshes()
        self.update({})

    def _parse(self) -> None:
        root = ET.parse(self.urdf_path).getroot()
        for joint in root.findall("joint"):
            axis_el = joint.find("axis")
            spec = JointSpec(
                name=joint.get("name") or "",
                jtype=joint.get("type") or "fixed",
                parent=(joint.find("parent").get("link") if joint.find("parent") is not None else ""),
                child=(joint.find("child").get("link") if joint.find("child") is not None else ""),
                origin=_parse_origin(joint),
                axis=_xyz(axis_el, "xyz", "0 0 1") if axis_el is not None else np.array([0.0, 0.0, 1.0]),
            )
            self.joints[spec.name] = spec
            self.children.setdefault(spec.parent, []).append(spec.name)
        for link in root.findall("link"):
            name = link.get("name") or ""
            for visual in link.findall("visual"):
                geom = visual.find("geometry")
                mesh = geom.find("mesh") if geom is not None else None
                if mesh is None:
                    continue
                filename = mesh.get("filename") or ""
                path = Path(filename)
                if not path.is_absolute():
                    path = self.mesh_dir / path.name
                color = np.array([180.0, 180.0, 180.0])
                mat = visual.find("material")
                if mat is not None:
                    col = mat.find("color")
                    if col is not None and col.get("rgba"):
                        rgba = np.fromstring(col.get("rgba"), sep=" ", dtype=np.float64)
                        if rgba.size >= 3:
                            color = rgba[:3] * (255.0 if rgba.max() <= 1.0 else 1.0)
                self.visuals.append(VisualSpec(name, _parse_origin(visual), path, color, _xyz(mesh, "scale", "1 1 1")))

    def _load_meshes(self) -> None:
        for vis in self.visuals:
            path = vis.mesh_path.resolve()
            stat = path.stat()  # Missing parts must not silently produce an incomplete model.
            geometry = _load_geometry(path, stat.st_mtime_ns, stat.st_size)
            self.meshes.append((vis, geometry))
        if not self.meshes:
            raise ValueError("URDF contains no visual meshes")

    def update(self, cfg: dict[str, float]) -> None:
        self.transforms = {"base": self.base.copy()}
        pending = list(self.children.get("base", []))
        seen = set()
        while pending:
            jname = pending.pop(0)
            if jname in seen:
                continue
            seen.add(jname)
            spec = self.joints[jname]
            if spec.parent not in self.transforms:
                pending.append(jname)
                continue
            q = float(cfg.get(spec.name, 0.0))
            T = self.transforms[spec.parent] @ spec.origin @ joint_motion(spec.jtype, spec.axis, q)
            self.transforms[spec.child] = T
            pending.extend(self.children.get(spec.child, []))

    def link_xyz(self, name: str) -> np.ndarray | None:
        T = self.transforms.get(name)
        if T is None:
            return None
        return T[:3, 3].astype(np.float32)

    def visual_instances(self):
        """Static local geometry and its current model matrix; no per-frame mesh copies."""
        for vis, geometry in self.meshes:
            T_link = self.transforms.get(vis.link)
            if T_link is None:
                continue
            Tw = T_link @ vis.origin @ np.diag([*vis.scale, 1.0])
            yield geometry, Tw, vis.color

    def world_meshes(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        out = []
        for geometry, Tw, color in self.visual_instances():
            verts, faces = geometry.vertices, geometry.faces
            ones = np.ones((len(verts), 1), dtype=np.float64)
            world = (Tw @ np.hstack([verts, ones]).T).T[:, :3]
            out.append((world, faces, color))
        return out
