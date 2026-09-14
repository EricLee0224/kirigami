"""Native desktop OpenGL viewport. Full URDF surfaces stay resident on the GPU."""

from __future__ import annotations

import numpy as np
import shiboken6
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QMatrix4x4, QPainter, QSurfaceFormat, QVector3D
from PySide6.QtOpenGL import QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram, QOpenGLVertexArrayObject
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QSizePolicy

from ..robot3d import Robot3DPlayer

GL_FLOAT = 0x1406
GL_TRIANGLES = 0x0004
GL_LINE_STRIP = 0x0003
GL_DEPTH_TEST = 0x0B71
GL_BLEND = 0x0BE2
GL_CULL_FACE = 0x0B44
GL_MULTISAMPLE = 0x809D
GL_LEQUAL = 0x0203
GL_COLOR_BUFFER_BIT = 0x4000
GL_DEPTH_BUFFER_BIT = 0x0100
GL_RENDERER = 0x1F01

VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
uniform mat4 model;
uniform mat4 viewProjection;
out vec3 worldPosition;
out vec3 worldNormal;
void main() {
    vec4 p = model * vec4(position, 1.0);
    worldPosition = p.xyz;
    worldNormal = transpose(inverse(mat3(model))) * normal;
    gl_Position = viewProjection * p;
}
"""

FRAGMENT_SHADER = """#version 330 core
in vec3 worldPosition;
in vec3 worldNormal;
uniform vec3 eye;
uniform vec3 material;
uniform vec3 leftBase;
uniform vec3 rightBase;
uniform int surface; // 0: robot, 1: studio floor, 2: trajectory
out vec4 fragColor;
void main() {
    if (surface == 1) {
        vec2 uv = worldPosition.xy / 0.10;
        vec2 grid = abs(fract(uv - 0.5) - 0.5) / max(fwidth(uv), vec2(0.0001));
        float line = 1.0 - min(min(grid.x, grid.y), 1.0);
        float fade = exp(-0.7 * dot(worldPosition.xy, worldPosition.xy));
        float dl = length(worldPosition.xy - leftBase.xy);
        float dr = length(worldPosition.xy - rightBase.xy);
        float shadow = exp(-dl * dl / 0.005) + exp(-dr * dr / 0.005);
        vec3 color = mix(vec3(0.91, 0.93, 0.95), vec3(0.96, 0.97, 0.98), fade);
        color -= line * 0.065 * fade + shadow * 0.16;
        vec2 markerL = worldPosition.xy - leftBase.xy - vec2(-0.085, 0.0);
        vec2 markerR = worldPosition.xy - rightBase.xy - vec2(-0.085, 0.0);
        float aa = max(length(fwidth(worldPosition.xy)), 0.0001);
        color = mix(color, vec3(0.09, 0.52, 0.47), 1.0 - smoothstep(0.010-aa, 0.010+aa, length(markerL)));
        color = mix(color, vec3(0.74, 0.53, 0.28), 1.0 - smoothstep(0.010-aa, 0.010+aa, length(markerR)));
        fragColor = vec4(color, 1.0);
    } else if (surface == 2) {
        fragColor = vec4(material, 1.0);
    } else {
        vec3 n = normalize(worldNormal);
        if (!gl_FrontFacing) n = -n;
        vec3 v = normalize(eye - worldPosition);
        vec3 key = normalize(vec3(0.4, -0.6, 1.0));
        vec3 fill = normalize(vec3(-0.6, 0.7, 0.45));
        float light = 0.24 + 0.58 * max(dot(n, key), 0.0) + 0.16 * max(dot(n, fill), 0.0);
        float spec = pow(max(dot(n, normalize(key + v)), 0.0), 48.0) * 0.20;
        vec3 color = material * light + vec3(spec);
        fragColor = vec4(pow(clamp(color, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
    }
}
"""


def desktop_gl_format() -> QSurfaceFormat:
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSamples(4)
    return fmt


class _GpuMesh:
    def __init__(self, data: np.ndarray, program: QOpenGLShaderProgram):
        self.count = len(data)
        self.vao = QOpenGLVertexArrayObject()
        self.vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        if not self.vao.create() or not self.vbo.create():
            self.destroy()
            raise RuntimeError("Could not allocate OpenGL mesh buffers")
        self.vao.bind()
        self.vbo.bind()
        self.vbo.setUsagePattern(QOpenGLBuffer.UsagePattern.StaticDraw)
        raw = np.ascontiguousarray(data, dtype=np.float32).tobytes()
        self.vbo.allocate(raw, len(raw))
        program.enableAttributeArray(0)
        program.setAttributeBuffer(0, GL_FLOAT, 0, 3, 24)
        program.enableAttributeArray(1)
        program.setAttributeBuffer(1, GL_FLOAT, 12, 3, 24)
        self.vao.release()
        self.vbo.release()

    def draw(self, functions, mode=GL_TRIANGLES):
        self.vao.bind()
        functions.glDrawArrays(mode, 0, self.count)
        self.vao.release()

    def destroy(self):
        self.vao.destroy()
        self.vbo.destroy()


class Robot3DPanel(QOpenGLWidget):
    orbitRequested = Signal(float, float)
    zoomRequested = Signal(float)
    expandRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFormat(desktop_gl_format())
        self.setMinimumSize(240, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.player: Robot3DPlayer | None = None
        self.renderer = ""
        self.error = ""
        self.ready = False
        self._message = "等待 URDF"
        self._program = None
        self._uniforms = {}
        self._meshes = {}
        self._floor = None
        self._trails = []
        self._drag_pos = None
        self.rendered_frames = 0

    def set_player(self, player: Robot3DPlayer) -> None:
        self.player = player
        self._message = player.status
        self.update()

    def set_status(self, text: str) -> None:
        self._message = text
        self.update()

    def initializeGL(self) -> None:
        self.context().aboutToBeDestroyed.connect(self.cleanup)
        try:
            f = self.context().functions()
            self.renderer = f.glGetString(GL_RENDERER)
            self._program = QOpenGLShaderProgram()
            for kind, source in ((QOpenGLShader.ShaderTypeBit.Vertex, VERTEX_SHADER),
                                 (QOpenGLShader.ShaderTypeBit.Fragment, FRAGMENT_SHADER)):
                if not self._program.addShaderFromSourceCode(kind, source):
                    raise RuntimeError(self._program.log())
            if not self._program.link():
                raise RuntimeError(self._program.log())
            self._uniforms = {name: self._program.uniformLocation(name) for name in
                              ("model", "viewProjection", "eye", "material", "leftBase", "rightBase", "surface")}
            self._program.bind()
            floor = np.array([[-5, -5, 0], [5, -5, 0], [5, 5, 0],
                              [-5, -5, 0], [5, 5, 0], [-5, 5, 0]], dtype=np.float32)
            self._floor = _GpuMesh(np.column_stack([floor, np.tile([0, 0, 1], (6, 1))]), self._program)
            self._program.release()
            self.error = ""
        except Exception as exc:
            self.error = f"OpenGL 初始化失败: {exc}"

    def _uniform(self, name, value):
        self._program.setUniformValue(self._uniforms[name], value)

    def _sync_meshes(self, instances):
        geometries = {id(geometry): geometry for geometry, _, _ in instances}
        for key in self._meshes.keys() - geometries.keys():
            _, gpu = self._meshes.pop(key)
            gpu.destroy()
        for key, geometry in geometries.items():
            if key not in self._meshes:
                self._meshes[key] = (geometry, _GpuMesh(geometry.vertex_data, self._program))

    def paintGL(self) -> None:
        f = self.context().functions()
        f.glClearColor(0.91, 0.93, 0.95, 1.0)
        f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        player = self.player
        if not self.error and self._program and player and player.available:
            try:
                self._program.bind()
                f.glEnable(GL_DEPTH_TEST)
                f.glDepthFunc(GL_LEQUAL)
                f.glEnable(GL_MULTISAMPLE)
                f.glDisable(GL_BLEND)
                f.glDisable(GL_CULL_FACE)  # Preserve thin/two-sided CAD surfaces.
                camera = player.camera
                projection, view = QMatrix4x4(), QMatrix4x4()
                projection.perspective(camera.fov_deg, self.width() / max(self.height(), 1), 0.01, 20.0)
                view.lookAt(QVector3D(*camera.eye), QVector3D(*camera.target), QVector3D(*camera.up))
                self._uniform("viewProjection", projection * view)
                self._uniform("eye", QVector3D(*camera.eye))
                self._uniform("leftBase", QVector3D(*player.config.left_base))
                self._uniform("rightBase", QVector3D(*player.config.right_base))
                floor_model = QMatrix4x4()
                floor_model.translate(0, 0, min(player.config.left_base[2], player.config.right_base[2]) - 0.002)
                self._uniform("model", floor_model)
                self._uniform("surface", 1)
                self._floor.draw(f)

                instances = list(player.visual_instances())
                self._sync_meshes(instances)
                self._uniform("surface", 0)
                for geometry, transform, color in instances:
                    self._uniform("model", QMatrix4x4(*transform.ravel().tolist()))
                    rgb = np.array([0.84, 0.86, 0.89]) if player.material_mode == "white" else np.power(color / 255.0, 2.2)
                    self._uniform("material", QVector3D(*rgb))
                    self._meshes[id(geometry)][1].draw(f)

                self._draw_trails(f)
                self._program.release()
                self.ready = True
                self.rendered_frames += 1
                self.setToolTip(f"{player.status}\n{self.renderer} · {player.triangle_count:,} triangles\nDouble-click to enlarge")
            except Exception as exc:
                self.error = f"OpenGL 渲染失败: {exc}"
                self.ready = False
                self._program.release()
        else:
            self.ready = False
        if self.error or not player or not player.available:
            painter = QPainter(self)
            painter.setPen(QColor("#64748b"))
            painter.drawText(self.rect().adjusted(20, 20, -20, -20),
                             Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                             self.error or self._message)
            painter.end()

    def _draw_trails(self, f):
        trails = self.player.trails()
        if len(self._trails) != 2 or any(old is not new for (old, _), new in zip(self._trails, trails)):
            for _, gpu in self._trails:
                if gpu:
                    gpu.destroy()
            self._trails = []
            for points in trails:
                data = np.column_stack([points, np.tile([0, 0, 1], (len(points), 1))])
                self._trails.append((points, _GpuMesh(data, self._program) if len(points) >= 2 else None))
        self._uniform("surface", 2)
        self._uniform("model", QMatrix4x4())
        for (_, gpu), color in zip(self._trails, ((0.20, 0.57, 0.52), (0.69, 0.51, 0.30))):
            if gpu:
                self._uniform("material", QVector3D(*color))
                gpu.draw(f, GL_LINE_STRIP)

    def cleanup(self) -> None:
        """Destroy GPU objects while their owning context is still current."""
        if not self.context() or not self.context().isValid():
            return
        self.makeCurrent()
        for _, gpu in self._meshes.values():
            gpu.destroy()
        self._meshes.clear()
        for _, gpu in self._trails:
            if gpu:
                gpu.destroy()
        self._trails.clear()
        if self._floor:
            self._floor.destroy()
            self._floor = None
        if self._program:
            shiboken6.delete(self._program)
            self._program = None
        self.ready = False
        self.doneCurrent()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position() - self._drag_pos
            self._drag_pos = event.position()
            self.orbitRequested.emit(delta.x(), delta.y())

    def mouseReleaseEvent(self, event) -> None:
        self._drag_pos = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = None
            self.expandRequested.emit()

    def wheelEvent(self, event) -> None:
        self.zoomRequested.emit(event.angleDelta().y() / 120.0)
        event.accept()
