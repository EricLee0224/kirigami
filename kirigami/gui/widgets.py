"""Kirigami widgets: cameras, timeline, joint plot, 3D placeholder."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap, QFont
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

from ..annotation import Annotation, Segment

PALETTE = [
    QColor(66, 133, 244),
    QColor(219, 68, 55),
    QColor(244, 180, 0),
    QColor(15, 157, 88),
    QColor(171, 71, 188),
    QColor(0, 172, 193),
    QColor(255, 112, 67),
    QColor(124, 77, 255),
]


def subtask_color(name: str) -> QColor:
    if not name:
        return QColor(90, 90, 90)
    return PALETTE[sum(ord(c) for c in name) % len(PALETTE)]


def numpy_to_pixmap(rgb: np.ndarray) -> QPixmap:
    h, w = rgb.shape[:2]
    buf = np.ascontiguousarray(rgb)
    qimg = QImage(buf.data, w, h, buf.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class CameraLabel(QLabel):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(240, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background:#111; color:#888; border:1px solid #333;")
        self.setText(title)
        self._pixmap: QPixmap | None = None

    def set_frame(self, rgb: np.ndarray | None) -> None:
        if rgb is None:
            self._pixmap = None
            self.setText(self._title)
            return
        self._pixmap = numpy_to_pixmap(rgb)
        self._rescale()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class TimelineBar(QWidget):
    frameClicked = Signal(int)
    markToggleRequested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.n_frames = 1
        self.frame = 0
        self.annotation: Annotation | None = None
        self.setMinimumHeight(56)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

    def set_state(self, n_frames: int, frame: int, annotation: Annotation | None) -> None:
        self.n_frames = max(int(n_frames), 1)
        self.frame = int(np.clip(frame, 0, self.n_frames - 1))
        self.annotation = annotation
        self.update()

    def _x_to_frame(self, x: int) -> int:
        w = max(self.width() - 2, 1)
        return int(np.clip(round((x - 1) / w * (self.n_frames - 1)), 0, self.n_frames - 1))

    def _frame_to_x(self, frame: int) -> int:
        w = max(self.width() - 2, 1)
        return 1 + int(round(frame / max(self.n_frames - 1, 1) * w))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        frame = self._x_to_frame(event.position().toPoint().x())
        if event.button() == Qt.MouseButton.RightButton:
            self.markToggleRequested.emit(frame)
        else:
            self.frameClicked.emit(frame)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.frameClicked.emit(self._x_to_frame(event.position().toPoint().x()))

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        self.markToggleRequested.emit(self._x_to_frame(event.position().toPoint().x()))

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(1, 8, -1, -8)
        p.fillRect(self.rect(), QColor(22, 22, 24))
        p.fillRect(r, QColor(40, 40, 44))
        segs: Sequence[Segment] = self.annotation.segments if self.annotation else []
        for seg in segs:
            x0 = self._frame_to_x(seg.start_frame)
            x1 = self._frame_to_x(max(seg.end_frame - 1, seg.start_frame))
            color = QColor(70, 70, 70) if seg.discard else subtask_color(seg.subtask)
            color.setAlpha(90 if seg.subtask or seg.discard else 40)
            p.fillRect(QRect(x0, r.top(), max(x1 - x0, 2), r.height()), color)
        p.setPen(QPen(QColor(70, 70, 74), 1))
        p.drawRect(r)
        if self.annotation:
            for m in self.annotation.marks:
                x = self._frame_to_x(m)
                p.setPen(QPen(QColor(255, 214, 10), 2))
                p.drawLine(x, r.top() - 4, x, r.bottom() + 4)
        x = self._frame_to_x(self.frame)
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.drawLine(x, r.top(), x, r.bottom())
        p.end()


class JointPlot(QWidget):
    frameClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.n_frames = 1
        self.frame = 0
        self.left = np.zeros((0, 7), dtype=np.float32)
        self.right = np.zeros((0, 7), dtype=np.float32)
        self.marks: list[int] = []
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(self, left: np.ndarray, right: np.ndarray, frame: int, marks: list[int]) -> None:
        self.left = np.asarray(left, dtype=np.float32)
        self.right = np.asarray(right, dtype=np.float32)
        self.n_frames = max(int(self.left.shape[0]), 1)
        self.frame = int(np.clip(frame, 0, self.n_frames - 1))
        self.marks = list(marks)
        self.update()

    def set_frame(self, frame: int) -> None:
        self.frame = int(np.clip(frame, 0, self.n_frames - 1))
        self.update()

    def _x_to_frame(self, x: int) -> int:
        w = max(self.width() - 2, 1)
        return int(np.clip(round(x / w * (self.n_frames - 1)), 0, self.n_frames - 1))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.frameClicked.emit(self._x_to_frame(event.position().toPoint().x()))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.frameClicked.emit(self._x_to_frame(event.position().toPoint().x()))

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(18, 18, 20))
        w, h = self.width(), self.height()
        if self.left.size == 0:
            p.setPen(QColor(120, 120, 120))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "no joint data")
            p.end()
            return

        def _norm(arr: np.ndarray) -> np.ndarray:
            out = arr.astype(np.float32).copy()
            for j in range(out.shape[1]):
                col = out[:, j]
                lo, hi = float(col.min()), float(col.max())
                if hi - lo < 1e-6:
                    out[:, j] = 0.5
                else:
                    out[:, j] = (col - lo) / (hi - lo)
            return out

        left_n = _norm(self.left)
        right_n = _norm(self.right) if self.right.size else left_n
        pad = 8
        inner_h = max(h - 2 * pad, 1)
        step = max(1, self.n_frames // 400)
        xs = np.linspace(0, w - 1, self.n_frames)

        def _draw(arr: np.ndarray, base: QColor) -> None:
            for j in range(arr.shape[1]):
                color = QColor(base)
                color.setAlpha(80 + j * 20)
                p.setPen(QPen(color, 1))
                last = None
                for i in range(0, self.n_frames, step):
                    pt = QPoint(int(xs[i]), int(pad + (1.0 - arr[i, j]) * inner_h))
                    if last is not None:
                        p.drawLine(last, pt)
                    last = pt
                end = QPoint(int(xs[-1]), int(pad + (1.0 - arr[-1, j]) * inner_h))
                if last is not None and last != end:
                    p.drawLine(last, end)

        _draw(left_n, QColor(66, 165, 245))
        _draw(right_n, QColor(255, 167, 38))
        p.setPen(QPen(QColor(255, 214, 10), 1, Qt.PenStyle.DashLine))
        for m in self.marks:
            x = int(m / max(self.n_frames - 1, 1) * (w - 1))
            p.drawLine(x, 0, x, h)
        x = int(self.frame / max(self.n_frames - 1, 1) * (w - 1))
        p.setPen(QPen(QColor(255, 255, 255), 1))
        p.drawLine(x, 0, x, h)
        p.setPen(QColor(160, 160, 160))
        p.setFont(QFont("Sans", 9))
        p.drawText(8, 14, "joints  left=blue  right=orange")
        p.end()


class Robot3DPanel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumSize(280, 220)
        self.setStyleSheet("background:#16161a; color:#c8c8c8; border:1px solid #333; padding:12px;")
        self.set_status("等待 URDF：加载后可在此查看双臂 3D 运动与末端轨迹")

    def set_status(self, text: str) -> None:
        self.setText(text)

    def set_trail_image(self, rgb: np.ndarray | None, caption: str) -> None:
        if rgb is None:
            self.set_status(caption)
            return
        pix = numpy_to_pixmap(rgb)
        self.setPixmap(
            pix.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )
