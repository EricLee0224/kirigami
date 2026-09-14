"""Kirigami widgets: cameras, timeline, joint plot, 3D placeholder."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import QPoint, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPainterPath, QPen, QPixmap, QFont
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

from ..annotation import Annotation, Segment

PALETTE = [
    QColor("#3c8d80"), QColor("#7694b8"), QColor("#b79965"), QColor("#9883b4"),
    QColor("#5d9ba5"), QColor("#b88586"), QColor("#85a17f"), QColor("#7d90ae"),
]


def subtask_color(name: str) -> QColor:
    if not name:
        return QColor("#c6d0db")
    return PALETTE[sum(ord(c) for c in name) % len(PALETTE)]


def numpy_to_pixmap(rgb: np.ndarray) -> QPixmap:
    h, w = rgb.shape[:2]
    buf = np.ascontiguousarray(rgb)
    qimg = QImage(buf.data, w, h, buf.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class CameraLabel(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self.setMinimumSize(160, 120)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._background = QColor("#14202e")
        self._message = "No episode loaded"
        self._pixmap: QPixmap | None = None

    def sizeHint(self) -> QSize:
        return QSize(360, 230)

    def set_frame(self, rgb: np.ndarray | None) -> None:
        if rgb is None:
            self._pixmap = None
        else:
            self._pixmap = numpy_to_pixmap(rgb)
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(self.rect()), 8, 8)
        p.setClipPath(clip)
        p.fillRect(self.rect(), self._background)
        if self._pixmap is not None:
            size = self._pixmap.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            target = QRect((self.width() - size.width()) // 2, (self.height() - size.height()) // 2, size.width(), size.height())
            p.drawPixmap(target, self._pixmap)
        else:
            cx, cy = self.width() // 2, self.height() // 2 - 17
            p.setPen(QPen(QColor("#536477"), 1.3))
            p.drawRoundedRect(QRectF(cx - 18, cy - 12, 36, 26), 6, 6)
            p.drawEllipse(QPoint(cx, cy + 1), 6, 6)
            p.setPen(QColor("#8292a7"))
            p.setFont(QFont("Noto Sans", 9))
            p.drawText(QRect(8, cy + 26, self.width() - 16, 45), Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap, self._message)
        p.end()


class TimelineBar(QWidget):
    frameClicked = Signal(int)
    markToggleRequested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.n_frames = 1
        self.frame = 0
        self.annotation: Annotation | None = None
        self.setFixedHeight(72)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

    def set_state(self, n_frames: int, frame: int, annotation: Annotation | None) -> None:
        self.n_frames = max(int(n_frames), 1)
        self.frame = int(np.clip(frame, 0, self.n_frames - 1))
        self.annotation = annotation
        self.update()

    def _x_to_frame(self, x: int) -> int:
        w = max(self.width() - 24, 1)
        return int(np.clip(round((x - 12) / w * (self.n_frames - 1)), 0, self.n_frames - 1))

    def _frame_to_x(self, frame: int) -> int:
        w = max(self.width() - 24, 1)
        return 12 + int(round(frame / max(self.n_frames - 1, 1) * w))

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
        r = QRect(12, 27, max(self.width() - 24, 1), 35)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#f0f3f7"))
        p.drawRoundedRect(r, 6, 6)
        p.setFont(QFont("Noto Sans", 8))
        for tick in np.unique(np.linspace(0, self.n_frames - 1, 6, dtype=int)):
            x = self._frame_to_x(int(tick))
            p.setPen(QPen(QColor("#dce3ec"), 1))
            p.drawLine(x, 18, x, 22)
            p.setPen(QColor("#99a5b5"))
            label = str(tick)
            text_width = p.fontMetrics().horizontalAdvance(label)
            text_x = max(2, min(x - text_width // 2, self.width() - text_width - 2))
            p.drawText(text_x, 12, label)
        segs: Sequence[Segment] = self.annotation.segments if self.annotation else []
        for seg in segs:
            x0 = self._frame_to_x(seg.start_frame)
            x1 = self._frame_to_x(max(seg.end_frame - 1, seg.start_frame))
            color = QColor("#cbd3de") if seg.discard else subtask_color(seg.subtask)
            fill = QColor(color)
            fill.setAlpha(48 if seg.discard or not seg.subtask else 65)
            rect = QRect(x0 + 1, r.top() + 2, max(x1 - x0 - 2, 2), r.height() - 4)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(fill)
            p.drawRoundedRect(rect, 4, 4)
            if rect.width() > 55:
                p.setPen(color.darker(140))
                text = "Skipped" if seg.discard else seg.subtask or "Unlabeled"
                text = p.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, rect.width() - 12)
                p.drawText(rect.adjusted(6, 0, -6, 0), Qt.AlignmentFlag.AlignVCenter, text)
        if self.annotation:
            for m in self.annotation.marks:
                x = self._frame_to_x(m)
                p.setPen(QPen(QColor("#9aaabc"), 1))
                p.drawLine(x, r.top() - 3, x, r.bottom() + 3)
            start, end = self.annotation.keep_range
            if start > 0 or end < self.n_frames:
                x0, x1 = self._frame_to_x(start), self._frame_to_x(end - 1)
                shade = QColor(176, 77, 62, 85)
                p.fillRect(QRect(r.left(), r.top(), max(x0 - r.left(), 0), r.height()), shade)
                p.fillRect(QRect(x1 + 1, r.top(), max(r.right() - x1, 0), r.height()), shade)
                p.setPen(QPen(QColor("#b05243"), 2))
                for marker in (x0, x1):
                    p.drawLine(marker, r.top() - 4, marker, r.bottom() + 4)
        x = self._frame_to_x(self.frame)
        p.setPen(QPen(QColor("#168477"), 2))
        p.drawLine(x, r.top() - 5, x, r.bottom() + 5)
        p.setBrush(QColor("#168477"))
        p.drawEllipse(QPoint(x, r.top() - 6), 3, 3)
        p.end()


class JointPlot(QWidget):
    frameClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.n_frames = 1
        self.frame = 0
        self.left = np.zeros((0, 7), dtype=np.float32)
        self.right = np.zeros((0, 7), dtype=np.float32)
        self._left_normalized = self.left
        self._right_normalized = self.right
        self.marks: list[int] = []
        self.setMinimumHeight(85)
        self.setMaximumHeight(130)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(self, left: np.ndarray, right: np.ndarray, frame: int, marks: list[int]) -> None:
        changed = left is not self.left or right is not self.right
        self.left = np.asarray(left, dtype=np.float32)
        self.right = np.asarray(right, dtype=np.float32)
        if changed:
            self._left_normalized = self._normalize(self.left)
            self._right_normalized = self._normalize(self.right)
        self.n_frames = max(int(self.left.shape[0]), 1)
        self.frame = int(np.clip(frame, 0, self.n_frames - 1))
        self.marks = list(marks)
        self.update()

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        if not arr.size:
            return arr
        lo = arr.min(axis=0)
        span = arr.max(axis=0) - lo
        return np.where(span < 1e-6, 0.5, (arr - lo) / np.maximum(span, 1e-6))

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
        p.fillRect(self.rect(), QColor("#ffffff"))
        w, h = self.width(), self.height()
        p.setPen(QPen(QColor("#edf1f5"), 1))
        for y in np.linspace(8, h - 8, 4):
            p.drawLine(0, int(y), w, int(y))
        if self.left.size == 0:
            p.setPen(QColor("#9aa7b7"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Joint motion appears here when an episode is loaded")
            p.end()
            return

        left_n = self._left_normalized
        right_n = self._right_normalized if self.right.size else left_n
        pad = 8
        inner_h = max(h - 2 * pad, 1)
        step = max(1, self.n_frames // 400)
        xs = np.linspace(0, w - 1, self.n_frames)

        def _draw(arr: np.ndarray, base: QColor) -> None:
            for j in range(arr.shape[1]):
                color = QColor(base)
                color.setAlpha(55 + j * 17)
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

        _draw(left_n, QColor("#168477"))
        _draw(right_n, QColor("#bd8747"))
        p.setPen(QPen(QColor("#bdc8d5"), 1, Qt.PenStyle.DashLine))
        for m in self.marks:
            x = int(m / max(self.n_frames - 1, 1) * (w - 1))
            p.drawLine(x, 0, x, h)
        x = int(self.frame / max(self.n_frames - 1, 1) * (w - 1))
        p.setPen(QPen(QColor("#168477"), 1))
        p.drawLine(x, 0, x, h)
        p.end()
