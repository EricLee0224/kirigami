"""Progress for background jobs without nested event processing."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QVBoxLayout


class TaskProgressDialog(QDialog):
    """QProgressBar updates don't pump events like a modal QProgressDialog does."""

    def __init__(self, title: str, message: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        self.setMinimumWidth(430)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)
        self.label = QLabel(message)
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        layout.addWidget(self.bar)

    def setLabelText(self, text: str) -> None:
        self.label.setText(text)

    def setValue(self, value: int) -> None:
        self.bar.setValue(value)

    def reject(self) -> None:
        # Trimming/encoding is not cancellable halfway through publication.
        pass

    def closeEvent(self, event) -> None:
        event.ignore()

    def finish(self) -> None:
        self.accept()
        self.deleteLater()
