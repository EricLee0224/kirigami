"""Manage reusable subtask names without changing episode annotations."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QPushButton, QVBoxLayout,
)
from yaml import YAMLError

from ..annotation import validate_subtask_name


class PresetDialog(QDialog):
    def __init__(self, names: list[str], path: Path,
                 save: Callable[[list[str]], None], parent=None):
        super().__init__(parent)
        self.setObjectName("PresetManager")
        self.setWindowTitle("Manage subtask presets")
        self.setMinimumSize(520, 440)
        self.resize(580, 560)
        self.names = list(names)
        self._save = save

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(12)
        header = QHBoxLayout()
        title = QLabel("Subtask presets")
        title.setProperty("role", "heading")
        header.addWidget(title)
        header.addStretch()
        self.count = QLabel()
        self.count.setProperty("role", "badge")
        header.addWidget(self.count)
        layout.addLayout(header)
        description = QLabel("Add, rename or delete reusable names. Changes save immediately.\n"
                             "Existing segment labels and exported files keep their names.")
        description.setProperty("role", "muted")
        description.setWordWrap(True)
        layout.addWidget(description)

        self.list = QListWidget()
        self.list.setObjectName("PresetList")
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.itemSelectionChanged.connect(self._selection_changed)
        self.list.itemDoubleClicked.connect(self._focus_name)
        layout.addWidget(self.list, 1)
        self.empty = QLabel("No presets yet. Enter a name below to add the first one.")
        self.empty.setProperty("role", "muted")
        self.empty.setWordWrap(True)
        layout.addWidget(self.empty)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Select a preset to rename, or enter a new name")
        self.name_edit.textChanged.connect(self._update_buttons)
        layout.addWidget(self.name_edit)
        actions = QHBoxLayout()
        self.add_btn = self._button("Add new", self._add, "primary")
        self.rename_btn = self._button("Rename", self._rename)
        self.delete_btn = self._button("Delete selected", self._delete, "trim")
        actions.addWidget(self.add_btn)
        actions.addWidget(self.rename_btn)
        actions.addStretch()
        actions.addWidget(self.delete_btn)
        layout.addLayout(actions)
        self.feedback = QLabel("Ctrl / Shift + click to select multiple presets.")
        self.feedback.setWordWrap(True)
        self.feedback.setTextFormat(Qt.TextFormat.PlainText)
        self.feedback.setProperty("role", "muted")
        layout.addWidget(self.feedback)

        footer = QHBoxLayout()
        location = QLabel(f"Saved to: {path.name}")
        location.setProperty("role", "muted")
        location.setTextFormat(Qt.TextFormat.PlainText)
        location.setToolTip(str(path.resolve()))
        footer.addWidget(location, 1)
        footer.addWidget(self._button("Close", self.accept))
        layout.addLayout(footer)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self.list, self._delete,
                  context=Qt.ShortcutContext.WidgetShortcut)
        QShortcut(QKeySequence(Qt.Key.Key_F2), self.list, self._focus_name,
                  context=Qt.ShortcutContext.WidgetShortcut)
        self._refresh()

    def _button(self, text, callback, variant=""):
        button = QPushButton(text)
        button.setAutoDefault(False)
        button.setProperty("variant", variant)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(callback)
        return button

    def _refresh(self, selected: str | None = None):
        self.list.clear()
        self.list.addItems(self.names)
        self.count.setText(f"{len(self.names)} PRESETS")
        self.empty.setVisible(not self.names)
        if selected in self.names:
            self.list.setCurrentRow(self.names.index(selected))
        else:
            self.name_edit.clear()
        self._update_buttons()

    def _selection_changed(self):
        selected = self.list.selectedItems()
        self.name_edit.setText(selected[0].text() if len(selected) == 1 else "")
        self._update_buttons()

    def _update_buttons(self):
        selected = self.list.selectedItems()
        name = self.name_edit.text().strip()
        self.add_btn.setEnabled(bool(name) and name not in self.names)
        self.rename_btn.setEnabled(len(selected) == 1 and bool(name) and name != selected[0].text())
        self.delete_btn.setEnabled(bool(selected))

    def _focus_name(self, *_):
        if len(self.list.selectedItems()) == 1:
            self.name_edit.setFocus()
            self.name_edit.selectAll()

    def _new_name(self):
        name = validate_subtask_name(self.name_edit.text())
        if name in self.names:
            raise ValueError(f'A preset named "{name}" already exists.')
        return name

    def _feedback(self, message: str, error: bool = False):
        self.feedback.setProperty("role", "error" if error else "muted")
        self.feedback.style().unpolish(self.feedback)
        self.feedback.style().polish(self.feedback)
        self.feedback.setText(message)

    def _apply(self, names: list[str], selected: str | None, message: str):
        try:
            self._save(names)
        except (ValueError, OSError, YAMLError) as exc:
            self._feedback(f"Could not save: {exc}", error=True)
            return
        self.names = names
        self._refresh(selected)
        self._feedback(message)

    def _add(self):
        try:
            name = self._new_name()
        except ValueError as exc:
            self._feedback(str(exc), error=True)
            return
        self._apply([*self.names, name], name, f'Added "{name}".')

    def _rename(self):
        selected = self.list.selectedItems()
        if len(selected) != 1:
            return
        old = selected[0].text()
        try:
            name = self._new_name()
        except ValueError as exc:
            self._feedback(str(exc), error=True)
            return
        self._apply([name if item == old else item for item in self.names], name,
                    f'Renamed "{old}" to "{name}". Existing segment labels keep their names.')

    def _delete(self):
        selected = {item.text() for item in self.list.selectedItems()}
        if not selected:
            return
        row = min(self.list.row(item) for item in self.list.selectedItems())
        names = [name for name in self.names if name not in selected]
        next_name = names[min(row, len(names) - 1)] if names else None
        self._apply(names, next_name,
                    f"Deleted {len(selected)} preset(s). Existing segment labels keep their names.")
