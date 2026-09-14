"""Shared desktop palette and controls for the annotation workspace."""

from PySide6.QtGui import QColor, QFont, QPalette
from pathlib import Path


STYLE = """
QMainWindow, QWidget#Workspace { background: #f1f4f8; }
QDialog#PresetManager { background: #f1f4f8; }
QWidget { color: #253448; font-size: 12px; }
QFrame#Topbar { background: #ffffff; border-bottom: 1px solid #e0e6ee; }
QFrame#Sidebar { background: #172436; border-radius: 14px; }
QFrame#Sidebar QLabel { color: #c1cddd; }
QFrame#Sidebar QLabel[role="eyebrow"] { color: #8294ad; }
QFrame[role="card"] { background: #ffffff; border: 1px solid #e0e6ee; border-radius: 12px; }
QFrame[role="card"] QLabel { background: transparent; border: none; }
QLabel[role="brand"] { font-size: 23px; font-weight: 700; color: #182a3a; }
QLabel[role="logo"] { color: #ffffff; background: #168477; border-radius: 9px; font-size: 23px; font-weight: 700; }
QLabel[role="eyebrow"] { font-size: 10px; font-weight: 600; color: #8b98aa; }
QLabel[role="heading"] { font-size: 14px; font-weight: 600; color: #253448; }
QLabel[role="muted"] { color: #8794a6; font-size: 11px; }
QLabel[role="error"] { color: #a04e3c; font-size: 11px; }
QLabel[role="badge"], QFrame[role="card"] QLabel[role="badge"] { background: #edf6f3; color: #168477; border-radius: 6px; padding: 5px 9px; font-size: 10px; font-weight: 600; }
QLabel[role="leftLegend"] { color: #168477; font-size: 10px; }
QLabel[role="rightLegend"] { color: #bd8747; font-size: 10px; }
QPushButton { background: #ffffff; border: 1px solid #dce3ec; border-radius: 7px; padding: 7px 11px; font-weight: 500; color: #43536a; }
QPushButton:hover { background: #f1f7f6; border-color: #9cc9c1; color: #16766c; }
QPushButton:pressed { background: #e4efec; }
QPushButton:disabled { color: #aeb7c5; border-color: #e7ebf0; background: #f6f8fa; }
QPushButton[variant="primary"] { background: #168477; border-color: #168477; color: #ffffff; font-weight: 600; }
QPushButton[variant="primary"]:hover { background: #107366; border-color: #107366; }
QPushButton[variant="primary"]:disabled { background: #9cbdb7; border-color: #9cbdb7; color: #edf4f2; }
QPushButton[variant="quiet"] { background: transparent; border-color: transparent; padding: 5px 8px; }
QPushButton[variant="trim"] { background: #fff5f1; border-color: #e7c1b5; color: #a04e3c; }
QPushButton[variant="trim"]:hover { background: #fce8df; border-color: #c78c78; }
QPushButton:checked { background: #e8f3f0; border-color: #a9d2c8; color: #147668; }
QFrame#Sidebar QPushButton { background: #24354c; border-color: #34465f; color: #dbe5f1; }
QFrame#Sidebar QPushButton:hover { background: #30445d; border-color: #587089; }
QFrame#Sidebar QPushButton[variant="primary"] { background: #168477; border-color: #168477; color: white; }
QFrame#Sidebar QLineEdit { background: #202f44; border-color: #34455e; color: #d2ddeb; selection-background-color: #287e79; }
QLineEdit, QSpinBox, QComboBox { background: #ffffff; border: 1px solid #dce3ec; border-radius: 6px; padding: 6px 8px; min-height: 18px; selection-background-color: #d7ebe5; selection-color: #154e45; }
QLineEdit:focus, QSpinBox:focus, QComboBox:focus { border-color: #168477; }
QSpinBox { padding-right: 18px; }
QSpinBox::up-button, QSpinBox::down-button { width: 16px; border: none; background: #eef2f6; }
QComboBox { padding-right: 22px; }
QComboBox::drop-down { width: 20px; border: none; }
QComboBox::down-arrow { image: url("CHEVRON_ASSET"); width: 12px; height: 12px; }
QComboBox#MaterialSelect { padding-top: 2px; padding-bottom: 2px; min-height: 16px; }
QPushButton#ResetView, QPushButton#ExpandView { padding: 0 5px; }
QComboBox QAbstractItemView { background: #ffffff; color: #253448; selection-background-color: #e2f1ec; selection-color: #126c61; border: 1px solid #dbe4eb; outline: none; }
QListWidget { background: transparent; border: none; outline: none; }
QListWidget::item { color: #aabbd0; padding: 11px 12px; border-radius: 8px; margin: 3px 0; }
QListWidget::item:hover { background: #23354c; }
QListWidget::item:selected { background: #2b494d; color: #e1f6ef; border: 1px solid #3b6562; }
QListWidget#PresetList { background: #ffffff; border: 1px solid #e0e6ee; border-radius: 10px; padding: 6px; }
QListWidget#PresetList::item { color: #43536a; padding: 9px 12px; margin: 2px 0; border: 1px solid transparent; }
QListWidget#PresetList::item:hover { background: #f1f7f6; }
QListWidget#PresetList::item:selected { background: #e2f1ec; color: #126c61; border-color: #b9d9d0; }
QTableWidget { background: #ffffff; alternate-background-color: #fafbfd; border: none; gridline-color: #edf1f5; selection-background-color: #e9f4f0; selection-color: #1f665b; outline: none; }
QTableWidget::item { padding: 6px 4px; border-bottom: 1px solid #edf1f5; }
QTableWidget QComboBox { margin: 3px 2px; border-color: transparent; background: transparent; padding: 5px 4px; padding-right: 20px; }
QTableWidget QComboBox:hover, QTableWidget QComboBox:focus { background: #f2f7f5; border-color: #d3e5dd; }
QHeaderView::section { background: #f7f9fb; color: #8b98a8; border: none; padding: 9px 4px; font-size: 10px; font-weight: 600; }
QCheckBox { spacing: 6px; color: #7c899c; }
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #cbd6df; border-radius: 4px; background: white; }
QCheckBox::indicator:checked { background: #168477; border-color: #168477; image: url("CHECK_ASSET"); }
QCheckBox::indicator:hover { border-color: #168477; }
QScrollBar:vertical { background: transparent; width: 7px; margin: 3px 0; }
QScrollBar::handle:vertical { background: #cad4df; border-radius: 3px; min-height: 28px; }
QScrollBar:horizontal { background: transparent; height: 7px; }
QScrollBar::handle:horizontal { background: #cad4df; border-radius: 3px; min-width: 28px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QSplitter::handle { background: transparent; width: 12px; height: 10px; }
QStatusBar { background: #f1f4f8; color: #8b98a8; font-size: 10px; border: none; }
QStatusBar::item { border: none; }
QToolTip { color: #ffffff; background: #253448; border: none; padding: 6px 9px; }
QProgressDialog, QMessageBox { background: #f6f8fa; }
QProgressBar { border: none; background: #e3ebe9; border-radius: 5px; height: 9px; text-align: center; }
QProgressBar::chunk { background: #168477; border-radius: 5px; }
"""


def apply_theme(window) -> None:
    font = QFont("Noto Sans", 10)
    window.setFont(font)
    palette = QPalette(window.palette())
    for role, value in (
        (QPalette.ColorRole.Window, "#f1f4f8"),
        (QPalette.ColorRole.WindowText, "#253448"),
        (QPalette.ColorRole.Base, "#ffffff"),
        (QPalette.ColorRole.Text, "#253448"),
        (QPalette.ColorRole.Button, "#ffffff"),
        (QPalette.ColorRole.ButtonText, "#43536a"),
        (QPalette.ColorRole.Highlight, "#d7ebe5"),
        (QPalette.ColorRole.HighlightedText, "#155b50"),
        (QPalette.ColorRole.PlaceholderText, "#9ba7b7"),
    ):
        palette.setColor(role, QColor(value))
    window.setPalette(palette)
    assets = Path(__file__).parent / "assets"
    window.setStyleSheet(STYLE.replace("CHECK_ASSET", str(assets / "check.svg")).replace("CHEVRON_ASSET", str(assets / "chevron.svg")))
