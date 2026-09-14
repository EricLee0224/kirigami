#!/usr/bin/env python
"""Launch the Kirigami GUI."""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path

from .robot3d import Robot3DConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Kirigami: slice Prometheus long-horizon episodes into subtasks")
    ap.add_argument(
        "raw_dirs",
        nargs="*",
        type=Path,
        help="task dirs (e.g. romoya-egg-stage2-0912_100) or single episode dirs",
    )
    ap.add_argument("--out-root", type=Path, default=None, help="override <task>_sliced output root")
    ap.add_argument("--subtasks", type=Path, default=repo_root / "subtasks.yaml", help="subtask preset yaml")
    ap.add_argument("--urdf", type=Path, default=None, help="dual-arm URDF (optional)")
    ap.add_argument("--mesh-dir", type=Path, default=None, help="URDF mesh directory")
    ap.add_argument("--robot-config", type=Path, default=repo_root / "robot3d.yaml")
    ap.add_argument("--check", action="store_true", help="open a real desktop window, check Qt/3D, then exit")
    return ap.parse_args(argv)


def desktop_startup_error() -> str | None:
    if not sys.platform.startswith("linux"):
        return None
    platform = os.environ.get("QT_QPA_PLATFORM", "")
    if platform in {"offscreen", "minimal", "minimalegl"}:
        return "Kirigami needs a visible desktop. Unset QT_QPA_PLATFORM and launch from your desktop terminal."
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return "No desktop session found. Run ./run_kirigami.sh in a terminal on the host's X11/Wayland desktop."
    if not platform:
        platform = "xcb" if os.environ.get("DISPLAY") else "wayland"
        os.environ["QT_QPA_PLATFORM"] = platform
    if platform == "xcb":
        from PySide6.QtCore import QLibraryInfo
        plugin = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)) / "platforms/libqxcb.so"
        try:
            ctypes.CDLL(str(plugin))
            x11 = ctypes.CDLL("libX11.so.6")
            x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
            x11.XOpenDisplay.restype = ctypes.c_void_p
            x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
            display = x11.XOpenDisplay(None)
            if not display:
                return f"Cannot connect to DISPLAY={os.environ.get('DISPLAY', '')}. Launch from the logged-in desktop session with its display permissions."
            x11.XCloseDisplay(display)
        except OSError as exc:
            return f"Missing desktop runtime library: {exc}. Run ./setup_env.sh to install the Qt/X11 dependencies."
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    error = desktop_startup_error()
    if error:
        print(error, file=sys.stderr)
        return 1
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QSurfaceFormat

    from .gui.main_window import MainWindow
    from .gui.robot_view import desktop_gl_format

    cfg = Robot3DConfig.from_yaml(args.robot_config)
    if args.urdf is not None:
        cfg.urdf = args.urdf
    if args.mesh_dir is not None:
        cfg.mesh_dir = args.mesh_dir

    QSurfaceFormat.setDefaultFormat(desktop_gl_format())
    app = QApplication(sys.argv)
    from . import APP_NAME

    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    win = MainWindow(
        initial_dirs=list(args.raw_dirs),
        out_root=args.out_root,
        presets_path=args.subtasks,
        robot_config=cfg,
    )
    win.show()
    if args.check:
        def check_window() -> None:
            visible = win.isVisible() and win.windowHandle().isExposed()
            ready = visible and win.robot.available and win.robot_panel.ready
            print(f"Desktop check: platform={app.platformName()}, display={os.environ.get('DISPLAY', os.environ.get('WAYLAND_DISPLAY', ''))}, visible={visible}, robot3d={win.robot.available}, opengl={win.robot_panel.ready}", flush=True)
            print(f"3D renderer: {win.robot_panel.renderer}; triangles={win.robot.triangle_count:,}", flush=True)
            if win.robot_panel.error:
                print(win.robot_panel.error, file=sys.stderr)
            if not win.robot.available:
                print(win.robot.status, file=sys.stderr)
            win.close()
            app.exit(0 if ready else 1)
        QTimer.singleShot(750, check_window)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
