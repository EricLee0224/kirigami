#!/usr/bin/env python
"""Launch the Kirigami GUI."""

from __future__ import annotations

import argparse
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
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from PySide6.QtWidgets import QApplication

    from .gui.main_window import MainWindow

    cfg = Robot3DConfig.from_yaml(args.robot_config)
    if args.urdf is not None:
        cfg.urdf = args.urdf
    if args.mesh_dir is not None:
        cfg.mesh_dir = args.mesh_dir

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
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
