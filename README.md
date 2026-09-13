# Kirigami

Desktop tool for cutting long-horizon robot demonstrations into short subtask episodes.

Kirigami loads Prometheus raw episodes (`prometheus_raw_episode_v1`), plays three cameras plus a dual YAM Ultra 2 view, lets you mark split points on a timeline, names each segment, and exports folders in the same raw format.

The GUI needs a display (X11, a desktop session, or X11 forwarding). A headless training server is fine for developing and pushing; run the app on a machine with a screen.

## Install

```bash
./setup_env.sh
```

This creates a standalone `kirigami` conda env (not `lerobot`).

## Launch

```bash
./run_kirigami.sh /path/to/romoya-egg-stage2-0912_100
```

Keyboard: Space play/pause, arrows step frames, Shift+arrows jump 1s, `M` add a split, Delete remove the nearest split.

## 3D arms

Two YAM Ultra 2 arms are placed in parallel with a **46 cm** base spacing. The URDF and STL meshes are vendored from i2rt under `models/yam_ultra/v2/` (MIT, I2RT Robotics).

To use another copy:

```bash
export KIRIGAMI_YAM_ULTRA_DIR=/path/to/yam_ultra/v2
```

Recorded state is 7-D per arm: `joint1`–`joint6` plus gripper stroke.

## Export layout

Default output:

```text
<task>_sliced/<subtask>/<src_ep>_<seg>/
```

Each short episode keeps `camera/`, `robot/`, `action/`, `event/`, `manifests/`. Source episodes store `annotations/slices.json` so you can reopen and re-export.

## Tests

From the repo root, with the `kirigami` env active:

```bash
python -m unittest tests.test_core
```
