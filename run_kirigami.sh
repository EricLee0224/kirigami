#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -z "${CONDA_BASE}" ]]; then
  for candidate in "${HOME}/miniconda3" "${HOME}/anaconda3" /root/miniconda3 /opt/conda; do
    if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
      CONDA_BASE="${candidate}"
      break
    fi
  done
fi
if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "Conda was not found. Install Miniconda, then run ./setup_env.sh." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
if ! conda activate kirigami; then
  echo "Kirigami environment is missing. Run $ROOT/setup_env.sh first." >&2
  exit 1
fi
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Use the host's real desktop session. Do not invent DISPLAY or use offscreen.
if [[ -z "${QT_QPA_PLATFORM:-}" ]]; then
  if [[ -n "${DISPLAY:-}" ]]; then
    export QT_QPA_PLATFORM=xcb
  elif [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
    export QT_QPA_PLATFORM=wayland
  fi
fi
cd "$ROOT"
exec python -m kirigami.app "$@"
