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
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate kirigami
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Headless servers have no X11. Override on a desktop if needed, e.g. QT_QPA_PLATFORM=xcb
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
cd "$ROOT"
python -m kirigami.app "$@"
