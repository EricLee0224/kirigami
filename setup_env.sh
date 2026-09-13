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
if conda env list | awk '{print $1}' | grep -qx kirigami; then
  conda env update -n kirigami -f "$ROOT/environment.yml" --prune
else
  conda env create -f "$ROOT/environment.yml"
fi
echo "kirigami env ready. Launch with: $ROOT/run_kirigami.sh"
