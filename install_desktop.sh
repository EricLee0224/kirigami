#!/usr/bin/env bash
set -euo pipefail
KIRIGAMI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 - "$KIRIGAMI_ROOT" <<'PY'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
applications = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))) / "applications"
applications.mkdir(parents=True, exist_ok=True)
# Desktop Entry Exec quoting is different from shell quoting.
executable = str(root / "run_kirigami.sh")
for char in ('\\', '"', '`', '$'):
    executable = executable.replace(char, '\\' + char)
executable = executable.replace('%', '%%')
entry = applications / "kirigami.desktop"
entry.write_text(
    '[Desktop Entry]\nType=Application\nName=Kirigami\n'
    'Comment=Slice and annotate robot demonstrations\n'
    f'Exec="{executable}" %F\n'
    'Terminal=false\nIcon=applications-science\n'
    'Categories=Science;\nStartupNotify=true\n',
    encoding="utf-8",
)
print(f"Installed {entry}. Search for Kirigami in the desktop application menu.")
PY
