#!/bin/bash
# One-step setup after copying "Stock Analysis" (and "Voice Genie" next to it) to a new Mac:   bash setup_mac_mini.sh
# Installs the packages, points git's notebook filter at this Mac's python, and installs the launchd jobs
# (launchd/install_schedule.sh: the 5 com.stockanalysis.* jobs + com.unicorn.voice when ../Voice Genie exists).
# Python: $PYTHON, else /opt/anaconda3/bin/python3 (same as the old Mac, recommended), else python3 on PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
VG="$(dirname "$ROOT")/Voice Genie"
PY="${PYTHON:-}"
[ -z "$PY" ] && { [ -x /opt/anaconda3/bin/python3 ] && PY=/opt/anaconda3/bin/python3 || PY="$(command -v python3)"; }
echo "folder: $ROOT"
echo "python: $PY ($("$PY" --version 2>&1))   (the pins in requirements.txt were made on Python 3.13)"
[ -f "$ROOT/.env" ] || { echo "STOP: $ROOT/.env is missing (the API keys) - copy it from the old Mac first"; exit 1; }

"$PY" -m pip install -r "$ROOT/requirements.txt"
[ -f "$VG/requirements.txt" ] && "$PY" -m pip install -r "$VG/requirements.txt"
git -C "$ROOT" config filter.nbstripout.clean "\"$PY\" -m nbstripout"

PYTHON="$PY" bash "$ROOT/launchd/install_schedule.sh"
echo
launchctl list | grep -E 'stockanalysis|unicorn' || true
echo "Check:  $PY \"$ROOT/run_all.py\" --dry-run     and open http://localhost:8502"
