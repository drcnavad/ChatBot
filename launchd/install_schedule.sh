#!/bin/bash
# Install (or refresh after editing) the launchd jobs:  bash launchd/install_schedule.sh   [--check = render + lint only]
#   com.stockanalysis.evening         the ONE daily run, Mon-Fri 11:00 AM + 2:30 PM + 3:05 PM CT, at login, on wake, every 30 min:
#                                     pipeline_watchdog.py --trade --scheduled. The decision on Mon-Fri at 2:30 PM
#                                     (holiday-shifted), then from 3:05 PM CT once per trading day the after-close step:
#                                     main_signal_analysis.ipynb with the day's final bar on non-decision days (none while Mon-Fri all decide)
#                                     and the forward test (Reports/forward_test_daily.csv + forward_strategies*.csv);
#                                     never 4:05-4:30 PM CT (Alpaca books deposits about 4:15 PM CT)
#   com.stockanalysis.support         Mon 1:00, Tue 1:07, Wed 1:14, Thu 1:21, Fri 1:28 PM CT: run_all.py --support --scheduled
#                                     (company reports, news sentiment, earnings dates: the quota APIs, each run > 24 h
#                                     after the last; trading days 12:30-2:00 PM only, never beside a trade run)
#   com.stockanalysis.morning         Mon-Fri 9:00 AM CT, at login, on wake, every 30 min: pipeline_watchdog.py --fill-check --scheduled
#   com.stockanalysis.drop_guard        every 15 min + at login: drop_guard.py --scheduled (LIVE intraday AI drop guard ~8:45 AM-2:15 PM CT; sells when mid is >=4.1% below prior close and the LLM agrees at >=0.75 confidence; optional replacement buy with proceeds; kill switch + dry_run in Reports/drop_guard_config.json)
#   com.stockanalysis.earningsstop    every 5 min + at login: earnings_stop.py --loop (LIVE earnings-day 5% drop stop;
#                                     30-second checks only while a held stock is in an earnings-day session, 3:00 AM-
#                                     7:00 PM CT on trading days; one loop at a time; otherwise exits after a quick look)
#   com.stockanalysis.tradeaudit      Mon-Fri every 30 min 8:30 AM-3:00 PM + 3:45 PM CT: trade_audit.py --write --log (read-only ledger, round
#                                     trips, reconciliation + alerts to the run log; GET only, never orders)
#   com.stockanalysis.dashboard-8502  always on (restarted if it stops): the app on http://localhost:8502
#   com.unicorn.voice                 Voice Genie (Unicorn), template in ../Voice Genie/launchd/ (when that folder exists)
# The plists in this folder are templates written on the old Mac: their paths (/Users/drcnavad/Desktop/..., /opt/anaconda3)
# are rewritten to this folder's real location and the python in $PYTHON (default: /opt/anaconda3/bin/python3, else
# python3 on PATH) before they are copied to ~/Library/LaunchAgents. run_all.py decides whether anything is due; run_state
# last_decision and lock files stop double runs, so extra starts are harmless.
# Uninstall: launchctl bootout gui/$(id -u)/<label>  and delete ~/Library/LaunchAgents/<label>.plist
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
VG="$(dirname "$ROOT")/Voice Genie"
PY="${PYTHON:-}"
[ -z "$PY" ] && { [ -x /opt/anaconda3/bin/python3 ] && PY=/opt/anaconda3/bin/python3 || PY="$(command -v python3)"; }
AGENTS="$HOME/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"
OUT="$AGENTS"
[ "${1:-}" = "--check" ] && OUT="$(mktemp -d)"
mkdir -p "$OUT" "$ROOT/Reports/logs"

render() {   # template -> $OUT/<label>.plist with this Mac's paths
    "$PY" - "$1" "$OUT/$(basename "$1")" "$ROOT" "$VG" "$PY" <<'PYEOF'
import os, plistlib, re, sys
src, dst, root, vg, py = sys.argv[1:]
subs = [("/Users/drcnavad/Desktop/Stock Analysis", root), ("/Users/drcnavad/Desktop/Voice Genie", vg),
        ("/opt/anaconda3/bin", os.path.dirname(py))]
def fix(x):
    if isinstance(x, str):
        if x in ("/opt/anaconda3/bin/python", "/opt/anaconda3/bin/python3"):
            return py
        for a, b in subs:
            x = x.replace(a, b)
        return x
    if isinstance(x, list):
        return [fix(v) for v in x]
    if isinstance(x, dict):
        return {k: fix(v) for k, v in x.items()}
    return x
p = fix(plistlib.loads(re.sub(rb"<!--.*?-->", b"", open(src, "rb").read(), flags=re.S)))   # comments: launchd-only
env = p.setdefault("EnvironmentVariables", {})
env["STOCK_ANALYSIS_PYTHON"] = py                 # run_dashboard_8502.sh
if p["Label"] == "com.unicorn.voice":
    env["VOICE_GENIE_STOCK_DIR"] = root           # Voice Genie's own setting (config.py), no code change
plistlib.dump(p, open(dst, "wb"))
PYEOF
    plutil -lint "$OUT/$(basename "$1")" >/dev/null
}

templates=("$HERE"/com.stockanalysis.*.plist)
[ -f "$VG/launchd/com.unicorn.voice.plist" ] && templates+=("$VG/launchd/com.unicorn.voice.plist")   # its template lives in Voice Genie
for f in "${templates[@]}"; do
    label="$(basename "$f" .plist)"
    if [ "$OUT" != "$AGENTS" ]; then render "$f"; echo "ok  $OUT/$label.plist"; continue; fi
    launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
    render "$f"
    for try in 1 2 3 4 5; do         # bootout finishes asynchronously; retry briefly
        launchctl bootstrap "$DOMAIN" "$AGENTS/$label.plist" 2>/dev/null && break
        sleep 1
    done
    launchctl print "$DOMAIN/$label" >/dev/null && echo "loaded  $label"
done
[ "$OUT" != "$AGENTS" ] && exit 0

echo
echo "To wake the Mac for the 2:30 PM run, paste this once in Terminal (asks for your password):"
echo "  sudo pmset repeat wakeorpoweron MTWRF 14:31:00"
echo "(1 minute after the slot on purpose: launchd runs the missed 2:30 job right on wake, before the Mac can doze off again)"
