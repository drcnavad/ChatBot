#!/bin/bash
# Install (or refresh after editing) the Stock Analysis launchd jobs:  bash launchd/install_schedule.sh
#   com.stockanalysis.evening         Mon-Fri 2:30 PM CT, at login, on wake, every 30 min: pipeline_watchdog.py --trade --scheduled
#   com.stockanalysis.morning         Mon-Fri 9:00 AM CT, at login, on wake, every 30 min: pipeline_watchdog.py --fill-check --scheduled
#   com.stockanalysis.refresh         Mon-Fri 3:45 PM CT, on wake: run_all.py --quick --scheduled (dashboard refresh on
#                                     non-decision trading days, normally Tue/Thu; no paid APIs, never orders)
#   com.stockanalysis.forwardtest     Mon-Fri 4:15 PM CT, on wake: forward_test.py --record (one row per trading day in
#                                     Reports/forward_test_daily.csv; read-only GETs, never orders; + the paper
#                                     strategies' days in Reports/forward_strategies*.csv from saved data + free daily bars)
#   com.stockanalysis.dashboard-8502  always on (restarted if it stops): the app on http://localhost:8502
# run_all.py decides whether anything is due (a decision after its 2:30 PM slot, or a missed one caught up at the next
# regular session until the next slot); idle starts print one "idle:" line. run_state last_decision and lock files
# stop double runs, so extra starts are harmless.
# Uninstall: launchctl bootout gui/$(id -u)/<label>  and delete ~/Library/LaunchAgents/<label>.plist
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
AGENTS="$HOME/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"
mkdir -p "$AGENTS" "$HERE/../Reports/logs"

for f in "$HERE"/com.stockanalysis.*.plist; do
    label="$(basename "$f" .plist)"
    plutil -lint "$f" >/dev/null
    launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
    cp "$f" "$AGENTS/"
    for try in 1 2 3 4 5; do         # bootout finishes asynchronously; retry briefly
        launchctl bootstrap "$DOMAIN" "$AGENTS/$label.plist" 2>/dev/null && break
        sleep 1
    done
    launchctl print "$DOMAIN/$label" >/dev/null && echo "loaded  $label"
done

echo
echo "To wake the Mac for the 2:30 PM run, paste this once in Terminal (asks for your password):"
echo "  sudo pmset repeat wakeorpoweron MTWRF 14:31:00"
echo "(1 minute after the slot on purpose: launchd runs the missed 2:30 job right on wake, before the Mac can doze off again)"
