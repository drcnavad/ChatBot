#!/bin/bash
# Install (or refresh after editing) the Stock Analysis launchd jobs:  bash launchd/install_schedule.sh
#   com.stockanalysis.evening         the ONE daily run, Mon-Fri 2:30 PM + 3:05 PM CT, at login, on wake, every 30 min:
#                                     pipeline_watchdog.py --trade --scheduled. The decision on Mon/Wed/Fri at 2:30 PM
#                                     (holiday-shifted), then from 3:05 PM CT once per trading day the after-close step:
#                                     main_signal_analysis.ipynb with the day's final bar on non-decision days (Tue/Thu)
#                                     and the forward test (Reports/forward_test_daily.csv + forward_strategies*.csv);
#                                     never 4:05-4:30 PM CT (Alpaca books deposits about 4:15 PM CT)
#   com.stockanalysis.morning         Mon-Fri 9:00 AM CT, at login, on wake, every 30 min: pipeline_watchdog.py --fill-check --scheduled
#   com.stockanalysis.earningsstop    every 5 min + at login: earnings_stop.py --loop (LIVE earnings-day 5% drop stop;
#                                     30-second checks only while a held stock is in an earnings-day session, 3:00 AM-
#                                     7:00 PM CT on trading days; one loop at a time; otherwise exits after a quick look)
#   com.stockanalysis.tradeaudit      Mon-Fri 9:45 AM + 3:45 PM CT: trade_audit.py --write --log (read-only ledger, round
#                                     trips, reconciliation + alerts to the run log; GET only, never orders)
#   com.stockanalysis.dashboard-8502  always on (restarted if it stops): the app on http://localhost:8502
# run_all.py decides whether anything is due (a decision after its 2:30 PM slot, or a missed one caught up at the next
# regular session until the next slot); idle starts print one "idle:" line. run_state last_decision and lock files
# stop double runs, so extra starts are harmless.
# Uninstall: launchctl bootout gui/$(id -u)/<label>  and delete ~/Library/LaunchAgents/<label>.plist
# Retired on 2026-10-07 (folded into com.stockanalysis.evening): com.stockanalysis.refresh and com.stockanalysis.forwardtest.
# This script unloads and deletes them if they are still installed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
AGENTS="$HOME/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"
mkdir -p "$AGENTS" "$HERE/../Reports/logs"

for label in com.stockanalysis.refresh com.stockanalysis.forwardtest; do      # retired jobs
    launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
    rm -f "$AGENTS/$label.plist"
done

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
