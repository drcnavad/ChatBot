#!/bin/bash
# Login catch-up for the evening trade run. Executed at every Mac login by the
# com.stockanalysis.trade-catchup launchd agent. It fires `pipeline_watchdog.py --trade`
# (the watchdog wraps run_all.py: same arguments, plus transient-failure retries)
# ONLY when ALL of these hold:
#   - today is Mon / Wed / Fri,
#   - it is after 3:15 PM CT but before 7:00 PM CT (a catch-up this late may finish
#     its pipeline after extended hours end - in that case the orders are staged
#     for the next morning's fill check instead of being submitted),
#   - today's scheduled 3:15 PM run did not happen (no successful --trade today).
# run_all.py re-checks the trading-day calendar and the once-per-window guard
# itself, so a login can never cause a 4th weekly run or duplicate orders.
set -u
PROJ="$HOME/Desktop/Stock Analysis"
cd "$PROJ" || exit 0
mkdir -p "$PROJ/Reports/logs"                # never fail just because the logs folder is missing

DOW=$(TZ=America/Chicago date +%u)            # 1=Mon ... 7=Sun
case "$DOW" in 1|3|5) ;; *) exit 0 ;; esac    # Mon/Wed/Fri only

HHMM=$((10#$(TZ=America/Chicago date +%H%M))) # 10# : 08xx/09xx are decimal, not octal
if [ "$HHMM" -lt 1515 ] || [ "$HHMM" -ge 1900 ]; then exit 0; fi

TODAY=$(TZ=America/Chicago date +%F)
PY3="$(command -v python3 || command -v python)"
LAST_TRADE_DAY="$("$PY3" -c "
import json
try:
    print((json.load(open('Reports/run_state.json')).get('last_trade_at') or '')[:10])
except Exception:
    print('')
" 2>/dev/null)"
if [ "$LAST_TRADE_DAY" = "$TODAY" ]; then exit 0; fi   # today's run already happened

PY="$(command -v python || command -v python3)"
LOG="$PROJ/Reports/logs/launchd_catchup.log"
echo "$(TZ=America/Chicago date '+%F %T %Z') catch-up: missed evening trade - running pipeline_watchdog.py --trade" >> "$LOG"
"$PY" pipeline_watchdog.py --trade >> "$LOG" 2>&1
