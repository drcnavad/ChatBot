#!/bin/bash
# Dashboard data refresh (launchd com.stockanalysis.tradeaudit): Mon-Fri every 30 min 8:30 AM-3:00 PM CT + 3:45 PM CT.
# Both steps are read-only on the LIVE account (GET only, never orders):
#   1. live sync = run_all.sync_live() -> alpaca_paper.sync_paper_account(): my_positions.csv + Reports/live_*.csv
#      (called directly, not `run_all.py --sync-live`, which would also re-run the signal notebook)
#   2. trade_audit.py --write --log: order ledger, round trips, new alerts to the run log
cd "$(dirname "$0")" || exit 1
PY="${STOCK_ANALYSIS_PYTHON:-/opt/anaconda3/bin/python3}"
echo "=== $(date '+%a %b %d %I:%M %p %Z') dashboard data refresh"
"$PY" -c "
import logging, run_all
logging.basicConfig(level=logging.INFO, format='%(message)s')
run_all.sync_live()" || echo "live sync failed (audit still runs)"
exec "$PY" trade_audit.py --write --log
