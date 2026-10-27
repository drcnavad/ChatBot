#!/bin/bash
# Dashboard data refresh (launchd com.stockanalysis.tradeaudit): Mon-Fri every 30 min 8:30 AM-3:00 PM CT + 3:45 PM CT.
# Read-only on the LIVE account (GET only, never orders):
#   trade_audit.py --write --log: order ledger, round trips, new alerts to the run log
cd "$(dirname "$0")" || exit 1
PY="${STOCK_ANALYSIS_PYTHON:-/opt/anaconda3/bin/python3}"
echo "=== $(date '+%a %b %d %I:%M %p %Z') dashboard data refresh"
exec "$PY" trade_audit.py --write --log
