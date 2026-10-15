#!/bin/bash
# Serve the Stock Analysis dashboard (app.py) on http://localhost:8502.
# Started at login by launchd/com.stockanalysis.dashboard-8502.plist (KeepAlive restarts it if it stops).
# Logs: Reports/logs/streamlit_8502.log. launchd has a minimal PATH, so python comes by full path: STOCK_ANALYSIS_PYTHON
# (set in the plist by launchd/install_schedule.sh), else the anaconda python.
cd "$(dirname "$0")" || exit 1
exec "${STOCK_ANALYSIS_PYTHON:-/opt/anaconda3/bin/python}" -m streamlit run app.py --server.port 8502 --server.headless true
