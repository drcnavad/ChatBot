#!/bin/bash
# Serve the Stock Analysis dashboard (app.py) on http://localhost:8502.
# Started at login by launchd/com.stockanalysis.dashboard-8502.plist (KeepAlive restarts it if it stops).
# Logs: Reports/logs/streamlit_8502.log. launchd has a minimal PATH, so the anaconda python is used by full path.
cd "/Users/drcnavad/Desktop/Stock Analysis" || exit 1
exec /opt/anaconda3/bin/python -m streamlit run app.py --server.port 8502 --server.headless true
