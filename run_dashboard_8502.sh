#!/bin/bash
# Serve the Stock Analysis dashboard (app.py) on port 8502.
# Port 8501 is reserved for the other dashboard in the backup folder.
# Launched at login by ~/Library/LaunchAgents/com.stockanalysis.dashboard-8502.plist
# (KeepAlive restarts it if it crashes). Logs: Reports/logs/streamlit_8502.log
#
# NOTE: launchd runs with a minimal PATH, so use the absolute python that has
# streamlit installed instead of relying on PATH lookup.
cd "/Users/drcnavad/Desktop/Stock Analysis" || exit 1
exec /opt/anaconda3/bin/python -m streamlit run app.py --server.port 8502 --server.headless true
