#!/bin/bash
# Install (or refresh after editing) the Stock Analysis launchd jobs:  bash launchd/install_schedule.sh
#   com.stockanalysis.evening         Mon-Fri 3:15 PM CT + at login: pipeline_watchdog.py --trade --scheduled
#   com.stockanalysis.morning         Mon-Fri 9:00 AM CT + at login: pipeline_watchdog.py --fill-check
#   com.stockanalysis.dashboard-8502  always on (restarted if it stops): the app on http://localhost:8502
# A slot missed while the Mac sleeps runs once on wake; a slot missed while it is off runs at the next login
# (RunAtLoad). run_all.py decides whether anything is due and a lock file stops double runs, so extra starts are harmless.
# Uninstall: launchctl bootout gui/$(id -u)/<label>  and delete ~/Library/LaunchAgents/<label>.plist
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
AGENTS="$HOME/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"
mkdir -p "$AGENTS" "$HERE/../Reports/logs"

# Retired: the old login catch-up job (now RunAtLoad + run_all.py --scheduled).
launchctl bootout "$DOMAIN/com.stockanalysis.trade-catchup" 2>/dev/null || true
[ -f "$AGENTS/com.stockanalysis.trade-catchup.plist" ] && mv "$AGENTS/com.stockanalysis.trade-catchup.plist" "$HOME/.Trash/"

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
echo "To wake the Mac for the 3:15 PM run, paste this once in Terminal (asks for your password):"
echo "  sudo pmset repeat wakeorpoweron MTWRF 15:05:00"
