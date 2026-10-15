# Moving Stock Analysis + Voice Genie to the Mac mini

## 1. Old Mac: stop every job (already done on Oct 15, 2026)
The old MacBook's jobs were removed on Oct 15, 2026 (copies of the plists are in `launchd/`), so nothing trades from
two Macs at once. If you ever need to do it again, paste this in Terminal:
```
for l in com.stockanalysis.evening com.stockanalysis.morning com.stockanalysis.tradeaudit \
         com.stockanalysis.earningsstop com.stockanalysis.dashboard-8502 com.unicorn.voice; do
  launchctl bootout gui/$(id -u)/$l 2>/dev/null; rm -f ~/Library/LaunchAgents/$l.plist
done
launchctl list | grep -E 'stockanalysis|unicorn'      # must print nothing
```

## 2. Copy
Copy BOTH folders from `~/Desktop` to the USB drive, then to `~/Desktop` on the Mac mini (side by side):
`Stock Analysis` (with the hidden `.env` and `.git`; in Finder press Cmd+Shift+. to see them) and `Voice Genie`.
Copy the whole folders, so the hidden files go too. Use the Terminal if you want: `cp -R ~/Desktop/"Stock Analysis" /Volumes/<USB>/`.
Format the USB as APFS or Mac OS Extended (exFAT is OK too, but FAT32 can break file names).

## 3. Mac mini: install
1. Install Anaconda (it goes to `/opt/anaconda3`, the same as the old Mac) or set `PYTHON=/path/to/python3`. For Voice Genie also:
   `brew install ollama portaudio` (see `Voice Genie/README.md`), and copy its `models` folder (it is inside the folder).
2. Run:
   ```
   cd ~/Desktop/"Stock Analysis" && bash setup_mac_mini.sh
   ```
   It installs the packages (both requirements.txt files), points git's notebook filter at that python, and installs the
   5 `com.stockanalysis.*` jobs + `com.unicorn.voice`, with paths taken from where the folders really are.
3. Check:
   ```
   launchctl list | grep -E 'stockanalysis|unicorn'      # 6 lines
   /opt/anaconda3/bin/python3 run_all.py --dry-run        # plan only, nothing runs
   /opt/anaconda3/bin/python3 tests/run_tests.py
   ```
   and open http://localhost:8502.

## 4. Mac mini settings
- **Stay awake:** System Settings > Energy: turn on "Prevent automatic sleeping when the display is off" and "Start up
  automatically after a power failure". Also run once: `sudo pmset repeat wakeorpoweron MTWRF 14:31:00`.
- **Log in automatically:** the jobs are per-user (LaunchAgents), so they run only while you are logged in. With
  FileVault on, the Mac waits at the password screen after a restart, and the jobs don't run until you log in. Either
  turn FileVault off and set automatic login (Users & Groups), or log in after every restart.
- **Permissions:** the first runs may ask for "python3" / "bash" to access the Desktop folder. Click Allow (or add
  them in Privacy & Security > Files and Folders / Full Disk Access). Voice Genie also needs Microphone (and Screen
  Recording if you use its screen tool).
- **Time zone:** America/Chicago (the schedule is in US Central time).
