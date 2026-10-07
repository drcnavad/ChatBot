"""The ONE daily run (launchd com.stockanalysis.evening: pipeline_watchdog.py --trade --scheduled, Mon-Fri 2:30 PM + 3:05 PM CT).
  - the decision part is unchanged (Mon/Wed/Fri 2:30 PM; test_catch_up / test_runner cover it)
  - from 3:05 PM CT, once per trading day, the after-close step (run_all.daily_step): the signal refresh
    (run_all.py --quick --scheduled, main_signal_analysis.ipynb with today's final bar) then forward_test.py --record;
    never 4:05-4:30 PM CT (Alpaca books deposits about 4:15 PM CT); idle on weekends / holidays / when done today
  - the refresh runs only on a trading day that is not a decision day (Mon/Wed/Fri, Tue after a Monday holiday, Thu
    before a Friday holiday: the 2:30 PM run wrote those rows); its plan is prices + main + checks: no trade, no paid API
  - the separate refresh and forward-test launchd jobs are retired; no job starts 4:05-4:30 PM CT
  - main_signal_analysis.ipynb (run by hand or by the run) contains no order code
Fake clocks and fake child processes: nothing real runs, nothing real is written. Run: python tests/run_tests.py"""
import contextlib
import io
import json
import os
import plistlib
import re
import sys
import tempfile
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import run_all as ra

FAIL = []

def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info and not ok else ""))
    if not ok:
        FAIL.append(name)

def at(s):
    return datetime.fromisoformat(s).replace(tzinfo=ra.CT)

# ---------------------------------------------------------------- the signal refresh (child of the after-close step)
check("Tue 3:05 PM CT (no decision, bar final for the refresh): refresh runs", ra.refresh_idle(at("2026-10-06 15:05")) is None)
check("Thu 3:45 PM CT: refresh runs", ra.refresh_idle(at("2026-10-08 15:45")) is None)
for when, word, what in (("2026-10-06 15:03", "not final", "Tue 3:03 PM CT: bar not final yet"),
                         ("2026-10-05 15:45", "decision day", "Mon (mid-week check): the 2:30 run refreshes"),
                         ("2026-10-09 15:45", "decision day", "Fri (rebalance): the 2:30 run refreshes"),
                         ("2026-10-03 15:45", "closed", "Saturday: closed"),
                         ("2026-11-26 15:45", "closed", "Thanksgiving: closed"),
                         ("2026-09-08 15:45", "decision day", "Tue after Labor Day (the week's check): decision day"),
                         ("2027-03-25 15:45", "decision day", "Thu before Good Friday (the rebalance): decision day")):
    why = ra.refresh_idle(at(when)) or ""
    check(f"{what} -> idle", word in why, why)

def dry(now):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = ra.main(["--quick", "--scheduled", "--dry-run", "--now", now])
    return rc, out.getvalue()

rc, out = dry("2026-10-06 15:05")
check("Tue dry run: plans main_signal_analysis.ipynb in quick mode", rc == 0 and "main_signal_analysis.ipynb" in out
      and "mode QUICK" in out, out)
check("Tue dry run: no trade, no fill check, no paid API step",
      not re.search(r"\btrade\b|fill-check|NewsAPI|Finnhub|Alpha Vantage|company_report_autofetch|sentiment_analysis", out), out)
rc, out = dry("2026-10-07 15:05")
check("Wed dry run: idle (decision day), nothing planned", rc == 0 and out.startswith("idle:") and "decision day" in out, out)

# ---------------------------------------------------------------- when the after-close step runs
for when, state, word, what in (
        ("2026-10-06 15:04", {}, "3:05 PM", "Tue 3:04 PM CT: not yet (bar final at 3:05)"),
        ("2026-10-06 16:10", {}, "4:15", "Tue 4:10 PM CT: Alpaca deposit booking window - waits"),
        ("2026-10-06 16:29", {}, "4:15", "Tue 4:29 PM CT: still the deposit window"),
        ("2026-10-06 17:00", {"last_daily_date": "2026-10-06"}, "already ran", "done today: idle"),
        ("2026-10-10 15:30", {}, "closed", "Saturday: closed"),
        ("2026-11-26 15:30", {}, "closed", "Thanksgiving: closed")):
    why = ra.daily_idle(at(when), state) or ""
    check(f"after-close step: {what}", word in why, why)
for when, what in (("2026-10-06 15:05", "Tue 3:05 PM CT"), ("2026-10-07 15:05", "Wed 3:05 PM CT (decision day: forward test)"),
                   ("2026-10-06 16:04", "Tue 4:04 PM CT"), ("2026-10-06 16:30", "Tue 4:30 PM CT (after the deposit window)"),
                   ("2026-10-06 21:00", "Tue 9 PM CT (a late wake)")):
    check(f"after-close step due: {what}", ra.daily_idle(at(when), {"last_daily_date": "2026-10-05"}) is None)

class Done:
    def __init__(self, rc):
        self.returncode = rc

tmp = tempfile.mkdtemp()
saved = {k: getattr(ra, k) for k in ("STATE_FILE", "DAILY_LOCK")}
ra.STATE_FILE, ra.DAILY_LOCK = os.path.join(tmp, "run_state.json"), os.path.join(tmp, ".daily_step.lock")
try:
    json.dump({"last_decision": "2026-10-07"}, open(ra.STATE_FILE, "w"))
    calls = []
    with contextlib.redirect_stdout(io.StringIO()) as out:
        rc = ra.daily_step(at("2026-10-08 15:05"), run=lambda cmd, cwd=None: calls.append(cmd[1:]) or Done(1 if len(calls) == 1 else 0))
    st = json.load(open(ra.STATE_FILE))
    check("daily_step: the signal refresh first (run_all.py --quick --scheduled), then forward_test.py --record; no order flags",
          [[os.path.basename(c[0])] + c[1:] for c in calls] == [["run_all.py", "--quick", "--scheduled"], ["forward_test.py", "--record"]]
          and not any(f in sum(calls, []) for f in ("--trade", "--fill-check", "--submit")), calls)
    check("daily_step: marked done for the day (last_daily_date) even when a part failed (it reported itself); state kept",
          rc == 0 and st.get("last_daily_date") == "2026-10-08" and st.get("last_decision") == "2026-10-07"
          and "refresh exit 1, forward test exit 0" in out.getvalue(), (st, out.getvalue()))
    check("daily_step: lock released afterwards", not os.path.exists(ra.DAILY_LOCK))
    ra.acquire_trade_lock(ra.DAILY_LOCK)                 # another after-close step holds the lock (this process)
    calls = []
    with contextlib.redirect_stdout(io.StringIO()) as out:
        ra.daily_step(at("2026-10-08 15:35"), run=lambda cmd, cwd=None: calls.append(cmd) or Done(0))
    check("daily_step: one at a time (a running step's lock -> idle, nothing started)",
          calls == [] and out.getvalue().startswith("idle:") and "already running" in out.getvalue(), out.getvalue())
    ra.release_trade_lock(ra.DAILY_LOCK)
finally:
    for k, v in saved.items():
        setattr(ra, k, v)

# main(): a scheduled start with no decision due runs the after-close step when it is due (real clock only)
saved = {k: getattr(ra, k) for k in ("decision_gate", "superseded_decision", "daily_idle", "daily_step", "load_state")}
try:
    ra.decision_gate = lambda now, state, scheduled=False: (None, None, "nothing due")
    ra.superseded_decision = lambda now, state: None
    ra.load_state = lambda path=None: {}
    ran = []
    ra.daily_step = lambda now: ran.append(now) or 0
    for idle, n in ((None, 1), ("the after-close step already ran today", 0)):
        ran.clear()
        ra.daily_idle = lambda now, state, _i=idle: _i
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rc = ra.main(["--trade", "--scheduled"])
        check(f"main --trade --scheduled, nothing due, step {'due' if n else 'not due'}: idle line"
              f"{', then the after-close step' if n else ' only'}",
              rc == 0 and out.getvalue().startswith("idle:") and len(ran) == n, (out.getvalue(), ran))
    ra.daily_idle = lambda now, state: None
    ran.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        ra.main(["--trade", "--scheduled", "--now", "2026-10-06 15:30"])
    check("main with the tests' fake clock (--now) never starts the after-close step", ran == [], ran)
finally:
    for k, v in saved.items():
        setattr(ra, k, v)
src = open(os.path.join(ROOT, "run_all.py")).read()
check("a scheduled decision run that ends after 3:05 PM CT runs the after-close step itself",
      "if daily_idle(later, load_state()) is None:" in src and "daily_step(later)" in src)

# ---------------------------------------------------------------- launchd: one daily run, retired jobs gone
job = plistlib.load(open(os.path.join(ROOT, "launchd", "com.stockanalysis.evening.plist"), "rb"))
args = job["ProgramArguments"]
check("evening plist: pipeline_watchdog.py --trade --scheduled (unchanged command)",
      args[1].endswith("/pipeline_watchdog.py") and args[2:] == ["--trade", "--scheduled"], args)
check("evening plist: Mon-Fri 2:30 PM (decision) + 3:05 PM CT (after-close step), RunAtLoad, every 30 min",
      sorted((d["Weekday"], d["Hour"], d["Minute"]) for d in job["StartCalendarInterval"])
      == sorted([(w, 14, 30) for w in range(1, 6)] + [(w, 15, 5) for w in range(1, 6)])
      and job.get("RunAtLoad") is True and job.get("StartInterval") == 1800, job)
names = sorted(os.listdir(os.path.join(ROOT, "launchd")))
check("retired jobs removed: no refresh / forwardtest plist",
      "com.stockanalysis.refresh.plist" not in names and "com.stockanalysis.forwardtest.plist" not in names, names)
sh = open(os.path.join(ROOT, "launchd", "install_schedule.sh")).read()
check("install_schedule.sh unloads and deletes the retired jobs",
      "for label in com.stockanalysis.refresh com.stockanalysis.forwardtest" in sh and 'rm -f "$AGENTS/$label.plist"' in sh)
plist = lambda n: plistlib.loads(re.sub(rb"<!--.*?-->", b"", open(os.path.join(ROOT, "launchd", n), "rb").read(), flags=re.S))
slots = [(n, d["Hour"], d["Minute"]) for n in names if n.endswith(".plist") for d in (plist(n).get("StartCalendarInterval") or [])]
check("no launchd slot in the 4:05-4:30 PM CT deposit booking window",
      not [x for x in slots if (16, 5) <= (x[1], x[2]) < (16, 30)], slots)
morning = plist("com.stockanalysis.morning.plist")
check("9 AM fill check unchanged (Mon-Fri 9:00 AM CT, --fill-check --scheduled)",
      morning["ProgramArguments"][2:] == ["--fill-check", "--scheduled"]
      and sorted((d["Weekday"], d["Hour"], d["Minute"]) for d in morning["StartCalendarInterval"]) == [(w, 9, 0) for w in range(1, 6)])

# the bar acceptance: default 30 min after the close for every job; 5 min (4:05 PM ET) only where a job asks for it
import pandas as pd  # noqa: E402
import backtest_engine as be  # noqa: E402
bars = pd.DataFrame({"Date": pd.to_datetime(["2026-10-05", "2026-10-06"]), "Symbol": "AAPL", "Close": [1.0, 2.0]})
et = lambda s: datetime.fromisoformat(s).replace(tzinfo=be.EASTERN)
old_env = os.environ.pop(be.BAR_FINAL_ENV, None)
check("default: Tue 4:10 PM ET drops today's bar (final at 4:30 PM ET, unchanged)",
      be.drop_partial_last_bar(bars, now=et("2026-10-06 16:10"))[1] is True)
check("after-close jobs: Tue 4:05 PM ET keeps today's bar", be.drop_partial_last_bar(
      bars, now=et("2026-10-06 16:05"), close_buffer_min=be.AFTER_CLOSE_BAR_MIN)[1] is False)
os.environ[be.BAR_FINAL_ENV] = str(be.AFTER_CLOSE_BAR_MIN)
check("refresh env (set by run_all for its notebook): 4:05 PM ET keeps it, 4:04 drops it",
      be.drop_partial_last_bar(bars, now=et("2026-10-06 16:05"))[1] is False
      and be.drop_partial_last_bar(bars, now=et("2026-10-06 16:04"))[1] is True)
os.environ.pop(be.BAR_FINAL_ENV)
if old_env is not None:
    os.environ[be.BAR_FINAL_ENV] = old_env

import forward_test as ft  # noqa: E402
check("forward test: fresh bars are taken from 3:05 PM CT (AFTER_CLOSE_BAR_MIN)",
      "close_buffer_min=be.AFTER_CLOSE_BAR_MIN" in open(os.path.join(ROOT, "forward_test.py")).read())
check("forward test: no --wait guard any more (the daily run orders the steps)", not hasattr(ft, "wait_for_inputs"))
ds = open(os.path.join(ROOT, "dashboard", "details", "forward_test.py")).read()
check("dashboard text says 3:05 PM CT (no stale 3:00 PM job / 4:15)", "3:05 PM CT" in ds and "3:00 PM CT job" not in ds and "4:15" not in ds)

nb = json.load(open(os.path.join(ROOT, "main_signal_analysis.ipynb")))
code = "\n".join(line.split("#")[0] for c in nb["cells"] if c["cell_type"] == "code" for line in "".join(c["source"]).splitlines())
check("main_signal_analysis.ipynb has no order code (safe to run by hand: no trading imports or order calls)",
      not re.search(r"import paper_trade|from paper_trade|alpaca\.trading|TradingClient|submit_order|OrderRequest|cancel_order", code))
print(f"\n{len(FAIL)} failed" if FAIL else "\nDAILY RUN OK")
sys.exit(1 if FAIL else 0)
