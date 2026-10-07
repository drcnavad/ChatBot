"""The weekday dashboard refresh (launchd com.stockanalysis.refresh: run_all.py --quick --scheduled).
  - runs only on a trading day that is not a decision day, after the day's bar is final for it (3:05 PM CT); idle on weekends,
    holidays and decision days (Mon/Wed/Fri, Tue after a Monday holiday, Thu before a Friday holiday)
  - its plan is prices + main_signal_analysis.ipynb + checks: no trade, no fill check, no paid API step
  - the plist never passes --trade / --fill-check, runs Mon-Fri 3:00 PM CT, no RunAtLoad / StartInterval; started at
    3:00 it waits until 3:05 (refresh_wait_seconds) and its notebook takes today's bar from 4:05 PM ET
  - the forward test (com.stockanalysis.forwardtest) also starts Mon-Fri 3:00 PM CT with --record --wait: it waits for
    the bar (3:05 PM CT) and while any run_all.py runs (the refresh), so it never races the refresh
  - main_signal_analysis.ipynb (run by hand or by the job) contains no order code
Dry runs on a fake clock (--now): nothing runs, nothing is written. Run: python tests/run_tests.py"""
import contextlib
import io
import json
import os
import plistlib
import re
import sys
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


check("Tue 3:05 PM CT (no decision, bar final for the refresh): refresh runs", ra.refresh_idle(at("2026-10-06 15:05")) is None)
check("Thu 3:45 PM CT: refresh runs", ra.refresh_idle(at("2026-10-08 15:45")) is None)
check("Tue 3:00 PM CT start: waits 5 min for the final bar", ra.refresh_wait_seconds(at("2026-10-06 15:00")) == 300)
check("Tue 3:10 PM CT / 9 AM / 2:30 PM: no wait", [ra.refresh_wait_seconds(at(t)) for t in
      ("2026-10-06 15:10", "2026-10-06 09:00", "2026-10-06 14:30")] == [0, 0, 0])
check("Mon/Wed/Fri 3:00 PM CT: no wait (decision day, the job idles)", [ra.refresh_wait_seconds(at(t)) for t in
      ("2026-10-05 15:00", "2026-10-07 15:00", "2026-10-09 15:00")] == [0, 0, 0])
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


rc, out = dry("2026-10-06 15:00")   # the 3:00 PM start plans as of 3:05 (no sleep in a dry run)
check("Tue dry run: plans main_signal_analysis.ipynb in quick mode", rc == 0 and "main_signal_analysis.ipynb" in out
      and "mode QUICK" in out, out)
check("Tue dry run: no trade, no fill check, no paid API step",
      not re.search(r"\btrade\b|fill-check|NewsAPI|Finnhub|Alpha Vantage|company_report_autofetch|sentiment_analysis", out), out)
rc, out = dry("2026-10-07 15:00")
check("Wed dry run: idle (decision day), nothing planned", rc == 0 and out.startswith("idle:") and "decision day" in out, out)

job = plistlib.load(open(os.path.join(ROOT, "launchd", "com.stockanalysis.refresh.plist"), "rb"))
args = job["ProgramArguments"]
check("plist: run_all.py --quick --scheduled, never --trade / --fill-check",
      args[1].endswith("/run_all.py") and args[2:] == ["--quick", "--scheduled"], args)
check("plist: Mon-Fri 3:00 PM CT only (no RunAtLoad, no StartInterval)",
      sorted((d["Weekday"], d["Hour"], d["Minute"]) for d in job["StartCalendarInterval"]) == [(w, 15, 0) for w in range(1, 6)]
      and "RunAtLoad" not in job and "StartInterval" not in job, job)

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

# the forward test job: 3:00 PM CT, after the refresh
import forward_test as ft  # noqa: E402
fj = plistlib.load(open(os.path.join(ROOT, "launchd", "com.stockanalysis.forwardtest.plist"), "rb"))
check("forward test plist: forward_test.py --record --wait, Mon-Fri 3:00 PM CT, never order flags",
      fj["ProgramArguments"][1].endswith("/forward_test.py") and fj["ProgramArguments"][2:] == ["--record", "--wait"]
      and sorted((d["Weekday"], d["Hour"], d["Minute"]) for d in fj["StartCalendarInterval"]) == [(w, 15, 0) for w in range(1, 6)]
      and "RunAtLoad" not in fj, fj)
check("forward test: a 3:00 PM CT start waits 5 min for the bar, 3:06 / 9 AM / Saturday do not",
      ft.wait_seconds_for_bar(at("2026-10-06 15:00")) == 300
      and [ft.wait_seconds_for_bar(at(t)) for t in ("2026-10-06 15:06", "2026-10-07 09:00", "2026-10-10 15:00")] == [0, 0, 0])
slept, state = [], iter([True, True, True, False])
waited = ft.wait_for_inputs(sleep=slept.append, running=lambda: next(state, False), now=at("2026-10-08 15:00"))
check("forward test --wait: sleeps to 3:05, then while run_all.py runs (the refresh), then goes",
      slept == [300, 30, 30, 30] and waited == 90, (slept, waited))
slept = []
ft.wait_for_inputs(sleep=slept.append, running=lambda: True, now=at("2026-10-08 15:20"))
check("forward test --wait: a stuck pipeline is given at most MAX_PIPELINE_WAIT_MIN, then it goes ahead",
      sum(slept) == ft.MAX_PIPELINE_WAIT_MIN * 60, sum(slept))
check("forward test: fresh bars are taken from 3:05 PM CT (AFTER_CLOSE_BAR_MIN)",
      "close_buffer_min=be.AFTER_CLOSE_BAR_MIN" in open(os.path.join(ROOT, "forward_test.py")).read())
check("refresh starts no later than the forward test (same 3:00 PM CT slot; the forward test waits for it)",
      {(d["Hour"], d["Minute"]) for d in job["StartCalendarInterval"]} == {(d["Hour"], d["Minute"]) for d in fj["StartCalendarInterval"]})
ds = open(os.path.join(ROOT, "dashboard", "details", "forward_test.py")).read()
check("dashboard text says 3:00 PM CT (no stale 4:15)", "3:00 PM CT" in ds and "4:15" not in ds)

nb = json.load(open(os.path.join(ROOT, "main_signal_analysis.ipynb")))
code = "\n".join(line.split("#")[0] for c in nb["cells"] if c["cell_type"] == "code" for line in "".join(c["source"]).splitlines())
check("main_signal_analysis.ipynb has no order code (safe to run by hand: no trading imports or order calls)",
      not re.search(r"import paper_trade|from paper_trade|alpaca\.trading|TradingClient|submit_order|OrderRequest|cancel_order", code))
print(f"\n{len(FAIL)} failed" if FAIL else "\nREFRESH JOB OK")
sys.exit(1 if FAIL else 0)
