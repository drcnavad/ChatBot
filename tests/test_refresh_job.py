"""The weekday dashboard refresh (launchd com.stockanalysis.refresh: run_all.py --quick --scheduled).
  - runs only on a trading day that is not a decision day, after the day's bar is final (3:30 PM CT); idle on weekends,
    holidays and decision days (Mon/Wed/Fri, Tue after a Monday holiday, Thu before a Friday holiday)
  - its plan is prices + main_signal_analysis.ipynb + checks: no trade, no fill check, no paid API step
  - the plist never passes --trade / --fill-check, runs Mon-Fri 3:45 PM CT, no RunAtLoad / StartInterval
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


check("Tue 3:45 PM CT (no decision, bar final): refresh runs", ra.refresh_idle(at("2026-10-06 15:45")) is None)
check("Thu 3:45 PM CT: refresh runs", ra.refresh_idle(at("2026-10-08 15:45")) is None)
for when, word, what in (("2026-10-06 15:20", "not final", "Tue 3:20 PM CT: bar not final yet"),
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


rc, out = dry("2026-10-06 15:45")
check("Tue dry run: plans main_signal_analysis.ipynb in quick mode", rc == 0 and "main_signal_analysis.ipynb" in out
      and "mode QUICK" in out, out)
check("Tue dry run: no trade, no fill check, no paid API step",
      not re.search(r"\btrade\b|fill-check|NewsAPI|Finnhub|Alpha Vantage|company_report_autofetch|sentiment_analysis", out), out)
rc, out = dry("2026-10-07 15:45")
check("Wed dry run: idle (decision day), nothing planned", rc == 0 and out.startswith("idle:") and "decision day" in out, out)

job = plistlib.load(open(os.path.join(ROOT, "launchd", "com.stockanalysis.refresh.plist"), "rb"))
args = job["ProgramArguments"]
check("plist: run_all.py --quick --scheduled, never --trade / --fill-check",
      args[1].endswith("/run_all.py") and args[2:] == ["--quick", "--scheduled"], args)
check("plist: Mon-Fri 3:45 PM CT only (no RunAtLoad, no StartInterval)",
      sorted((d["Weekday"], d["Hour"], d["Minute"]) for d in job["StartCalendarInterval"]) == [(w, 15, 45) for w in range(1, 6)]
      and "RunAtLoad" not in job and "StartInterval" not in job, job)

nb = json.load(open(os.path.join(ROOT, "main_signal_analysis.ipynb")))
code = "\n".join(line.split("#")[0] for c in nb["cells"] if c["cell_type"] == "code" for line in "".join(c["source"]).splitlines())
check("main_signal_analysis.ipynb has no order code (safe to run by hand: no trading imports or order calls)",
      not re.search(r"import paper_trade|from paper_trade|alpaca\.trading|TradingClient|submit_order|OrderRequest|cancel_order", code))
print(f"\n{len(FAIL)} failed" if FAIL else "\nREFRESH JOB OK")
sys.exit(1 if FAIL else 0)
