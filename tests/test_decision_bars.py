"""Decision days keep their 2:30 PM CT bar (backtest_engine.keep_decision_bars), so the record matches what was traded.
  - the 2:30 run of a decision day saves today's bar as ratios to the previous session and uses the bar as is
  - later runs (after the close, the next days) rebuild that day from the ratios, not from the final bar
  - a split later on (all history rescaled) gives the same bar in the new units
  - once the decision traded (run_state last_decision), a later run never overwrites it; a retry before the trade does
  - no saved file, a non-decision day, or a bad file: the bars come back unchanged (never raises)
Temporary files only. Run: python tests/run_tests.py  (or PYTHONPATH=. python tests/test_decision_bars.py)"""
import json
import os
import sys
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd

import backtest_engine as be

FAIL = []
CT = ZoneInfo("America/Chicago")

def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info and not ok else ""))
    if not ok:
        FAIL.append(name)

DAYS = pd.to_datetime(["2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02", "2026-10-05"])   # Mon 10/5 = mid-week check

def bars(mon_close, scale=1.0):
    """Two stocks; Monday's bar closes at mon_close (AAA) / mon_close / 2 (BBB). scale = a later split (0.5 = 2-for-1)."""
    rows = []
    for sym, k in (("AAA", 1.0), ("BBB", 0.5)):
        for i, d in enumerate(DAYS):
            c = (mon_close if i == 4 else 100.0 + i) * k
            rows.append({"Symbol": sym, "Date": d, "Open": (c - 1) * scale, "High": (c + 2) * scale, "Low": (c - 2) * scale,
                         "Close": c * scale, "Volume": (1000.0 + i) / scale})
    return pd.DataFrame(rows)

tmp = tempfile.mkdtemp()
P, S = os.path.join(tmp, "decision_bars.csv"), os.path.join(tmp, "run_state.json")
json.dump({"last_decision": "2026-10-02"}, open(S, "w"))
at = lambda d, h, m: datetime(2026, 10, d, h, m, tzinfo=CT)
run = lambda b, now: be.keep_decision_bars(b, now=now, path=P, state_path=S)

b230 = bars(110.0)
out = run(b230, at(5, 14, 30))
check("2:30 run on Mon (decision day, not traded): today's bar saved", os.path.exists(P)
      and sorted(pd.read_csv(P)["Symbol"]) == ["AAA", "BBB"] and (pd.read_csv(P)["Date"] == "2026-10-05").all())
check("2:30 run: its bars are used as is", out.equals(b230))
saved = pd.read_csv(P).set_index("Symbol")
check("saved as ratios to the previous session (AAA close 110 / 103)", abs(saved.at["AAA", "Close"] - 110 / 103) < 1e-12,
      saved.loc["AAA"].to_dict())

final = bars(112.0)                                  # the final close differs from 2:30
out = run(final, at(5, 17, 0))
mon = out["Date"] == DAYS[-1]
check("after the close: Monday rebuilt from the 2:30 bar, not the final close",
      np.allclose(out.loc[mon, be.BAR_COLS].to_numpy(), b230.loc[mon, be.BAR_COLS].to_numpy(), rtol=1e-12))
check("after the close: every other day untouched", out.loc[~mon].equals(final.loc[~mon]))
out = run(final, at(6, 9, 0))
check("next morning: still the 2:30 bar", np.allclose(out.loc[mon, "Close"], [110.0, 55.0]))

split = bars(112.0, scale=0.5)                       # a 2-for-1 split later: every past bar halved, volume doubled
out = run(split, at(9, 15, 0))
check("after a split: the 2:30 bar in the new units (close 55 / 27.5, volume x 2)",
      np.allclose(out.loc[mon, "Close"], [55.0, 27.5]) and np.allclose(out.loc[mon, "Volume"], b230.loc[mon, "Volume"] * 2))

before = open(P).read()
json.dump({"last_decision": "2026-10-05"}, open(S, "w"))
out = run(bars(111.0), at(5, 14, 50))                # a later run the same afternoon, after the decision traded
check("after the trade: a later 2:50 run never overwrites the saved bar", open(P).read() == before)
check("after the trade: the 2:50 run uses the 2:30 bar", np.allclose(out.loc[mon, "Close"], [110.0, 55.0]))
json.dump({"last_decision": "2026-10-02"}, open(S, "w"))
run(bars(111.0), at(5, 14, 50))                      # the trade failed and is retried: the retry's bar is the one traded
check("retry before the trade: the retry's bar replaces it", abs(pd.read_csv(P).set_index("Symbol").at["AAA", "Close"] - 111 / 103) < 1e-12)

os.remove(P)
b = bars(110.0)
check("no saved file: bars unchanged, nothing written (Tue 2:45 is no decision day)",
      run(b, at(6, 14, 45)).equals(b) and not os.path.exists(P))
check("before 2:30 on a decision day: nothing saved", run(b, at(5, 14, 0)).equals(b) and not os.path.exists(P))
open(P, "w").write("not,a\nvalid file")
check("a bad file: bars unchanged, no crash", run(b, at(6, 9, 0)).equals(b))
check("the live file lives in Reports/", be.DECISION_BARS_CSV == be.REPORTS_DIR / "decision_bars.csv")
print(f"\n{len(FAIL)} failed" if FAIL else "\nDECISION BARS OK")
sys.exit(1 if FAIL else 0)
