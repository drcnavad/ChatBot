"""Earnings planner (earnings_planner.py + app.render_earnings_planner) with FAKE bars and calendar (no network):
  - held stocks reporting within 14 days, sorted by date; none in range -> the next 3 upcoming instead
  - time label (before open / after close / not set), days until, stop window (opens 7 days before, active through the
    reaction day: AM = that day, PM = the next trading day)
  - stop = highest close since the first buy - 3 x ATR(14) (independent Wilder ATR here) and its % below the last price
  - past reactions: gap = reaction-day open vs the close before, 5-day = 5th trading day's close vs that close; reports
    without bars are not counted; medians
  - the local bar caches are combined (newer file wins), nothing is downloaded
Run: python tests/run_tests.py  (or python tests/test_earnings_planner.py)"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
import pandas as pd

import earnings_planner as ep
import earnings_stop as es

FAIL = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info and not ok else ""))
    if not ok:
        FAIL.append(name)


def close(a, b, tol=1e-6):
    return a == a and b == b and abs(a - b) <= tol


days = pd.bdate_range("2025-06-02", "2026-10-02")
rows = []
for sym, base in (("AAA", 100.0), ("BBB", 50.0), ("CCC", 20.0), ("DDD", 80.0)):
    for i, d in enumerate(days):
        c = base + 5 * np.sin(i / 9) + i * 0.05
        rows.append({"Symbol": sym, "Date": d, "Open": c - 0.2, "High": c + 1.0 + (i % 3) * 0.3, "Low": c - 1.1, "Close": c, "Volume": 1})
bars = pd.DataFrame(rows)


def setbar(sym, d, **kw):
    m = (bars["Symbol"] == sym) & (bars["Date"] == pd.Timestamp(d))
    for k, v in kw.items():
        bars.loc[m, k] = v


# AAA: Jan 15, 2026 after close (reaction Fri Jan 16), Apr 16, 2026 before open (reaction Apr 16), next Oct 12 after close
prev = lambda sym, d: float(bars[(bars.Symbol == sym) & (bars.Date < pd.Timestamp(d))].Close.iloc[-1])
p1, p2 = prev("AAA", "2026-01-16"), prev("AAA", "2026-04-16")
setbar("AAA", "2026-01-16", Open=p1 * 1.06)
setbar("AAA", "2026-01-22", Close=p1 * 1.10)          # Jan 16, 19 (bdate), 20, 21, 22 -> 5th day
setbar("AAA", "2026-04-16", Open=p2 * 0.96)
setbar("AAA", "2026-04-22", Close=p2 * 0.93)          # Apr 16, 17, 20, 21, 22
earn = pd.DataFrame([("AAA", "2026-01-15", "PM"), ("AAA", "2026-04-16", "AM"), ("AAA", "2026-10-12", "PM"),
                     ("AAA", "2024-01-10", "PM"),                                     # before the bars: not counted
                     ("BBB", "2026-11-30", "AM"), ("CCC", "2026-12-03", None), ("DDD", "2026-11-05", "PM"),
                     ("DDD", "2027-02-05", "PM"), ("ZZZ", "2026-10-06", "AM")], columns=["Symbol", "Earnings Date", "Time"])
earn["Earnings Date"] = pd.to_datetime(earn["Earnings Date"])
held = {"AAA": 110.0, "BBB": 60.0, "CCC": 25.0, "DDD": 90.0, "EEE": 10.0}
entry = {s: pd.Timestamp("2026-08-03").date() for s in held}

t, in_range = ep.plan(held, entry, earn, bars, pd.Timestamp("2026-10-04"))
check("in range: only AAA (Oct 12 within 14 days); ZZZ is not held", in_range and list(t["Stock"]) == ["AAA"], t.to_dict("list"))
a = t.iloc[0]
check("AAA: after close, 8 days, window opens Mon Oct 5", a["Time"] == "After close" and a["Days"] == 8 and a["Stop window"] == "Opens Mon Oct 5", a.to_dict())

# independent Wilder ATR(14) + peak close since entry
b = bars[(bars.Symbol == "AAA") & (bars.Date < pd.Timestamp("2026-10-04"))].reset_index(drop=True)
tr = [b.High[0] - b.Low[0]] + [max(b.High[i] - b.Low[i], abs(b.High[i] - b.Close[i - 1]), abs(b.Low[i] - b.Close[i - 1])) for i in range(1, len(b))]
atr = None
for i, x in enumerate(tr):
    atr = x if i == 0 else atr + (x - atr) / 14            # ewm(alpha=1/14, adjust=False)
peak = b.loc[b.Date >= pd.Timestamp("2026-08-03"), "Close"].max()
check("stop = peak close since entry - 3 x ATR(14) (independent calculation)", close(a["Stop"], peak - 3 * atr, 1e-6) and es.K_ATR == 3.0,
      (a["Stop"], peak - 3 * atr))
check("To stop % = (last - stop) / last", close(a["To stop %"], (110 - a["Stop"]) / 110 * 100))
check("AAA reactions: 2 reports with bars (2024 one has none); median gap = mean of +6% and -4% = +1%, 5-day = mean of +10% / -7% = +1.5%",
      a["Reports"] == 2 and close(a["Median gap %"], 1.0, 1e-6) and close(a["Median 5-day %"], 1.5, 1e-6), a.to_dict())
r = ep.reactions("AAA", earn, bars, pd.Timestamp("2026-10-04"))
check("reaction days: PM Jan 15 -> Jan 16 (next trading day), AM Apr 16 -> same day",
      [round(g, 6) for _, g, _ in r] == [6.0, -4.0], r)

t2, _ = ep.plan(held, entry, earn, bars, pd.Timestamp("2026-10-06"))
check("Oct 6: AAA window active through Tue Oct 13 (after-close report reacts the next trading day)",
      t2.iloc[0]["Stop window"] == "Active through Tue Oct 13" and t2.iloc[0]["Days"] == 6, t2.iloc[0].to_dict())
t3, in3 = ep.plan(held, entry, earn, bars, pd.Timestamp("2026-10-14"))
check("none in range (Oct 14): the next 3 upcoming, by date: DDD Nov 5, BBB Nov 30, CCC Dec 3 (EEE has no date)",
      not in3 and list(t3["Stock"]) == ["DDD", "BBB", "CCC"], t3.to_dict("list"))
check("time labels: AM = Before open, PM = After close, missing = Time not set",
      list(t3["Time"]) == ["After close", "Before open", "Time not set"])
check("labels", ep.time_label("am") == "Before open" and ep.time_label(float("nan")) == "Time not set")
t4, in4 = ep.plan({"EEE": 10.0}, {}, earn, bars, pd.Timestamp("2026-10-04"))
check("no upcoming date for any held stock: empty table", t4.empty and not in4)
t5, _ = ep.plan({"AAA": 110.0}, {}, earn, bars, pd.Timestamp("2026-10-04"))
check("no entry date: no stop (blank), the rest still shown", len(t5) == 1 and t5["Stop"].isna().all() and t5["Reports"][0] == 2)

# caches: combined, newer file wins, no download
d = tempfile.mkdtemp()
old = bars[bars.Symbol == "AAA"].head(30).copy()
new = old.tail(10).copy()
new["Close"] = new["Close"] + 1000
old.to_pickle(os.path.join(d, ep.CACHE_FILES[0]))
new.to_pickle(os.path.join(d, ep.CACHE_FILES[1]))
c = ep.cached_bars(d)
check("cached bars: both files combined, the forward cache wins on overlap", len(c) == 30 and (c.tail(10)["Close"] > 1000).all()
      and (c.head(20)["Close"] < 1000).all())
check("cached bars: no files -> empty frame", ep.cached_bars(tempfile.mkdtemp()).empty)
src = open(os.path.join(ROOT, "earnings_planner.py")).read()
check("earnings_planner.py downloads nothing and sends nothing", not any(w in src for w in ("fetch_daily_bars", "submit_order", "requests", "urlopen")))
print(f"\n{len(FAIL)} failed" if FAIL else "\nEARNINGS PLANNER OK")
sys.exit(1 if FAIL else 0)
