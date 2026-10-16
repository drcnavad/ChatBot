"""forward_combos (Chirag, Oct 15, 2026 evening): the forward-test lines with two independent halves, swap-only halves and
the low-volatility keep / Friday falling rules; the live line renamed "Live 50/50 combined (no overlap)" and switched to
the 50/50 at the Oct 16 close; the Live vs benchmark card's numbers (forward_test.live_vs_benchmark). Real saved data,
temp copies of the forward files (the real Reports files are never written); no broker calls, no orders.

Covers: deterministic replay (two runs equal; a shorter replay is a prefix of the longer one = saved days never change);
no overlap between the halves at any close; at most 10 names and 20% of its half per stock; never below zero cash;
do_not_buy never held; the 63-day sector score = forward_test's Sector rotation score; values 1.0 at the start close;
update_strategies appends the new lines and migrates the old live name; the live switch (shown on Oct 13 so the saved data
covers it): the Oct 12 saved day unchanged, the switch-day value = the old holdings marked to that close minus the sale
cost and the buys' cost, two halves held after; the live halves are not ranked; Winning / Losing vs SPY."""
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
import pandas as pd

import backtest_engine as be
import forward_combos as fc
import forward_test as ft
import sector_mapping as sm

FAIL = []
def check(ok, what, detail=""):
    print(("PASS " if ok else "FAIL ") + what + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(what)

if not (os.path.exists(ft.SIGNAL_CSV) and os.path.exists(ft.STRATEGIES_CSV) and os.path.exists(ft.FWD_BARS)):
    print("SKIP no saved data"); sys.exit(0)
sig = pd.read_csv(ft.SIGNAL_CSV, usecols=ft.SIG_COLS, parse_dates=["Date"])
bars = ft.forward_bars(fetch=False)
inp = ft.strategy_inputs(sig, ft._read(ft.FACTOR_CSV), None, ft._read(ft.BENCH_CSV), bars)
x = fc.Ctx(inp)
check(((x.leg("sr63")[0] - inp["scores"]["sector_rot"]).abs().fillna(0) < 1e-12).all().all()
      and (x.leg("sr63")[0].isna() == inp["scores"]["sector_rot"].isna()).all().all(),
      "sr63 score = forward_test's Sector rotation score")

start = "2026-07-01"
combos = [(ft.LIVE_LEGS, "full weekly SR"), ((ft.LIVE_LEGS[0], ("sr63", "swap10", "day", 0.5)), "daily swap"),
          ((ft.LIVE_LEGS[0], ("calm", "full", "week", 0.5)), "calmer half")]
never = set(sm.do_not_buy)
for legs, label in combos:
    a = fc.simulate(inp, legs, start)
    fc._CACHE.clear()
    b = fc.simulate(inp, legs, start)
    fc._CACHE.clear()
    short = fc.simulate(inp, legs, start, through=a["days"][len(a["days"]) // 2])
    fc._CACHE.clear()
    check(a["legs"] == b["legs"], f"{label}: two replays equal (deterministic)")
    n = len(short["days"])
    check(a["legs"][0][:n] == short["legs"][0][:n] and a["legs"][1][:n] == short["legs"][1][:n],
          f"{label}: a shorter replay is a prefix of the longer one (saved days never change)")
    lv, sr = a["legs"]
    overlap = [d for d, l0, l1 in zip(a["days"], lv, sr) if set(l0[2]) & set(l1[2])]
    check(not overlap, f"{label}: no stock in both halves at any close", overlap[:3])
    worst = max(max((q * p / v for q, p in l[2].values()), default=0) for leg in a["legs"] for (v, c, l) in [(e[0], e[1], e) for e in leg])
    check(worst <= ft.MAX_WEIGHT + 0.012, f"{label}: no stock above 20% of its half (+ drift until a rebalance)", worst)
    check(all(len(e[2]) <= 10 for leg in a["legs"] for e in leg), f"{label}: at most 10 names per half")
    check(min(e[1] for leg in a["legs"] for e in leg) >= -1e-9, f"{label}: cash never below zero")
    check(not any(set(e[2]) & never for leg in a["legs"] for e in leg), f"{label}: do_not_buy never held")
    rows, hold = fc.line_rows("t", a)
    check(abs(rows[0]["Value"] - 1.0) < 1e-12 and abs(sum(h["Shares"] * h["Price"] for h in hold if h["Date"] == rows[0]["Date"])
                                                     + rows[0]["Cash"] - 1.0) < 1e-9, f"{label}: 1.0 at the start close after its buys")

# keep (A) / B-final differ from plain monthly only by their rules
m = fc.simulate(inp, (("lowvol", "full", "month", 1.0),), start)
bf = fc.simulate(inp, (("lowvol", "bfinal", "month", 1.0),), start)
kp = fc.simulate(inp, (("lowvol", "keep", "month", 1.0),), start)
check(m["legs"][0][0] == bf["legs"][0][0] == kp["legs"][0][0], "low vol variants: same start-day buys")
check(m["legs"][0][-1] != bf["legs"][0][-1], "B-final trades on Fridays between month ends (differs from plain monthly)")

# update_strategies on temp copies: new lines appended, the old live name migrated, saved days kept
tmp = tempfile.mkdtemp()
vp, hp = os.path.join(tmp, "v.csv"), os.path.join(tmp, "h.csv")
v0 = pd.read_csv(ft.STRATEGIES_CSV)
h0 = pd.read_csv(ft.HOLDINGS_CSV)
old_live = "Live rules (C6)"
v0, h0 = (d.assign(Strategy=d["Strategy"].replace({ft.LIVE: old_live})) for d in (v0, h0))   # as saved before the rename
v0 = v0[~((v0["Strategy"] == old_live) & (v0["Date"] >= "2026-10-13"))]
h0 = h0[~((h0["Strategy"] == old_live) & (h0["Date"] >= "2026-10-13"))]
names = [c["name"] for c in ft.STRATEGIES if c.get("combo")]
v0, h0 = v0[~v0["Strategy"].isin(names)], h0[~h0["Strategy"].isin(names)]
v0.to_csv(vp, index=False); h0.to_csv(hp, index=False)
saved_sleeves, saved_switch = be.WINNER.get("sleeves"), ft.LIVE_SWITCH
try:
    be.WINNER["sleeves"] = {"lowvol": 0.5, "sector": 0.5}
    ft.LIVE_SWITCH = "2026-10-13"                     # a switch the saved data covers
    fc._CACHE.clear()
    ft.update_strategies(sig=sig, bars=bars, path=vp, hold_path=hp)
finally:
    if saved_sleeves is None:
        be.WINNER.pop("sleeves", None)
    else:
        be.WINNER["sleeves"] = saved_sleeves
    ft.LIVE_SWITCH = saved_switch
v1, h1 = pd.read_csv(vp), pd.read_csv(hp)
check(old_live not in set(v1["Strategy"]) and old_live not in set(h1["Strategy"]), "old live name migrated in both files")
L = v1[v1["Strategy"] == ft.LIVE].set_index("Date")
L0 = v0[v0["Strategy"] == old_live].set_index("Date")
check(np.allclose(L.loc[L0.index, "Value"], L0["Value"]), "the live line's saved days before the switch are unchanged")
ho = h0[(h0["Strategy"] == old_live) & (h0["Date"] == "2026-10-12")]
px = inp["close"].loc["2026-10-13"]
inv = float((ho["Shares"] * ho["Symbol"].map(px).fillna(ho["Price"])).sum())   # a stock gone from the list: its saved price
base = float(L0.loc["2026-10-12", "Cash"]) + inv * (1 - be.COST)
sim = fc.simulate(inp, ft.LIVE_LEGS, "2026-10-13")
check(abs(L.loc["2026-10-13", "Value"] - base * sum(l[0][0] for l in sim["legs"])) < 1e-9,
      "switch day = old holdings marked to the close, sold (0.1%), then the halves' buys", (L.loc["2026-10-13", "Value"], base))
hl = h1[(h1["Strategy"] == ft.LIVE) & (h1["Date"] == "2026-10-13")]
check(len(hl) > 10 and abs(float((hl["Shares"] * hl["Price"]).sum()) + float(L.loc["2026-10-13", "Cash"]) - L.loc["2026-10-13", "Value"]) < 1e-9,
      "live holdings after the switch = both halves (shares x price + cash = value)", len(hl))
new = v1[v1["Strategy"].isin(names)]
starts = new.groupby("Strategy")["Date"].min()
halves = list(ft.LIVE_HALVES)
check(set(starts.drop(halves, errors="ignore")) == {"2026-10-02"}, "the new lines (not the live halves) start on Oct 2", starts.to_dict())
check(all(abs(v - 1.0) < 1e-12 for v in new[new["Date"] == "2026-10-02"]["Value"]), "each new line is 1.0 at the Oct 2 close")
lb = ft.leaderboard(values=v1.assign(Date=pd.to_datetime(v1["Date"])))
check(lb.set_index("Strategy").loc[halves, "Rank"].isna().all(), "the live halves are not ranked")
again = ft.update_strategies(sig=sig, bars=bars, path=vp, hold_path=hp)
check(again == 0, "a second update adds nothing (saved days never redone)", again)
shutil.rmtree(tmp, ignore_errors=True)

# live vs benchmark (synthetic)
d = pd.bdate_range("2026-10-16", periods=8)
vals = pd.concat([pd.DataFrame({"Date": d, "Strategy": n, "Value": np.linspace(1, 1 + g, 8), "Cash": 0.0})
                  for n, g in ((ft.LIVE, 0.05), (halves[0], 0.02), (halves[1], 0.08))])
bench = pd.DataFrame({"Date": d, "SPY": np.linspace(100, 103, 8), "QQQ": np.linspace(100, 106, 8)})
r = ft.live_vs_benchmark(vals, bench)
check(r["status"] == "Winning" and abs(r["gap"] - 2.0) < 1e-9 and r["days"] == 7, "Winning vs SPY by 2.00 pts", (r["status"], r["gap"]))
check("Winning vs SPY by 2.00 pts" in ft.benchmark_verdict(r) and "Thu Dec 24, 2026" in ft.benchmark_verdict(r), "verdict line + decision date")
check([c for c in r["table"].columns if c.startswith("Week ")] == ["Week 1 %", "Week 2 (so far) %"]
      and list(r["table"]["Line"]) == [ft.LIVE, *halves, "SPY", "QQQ"], "rows: live, halves, SPY, QQQ; weekly columns", list(r["table"].columns))
r2 = ft.live_vs_benchmark(vals.assign(Value=lambda v: np.where(v["Strategy"] == ft.LIVE, 1 - (v["Value"] - 1), v["Value"])), bench)
check(r2["status"] == "Losing" and abs(r2["gap"] + 8.0) < 1e-9, "Losing vs SPY by 8.00 pts", (r2["status"], r2["gap"]))
r3 = ft.live_vs_benchmark(vals[vals["Date"] < "2026-10-16"], bench)
check(r3["status"] is None and "starts at the Fri Oct 16 close" in ft.benchmark_verdict(r3), "before Oct 16: not started")
src = open(os.path.join(ROOT, "forward_combos.py")).read()
check(not any(k in src for k in ("submit", "alpaca", "requests", "to_csv", "WINNER[")), "forward_combos: no orders, no writes")

print("FORWARD COMBOS OK" if not FAIL else f"{len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
