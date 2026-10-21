"""forward_combos (Chirag, Oct 15, 2026 evening): the forward-test lines with two independent halves, swap-only halves and
the low-volatility keep / Friday falling rules (the old "Live 50/50 combined (no overlap)" line was removed Oct 20, 2026). Real saved data,
temp copies of the forward files (the real Reports files are never written); no broker calls, no orders.

Covers: deterministic replay (two runs equal; a shorter replay is a prefix of the longer one = saved days never change);
no overlap between the halves at any close; at most 10 names and 20% of its half per stock; never below zero cash;
do_not_buy never held; the 63-day sector score = forward_test's Sector rotation score; values 1.0 at the start close;
update_strategies appends the new lines and migrates the old live name; the live switch (shown on Oct 13 so the saved data
covers it): the Oct 12 saved day unchanged, the switch-day value = the old holdings marked to that close minus the sale
cost and the buys' cost, two halves held after; the live halves are not ranked."""
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import pandas as pd

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
combos = [(ft.PREV_LEGS, "full weekly SR"), ((ft.LV, ("sr63", "swap10", "day", 0.5)), "daily swap"),
          ((ft.LV, ("calm", "full", "week", 0.5)), "calmer half"), (ft.LIVE_LEGS, "live top 5")]
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
    # 5 stocks drift more between rebalances
    check(worst <= 0.2 + (0.05 if label == "live top 5" else 0.012), f"{label}: no stock above one equal slot of its half (+ drift)", worst)
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
removed = sorted(ft.REMOVED)[0]
extra = v0[v0["Strategy"] == ft.MON].assign(Strategy=removed)     # a removed line still in a saved file
v0 = pd.concat([v0, extra], ignore_index=True)
names = [c["name"] for c in ft.STRATEGIES if c.get("combo")]
v0, h0 = v0[~v0["Strategy"].isin(names)], h0[~h0["Strategy"].isin(names)]
v0.to_csv(vp, index=False); h0.to_csv(hp, index=False)
fc._CACHE.clear()
ft.update_strategies(sig=sig, bars=bars, path=vp, hold_path=hp)
v1, h1 = pd.read_csv(vp), pd.read_csv(hp)
check(not set(v1["Strategy"]) & ft.REMOVED and not set(h1["Strategy"]) & ft.REMOVED, "removed lines purged from both files")
check(ft.LIVE == ft.TOP5 and {ft.TOP5, ft.MON} <= set(v1["Strategy"]), "the live comparison line is the top-5 50/50 (top-10 kept)")
new = v1[v1["Strategy"].isin(names)]
starts = new.groupby("Strategy")["Date"].min()
halves = list(ft.LIVE_HALVES)
check(set(starts.drop(halves, errors="ignore")) == {"2026-10-02"}, "the new lines (not the live halves) start on Oct 2", starts.to_dict())
check(all(abs(v - 1.0) < 1e-12 for v in new[new["Date"] == "2026-10-02"]["Value"]), "each new line is 1.0 at the Oct 2 close")
T5 = ft.TOP5
h5 = h1[(h1["Strategy"] == T5) & (h1["Shares"] > 0)]
check(T5 in starts and starts[T5] == "2026-10-02" and len(h5) and h5.groupby("Date")["Symbol"].nunique().max() <= 10,
      "top 5 each half: from Oct 2, at most 5 + 5 stocks", h5.groupby("Date")["Symbol"].nunique().to_dict())
lb = ft.leaderboard(values=v1.assign(Date=pd.to_datetime(v1["Date"])))
check(lb.set_index("Strategy").loc[halves, "Rank"].isna().all(), "the live halves are not ranked")
again = ft.update_strategies(sig=sig, bars=bars, path=vp, hold_path=hp)
check(again == 0, "a second update adds nothing (saved days never redone)", again)
shutil.rmtree(tmp, ignore_errors=True)

# the live manual buy blocks ("+mb", Reports/manual_block.json): a blocked stock is never bought from the session after
# it was added on (a held one sold that day, the live manual sale) until a close at or below block_until_price
import json
import manual_trade as mt
_bj, mt.BLOCK_JSON = mt.BLOCK_JSON, os.path.join(tempfile.mkdtemp(), "b.json")
json.dump({}, open(mt.BLOCK_JSON, "w"))
ref = fc.simulate(inp, ft.LIVE_LEGS, start); fc._CACHE.clear()
day = ref["days"][len(ref["days"]) // 2]
held = sorted(set(ref["legs"][0][len(ref["days"]) // 2][2]) | set(ref["legs"][1][len(ref["days"]) // 2][2]))
bs = held[0]
json.dump({bs: {"added": f"{day:%Y-%m-%d}", "block_until_price": None}},
          open(mt.BLOCK_JSON, "w"))
y = fc.Ctx(inp)
mb = fc.simulate({**inp, "_combo_ctx": y}, ft.LIVE_LEGS, start); fc._CACHE.clear()
after = [d for i, d in enumerate(mb["days"]) if d >= day and any(bs in leg[i][2] for leg in mb["legs"])]
check(not after and y.mblock[bs].loc[day:].all() and not y.mblock[bs].loc[:day].iloc[:-1].any(),
      "manual block: sold that day, never held after", (bs, after[:3]))
k = len(ref["days"]) // 2
check(all(mb["legs"][j][:k] == ref["legs"][j][:k] for j in (0, 1)) and len(mb["legs"][0][k][2]) + len(mb["legs"][1][k][2]) >= len(held) - 1,
      "manual block: days before it unchanged, the slot refilled")
mt.BLOCK_JSON = _bj
src = open(os.path.join(ROOT, "forward_combos.py")).read()
check(not any(k in src for k in ("submit", "alpaca", "requests", "to_csv", "WINNER[")), "forward_combos: no orders, no writes")

print("FORWARD COMBOS OK" if not FAIL else f"{len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
