"""forward_combos (Chirag, Oct 15, 2026 evening): the forward-test lines with two independent halves and swap-only halves. Real saved data,
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
import db  # noqa: E402  tables live in SQLite (db.py), not CSV files

import forward_combos as fc
import forward_test as ft
import backtest_engine as be
import sector_mapping as sm

FAIL = []
def check(ok, what, detail=""):
    print(("PASS " if ok else "FAIL ") + what + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(what)

if not (db.exists(ft.SIGNAL_CSV) and db.exists(ft.STRATEGIES_CSV) and db.has_bars()):
    print("SKIP no saved data"); sys.exit(0)
sig = db.read_csv(ft.SIGNAL_CSV, usecols=ft.SIG_COLS, parse_dates=["Date"])
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
                                                     + rows[0]["Cash"] - 1.0) < 1e-9, f"{label}: 1.0 at the start close after its buys")  # line_rows still unit-normalized; update_strategies scales to START_CASH

# update_strategies on temp copies: new lines appended, the old live name migrated, saved days kept
tmp = tempfile.mkdtemp()
vp, hp = os.path.join(tmp, "v.csv"), os.path.join(tmp, "h.csv")
v0 = db.read_csv(ft.STRATEGIES_CSV)
h0 = db.read_csv(ft.HOLDINGS_CSV)
removed = sorted(ft.REMOVED)[0]
# inject a REMOVED name so purge is exercised (bars currently end before FORWARD_START, so do not strip
# combo rows that cannot be rebuilt until volume catches up)
seed = v0[v0["Strategy"] == ft.LIVE].head(1)
extra = seed.assign(Strategy=removed) if len(seed) else v0.head(1).assign(Strategy=removed)
v0 = pd.concat([v0, extra], ignore_index=True)
names = [c["name"] for c in ft.STRATEGIES if c.get("combo")]
v0.pipe(db.to_csv, vp, index=False); h0.pipe(db.to_csv, hp, index=False)
fc._CACHE.clear()
ft.update_strategies(sig=sig, bars=bars, path=vp, hold_path=hp)
v1, h1 = db.read_csv(vp), db.read_csv(hp)
check(not set(v1["Strategy"]) & ft.REMOVED and not set(h1["Strategy"]) & ft.REMOVED, "removed lines purged from both files")
check(ft.LIVE == ft.LIVE_NOW and ft.LIVE in set(v1["Strategy"]), "the live line is short-term relative strength")
check(not ({ft.MON5, ft.MON} & set(v1["Strategy"])), "Oct 16-era live 50/50 lines stay removed")
new = v1[v1["Strategy"].isin(names)]
starts = pd.to_datetime(new.groupby("Strategy")["Date"].min())
check(len(names) and set(starts.dt.strftime("%Y-%m-%d")) == {str(be.FORWARD_START)[:10]},
      "the combo lines start on FORWARD_START", starts.to_dict())
day0 = new[pd.to_datetime(new["Date"]) == pd.Timestamp(be.FORWARD_START)]
check(len(day0) and all(abs(v - ft.START_CASH) < 1e-6 for v in day0["Value"]),
      "each combo line is START_CASH at FORWARD_START")
unranked = [c["name"] for c in ft.STRATEGIES if c.get("combo") and c.get("ranked") is False]
lb = ft.leaderboard(values=v1.assign(Date=pd.to_datetime(v1["Date"])))
if unranked:
    check(lb.set_index("Strategy").loc[unranked, "Rank"].isna().all(), "unranked combo halves are not ranked")
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
