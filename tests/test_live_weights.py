"""Live weights target 99% invested and are rounded DOWN to 2-decimal percents, so rounding never pushes the total over 100%.
  - rule weights summing to 1 whose 4-decimal ROUNDING sums above 1 (the 2026-09-30 case: 1.0001) -> live weights sum <= 99%
  - floor, not nearest: 0.0999 x 0.99 = 0.098901 -> 0.0989 (9.89%); exact values stay exact (0.10 -> 0.0990)
  - regime-off halving still applies on top (<= 49.5%); fewer picks -> less invested
  - build_orders accepts them, still rejects a total over 100%; the 1% buying-power cushion does not trim a 99% plan
  - the saved Reports/strategy_picks.csv weights obey the same rule
Run: python tests/run_tests.py  (or PYTHONPATH=. python tests/test_live_weights.py)"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
import pandas as pd

import backtest_engine as be
import paper_trade as pt

FAIL = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info and not ok else ""))
    if not ok:
        FAIL.append(name)


def on_grid(w):
    return bool(np.all(np.abs(np.asarray(w) * 10_000 - np.round(np.asarray(w) * 10_000)) < 1e-6))


check("target is 99%", be.LIVE_INVESTED == 0.99)
rng = np.random.default_rng(7)
over, worst = 0, 0.0
for _ in range(5000):
    inv = 1 / rng.uniform(0.01, 0.06, 10)          # inverse-volatility weights of 10 picks, summing to exactly 1
    w = inv / inv.sum()
    over += np.round(w, 4).sum() > 1 + 1e-9
    live = be.live_weights(w)
    worst = max(worst, live.sum())
    if not (live.sum() <= 0.99 + 1e-12 and on_grid(live) and (live <= w * 0.99 + 1e-12).all()
            and (live > w * 0.99 - 0.0001 - 1e-12).all()):
        check("random weights: live <= 99%, floored to 0.0001", False, list(w))
        break
else:
    check("5000 random 10-pick portfolios: every live total <= 99%, each weight floored to 0.0001", True)
check("the old 4-decimal rounding went over 100% in some of them", over > 0, over)
print(f"     ({over} of 5000 rounded above 100% the old way; highest live total {worst:.4f})")

check("floor, not nearest: 0.0999 -> 0.0989", be.live_weights(0.0999) == 0.0989, be.live_weights(0.0999))
check("exact stays exact: 0.10 -> 0.0990", be.live_weights(0.10) == 0.099, be.live_weights(0.10))
check("no float drop for 0.0001 .. 0.2000", all(be.live_weights(k / 10_000) == np.floor(round(k * 0.99, 6)) / 10_000
                                                for k in range(1, 2001)))
check("zero and NaN pass through", be.live_weights(0.0) == 0.0 and np.isnan(be.live_weights(np.nan)))
half = be.live_weights(pd.Series(np.full(10, 0.1)) * 0.5)
check("regime-off halving on top: 10 x 5% -> 10 x 4.95% = 49.5%", np.isclose(half.sum(), 0.495) and (half == 0.0495).all())
check("7 of 10 picks -> at most 69.3%", be.live_weights(pd.Series(np.full(7, 0.1))).sum() <= 0.693 + 1e-12)
frame = be.live_weights(pd.DataFrame({"A": [0.12345, 0.0], "B": [0.87655, np.nan]}))
check("works on a DataFrame", isinstance(frame, pd.DataFrame) and frame.iloc[0].sum() <= 0.99 and frame.iloc[0, 0] == 0.1222)

# order sizing: the trade step takes the 99% weights as they are (no second scale-down)
syms = [f"S{i}" for i in range(10)]
w = 1 / np.linspace(0.015, 0.05, 10)
targets = pd.DataFrame({"Symbol": syms, "Weight": be.live_weights(w / w.sum()), "Price": 10.0})
orders = pt.build_orders(targets, 100_000.0, positions={}, fractional=True, statuses={s: "add" for s in syms})
buy = (orders["Shares"] * orders["Price"])[orders["Side"] == "BUY"].sum()
check("build_orders accepts the live weights; buys ~99% of the account",
      98_900 <= buy <= 99_000 + 1e-6, f"{buy:.2f}")
guarded = pt.apply_buying_power_guard(orders, 100_000.0, fractional=True)
gbuy = (guarded["Shares"] * guarded["Price"])[guarded["Side"] == "BUY"].sum()
check("the 1% cash cushion does not trim a 99% plan (no ~98%)", abs(gbuy - buy) < 1e-6, f"{gbuy:.2f} vs {buy:.2f}")
bad = targets.assign(Weight=np.round(w / w.sum(), 4))
bad.loc[0, "Weight"] += max(0.0, 1.0001 - bad["Weight"].sum())
try:
    pt.build_orders(bad, 100_000.0, positions={}, fractional=True, statuses={s: "add" for s in syms})
    check("build_orders still rejects weights summing to 1.0001", False)
except ValueError as e:
    check("build_orders still rejects weights summing to 1.0001", ">100%" in str(e), str(e))

picks_csv = os.path.join(ROOT, "Reports", "strategy_picks.csv")
if os.path.exists(picks_csv):
    p = pd.read_csv(picks_csv)
    for col in ("Strategy_Weight", "Provisional_Weight"):
        s = p[col].fillna(0)
        check(f"saved strategy_picks {col}: total {s.sum():.4f} <= 99%, 2-decimal percents",
              s.sum() <= 0.99 + 1e-9 and on_grid(s), f"{s.sum():.6f}")
print("LIVE WEIGHTS OK" if not FAIL else f"LIVE WEIGHTS FAILURES: {FAIL}")
sys.exit(1 if FAIL else 0)
