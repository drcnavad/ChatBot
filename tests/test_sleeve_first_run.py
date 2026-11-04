"""First sleeve run (no / old ledger) must size against the equity once: before Nov 4, 2026 the start cash was the whole
equity AND the held positions were folded back in, so targets came to ~2x equity. Also: the cap guard in sleeves.plan."""
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402
import sleeves  # noqa: E402

FAIL = []
def check(c, name, detail=""):
    print(f"{'PASS' if c else 'FAIL'} {name}" + ("" if c else f" {detail}"))
    c or FAIL.append(name)

H = sleeves.HALVES[0]
state, unowned, info = sleeves.reconcile(None, {"AAA": 10, "BBB": 5}, {"AAA": 100.0, "BBB": 200.0}, 10_000.0, {H: 1.0})
check(info["init"] and abs(state[H]["cash"] - 8_000.0) < 1e-9, "first run: cash = equity - holdings", state)

inp = sleeves.inputs()
day = inp["close"].index[-1]
picks = sleeves.ranking(H, day, inp).index[:3].tolist()
px = {s: float(inp["close"].at[day, s]) for s in picks}
equity = 100_000.0
pos = {s: round(20_000.0 / px[s], 4) for s in picks}            # $60k held, $40k cash
orders, info = sleeves.plan(day, pos, equity, px, ledger=None, inp=inp)
tv = sum(q * (px.get(s) or float(inp["close"].at[day, s])) for s, q in info["half_shares"][H].items())
check(tv <= equity * 0.99 + 1, "first run: sleeve targets <= 99% of equity", tv)
buys = orders.loc[orders["Side"] == "BUY", "Est_Value"].sum() - orders.loc[orders["Side"] == "SELL", "Est_Value"].sum()
check(buys <= equity - 60_000.0 + 1, "first run: net buys <= the cash", buys)

fat = lambda led, positions, prices, equity, shares=None, **k: ({h: {"shares": {}, "cash": 5 * equity} for h in sleeves.HALVES},
                                                                 {}, {"init": True, "residual": 0.0, "dropped": [], "tags": {}})
with mock.patch.object(sleeves, "reconcile", fat):
    try:
        sleeves.plan(day, {}, equity, px, ledger=None, inp=inp)
        check(False, "cap guard raises when targets exceed the equity")
    except ValueError as e:
        check("exceed" in str(e), "cap guard raises when targets exceed the equity", e)

print(f"\n{len(FAIL)} failures" if FAIL else "\nALL PASSED")
sys.exit(1 if FAIL else 0)
