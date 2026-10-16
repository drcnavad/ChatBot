"""The live rules from Fri Oct 16, 2026 (Chirag): Low volatility in uptrend (monthly) = forward_test.py's "Low volatility in
uptrend" (score low_vol, monthly, equal weight, max 4 per sector, 20% cap, no earnings rule, no sleeve) + do_not_buy +
the optional Friday check (WINNER['friday_falling_check'], OFF live; tested with the flag on): sell a holding below its entry price, at/below its 200-day
average or out of the top 10, same dollars into the best pick not held; no Mon-Thu trading; the earnings-day stop off.
Mocked: no broker calls, temporary files only."""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
os.chdir(ROOT)
import pandas as pd  # noqa: E402

import backtest_engine as be  # noqa: E402
import forward_test as ft  # noqa: E402
import paper_trade as pt  # noqa: E402
import sector_mapping as sm  # noqa: E402

FAIL = []

def check(ok, what, got=""):
    print(("PASS " if ok else "FAIL ") + what + ("" if ok else f"  ({got})"))
    if not ok:
        FAIL.append(what)

W = be.WINNER
cfg = next(c for c in ft.STRATEGIES if c["name"] == "Low volatility in uptrend")
sel = {"max_weight": ft.MAX_WEIGHT, **ft.PIN, **cfg["select"]}
check(cfg["score"] == W["score"] == "low_vol" and cfg["calendar"] == W["calendar"] == "monthly", "score / calendar = forward strategy")
for k in ("vol_sizing", "sector_cap", "max_weight", "max_pick_rank", "cap_soft"):
    check(sel[k] == W[k], f"selection {k} = forward strategy", (sel[k], W[k]))
check(cfg["earnings"] is None and not W["earnings_block_days"], "no earnings rule (forward strategy: earnings None)")
check(not W["midweek_swap"] and not W["midweek_exit_below"], "no Mon-Thu checks")
check(not W["sleeve"] and "sleeve" not in be.winner_rank_args(pd.Series(dtype=bool)) and sm.do_not_sell == [], "no sleeve, do_not_sell empty")
check(not W["keep_profit_at_rebalance"] and not W["earnings_day_stop"] and not W["friday_falling_check"], "keep rule off, stop off, Friday check off (plain monthly)")
check(pt.MAX_ORDER_PCT == W["max_weight"], "order backstop = 20% cap")

# calendar: Oct 16 switch, then the last trading day of each month; nothing in between
days = pd.date_range("2026-10-12", "2027-01-31", freq="D")
W["friday_falling_check"] = False
dec = {d.date().isoformat(): be.decision_kind(d) for d in days if be.decision_kind(d)}
W["friday_falling_check"] = True   # the flag's own calendar
check(dec == {"2026-10-16": "full rebalance", "2026-10-30": "full rebalance", "2026-11-30": "full rebalance",
              "2026-12-31": "full rebalance", "2027-01-29": "full rebalance"}, "decision days Oct 16, Oct 30, Nov 30, Dec 31, Jan 29", dec)
fridays = [d for d in days if d.dayofweek == 4 and d.date().isoformat() not in dec and be.is_session(d)]
check(all(be.decision_kind(d) == "Friday check" for d in fridays) and be.decision_kind(pd.Timestamp("2026-10-30")) == "full rebalance"
      and be.decision_kind(pd.Timestamp("2026-10-20")) is None, "flag on: every other Friday is a 'Friday check', Mon-Thu still none")
W["friday_falling_check"] = False
check(be.decision_kind(pd.Timestamp("2026-10-23")) is None, "flag off: Fri Oct 23 is no decision")
W["friday_falling_check"] = True   # flag on for the friday_falling_orders checks below

# keep rule + Friday falling check on a small signal file
tmp = tempfile.mkdtemp()
sig = os.path.join(tmp, "signal_analysis.csv")
rows = [("2026-10-16", s, c, m, r, sc) for s, c, m, r, sc in [
    ("WIN", 120.0, 100.0, 15, 0.1),     # in profit, above the 200-day -> kept
    ("LOSS", 90.0, 80.0, 16, 0.1),      # below entry -> sold
    ("DOWN", 120.0, 130.0, None, -1.0), # in profit but below its 200-day -> sold
    ("NEW1", 50.0, 40.0, 1, 0.5), ("NEW2", 60.0, 50.0, 2, 0.4), ("OUT", 70.0, 50.0, 30, 0.05), ("AMD", 70.0, 50.0, 3, 0.45),
    ("PICK", 110.0, 90.0, 4, 0.3)]]
pw = {"WIN": 0.099, "LOSS": 0.099, "NEW1": 0.099, "NEW2": 0.099, "AMD": 0.099, "PICK": 0.099}
pd.DataFrame(rows, columns=["Date", "Symbol", "Close", "ma_200", "Strategy_Rank", "Strategy_Score"]).assign(
    Provisional_Weight=lambda d: d.Symbol.map(pw).fillna(0.0), Regime_On=1, Midweek_Check=0).to_csv(sig, index=False)
pos = {"WIN": 10, "LOSS": 10, "DOWN": 10, "NOPRICE": 5}
entry = {"WIN": 100.0, "LOSS": 100.0, "DOWN": 100.0}
px = {"WIN": 120.0, "LOSS": 90.0, "DOWN": 120.0, "NOPRICE": 10.0}
kept, sold = pt.keep_in_profit(pos, {"NEW1"}, px, entry, "2026-10-16", sig)
check(set(kept) == {"WIN"} and set(sold) == {"LOSS", "DOWN", "NOPRICE"}, "keep rule: WIN kept; LOSS (below entry), DOWN (200-day), NOPRICE (no entry) sold", (kept, sold))
entry2, px2 = {**entry, "OUT": 60.0}, {**px, "OUT": 70.0}
o, info = pt.friday_falling_orders(pd.DataFrame(columns=pt.ORDER_COLUMNS), {"WIN": 10, "LOSS": 10, "DOWN": 10, "OUT": 10},
                                   10_000.0, "2026-10-16", entry2, px2, sig, no_buy={"AMD": "do_not_buy"})
sells = set(o.loc[o.Side == "SELL", "Symbol"])
buys = dict(zip(o.loc[o.Side == "BUY", "Symbol"], o.loc[o.Side == "BUY", "Est_Value"]))
check(sells == {"LOSS", "DOWN", "OUT"} and set(buys) == {"NEW1", "NEW2", "PICK"} and set(o.Symbol) - sells - set(buys) <= {"WIN"},
      "Friday check: LOSS (below entry), DOWN (200-day), OUT (not a pick) sold; refilled by NEW1, NEW2, PICK (AMD do_not_buy "
      "skipped); WIN untouched", (sells, buys, info["why"]))
pairs = {a: (b, v) for a, _ra, b, _rb, v in info["replaced"]}
sold_usd = dict(zip(o.loc[o.Side == "SELL", "Symbol"], o.loc[o.Side == "SELL", "Est_Value"]))
check(all(v <= sold_usd[a] + 1e-6 and sold_usd[a] - v < 120 and v <= 0.2 * 0.99 * 10_000 + 1e-6 for a, (b, v) in pairs.items()),
      "each refill = its sale's dollars (whole shares), within the 20% cap", (pairs, sold_usd))

print("LOW VOL MONTHLY OK" if not FAIL else f"LOW VOL MONTHLY FAILURES ({len(FAIL)}): {FAIL}")
sys.exit(1 if FAIL else 0)
