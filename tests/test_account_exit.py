"""Mon/Wed sell rule on EVERY actual account position (Chirag, t188u, 2026-10-07): after the strategy's own mid-week orders,
any Alpaca position ranked worse than 20 - or with no rank (score 0 or below, or not in the stock list) - is sold in full,
worst first, and replaced 1-for-1 (same dollars, 19.8% cap) by the best top-10 stock the account does not hold (earnings /
earnings-day-stop blocks apply); no refill -> the cash goes to the spare-cash step. One SELL row per stock. Friday and
non-check days are unchanged. Hand-worked cases with fake CSVs; no network, no orders.
Run: python tests/run_tests.py  (or python tests/test_account_exit.py)"""
import os
import sys
import tempfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import backtest_engine as be
import paper_trade as pt

FAIL = []

def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info and not ok else ""))
    if not ok:
        FAIL.append(name)

def ranking(rows):
    """rows: (symbol, rank, close)."""
    return pd.DataFrame([{"Symbol": s, "Rank": r, "Score": 80 - r, "Provisional_Weight": 0.1 if r <= 10 else 0.0,
                          "Close": c, "Regime_On": 1, "Midweek_Check": 1} for s, r, c in rows])

def side(o, sym):
    return o.loc[o["Symbol"] == sym, "Side"].tolist()

EQ = 100_000.0
R = ranking([(f"S{i:02d}", i, 100.0) for i in range(1, 26)])          # S01 = rank 1 ... S25 = rank 25, all $100
PX = {f"S{i:02d}": 100.0 for i in range(1, 26)}
check("live settings: sell worse than rank 20, refill from the top 10",
      be.WINNER["midweek_exit_below"] == 20 and be.WINNER["midweek_exit_to_top"] == 10)

# 1) an extra position ranked 24, every top-10 stock already held: sold in full, no refill (cash -> spare-cash step)
pos = {**{f"S{i:02d}": 100 for i in range(1, 11)}, "S24": 12}
o, info = pt.build_account_exit_orders(pt.build_hold_orders(pos, PX), pos, EQ, R, prices=PX, fractional=True)
row = o[o["Symbol"] == "S24"]
check("rank 24 (like BE): one SELL of all 12 shares, $1,200; nothing bought (top 10 all held)",
      side(o, "S24") == ["SELL"] and row["Shares"].iloc[0] == 12 and row["Est_Value"].iloc[0] == 1_200
      and (o["Side"] == "BUY").sum() == 0 and info["sold"] == [("S24", 24, 12, 1_200.0)] and info["replaced"] == [],
      (o.values.tolist(), info))
check("the other positions stay HOLD (ranks 1-10)", all(side(o, f"S{i:02d}") == ["HOLD"] for i in range(1, 11)))

# 2) rank 20 is kept, rank 21 sold; the refill = the best top-10 stock not held, same dollars
pos = {**{f"S{i:02d}": 100 for i in (1, 2, 4, 5, 6, 7, 8, 9)}, "S20": 50, "S21": 80}   # S03, S10 not held
o, info = pt.build_account_exit_orders(pt.build_hold_orders(pos, PX), pos, EQ, R, prices=PX, fractional=True)
b = o[o["Side"] == "BUY"]
check("rank 20 kept; rank 21 sold and replaced by S03 (rank 3, the best not held) for the same $8,000",
      side(o, "S20") == ["HOLD"] and side(o, "S21") == ["SELL"] and list(b["Symbol"]) == ["S03"]
      and b["Est_Value"].iloc[0] == 8_000 and info["replaced"] == [("S21", 21, "S03", 3, 8_000.0)], (o.values.tolist(), info))

# 3) worst first: no rank (not in the list / score 0 or below) before the worst rank; best refill to the worst
pos = {**{f"S{i:02d}": 100 for i in (2, 4, 5, 6, 7, 8, 9, 10)}, "S22": 30, "S25": 40, "XYZ": 50, "NEG": 60}
listed = set(R["Symbol"]) | {"NEG"}                                    # NEG: in the stock list, score below 0 (no rank)
px = {**PX, "XYZ": 40.0, "NEG": 50.0}
o, info = pt.build_account_exit_orders(pt.build_hold_orders(pos, px), pos, EQ, R, listed=listed, prices=px, fractional=True)
check("sold worst first: NEG and XYZ (no rank), then S25, then S22", [s for s, *_ in info["sold"]] == ["NEG", "XYZ", "S25", "S22"],
      info["sold"])
check("refills in that order: NEG -> S01 (rank 1, $3,000), XYZ -> S03 (rank 3, $2,000); S25 / S22 sold to cash (no top-10 left)",
      info["replaced"] == [("NEG", None, "S01", 1, 3_000.0), ("XYZ", None, "S03", 3, 2_000.0)]
      and side(o, "S25") == ["SELL"] and side(o, "S22") == ["SELL"], info["replaced"])
check("reported: XYZ not in the stock list, NEG listed but not ranked", info["unlisted"] == ["XYZ"] and info["unranked"] == ["NEG"],
      (info["unlisted"], info["unranked"]))
check("orders: every SELL before every BUY, one row per stock", list(o["Side"]).index("BUY") > max(
    i for i, s_ in enumerate(o["Side"]) if s_ == "SELL") and o["Symbol"].is_unique, o[["Symbol", "Side"]].values.tolist())

# 4) blocks: an earnings / stop-blocked top-10 stock is not bought; the next best one is
pos = {**{f"S{i:02d}": 100 for i in (2, 4, 5, 6, 7, 8, 9)}, "S23": 50}           # S01, S03, S10 not held
o, info = pt.build_account_exit_orders(pt.build_hold_orders(pos, PX), pos, EQ, R, prices=PX, fractional=True,
                                       blocked={"S01": "earnings Thu Oct 08, in 1 day", "S03": "sold by the earnings-day stop"})
check("S01 (earnings) and S03 (stop sale) skipped -> S23 replaced by S10 (rank 10)",
      info["replaced"] == [("S23", 23, "S10", 10, 5_000.0)] and [s for s, _ in info["skipped"]] == ["S01", "S03"]
      and "S01" not in set(o["Symbol"]) and "S03" not in set(o["Symbol"]), (info["replaced"], info["skipped"]))

# 5) the strategy's own orders come first: no second SELL, a partial sell becomes a full one, its BUY is not reused
strat = pd.DataFrame([
    {"Symbol": "S22", "Side": "SELL", "Shares": 30, "Price": 100.0, "Est_Value": 3_000.0, "Current_Shares": 30,
     "Target_Shares": 0, "Target_Weight_%": 0.0, "Target_Value": 0.0},
    {"Symbol": "S03", "Side": "BUY", "Shares": 30, "Price": 100.0, "Est_Value": 3_000.0, "Current_Shares": 0,
     "Target_Shares": 30, "Target_Weight_%": 3.0, "Target_Value": 3_000.0},
    {"Symbol": "S24", "Side": "SELL", "Shares": 10, "Price": 100.0, "Est_Value": 1_000.0, "Current_Shares": 40,
     "Target_Shares": 30, "Target_Weight_%": 3.0, "Target_Value": 3_000.0}], columns=pt.ORDER_COLUMNS)
pos = {**{f"S{i:02d}": 100 for i in (2, 4, 5, 6, 7, 8, 9, 10)}, "S22": 30, "S24": 40}   # S01, S03 not held
o, info = pt.build_account_exit_orders(strat, pos, EQ, R, prices=PX, fractional=True)
s24 = o[o["Symbol"] == "S24"]
check("S22 sold by the strategy and ranked 22: exactly one SELL row (30 shares), not sold twice",
      side(o, "S22") == ["SELL"] and o.loc[o.Symbol == "S22", "Shares"].iloc[0] == 30
      and "S22" not in [s for s, *_ in info["sold"]], o[["Symbol", "Side", "Shares"]].values.tolist())
check("S24 partial strategy SELL (10 of 40) becomes one full SELL of 40, refilled by S01 (S03 is the strategy's buy)",
      len(s24) == 1 and s24["Shares"].iloc[0] == 40 and s24["Target_Shares"].iloc[0] == 0
      and info["replaced"] == [("S24", 24, "S01", 1, 4_000.0)] and side(o, "S03") == ["BUY"]
      and o.loc[o.Symbol == "S03", "Shares"].iloc[0] == 30, (s24.values.tolist(), info["replaced"]))

# 6) cap and small sales: a refill never exceeds 19.8% of equity; under max($100, 1% of equity) -> no refill
pos = {**{f"S{i:02d}": 1 for i in (2, 4, 5, 6, 7, 8, 9, 10)}, "S21": 300, "S25": 5}   # $30,000 and $500
o, info = pt.build_account_exit_orders(pt.build_hold_orders(pos, PX), pos, EQ, R, prices=PX, fractional=True)
check("$30,000 sale -> S01 refill capped at $19,800; the $500 sale (under $1,000) gets no refill, S03 stays unbought",
      info["replaced"] == [("S21", 21, "S01", 1, 19_800.0)] and "S03" not in set(o["Symbol"]), info["replaced"])
o, info = pt.build_account_exit_orders(pt.build_hold_orders(pos, PX), pos, EQ, R.iloc[:0], prices=PX, fractional=True)
check("no ranking for the day -> nothing sold (never a sell-everything on missing data)",
      info["sold"] == [] and (o["Side"] == "HOLD").all() and "no ranking" in info["note"], info)
o, _ = pt.build_account_exit_orders(pt.build_hold_orders({"S21": 33}, PX), {"S21": 33}, EQ,
                                    ranking([("S01", 1, 333.0), ("S21", 21, 100.0)]), prices={"S21": 100.0, "S01": 333.0})
check("whole shares when not fractional: $3,300 / $333 -> 9 shares", o.loc[o.Symbol == "S01", "Shares"].tolist() == [9],
      o.values.tolist())
txt = pt.account_exit_text(info | {"note": ""})
check("summary text names the rule and the positions with no rank", "no position is worse than rank 20" in txt
      and "positions with no rank: none" in txt, txt)

# --- end to end through plan_orders with fake report files ---
tmp = tempfile.mkdtemp()
pt.EARNINGS_STOP_STATE = os.path.join(tmp, "no_stop_state.json")      # no earnings-day stop sales
picks, sig, mid = (os.path.join(tmp, n) for n in ("picks.csv", "sig.csv", "mid.csv"))
pd.DataFrame([{"As_Of": "2026-10-07", "Last_Rebalance": "2026-10-02", "Last_Decision": "2026-10-02", "Strategy": "C6",
               "Symbol": "S01", "Strategy_Weight": 0.1, "Provisional_Weight": 0.10, "Close": 100.0}]).to_csv(picks, index=False)
rows = []
for d, chk in (("2026-10-06", 0), ("2026-10-07", 1)):
    for i in range(1, 26):
        rows.append({"Date": d, "Symbol": f"S{i:02d}", "Close": 100.0, "Strategy_Score": 70 - i, "Strategy_Rank": i,
                     "Strategy_Weight": 0, "Provisional_Weight": 0.098 if i <= 10 else 0.0, "Regime_On": 1, "Midweek_Check": chk})
    rows.append({"Date": d, "Symbol": "NEG", "Close": 5.0, "Strategy_Score": -3, "Strategy_Rank": 26, "Strategy_Weight": 0,
                 "Provisional_Weight": 0.0, "Regime_On": 1, "Midweek_Check": chk})
pd.DataFrame(rows).to_csv(sig, index=False)
pd.DataFrame(columns=["As_Of", "Event", "Event_Date", "Action", "Sell", "Buy", "Message", "Weight_%"]).to_csv(mid, index=False)
real_cp = pt.current_prices
asked = []
pt.current_prices = lambda syms: asked.append(sorted(syms)) or {"XYZ": 20.0}  # Alpaca market data stand-in (read-only)
try:
    held = {**{f"S{i:02d}": 100 for i in (1, 2, 4, 5, 6, 7, 8, 9, 10)}, "S24": 50, "XYZ": 10, "NEG": 30}
    o, meta, _ = pt.plan_orders("auto", EQ, held, picks_csv=picks, signal_csv=sig, midweek_csv=mid, fractional=True, cash=1_000)
    ex = meta["account_exits"]
    check("quiet Wed check: NEG ($150) and XYZ ($200, priced from market data) sold, too small to refill; S24 ($5,000) "
          "-> S03 (rank 3); source hold+exits",
          [s for s, *_ in ex["sold"]] == ["NEG", "XYZ", "S24"] and ex["replaced"] == [("S24", 24, "S03", 3, 5_000.0)]
          and meta["source"] == "hold+exits" and asked == [["XYZ"]], (ex, meta["source"], asked))
    check("then the spare-cash step: $1,000 + $5,350 sold - $5,000 bought - $1,000 kept = $350 < $1,000 -> no buy",
          abs(meta["cash_deploy"]["spare"] - 350) < 0.01 and not meta["cash_deploy"]["buys"] and not meta["cash_deploy"]["topups"],
          meta["cash_deploy"])
    rep, ok = pt.reconcile_positions("hold+exits")
    check("reconcile_positions('hold+exits') -> empty, ok (the account differs from Strategy_Weight on purpose)",
          rep.empty and ok is True, (rep, ok))
    # the strategy itself sells S22 (REPLACE S22 -> S03); the account also holds S22 and S24
    pd.DataFrame([{"As_Of": "2026-10-07", "Event": "mid-week check", "Event_Date": "2026-10-07", "Action": "REPLACE",
                   "Sell": "S22", "Buy": "S03", "Message": "x", "Weight_%": 9.8}]).to_csv(mid, index=False)
    held = {**{f"S{i:02d}": 100 for i in (1, 2, 4, 5, 6, 7, 8, 9, 10)}, "S22": 98, "S24": 40}
    o, meta, _ = pt.plan_orders("auto", EQ, held, picks_csv=picks, signal_csv=sig, midweek_csv=mid, fractional=True, cash=1_000)
    ex = meta["account_exits"]
    check("strategy REPLACE S22 -> S03 kept as is (one SELL S22, one BUY S03); the account rule adds only S24, no refill left",
          side(o, "S22") == ["SELL"] and side(o, "S03") == ["BUY"] and [s for s, *_ in ex["sold"]] == ["S24"]
          and ex["replaced"] == [] and list(o["Symbol"]).count("S24") == 1, o[["Symbol", "Side", "Shares"]].values.tolist())
    check("S24's $4,000 goes to the spare-cash step: tops up rank 1 S01 ($10,000 -> +$4,000); source midweek+exits+cash",
          meta["source"] == "midweek+exits+cash" and meta["cash_deploy"]["topups"] == [("S01", 1, 40.0, 4_000.0)],
          (meta["source"], meta["cash_deploy"]))
    # not a check day (Tue): nothing sold by the account rule
    pd.DataFrame([{**pd.read_csv(picks).iloc[0].to_dict(), "As_Of": "2026-10-06"}]).to_csv(picks, index=False)
    o, meta, _ = pt.plan_orders("auto", EQ, {"S01": 100, "S24": 50}, picks_csv=picks, signal_csv=sig, midweek_csv=mid,
                                fractional=True, cash=1_000)
    check("Tue (no check): no account sell rule, S24 held", meta["source"] == "hold" and "account_exits" not in meta
          and side(o, "S24") == ["HOLD"], (meta["source"], o.values.tolist()))
    # Friday rebalance unchanged (provisional): no account-exit step (non-targets are sold by build_orders as before)
    pd.DataFrame([{**pd.read_csv(picks).iloc[0].to_dict(), "As_Of": "2026-10-02"}]).to_csv(picks, index=False)
    o, meta, _ = pt.plan_orders("auto", EQ, {"S24": 50}, picks_csv=picks, signal_csv=sig, midweek_csv=mid, fractional=True,
                                cash=1_000, decision="2026-10-02")
    check("Friday (provisional) is unchanged: no account_exits step", meta["source"] == "provisional"
          and "account_exits" not in meta, meta["source"])
finally:
    pt.current_prices = real_cp

print(f"\n{len(FAIL)} failed" if FAIL else "\nACCOUNT EXIT OK")
sys.exit(1 if FAIL else 0)
