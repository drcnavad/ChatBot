"""Strategy tab (Chirag, Oct 16, 2026), pure functions with fake Alpaca data - no network, no orders:
  - Last decision (dashboard/details/last_decision.decision_table): Bought / Sold / Held from the fills since the decision
    day, sleeve from the ledger, ranks at the decision and today, weight = market value / equity
  - Live vs benchmark (dashboard/details/live_vs_benchmark): time-weighted account (deposits are not returns), price
    returns, 'since the switch' blank (not 0) until a session after it
  - decision_ranks.save / load round trip (temp file)
  - Home tab sector performance (dashboard/sector_table.sector_returns): equal-weight average per sector
Run: python tests/run_tests.py  (or python tests/test_strategy_tab.py)"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import pandas as pd

from dashboard.details.last_decision import COLUMNS, decision_table
from dashboard.details.live_vs_benchmark import account_index, price_index, returns_table

FAIL = []

def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info and not ok else ""))
    if not ok:
        FAIL.append(name)

# ---------------------------------------------------------------- Last decision
fill = lambda s, side, q, t: {"symbol": s, "side": side, "qty": str(q), "transaction_time": t}
FILLS = [fill("OLD", "buy", 5, "2026-10-09T19:30:00Z"),                       # before the decision: ignored
         fill("NEW", "buy", 10, "2026-10-16T19:30:22Z"),                      # bought
         fill("GONE", "sell", 7, "2026-10-16T19:30:10Z"),                     # sold in full
         fill("TRIM", "sell", 2, "2026-10-16T19:30:11Z"),                     # held, trimmed
         fill("LATE", "buy", 3, "2026-10-19T14:00:05Z"),                      # the 9 AM rest the next day: bought
         fill("FLIP", "buy", 1, "2026-10-16T19:30:00Z"), fill("FLIP", "sell", 1, "2026-10-16T20:00:00Z")]  # in and out
POS = [{"symbol": "NEW", "qty": "10", "market_value": "1000"}, {"symbol": "TRIM", "qty": "8", "market_value": "800"},
       {"symbol": "KEEP", "qty": "4", "market_value": "400"}, {"symbol": "LATE", "qty": "3", "market_value": "300"},
       {"symbol": "OLD", "qty": "5", "market_value": "500"}]
LED = {"day": "2026-10-16", "halves": {"lowvol": {"shares": {"NEW": 10, "KEEP": 4, "OLD": 5}},
                                       "sector": {"shares": {"TRIM": 8, "LATE": 3, "GONE": 0.0}}}}
THEN = pd.DataFrame({"Symbol": ["NEW", "GONE", "TRIM"], "Rank_Vol": [1, 30, 12], "Rank_Sector": [pd.NA, 9, 2]})
NOW = pd.DataFrame({"Symbol": ["NEW", "TRIM"], "Rank_Vol": [2, 11], "Rank_Sector": [pd.NA, 3]})
t = decision_table("2026-10-16", POS, FILLS, 5000.0, LED, THEN, NOW)
a = t.set_index("Symbol")
check("columns: Symbol, Action, Sleeve, 4 ranks, Current weight % (no Signal / plan / weight before / after)",
      list(t.columns) == COLUMNS and not {"Signal", "Next rebalance plan", "Weight before %"} & set(t.columns), list(t.columns))
check("Bought = not held before the decision, held now (also the 9 AM rest next day)",
      a.at["NEW", "Action"] == "Bought" and a.at["LATE", "Action"] == "Bought", a["Action"].to_dict())
check("Sold = held before, none now; weight 0; no sleeve", a.at["GONE", "Action"] == "Sold" and a.at["GONE", "Current weight %"] == 0
      and a.at["GONE", "Sleeve"] == "—", a.loc["GONE"].to_dict())
check("Held: trimmed / untouched", a.at["TRIM", "Action"] == "Held (trimmed)" and a.at["KEEP", "Action"] == "Held"
      and a.at["OLD", "Action"] == "Held", a["Action"].to_dict())
check("bought and sold again after the decision: not listed", "FLIP" not in a.index)
check("sleeve from the ledger", a.at["NEW", "Sleeve"] == "Low vol" and a.at["TRIM", "Sleeve"] == "Sector rotation")
check("ranks at the decision and today", a.at["TRIM", "Rank Vol at decision"] == 12 and a.at["TRIM", "Rank Vol today"] == 11
      and a.at["TRIM", "Rank Sector at decision"] == 2 and a.at["TRIM", "Rank Sector today"] == 3
      and pd.isna(a.at["NEW", "Rank Sector today"]) and pd.isna(a.at["KEEP", "Rank Vol at decision"]), a.loc["TRIM"].to_dict())
check("weight = market value / equity", a.at["NEW", "Current weight %"] == 20.0 and a.at["TRIM", "Current weight %"] == 16.0)
check("order: Bought, Held, Sold", list(t["Action"].str.split().str[0]) == sorted(t["Action"].str.split().str[0],
      key=["Bought", "Held", "Sold"].index), list(t["Action"]))
t2 = decision_table("2026-10-16", POS, FILLS, 5000.0, LED, None, NOW)
check("no saved ranks for the day: 'at decision' blank, today shown", t2["Rank Vol at decision"].isna().all()
      and t2.set_index("Symbol").at["NEW", "Rank Vol today"] == 2)

# ---------------------------------------------------------------- Live vs benchmark
H = pd.DataFrame({"Date": pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-15"]),
                  "Equity": [1000.0, 1100.0, 2100.0, 2205.0], "Cashflow": [1000.0, 0.0, 1000.0, 0.0]})
idx = account_index(H)
check("time-weighted: a deposit is not a return", abs(idx.iloc[1] - 1.1) < 1e-12 and abs(idx.iloc[2] - 1.1) < 1e-12
      and abs(idx.iloc[3] - 1.1 * 1.05) < 1e-12, idx.round(4).tolist())
today = pd.Timestamp("2026-10-16")
idx = account_index(H, 2100.0, 0.0, today)
check("the equity now extends today", idx.index[-1] == today and abs(idx.iloc[-1] / idx.iloc[-2] - 2100 / 2205) < 1e-12)
idx2 = account_index(pd.concat([H, pd.DataFrame({"Date": [today], "Equity": [9.0], "Cashflow": [0.0]})]), 2100.0, 0.0, today)
check("a history row for today is replaced by the live equity", abs(idx2.iloc[-1] - idx.iloc[-1]) < 1e-12)
q = price_index(pd.Series([100.0, 110.0, 121.0], index=pd.to_datetime(["2026-10-02", "2026-10-15", "2026-10-16"])), 99.0, today)
check("price index: the latest quote replaces today's bar", q.iloc[-1] == 99.0 and len(q) == 3)
tb, col = returns_table([("Acct", idx), ("QQQ", q)], "2026-10-16", "2026-10-02")
r = tb.set_index("Line")
check("on the switch day 'since the switch' is blank (not a fake 0)", r[col].isna().all(), r[col].tolist())
check("today and since-start returns", abs(r.at["QQQ", "Today %"] - (99 / 110 - 1) * 100) < 1e-9
      and abs(r.at["QQQ", "Since Oct 2 %"] - (-1.0)) < 1e-9 and abs(r.at["Acct", "Today %"] - (2100 / 2205 - 1) * 100) < 1e-9,
      r.to_dict())
nxt = pd.Timestamp("2026-10-19")
tb3, _ = returns_table([("QQQ", pd.concat([q, pd.Series([108.9], index=[nxt])]))], "2026-10-16", "2026-10-02")
check("a session after the switch: since-the-switch return from the switch close", abs(tb3[col].iloc[0] - 10.0) < 1e-9,
      tb3.to_dict())

# ---------------------------------------------------------------- decision_ranks round trip (temp file, real saved data)
import decision_ranks  # noqa: E402
tmp = os.path.join(tempfile.mkdtemp(), "ranks.csv")
d, live = decision_ranks.ranks_on()
decision_ranks.save(d, path=tmp)
decision_ranks.save(d, path=tmp)                       # the same day twice: replaced, not duplicated
back = decision_ranks.load(d, path=tmp)
check("decision_ranks: save twice + load = one copy of that day's ranks", back is not None and len(back) == len(live)
      and back["Rank_Vol"].equals(live["Rank_Vol"].reset_index(drop=True)), len(back) if back is not None else None)
check("decision_ranks: an unsaved day -> None", decision_ranks.load("2020-01-02", path=tmp) is None)
sv = live.dropna(subset=["Rank_Sector"])["Rank_Sector"]
check("Rank_Sector = 1..n", sorted(sv.tolist()) == list(range(1, len(sv) + 1)), sorted(sv.tolist())[:5])

# ---------------------------------------------------------------- Home tab: sector performance (equal weight per sector)
from dashboard.sector_table import WINDOWS, sector_returns  # noqa: E402
import sector_mapping as sm  # noqa: E402
tech = [x for x in sm.tradable_symbols if sm.symbol_sector.get(x) == "Technology"][:2]
en = [x for x in sm.tradable_symbols if sm.symbol_sector.get(x) == "Energy"][:1]
days = pd.bdate_range("2026-06-01", "2026-10-16")
px = {tech[0]: 100.0, tech[1]: 100.0, en[0]: 100.0}
rows = [{"Date": d, "Symbol": x, "Close": p0 * (1.2 if (x == tech[0] and d > pd.Timestamp("2026-09-16")) else 1.0)
         * (0.9 if (x == en[0] and d > pd.Timestamp("2026-10-02")) else 1.0)} for d in days for x, p0 in px.items()]
sr = sector_returns(pd.DataFrame(rows), symbols=list(px)).set_index("Sector")
check("sector return = equal-weight mean of its stocks (one stock +20% a month ago, the other flat -> +10%)",
      abs(sr.at["Technology", "1 month %"] - 10.0) < 1e-9 and abs(sr.at["Technology", "14 days %"]) < 1e-9
      and sr.at["Technology", "Stocks"] == 2, sr.to_dict())
check("14 days catches a move 10 sessions ago; sorted by 1 month", abs(sr.at["Energy", "14 days %"] + 10.0) < 1e-9
      and list(sr.index) == ["Technology", "Energy"] and list(sr.columns) == ["Stocks", *WINDOWS], list(sr.index))

print("STRATEGY TAB OK" if not FAIL else f"{len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
