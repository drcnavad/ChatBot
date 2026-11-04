"""Strategy tab (Chirag, Oct 16, 2026), pure functions with fake Alpaca data - no network, no orders:
  - Decisions table (dashboard/details/last_decision): last Friday from the order log, next Friday via sleeves.plan(rebalance=True)
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


FAIL = []

def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info and not ok else ""))
    if not ok:
        FAIL.append(name)

import decision_ranks  # noqa: E402
# ---------------------------------------------------------------- Decisions table: last Friday from the order log
from dashboard.details.last_decision import last_rebalance  # noqa: E402
log = os.path.join(tempfile.mkdtemp(), "orders.csv")
pd.DataFrame([{"Symbol": "AAA", "Side": "BUY", "Status": "FILLED", "As_Of": "2026-10-30"},
              {"Symbol": "BBB", "Side": "SELL", "Status": "FILLED", "As_Of": "2026-10-30"},
              {"Symbol": "CCC", "Side": "BUY", "Status": "STAGED", "As_Of": "2026-10-30"},
              {"Symbol": "DDD", "Side": "BUY", "Status": "FILLED", "As_Of": "2026-11-02"}]).to_csv(log, index=False)
day, side = last_rebalance(log)
check("last rebalance = the latest Friday As_Of (+ a later Monday trade); staged rows ignored",
      day == pd.Timestamp("2026-10-30") and side == {"AAA": "Buy Oct 30", "BBB": "Sell Oct 30", "DDD": "Buy Nov 2"}, (day, side))
# Next decision preview: sleeves.plan(rebalance=True) on a weekday = the Friday book (real data, no orders)
import sleeves  # noqa: E402
d, inp = sleeves.rebalance_inputs()
o, info = sleeves.plan(d, {}, 100000.0, {}, inp=inp, rebalance=True)
buys = o[o["Side"] == "BUY"]["Symbol"].tolist()
from sector_mapping import symbol_sector as _ss  # noqa: E402
check("rebalance preview: up to 10 buys, max 4 per sector", 0 < len(buys) <= 10
      and pd.Series([_ss.get(x) for x in buys]).value_counts().max() <= 4, buys)
check("rebalance preview leaves the live inputs alone", not bool(sleeves.inputs()["weekly"].get(d, False)) or d.weekday() == 4)

# ---------------------------------------------------------------- decision_ranks round trip (temp file, real saved data)
import decision_ranks  # noqa: E402
tmp = os.path.join(tempfile.mkdtemp(), "ranks.csv")
d, live = decision_ranks.ranks_on()
decision_ranks.save(d, path=tmp)
decision_ranks.save(d, path=tmp)                       # the same day twice: replaced, not duplicated
back = decision_ranks.load(d, path=tmp)
check("decision_ranks: save twice + load = one copy of that day's ranks", back is not None and len(back) == len(live)
      and back["Rank_RS_Short"].equals(live["Rank_RS_Short"].reset_index(drop=True)), len(back) if back is not None else None)
check("decision_ranks: an unsaved day -> None", decision_ranks.load("2020-01-02", path=tmp) is None)
sv = live.dropna(subset=["Rank_RS_Short"])["Rank_RS_Short"]
check("Rank_RS_Short = 1..n", sorted(sv.tolist()) == list(range(1, len(sv) + 1)), sorted(sv.tolist())[:5])

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
