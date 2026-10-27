"""Strategy tab (Chirag, Oct 16, 2026), pure functions with fake Alpaca data - no network, no orders:
  - Last decision (dashboard/details/last_decision.ranking_table): the RS short-term top 20 at its own decision, held marked
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
# ---------------------------------------------------------------- Last decision: one ranking table per half
from dashboard.details.last_decision import half_decision_day, ranking_table  # noqa: E402
REAL_RANKS = os.path.join(ROOT, "Reports", "decision_ranks.csv")   # the runner points decision_ranks at a temp file
for half in ("rs_short",):
    day = half_decision_day(half, path=REAL_RANKS)
    if day is None:
        continue
    col = decision_ranks.RANK_COL[half]
    ranks = decision_ranks.load(day, path=REAL_RANKS)
    top = ranks.dropna(subset=[col]).sort_values(col)["Symbol"].tolist()
    held = top[:1] + ["ZZZ"]                                      # one ranked holding, one unranked
    led = {"version": 3, "halves": {h: {"shares": {x: 1.0 for x in held} if h == half else {}, "cash": 100.0}
                                    for h in ("rs_short",)}}
    pos = [{"symbol": x, "qty": "1", "current_price": "450", "market_value": "450"} for x in held]
    t, others = ranking_table(half, day, pos, led, path=REAL_RANKS, equity=1100.0)
    check(f"{half}: ranks 1..n, at most 20", list(t["Rank"]) == list(range(1, len(t) + 1)) and len(t) <= 20, list(t["Rank"]))
    check(f"{half}: held stock marked", t.iloc[0]["Held by"].startswith("✓"), t.iloc[0].to_dict())
    check(f"{half}: an unranked holding is listed separately", others == ["ZZZ"], others)

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
