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

# ---------------------------------------------------------------- Nov 8, 2026: the rank table never disappears
import dashboard.details.last_decision as ld  # noqa: E402
from datetime import datetime  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402
t, last_d, now_d, prev_d, err = ld.decision_table(None, 0.0)          # no live account read at all
check("decisions: no live account -> the rank columns still show (saved snapshot + latest close)",
      len(t) > 0 and t[ld.RANK_NEXT].notna().any() and (t[ld.NEXT] == "—").all() and prev_d is None and err is None,
      (len(t), err))
_real = ld.preview
ld.preview = lambda *a, **k: (_ for _ in ()).throw(ValueError("targets exceed equity"))
try:
    t2, *_rest, err2 = ld.decision_table([{"symbol": "AAA", "qty": 1, "current_price": 1.0}], 1000.0)
finally:
    ld.preview = _real
check("decisions: a failing preview keeps the table and reports why", len(t2) > 0 and "targets exceed equity" in (err2 or ""), err2)
_ct = ZoneInfo("America/Chicago")
check("next rebalance after Sun Nov 8, 2026 = Fri Nov 13 2:30 PM CT",
      ld.next_rebalance_slot(datetime(2026, 11, 8, 21, 0, tzinfo=_ct)) == datetime(2026, 11, 13, 14, 30, tzinfo=_ct))
check("next rebalance after Fri Nov 13 2:30 PM = Fri Nov 20 (not the Mon-Thu runs)",
      ld.next_rebalance_slot(datetime(2026, 11, 13, 14, 31, tzinfo=_ct)) == datetime(2026, 11, 20, 14, 30, tzinfo=_ct))
check("Thanksgiving week: Fri Nov 27 is an early close -> 11:00 AM CT",
      ld.next_rebalance_slot(datetime(2026, 11, 23, 9, 0, tzinfo=_ct)) == datetime(2026, 11, 27, 11, 0, tzinfo=_ct))

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
