"""Shared live / forward-test book rules (sleeves.finalize_book): sector cap on replacements,
15% per-stock cap, rank-20 new buys, drop unheld starting picks worse than rank 20 (held kept),
and cool_buys matching replace_blocked_new_slots. No broker calls."""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ["STOCK_ANALYSIS_SLEEVE_LEDGER"] = os.path.join(tempfile.mkdtemp(), "sleeve_ledger.json")

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
import backtest_engine as be  # noqa: E402
import forward_test as ft  # noqa: E402
import sleeves  # noqa: E402
import paper_trade as pt  # noqa: E402

FAIL = []

def check(ok, what, got=""):
    print(("PASS " if ok else "FAIL ") + what + ("" if ok else f"  ({got})"))
    if not ok:
        FAIL.append(what)

check(be.MAX_WEIGHT == 0.15 and pt.MAX_ORDER_PCT == 0.15, "MAX_WEIGHT and MAX_ORDER_PCT are 15%")
check(be.WINNER.get("score") == "rs_short" and be.WINNER.get("n") == 10 and be.WINNER.get("calendar") == "weekly",
      "WINNER is live short-term RS, n=10, weekly", be.WINNER.get("score"))
check(be.RS_SHORT_WINDOWS == (10, 21, 63, 63), "RS_SHORT_WINDOWS = 10/21/63/63")

# --- sector cap on swapped-in names (fake tickers have no sector: unique per symbol unless we assign) ---
# Five healthcare names plus fillers. Patch symbol_sector for this test only.
import sector_mapping as sm
hc = ["HC1", "HC2", "HC3", "HC4", "HC5"]
oth = ["O1", "O2", "O3", "O4", "O5", "O6"]
all_n = hc + oth
rank_of = {s: i + 1 for i, s in enumerate(all_n)}  # HC1..HC5 are ranks 1-5
orig = dict(sm.symbol_sector)
try:
    sm.symbol_sector.update({s: "Health Care" for s in hc})
    sm.symbol_sector.update({s: f"Sec{s}" for s in oth})
    # starting book: 4 HC + 6 other; swap HC5 in as a replacement for nothing — fill should refuse a 5th HC
    book = sleeves.finalize_book(
        names=hc[:4] + oth[:6], held=set(), rank_of=rank_of, blocked={}, stretched=set(), soon=set(),
        priced=set(all_n), n_picks=10, invested=0.99, max_rank=20, sector_cap=0.4, max_weight=0.15)
    check(sum(1 for s in book if s in hc) == 4 and "HC5" not in book,
          "sector cap: no 5th healthcare name in a 10-slot book", book)
    # replacement path: drop O6 (blocked), next best is HC5 — must skip it and take... there is no 11th name.
    # Use a starting book of 4 HC + 5 other (9 slots) so the fill tries HC5 then would need another Other — only Oth are O1-O6, O1-O5 in book, O6 not starting.
    book2 = sleeves.finalize_book(
        names=hc[:4] + oth[:5], held=set(), rank_of=rank_of, blocked={}, stretched=set(), soon=set(),
        priced=set(all_n), n_picks=10, invested=0.99, max_rank=20, sector_cap=0.4)
    check("HC5" not in book2 and "O6" in book2 and sum(1 for s in book2 if s in hc) == 4,
          "replacement fill skips a 5th HC and takes the next other-sector name", book2)
finally:
    sm.symbol_sector.clear()
    sm.symbol_sector.update(orig)

# --- 15% cap: 6 names at 99% would be 16.5% each ---
names = [f"S{i}" for i in range(6)]
book = sleeves.finalize_book(names, set(), {s: i + 1 for i, s in enumerate(names)}, {}, set(), set(),
                             set(names), 6, 0.99, max_rank=20, sector_cap=1.0, max_weight=0.15)
check(len(book) == 6 and all(abs(v - 0.15) < 1e-12 for v in book.values()) and abs(sum(book.values()) - 0.90) < 1e-12,
      "15% cap: 6 names at 99% invest become 15% each, 10% cash", book)

# --- rank-20: unheld starting pick #21 is dropped; held #23 is kept when the book is already 10 ---
ranked = [f"R{i}" for i in range(1, 25)]
rank_of = {s: i for i, s in enumerate(ranked, 1)}
start = ranked[:9] + ["R21"]          # 9 good + unheld #21
held = set()
book = sleeves.finalize_book(start, held, rank_of, {}, set(), set(), set(ranked), 10, 0.99,
                             max_rank=20, sector_cap=1.0)
check("R21" not in book and "R10" in book and len(book) == 10,
      "unheld starting pick worse than rank 20 is dropped and filled from rank <= 20", book)

start10 = ranked[:9] + ["R23"]
held = {"R23"}
book = sleeves.finalize_book(start10, held, rank_of, {}, set(), set(), set(ranked), 10, 0.99,
                             max_rank=20, sector_cap=1.0)
check("R23" in book and len(book) == 10 and "R10" not in book,
      "held rank 23 stays when the book is already 10 (R10 is not swapped in)", book)

# more than 10 names (9 good + held 23 + held 24) trims to best 10, dropping 23/24
book = sleeves.finalize_book(ranked[:9] + ["R23", "R24"], {"R23", "R24"}, rank_of, {}, set(), set(),
                             set(ranked), 10, 0.99, max_rank=20, sector_cap=1.0)
check("R24" not in book and "R23" in book and "R10" not in book and len(book) == 10,
      "when the book is over n, trim the worst already-in name (R24); do not pull in R10", book)

# --- cool_buys last-day membership matches finalize_book / replace_blocked (synthetic frame) ---
# 3 sessions, Friday-like weekly on first and last; start with a stretched new name and a rank-21 unheld.
idx = pd.to_datetime(["2026-10-30", "2026-11-02", "2026-11-06"])
cols = ["A", "B", "C", "D", "E", "F"]
# ranks by score: A B C D E F ; start weights on 10/30 and 11/06 include F (we'll mark F stretched on 11/06) and a dummy
w = pd.DataFrame(0.0, index=idx, columns=cols)
w.loc["2026-10-30", ["A", "B", "C"]] = 0.33
w.loc["2026-11-02"] = w.loc["2026-10-30"]
w.loc["2026-11-06", ["A", "B", "F"]] = 0.33   # F is a new stretched pick; D should replace it
score = pd.DataFrame({"A": 6, "B": 5, "C": 4, "D": 3, "E": 2, "F": 1}, index=idx, dtype=float)
ok = pd.DataFrame(True, index=idx, columns=cols)
stretched = pd.DataFrame(False, index=idx, columns=cols)
stretched.loc["2026-11-06", "F"] = True
weekly = pd.Series([True, False, True], index=idx)
close = pd.DataFrame(10.0, index=idx, columns=cols)
out = ft.cool_buys(w, score, ok, stretched, weekly, held={"A", "B", "C"}, n_picks=3, sector_cap=1.0,
                   soon=None, close=close, max_rank=20, max_weight=0.15)
# last day: F dropped (stretched, not held — held is A,B,C but C not in starting last-day names; A,B held; F replaced by D)
last = out.loc["2026-11-06"]
# F is a new stretched pick, so it is dropped; held C (previous day + ledger) is added back;
# the book is already n=3 so D is not filled. Matches live replace_blocked (held stay).
check(set(last[last > 0].index) == {"A", "B", "C"} and "F" not in last[last > 0].index,
      "cool_buys last day drops stretched new F and keeps held C", last.to_dict())
check(all(abs(v - 0.15) < 1e-12 for v in last[last > 0]), "cool_buys applies the 15% cap", last.to_dict())

# first rebalance day also goes through finalize_book (t=0)
first = out.loc["2026-10-30"]
check(set(first[first > 0].index) == {"A", "B", "C"}, "cool_buys runs on the first rebalance day too", first.to_dict())

print("LIVE BOOK RULES OK" if not FAIL else f"LIVE BOOK RULES FAILURES ({len(FAIL)}): {FAIL}")
sys.exit(1 if FAIL else 0)
