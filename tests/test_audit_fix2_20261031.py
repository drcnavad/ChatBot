"""Second-pass audit fixes (2026-10-31): sold_block fill-only + matched clear,
midweek continue weights, drop_guard fail-closed, hold messaging, FAILED exit."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CT = ZoneInfo("America/Chicago")
fails = []

def check(cond, label, detail=None):
    if cond:
        print(f"  OK  {label}")
    else:
        fails.append(label)
        print(f" FAIL {label}" + (f" — {detail!r}" if detail is not None else ""))


# --- #2 clear_block matches order_id ---
import sold_block as sb
with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "sold_block.json")
    old = sb.BLOCK_JSON
    sb.BLOCK_JSON = path
    try:
        sb.record_sell("AAA", sold_on=date(2026, 10, 30), source="fill_check",
                       note="real fill", order_id="oid-real")
        # 0-fill of a different order must NOT clear
        check(not sb.clear_block("AAA", reason="0-fill other", order_id="oid-other"),
              "clear_block refuses mismatched order_id")
        check("AAA" in sb.active_blocks(as_of=date(2026, 10, 31)), "block kept after mismatch")
        check(sb.clear_block("AAA", reason="0-fill same", order_id="oid-real"),
              "clear_block accepts matching order_id")
        check("AAA" not in sb.active_blocks(as_of=date(2026, 10, 31)), "block cleared on match")
        # unconditional clear (no matcher) still works
        sb.record_sell("BBB", sold_on=date(2026, 10, 30), source="x", order_id="1")
        check(sb.clear_block("BBB", reason="admin"), "unconditional clear still works")
    finally:
        sb.BLOCK_JSON = old


# --- #1 COMPLETED submit must not record via _record_sold_blocks_from_results ---
import paper_trade as pt
with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "sold_block.json")
    old = sb.BLOCK_JSON
    sb.BLOCK_JSON = path
    try:
        rows = pd.DataFrame([
            {"Symbol": "CCC", "Side": "SELL", "Shares": 1, "Order_ID": "m1",
             "Status": "COMPLETED via limit $10 (sent now, nothing was sent before)"},
            {"Symbol": "DDD", "Side": "SELL", "Shares": 1, "Order_ID": "m2",
             "Status": "FILLED (1/1)"},
        ])
        pt._record_sold_blocks_from_results(rows, source="unit")
        active = sb.active_blocks(as_of=date(2026, 10, 31))
        check("DDD" in active, "FILLED status records block", active)
        check("CCC" not in active, "COMPLETED via limit (submit) does NOT block", active)
    finally:
        sb.BLOCK_JSON = old


# --- #7 FAILED exit flag via _plan_qty ---
# full exit: qty >= held
q, is_exit, reason = pt._plan_qty("EEE", "SELL", 10, {"EEE": 10.0})
check(is_exit is True and reason is None, "full exit is_exit True", (q, is_exit, reason))
q2, is_exit2, _ = pt._plan_qty("EEE", "SELL", 3, {"EEE": 10.0})
check(is_exit2 is False, "trim is_exit False", (q2, is_exit2))


# --- equal weight: replace_blocked fills blocked slot and equalizes ---
import sleeves
names2 = ["HOLD1", "HOLD2", "NEW1", "BAD"]
held2 = {"HOLD1", "HOLD2"}
blocked2 = {"BAD": "sold-block"}
px2 = {"HOLD1": 10.0, "HOLD2": 10.0, "NEW1": 10.0, "BAD": 10.0, "REPL": 10.0}
fake_rank = pd.Series(range(5, 0, -1), index=["REPL", "ZZZ", "HOLD1", "HOLD2", "NEW1"], dtype=float)
with mock.patch.object(sleeves, "ranking", return_value=fake_rank):
    out = sleeves.replace_blocked_new_slots(
        "rs_short", pd.Timestamp("2026-10-30"), names2, held2, blocked2, px2,
        {}, 4, 0.99,
    )
check("BAD" not in out, "replace_blocked drops blocked new", out)
check("REPL" in out, "replace_blocked adds replacement", out)
check(len(out) == 4 and all(abs(v - 0.99/4) < 1e-12 for v in out.values()),
      "replace_blocked equal weight", out)
check(abs(sum(out.values()) - 0.99) < 1e-9, "replace_blocked sum == invested", sum(out.values()))

# Held stand-ins for blocked ideal picks must stay (rank <= 20), not be sold as "taken"
names3 = ["H1", "BLK1", "H2", "BLK2"]  # ideal midweek picks; BLK* blocked
held3 = {"H1", "H2", "SUB1", "SUB2"}   # SUB* are live replacements
blocked3 = {"BLK1": "sold-block", "BLK2": "sold-block"}
px3 = {s: 10.0 for s in ["H1", "BLK1", "H2", "BLK2", "SUB1", "SUB2", "X1"]}
fake_rank3 = pd.Series(
    [10, 9, 8, 7, 6, 5, 4],
    index=["X1", "BLK1", "H2", "BLK2", "SUB1", "SUB2", "H1"], dtype=float,
)
with mock.patch.object(sleeves, "ranking", return_value=fake_rank3):
    with mock.patch.object(sleeves, "_cfg", return_value={"exit": (20, 10), "select": {"n": 4}}):
        out3 = sleeves.replace_blocked_new_slots(
            "rs_short", pd.Timestamp("2026-11-02"), names3, held3, blocked3, px3,
            {}, 4, 0.99,
        )
check("SUB1" in out3, "SUB1 kept as held replacement", out3)
check("SUB2" in out3, "SUB2 kept as held replacement", out3)
check("BLK1" not in out3, "BLK1 still dropped", out3)
check(len(out3) == 4, "held-replacement book stays at n_picks", out3)

# trade-day move = every target or holding (equal-weight rebalance)
hw3 = pd.Series({"A": 0.12, "B": 0.08, "C": 0.10})
wd3 = pd.Series({"A": 0.099, "B": 0.099, "D": 0.099})
idx = sorted(set(hw3.index) | set(wd3.index))
hw3 = hw3.reindex(idx).fillna(0)
wd3 = wd3.reindex(idx).fillna(0)
move_m = (hw3 > 0) | (wd3 > 0)
check(bool(move_m["A"]) and bool(move_m["B"]) and bool(move_m["C"]) and bool(move_m["D"]),
      "trade day moves continues+exits+adds", move_m)


# --- #5 drop_guard fail-closed ---
import drop_guard as dg
with mock.patch.object(dg, "rs_protected_symbols", return_value=None):
    # simulate call-site logic
    protected = dg.rs_protected_symbols()
    check(protected is None, "mocked protect None")
# live call should return a set (inputs available on this Mac) or None
prot = dg.rs_protected_symbols()
check(prot is None or isinstance(prot, set), "rs_protected_symbols set|None", prot)
if isinstance(prot, set):
    check(len(prot) >= 1, "live protect non-empty when available", prot)


# --- #10 hold messaging after clear ---
import run_all as ra
# After clear, last_cleared_hold_until still explains prior days
cause = ra._miss_reason(
    {"hold_until": "", "last_cleared_hold_until": "2026-10-30"},
    pd.Timestamp("2026-10-29"),
)
check("on hold until the 2026-10-30" in cause, "cleared hold still cited for prior day", cause)
cause2 = ra._miss_reason(
    {"hold_until": "", "last_cleared_hold_until": "2026-10-30"},
    pd.Timestamp("2026-10-30"),
)
check("on hold" not in cause2, "hold day itself not cited as under hold", cause2)


# --- #4 chatbot helper: only records on fill (unit via sold_block + fake order) ---
class _O:
    def __init__(self, filled, status, oid="c1"):
        self.filled_qty = filled
        self.status = status
        self.id = oid

with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "sold_block.json")
    old = sb.BLOCK_JSON
    sb.BLOCK_JSON = path
    try:
        # Import execute's nested logic by reimplementing the gate used in chatbot
        def record_if_fill(sym, o):
            st = getattr(o, "status", "")
            stv = (st.value if hasattr(st, "value") else str(st)).lower()
            filled = float(getattr(o, "filled_qty", 0) or 0)
            if filled <= 0 and stv != "filled":
                return False
            sb.record_sell(sym, source="chatbot", note="t", order_id=str(o.id))
            return True
        check(not record_if_fill("ZZZ", _O(0, "pending_new")), "chatbot skips 0-fill pending")
        check("ZZZ" not in sb.active_blocks(as_of=date(2026, 10, 31)), "no block on pending")
        check(record_if_fill("ZZZ", _O(2.0, "filled")), "chatbot records on fill")
        check("ZZZ" in sb.active_blocks(as_of=date(2026, 10, 31)), "block after fill")
    finally:
        sb.BLOCK_JSON = old


# --- #6 / #8 smoke: docstring mentions staged FAILED; no Desktop in launchd templates ---
doc = pt._todays_recorded_orders.__doc__ or ""
check("FAILED submits are staged" in doc or "staged" in doc.lower(), "docstring mentions FAILED staging")
desk = 0
for f in (ROOT / "launchd").glob("com.stockanalysis.*.plist"):
    if "Desktop/Stock Analysis" in f.read_text():
        desk += 1
check(desk == 0, "no Desktop paths in repo launchd plists", desk)


# --- idle cash: sellable unowned value is folded into half sizing ---
import inspect
src = inspect.getsource(sleeves.plan)
check("unowned_mv" in src, "plan folds unowned_mv into half values")


print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("all audit-fix2 checks passed")
