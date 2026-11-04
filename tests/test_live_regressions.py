"""Live-trading regressions from the Oct 31, 2026 audits: sold_block only on fills (+ order-matched clear), full-exit
flag, blocked-pick replacement in sleeves, evening-retry deferral, ledger v4, missed-decision wording. No broker orders."""
import json
import os
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import paper_trade as pt  # noqa: E402
import run_all as ra  # noqa: E402
import sleeves  # noqa: E402
import sold_block as sb  # noqa: E402

CT = ZoneInfo("America/Chicago")
FAIL = []

def check(cond, name, detail=""):
    print(f"{'PASS' if cond else 'FAIL'} {name}" + ("" if cond else f" {detail}"))
    if not cond:
        FAIL.append(name)

def temp_blocks():
    td = tempfile.mkdtemp()
    sb.BLOCK_JSON = os.path.join(td, "sold_block.json")

OLD_BLOCK = sb.BLOCK_JSON
try:
    # --- sold_block: only real fills block; clear needs the matching order id ---
    temp_blocks()
    pt._record_sold_blocks_from_results(pd.DataFrame([
        {"Symbol": "AAA", "Side": "SELL", "Shares": 1, "Order_ID": "1", "Status": "submitted (pending_new) at $10"},
        {"Symbol": "BBB", "Side": "SELL", "Shares": 1, "Order_ID": "2", "Status": "submitted (filled) at $10"},
        {"Symbol": "CCC", "Side": "SELL", "Shares": 1, "Order_ID": "3", "Status": "FILLED (1/1)"},
        {"Symbol": "DDD", "Side": "SELL", "Shares": 1, "Order_ID": "4",
         "Status": "COMPLETED via limit $10 (sent now, nothing was sent before)"},
    ]), source="unit")
    active = sb.active_blocks(as_of=date(2026, 10, 31))
    check("AAA" not in active and "DDD" not in active, "pending_new / COMPLETED-on-submit do not block", active)
    check("BBB" in active and "CCC" in active, "filled statuses block", active)
    temp_blocks()
    sb.record_sell("EEE", sold_on=date(2026, 10, 30), source="fill_check", note="real fill", order_id="oid-real")
    check(not sb.clear_block("EEE", reason="0-fill other", order_id="oid-other")
          and "EEE" in sb.active_blocks(as_of=date(2026, 10, 31)), "clear_block refuses a mismatched order_id")
    check(sb.clear_block("EEE", reason="0-fill same", order_id="oid-real")
          and "EEE" not in sb.active_blocks(as_of=date(2026, 10, 31)), "clear_block accepts the matching order_id")
    sb.record_sell("FFF", sold_on=date(2026, 10, 30), source="x", order_id="1")
    check(sb.clear_block("FFF", reason="admin"), "unconditional clear still works")
finally:
    sb.BLOCK_JSON = OLD_BLOCK

# --- full exit flag ---
_, is_exit, reason = pt._plan_qty("EEE", "SELL", 10, {"EEE": 10.0})
check(is_exit is True and reason is None, "full exit is_exit True")
check(pt._plan_qty("EEE", "SELL", 3, {"EEE": 10.0})[1] is False, "trim is_exit False")

# --- sleeves.replace_blocked_new_slots: blocked new pick replaced, equal weights, held stand-ins kept ---
with mock.patch.object(sleeves, "ranking",
                       return_value=pd.Series(range(5, 0, -1), index=["REPL", "ZZZ", "HOLD1", "HOLD2", "NEW1"], dtype=float)):
    out = sleeves.replace_blocked_new_slots("rs_short", pd.Timestamp("2026-10-30"), ["HOLD1", "HOLD2", "NEW1", "BAD"],
                                            {"HOLD1", "HOLD2"}, {"BAD": "sold-block"},
                                            {s: 10.0 for s in ["HOLD1", "HOLD2", "NEW1", "BAD", "REPL"]},
                                            {"stretched": pd.DataFrame()}, 4, 0.99)
check("BAD" not in out and "REPL" in out, "replace_blocked swaps the blocked new pick", out)
check(len(out) == 4 and all(abs(v - 0.99 / 4) < 1e-12 for v in out.values()), "replace_blocked equal weight", out)
with mock.patch.object(sleeves, "ranking", return_value=pd.Series(
        [10, 9, 8, 7, 6, 5, 4], index=["X1", "BLK1", "H2", "BLK2", "SUB1", "SUB2", "H1"], dtype=float)), \
        mock.patch.object(sleeves, "_cfg", return_value={"exit": (20, 10), "select": {"n": 4}}):
    out3 = sleeves.replace_blocked_new_slots("rs_short", pd.Timestamp("2026-11-02"), ["H1", "BLK1", "H2", "BLK2"],
                                             {"H1", "H2", "SUB1", "SUB2"}, {"BLK1": "sold-block", "BLK2": "sold-block"},
                                             {s: 10.0 for s in ["H1", "BLK1", "H2", "BLK2", "SUB1", "SUB2", "X1"]},
                                             {}, 4, 0.99)
check({"SUB1", "SUB2"} <= set(out3) and "BLK1" not in out3 and len(out3) == 4,
      "held replacements stay, blocked dropped, book stays at n_picks", out3)

# --- evening retry deferred after a failed session attempt ---
t = lambda s: datetime.fromisoformat(s).replace(tzinfo=CT)
fail_session = {"decision_attempts": {"decision": "2026-10-30", "n": 1, "at": "2026-10-30T14:30:15-05:00"}}
fail_evening = {"decision_attempts": {"decision": "2026-10-30", "n": 1, "at": "2026-10-30T15:15:00-05:00"}}
check(ra.decision_gate(t("2026-10-30 16:00:00"), fail_session, scheduled=True)[1] == "wait", "scheduled retry after a session fail waits")
check(ra.decision_gate(t("2026-10-30 16:20:00"), fail_evening, scheduled=True)[1] == "evening", "a prior evening fail may retry evening")
check(ra.decision_gate(t("2026-10-30 16:00:00"), fail_session, scheduled=False)[1] == "evening", "manual --trade still evening")
check(ra.decision_gate(t("2026-10-30 14:35:00"), {}, scheduled=True)[1] == "session", "first attempt in session is session")

# --- ledger: only v4 loads ---
with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "led.json")
    with mock.patch.dict(os.environ, {"STOCK_ANALYSIS_SLEEVE_LEDGER": path}):
        import importlib
        importlib.reload(sleeves)
        json.dump({"version": 2, "day": "2026-10-23", "halves": {"lowvol": {}, "sector": {}}}, open(path, "w"))
        check(sleeves.load_ledger() is None, "v2 ledger ignored")
        json.dump({"version": 4, "day": "2026-10-30", "halves": {"rs_short": {"shares": {"ACN": 1}, "cash": 0, "tags": ["ACN"],
                                                                             "rebalanced": True, "value": 1}}}, open(path, "w"))
        led = sleeves.load_ledger()
        check(led is not None and led["version"] == 4, "v4 ledger loads", led)
    importlib.reload(sleeves)

# --- missed-decision wording ---
D = pd.Timestamp("2026-10-29")
c = ra._miss_reason({"hold_until": "2026-10-30"}, D)
check("hold_until" in c and "asleep" not in c, "miss reason names hold_until", c)
check("asleep" not in ra._miss_reason({"hold_until": ""}, D), "miss reason without hold does not say asleep")
check("on hold until the 2026-10-30" in ra._miss_reason({"hold_until": "", "last_cleared_hold_until": "2026-10-30"}, D),
      "cleared hold still cited for a prior day")
check("on hold" not in ra._miss_reason({"hold_until": "", "last_cleared_hold_until": "2026-10-30"}, pd.Timestamp("2026-10-30")),
      "the hold day itself is not cited as under hold")

# --- launchd templates carry no old Desktop paths ---
check(not any("Desktop/Stock Analysis" in f.read_text() for f in (ROOT / "launchd").glob("com.stockanalysis.*.plist")),
      "no Desktop paths in launchd plists")

print(f"\n{len(FAIL)} failures" if FAIL else "\nALL PASSED")
sys.exit(1 if FAIL else 0)
