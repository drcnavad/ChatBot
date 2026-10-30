"""Unit tests for sold_block (5 business days) and target_sells helpers."""
import os
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backtest_engine as be
import sold_block as sb
import target_sells as ts

CT = ZoneInfo("America/Chicago")
passed = []
failed = []

def check(name, cond, detail=""):
    if cond:
        passed.append(name)
        print(f"PASS {name}")
    else:
        failed.append(name)
        print(f"FAIL {name} {detail}")


# --- sold_block business days (skip weekend) ---
# Tue Oct 27 2026 + 5 sessions = Mon Nov 2? Let's compute carefully.
# Oct 27 Tue -> 28 Wed, 29 Thu, 30 Fri, (31 Sat 1 Sun skip), 2 Mon, 3 Tue
# So Tue + 5 business days = next Tuesday Nov 3? 
# business_days_ahead counts STRICTLY after start:
# after Oct 27: 28,29,30, Nov 2, Nov 3 → through = Nov 3
d = date(2026, 10, 27)  # Tuesday
through = sb.business_days_ahead(d, 5)
check("Tue+5 sessions = next Tue Nov 3", through == date(2026, 11, 3), str(through))

# Holiday skip: Thanksgiving week 2026 — Thanksgiving is Thu Nov 26
# Mon Nov 23 + 3 sessions: 24 Tue, 25 Wed, (26 Thu holiday?), 27 Fri → need is_session
# Just check a known holiday if be knows it
th = date(2026, 11, 26)
check("Thanksgiving 2026 not a session", not be.is_session(th), str(be.is_session(th)))

# --- sold_block persist ---
with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "sold_block.json")
    old = sb.BLOCK_JSON
    sb.BLOCK_JSON = path
    try:
        e = sb.record_sell("CRL", sold_on=date(2026, 10, 27), source="test", note="unit", now=datetime(2026, 10, 27, 12, tzinfo=CT))
        check("record through Nov 3", e["through"] == "2026-11-03", e)
        active = sb.active_blocks(as_of=date(2026, 10, 30))
        check("active on Thu same week", "CRL" in active, active)
        check("is_blocked true while active", sb.is_blocked("CRL", as_of=date(2026, 10, 28)))
        active2 = sb.active_blocks(as_of=date(2026, 11, 3))
        check("still active on through day", "CRL" in active2, active2)
        active3 = sb.active_blocks(as_of=date(2026, 11, 4))
        check("expired day after through", "CRL" not in active3, active3)
    finally:
        sb.BLOCK_JSON = old

# --- target_sells helpers ---
check("cid shape", ts.cid_for("CRL", date(2026, 10, 30)).startswith("live-tsell-20261030-CRL"))
check("is_target_cid", ts.is_target_cid("live-tsell-20261030-CRL"))
check("not foreign cid", not ts.is_target_cid("live-dguard-sell-20261030-CRL"))
check("full qty floors", ts.target_qty("CRL", {"qty": "full"}, 34.7) == 34)
check("qty capped to held", ts.target_qty("CRL", {"qty": 100}, 34) == 34)

# window: Fri 10 AM CT in, Fri 10 PM CT out, Sat out
fri = datetime(2026, 10, 30, 10, 0, tzinfo=CT)
ok, why = ts.in_window(fri, ts.DEFAULT_CONFIG)
check("Fri 10 CT in window", ok, why)
ok2, why2 = ts.in_window(datetime(2026, 10, 30, 22, 0, tzinfo=CT), ts.DEFAULT_CONFIG)
check("Fri 22 CT outside", not ok2, why2)
ok3, _ = ts.in_window(datetime(2026, 10, 31, 10, 0, tzinfo=CT), ts.DEFAULT_CONFIG)
check("Sat idle", not ok3)

# CRL not do_not_sell
import sector_mapping as sm
check("CRL not do_not_sell", "CRL" not in {str(x).upper() for x in sm.do_not_sell})

# plist
pl = Path("launchd/com.stockanalysis.target_sells.plist")
check("plist present", pl.exists())
txt = pl.read_text()
check("plist --scheduled", "--scheduled" in txt)
check("plist interval 900", "<integer>900</integer>" in txt)


# paper_trade: STAGED/no Order_ID must not record sold_block or remove targets
import pandas as pd
import paper_trade as pt
with tempfile.TemporaryDirectory() as td:
    sb_path = os.path.join(td, "sold_block.json")
    cfg_path = os.path.join(td, "targets.json")
    old_sb, old_cfg = sb.BLOCK_JSON, ts.CONFIG_JSON
    sb.BLOCK_JSON = sb_path
    ts.CONFIG_JSON = cfg_path
    try:
        ts.save_config({"enabled": True, "dry_run": True, "targets": {
            "AAA": {"limit": 10.0, "qty": "full", "added": "2026-10-29"},
            "BBB": {"limit": 20.0, "qty": "full", "added": "2026-10-29"},
        }})
        staged = pd.DataFrame([
            {"Symbol": "AAA", "Side": "SELL", "Shares": 1, "Order_ID": None,
             "Status": "STAGED for morning (bad quote)"},
            {"Symbol": "BBB", "Side": "SELL", "Shares": 2, "Order_ID": "oid-1",
             "Status": "submitted (new)"},
            {"Symbol": "CCC", "Side": "SELL", "Shares": 1, "Order_ID": None,
             "Status": "SKIPPED: not held"},
        ])
        pt._record_sold_blocks_from_results(staged, source="unit_test")
        blocks = sb.active_blocks(as_of=date(2026, 10, 29))
        check("Order_ID gate: STAGED not blocked", "AAA" not in blocks, blocks)
        check("Order_ID gate: submitted is blocked", "BBB" in blocks, blocks)
        check("Order_ID gate: SKIPPED not blocked", "CCC" not in blocks, blocks)
        cfg = ts.load_config()
        check("STAGED did not remove target AAA", "AAA" in cfg["targets"], cfg["targets"])
        check("submitted removed target BBB", "BBB" not in cfg["targets"], cfg["targets"])
    finally:
        sb.BLOCK_JSON = old_sb
        ts.CONFIG_JSON = old_cfg

print(f"\n{len(passed)} passed, {len(failed)} failed")
if failed:
    sys.exit(1)
print("ALL PASSED")
