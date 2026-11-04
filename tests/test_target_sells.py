"""Unit tests for target_sells helpers."""
import os
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backtest_engine as be
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


# Holiday skip: Thanksgiving week 2026 — Thanksgiving is Thu Nov 26
th = date(2026, 11, 26)
check("Thanksgiving 2026 not a session", not be.is_session(th), str(be.is_session(th)))

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


# paper_trade: STAGED/no Order_ID must not remove targets
import pandas as pd
import paper_trade as pt
with tempfile.TemporaryDirectory() as td:
    cfg_path = os.path.join(td, "targets.json")
    old_cfg = ts.CONFIG_JSON
    ts.CONFIG_JSON = cfg_path
    try:
        ts.save_config({"enabled": True, "dry_run": True, "targets": {
            "AAA": {"limit": 10.0, "qty": "full", "added": "2026-10-29"},
            "BBB": {"limit": 20.0, "qty": "full", "added": "2026-10-29"},
            "DDD": {"limit": 30.0, "qty": "full", "added": "2026-10-29"},
        }})
        staged = pd.DataFrame([
            {"Symbol": "AAA", "Side": "SELL", "Shares": 1, "Order_ID": None,
             "Status": "STAGED for morning (bad quote)"},
            {"Symbol": "BBB", "Side": "SELL", "Shares": 2, "Order_ID": "oid-1",
             "Status": "submitted (pending_new) at $10"},
            {"Symbol": "DDD", "Side": "SELL", "Shares": 2, "Order_ID": "oid-2",
             "Status": "submitted (filled) at $10"},
            {"Symbol": "CCC", "Side": "SELL", "Shares": 1, "Order_ID": None,
             "Status": "SKIPPED: not held"},
        ])
        pt._remove_targets_from_results(staged, source="unit_test")
        cfg = ts.load_config()
        check("STAGED did not remove target AAA", "AAA" in cfg["targets"], cfg["targets"])
        check("pending_new did not remove target BBB", "BBB" in cfg["targets"], cfg["targets"])
        check("filled removed target DDD", "DDD" not in cfg.get("targets", {"DDD": 1}), cfg)
    finally:
        ts.CONFIG_JSON = old_cfg

print(f"\n{len(passed)} passed, {len(failed)} failed")
if failed:
    sys.exit(1)
print("ALL PASSED")
