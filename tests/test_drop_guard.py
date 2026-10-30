"""Unit tests for drop_guard.py (mocked: no broker, no network LLM unless explicitly tested)."""
import json
import math
import os
import sys
import tempfile
from datetime import datetime
from types import SimpleNamespace as NS
from unittest import mock

TMP = tempfile.mkdtemp()
os.environ["STOCK_ANALYSIS_RUN_LOG"] = os.path.join(TMP, "run_log.csv")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.modules.setdefault("dotenv", NS(load_dotenv=lambda *a, **k: None))

import numpy as np
import pandas as pd
import backtest_engine as be
import drop_guard as dg

FAIL = []

def check(what, ok, got=""):
    print(("PASS " if ok else "FAIL ") + what + ("" if ok else f"  -> {got}"))
    if not ok:
        FAIL.append(what)

# config defaults
cfg = dict(dg.DEFAULT_CONFIG)
check("default drop 4.1%", abs(cfg["drop_pct"] - 0.041) < 1e-12)
check("default confidence 0.75", cfg["min_confidence"] == 0.75)
check("default max_weight 15%", cfg["max_weight"] == 0.15)
check("window ends before 2:30", tuple(cfg["window_end_ct"]) == (14, 15))

# window: Friday regular hours CT
fri = datetime(2026, 10, 30, 11, 0, tzinfo=be.CENTRAL)
ok, why = dg.in_window(fri, cfg)
check("Fri 11:00 CT in window", ok, why)
ok2, why2 = dg.in_window(datetime(2026, 10, 30, 14, 20, tzinfo=be.CENTRAL), cfg)
check("Fri 14:20 CT outside window", not ok2, why2)
ok3, _ = dg.in_window(datetime(2026, 10, 31, 11, 0, tzinfo=be.CENTRAL), cfg)  # Saturday
check("Saturday idle", not ok3)
ok4, _ = dg.in_window(datetime(2026, 10, 30, 8, 0, tzinfo=be.CENTRAL), cfg)
check("before open idle", not ok4)

# technical snapshot
idx = pd.bdate_range("2025-10-01", "2026-10-29")
rng = np.random.default_rng(0)
close = pd.Series(100 + np.cumsum(rng.normal(0, 1, len(idx))), index=idx)
daily = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1e6}, index=idx)
tech = dg.technical_snapshot(daily, None, float(close.iloc[-1]) * 0.94, float(close.iloc[-1]), cfg, qqq_ret=-0.01, sector_ret=-0.005)
check("snapshot has sma_200", "sma_200" in tech["smas"])
check("snapshot has rsi", tech["rsi_14"] is not None)
check("snapshot has macd", tech["macd"] is not None)
check("snapshot has fib", "fib_0.618" in tech["fibonacci"])
check("day_return negative when price down", tech["day_return_pct"] < 0)

# AI JSON parse
parsed = dg._parse_ai_json('{"action":"SELL","confidence":0.8,"reasons":["broke SMA50","volume spike"]}')
check("parse SELL", parsed and parsed["action"] == "SELL" and parsed["confidence"] == 0.8)
check("reject bad action", dg._parse_ai_json('{"action":"BUY","confidence":1}') is None)
check("clamp confidence", dg._parse_ai_json('{"action":"HOLD","confidence":2}')["confidence"] == 1.0)

# cid
check("sell cid shape", dg._cid("sell", "2026-10-30", "ACN") == "live-dguard-sell-20261030-ACN")
check("buy retry cid", dg._cid("buy", "2026-10-30", "MSFT", 2).endswith("-r2"))

# replacement picker respects do_not_buy / sold_today / under ma200
sig = pd.DataFrame({
    "Date": pd.Timestamp("2026-10-28"),
    "Symbol": ["AAA", "BBB", "CCC", "DDD"],
    "Strategy_Rank": [1, 2, 3, 4],
    "Strategy_Score": [1.0, 0.9, 0.8, 0.7],
    "Close": [50.0, 40.0, 30.0, 20.0],
    "ma_200": [40.0, 45.0, 25.0, 10.0],  # BBB under MA200
})
# monkeypatch latest_ranking
real = dg.latest_ranking
dg.latest_ranking = lambda: sig
import sector_mapping as sm
old_dnb = set(sm.do_not_buy)
sm.do_not_buy = set(old_dnb) | {"AAA"}
pick, why = dg.pick_replacement("ZZZ", proceeds=5000, equity=100000, held={}, sold_today={}, cfg=cfg, now=fri,
                                price_map={"AAA": 50, "BBB": 40, "CCC": 30, "DDD": 20})
# AAA blocked do_not_buy, BBB under ma200 -> CCC
check("replacement skips do_not_buy and under-MA200", pick and pick["symbol"] == "CCC", f"{pick} {why}")
pick2, why2 = dg.pick_replacement("ZZZ", 5000, 100000, held={}, sold_today={"CCC": {}}, cfg=cfg, now=fri,
                                  price_map={"CCC": 30, "DDD": 20})
check("replacement skips sold_today", pick2 and pick2["symbol"] == "DDD", f"{pick2} {why2}")
sm.do_not_buy = old_dnb
dg.latest_ranking = real

# trigger math
prev = 100.0
trig = prev * (1 - cfg["drop_pct"])
check("trigger at 95.9", abs(trig - 95.9) < 1e-9)
check("95.8 triggers", 95.8 <= trig)
check("96.0 does not", 96.0 > trig)

# dry-run path does not call trading client
calls = {"trade": 0}
dg.trading_client = lambda: (_ for _ in ()).throw(RuntimeError("no client in dry-run"))
# execute_sell_and_replace dry_run
c = {"symbol": "TEST", "shares": 10, "prev": 100.0, "trigger": 95.0, "earnings": None}
out = dg.execute_sell_and_replace(c, 90.0, 89.5, 90.5, {}, {"confidence": 0.9, "reasons": ["x"]},
                                  cfg, {"sold_today": {}}, fri, True, 185000, {"TEST": 10}, None)
check("dry-run action WOULD_SELL", out["Action"] == "WOULD_SELL", out)
check("dry-run no order id", not out.get("Sell_Order_Id"))

# plist exists
from pathlib import Path
pl = Path("launchd/com.stockanalysis.drop_guard.plist")
check("plist present", pl.exists())
import plistlib, re as _re
pdata = plistlib.loads(_re.sub(rb"<!--.*?-->", b"", pl.read_bytes(), flags=_re.S))
check("plist label", pdata["Label"] == "com.stockanalysis.drop_guard")
check("plist interval 900", pdata["StartInterval"] == 900)
check("plist --scheduled", "--scheduled" in pdata["ProgramArguments"])

print()
if FAIL:
    print(f"{len(FAIL)} FAILED")
    sys.exit(1)
print("ALL PASSED")
