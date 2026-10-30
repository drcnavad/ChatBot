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
check("default drop 4.0%", abs(cfg["drop_pct"] - 0.04) < 1e-12)
check("no AI min_confidence in defaults", "min_confidence" not in cfg)
check("default max_weight 15%", cfg["max_weight"] == 0.15)
check("window ends before 2:30", tuple(cfg["window_end_ct"]) == (14, 15))
check("window starts 8:30 CT", tuple(cfg["window_start_ct"]) == (8, 30))
check("default dry_run True (safe)", cfg["dry_run"] is True)

# window: Friday regular hours CT
fri = datetime(2026, 10, 30, 11, 0, tzinfo=be.CENTRAL)
ok, why = dg.in_window(fri, cfg)
check("Fri 11:00 CT in window", ok, why)
ok2, why2 = dg.in_window(datetime(2026, 10, 30, 14, 20, tzinfo=be.CENTRAL), cfg)
check("Fri 14:20 CT outside window", not ok2, why2)
ok3, _ = dg.in_window(datetime(2026, 10, 31, 11, 0, tzinfo=be.CENTRAL), cfg)  # Saturday
check("Saturday idle", not ok3)
ok4, why4 = dg.in_window(datetime(2026, 10, 30, 8, 0, tzinfo=be.CENTRAL), cfg)
check("before 8:30 idle", not ok4, why4)
ok4b, why4b = dg.in_window(datetime(2026, 10, 30, 8, 30, tzinfo=be.CENTRAL), cfg)
# 8:30 CT = 9:30 ET = regular open — should be in window on a session day
check("Fri 8:30 CT in window", ok4b, why4b)

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

# AI JSON parse removed Oct 30 2026 (auto-sell, no LLM)

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
check("trigger at 96.0", abs(trig - 96.0) < 1e-9)
check("95.9 triggers", 95.9 <= trig)
check("96.1 does not", 96.1 > trig)


# --- entry OR prev trigger (Chirag Oct 30) ---
# Fake account with avg_entry_price + bars/prev + today's fills
class _FakeAcct:
    def position_dicts(self):
        return [
            {"symbol": "OLD", "qty": 10, "avg_entry_price": 100.0, "current_price": 90, "market_value": 900},
            {"symbol": "TODAY", "qty": 5, "avg_entry_price": 50.0, "current_price": 49, "market_value": 245},
            {"symbol": "ENTRY_ONLY", "qty": 8, "avg_entry_price": 200.0, "current_price": 190, "market_value": 1520},
        ]
    def account_summary(self):
        return {"Equity": 100000}

# bars: OLD prev=110 (so prev trig=105.6, entry trig=96 → trigger=max=105.6)
# TODAY prev=60 but opened today → entry only trig=48
# ENTRY_ONLY prev=195 → entry trig=192, prev trig=187.2 → trigger=192
bars = pd.DataFrame({
    "Symbol": ["OLD", "OLD", "TODAY", "TODAY", "ENTRY_ONLY", "ENTRY_ONLY", "QQQ", "QQQ"],
    "Date": pd.to_datetime(
        ["2026-10-28", "2026-10-29"] * 4
    ),
    "Open": [1]*8, "High": [1]*8, "Low": [1]*8,
    "Close": [108, 110, 58, 60, 194, 195, 400, 401],
    "Volume": [1e6]*8,
})
real_daily = dg.daily_bars
real_fills = dg._todays_buy_fills
real_earn = be.load_earnings
dg.daily_bars = lambda syms, now, lookback=260: bars
dg._todays_buy_fills = lambda now=None, client=None: (
    {"TODAY": 50.0}, {"TODAY": 5.0}
)
be.load_earnings = lambda *a, **k: pd.DataFrame(columns=["Symbol","Report_Date","EPS","Estimate"])
rows, held, _, prev, _ = dg.held_candidates(fri, cfg, acct=_FakeAcct(),
    earnings=pd.DataFrame(columns=["Symbol","Report_Date","EPS","Estimate"]))
by = {r["symbol"]: r for r in rows}
check("held_candidates has OLD/TODAY/ENTRY_ONLY", set(by) == {"OLD", "TODAY", "ENTRY_ONLY"}, set(by))
check("OLD basis_kind entry_or_prev", by["OLD"]["basis_kind"] == "entry_or_prev", by["OLD"]["basis_kind"])
check("OLD trigger = max(entry,prev) thresh", abs(by["OLD"]["trigger"] - max(100*0.96, 110*0.96)) < 1e-9, by["OLD"]["trigger"])
check("TODAY opened_today", by["TODAY"]["opened_today"] is True)
check("TODAY basis_kind entry_today", by["TODAY"]["basis_kind"] == "entry_today", by["TODAY"]["basis_kind"])
check("TODAY trigger = entry only", abs(by["TODAY"]["trigger"] - 50*0.96) < 1e-9, by["TODAY"]["trigger"])
check("TODAY ignores higher prev thresh", by["TODAY"]["trigger"] < 60*0.96)
check("ENTRY_ONLY trigger uses higher of entry/prev", abs(by["ENTRY_ONLY"]["trigger"] - max(200*0.96, 195*0.96)) < 1e-9, by["ENTRY_ONLY"]["trigger"])
# mid 97 on OLD: below entry thresh 96? no; below prev thresh 105.6? yes → would trigger via prev
check("OLD mid 97 hits prev not entry", 97 <= by["OLD"]["prev_trigger"] and 97 > by["OLD"]["entry_trigger"])
# mid 49 on TODAY: below entry 48? no → no trigger even though vs prev 60 is -18%
check("TODAY mid 49 does NOT trigger (entry-only)", 49 > by["TODAY"]["trigger"])
check("TODAY mid 47 does trigger", 47 <= by["TODAY"]["trigger"])
dg.daily_bars = real_daily
dg._todays_buy_fills = real_fills
be.load_earnings = real_earn

# fractional dry-run note
c_frac = {"symbol": "FRAC", "shares": 10.25, "prev": 100.0, "entry": 100.0, "basis": 100.0,
          "basis_kind": "entry_or_prev", "trigger": 96.0, "opened_today": False, "earnings": None}
out_f = dg.execute_sell_and_replace(c_frac, 90.0, 89.5, 90.5,
                                    cfg, {"sold_today": {}}, fri, True, 185000, {"FRAC": 10.25}, None)
check("frac dry-run WOULD_SELL", out_f["Action"] == "WOULD_SELL", out_f)
check("frac dry-run mentions market", "fractional DAY market" in str(out_f.get("Note_extra","")), out_f.get("Note_extra"))
check("_submit_market exists", callable(getattr(dg, "_submit_market", None)))

# replacements already down are NOT skipped by pick_replacement (Chirag dropped item 2)
dg.latest_ranking = lambda: sig
sm.do_not_buy = set(old_dnb) | {"AAA"}  # AAA blocked; BBB under MA200
# CCC Close=30 ma200=25; live 27 is above MA200 but already down ~10% vs Close — still eligible
pick_down, why_down = dg.pick_replacement("ZZZ", 5000, 100000, held={}, sold_today={}, cfg=cfg, now=fri,
                                          price_map={"AAA": 50, "BBB": 40, "CCC": 27, "DDD": 20})
check("down replacement still eligible", pick_down and pick_down["symbol"] == "CCC", f"{pick_down} {why_down}")
sm.do_not_buy = old_dnb
dg.latest_ranking = real

# dry-run path does not call trading client
calls = {"trade": 0}
dg.trading_client = lambda: (_ for _ in ()).throw(RuntimeError("no client in dry-run"))
# execute_sell_and_replace dry_run
c = {"symbol": "TEST", "shares": 10, "prev": 100.0, "entry": 100.0, "basis": 100.0,
     "basis_kind": "entry_or_prev", "trigger": 95.0, "opened_today": False, "earnings": None}
out = dg.execute_sell_and_replace(c, 90.0, 89.5, 90.5,
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
