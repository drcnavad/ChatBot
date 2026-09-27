"""Unit tests for strategy_health.py: pure functions, synthetic data, no network, no streamlit."""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import strategy_health as sh

FAIL = []


def expect(ok, what):
    if not ok:
        FAIL.append(what)
        print("   !! FAIL:", what)


def _tmp(csv_text):
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    f.write(csv_text)
    f.close()
    return f.name


# --- reference parsing ---------------------------------------------------------------
SUMMARY = """Strategy,Period,Start,End,Total Return %,CAGR %,Sharpe,Max DD %,Trades,Win rate %,Median trade %,Median hold (sessions)
C6-U96-T20-MW30-E5,Walk-forward,2022-04-01,2026-09-24,399.9,43.45,1.3483,-30.76,1021.0,45.45,-0.74,6.0
C6-U96-T20-MW30-E5,Never-seen 2022-04 → 2024-09,2022-04-01,2024-09-16,33.2,12.42,0.5958,-29.33,,,,
QQQ buy & hold,Walk-forward,2022-04-01,2026-09-24,110.1,18.11,0.8458,-29.07,,,,
"""
p = _tmp(SUMMARY)
ref = sh.load_reference(p)
expect(abs(ref["sharpe"] - 1.3483) < 1e-9, "ref sharpe")
expect(abs(ref["sharpe_neverseen"] - 0.5958) < 1e-9, "ref never-seen sharpe")
expect(abs(ref["max_dd_pct"] - -30.76) < 1e-9, "ref max dd")
expect(abs(ref["win_rate_pct"] - 45.45) < 1e-9, "ref win rate")
expect(abs(ref["cagr_pct"] - 43.45) < 1e-9, "ref cagr")
expect(abs(ref["qqq_sharpe"] - 0.8458) < 1e-9, "ref qqq sharpe")
os.remove(p)

ref_missing = sh.load_reference("/nonexistent/file.csv")
expect(abs(ref_missing["sharpe"] - 1.3483) < 1e-9, "fallback ref used when CSV missing")


def _hist(prices, start="2026-09-28"):
    dates = pd.date_range(start, periods=len(prices), freq="B")
    lines = ["As_Of,Equity,Cash,Buying_Power,Positions"]
    lines += [f"{d:%Y-%m-%d %H:%M},{p:.2f},{p:.2f},{p:.2f},10" for d, p in zip(dates, prices)]
    return _tmp("\n".join(lines) + "\n")


def _bench(n=300, start="2026-01-01", drift=0.001):
    dates = pd.date_range(start, periods=n, freq="B")
    q = 500 * np.cumprod(1 + drift + np.random.default_rng(7).normal(0, 0.008, n))
    lines = ["Date,SPY,QQQ"] + [f"{d:%Y-%m-%d},{s:.2f},{v:.2f}" for d, s, v in zip(dates, q * 1.1, q)]
    return _tmp("\n".join(lines) + "\n")


# --- healthy: steady growth ----------------------------------------------------------
rng = np.random.default_rng(1)
prices = 100000 * np.cumprod(1 + 0.004 + rng.normal(0, 0.006, 120))
hp, bp = _hist(prices), _bench()
rep = sh.build_report(hp, bp, _tmp(SUMMARY))
expect(rep["data_state"] == "full", f"healthy data_state {rep['data_state']}")
expect(rep["level"] == "green", f"healthy level {rep['level']} score {rep['score']:.0f}")
expect(rep["stats"]["current_dd_pct"] > -15, "healthy dd small")
expect(rep["regime"] is not None and rep["regime"]["above_200dma"], "regime risk-on for drifting-up QQQ")
expect(rep["live_rebased"].iloc[0] == 100.0 and rep["live_rebased"].iloc[-1] > 100, "rebased curve")
expect(rep["qqq"] is not None and rep["qqq"].iloc[0] == 100.0, "qqq rebased")
expect(rep["expected"].iloc[0] == 100.0 and rep["expected"].iloc[-1] > 100, "expected path grows at CAGR")

# --- crash: drawdown beyond the backtest worst -> red ----------------------------------
crash = np.concatenate([100000 * np.cumprod(1 + rng.normal(0.002, 0.005, 60)),
                        np.linspace(101000, 65000, 40)])  # -35% drawdown, worse than -30.8%
hp2 = _hist(crash)
rep2 = sh.build_report(hp2, bp, _tmp(SUMMARY))
expect(rep2["drawdown"][1] == "red", f"crash dd level {rep2['drawdown'][1]}: {rep2['drawdown'][2]}")
expect(rep2["level"] == "red", f"crash composite {rep2['level']}")
expect("BEYOND" in rep2["drawdown"][2], "crash message names the breach")

# --- flat market: score degrades but not red ------------------------------------------
flat = 100000 * np.cumprod(1 + rng.normal(0, 0.004, 120))
rep3 = sh.build_report(_hist(flat), bp, _tmp(SUMMARY))
expect(rep3["level"] in ("yellow", "green"), f"flat level {rep3['level']}")

# --- warming up: < 10 sessions -> no score ---------------------------------------------
rep4 = sh.build_report(_hist(prices[:5]), bp, _tmp(SUMMARY))
expect(rep4["data_state"] == "warming_up", "warming_up state")
expect("score" not in rep4, "no score while warming up")

# --- provisional: 10-63 sessions -> score flagged ---------------------------------------
rep5 = sh.build_report(_hist(prices[:40]), bp, _tmp(SUMMARY))
expect(rep5["data_state"] == "provisional", "provisional state")
expect("score" in rep5, "score present when provisional")

# --- empty / missing files -> never raises ----------------------------------------------
rep6 = sh.build_report("/nonexistent/a.csv", "/nonexistent/b.csv", "/nonexistent/c.csv")
expect(rep6["data_state"] == "empty", "empty state on missing files")
expect(rep6["ref"]["sharpe"] > 0, "fallback ref on missing summary")

# --- duplicate As_Of rows (as seen in production) are deduped ---------------------------
dup = ("As_Of,Equity,Cash,Buying_Power,Positions\n2026-09-25 01:33,99789.89,99789.89,399159.56,0\n"
       "2026-09-25 01:33,99789.89,99789.89,399059.56,0\n2026-09-26 01:33,99800.00,99800.00,399000.00,5\n")
eq = sh.load_live_equity(_tmp(dup))
expect(len(eq) == 2, f"dup rows deduped: {len(eq)}")

# --- same calendar date, different sync times -> one row, last sync wins ---------------
sameday = ("As_Of,Equity,Cash,Buying_Power,Positions\n2026-09-25 09:15,99789.89,99789.89,399159.56,0\n"
           "2026-09-25 16:05,99850.25,99850.25,399000.00,5\n2026-09-28 09:15,99900.00,99900.00,399000.00,5\n")
eq2 = sh.load_live_equity(_tmp(sameday))
expect(len(eq2) == 2, f"same-day syncs deduped: {len(eq2)}")
expect(abs(eq2["Equity"].iloc[0] - 99850.25) < 1e-9, "last sync of the day wins")

# --- drawdown watch band: score falls smoothly from 100 to 40 ---------------------------
s_lo, lvl_lo, _ = sh.drawdown_assessment(-15.4, -30.8)   # frac = 0.5 -> boundary
s_mid, _, _ = sh.drawdown_assessment(-23.1, -30.8)       # frac = 0.75
s_hi, lvl_hi, _ = sh.drawdown_assessment(-30.8, -30.8)   # frac = 1.0 -> boundary
expect(abs(s_lo - 100.0) < 1e-9, f"watch band starts at 100 (got {s_lo})")
expect(abs(s_hi - 40.0) < 1e-9, f"watch band ends at 40 (got {s_hi})")
expect(s_lo > s_mid > s_hi, f"watch band decreases smoothly: {s_lo:.1f} {s_mid:.1f} {s_hi:.1f}")
expect(lvl_lo == "yellow" and lvl_hi == "yellow", "watch band stays yellow")

# --- dollar fields ----------------------------------------------------------------------
st_h = sh.equity_stats(sh.load_live_equity(hp))
expect(abs(st_h["pnl_dollars"] - (st_h["equity_now"] - st_h["equity_start"])) < 1e-6, "pnl dollars = now - start")
expect(st_h["dd_dollars"] <= 0, "dd dollars never positive")
expect(abs(st_h["dd_dollars"] - (st_h["equity_now"] - st_h["peak_equity"])) < 1e-6, "dd dollars = now - peak")

# --- Sharpe annualization follows the actual cadence ------------------------------------
ann = float(np.sqrt(365.25 / 1.4))  # ~daily trading-day syncs
s_ann = sh._sharpe([0.001, -0.0005, 0.002, -0.001, 0.0015], ann)
s_252 = sh._sharpe([0.001, -0.0005, 0.002, -0.001, 0.0015], float(np.sqrt(252)))
expect(abs(s_ann / s_252 - ann / np.sqrt(252)) < 1e-9, "annualization factor applied")

# --- regime: QQQ below its 200DMA --------------------------------------------------------
down = _bench(n=300, drift=-0.002)
reg = sh.regime_status(sh.load_benchmark(down))
expect(reg is not None and not reg["above_200dma"], "risk-off regime detected")
expect("half exposure" in reg["text"], "regime text explains the exposure rule")

# --- review triggers are documented ------------------------------------------------------
expect(len(sh.REVIEW_TRIGGERS) >= 4 and any("1.25" in t for t in sh.REVIEW_TRIGGERS), "review triggers list")

for f in (hp, bp, hp2):
    os.remove(f)

print("\nSTRATEGY HEALTH TESTS OK" if not FAIL else f"\nFAILURES ({len(FAIL)}): {FAIL}")
sys.exit(1 if FAIL else 0)
