"""Strategy health monitor: is the live strategy behaving like its backtest said it would?

The backtest (Reports/backtest_summary.csv, live config C6-U91-T20-MW30-E5, live 99% weights, walk-forward
2022-04-01 -> 2026-09-24; rerun 2026-10-01 for the 91-stock universe) says:
    Sharpe 1.42 | never-seen Sharpe 0.76 | Max drawdown -26.9% | Win rate 56.3% | CAGR 45.3%
This module compares the LIVE account against those reference stats and answers:
"normal pain, or is the edge gone?"

Live inputs (all local CSVs written by the pipeline; everything is fail-soft):
    Reports/live_account_history.csv   As_Of,Equity,Cash,Buying_Power,Positions,Net_Deposits (one row per sync;
                                        deduped to one row per calendar date, last sync wins)
    Reports/benchmark_prices.csv        Date,SPY,QQQ,... (QQQ leg + regime check)

Deposits and withdrawals are NOT profit or loss: returns are time-weighted (each step's
return = (equity - net deposits made in between) / previous equity - 1), and P&L dollars =
equity change minus net deposits since the first sync.
Dollar values ride along everywhere: equity now, P&L since the first sync, and the
current drawdown in dollars — the dashboard shows them next to the percentages.
The Sharpe annualization follows the actual sync cadence (sqrt(365.25 / avg days per
observation)), so it stays comparable to the backtest's sqrt(252) whether the account
syncs daily or a few times a week.

Scoring (each component 0-100, documented bands below):
    trend       live CAGR vs backtest CAGR
    drawdown    current drawdown vs backtest max drawdown
    consistency rolling 63-session Sharpe vs backtest Sharpe bands
Composite = 0.35*trend + 0.35*drawdown + 0.30*consistency
    >= 70 green  "healthy" | 40-70 yellow "under stress" | < 40 red "edge deteriorating"

Data sufficiency: < 10 sessions -> "warming_up" (no score, shows reference + what is tracked);
10-63 sessions -> "provisional" (score shown, flagged); >= 63 -> full.

No streamlit, no network, no broker calls. The dashboard renders what this returns.
"""

import os

import numpy as np
import pandas as pd

ROLL_WINDOW = 63          # sessions for the rolling Sharpe
WARMUP_MIN = 10           # sessions before any score is shown
FULL_MIN = 63             # sessions before the score is out of "provisional"
LIVE_STRATEGY = "C6-U91-T20-MW30-E5"   # == backtest_engine.WINNER["tag"] (checked in tests/test_strategy_health.py)

# --- reference fallbacks (from backtest_summary.csv walk-forward row, 2026-10-01: C6-U91, live 99% weights) ---
# Used only if the CSV is missing; the CSV is the authority when present.
_REF_FALLBACK = {"sharpe": 1.4178, "sharpe_neverseen": 0.7637, "max_dd_pct": -26.9070,
                 "win_rate_pct": 56.3043, "cagr_pct": 45.2852, "qqq_sharpe": 0.8458}


def load_reference(summary_csv):
    """Reference stats for the live strategy + QQQ, from backtest_summary.csv.

    Returns dict(sharpe, sharpe_neverseen, max_dd_pct, win_rate_pct, cagr_pct, qqq_sharpe)
    or the documented fallback if the file/rows are missing.
    """
    ref = dict(_REF_FALLBACK)
    try:
        df = pd.read_csv(summary_csv)
    except (OSError, pd.errors.ParserError):
        return ref
    live = df[(df["Strategy"] == LIVE_STRATEGY) & (df["Period"] == "Walk-forward")]
    if not live.empty:
        r = live.iloc[0]
        for k, col in (("sharpe", "Sharpe"), ("max_dd_pct", "Max DD %"),
                       ("win_rate_pct", "Win rate %"), ("cagr_pct", "CAGR %")):
            v = r.get(col)
            if pd.notna(v):
                ref[k] = float(v)
    ns = df[(df["Strategy"] == LIVE_STRATEGY) & df["Period"].astype(str).str.startswith("Never-seen")]
    if not ns.empty and pd.notna(ns.iloc[0].get("Sharpe")):
        ref["sharpe_neverseen"] = float(ns.iloc[0]["Sharpe"])
    qqq = df[(df["Strategy"] == "QQQ buy & hold") & (df["Period"] == "Walk-forward")]
    if not qqq.empty and pd.notna(qqq.iloc[0].get("Sharpe")):
        ref["qqq_sharpe"] = float(qqq.iloc[0]["Sharpe"])
    return ref


def load_live_equity(history_csv):
    """Live equity curve from live_account_history.csv -> DataFrame(Date, Equity, Net_Deposits).

    One row per calendar date (the last sync of the day wins): the pipeline can sync
    twice on the same date (evening trade + morning fill-check) and those must not
    count as two sessions.
    """
    try:
        df = pd.read_csv(history_csv)
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame(columns=["Date", "Equity", "Net_Deposits"])
    if "Equity" not in df.columns:
        return pd.DataFrame(columns=["Date", "Equity", "Net_Deposits"])
    date_col = "As_Of" if "As_Of" in df.columns else df.columns[0]
    out = pd.DataFrame({"Date": pd.to_datetime(df[date_col], errors="coerce"),
                        "Equity": pd.to_numeric(df["Equity"], errors="coerce"),
                        "Net_Deposits": pd.to_numeric(df.get("Net_Deposits", 0.0), errors="coerce")})
    out = out.dropna().sort_values("Date")
    out["_day"] = out["Date"].dt.date
    out = out.drop_duplicates("_day", keep="last").drop(columns="_day").reset_index(drop=True)
    return out[out["Equity"] > 0].reset_index(drop=True)


def load_benchmark(bench_csv):
    """QQQ closes from benchmark_prices.csv -> DataFrame(Date, QQQ), sorted."""
    try:
        df = pd.read_csv(bench_csv, usecols=["Date", "QQQ"], parse_dates=["Date"])
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame(columns=["Date", "QQQ"])
    return df.dropna().sort_values("Date").reset_index(drop=True)


def _sharpe(returns, annualization):
    """Annualized Sharpe of a per-observation return series; NaN when undefined."""
    r = pd.Series(returns).dropna()
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * annualization)


def growth_index(eq):
    """Deposit-neutral growth of 1.0 over the live curve (time-weighted: deposits/withdrawals are not returns)."""
    e = eq["Equity"].to_numpy(float)
    r = (e[1:] - np.diff(eq["Net_Deposits"].to_numpy(float))) / e[:-1] - 1
    return np.concatenate([[1.0], np.cumprod(1 + r)])


def equity_stats(eq):
    """Core stats of a live equity curve. Returns dict; empty-ish when < 2 points.

    The Sharpe annualization is derived from the actual sync cadence
    (sqrt(365.25 / avg calendar days per observation) ~= sqrt(252) for roughly
    daily trading-day syncs), so a 3x/week sync does not inflate the number.
    Dollar fields: equity_start/now, pnl_dollars (net of deposits), peak_equity (the peak restated in
    today's dollars), dd_dollars.
    """
    out = {"n": int(len(eq)), "total_return_pct": float("nan"), "cagr_pct": float("nan"),
           "current_dd_pct": 0.0, "max_dd_pct": 0.0, "sharpe_full": float("nan"),
           "sharpe_63": float("nan"), "sharpe_provisional": True,
           "equity_start": float("nan"), "equity_now": float("nan"), "pnl_dollars": float("nan"),
           "peak_equity": float("nan"), "dd_dollars": 0.0}
    if len(eq) < 2:
        return out
    e, deposits, g = eq["Equity"].to_numpy(float), eq["Net_Deposits"].to_numpy(float), growth_index(eq)
    days = max((eq["Date"].iloc[-1] - eq["Date"].iloc[0]).days, 1)
    avg_gap = days / (len(eq) - 1)
    ann = float(np.sqrt(365.25 / max(avg_gap, 1.0)))
    out["total_return_pct"] = float(g[-1] - 1) * 100
    out["cagr_pct"] = float(g[-1] ** (365.25 / days) - 1) * 100
    out["equity_start"] = float(e[0])
    out["equity_now"] = float(e[-1])
    out["pnl_dollars"] = float(e[-1] - e[0] - (deposits[-1] - deposits[0]))
    peak = np.maximum.accumulate(g)
    out["peak_equity"] = float(e[-1] * peak[-1] / g[-1])
    out["dd_dollars"] = float(e[-1] - out["peak_equity"])
    dd = (g - peak) / peak * 100
    out["current_dd_pct"] = float(dd[-1])
    out["max_dd_pct"] = float(dd.min())
    r = np.diff(g) / g[:-1]
    out["sharpe_full"] = _sharpe(r, ann)
    out["sharpe_63"] = _sharpe(r[-(ROLL_WINDOW - 1):], ann)
    out["sharpe_provisional"] = len(r) < ROLL_WINDOW - 1
    return out


def _band_score(x, lo, hi):
    """Linear 0-100 map: x <= lo -> 0, x >= hi -> 100."""
    if hi <= lo:
        return 50.0
    return float(np.clip((x - lo) / (hi - lo), 0, 1) * 100)


def trend_assessment(cagr_pct, ref_cagr_pct):
    """Live CAGR vs backtest CAGR. 100 at/above reference, 50 at zero, 0 at -50% CAGR."""
    if pd.isna(cagr_pct):
        return 0.0, "grey", "no live history yet"
    if cagr_pct >= ref_cagr_pct:
        s, lvl, txt = 100.0, "green", f"CAGR {cagr_pct:.1f}% — at/above the backtest's {ref_cagr_pct:.1f}%"
    elif cagr_pct >= 0:
        s = 50 + 50 * cagr_pct / ref_cagr_pct
        lvl, txt = "yellow", f"CAGR {cagr_pct:.1f}% — positive but below the backtest's {ref_cagr_pct:.1f}%"
    else:
        s = max(0.0, 50 * (1 + cagr_pct / 50))
        lvl, txt = "red", f"CAGR {cagr_pct:.1f}% — losing money; backtest expected +{ref_cagr_pct:.1f}%"
    return s, lvl, txt


def drawdown_assessment(current_dd_pct, ref_max_dd_pct):
    """Current drawdown as a fraction of the backtest's worst drawdown.

    OK below half the backtest max, WATCH up to the max, ALERT beyond anything seen.
    """
    m = abs(ref_max_dd_pct)
    if m == 0:
        return 50.0, "grey", "no backtest drawdown reference"
    frac = abs(current_dd_pct) / m
    if frac < 0.5:
        return 100.0, "green", f"drawdown {current_dd_pct:.1f}% — inside normal range (backtest worst {ref_max_dd_pct:.1f}%)"
    if frac <= 1.0:
        s = 100 - 0.6 * _band_score(frac, 0.5, 1.0)  # 100 at half the worst -> 40 at the worst
        return s, "yellow", (f"drawdown {current_dd_pct:.1f}% — {frac:.0%} of the backtest worst "
                             f"({ref_max_dd_pct:.1f}%); stress, not yet breakage")
    s = max(0.0, 40 - 40 * min(frac - 1.0, 1.0))
    return s, "red", (f"drawdown {current_dd_pct:.1f}% — BEYOND the backtest worst "
                      f"({ref_max_dd_pct:.1f}%): review the strategy")


def sharpe_assessment(sharpe, ref, provisional):
    """Rolling Sharpe vs backtest bands: 1.35 walk-forward, 0.60 never-seen floor."""
    tag = " (provisional — < 63 sessions)" if provisional else ""
    if pd.isna(sharpe):
        return 0.0, "grey", "Sharpe undefined on flat history" + tag
    hi, flo = ref["sharpe"], ref["sharpe_neverseen"]
    if sharpe >= 1.0:
        return 100.0, "green", f"rolling Sharpe {sharpe:.2f}{tag} — edge intact (backtest {hi:.2f})"
    if sharpe >= 0.3:
        return _band_score(sharpe, 0.3, 1.0) * 0.4 + 60, "yellow", \
            f"rolling Sharpe {sharpe:.2f}{tag} — below the backtest's {hi:.2f}, above the never-seen floor {flo:.2f}"
    if sharpe >= 0:
        return _band_score(sharpe, 0, 0.3) * 0.3 + 30, "yellow", \
            f"rolling Sharpe {sharpe:.2f}{tag} — near zero; the edge is not showing up"
    return max(0.0, 30 + sharpe * 30), "red", \
        f"rolling Sharpe {sharpe:.2f}{tag} — negative risk-adjusted return; edge deteriorating"


def regime_status(bench):
    """QQQ vs its 200-day average. The strategy halves exposure when QQQ <= 200DMA.

    Returns dict(above_200dma, distance_pct, text) or None when unavailable.
    """
    if bench is None or len(bench) < 200 or "QQQ" not in bench.columns:
        return None
    q = bench["QQQ"].to_numpy(float)
    dma = float(pd.Series(q).rolling(200).mean().iloc[-1])
    last = float(q[-1])
    if dma <= 0 or pd.isna(dma):
        return None
    dist = (last / dma - 1) * 100
    above = last > dma
    return {"above_200dma": bool(above), "distance_pct": float(dist),
            "text": (f"QQQ {dist:+.1f}% vs 200-day average — "
                     + ("risk-on regime (full exposure)" if above else
                        "RISK-OFF regime — the strategy is at half exposure by design"))}


def expected_path(dates, ref_cagr_pct):
    """Backtest-CAGR-implied equity path, rebased to 100 at the first date (for the chart)."""
    dates = pd.to_datetime(pd.Series(dates))
    if len(dates) == 0:
        return pd.Series(dtype=float)
    yrs = (dates - dates.iloc[0]).dt.total_seconds() / (365.25 * 86400)
    return pd.Series(100 * (1 + ref_cagr_pct / 100) ** yrs.to_numpy(), index=dates.dt.date)


def composite_score(trend_s, dd_s, sh_s):
    """0-100 composite + level + one-line verdict."""
    score = float(0.35 * trend_s + 0.35 * dd_s + 0.30 * sh_s)
    if score >= 70:
        return score, "green", "Healthy — live behavior matches the backtest. No action needed."
    if score >= 40:
        return score, "yellow", "Under stress — within the range the backtest saw. Watch, don't overhaul."
    return score, "red", "Edge deteriorating — review the strategy before adding capital."


REVIEW_TRIGGERS = [
    "Drawdown pushes past ~1.25x the backtest worst (about -38%) and stays there — the tab already "
    "alerts at 1.0x (-30.8%); 1.25x sustained is the review-the-strategy line.",
    "Rolling 63-day Sharpe stays negative for a full quarter — the edge is not showing up.",
    "Live CAGR trails QQQ by > 10 pts over any 6-month window — buy-and-hold is winning.",
    "Two consecutive quarters in the red zone with no risk-off regime to explain it.",
]


def build_report(history_csv, bench_csv, summary_csv):
    """Everything the dashboard tab needs. Never raises on missing/bad files."""
    ref = load_reference(summary_csv)
    eq = load_live_equity(history_csv)
    bench = load_benchmark(bench_csv)
    stats = equity_stats(eq)
    n = stats["n"]

    report = {"ref": ref, "n_sessions": n, "stats": stats, "regime": regime_status(bench),
              "review_triggers": REVIEW_TRIGGERS, "qqq": None, "expected": None, "live_rebased": None}
    if n == 0:
        report["data_state"] = "empty"
        return report
    # QQQ leg aligned to the live window (rebased to 100)
    if bench is not None and len(bench):
        b = bench[bench["Date"] >= eq["Date"].iloc[0].normalize()]
        if len(b):
            q = b["QQQ"].to_numpy(float)
            report["qqq"] = pd.Series(100 * q / q[0], index=pd.to_datetime(b["Date"]).dt.date)
    report["live_rebased"] = pd.Series(100 * growth_index(eq), index=eq["Date"].dt.date)
    report["expected"] = expected_path(eq["Date"], ref["cagr_pct"])
    if n < WARMUP_MIN:
        report["data_state"] = "warming_up"
        return report
    report["data_state"] = "provisional" if n < FULL_MIN else "full"
    t_s, t_lvl, t_txt = trend_assessment(stats["cagr_pct"], ref["cagr_pct"])
    d_s, d_lvl, d_txt = drawdown_assessment(stats["current_dd_pct"], ref["max_dd_pct"])
    s_s, s_lvl, s_txt = sharpe_assessment(stats["sharpe_63"], ref, stats["sharpe_provisional"])
    score, lvl, verdict = composite_score(t_s, d_s, s_s)
    report.update({"trend": (t_s, t_lvl, t_txt), "drawdown": (d_s, d_lvl, d_txt),
                   "sharpe": (s_s, s_lvl, s_txt),
                   "score": score, "level": lvl, "verdict": verdict})
    return report


def default_paths(root=None):
    """Resolve the three input CSV paths under the project root."""
    root = root or os.path.dirname(os.path.abspath(__file__))
    rep = os.path.join(root, "Reports")
    return (os.path.join(rep, "live_account_history.csv"), os.path.join(rep, "benchmark_prices.csv"),
            os.path.join(rep, "backtest_summary.csv"))
