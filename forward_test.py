"""Forward test from backtest_engine.FORWARD_START (Fri Oct 2, 2026), counted only from that close on, never backtested:

  1. the real Alpaca account (read-only GET requests through alpaca_paper; no orders, ever), and
  2. STRATEGIES: 35 paper-only strategies next to it (the live rules recomputed the same way + 34 others). They are
     never traded; each one is a list of target weights built from data the pipeline already saves every day
     (Reports/signal_analysis.csv, factor_history.csv, earnings_date.csv, benchmark_prices.csv) + daily volume from
     Alpaca's free market data (forward_bars(): one bar download per run, the same request the pipeline makes).

  Reports/forward_test_daily.csv           one row per trading day (re-recording a day replaces it): account equity, cash,
                                           lifetime net deposits, positions, closed picks / winners, $ traded and its cost
                                           vs the decision price (the stock's close in signal_analysis.csv on the order's
                                           As_Of day, which on a decision day is the 2:30 PM bar the strategy decided on)
  Reports/forward_strategies.csv           one row per strategy and trading day: Value (1.0 at the Oct 2 close) and Cash
  Reports/forward_strategies_holdings.csv  each strategy's holdings every trading day: Symbol, target Weight, Shares, Price
  leaderboard()                            every strategy + the account + QQQ / SPY: total return, median weekly return,
                                           max drawdown, weeks; ranked by RANK_RULE (chosen before any result)

Same accounting for every strategy: decision at that day's close (the 2:30 PM bar on decision days, as saved), trades at
that price whenever its target changes, be.COST (0.1%) per side, no stock above 20%, the live 99% invested convention
(be.live_weights), cash earns 0. A saved day is never redone: a re-run only adds the days after the last saved one (so
running twice never double-counts) and a missed day is caught up from the saved bars.

    python forward_test.py --record   # strategies + today's account row (launchd com.stockanalysis.forwardtest, 4:15 PM CT)
    python forward_test.py            # update the strategies (bars only, no account request) and print the leaderboard
"""
import argparse
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import backtest_engine as be  # noqa: E402

REPORTS = os.path.join(ROOT, "Reports")
DAILY_CSV = os.path.join(REPORTS, "forward_test_daily.csv")
STRATEGIES_CSV = os.path.join(REPORTS, "forward_strategies.csv")
HOLDINGS_CSV = os.path.join(REPORTS, "forward_strategies_holdings.csv")
BENCH_CSV = os.path.join(REPORTS, "benchmark_prices.csv")
ORDERS_CSV = os.path.join(REPORTS, "live_orders_log.csv")
SIGNAL_CSV = os.path.join(REPORTS, "signal_analysis.csv")
FACTOR_CSV = os.path.join(REPORTS, "factor_history.csv")
CT = ZoneInfo("America/Chicago")
DAILY_COLS = ["Date", "Time_CT", "Equity", "Cash", "Net_Deposits", "Positions", "Closed_Picks", "Winning_Picks",
              "Traded_USD", "Cost_USD"]
SIG_COLS = ["Date", "Symbol", "Close", "ma_50", "ma_10", "ma_30", "ma_100", "ma_200", "RSI", "macd", "MACD Signal",
            "Technical_Score", "RS_Score", "Strategy_Score", "Regime_On", "final_trade"]
MAX_WEIGHT = 0.20            # no stock above 20% (the live max_weight rule); the extra stays cash
WEEKS_TO_WIN = 12
LIVE = "Live rules (C6)"
LIVE_MARK = " ◀ LIVE"          # the leaderboard row of the live rules
ACCOUNT = "Your Alpaca account (real)"
RANK_RULE = ("Ranked by median weekly return (Friday to Friday); a tie goes to the smaller max drawdown. Chosen on Oct 2, "
             f"2026, before any result. After {WEEKS_TO_WIN} full weeks the top strategy is the winner only if its total "
             f"return also beats {LIVE} and QQQ; otherwise the live rules stay. Nothing switches on its own: the live "
             "rules change only when you ask.")


# ---------------------------------------------------------------------------------------------------- the strategies
# One dict per strategy. Ranked strategies go through the engine's own selection (be.winner_targets); their defaults = the
# live rules: score "live" (0.5 x technical + 0.5 x relative strength), relative strength tiebreak, Friday rebalance +
# Mon/Wed swaps and exits ("mwf"), top 10 from ranks 1-20, max 4 per sector (relaxed to fill the slots), 1/volatility
# weights, half size when QQQ is below its 200-day average, no buys 5 days before earnings. "select" changes rank_targets
# keys (n, sector_cap, vol_sizing, regime, ...), "earnings" None drops the earnings skip, "calendar" is "mwf" / "weekly"
# (Friday only) / "monthly" (last session of the month). "weights" = a rule that is not a ranking (its own weights below);
# "reweight" = the same picks with other weights; "stop" = the live rules + an ATR stop around earnings (STOPS);
# "spare_dips" = the live rules + ATR dip buys with the cash they leave unused.
PLAIN = {"max_pick_rank": None, "cap_soft": False}       # walk every rank, strict max 4 per sector
EQ = {**PLAIN, "vol_sizing": False}                      # ... and an equal weight per stock
SIMPLE = dict(select=PLAIN, earnings=None)                # a plain top-10 strategy: no rank-20 limit, no earnings skip
STRATEGIES = [
    dict(name=LIVE, rule="What the bot trades: 50% technical + 50% relative strength, top 10, Fri rebalance + Mon/Wed swaps."),
    dict(name="Live without rank-20 limit / earnings skip", select=PLAIN, earnings=None,
         rule="The live rules, but picks may come from any rank and earnings never block a buy."),
    dict(name="Technical score only", score="tech", rule="Live rules ranking on the technical score alone."),
    dict(name="Relative strength only", score="rs",
         rule="Live rules ranking on relative strength (vs sector ETF and SPY) alone."),
    dict(name="Quant Score", score="quant", tiebreak=None,
         rule="Live rules ranking on the copy folder's Quant Score (trend, MAs, VWAP, RSI, MACD, Bollinger); needs a score above 50."),
    dict(name="Quant Score top 5", score="quant", tiebreak=None, select={"n": 5},
         rule="The copy folder's main Quant leg: the Quant Score rules with 5 stocks."),
    dict(name="Quant + relative strength", score="quant_rs",
         rule="Live rules ranking on a 50/50 blend of Quant Score rank and relative strength rank."),
    dict(name="Live + news sentiment", score="sentiment",
         rule="Live rules, ranking 80% on the live score and 20% on the latest news sentiment."),
    dict(name="Company fundamentals", score="fundamental", calendar="monthly", select=EQ, earnings=None,
         rule="Top 10 by the company-report score (Fundamental_Weight), equal weight, monthly."),
    dict(name="12-1 month momentum", score="mom_12_1", calendar="monthly", **SIMPLE,
         rule="Top 10 by the 12-month return skipping the last month, monthly, 1/volatility weights."),
    dict(name="6-month momentum", score="mom_6", calendar="weekly", **SIMPLE,
         rule="Top 10 by the 6-month return, every Friday, 1/volatility weights."),
    dict(name="Low volatility in uptrend", score="low_vol", calendar="monthly", select=EQ, earnings=None,
         rule="The 10 calmest stocks (63-day volatility) above their 200-day average, equal weight, monthly."),
    dict(name="Buy the dip", weights="dip",
         rule="The copy folder's dip test: buy a stock 8% below its 50-day average, sell when it is back at the average or "
              "after 30 days; 10% each, at most 10, deepest first, checked daily."),
    dict(name="Legacy BUY/SELL signals", weights="legacy",
         rule="The Legacy folder's rule: 60% technical + 25% fundamentals + 10% sector week + 5% news; hold from BUY (above "
              "20) to SELL (below -25), an equal slice of the money per stock, checked daily."),
    dict(name="Top 5", select={"n": 5}, rule="The live rules with 5 stocks (max 2 per sector)."),
    dict(name="Top 20", select={"n": 20}, rule="The live rules with 20 stocks (= ranks 1-20, max 8 per sector)."),
    dict(name="Equal weight", select={"vol_sizing": False}, rule="The live rules with an equal weight per stock."),
    dict(name="Friday only", calendar="weekly", rule="The live rules without the Mon/Wed swaps and exits."),
    dict(name="Monthly", calendar="monthly", rule="The live rules, rebalanced on the last trading day of each month only."),
    dict(name="No sector limit", select={"sector_cap": 1.0, "cap_soft": True},
         rule="The live rules without the max-4-per-sector rule (pure top 10)."),
    dict(name="Dual momentum", score="dual", calendar="monthly", select={**EQ, "regime": None, "regime_scale": None},
         earnings=None, rule="Top 10 by 12-month return, only stocks that are up over the year; all cash while QQQ is "
                             "below its 200-day average; equal weight, monthly."),
    dict(name="Trend following 50/200", weights="trend",
         rule="An equal slice of the money for every stock whose 50-day average is above its 200-day average; checked Fridays."),
    dict(name="Volatility-adjusted momentum", score="mom_risk", calendar="weekly", select=EQ, earnings=None,
         rule="Top 10 by 6-month return divided by 6-month volatility, equal weight, every Friday."),
    dict(name="52-week high", score="high_52", calendar="monthly", select=EQ, earnings=None,
         rule="The 10 stocks closest to their 52-week high (George & Hwang), equal weight, monthly."),
    dict(name="Short-term reversal", score="reversal", calendar="weekly", select=EQ, earnings=None,
         rule="The 10 biggest 1-week losers among stocks above their 200-day average, equal weight, every Friday."),
    dict(name="Quality + momentum", score="quality_mom", calendar="monthly", **SIMPLE,
         rule="Top 10 by 50% fundamentals rank + 50% 12-1 month momentum rank, monthly, 1/volatility weights."),
    dict(name="Earnings drift", score="drift", calendar="weekly", **SIMPLE,
         rule="Stocks that beat the market on their last earnings day (within 60 trading days), biggest jump first; top 10, "
              "every Friday. (No free EPS-surprise data, so the price jump stands in for the beat.)"),
    dict(name="Sector rotation", score="sector_rot", calendar="weekly", **SIMPLE,
         rule="The 3 sector ETFs with the best 3-month return, then the best live-score stocks in those sectors; top 10, Fridays."),
    dict(name="Breakout with volume", score="breakout", calendar="weekly", select=EQ, earnings=None,
         rule="Stocks that closed within 2% of their 55-day high on 1.3x+ normal volume (5-day vs 50-day average) in the "
              "last 20 trading days and are still above their 50-day average; top 10 by that volume jump, equal weight, Fridays."),
    dict(name="Live + pre-earnings 3.5x ATR stop", stop="pre",
         rule="The live rules + a stop for a stock kept at the Friday rebalance with earnings in the next 7 days, from that "
              "Friday through its first trading day after the report: sold in full at the close at or below its highest "
              "close since bought minus 3.5x ATR(14), or at the open if that day opens below it; the best stock the live "
              "rules would buy takes the slot."),
    dict(name="Live + post-earnings ATR stop", stop="post",
         rule="The live rules + the same stop from a holding's first trading day after earnings through 10 trading days "
              "later: sold at the close at or below its highest close since bought minus 3x ATR(14); the best stock the "
              "live rules would buy takes the slot."),
    dict(name="ATR dip buy (top 30)", weights="atr_dip",
         rule="Buy a stock ranked in the live top 30 when it closes 3x ATR(14) or more below its 20-day high close; 10% "
              "each (at most 10, best rank first), sold after 10 trading days."),
    dict(name="ATR dip buy, not after earnings", weights="atr_dip_ex_earn",
         rule="The ATR dip buy, but not within 10 trading days after the stock's earnings."),
    dict(name="Live + ATR dip buys in spare cash", spare_dips=True,
         rule="The live rules; cash they leave unused (empty slots, half size when QQQ is weak) buys the not-after-earnings "
              "ATR dips at 10% each (not within 5 days before earnings either), sold after 10 trading days or when the "
              "live rules buy that stock or need the cash."),
    dict(name="Risk parity", reweight="risk_parity",
         rule="The live picks, weighted so each stock adds the same risk (63-day volatility and correlation)."),
]
STOPS = {"pre": dict(k=3.5, arm_days=7, gap_open=True), "post": dict(k=3.0, after=10)}   # atr_stop_targets settings
FWD_BARS = os.path.join(REPORTS, "cache", "forward_bars.pkl")
FWD_BARS_START = "2026-05-01"      # ~100 sessions before the start: the 50-day volume average is full by Oct 2
_NO_SAVE = datetime(2000, 1, 1, tzinfo=be.EASTERN)   # be.keep_decision_bars(now=this) only reads Reports/decision_bars.csv


def _wide(df, col, index="Date"):
    return df.pivot(index=index, columns="Symbol", values=col).sort_index()


def _pct(w):
    return w.rank(axis=1, pct=True)


def _carry(w, days):
    """Weights that only change on `days` (carried in between)."""
    return w[days.to_numpy(bool)].reindex(w.index).ffill().fillna(0.0)


def forward_bars(fetch=True):
    """Daily bars with volume for the VWAP / volume rules: a fresh download from FWD_BARS_START (Alpaca free market data
    through be.fetch_daily_bars, the pipeline's own client; saved to Reports/cache/forward_bars.pkl and reused when a
    download fails) + the backtest bar cache (Reports/cache/bars_daily_long.pkl, read only) for the older days."""
    if fetch:
        try:
            new, _ = be.drop_partial_last_bar(be.fetch_daily_bars(be.TRADABLE, start=FWD_BARS_START))
            new.to_pickle(FWD_BARS + ".tmp")
            os.replace(FWD_BARS + ".tmp", FWD_BARS)
        except Exception as e:
            print(f"bars not refreshed ({type(e).__name__}: {e}); using the saved copy")
    return [be.apply_history_start(pd.read_pickle(p)) for p in (FWD_BARS, be.CACHE_DIR / be.LONG_CACHE) if os.path.exists(p)]


def bar_features(sources, dates, cols):
    """Close / 20-day VWAP (the copy's daily-bar VWAP: typical price x volume), the volume jump (5-day / 50-day average
    volume), ATR(14) / close and open / close, computed inside each download so splits and dividends stay consistent; the newest download wins where its
    window is full. Decision days use their 2:30 PM bar, like signal_analysis.csv."""
    out = {k: pd.DataFrame(index=dates, columns=cols, dtype=float) for k in ("vwap_ratio", "vol_jump", "atr_ratio", "open_ratio")}
    for b in sources:
        b = be.keep_decision_bars(b.sort_values(["Symbol", "Date"]).reset_index(drop=True), now=_NO_SAVE)
        w = {c: b.pivot_table(index="Date", columns="Symbol", values=c) for c in be.BAR_COLS}
        v = w["Volume"].replace(0, np.nan)
        vwap = ((w["High"] + w["Low"] + w["Close"]) / 3 * v).rolling(20).sum() / v.rolling(20).sum()
        pc = w["Close"].shift(1)                         # ATR: Wilder 14, as in be.calculate_technical_indicators
        tr = np.fmax(w["High"] - w["Low"], np.fmax((w["High"] - pc).abs(), (w["Low"] - pc).abs()))
        atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        new = {"vwap_ratio": w["Close"] / vwap, "vol_jump": v.rolling(5).mean() / v.rolling(50).mean(),
               "atr_ratio": atr / w["Close"], "open_ratio": w["Open"] / w["Close"]}
        for k in out:
            out[k] = out[k].combine_first(new[k].reindex(index=dates, columns=cols))
    return out


def quant_score(f, vwap_ratio):
    """The copy folder's Quant_Score (Stock Analysis Test Strategy/backtest_engine.quant_score), 0-100, from the saved
    signal_analysis.csv columns + the bars' VWAP: the average of price above its 5 moving averages, the moving-average
    stack, price vs the 20-day VWAP, RSI(14), the MACD histogram (point-in-time scaled), Bollinger %B (20 days) and the
    20-day slope of the 200-day average (a missing part counts as 50)."""
    c, ma = f["Close"], {n: f[f"ma_{n}"] for n in (10, 30, 50, 100, 200)}
    mid, sd = c.rolling(be.BB_WINDOW).mean(), c.rolling(be.BB_WINDOW).std()
    parts = [sum((c > m).astype(float) for m in ma.values()) / 5 * 100,
             sum((ma[a] > ma[b]).astype(float) for a, b in ((10, 30), (30, 50), (50, 100), (100, 200))) / 4 * 100,
             (50 + 2500 * (vwap_ratio - 1)).clip(0, 100),
             f["RSI"].clip(0, 100),
             (50 + (f["macd"] - f["MACD Signal"]).apply(be.pit_scale) / 2).clip(0, 100),
             ((c - (mid - 2 * sd)) / (4 * sd).replace(0, np.nan)).clip(0, 1) * 100,
             (50 + 1250 * (ma[200] / ma[200].shift(20) - 1)).clip(0, 100)]
    return sum(p.fillna(50.0) for p in parts) / len(parts)


def dip_weights(c, ma50, eligible, depth=0.08, max_hold=30, n=10):
    """test_buy_high_sell_low.ipynb as a portfolio: buy at the close 8% or more below the 50-day average, sell at the close
    back at/above it or after max_hold sessions (not bought back that day); 1/n each, at most n at once (deepest first)."""
    C, M, E = c.to_numpy(float), ma50.to_numpy(float), eligible.to_numpy(bool)
    out, held = np.zeros(C.shape), {}
    for t in range(len(C)):
        sold = set()
        for j in list(held):
            held[j] += 1
            if not C[t, j] < M[t, j] or held[j] >= max_hold:
                del held[j]
                sold.add(j)
        gap = np.nan_to_num(1 - C[t] / M[t], nan=-1.0)
        new = [j for j in np.argsort(-gap, kind="stable") if E[t, j] and gap[j] >= depth and j not in held and j not in sold]
        for j in new[:n - len(held)]:
            held[j] = 0
        out[t, list(held)] = 1 / n
    return pd.DataFrame(out, index=c.index, columns=c.columns)


def earnings_drift(c, spy, earnings, hold=60):
    """Excess return vs SPY (%) of each stock's last earnings reaction day (the report day for AM reports, else the next
    session), kept for `hold` sessions from that close; NaN otherwise."""
    out, idx = np.full(c.shape, np.nan), c.index
    pos = {s: j for j, s in enumerate(c.columns)}
    e = earnings[earnings["Symbol"].isin(pos)].sort_values("Earnings Date")
    for sym, day, when in zip(e["Symbol"], e["Earnings Date"], e.get("Time", pd.Series("", index=e.index)).astype(str)):
        k = idx.searchsorted(day, side="left" if when.strip().upper() == "AM" else "right")
        if 1 <= k < len(idx):
            j = pos[sym]
            out[k:k + hold, j] = (c.iat[k, j] / c.iat[k - 1, j] - spy.iat[k] / spy.iat[k - 1]) * 100
    return pd.DataFrame(out, index=idx, columns=c.columns)


def risk_parity(tgt, close, days, window=63):
    """Same picks and total weight, re-split for equal risk contributions (63-day covariance; cyclical coordinate descent),
    recomputed when the picks change and on `days`; 1/volatility weights stay where the covariance is incomplete."""
    r, T = close.pct_change(fill_method=None), tgt.to_numpy(float)
    out, key, cur, reb = T.copy(), None, None, days.reindex(tgt.index).fillna(False).to_numpy(bool)
    for t in range(len(T)):
        picked = np.flatnonzero(T[t] > 0)
        k = (tuple(picked), round(T[t].sum(), 8))
        if k != key or reb[t]:
            key, cur = k, T[t].copy()
            cov = r.iloc[max(0, t - window + 1):t + 1, picked].cov().to_numpy()
            if len(picked) > 1 and np.isfinite(cov).all():
                x = np.full(len(picked), 1 / len(picked))
                for _ in range(100):
                    for i in range(len(x)):
                        b = cov[i] @ x - cov[i, i] * x[i]
                        x[i] = (-b + np.sqrt(b * b + 4 * cov[i, i] / len(x))) / (2 * cov[i, i])
                cur[picked] = x / x.sum() * T[t].sum()
        out[t] = cur
    return pd.DataFrame(out, index=tgt.index, columns=tgt.columns)


def earnings_events(dates, cols, earnings):
    """[(column, report day, reaction-day row)] per report in `dates`: the reaction day is the report day for AM reports,
    else the next session (after-close reports)."""
    pos, ev = {s: j for j, s in enumerate(cols)}, []
    e = earnings[earnings["Symbol"].isin(pos)]
    for sym, day, when in zip(e["Symbol"], e["Earnings Date"], e.get("Time", pd.Series("", index=e.index)).astype(str)):
        r = dates.searchsorted(day, side="left" if when.strip().upper() == "AM" else "right")
        if r < len(dates):
            ev.append((pos[sym], day, r))
    return ev


def atr_stop_targets(inp, k=3.0, after=None, arm_days=None, gap_open=False):
    """The live rules + an ATR trailing stop around earnings. The live rules are replayed day by day with the engine's own
    pieces, exactly as be.winner_targets does (Friday selection with the real holdings, Mon/Wed swaps and exits). Stop
    window: after = from the reaction day through `after` sessions later; arm_days = a stock kept (HOLD) at a Friday
    rebalance with earnings within the next arm_days calendar days, from that Friday through its reaction day. Inside it
    a holding whose close is at or below its highest close since it was bought - k x ATR(14) is sold in full at that close
    (gap_open: on the reaction day an open already below the stop sells at the open); its weight goes to the best stock
    the live rules would buy: ranks 1-20, max 4 per sector unless none fits, no buy within 5 days of earnings, and not a
    stock stopped out in its window. Returns (raw weights, open-sale prices)."""
    W, args = be.WINNER, be.winner_rank_args(inp["regime"])
    score, elig, vol, tb = inp["scores"]["live"], inp["eligible"], inp["vol"], inp["scores"]["rs"]
    dates, cols = score.index, list(score.columns)
    arr = lambda x: x.reindex(index=dates, columns=cols).to_numpy(float)
    S, V, TB, C, A, O = arr(score), arr(vol), arr(tb), arr(inp["close"]), arr(inp["atr"]), arr(inp["open"])
    E = elig.reindex(index=dates, columns=cols).astype("boolean").fillna(False).to_numpy(bool)
    BB = arr(be.earnings_days_ahead(dates, cols, inp["earnings"], W["earnings_block_days"]))   # not NaN = not bought
    mw, n, min_score = W["midweek_swap"], W["n"], W["min_score"]
    reb = inp["weekly"].reindex(dates).fillna(False).to_numpy(bool)
    chk = be.midweek_check_days(dates, mw.get("days", ("Mon", "Wed")), inp["weekly"]).to_numpy(bool)
    name_pos, sectors = be._name_positions(cols), np.array([be.sector_mapping.symbol_sector.get(c, "Other") for c in cols])
    per_sector = max(1, int(np.floor(W["sector_cap"] * n)))
    win, react, reports = np.zeros((len(dates), len(cols)), bool), np.zeros((len(dates), len(cols)), bool), {}
    for j, day, r in earnings_events(dates, cols, inp["earnings"]):
        react[r, j] = True
        reports.setdefault(j, []).append((day, r))
        if after is not None:
            win[r:r + after + 1, j] = True
    out, opens = np.zeros((len(dates), len(cols))), np.full((len(dates), len(cols)), np.nan)
    cur, peak, armed = np.zeros(len(cols)), np.full(len(cols), np.nan), np.full(len(cols), -1)

    def ranked(t):
        ok = E[t] & ~np.isnan(S[t]) & ~np.isnan(V[t]) & (V[t] > 0) & (S[t] > min_score)
        return be.ranking_order(np.where(ok)[0], S[t], TB[t], name_pos)

    def stop_out(t, j):                                   # sold in full; not bought back until its window ends
        end = armed[j] + 1 if armed[j] >= t else (t + np.argmin(win[t:, j]) if not win[t:, j].all() else len(dates))
        BB[t:end, j], peak[j], armed[j], w = 0.0, np.nan, -1, cur[j]
        cur[j] = 0.0
        return w

    for t in range(len(dates)):
        freed = []
        if gap_open and t:                                # the reaction day opens below yesterday's stop: sold at the open
            inside = win[t] | (armed >= t)
            for j in np.where(react[t] & inside & (cur > 0) & (O[t] <= peak - k * A[t - 1]))[0]:
                opens[t, j] = O[t, j]
                freed.append(stop_out(t, j))
        if reb[t]:
            day, before = dates[[t]], cur.copy()
            cur = be.rank_targets(score.iloc[[t]], elig, vol, rebalance_days=pd.Series(True, index=day), tiebreak_w=tb,
                                  buy_block=pd.DataFrame(BB[[t]], index=day, columns=cols), start_holdings=cur.copy(),
                                  **args).iloc[0].to_numpy(float)
            freed = []                                    # the Friday selection already refilled
            if arm_days:                                  # kept at this rebalance, earnings within arm_days: armed
                for j in np.where((before > 0) & (cur > 0))[0]:
                    for day_, r in reports.get(j, []):
                        if dates[t] < day_ <= dates[t] + pd.Timedelta(days=arm_days):
                            armed[j] = max(armed[j], r)
        elif chk[t] and cur.sum() > 0:
            order = ranked(t)
            rank = {j: r + 1 for r, j in enumerate(order)}
            skip = {j for j in order[:mw["enter_top"]] if cur[j] == 0 and not np.isnan(BB[t, j])}
            be.midweek_swap_pairs(cur, order, rank, sectors, mw["enter_top"], mw["exit_below"],
                                  10 ** 6 if args["cap_soft"] else per_sector, skip=skip)
            be.midweek_exit_sells(cur, rank, W.get("midweek_exit_below"))
        peak, armed = np.where(cur > 0, peak, np.nan), np.where(cur > 0, armed, -1)   # sold by the live rules: reset
        inside = ~np.isnan(peak) & (cur > 0) & (win[t] | (armed >= t))       # held before today, inside its window
        for j in np.where(inside & (C[t] <= np.fmax(peak, C[t]) - k * A[t]))[0]:
            freed.append(stop_out(t, j))
        for w in freed:                                   # refill each freed slot (its weight) like a live buy
            cands = [j for j in ranked(t)[:args["max_pick_rank"]] if cur[j] == 0 and np.isnan(BB[t, j])]
            fits = [j for j in cands if sum(sectors[cur > 0] == sectors[j]) < per_sector] or (cands if args["cap_soft"] else [])
            if fits:
                cur[fits[0]] = w
        peak = np.where(cur > 0, np.fmax(peak, C[t]), np.nan)
        out[t] = cur
    return pd.DataFrame(out, index=dates, columns=cols), pd.DataFrame(opens, index=dates, columns=cols)


def atr_dip_weights(signal, priority, hold=10, n=10, room=None, taken=None):
    """ATR dip buys: buy at the close of a signal day (1/n each, best priority first, at most `room` positions that day,
    default n), sell at the close `hold` sessions later. taken = stocks it may not hold that day (sold if held); with
    fewer rooms than positions the oldest are sold first."""
    S, P = signal.to_numpy(bool), np.nan_to_num(priority.reindex_like(signal).to_numpy(float), nan=-np.inf)
    room = np.full(len(S), n) if room is None else room
    tk = np.zeros(S.shape, bool) if taken is None else taken
    out, held = np.zeros(S.shape), {}                     # column -> sessions held
    for t in range(len(S)):
        for j in list(held):
            held[j] += 1
            if held[j] >= hold or tk[t, j]:
                del held[j]
        while len(held) > room[t]:
            del held[max(held, key=held.get)]
        new = [j for j in np.argsort(-P[t], kind="stable") if S[t, j] and j not in held and not tk[t, j]]
        for j in new[:max(0, room[t] - len(held))]:
            held[j] = 0
        out[t, list(held)] = 1 / n
    return pd.DataFrame(out, index=signal.index, columns=signal.columns)


def strategy_inputs(sig, facts=None, earnings=None, bench=None, bars=None):
    """Everything the strategies rank on, as Date x Symbol frames, from saved data only: `sig` = signal_analysis.csv rows
    (SIG_COLS), `facts` = factor_history.csv (the daily snapshot of news sentiment and the company-report score; each day
    uses the latest snapshot on or before it), `bench` = benchmark_prices.csv (SPY, QQQ, sector ETFs), `bars` =
    forward_bars() (volume). "through" = the last day the bars cover (later days wait for the bars)."""
    sig = sig[sig["Symbol"].isin(sig.loc[sig["Strategy_Score"].notna(), "Symbol"].unique())]   # scored stocks (no QQQ)
    f = {col: _wide(sig, col) for col in SIG_COLS[2:] if col not in ("Regime_On", "final_trade")}
    dates, c, live, rs, tech = f["Close"].index, f["Close"], f["Strategy_Score"], f["RS_Score"], f["Technical_Score"]
    cols, eligible, earnings = c.columns, f["Strategy_Score"].notna(), be.load_earnings() if earnings is None else earnings
    snap = {}
    for col in ("SentimentScore", "Fundamental_Weight"):
        snap[col] = pd.DataFrame(index=dates, columns=cols, dtype=float)
        if facts is not None and len(facts):
            fa = facts.assign(bar_date=pd.to_datetime(facts["bar_date"])).sort_values("as_of")
            w = _wide(fa.drop_duplicates(["bar_date", "Symbol"], keep="last"), col, index="bar_date").reindex(columns=cols)
            snap[col] = w.reindex(w.index.union(dates)).ffill().reindex(dates)
    b = pd.DataFrame(index=dates) if bench is None else \
        bench.assign(Date=pd.to_datetime(bench["Date"])).set_index("Date").sort_index()
    b = b.reindex(b.index.union(dates)).ffill().reindex(dates)
    etf_of = {s: be.sector_mapping.sector_etf_for(s) for s in cols}
    by_etf = lambda x: pd.DataFrame({s: x[etf_of[s]] if etf_of[s] in x else np.nan for s in cols}, index=dates)
    bx = bar_features(forward_bars() if bars is None else bars, dates, cols)
    quant, vol, up = quant_score(f, bx["vwap_ratio"]), be.volatility(c), c > f["ma_200"]
    regime = sig.groupby("Date")["Regime_On"].first().reindex(dates).fillna(0).astype(bool)
    weekly = be.weekly_rebalance_days(dates, live=True)
    ret = lambda n: c / c.shift(n) - 1
    mom_12_1, etf_3m = (c.shift(21) / c.shift(252) - 1) * 100, by_etf(b / b.shift(63) - 1)
    scores = {
        "live": live, "tech": tech, "rs": rs,
        "quant": quant - 50,                                  # > 0 = Quant Score above 50 (the copy's min score)
        "quant_rs": 100 * (0.5 * _pct(quant) + 0.5 * _pct(rs)),
        "sentiment": (100 * (0.8 * _pct(live) + 0.2 * _pct(snap["SentimentScore"].fillna(0.0)))).where(live > 0),
        "fundamental": snap["Fundamental_Weight"],
        "mom_12_1": mom_12_1,
        "mom_6": ret(126) * 100,
        "low_vol": (1 / vol).where(up),
        "dual": (ret(252) * 100).where(regime, axis=0),
        "mom_risk": ret(126) / (c.pct_change(fill_method=None).rolling(126).std() * np.sqrt(126)),
        "high_52": c / c.rolling(252).max(),
        "reversal": (-ret(5) * 100).where(up),
        "quality_mom": (100 * (0.5 * _pct(snap["Fundamental_Weight"]) + 0.5 * _pct(mom_12_1))),
        "drift": earnings_drift(c, b["SPY"] if "SPY" in b else pd.Series(1.0, index=dates), earnings),
        "sector_rot": live.where(etf_3m.rank(axis=1, method="dense", ascending=False).le(3)),   # top 3 ETF returns
        "breakout": bx["vol_jump"].where((c >= 0.98 * c.rolling(55).max()) & (bx["vol_jump"] >= 1.3))
                    .rolling(20, min_periods=1).max().where(c > f["ma_50"]),     # a breakout in the last 20 sessions
    }
    # Legacy combined_signal (Stock Analysis Legacy/main_signal_analysis.ipynb): missing news / fundamentals = that day's mean
    fill = lambda x: x.T.fillna(x.mean(axis=1)).T.fillna(0.0)
    legacy = (0.60 * tech + 0.25 * fill(snap["Fundamental_Weight"]).clip(-10, 10) * 10
              + 0.10 * (by_etf(b / b.shift(4) - 1) * 300).clip(-30, 30) / 30 * 100
              + 0.05 * fill(snap["SentimentScore"]).clip(-10, 10) * 10)
    held = legacy.gt(20).astype(float).where(legacy.gt(20) | legacy.lt(-25)).ffill().fillna(0.0).where(eligible, 0.0)
    atr = bx["atr_ratio"] * c
    after_earn = np.zeros(c.shape, bool)                  # the reaction day + 10 sessions after each report
    for j, _, r in earnings_events(dates, cols, earnings):
        after_earn[r:r + 11, j] = True
    top30 = live.where(eligible & (live > 0)).rank(axis=1, ascending=False, method="first") <= 30
    dip_a = (c <= c.rolling(20).max() - 3 * atr) & top30                 # 3x ATR(14) below the 20-day high close
    dip_b = dip_a & ~after_earn
    soon = be.earnings_days_ahead(dates, list(cols), earnings, be.WINNER.get("earnings_block_days")).notna()
    n = eligible.sum(axis=1)
    weights = {"atr_dip": atr_dip_weights(dip_a, live), "atr_dip_ex_earn": atr_dip_weights(dip_b, live),"legacy": held.div(n, axis=0),
               "dip": dip_weights(c, f["ma_50"], eligible),
               "trend": _carry(((f["ma_50"] > f["ma_200"]) & eligible).astype(float).div(n, axis=0), weekly)}
    have = bx["vwap_ratio"].notna().any(axis=1)
    return {"close": c.ffill(), "scores": scores, "weights": weights, "eligible": eligible, "vol": vol, "regime": regime,
            "weekly": weekly, "earnings": earnings, "atr": atr,
            "dip_spare": dip_b & ~soon.reindex(index=dates, columns=cols).fillna(False), "open": bx["open_ratio"] * c,
            "monthly": pd.Series([be.next_sessions(d, 1)[0].month != d.month for d in dates], index=dates),
            "through": have[have].index.max() if have.any() else pd.Timestamp(0)}


def strategy_targets(cfg, inp):
    """Daily live target weights of one STRATEGIES entry (Date x Symbol). Ranked ones go through the engine's own selection
    code (be.winner_targets): only overrides are passed, WINNER itself is never changed. A "stop" strategy also leaves its
    open-sale prices in inp["open_sells"][name]."""
    if cfg.get("weights"):
        tgt = inp["weights"][cfg["weights"]]
    elif cfg.get("stop"):
        tgt, inp.setdefault("open_sells", {})[cfg["name"]] = atr_stop_targets(inp, **STOPS[cfg["stop"]])
    else:
        cal, tb = cfg.get("calendar", "mwf"), cfg.get("tiebreak", "rs")
        tgt, _ = be.winner_targets(inp["scores"][cfg.get("score", "live")], inp["eligible"], inp["vol"], inp["regime"],
                                   inp["weekly" if cal == "mwf" else cal], tiebreak_w=inp["scores"].get(tb),
                                   midweek=None if cal == "mwf" else False, selection=cfg.get("select"),
                                   earnings_block_days=cfg.get("earnings", "winner"), earnings=inp["earnings"])
        if cfg.get("reweight") == "risk_parity":
            tgt = risk_parity(tgt, inp["close"][tgt.columns], inp["weekly"])
        if cfg.get("spare_dips"):                         # the cash the live rules leave unused buys ATR dips (10% each)
            tgt = tgt + atr_dip_weights(inp["dip_spare"][tgt.columns], inp["scores"]["live"], taken=(tgt > 0).to_numpy(),
                                        room=np.floor((1 - tgt.sum(axis=1)).to_numpy() * 10 + 1e-9).astype(int))
    return be.live_weights(tgt.clip(upper=MAX_WEIGHT)).reindex(columns=inp["close"].columns).fillna(0.0)


def update_strategies(sig=None, facts=None, earnings=None, bench=None, bars=None, path=STRATEGIES_CSV,
                      hold_path=HOLDINGS_CSV, start=be.FORWARD_START):
    """Add every trading day after each strategy's last saved day (the first one: the start close, value 1.0 after its
    buys), up to the last day the volume bars cover. Saved days are never redone. Returns the number of rows added."""
    sig = _read(SIGNAL_CSV, usecols=SIG_COLS, parse_dates=["Date"]) if sig is None else sig
    facts = _read(FACTOR_CSV) if facts is None else facts
    bench = _read(BENCH_CSV) if bench is None else bench
    inp = strategy_inputs(sig, facts, earnings, bench, bars)
    close = inp["close"]
    days = close.index[close.index >= pd.Timestamp(start)]
    if inp.get("through") is not None and len(days) and days[-1] > inp["through"]:
        print(f"waiting for volume bars after {inp['through']:%b %-d} (days after it are added on a later run)")
        days = days[days <= inp["through"]]
    old = _read(path, parse_dates=["Date"])
    old_h = _read(hold_path, parse_dates=["Date"])
    rows, hold = [], []
    for cfg in STRATEGIES:
        name = cfg["name"]
        mine = old[old["Strategy"] == name] if old is not None else pd.DataFrame(columns=["Date"])
        last = mine["Date"].max() if len(mine) else None
        todo = days if last is None else days[days > last]
        if not len(todo):
            continue
        h = pd.DataFrame(columns=["Symbol", "Weight", "Shares", "Price"])
        if last is not None and old_h is not None:
            h = old_h[(old_h["Strategy"] == name) & (old_h["Date"] == last)]
        h = h.set_index("Symbol")
        cols = close.columns.union(h.index)
        zero = pd.Series(0.0, index=cols)
        cash, shares, prev, last_px = 1.0, zero, None, h["Price"].reindex(cols).astype(float)
        if last is not None:                                  # resume from the last saved day (no rows = all cash)
            cash = float(mine.loc[mine["Date"] == last, "Cash"].iloc[0])
            shares, prev = zero.add(h["Shares"], fill_value=0.0), zero.add(h["Weight"], fill_value=0.0)
        w = strategy_targets(cfg, inp).reindex(columns=cols).fillna(0.0)
        opens = inp.get("open_sells", {}).get(name, pd.DataFrame()).reindex(index=todo, columns=cols)
        for d in todo:
            px = close.loc[d].reindex(cols).fillna(last_px)  # a stock gone from the list keeps its last saved price
            at_open = opens.loc[d][(shares > 0) & opens.loc[d].notna()]
            cash += (shares[at_open.index] * at_open * (1 - be.COST)).sum()   # stop sales at the open (0.1% cost)
            shares = shares.where(~shares.index.isin(at_open.index), 0.0)
            value = cash + (shares * px).sum()
            wd = w.loc[d]
            if prev is None or not np.allclose(wd, prev):
                value -= be.COST * (wd * value - shares * px).abs().sum()
                shares = (wd * value / px.where(px > 0)).fillna(0.0)
                cash, prev = value - (shares * px).sum(), wd
            if last is None and d == todo[0]:                 # 1.0 at the start close, after its buys
                shares, cash, value = shares / value, cash / value, 1.0
            last_px = px
            rows.append({"Date": d, "Strategy": name, "Value": value, "Cash": cash})
            on = (wd > 0) | (shares > 0)
            hold += [{"Date": d, "Strategy": name, "Symbol": s, "Weight": wd[s], "Shares": shares[s], "Price": px[s]}
                     for s in cols[on.to_numpy()]]
    if rows:
        for p, new, prior in ((path, rows, old), (hold_path, hold, old_h)):
            out = pd.concat([prior, pd.DataFrame(new)], ignore_index=True) if prior is not None else pd.DataFrame(new)
            out = out.assign(Date=pd.to_datetime(out["Date"])).sort_values("Date", kind="stable")   # by day, registry order
            out.assign(Date=out["Date"].dt.strftime("%Y-%m-%d")).to_csv(p, index=False)
    return len(rows)


# ---------------------------------------------------------------------------------------------------- the live account
def _fill_rows(fills):
    """GET /account/activities FILL dicts -> DataFrame (Time in CT, naive), oldest first."""
    f = pd.DataFrame([{"Time": x.get("transaction_time"), "Symbol": x.get("symbol"), "Side": str(x.get("side", "")).lower(),
                       "Qty": float(x.get("qty") or 0), "Price": float(x.get("price") or 0)} for x in fills],
                     columns=["Time", "Symbol", "Side", "Qty", "Price"])
    f["Time"] = pd.to_datetime(f["Time"], utc=True).dt.tz_convert(CT).dt.tz_localize(None)
    return f.sort_values("Time", kind="stable").reset_index(drop=True)


def picks(fills, start=be.FORWARD_START):
    """Closed picks: a pick opens with a buy into a flat position on/after `start` and closes when the position is back to
    0; P/L = sell proceeds - buy cost. A position already held before `start` is not counted (its round trip began
    earlier)."""
    f, start, out, book = _fill_rows(fills), pd.Timestamp(start), [], {}
    for r in f.itertuples():
        b = book.setdefault(r.Symbol, {"qty": 0.0, "counted": False, "entry": None, "buy": 0.0, "sell": 0.0})
        if b["qty"] <= 1e-9 and r.Side == "buy":
            b.update(counted=r.Time >= start, entry=r.Time, buy=0.0, sell=0.0)
        b["qty"] += r.Qty if r.Side == "buy" else -r.Qty
        b["buy" if r.Side == "buy" else "sell"] += r.Qty * r.Price
        if b["qty"] <= 1e-9 and r.Side != "buy":
            if b["counted"]:
                out.append({"Symbol": r.Symbol, "Entry": b["entry"], "Exit": r.Time, "P/L $": b["sell"] - b["buy"],
                            "Return %": (b["sell"] / b["buy"] - 1) * 100 if b["buy"] else np.nan})
            b.update(qty=0.0, counted=False)
    return pd.DataFrame(out, columns=["Symbol", "Entry", "Exit", "P/L $", "Return %"])


def trade_cost(fills, orders, closes, start=be.FORWARD_START):
    """($ traded, $ cost) of the fills on/after `start` vs the decision price: buys (fill - decision price) x qty, sells
    (decision price - fill) x qty, so + = it cost money. The decision price is the stock's close on the As_Of day of the
    latest live_orders_log row with the same symbol and side sent before the fill; fills with no such row are left out."""
    f = _fill_rows(fills)
    f = f[f["Time"] >= pd.Timestamp(start)]
    if f.empty or orders is None or orders.empty:
        return 0.0, 0.0
    o = orders.assign(Sent=pd.to_datetime(orders["Submitted_At_CT"], errors="coerce"),
                      Side=orders["Side"].astype(str).str.lower(), As_Of=pd.to_datetime(orders["As_Of"], errors="coerce"))
    traded = cost = 0.0
    for r in f.itertuples():
        m = o[(o["Symbol"] == r.Symbol) & (o["Side"] == r.Side) & (o["Sent"] <= r.Time)]
        ref = closes.get((m["As_Of"].iloc[-1], r.Symbol)) if len(m) else None
        if ref is None or not ref == ref:
            continue
        traded += r.Qty * r.Price
        cost += (r.Price - ref) * r.Qty * (1 if r.Side == "buy" else -1)
    return traded, cost


def _read(path, **kw):
    return pd.read_csv(path, **kw) if os.path.exists(path) else None


def record(account=None, now=None, path=DAILY_CSV):
    """Save today's row (only on an NYSE session; replaces a row already saved today). Returns the row, or None."""
    now = (now or datetime.now(CT)).astimezone(CT)
    if not be.is_session(now.date()) or now.date() < pd.Timestamp(be.FORWARD_START).date():
        print(f"idle: {now:%a %b %-d} is not a trading day on/after the forward-test start")
        return None
    if account is None:
        import alpaca_paper as ap
        account = ap.PaperAccount()
    s, fills = account.account_summary(), account.fills()
    sig = _read(SIGNAL_CSV, usecols=["Date", "Symbol", "Close"], parse_dates=["Date"])
    closes = {} if sig is None else dict(zip(zip(sig["Date"], sig["Symbol"]), sig["Close"]))
    pk = picks(fills)
    traded, cost = trade_cost(fills, _read(ORDERS_CSV), closes)
    row = {"Date": f"{now:%Y-%m-%d}", "Time_CT": f"{now:%H:%M}", "Equity": s["Equity"], "Cash": s["Cash"],
           "Net_Deposits": account.net_deposits(), "Positions": len(account.position_dicts()),
           "Closed_Picks": len(pk), "Winning_Picks": int((pk["P/L $"] > 0).sum()), "Traded_USD": round(traded, 2),
           "Cost_USD": round(cost, 2)}
    old = _read(path, dtype={"Date": str})
    new = pd.DataFrame([row])
    if old is not None and (old["Date"] != row["Date"]).any():
        new = pd.concat([old[old["Date"] != row["Date"]], new], ignore_index=True)
    new[DAILY_COLS].to_csv(path, index=False)
    return row


# ---------------------------------------------------------------------------------------------------- leaderboard
def _stats(v):
    """Total return %, median weekly return % (week end to week end: the last session of each finished week, from the
    start close), max drawdown % and finished weeks of a value series by date."""
    v = v.dropna().sort_index()
    if len(v) < 1:
        return np.nan, np.nan, np.nan, 0
    weekly = v[be.weekly_rebalance_days(v.index, live=True).to_numpy() | (v.index == v.index[0])].pct_change().dropna()
    return ((v.iloc[-1] / v.iloc[0] - 1) * 100, weekly.median() * 100 if len(weekly) else np.nan,
            (v / v.cummax() - 1).min() * 100, len(weekly))


def leaderboard(daily=None, bench=None, values=None, start=be.FORWARD_START):
    """One row per strategy (STRATEGIES order) + the real account (equity net of deposits made after the first saved
    day: deposits are not returns) + QQQ / SPY (closes, comparison only), sorted by RANK_RULE. Rank counts strategies
    only; the account and QQQ / SPY show where they would sit."""
    daily = _read(DAILY_CSV, parse_dates=["Date"]) if daily is None else daily
    bench = _read(BENCH_CSV, parse_dates=["Date"]) if bench is None else bench
    values = _read(STRATEGIES_CSV, parse_dates=["Date"]) if values is None else values
    start, series = pd.Timestamp(start), []

    def since(df):
        return df.assign(Date=pd.to_datetime(df["Date"])).set_index("Date").sort_index().loc[start:]
    for cfg in STRATEGIES:
        v = since(values[values["Strategy"] == cfg["name"]])["Value"] if values is not None else pd.Series(dtype=float)
        series.append((cfg["name"] + (LIVE_MARK if cfg["name"] == LIVE else ""), True, v))
    if daily is not None and len(daily):
        d = since(daily)
        if len(d):
            series.append((ACCOUNT, False, d["Equity"] - (d["Net_Deposits"] - d["Net_Deposits"].iloc[0])))
    if bench is not None:
        b = since(bench)
        series += [(f"{s} (comparison)", False, b[s]) for s in ("QQQ", "SPY") if s in b]
    cols = ["Total return %", "Median weekly return %", "Max drawdown %", "Weeks"]
    t = pd.DataFrame([{"Strategy": n, "ranked": r, **dict(zip(cols, _stats(v)))} for n, r, v in series])
    if t.empty:
        return pd.DataFrame(columns=["Rank", "Strategy"] + cols)
    t = t.sort_values(["Median weekly return %", "Max drawdown %"], ascending=False, na_position="last", kind="stable")
    t["Rank"] = t["ranked"].cumsum().where(t["ranked"] & (t["Weeks"] > 0)).astype("Int64")   # no rank before a full week
    return t[["Rank", "Strategy"] + cols].reset_index(drop=True)


def verdict(board):
    """One plain line: too early / the winner / no winner (RANK_RULE)."""
    by = board.set_index(board["Strategy"].str.replace(LIVE_MARK, "", regex=False))
    weeks = int(by["Weeks"].get(LIVE, 0) or 0)
    if weeks < WEEKS_TO_WIN:
        return f"Week {weeks} of {WEEKS_TO_WIN}: too early to name a winner."
    top = by[board["Rank"].notna().to_numpy()].index[0]
    bar = max(by.at[LIVE, "Total return %"], by["Total return %"].get("QQQ (comparison)", -np.inf))
    if top == LIVE:
        return f"After {weeks} weeks the live rules rank first: keep them."
    if by.at[top, "Total return %"] > bar:
        return f"After {weeks} weeks the winner is {top}: it ranks first and beats {LIVE} and QQQ."
    return f"After {weeks} weeks no winner: {top} ranks first but does not beat both {LIVE} and QQQ. Keep the live rules."


def holdings(path=HOLDINGS_CSV):
    """Strategy -> 'NVDA 14.2%, ...' of its latest saved day (target weights, largest first)."""
    h = _read(path, parse_dates=["Date"])
    if h is None or h.empty:
        return {}
    h = h[h["Date"] == h.groupby("Strategy")["Date"].transform("max")]
    h = h[h["Weight"] > 0].sort_values(["Strategy", "Weight"], ascending=[True, False])
    return {s: ", ".join(f"{r.Symbol} {r.Weight:.1%}" for r in g.itertuples()) for s, g in h.groupby("Strategy")}


def _log_problem(message, error):
    """A failed forward-test update -> one Reports/run_log.csv row (the dashboard's run log), so it is not only in the
    launchd log. Never raises."""
    print(f"{message} ({type(error).__name__}: {error})")
    try:
        import run_all
        run_all.log_event("Forward test", "failed", "no", message, f"{type(error).__name__}: {error}")
    except Exception as e:
        print(f"run log not written ({e})")


def main(argv=None):
    """Update the paper strategies (and with --record save today's account row), then print the leaderboard. Returns the
    exit code: 1 when a part failed (each failure is one run-log row; a saved day is never redone or lost)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--record", action="store_true", help="also save today's account row (reads the live account, GET only)")
    a = p.parse_args(argv)
    rc = 0
    try:
        print(f"strategies: {update_strategies()} new row(s) -> {os.path.relpath(STRATEGIES_CSV, ROOT)}")
    except Exception as e:  # never stops the account row
        _log_problem("The paper strategies were not updated today; the next run adds the missing days.", e)
        rc = 1
    if a.record:
        try:
            row = record()
        except Exception as e:  # Alpaca down / keys missing: the next trading day's run still adds its own row
            _log_problem("Today's account row for the forward test was not saved (the Alpaca account read failed). "
                         "No money moved.", e)
            row, rc = None, 1
        if row:
            print("saved", {k: row[k] for k in ("Date", "Time_CT", "Positions", "Closed_Picks")}, "->",
                  os.path.relpath(DAILY_CSV, ROOT))
    board = leaderboard()
    print(board.round(2).to_string(index=False))
    print(verdict(board))
    return rc


if __name__ == "__main__":
    sys.exit(main())
