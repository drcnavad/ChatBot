"""Forward test from backtest_engine.FORWARD_START (Fri Oct 2, 2026), counted only from that close on, never backtested:

  1. the real Alpaca account (read-only GET requests through alpaca_paper; no orders, ever), and
  2. STRATEGIES: about 20 paper-only strategies next to it (the live rules recomputed the same way + 19 others). They are
     never traded; each one is a list of target weights built from data the pipeline already saves every day
     (Reports/signal_analysis.csv, Reports/factor_history.csv, Reports/earnings_date.csv), so no API quota is used.

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
    python forward_test.py            # update the strategies (no requests) and print the leaderboard
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
             f"return also beats {LIVE} and QQQ; otherwise the live rules stay.")


# ---------------------------------------------------------------------------------------------------- the strategies
# One dict per strategy. Defaults = the live rules: score "live" (0.5 x technical + 0.5 x relative strength), relative
# strength tiebreak, Friday rebalance + Mon/Wed swaps and exits ("mwf"), top 10 from ranks 1-20, max 4 per sector (relaxed
# to fill the slots), 1/volatility weights, half size when QQQ is below its 200-day average, no buys 5 days before
# earnings. "select" changes rank_targets keys (n, sector_cap, vol_sizing, regime, ...), "earnings" None drops the
# earnings skip, "calendar" is "mwf" / "weekly" (Friday only) / "monthly" (last session of the month) / "daily".
PLAIN = {"max_pick_rank": None, "cap_soft": False}       # walk every rank, strict max 4 per sector
STRATEGIES = [
    dict(name=LIVE, rule="What the bot trades: 50% technical + 50% relative strength, top 10, Fri rebalance + Mon/Wed swaps."),
    dict(name="Live without rank-20 limit / earnings skip", select=PLAIN, earnings=None,
         rule="The live rules, but picks may come from any rank and earnings never block a buy."),
    dict(name="Technical score only", score="tech", rule="Live rules ranking on the technical score alone."),
    dict(name="Relative strength only", score="rs", tiebreak="tech",
         rule="Live rules ranking on relative strength (vs sector ETF and SPY) alone."),
    dict(name="Quant Score", score="quant", tiebreak=None,
         rule="Live rules ranking on the copy folder's Quant Score (trend, MAs, RSI, MACD, Bollinger); needs a score above 50."),
    dict(name="Quant + relative strength", score="quant_rs",
         rule="Live rules ranking on a 50/50 blend of Quant Score rank and relative strength rank."),
    dict(name="Live + news sentiment", score="sentiment",
         rule="Live rules, ranking 80% on the live score and 20% on the latest news sentiment."),
    dict(name="Company fundamentals", score="fundamental", calendar="monthly", select={**PLAIN, "vol_sizing": False},
         earnings=None, rule="Top 10 by the company-report score (Fundamental_Weight), equal weight, monthly."),
    dict(name="12-1 month momentum", score="mom_12_1", calendar="monthly", select=PLAIN, earnings=None,
         rule="Top 10 by the 12-month return skipping the last month, monthly, 1/volatility weights."),
    dict(name="6-month momentum", score="mom_6", calendar="weekly", select=PLAIN, earnings=None,
         rule="Top 10 by the 6-month return, every Friday, 1/volatility weights."),
    dict(name="Low volatility in uptrend", score="low_vol", calendar="monthly", select={**PLAIN, "vol_sizing": False},
         earnings=None, rule="The 10 calmest stocks (63-day volatility) above their 200-day average, equal weight, monthly."),
    dict(name="Buy the dip", score="dip", calendar="weekly", select={**PLAIN, "vol_sizing": False}, earnings=None,
         rule="Stocks above their 200-day average that fell furthest below their 50-day average; top 10, equal weight, "
              "every Friday (sold once back above the 50-day)."),
    dict(name="Legacy BUY/SELL signals", score="legacy", calendar="daily",
         rule="The Legacy folder's rule: hold a stock from its BUY signal to its SELL signal, an equal slice per stock."),
    dict(name="Top 5", calendar="weekly", select={"n": 5, "max_pick_rank": 10},
         rule="Live score, 5 stocks from ranks 1-10 (max 2 per sector), Friday only."),
    dict(name="Top 20", calendar="weekly", select={"n": 20, "max_pick_rank": 40},
         rule="Live score, 20 stocks from ranks 1-40 (max 8 per sector), Friday only."),
    dict(name="Equal weight", select={"vol_sizing": False}, rule="The live rules with an equal weight per stock."),
    dict(name="Friday only", calendar="weekly", rule="The live rules without the Mon/Wed swaps and exits."),
    dict(name="Monthly", calendar="monthly", rule="The live rules, rebalanced on the last trading day of each month only."),
    dict(name="No QQQ filter", select={"regime": None, "regime_scale": None},
         rule="The live rules, always fully invested (no halving when QQQ is below its 200-day average)."),
    dict(name="No sector limit", select={"sector_cap": 1.0, "cap_soft": True},
         rule="The live rules without the max-4-per-sector rule (pure top 10)."),
]


def _wide(df, col, index="Date"):
    return df.pivot(index=index, columns="Symbol", values=col).sort_index()


def _pct(w):
    return w.rank(axis=1, pct=True)


def quant_score(f):
    """The copy folder's Quant_Score (Stock Analysis Test Strategy/backtest_engine.quant_score), 0-100, from the saved
    signal_analysis.csv columns: the average of price above its 5 moving averages, the moving-average stack, RSI(14), the
    MACD histogram (point-in-time scaled), Bollinger %B (20 days) and the 20-day slope of the 200-day average (a missing
    part counts as 50). Its VWAP part is left out because daily volume is not saved."""
    c, ma = f["Close"], {n: f[f"ma_{n}"] for n in (10, 30, 50, 100, 200)}
    mid, sd = c.rolling(be.BB_WINDOW).mean(), c.rolling(be.BB_WINDOW).std()
    parts = [sum((c > m).astype(float) for m in ma.values()) / 5 * 100,
             sum((ma[a] > ma[b]).astype(float) for a, b in ((10, 30), (30, 50), (50, 100), (100, 200))) / 4 * 100,
             f["RSI"].clip(0, 100),
             (50 + (f["macd"] - f["MACD Signal"]).apply(be.pit_scale) / 2).clip(0, 100),
             ((c - (mid - 2 * sd)) / (4 * sd).replace(0, np.nan)).clip(0, 1) * 100,
             (50 + 1250 * (ma[200] / ma[200].shift(20) - 1)).clip(0, 100)]
    return sum(p.fillna(50.0) for p in parts) / len(parts)


def strategy_inputs(sig, facts=None, earnings=None):
    """Everything the strategies rank on, as Date x Symbol frames, from the saved files only. `sig` = signal_analysis.csv
    rows (SIG_COLS), `facts` = factor_history.csv (the daily snapshot of news sentiment and the company-report score; each
    day uses the latest snapshot on or before it)."""
    sig = sig[sig["Symbol"].isin(sig.loc[sig["Strategy_Score"].notna(), "Symbol"].unique())]   # scored stocks (no QQQ)
    f = {c: _wide(sig, c) for c in SIG_COLS[2:] if c not in ("Regime_On", "final_trade")}
    dates, c, live, rs, tech = f["Close"].index, f["Close"], f["Strategy_Score"], f["RS_Score"], f["Technical_Score"]
    snap = {}
    for col in ("SentimentScore", "Fundamental_Weight"):
        s = pd.DataFrame(index=dates, columns=c.columns, dtype=float)
        if facts is not None and len(facts):
            fa = facts.assign(bar_date=pd.to_datetime(facts["bar_date"])).sort_values("as_of")
            fa = fa.drop_duplicates(["bar_date", "Symbol"], keep="last")
            w = _wide(fa, col, index="bar_date").reindex(columns=c.columns)
            s = w.reindex(w.index.union(dates)).ffill().reindex(dates)
        snap[col] = s
    quant, vol = quant_score(f), be.volatility(c)
    uptrend = c > f["ma_200"]
    scores = {
        "live": live, "tech": tech, "rs": rs,
        "quant": quant - 50,                                  # > 0 = Quant Score above 50 (the copy's min score)
        "quant_rs": 100 * (0.5 * _pct(quant) + 0.5 * _pct(rs)),
        "sentiment": (100 * (0.8 * _pct(live) + 0.2 * _pct(snap["SentimentScore"].fillna(0.0)))).where(live > 0),
        "fundamental": snap["Fundamental_Weight"],
        "mom_12_1": (c.shift(21) / c.shift(252) - 1) * 100,
        "mom_6": (c / c.shift(126) - 1) * 100,
        "low_vol": (1 / vol).where(uptrend),
        "dip": ((f["ma_50"] / c - 1) * 100).where(uptrend),
    }
    held = _wide(sig, "final_trade").map(lambda x: {"BUY": 1.0, "SELL": 0.0}.get(x, np.nan)).ffill().fillna(0.0)
    eligible = live.notna()
    legacy = (held.where(eligible, 0.0)).div(eligible.sum(axis=1), axis=0)
    return {"close": c.ffill(), "scores": scores, "eligible": eligible, "vol": vol, "legacy": legacy,
            "regime": sig.groupby("Date")["Regime_On"].first().reindex(dates).fillna(0).astype(bool),
            "weekly": be.weekly_rebalance_days(dates, live=True),
            "monthly": pd.Series([be.next_sessions(d, 1)[0].month != d.month for d in dates], index=dates),
            "earnings": be.load_earnings() if earnings is None else earnings}


def strategy_targets(cfg, inp):
    """Daily live target weights of one STRATEGIES entry (Date x Symbol), through the engine's own selection code
    (be.winner_targets): only overrides are passed, WINNER itself is never changed."""
    if cfg.get("score") == "legacy":
        tgt = inp["legacy"]
    else:
        cal, tb = cfg.get("calendar", "mwf"), cfg.get("tiebreak", "rs")
        tgt, _ = be.winner_targets(inp["scores"][cfg.get("score", "live")], inp["eligible"], inp["vol"], inp["regime"],
                                   inp["weekly" if cal == "mwf" else cal], tiebreak_w=inp["scores"].get(tb),
                                   midweek=None if cal == "mwf" else False, selection=cfg.get("select"),
                                   earnings_block_days=cfg.get("earnings", "winner"), earnings=inp["earnings"])
    return be.live_weights(tgt.clip(upper=MAX_WEIGHT)).reindex(columns=inp["close"].columns).fillna(0.0)


def update_strategies(sig=None, facts=None, earnings=None, path=STRATEGIES_CSV, hold_path=HOLDINGS_CSV,
                      start=be.FORWARD_START):
    """Add every trading day after each strategy's last saved day (the first one: the start close, value 1.0 after its
    buys). Saved days are never redone. Returns the number of (strategy, day) rows added."""
    sig = _read(SIGNAL_CSV, usecols=SIG_COLS, parse_dates=["Date"]) if sig is None else sig
    facts = _read(FACTOR_CSV) if facts is None else facts
    inp = strategy_inputs(sig, facts, earnings)
    close = inp["close"]
    days = close.index[close.index >= pd.Timestamp(start)]
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
        for d in todo:
            px = close.loc[d].reindex(cols).fillna(last_px)  # a stock gone from the list keeps its last saved price
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--record", action="store_true", help="also save today's account row (reads the live account, GET only)")
    a = p.parse_args(argv)
    try:
        print(f"strategies: {update_strategies()} new row(s) -> {os.path.relpath(STRATEGIES_CSV, ROOT)}")
    except Exception as e:  # never stops the account row
        print(f"strategies not updated: {type(e).__name__}: {e}")
    if a.record:
        row = record()
        if row:
            print("saved", {k: row[k] for k in ("Date", "Time_CT", "Positions", "Closed_Picks")}, "->",
                  os.path.relpath(DAILY_CSV, ROOT))
    board = leaderboard()
    print(board.round(2).to_string(index=False))
    print(verdict(board))


if __name__ == "__main__":
    main()
