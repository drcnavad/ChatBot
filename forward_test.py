"""Forward test of the LIVE strategy from backtest_engine.FORWARD_START (Fri Oct 2, 2026): the real Alpaca account
against QQQ and SPY, counted only from that close on. Read-only: GET requests through alpaca_paper (no orders, ever).

  Reports/forward_test_daily.csv   one row per trading day (re-recording a day replaces it): account equity, cash, lifetime
                                   net deposits, positions, closed picks / winners, $ traded and its cost vs the decision
                                   price (the stock's close in signal_analysis.csv on the order's As_Of day, which on a
                                   decision day is the 2:30 PM bar the strategy decided on)
                                   and the no-orders shadow's value (shadow_values)
  summary()                        Strategy (equity net of new deposits) / shadow / QQQ / SPY: total return, median weekly
                                   return, max drawdown (QQQ/SPY closes from Reports/benchmark_prices.csv)

    python forward_test.py --record   # save today's row (launchd com.stockanalysis.forwardtest, Mon-Fri 4:15 PM CT)
    python forward_test.py            # print the summary from the saved rows (no requests)
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
BENCH_CSV = os.path.join(REPORTS, "benchmark_prices.csv")
ORDERS_CSV = os.path.join(REPORTS, "live_orders_log.csv")
SIGNAL_CSV = os.path.join(REPORTS, "signal_analysis.csv")
CT = ZoneInfo("America/Chicago")
DAILY_COLS = ["Date", "Time_CT", "Equity", "Cash", "Net_Deposits", "Positions", "Closed_Picks", "Winning_Picks",
              "Traded_USD", "Cost_USD", "Shadow_Value"]
SHADOW = "Shadow: no rank-20 limit, no earnings skip (no orders)"


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


def shadow_values(sig, start=be.FORWARD_START):
    """Value (1.0 at the `start` close, after its buys) of the no-orders shadow: the live rules without the rank-20 pick limit (T20 /
    cap_soft) and without the earnings skip (E5), on the saved inputs in signal_analysis.csv `sig` (scores, RS tiebreak,
    market filter, closes; the live targets rebuild exactly from them). Trades at the close whenever its target changes,
    be.COST (0.1%) per side, 99% invested like the live weights. Only overrides are passed: WINNER is never changed."""
    wide = lambda c: sig.pivot(index="Date", columns="Symbol", values=c)
    score, close = wide("Strategy_Score"), wide("Close")
    tgt, _ = be.winner_targets(score, score.notna(), be.volatility(close[score.columns]),
                               sig.groupby("Date")["Regime_On"].first().astype(bool),
                               be.weekly_rebalance_days(score.index, live=True), tiebreak_w=wide("RS_Score"),
                               selection={"max_pick_rank": None, "cap_soft": False}, earnings_block_days=None)
    w = be.live_weights(tgt).reindex(columns=close.columns).fillna(0.0).loc[pd.Timestamp(start):]
    cash, shares, prev, out = 1.0, pd.Series(0.0, index=close.columns), None, {}
    for d, wd in w.iterrows():
        px = close.loc[d].fillna(0.0)
        value = cash + (shares * px).sum()
        if prev is None or not np.allclose(wd, prev):
            value -= be.COST * (wd * value - shares * px).abs().sum()
            shares = (wd * value / px.where(px > 0)).fillna(0.0)
            cash, prev = value - (shares * px).sum(), wd
        out[d] = value
    out = pd.Series(out, dtype=float)
    return out / out.iloc[0] if len(out) else out      # 1.0 after the start close's buys, like the account baseline


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
    sig = _read(SIGNAL_CSV, usecols=["Date", "Symbol", "Close", "RS_Score", "Strategy_Score", "Regime_On"], parse_dates=["Date"])
    shadow = shadow_values(sig) if sig is not None else pd.Series(dtype=float)
    closes = {} if sig is None else dict(zip(zip(sig["Date"], sig["Symbol"]), sig["Close"]))
    pk = picks(fills)
    traded, cost = trade_cost(fills, _read(ORDERS_CSV), closes)
    row = {"Date": f"{now:%Y-%m-%d}", "Time_CT": f"{now:%H:%M}", "Equity": s["Equity"], "Cash": s["Cash"],
           "Net_Deposits": account.net_deposits(), "Positions": len(account.position_dicts()),
           "Closed_Picks": len(pk), "Winning_Picks": int((pk["P/L $"] > 0).sum()), "Traded_USD": round(traded, 2),
           "Cost_USD": round(cost, 2), "Shadow_Value": round(float(shadow.iloc[-1]), 6) if len(shadow) else np.nan}
    old = _read(path, dtype={"Date": str})
    new = pd.DataFrame([row])
    if old is not None and (old["Date"] != row["Date"]).any():
        new = pd.concat([old[old["Date"] != row["Date"]], new], ignore_index=True)
    new[DAILY_COLS].to_csv(path, index=False)
    return row


def _stats(v):
    """Total return %, median weekly return % (Friday to Friday) and max drawdown % of a value series by date."""
    v = v.dropna()
    if len(v) < 1:
        return np.nan, np.nan, np.nan
    weekly = v.resample("W-FRI").last().dropna().pct_change().dropna()
    return ((v.iloc[-1] / v.iloc[0] - 1) * 100, weekly.median() * 100 if len(weekly) else np.nan,
            (v / v.cummax() - 1).min() * 100)


def summary(daily=None, bench=None, start=be.FORWARD_START):
    """Strategy / QQQ / SPY from `start`: total return, median weekly return, max drawdown, days. The strategy line is
    equity minus deposits made after the first saved day (deposits are not returns)."""
    daily = _read(DAILY_CSV, parse_dates=["Date"]) if daily is None else daily
    bench = _read(BENCH_CSV, parse_dates=["Date"]) if bench is None else bench
    rows, start = [], pd.Timestamp(start)
    if daily is not None and len(daily):
        d = daily.assign(Date=pd.to_datetime(daily["Date"])).set_index("Date").sort_index().loc[start:]
        if len(d):
            rows.append(("Strategy (live account)", d["Equity"] - (d["Net_Deposits"] - d["Net_Deposits"].iloc[0])))
            if "Shadow_Value" in d:
                rows.append((SHADOW, d["Shadow_Value"]))
    if bench is not None:
        b = bench.assign(Date=pd.to_datetime(bench["Date"])).set_index("Date").sort_index().loc[start:]
        rows += [(f"{s} (comparison, not traded)", b[s]) for s in ("QQQ", "SPY") if s in b]
    out = [{"Series": name, "From": v.dropna().index.min(), "Days": int(v.notna().sum()),
            **dict(zip(["Total return %", "Median weekly return %", "Max drawdown %"], _stats(v)))} for name, v in rows]
    return pd.DataFrame(out, columns=["Series", "From", "Days", "Total return %", "Median weekly return %", "Max drawdown %"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--record", action="store_true", help="save today's row first (reads the live account, GET only)")
    a = p.parse_args(argv)
    if a.record:
        row = record()
        if row:
            print("saved", {k: row[k] for k in ("Date", "Time_CT", "Positions", "Closed_Picks")}, "->",
                  os.path.relpath(DAILY_CSV, ROOT))
    print(summary().round(2).to_string(index=False))


if __name__ == "__main__":
    main()
