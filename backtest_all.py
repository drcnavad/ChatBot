"""Backtest every forward-test strategy (forward_test.STRATEGIES, incl. the live rules and the combos) plus QQQ / SPY
buy-and-hold from START to the latest close, and append the ranked results to the SQLite table `backtest_results`.
Simulated only: no Alpaca calls, no orders. The replay runs against a temporary copy of the database, so the saved
forward-test history is never touched. Usage: python backtest_all.py [--start 2026-01-01]"""
import argparse, os, shutil, sys, tempfile
from datetime import date
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import pandas as pd
import db

TABLE = "backtest_results"
COLS = ["run_date", "start_date", "end_date", "strategy", "total_return", "max_drawdown", "trades", "cost", "rank", "notes"]

def run(start):
    real = db.db_path()
    tmp = tempfile.mkdtemp()
    os.environ["STOCK_ANALYSIS_DB"] = os.path.join(tmp, "bt.db")
    import sqlite3
    with sqlite3.connect(real) as src, sqlite3.connect(os.environ["STOCK_ANALYSIS_DB"]) as dst:
        src.backup(dst)            # consistent copy incl. the WAL
    try:
        import forward_test as ft
        P, H = os.path.join(ft.REPORTS, "bt_strat.csv"), os.path.join(ft.REPORTS, "bt_hold.csv")
        bars, rows, fails = ft.forward_bars(fetch=False), [], []
        for c in list(ft.STRATEGIES):
            try:
                ft.update_strategies(bars=bars, path=P, hold_path=H, start=start, strategies=[c]) \
                    if "strategies" in ft.update_strategies.__code__.co_varnames else _one(ft, c, bars, P, H, start)
            except Exception as e:
                fails.append((c["name"], f"{type(e).__name__}: {e}"[:200]))
        v = db.read_csv(P, parse_dates=["Date"]) if db.exists(P) else pd.DataFrame(columns=["Date", "Strategy", "Value"])
        h = db.read_csv(H, parse_dates=["Date"]) if db.exists(H) else pd.DataFrame(columns=["Date", "Strategy", "Symbol", "Shares", "Price"])
        if os.environ.get("BACKTEST_DUMP"):                    # optional: keep the simulated values / holdings for audits
            v.to_pickle(os.environ["BACKTEST_DUMP"] + "_values.pkl"); h.to_pickle(os.environ["BACKTEST_DUMP"] + "_hold.pkl")
        for n, g in v.groupby("Strategy"):
            s = g.set_index("Date")["Value"].sort_index()
            hh = h[h.Strategy == n]
            sh = hh.pivot_table(index="Date", columns="Symbol", values="Shares", aggfunc="sum").reindex(s.index).fillna(0)
            px = hh.pivot_table(index="Date", columns="Symbol", values="Price", aggfunc="last").reindex(s.index).ffill()
            d = sh.diff(); d.iloc[0] = sh.iloc[0]
            traded = (d.abs() * px).sum().sum()
            rows.append(dict(strategy=n, total_return=(s.iloc[-1] / s.iloc[0] - 1) * 100,
                             max_drawdown=(s / s.cummax() - 1).min() * 100, trades=int((d.abs() > 1e-9).sum().sum()),
                             cost=traded * ft.be.COST * 100, start_date=s.index[0].date(), end_date=s.index[-1].date(),
                             notes="simulated; cost = % of starting capital (0.1% per $ traded)"))
        b = db.read_csv(ft.BENCH_CSV, parse_dates=["Date"]).set_index("Date").sort_index()
        end = v["Date"].max() if len(v) else b.index.max()
        for k in ("QQQ", "SPY"):
            s = b[k][(b.index >= pd.Timestamp(start)) & (b.index <= end)].dropna()
            rows.append(dict(strategy=f"{k} buy & hold", total_return=(s.iloc[-1] / s.iloc[0] - 1) * 100,
                             max_drawdown=(s / s.cummax() - 1).min() * 100, trades=1, cost=None,
                             start_date=s.index[0].date(), end_date=s.index[-1].date(), notes="benchmark, close to close"))
        for n, e in fails:
            rows.append(dict(strategy=n, notes=f"could not backtest: {e}"))
    finally:
        os.environ.pop("STOCK_ANALYSIS_DB", None)
        shutil.rmtree(tmp, ignore_errors=True)
    df = pd.DataFrame(rows)
    df = df.sort_values(["total_return", "max_drawdown"], ascending=[False, False], na_position="last").reset_index(drop=True)
    df["rank"] = [i + 1 if pd.notna(r) else None for i, r in enumerate(df["total_return"])]
    df["run_date"] = date.today().isoformat()
    df = df.reindex(columns=COLS)
    db.save_table(TABLE, df, mode="a", header=not db.has_table(TABLE), index=False)
    return df

def _one(ft, c, bars, P, H, start):
    import forward_combos as fc
    keep = ft.STRATEGIES[:]
    fc._cfg = lambda name: next(x for x in keep if x["name"] == name)   # combos look up their legs in the full list
    ft.STRATEGIES[:] = [c]
    try:
        ft.update_strategies(bars=bars, path=P, hold_path=H, start=start)
    finally:
        ft.STRATEGIES[:] = keep

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--start", default="2026-01-01")
    out = run(a.parse_args().start)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(out.round(2).to_string(index=False))
