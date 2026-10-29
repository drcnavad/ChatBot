"""Share-based $100k backtest of every forward-test strategy from 2025-01-02 (first session on/after
2025-01-01) through the latest bars_daily close, on the CURRENT universe + do_not_buy (same as
backtest_universe_report). Simulated only — no orders. Appends ranked rows to backtest_results.

Usage: python backtest_100k_from_2025.py
"""
from __future__ import annotations
import os, shutil, sys, tempfile, time, sqlite3
from datetime import date

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import db
import sector_mapping as sm
import forward_test as ft
import forward_combos as fc
import backtest_all as ba

TABLE = "backtest_results"
COLS = ba.COLS
START_CASH = 100_000.0
START = "2025-01-02"   # first NYSE session on/after 2025-01-01
LIVE = "RS short-term (weekly + Mon–Thu sells)"
NOTE_TAG = (
    f"$100k share-based from {START}; universe {date.today().isoformat()} "
    f"({len([s for s in sm.tradable_symbols if s not in set(sm.do_not_buy)])} buyable; "
    f"{len(sm.do_not_buy)} do_not_buy; ALAB+MRVL unblocked); same as backtest_universe_report"
)


def _yearly(s: pd.Series):
    """Return (ret_2025_pct, ret_2026_ytd_pct) from a Value series starting in 2025."""
    s = s.sort_index()
    y25 = s[s.index <= "2025-12-31"]
    y26 = s[s.index >= "2026-01-01"]
    r25 = float(y25.iloc[-1] / y25.iloc[0] - 1) * 100 if len(y25) >= 2 else float("nan")
    # 2026 YTD: from first 2026 close in the book (or last 2025 close as basis)
    if len(y26) >= 1 and len(s) >= 2:
        base = y25.iloc[-1] if len(y25) else s.iloc[0]
        r26 = float(y26.iloc[-1] / base - 1) * 100
    else:
        r26 = float("nan")
    return r25, r26


def run():
    real = db.db_path()
    latest = sqlite3.connect(real).execute("SELECT MAX(Date) FROM bars_daily").fetchone()[0]
    tmp = tempfile.mkdtemp()
    os.environ["STOCK_ANALYSIS_DB"] = os.path.join(tmp, "bt.db")
    with sqlite3.connect(real) as src, sqlite3.connect(os.environ["STOCK_ANALYSIS_DB"]) as dst:
        src.backup(dst)
    note = f"simulated; cost = % of starting capital (0.1% per $ traded); {NOTE_TAG}; 2025→{latest}"
    fails = []
    P = os.path.join(ft.REPORTS, "bt_100k_2025_values.csv")
    H = os.path.join(ft.REPORTS, "bt_100k_2025_hold.csv")
    for p in (P, H):
        if os.path.exists(p):
            os.remove(p)
    try:
        ft.START_CASH = START_CASH
        ft.be.FORWARD_START = START
        bars = ft.forward_bars(fetch=False)
        cut = lambda df, col="Date": df[pd.to_datetime(df[col]) <= pd.Timestamp(latest)]
        data = dict(
            sig=cut(ft._read(ft.SIGNAL_CSV, usecols=ft.SIG_COLS, parse_dates=["Date"])),
            facts=(lambda f: None if f is None else cut(f, "bar_date"))(ft._read(ft.FACTOR_CSV)),
            bench=cut(ft._read(ft.BENCH_CSV)),
            bars=[cut(b) for b in bars],
        )
        keep = ft.STRATEGIES[:]
        fc._cfg = lambda name: next(x for x in keep if x["name"] == name)
        for c in list(ft.STRATEGIES):
            cfg = c
            if cfg.get("combo", {}).get("start"):
                cfg = {**cfg, "combo": {k: v for k, v in cfg["combo"].items() if k != "start"}}
            t0 = time.time()
            try:
                ba._one(ft, cfg, data, P, H, START)
                print(f"  ok {cfg['name']} ({time.time()-t0:.1f}s)", flush=True)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"[:200]
                fails.append((cfg["name"], msg))
                print(f"  FAIL {cfg['name']}: {msg}", flush=True)

        v = db.read_csv(P, parse_dates=["Date"]) if db.exists(P) else pd.DataFrame(
            columns=["Date", "Strategy", "Value"])
        h = db.read_csv(H, parse_dates=["Date"]) if db.exists(H) else pd.DataFrame(
            columns=["Date", "Strategy", "Symbol", "Shares", "Price"])
        rows = []
        detail = []
        for n, g in v.groupby("Strategy"):
            s = g.set_index("Date")["Value"].sort_index().astype(float)
            hh = h[h.Strategy == n]
            traded = 0.0
            ntrades = 0
            if len(hh):
                sh = hh.pivot_table(index="Date", columns="Symbol", values="Shares",
                                    aggfunc="sum").reindex(s.index).fillna(0)
                px = hh.pivot_table(index="Date", columns="Symbol", values="Price",
                                    aggfunc="last").reindex(s.index).ffill()
                d = sh.diff(); d.iloc[0] = sh.iloc[0]
                traded = float((d.abs() * px).sum().sum())
                ntrades = int((d.abs() > 1e-9).sum().sum())
            total = float(s.iloc[-1] / s.iloc[0] - 1) * 100
            r25, r26 = _yearly(s)
            end_val = float(s.iloc[-1])
            rows.append(dict(
                strategy=n, total_return=total, max_drawdown=float((s / s.cummax() - 1).min() * 100),
                trades=ntrades, cost=traded * ft.be.COST * 100,
                start_date=s.index[0].date(), end_date=s.index[-1].date(), notes=note,
            ))
            detail.append(dict(strategy=n, end_value=end_val, total_return=total,
                               ret_2025=r25, ret_2026_ytd=r26,
                               max_drawdown=float((s / s.cummax() - 1).min() * 100), trades=ntrades))

        b = db.read_csv(ft.BENCH_CSV, parse_dates=["Date"]).set_index("Date").sort_index()
        last = v["Date"].max() if len(v) else b.index.max()
        for k in ("QQQ", "SPY"):
            s = b[k][(b.index >= pd.Timestamp(START)) & (b.index <= last)].dropna().astype(float)
            if len(s) < 2:
                continue
            # $100k buy & hold: shares = START_CASH / first close
            eq = START_CASH * (s / s.iloc[0])
            total = float(eq.iloc[-1] / eq.iloc[0] - 1) * 100
            r25, r26 = _yearly(eq)
            rows.append(dict(
                strategy=f"{k} buy & hold", total_return=total,
                max_drawdown=float((eq / eq.cummax() - 1).min() * 100), trades=1, cost=None,
                start_date=s.index[0].date(), end_date=s.index[-1].date(),
                notes=f"benchmark $100k close-to-close; {NOTE_TAG}; 2025→{latest}",
            ))
            detail.append(dict(strategy=f"{k} buy & hold", end_value=float(eq.iloc[-1]),
                               total_return=total, ret_2025=r25, ret_2026_ytd=r26,
                               max_drawdown=float((eq / eq.cummax() - 1).min() * 100), trades=1))
        for n, e in fails:
            rows.append(dict(strategy=n, notes=f"could not backtest: {e}; {NOTE_TAG}"))
    finally:
        os.environ.pop("STOCK_ANALYSIS_DB", None)
        shutil.rmtree(tmp, ignore_errors=True)

    df = pd.DataFrame(rows)
    df = df.sort_values(["total_return", "max_drawdown"], ascending=[False, False],
                        na_position="last").reset_index(drop=True)
    df["rank"] = [i + 1 if pd.notna(r) else None for i, r in enumerate(df["total_return"])]
    df["run_date"] = date.today().isoformat()
    df = df.reindex(columns=COLS)

    det = pd.DataFrame(detail)
    det = det.merge(df[["strategy", "rank"]], on="strategy", how="left")
    det = det.sort_values("end_value", ascending=False, na_position="last").reset_index(drop=True)
    det["end_rank"] = range(1, len(det) + 1)

    out_csv = os.path.join(ROOT, "Reports", f"backtest_100k_2025_{date.today().isoformat()}.csv")
    det.to_csv(out_csv, index=False)
    # keep value/hold dumps for audit
    print(f"detail CSV {out_csv}", flush=True)
    return df, det, latest, fails


def main():
    buyable = [s for s in sm.tradable_symbols if s not in set(sm.do_not_buy)]
    print(f"START_CASH={START_CASH:.0f} START={START} tradable={len(sm.tradable_symbols)} "
          f"do_not_buy={len(sm.do_not_buy)} buyable={len(buyable)} strategies={len(ft.STRATEGIES)}",
          flush=True)
    print(f"do_not_buy={sorted(sm.do_not_buy)}", flush=True)
    print(f"ALAB/MRVL blocked? ALAB={'ALAB' in sm.do_not_buy} MRVL={'MRVL' in sm.do_not_buy}", flush=True)
    t0 = time.time()
    df, det, latest, fails = run()
    db.save_table(TABLE, df, mode="a", header=not db.has_table(TABLE), index=False)
    print(f"\nInserted {len(df)} rows into {TABLE} in {time.time()-t0:.1f}s; fails={len(fails)}", flush=True)
    print(f"\n=== Ranked by ending $ (start ${START_CASH:,.0f}, {START} → {latest}) ===", flush=True)
    show = det.head(10)[["end_rank", "strategy", "end_value", "total_return", "ret_2025",
                         "ret_2026_ytd", "max_drawdown", "trades"]]
    print(show.to_string(index=False), flush=True)
    live = det[det["strategy"] == LIVE]
    if len(live):
        r = live.iloc[0]
        print(f"\nLIVE {LIVE}: end_rank={int(r.end_rank)} end=${r.end_value:,.0f} "
              f"total={r.total_return:.2f}% 2025={r.ret_2025:.2f}% 2026YTD={r.ret_2026_ytd:.2f}% "
              f"maxDD={r.max_drawdown:.2f}% trades={int(r.trades)}", flush=True)
    for name in ("RS standard (Fridays only)", LIVE):
        sub = det[det["strategy"] == name]
        if len(sub):
            print(f"  reconcile {name}: combined {sub.iloc[0].total_return:.2f}%", flush=True)


if __name__ == "__main__":
    main()
