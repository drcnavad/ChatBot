"""One-off: backtest ALL strategies on the CURRENT universe + do_not_buy for
2025, 2026 YTD, and combined 2025→latest. Appends new rows with a universe note;
leaves older rows untouched. Simulation only.

Usage: python backtest_universe_report.py
"""
from __future__ import annotations
import os, shutil, sys, tempfile, time
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
PLUS15 = [
    ("Worst 3 sectors 14d, +15%/-3%", dict(look=10, tp=0.15, stop=0.03)),
    ("Worst 3 sectors 1M, +15%/-3%", dict(look=21, tp=0.15, stop=0.03)),
]

NOTE_TAG = (
    f"universe {date.today().isoformat()} "
    f"({len([s for s in sm.tradable_symbols if s not in set(sm.do_not_buy)])} buyable; "
    f"+15 added, {len(sm.do_not_buy)} do_not_buy)"
)


def _extra_cfgs():
    return [dict(name=n, worst=w, earnings=None, group="event",
                 rule="Same as +10%/-3% twin but take-profit +15%.")
            for n, w in PLUS15]


def _summarize_from_paths(P, H, start, note):
    rows = []
    v = db.read_csv(P, parse_dates=["Date"]) if db.exists(P) else pd.DataFrame(
        columns=["Date", "Strategy", "Value"])
    h = db.read_csv(H, parse_dates=["Date"]) if db.exists(H) else pd.DataFrame(
        columns=["Date", "Strategy", "Symbol", "Shares", "Price"])
    for n, g in v.groupby("Strategy"):
        s = g.set_index("Date")["Value"].sort_index()
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
        rows.append(dict(
            strategy=n,
            total_return=float((s.iloc[-1] / s.iloc[0] - 1) * 100),
            max_drawdown=float((s / s.cummax() - 1).min() * 100),
            trades=ntrades,
            cost=traded * ft.be.COST * 100,
            start_date=s.index[0].date(),
            end_date=s.index[-1].date(),
            notes=note,
        ))
    # benchmarks
    b = db.read_csv(ft.BENCH_CSV, parse_dates=["Date"]).set_index("Date").sort_index()
    last = v["Date"].max() if len(v) else b.index.max()
    for k in ("QQQ", "SPY"):
        s = b[k][(b.index >= pd.Timestamp(start)) & (b.index <= last)].dropna()
        if len(s) >= 2:
            rows.append(dict(
                strategy=f"{k} buy & hold",
                total_return=float((s.iloc[-1] / s.iloc[0] - 1) * 100),
                max_drawdown=float((s / s.cummax() - 1).min() * 100),
                trades=1, cost=None,
                start_date=s.index[0].date(), end_date=s.index[-1].date(),
                notes=f"benchmark, close to close; {NOTE_TAG}",
            ))
    return rows, v, h


def run_window(start, end, label):
    """Replay every strategy (+ PLUS15) from start through end on a temp DB copy."""
    real = db.db_path()
    tmp = tempfile.mkdtemp()
    os.environ["STOCK_ANALYSIS_DB"] = os.path.join(tmp, "bt.db")
    import sqlite3
    with sqlite3.connect(real) as src, sqlite3.connect(os.environ["STOCK_ANALYSIS_DB"]) as dst:
        src.backup(dst)
    note = (f"simulated; cost = % of starting capital (0.1% per $ traded); {NOTE_TAG}; {label}")
    fails = []
    try:
        ft.be.FORWARD_START = start
        P = os.path.join(ft.REPORTS, f"bt_univ_{label.replace(' ', '_')}.csv")
        H = os.path.join(ft.REPORTS, f"bt_univ_{label.replace(' ', '_')}_hold.csv")
        for p in (P, H):
            if os.path.exists(p):
                os.remove(p)
        bars = ft.forward_bars(fetch=False)
        cut = (lambda df, col="Date": df) if end is None else \
            (lambda df, col="Date": df[pd.to_datetime(df[col]) <= pd.Timestamp(end)])
        data = dict(
            sig=cut(ft._read(ft.SIGNAL_CSV, usecols=ft.SIG_COLS, parse_dates=["Date"])),
            facts=(lambda f: None if f is None else cut(f, "bar_date"))(ft._read(ft.FACTOR_CSV)),
            bench=cut(ft._read(ft.BENCH_CSV)),
            bars=[cut(b) for b in bars],
        )
        cfgs = list(ft.STRATEGIES) + _extra_cfgs()
        keep = ft.STRATEGIES[:]
        fc._cfg = lambda name: next(x for x in keep if x["name"] == name)
        for c in cfgs:
            cfg = c
            if cfg.get("combo", {}).get("start"):
                cfg = {**cfg, "combo": {k: v for k, v in cfg["combo"].items() if k != "start"}}
            t0 = time.time()
            try:
                ba._one(ft, cfg, data, P, H, start)
                print(f"  ok {cfg['name']} ({time.time()-t0:.1f}s)", flush=True)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"[:200]
                fails.append((cfg["name"], msg))
                print(f"  FAIL {cfg['name']}: {msg}", flush=True)
        rows, v, h = _summarize_from_paths(P, H, start, note)
        for n, e in fails:
            if not any(r.get("strategy") == n for r in rows):
                rows.append(dict(strategy=n, notes=f"could not backtest: {e}; {NOTE_TAG}; {label}"))
        # clean temp report files
        for p in (P, H):
            if os.path.exists(p):
                os.remove(p)
    finally:
        os.environ.pop("STOCK_ANALYSIS_DB", None)
        shutil.rmtree(tmp, ignore_errors=True)

    df = pd.DataFrame(rows)
    df = df.sort_values(["total_return", "max_drawdown"], ascending=[False, False],
                        na_position="last").reset_index(drop=True)
    df["rank"] = [i + 1 if pd.notna(r) else None for i, r in enumerate(df.get("total_return", []))]
    df["run_date"] = date.today().isoformat()
    df = df.reindex(columns=COLS)
    return df, fails


def _existing_universe_keys():
    """Keys that already have this universe note (avoid re-inserting same run)."""
    if not db.has_table(TABLE):
        return set()
    df = db.load_table(TABLE)
    tag_prefix = f"universe {date.today().isoformat()}"
    m = df["notes"].astype(str).str.contains(tag_prefix, na=False, regex=False)
    sub = df[m]
    return set(zip(sub["strategy"].astype(str), sub["start_date"].astype(str),
                   sub["end_date"].astype(str), sub["notes"].astype(str)))


def save_new(df):
    if df is None or not len(df):
        return 0
    existing = _existing_universe_keys()
    fresh = []
    for _, r in df.iterrows():
        key = (str(r["strategy"]), str(r["start_date"]), str(r["end_date"]), str(r["notes"]))
        if key in existing:
            continue
        fresh.append(r)
        existing.add(key)
    if not fresh:
        return 0
    out = pd.DataFrame(fresh).reindex(columns=COLS)
    db.save_table(TABLE, out, mode="a", header=not db.has_table(TABLE), index=False)
    return len(out)


def main():
    buyable = [s for s in sm.tradable_symbols if s not in set(sm.do_not_buy)]
    print(f"tradable={len(sm.tradable_symbols)} do_not_buy={len(sm.do_not_buy)} "
          f"buyable={len(buyable)} NOTE_TAG={NOTE_TAG}", flush=True)
    print(f"strategies={len(ft.STRATEGIES)} + {len(PLUS15)} extras + QQQ/SPY", flush=True)

    # latest close in bars
    import sqlite3
    latest = sqlite3.connect(db.db_path()).execute("SELECT MAX(Date) FROM bars_daily").fetchone()[0]
    print(f"bars_daily latest={latest}", flush=True)

    windows = [
        ("2025-01-02", "2025-12-31", "2025 full year"),
        ("2026-01-02", latest, "2026 YTD"),
        ("2025-01-02", latest, "2025-2026 combined"),
    ]
    all_dfs = []
    t0 = time.time()
    for start, end, label in windows:
        print(f"\n=== {label}: {start} → {end} ===", flush=True)
        t1 = time.time()
        df, fails = run_window(start, end, label)
        print(df[["rank", "strategy", "total_return", "max_drawdown", "trades"]].head(15).to_string(index=False),
              flush=True)
        n = save_new(df)
        print(f"  inserted {n}/{len(df)} rows in {time.time()-t1:.1f}s; fails={len(fails)}", flush=True)
        all_dfs.append(df.assign(window=label))

    # Write compact report CSV
    out = pd.concat(all_dfs, ignore_index=True)
    path = os.path.join(ROOT, "Reports", "backtest_universe_2026-10-28.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved {path}. Done in {time.time()-t0:.1f}s", flush=True)

    # Quick top-5 by combined
    comb = all_dfs[2].dropna(subset=["total_return"]).sort_values("total_return", ascending=False)
    print("\nTOP 5 by combined return:")
    print(comb[["rank", "strategy", "total_return", "max_drawdown", "trades"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
