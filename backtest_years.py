"""Backtest every forward_test strategy for each calendar year 2017..N and append to backtest_results.

Historical years rebuild signal_analysis from bars_daily (point-in-time Technical_Score + RS_Score).
Strategy_Score = RS_Score (matches post-Oct-13 "now" / worst-sector ranking). No Alpaca orders.
Usage:
  python backtest_years.py                  # 2017-2024 + +15% variants; reuse stored 2025/2026
  python backtest_years.py --also-2025      # also recompute 2025 as sanity check (does not overwrite)
  python backtest_years.py --years 2017,2018
"""
from __future__ import annotations
import argparse, os, shutil, sys, tempfile, time, traceback
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import db
import backtest_engine as be
import sector_mapping as sm
import forward_test as ft
import forward_combos as fc

TABLE = "backtest_results"
COLS = ["run_date", "start_date", "end_date", "strategy", "total_return", "max_drawdown",
        "trades", "cost", "rank", "notes"]
PLUS15 = [
    ("Worst 3 sectors 14d, +15%/-3%", dict(look=10, tp=0.15, stop=0.03)),
    ("Worst 3 sectors 1M, +15%/-3%", dict(look=21, tp=0.15, stop=0.03)),
]


def _year_bounds(y: int, end_cap: str | None = None):
    """Trading-year window: first session on/after Jan 1 through last session of Dec 31 (or end_cap)."""
    start = f"{y}-01-01"
    end = end_cap or f"{y}-12-31"
    return start, end


def build_hist_signal(start="2016-01-01", end=None) -> pd.DataFrame:
    """Point-in-time signal panel from bars_daily for SIG_COLS. Pre-IPO symbols are simply absent."""
    syms = list(sm.tradable_symbols)
    bars = be.apply_history_start(
        be.load_long_bars(syms + list(be.BENCHMARKS) + list(be.SECTOR_ETFS), start=start, end=end)
    )
    if not len(bars):
        raise RuntimeError("bars_daily empty")
    tech = be.build_technical(bars, symbols=syms)
    close_all = be.wide(bars, "Close")
    have = [s for s in syms if s in close_all.columns]
    dates = close_all.index
    elig = be.bool_wide(tech, "eligible", dates, have)
    rs = be.relative_strength(close_all, have, eligible_w=elig)
    regime = be.regime_series(close_all, be.WINNER.get("regime_symbol") or "QQQ")
    tech2 = tech.rename(columns={"rsi": "RSI", "macd_signal": "MACD Signal"})
    keep = ["Date", "Symbol", "Close", "ma_10", "ma_30", "ma_50", "ma_100", "ma_200",
            "RSI", "macd", "MACD Signal", "Technical_Score"]
    rs_long = rs.reindex(columns=have).stack(future_stack=True).rename("RS_Score").reset_index()
    rs_long.columns = ["Date", "Symbol", "RS_Score"]
    sig = tech2[keep].merge(rs_long, on=["Date", "Symbol"], how="left")
    # Post-Oct-13 convention: Strategy_Score is relative strength (worst-sector / "now" ranking).
    sig["Strategy_Score"] = sig["RS_Score"]
    sig["Regime_On"] = sig["Date"].map(regime).fillna(False).astype(int)
    sig["final_trade"] = np.where(sig["Strategy_Score"].fillna(-999) < float(be.WINNER.get("min_score") or 0),
                                  "SELL", "HOLD")
    elig_long = elig.reindex(columns=have).stack(future_stack=True).rename("eligible").reset_index()
    elig_long.columns = ["Date", "Symbol", "eligible"]
    sig = sig.merge(elig_long, on=["Date", "Symbol"], how="left")
    # Warm-up / ineligible: blank Strategy_Score so strategy_inputs eligibility matches live.
    cold = ~sig["eligible"].fillna(False)
    sig.loc[cold, ["Strategy_Score", "RS_Score"]] = np.nan
    sig = sig.drop(columns=["eligible"])
    # Drop symbols that never became eligible (keeps QQQ/ETF-only noise out of the stock panel).
    scored = set(sig.loc[sig["Strategy_Score"].notna(), "Symbol"].unique())
    sig = sig[sig["Symbol"].isin(scored)].reset_index(drop=True)
    return sig, bars


def _cut(df, end, col="Date"):
    if df is None or not len(df):
        return df
    return df[pd.to_datetime(df[col]) <= pd.Timestamp(end)]


def _existing_keys():
    if not db.has_table(TABLE):
        return set()
    df = db.load_table(TABLE)
    return set(zip(df["strategy"].astype(str), df["start_date"].astype(str), df["end_date"].astype(str)))


def _summarize(name, v, h, notes):
    if v is None or not len(v) or name not in set(v["Strategy"]):
        return dict(strategy=name, notes=notes or "no equity curve")
    g = v[v.Strategy == name].set_index("Date")["Value"].sort_index()
    if len(g) < 2:
        return dict(strategy=name, notes="too few days")
    hh = h[h.Strategy == name] if h is not None and len(h) else pd.DataFrame()
    traded = 0.0
    ntrades = 0
    if len(hh):
        sh = hh.pivot_table(index="Date", columns="Symbol", values="Shares", aggfunc="sum").reindex(g.index).fillna(0)
        px = hh.pivot_table(index="Date", columns="Symbol", values="Price", aggfunc="last").reindex(g.index).ffill()
        d = sh.diff(); d.iloc[0] = sh.iloc[0]
        traded = float((d.abs() * px).sum().sum())
        ntrades = int((d.abs() > 1e-9).sum().sum())
    return dict(
        strategy=name,
        total_return=float((g.iloc[-1] / g.iloc[0] - 1) * 100),
        max_drawdown=float((g / g.cummax() - 1).min() * 100),
        trades=ntrades,
        cost=traded * ft.be.COST * 100,
        start_date=g.index[0].date(),
        end_date=g.index[-1].date(),
        notes=notes,
    )


def run_year(y, sig, bars_full, facts, bench, end_cap=None, extra_cfgs=None, note_suffix=""):
    """Replay every STRATEGIES (+ extras) for calendar year y. Returns a ranked DataFrame."""
    start_hint, end = _year_bounds(y, end_cap)
    # First trading day on/after Jan 1 with signal coverage.
    days = sorted(pd.to_datetime(sig["Date"]).unique())
    days = [d for d in days if d >= pd.Timestamp(start_hint) and d <= pd.Timestamp(end)]
    if not days:
        return pd.DataFrame(columns=COLS), [f"{y}: no trading days in signal"]
    start = days[0].strftime("%Y-%m-%d")
    end_s = days[-1].strftime("%Y-%m-%d")

    tmp = tempfile.mkdtemp()
    P, H = os.path.join(tmp, "bt_v.csv"), os.path.join(tmp, "bt_h.csv")
    fails = []
    rows = []
    note = ("simulated from bars_daily-built signal (Strategy_Score=RS); "
            "cost = % of starting capital (0.1% per $ traded)" + note_suffix)

    # Warm-up: signal/bars must include history before `start` (RS 252d, MA200, MIN_BARS).
    sig_y = _cut(sig, end_s)
    bench_y = _cut(bench, end_s)
    facts_y = None if facts is None else _cut(facts, end_s, "bar_date")
    bars_y = [_cut(bars_full, end_s)]

    ft.be.FORWARD_START = start
    cfgs = list(ft.STRATEGIES) + list(extra_cfgs or [])
    keep = ft.STRATEGIES[:]
    fc._cfg = lambda name: next(x for x in keep if x["name"] == name)
    try:
        for c in cfgs:
            cfg = c
            if cfg.get("combo", {}).get("start"):
                cfg = {**cfg, "combo": {k: v for k, v in cfg["combo"].items() if k != "start"}}
            ft.STRATEGIES[:] = [cfg]
            try:
                # Accumulate into P/H like backtest_all (blend legs must already be on disk).
                ft.update_strategies(sig=sig_y, facts=facts_y, bench=bench_y, bars=bars_y,
                                     path=P, hold_path=H, start=start)
            except Exception as e:
                fails.append((cfg["name"], f"{type(e).__name__}: {e}"[:200]))
                rows.append(dict(strategy=cfg["name"], notes=f"could not backtest: {fails[-1][1]}"))
        v = db.read_csv(P, parse_dates=["Date"]) if os.path.exists(P) else pd.DataFrame()
        h = db.read_csv(H, parse_dates=["Date"]) if os.path.exists(H) else pd.DataFrame()
        done = set(v["Strategy"]) if len(v) else set()
        for c in cfgs:
            if c["name"] in done:
                rows.append(_summarize(c["name"], v, h, note))
            elif not any(r.get("strategy") == c["name"] for r in rows):
                rows.append(dict(strategy=c["name"], notes="no equity curve"))
    finally:
        ft.STRATEGIES[:] = keep
        shutil.rmtree(tmp, ignore_errors=True)

    # Benchmarks
    b = bench_y.assign(Date=pd.to_datetime(bench_y["Date"])).set_index("Date").sort_index()
    last = pd.Timestamp(end_s)
    for k in ("QQQ", "SPY"):
        s = b[k][(b.index >= pd.Timestamp(start)) & (b.index <= last)].dropna()
        if len(s) >= 2:
            rows.append(dict(
                strategy=f"{k} buy & hold",
                total_return=float((s.iloc[-1] / s.iloc[0] - 1) * 100),
                max_drawdown=float((s / s.cummax() - 1).min() * 100),
                trades=1, cost=None,
                start_date=s.index[0].date(), end_date=s.index[-1].date(),
                notes="benchmark, close to close",
            ))

    df = pd.DataFrame(rows)
    df = df.sort_values(["total_return", "max_drawdown"], ascending=[False, False],
                        na_position="last").reset_index(drop=True)
    df["rank"] = [i + 1 if pd.notna(r) else None for i, r in enumerate(df.get("total_return", []))]
    df["run_date"] = date.today().isoformat()
    df = df.reindex(columns=COLS)
    return df, fails


def save_new(df, existing):
    """Append only rows whose (strategy, start, end) is not already stored."""
    if df is None or not len(df):
        return 0, existing
    fresh = []
    for _, r in df.iterrows():
        key = (str(r["strategy"]), str(r["start_date"]), str(r["end_date"]))
        if key in existing:
            continue
        fresh.append(r)
        existing.add(key)
    if not fresh:
        return 0, existing
    out = pd.DataFrame(fresh).reindex(columns=COLS)
    db.save_table(TABLE, out, mode="a", header=not db.has_table(TABLE), index=False)
    return len(out), existing


def pivot_table(path_csv, path_md):
    df = db.load_table(TABLE)
    df["total_return"] = pd.to_numeric(df["total_return"], errors="coerce")
    df["max_drawdown"] = pd.to_numeric(df["max_drawdown"], errors="coerce")
    df["year"] = pd.to_datetime(df["start_date"]).dt.year.astype(str)
    # Prefer YTD label for 2026
    end = pd.to_datetime(df["end_date"])
    df.loc[end.dt.year == 2026, "year"] = "2026YTD"
    # One row per strategy/year: keep best (highest return) if duplicates
    df = (df.sort_values("total_return", ascending=False)
            .drop_duplicates(["strategy", "year"], keep="first"))
    years = [str(y) for y in range(2017, 2026)] + ["2026YTD"]
    years = [y for y in years if y in set(df["year"])]
    piv = df.pivot(index="strategy", columns="year", values="total_return").reindex(columns=years)

    # Compounded total over available years (skip NaN years)
    def _comp(row):
        vals = [v for v in row if pd.notna(v)]
        if not vals:
            return np.nan
        x = 1.0
        for v in vals:
            x *= (1 + v / 100.0)
        return (x - 1) * 100.0

    piv["Compounded"] = piv.apply(_comp, axis=1)
    # Worst max-drawdown across years (most negative)
    dd = (df.pivot(index="strategy", columns="year", values="max_drawdown")
            .reindex(columns=years).min(axis=1))
    piv["WorstDD"] = dd
    piv = piv.sort_values("Compounded", ascending=False)
    piv.round(2).to_csv(path_csv)
    # markdown
    lines = ["| Strategy | " + " | ".join(years) + " | Compounded | WorstDD |",
             "|---| " + " | ".join(["---:"] * len(years)) + " | ---: | ---: |"]
    for name, row in piv.iterrows():
        cells = [f"{row[y]:.1f}" if pd.notna(row.get(y)) else "—" for y in years]
        comp = f"{row['Compounded']:.1f}" if pd.notna(row["Compounded"]) else "—"
        wdd = f"{row['WorstDD']:.1f}" if pd.notna(row["WorstDD"]) else "—"
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {comp} | {wdd} |")
    Path(path_md).write_text("\n".join(lines) + "\n")
    return piv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2017,2018,2019,2020,2021,2022,2023,2024")
    ap.add_argument("--also-2025", action="store_true")
    ap.add_argument("--skip-save", action="store_true")
    ap.add_argument("--end-2026", default="2026-10-26")
    args = ap.parse_args()
    years = [int(x) for x in args.years.split(",") if x.strip()]
    if args.also_2025:
        years.append(2025)

    t0 = time.time()
    print(f"building historical signal from bars_daily…", flush=True)
    sig, bars = build_hist_signal(start="2016-01-01", end=None)
    print(f"  signal rows={len(sig):,}  dates={sig.Date.min().date()}→{sig.Date.max().date()}  "
          f"symbols={sig.Symbol.nunique()}  ({time.time()-t0:.1f}s)", flush=True)

    facts = ft._read(ft.FACTOR_CSV)  # sparse; fundamentals/sentiment strategies will be weak pre-2026
    bench = db.load_table("benchmark_prices")
    bench["Date"] = pd.to_datetime(bench["Date"])

    extra = [dict(name=n, worst=w, earnings=None, group="event",
                  rule=f"Same as +10%/-3% twin but take-profit +15%.")
             for n, w in PLUS15]

    existing = _existing_keys()
    inserted = 0
    all_fails = []
    summaries = []

    for y in years:
        end_cap = None
        print(f"\n=== {y} ===", flush=True)
        t1 = time.time()
        # For 2025 sanity, prefer the stored signal_analysis (decision-bar prices) so numbers match.
        if y == 2025:
            sig_use = ft._read(ft.SIGNAL_CSV, usecols=ft.SIG_COLS, parse_dates=["Date"])
            bars_use = ft.forward_bars(fetch=False)
            # merge into one frame for _cut compatibility
            bars_flat = pd.concat(bars_use, ignore_index=True).drop_duplicates(["Symbol", "Date"], keep="last")
            note_suffix = "; sanity recompute from stored signal_analysis"
            df, fails = run_year(y, sig_use, bars_flat, facts, bench, end_cap=end_cap,
                                 extra_cfgs=extra, note_suffix=note_suffix)
        else:
            df, fails = run_year(y, sig, bars, facts, bench, end_cap=end_cap, extra_cfgs=extra)
        print(df[["rank", "strategy", "total_return", "max_drawdown", "trades"]].head(12).to_string(index=False),
              flush=True)
        print(f"  … {len(df)} rows in {time.time()-t1:.1f}s; fails={len(fails)}", flush=True)
        all_fails.extend([(y, n, e) for n, e in fails])
        if not args.skip_save:
            n, existing = save_new(df, existing)
            inserted += n
            print(f"  inserted {n} new rows", flush=True)
        summaries.append(df.assign(year=y))

    # Pivot over everything now in the table (incl. prior 2025/2026)
    out_dir = Path(ROOT) / "reports"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "backtest_years_pivot.csv"
    md_path = out_dir / "backtest_years_pivot.md"
    piv = pivot_table(csv_path, md_path)
    print(f"\nInserted {inserted} rows. Pivot → {csv_path} / {md_path}", flush=True)
    print(f"Top 15 by compounded:\n{piv.head(15).round(1).to_string()}", flush=True)
    if all_fails:
        print("\nFailures:", flush=True)
        for y, n, e in all_fails:
            print(f"  {y} {n}: {e}", flush=True)
    print(f"Done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
