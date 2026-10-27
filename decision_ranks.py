"""Each stock's two ranks at every decision (Chirag, Oct 16, 2026) - Reports/decision_ranks.csv.

One row per stock per decision day: Date, Symbol, Rank_Vol, Rank_Sector.
  Rank_Vol    = the low-volatility half's rank (Strategy_Rank in signal_analysis.csv: calmest first, above the 200-day only).
  Rank_Sector = the sector-rotation half's rank: the C6 score among the stocks of the 3 sectors with the best 63-day ETF
                return (forward_test's "sector_rot" score, via sleeves.sector_inputs); blank outside those sectors.
Written by the trade run right after it saves the sleeve ledger (paper_trade.auto_trade), so the Strategy tab can show the
ranks the decision used next to today's. Display only: nothing here feeds orders."""
import os

import pandas as pd
import db  # noqa: E402  tables live in SQLite (db.py), not CSV files

ROOT = os.path.dirname(os.path.abspath(__file__))
RANKS_CSV = os.environ.get("STOCK_ANALYSIS_DECISION_RANKS") or os.path.join(ROOT, "Reports", "decision_ranks.csv")
SIGNAL_CSV = os.path.join(ROOT, "Reports", "signal_analysis.csv")
COLS = ["Date", "Symbol", "Rank_Vol", "Rank_Sector"]

def ranks_on(day=None, signal_csv=SIGNAL_CSV, inp=None):
    """(date, DataFrame Symbol, Rank_Vol, Rank_Sector) for session `day` (default: the latest in signal_analysis.csv)."""
    import sleeves
    sig = db.read_csv(signal_csv, usecols=["Date", "Symbol", "Strategy_Rank"], parse_dates=["Date"])
    d = sig["Date"].max() if day is None else pd.Timestamp(day).normalize()
    vol = sig[sig["Date"] == d].drop_duplicates("Symbol").set_index("Symbol")["Strategy_Rank"]
    inp = inp or sleeves.sector_inputs()
    sec = pd.Series(dtype=float)
    if d in inp["close"].index:
        s = inp["scores"]["sector_rot"].loc[d].where(inp["eligible"].loc[d])
        sec = s[s > 0].rank(ascending=False, method="first")
    syms = sorted(set(vol.index) | set(sec.index))
    out = pd.DataFrame({"Symbol": syms, "Rank_Vol": [vol.get(x) for x in syms], "Rank_Sector": [sec.get(x) for x in syms]})
    for c in ("Rank_Vol", "Rank_Sector"):
        out[c] = pd.to_numeric(out[c], errors="coerce").round().astype("Int64")
    return d, out

def save(day=None, path=None, **kw):
    """Write (replace) the rows of decision day `day` in Reports/decision_ranks.csv. Returns the path."""
    path = path or RANKS_CSV
    d, r = ranks_on(day, **kw)
    r.insert(0, "Date", d.date().isoformat())
    old = db.read_csv(path, dtype={"Date": str}) if db.exists(path) else None
    keep = old[old["Date"] != d.date().isoformat()] if old is not None else None
    out = pd.concat([keep, r], ignore_index=True) if keep is not None and len(keep) else r
    out[COLS].pipe(db.to_csv, path, index=False)        # one SQLite transaction (all or nothing)
    return path

def load(day, path=None):
    """DataFrame Symbol, Rank_Vol, Rank_Sector saved for decision day `day`, or None when that day was not saved."""
    path = path or RANKS_CSV
    if not db.exists(path):
        return None
    r = db.read_csv(path, dtype={"Date": str})
    r = r[r["Date"] == pd.Timestamp(day).date().isoformat()]
    if not len(r):
        return None
    r = r.drop(columns="Date").reset_index(drop=True)
    for c in ("Rank_Vol", "Rank_Sector"):
        r[c] = pd.to_numeric(r[c], errors="coerce").round().astype("Int64")
    return r
