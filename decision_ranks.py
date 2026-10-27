"""Each stock's two ranks at every decision - Reports/decision_ranks.csv (Oct 27, 2026: the two live halves).

One row per stock per decision day: Date, Symbol, Rank_RS_Short, Rank_Friday.
  Rank_RS_Short = the "RS short-term (weekly + Mon–Thu sells)" half's rank (sleeves.ranking: eligible, score above 0).
  Rank_Friday   = the "RS standard (Fridays only)" half's rank (21/63/126-day relative strength).
Written by the trade run right after it saves the sleeve ledger (paper_trade.auto_trade). Display only."""
import os

import pandas as pd
import db  # noqa: E402  tables live in SQLite (db.py), not CSV files

ROOT = os.path.dirname(os.path.abspath(__file__))
RANKS_CSV = os.environ.get("STOCK_ANALYSIS_DECISION_RANKS") or os.path.join(ROOT, "Reports", "decision_ranks.csv")
SIGNAL_CSV = os.path.join(ROOT, "Reports", "signal_analysis.csv")
COLS = ["Date", "Symbol", "Rank_RS_Short", "Rank_Friday"]
RANK_COL = {"rs_short": "Rank_RS_Short", "friday": "Rank_Friday"}

def ranks_on(day=None, signal_csv=SIGNAL_CSV, inp=None):
    """(date, DataFrame Symbol, Rank_RS_Short, Rank_Friday) for session `day` (default: the latest saved session)."""
    import sleeves
    inp = inp or sleeves.inputs(signal_csv)
    idx = inp["close"].index
    d = idx[-1] if day is None else pd.Timestamp(day).normalize()
    r = {h: (sleeves.ranking(h, d, inp).rank(ascending=False, method="first") if d in idx else pd.Series(dtype=float))
         for h in sleeves.HALVES}
    syms = sorted(set().union(*[set(x.index) for x in r.values()]))
    out = pd.DataFrame({"Symbol": syms, **{RANK_COL[h]: [r[h].get(x) for x in syms] for h in sleeves.HALVES}})
    for c in COLS[2:]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round().astype("Int64")
    return d, out

def save(day=None, path=None, **kw):
    """Write (replace) the rows of decision day `day` in Reports/decision_ranks.csv. Returns the path."""
    path = path or RANKS_CSV
    d, r = ranks_on(day, **kw)
    r.insert(0, "Date", d.date().isoformat())
    old = db.read_csv(path, dtype={"Date": str}) if db.exists(path) else None
    if old is not None and not set(COLS) <= set(old.columns):   # the pre-Oct 27 Rank_Vol / Rank_Sector file
        old = None
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
    if not set(COLS) <= set(r.columns):
        return None
    r = r[r["Date"] == pd.Timestamp(day).date().isoformat()]
    if not len(r):
        return None
    r = r.drop(columns="Date").reset_index(drop=True)
    for c in COLS[2:]:
        r[c] = pd.to_numeric(r[c], errors="coerce").round().astype("Int64")
    return r
