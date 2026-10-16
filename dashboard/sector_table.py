"""Home tab, bottom: sector performance over 14 days, 1, 2 and 3 months (Chirag, Oct 16, 2026).

Each sector (sector_mapping.symbol_sector) = the equal-weighted average of its tradable stocks' price returns, from the
closes in Reports/signal_analysis.csv: the latest close vs the last close on or before that many calendar days / months
earlier (a stock without a close that far back is left out of that column). Display only. (The sector-rotation half
ranks sectors by their ETF's 63-day return instead - forward_test.strategy_inputs; this table is per stock.)"""
import pandas as pd
import streamlit as st

from dashboard.settings import symbol_sector
from dashboard.style import PCT_COL, section, toned

try:
    from sector_mapping import tradable_symbols
except Exception:
    tradable_symbols = []

WINDOWS = {"14 days %": pd.DateOffset(days=14), "1 month %": pd.DateOffset(months=1),
           "2 months %": pd.DateOffset(months=2), "3 months %": pd.DateOffset(months=3)}

def sector_returns(df, symbols=None):
    """DataFrame Sector, Stocks, one column per WINDOWS entry (%), sorted by the 1-month return (best first)."""
    symbols = set(symbols or tradable_symbols)
    c = (df[df["Symbol"].isin(symbols)].pivot_table(index="Date", columns="Symbol", values="Close", aggfunc="last")
         .sort_index().ffill())
    last = c.index.max()
    rets = {}
    for col, off in WINDOWS.items():
        past = c[c.index <= last - off]
        rets[col] = (c.loc[last] / past.iloc[-1] - 1) * 100 if len(past) else pd.Series(dtype=float)
    r = pd.DataFrame(rets)
    r["Sector"] = [symbol_sector.get(s, "—") for s in r.index]
    out = r.groupby("Sector").agg(Stocks=("1 month %", "size"), **{k: (k, "mean") for k in WINDOWS})
    return out.reset_index().sort_values("1 month %", ascending=False, na_position="last").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def _cached(_df, mtime):
    return sector_returns(_df)

def render_sector_table(p):
    section("Sector performance")
    t = _cached(p.df, p.mtime)
    st.dataframe(toned(t.round(2), list(WINDOWS)), hide_index=True, width="stretch",
                 column_config={k: PCT_COL for k in WINDOWS})
    st.caption(f"Each sector = the average price return of its stocks (equal weight), to the {p.df['Date'].max():%a %b %-d} "
               "close. Sorted by the 1-month return.")
