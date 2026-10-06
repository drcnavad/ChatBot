"""Trading Account tab: latest signals (every stock at the latest close + the next-rebalance plan)."""
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.history import daily_status
from dashboard.page import open_symbol
from dashboard.settings import symbol_sector
from dashboard.signals import plan_text
from dashboard.style import PCT_COL, SCORE_COL, SCORE_GOOD, num, toned


def latest_signals_title(p):
    return f"Latest signals · {p.df['Date'].max():%a %b %-d} close"


def render_latest_signals(p):
    """Every stock at the LATEST close (signal_analysis.csv) + the next-rebalance plan (strategy_picks.csv)."""
    day = p.df["Date"].max()
    la = p.latest[p.latest["Date"] == day].set_index("Symbol")
    plan_when = f"{p.plan_day:%a %b %-d}" if p.plan_day is not None else "next"
    decided_today = pd.Timestamp(p.off_date) == pd.Timestamp(day)
    rows = []
    for sym, r in la.iterrows():
        w = num(r.get("Strategy_Weight")) or 0.0
        rows.append({"Symbol": sym,
                     "Signal today": p.sig_off.get(sym, "Not ranked") if decided_today else daily_status(w, r.get("Strategy_Score")),
                     "Rank today": r.get("Strategy_Rank"), "Score today": r.get("Strategy_Score"),
                     f"{plan_when} plan": plan_text(p.plan, sym) if sym in symbol_sector or sym in p.plan else "—",
                     "Held now %": w * 100 if w > 0 else np.nan, "Sector": symbol_sector.get(sym, "—")})
    table = pd.DataFrame(rows).sort_values(["Rank today", "Symbol"], na_position="last").reset_index(drop=True)
    table["Rank today"] = table["Rank today"].round(0).astype("Int64")
    st.caption(f"Every stock at the {day:%a %b %-d} close (Reports/signal_analysis.csv) · Signal today = "
               + ("the decision made at this close" if decided_today else
                  "no decision today: Hold = in the portfolio, Watch = ranked but not held, Score below 0 = not eligible")
               + f" · {plan_when} plan = what the {plan_when} full rebalance would do at this close (Reports/strategy_picks.csv, "
               "the numbers the trade step uses; final at that close). Click a row to open the stock.")
    table = table.round({"Score today": 1, "Held now %": 2})
    event = st.dataframe(toned(toned(table, ["Signal today", f"{plan_when} plan"]), ["Score today"], good=SCORE_GOOD),
                         hide_index=True, width="stretch",
                         on_select="rerun", selection_mode="single-row", key="latest_signals_tbl",
                         column_config={"Score today": SCORE_COL, "Held now %": PCT_COL})
    open_symbol(table, event, "latest_signals")
