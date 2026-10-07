"""Strategy tab: the last decision (decisions in force, every stock)."""
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.data import read_report_csv
from dashboard.page import open_symbol
from dashboard.settings import CHANGES_CSV, MIDWEEK, N_PICKS, symbol_sector
from dashboard.signals import SIGNALS, plan_text
from dashboard.style import PCT_COL, SCORE_COL, SCORE_GOOD, toned


def render_last_decision(p):
    """The decisions in force, in ONE view: decision dates, then every stock with its signal, reason, ranks, weight before
    and after, and the next rebalance plan (filters: portfolio & changes / watch list / all). Click a row to open the stock."""
    reb_days = p.df.loc[p.df["Rebalance_Day"] == 1, "Date"]
    mw = p.midweek[p.midweek["Event"] == "mid-week check"] if p.midweek is not None else None
    if mw is not None and len(mw):
        last_day = mw["Event_Date"].iloc[-1]
        acts = set(mw.loc[mw["Event_Date"] == last_day, "Action"])
        mw_val = (f"{pd.Timestamp(last_day):%a %b %-d} · "
                  + (" + ".join(x for x, k in (("swap", {"SWAP"}), ("exit", {"REPLACE", "SELL"})) if k & acts) or "no trade"))
    else:
        mw_val = "none yet this week" if MIDWEEK else "off"
    c = st.columns(3)
    c[0].metric("Last weekly rebalance", f"{reb_days.max():%a %b %-d}" if not reb_days.empty else "—")
    c[1].metric("Last mid-week check", mw_val)
    c[2].metric("Next decision", f"{p.next_kind} · {p.next_dec:%a %b %-d}" if p.next_dec is not None else "—")
    today = p.df["Date"].max()
    plan_when = f"{p.plan_day:%a %b %-d}" if p.plan_day is not None else "next"
    rank_then = f"Rank at {p.off_date:%b %-d} decision"
    board = p.board_off.rename(columns={"Rank": rank_then})
    board[rank_then] = board[rank_then].round(0).astype("Int64")
    board.insert(4, "Rank today", board["Symbol"].map(p.by_symbol["Strategy_Rank"]).round(0).astype("Int64"))
    board.insert(5, "Next rebalance plan", [plan_text(p.plan, s) if s in symbol_sector or s in p.plan else "—"
                                            for s in board["Symbol"]])
    board.insert(6, "Rank change", board["Symbol"].map(lambda s: p.rank_change.get(s, (np.nan,))[0]))
    ch = read_report_csv(CHANGES_CSV)
    ok = ch is not None and {"Symbol", "Old_Weight"} <= set(ch.columns)
    # Newest decision first so drop_duplicates keeps the latest Old_Weight per symbol.
    _ch = ch.sort_values("Date", ascending=False) if ok else None
    before = _ch[_ch["Symbol"].notna()].drop_duplicates("Symbol").set_index("Symbol")["Old_Weight"] if ok else {}
    board.insert(board.columns.get_loc("Portfolio weight %"), "Weight before %",
                 board["Symbol"].map(lambda s: before.get(s, np.nan) * 100))
    board.loc[board["Signal"] == "Sold", "Portfolio weight %"] = 0.0      # after the decision a sold stock weighs 0
    counts = board["Signal"].value_counts()
    filters = {"Portfolio & changes": ["Buy", "Hold", "Sold"],
               "Watch list": ["Watch"],
               "All stocks": SIGNALS}
    show = st.radio("Show", list(filters), horizontal=True, key="signals_filter", label_visibility="collapsed")
    part = board[board["Signal"].isin(filters[show])].reset_index(drop=True)
    st.caption(f"Buy {counts.get('Buy', 0)} · Hold {counts.get('Hold', 0)} · "
               f"Sold {counts.get('Sold', 0)} · Watch {counts.get('Watch', 0)} · "
               f"Score below 0 {counts.get('Score below 0', 0)}. {rank_then} = the rank the decision used; Rank today = at "
               f"the {today:%a %b %-d} close (Rank change: + = moved up since the decision day); Weight before % → Portfolio "
               f"weight % = the portfolio before and after the decision; Portfolio slot = position among the {N_PICKS} picks; "
               f"Next rebalance plan = what the {plan_when} rebalance would do at the latest close (the numbers the trade "
               "step uses). Orders for a decision go out that day at 2:30 PM CT. Click a row to open the stock on the "
               "Home tab.")
    shown = toned(part.round({"Score": 1, "Weight before %": 2, "Portfolio weight %": 2}),
                  ["Signal", "Next rebalance plan", "Rank change", "Earnings soon"])
    event = st.dataframe(toned(shown, ["Score"], good=SCORE_GOOD), hide_index=True,
                         width="stretch", on_select="rerun", selection_mode="single-row", key=f"sig_tbl_{show}",
                         column_config={"Why": st.column_config.TextColumn("Why", width="large"), "Score": SCORE_COL,
                                        "Rank change": st.column_config.NumberColumn(format="%+.0f"),
                                        "Weight before %": PCT_COL, "Portfolio weight %": PCT_COL})
    open_symbol(part, event, "signals")
