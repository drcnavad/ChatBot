"""Details tab: the forward test (paper strategies vs the account, QQQ and SPY) and the paper strategies' rules."""
import pandas as pd
import streamlit as st

from dashboard.data import load_benchmarks, read_report_csv
from dashboard.style import INK, MUTED, PCT_COL, drawdown_tone, live_row, toned


def render_forward_test():
    """Details tab: the forward-test leaderboard from FORWARD_START (forward_test.py): every paper strategy
    (forward_test.STRATEGIES, Reports/forward_strategies.csv), the real account (Reports/forward_test_daily.csv), QQQ, SPY."""
    import forward_test as ft
    from backtest_engine import FORWARD_START
    start = f"{pd.Timestamp(FORWARD_START):%b %-d, %Y}"
    daily, values = read_report_csv(ft.DAILY_CSV), read_report_csv(ft.STRATEGIES_CSV)
    if (daily is None or daily.empty) and (values is None or values.empty):
        st.info(f"Forward test started {start}; the first daily row is saved after the close (4:15 PM CT on trading days).")
        return
    bench = load_benchmarks()
    board = ft.leaderboard(daily, bench.reset_index() if bench is not None else None, values)
    board = board.rename(columns={"Total return %": f"Total return since {start[:-6]} %"}).assign(
        Rank=lambda b: b["Rank"].map(lambda r: "–" if pd.isna(r) else str(r)))   # – = not ranked (comparison / no full week yet)
    sty = toned(toned(board, [c for c in board.columns if "return" in c]), ["Max drawdown %"], fn=drawdown_tone)
    st.dataframe(live_row(sty, "Strategy", ft.LIVE), hide_index=True, width="stretch",
                 height=35 * (len(board) + 1) + 3,                                                # every row, no scrolling
                 column_config={c: PCT_COL for c in board.columns if c.endswith("%")})
    acct = ""
    if daily is not None and len(daily):
        last = daily.iloc[-1]
        n, w, traded, cost = int(last["Closed_Picks"]), int(last["Winning_Picks"]), float(last["Traded_USD"]), float(last["Cost_USD"])
        acct = (f" Your account as of {last['Date']} {last['Time_CT']} CT: "
                + (f"closed picks {n}, win rate {w / n:.0%}" if n else "no closed picks yet")
                + f", traded \\${traded:,.0f}, cost vs the decision price {'-' if cost < 0 else ''}\\${abs(cost):,.2f}"
                + (f" ({cost / traded * 1e4:+.1f} bps; + = it cost money)" if traded else "") + ".")
    st.caption(f"**{ft.verdict(board)}** {ft.RANK_RULE} Every strategy is paper only (never traded): it decides at the "
               "day's close, pays 0.1% per trade side, holds no stock above 20%, invests at most 99% and earns nothing on cash; each "
               f"starts at 1.0 on the {start} close. Your account = equity net of new deposits; QQQ / SPY = closes, "
               "comparison only (not ranked). Weekly = Friday to Friday; None / – = no full week yet." + acct
               + " Saved by the 4:15 PM CT job (no orders); each strategy's rule and holdings are in the next section.")


def render_forward_rules():
    """Details tab: each forward-test strategy's one-line rule and its holdings on the latest saved day."""
    import forward_test as ft
    held, h = ft.holdings(), read_report_csv(ft.HOLDINGS_CSV)
    rules = pd.DataFrame([{"Strategy": c["name"], "Rule": c["rule"], "Holdings now (target weight)": held.get(c["name"], "cash")}
                          for c in ft.STRATEGIES])
    sty = toned(rules, ["Holdings now (target weight)"], fn=lambda v: MUTED if v == "cash" else INK)
    st.dataframe(live_row(sty, "Strategy", ft.LIVE), hide_index=True, width="stretch")
    if h is not None and len(h):
        st.caption(f"Holdings as of the {pd.Timestamp(h['Date'].max()):%a %b %-d} close (Reports/forward_strategies_holdings.csv "
                   "has every day). Paper only: none of these is traded.")
