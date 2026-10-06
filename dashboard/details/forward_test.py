"""Strategy tab: the forward test (paper strategies vs the account, QQQ and SPY) and the paper strategies' rules."""
import pandas as pd
import streamlit as st

from dashboard.data import load_benchmarks, read_report_csv
from dashboard.style import INK, MUTED, PCT_COL, drawdown_tone, live_row, toned


def _board(daily, values):
    """(leaderboard, values) with the provisional close rows (forward_test.provisional), as shown on the dashboard."""
    import forward_test as ft
    bench = load_benchmarks()
    values, bench = ft.provisional(values, bench.reset_index() if bench is not None else None)
    return ft.leaderboard(daily, bench, values), values


def render_forward_test():
    """Strategy tab: the forward-test leaderboard from FORWARD_START (forward_test.py): every paper strategy
    (forward_test.STRATEGIES, Reports/forward_strategies.csv), the real account (Reports/forward_test_daily.csv), QQQ, SPY."""
    import forward_test as ft
    from backtest_engine import FORWARD_START
    start = f"{pd.Timestamp(FORWARD_START):%b %-d, %Y}"
    daily, values = read_report_csv(ft.DAILY_CSV), read_report_csv(ft.STRATEGIES_CSV)
    if (daily is None or daily.empty) and (values is None or values.empty):
        st.info(f"Forward test started {start}; the first daily row is saved after the close (4:15 PM CT on trading days).")
        return
    board, values = _board(daily, values)
    marked = ""
    if values is not None and "Provisional" in values and values["Provisional"].fillna(False).astype(bool).any():
        marked = (f" {pd.Timestamp(values['Date'].max()):%a %b %-d}: each strategy's holdings from the last saved day marked "
                  "to that day's close (provisional, no trades yet); the 4:15 PM CT job saves that day with its trades.")
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
               + marked + " Saved by the 4:15 PM CT job (no orders); each strategy's rule and holdings are in the next section.")


def _top10_stocks(board, held):
    """'SYM (count, avg%)' for the 10 most common stocks held by the rank 1-10 strategies.
    `board` is the raw leaderboard (numeric Rank); `held` maps strategy name -> 'SYM 14.2%, ...'.
    Returns '–' when no strategy is ranked yet."""
    import forward_test as ft
    ranked = board[board["Rank"].notna() & (board["Rank"] <= 10)]
    counts, weights = {}, {}
    for strat in ranked["Strategy"].str.replace(ft.LIVE_MARK, "", regex=False):
        for part in held.get(strat, "").split(", "):
            if " " not in part:
                continue
            sym, pct = part.rsplit(" ", 1)
            try:
                w = float(pct.rstrip("%")) / 100
            except ValueError:
                continue
            counts[sym] = counts.get(sym, 0) + 1
            weights[sym] = weights.get(sym, 0.0) + w
    if not counts:
        return "–"
    top = sorted(counts, key=lambda s: (-counts[s], -weights[s] / counts[s]))[:10]
    return ", ".join(f"{s} ({counts[s]}, {weights[s] / counts[s]:.0%})" for s in top)


def render_forward_rules():
    """Strategy tab: each forward-test strategy's one-line rule and its holdings on the latest saved day."""
    import forward_test as ft
    held, h = ft.holdings(), read_report_csv(ft.HOLDINGS_CSV)
    rules = pd.DataFrame([{"Strategy": c["name"], "Rule": c["rule"], "Holdings now (target weight)": held.get(c["name"], "cash")}
                          for c in ft.STRATEGIES])
    daily, values = read_report_csv(ft.DAILY_CSV), read_report_csv(ft.STRATEGIES_CSV)
    top10 = "–"
    if daily is not None and values is not None and not daily.empty and not values.empty:
        top10 = _top10_stocks(_board(daily, values)[0], held)   # same ranks as the leaderboard above
    top_row = pd.DataFrame([{"Strategy": "Top 10",
                             "Rule": "Most common stocks across the rank 1-10 strategies (times held, avg target weight).",
                             "Holdings now (target weight)": top10}])
    rules = pd.concat([top_row, rules], ignore_index=True)
    sty = toned(rules, ["Holdings now (target weight)"], fn=lambda v: MUTED if v == "cash" else INK)
    st.dataframe(live_row(sty, "Strategy", ft.LIVE), hide_index=True, width="stretch")
    if h is not None and len(h):
        st.caption(f"Holdings as of the {pd.Timestamp(h['Date'].max()):%a %b %-d} close (Reports/forward_strategies_holdings.csv "
                   "has every day). Paper only: none of these is traded.")
