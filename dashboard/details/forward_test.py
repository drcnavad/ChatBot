"""Strategy tab: the forward test (paper strategies vs the account, QQQ and SPY) and the paper strategies' rules."""
import pandas as pd
import streamlit as st

from dashboard.data import load_benchmarks, read_report_csv
from dashboard.style import INK, MUTED, PCT_COL, drawdown_tone, esc, live_row, show_html, symbol_link, toned


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
        st.info(f"Forward test started {start}; the first daily row is saved after the close (the daily run's 3:05 PM CT step on trading days, after the signal refresh).")
        return
    board, values = _board(daily, values)
    marked = ""
    if values is not None and "Provisional" in values and values["Provisional"].fillna(False).astype(bool).any():
        marked = (f" {pd.Timestamp(values['Date'].max()):%a %b %-d}: each strategy's holdings from the last saved day marked "
                  "to that day's close (provisional, no trades yet); the daily run's 3:05 PM CT step saves that day with its trades.")
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
               f"starts at 1.0 on the {start} close. Your account = time-weighted (deposits and withdrawals are not returns); QQQ / SPY = closes, "
               "comparison only (not ranked). Weekly = Friday to Friday; None / – = no full week yet." + acct
               + marked + " Saved daily by the daily run's 3:05 PM CT step after the signal refresh (no orders); each strategy's rule and holdings are in the next section.")


CONSENSUS_GOOD, CONSENSUS_WARN, EXIT_RANK = 3, 2, 30   # chip colors: held by 3+ / 2 of the 5; ranked worse than 30


def render_top10(ranks=None):
    """Top 5 consensus card (display only): the 10 stocks most held by the 5 best forward-test strategies
    (forward_test.consensus), same leaderboard as the table above (provisional close included)."""
    import forward_test as ft
    daily, values = read_report_csv(ft.DAILY_CSV), read_report_csv(ft.STRATEGIES_CSV)
    c = None
    if values is not None and not values.empty:
        c = ft.consensus(_board(daily, values)[0], read_report_csv(ft.HOLDINGS_CSV), ranks)
    if c is None or c["stocks"].empty:
        show_html('<div class="sa-card"><div class="sa-card-title">Top 5 consensus<span>no saved strategy holdings yet</span></div></div>')
        return
    n = len(c["strategies"])

    def chip(r):
        cls = ("t-bad" if pd.isna(r.Rank) or r.Rank > EXIT_RANK else
               "t-good" if r.Count >= CONSENSUS_GOOD else "t-warn" if r.Count >= CONSENSUS_WARN else "")
        rk = f"rank #{r.Rank:.0f}" if pd.notna(r.Rank) else "not ranked"
        return (f'<span title="{esc(f"summed target weight {r.Weight:.0%} · {rk} at the latest close")}">'
                f'{symbol_link(r.Symbol, cls)} {r.Count}/{n} strategies</span>')
    strats = " ".join(f'<span>{i}. {esc(nm)}' + (f' <i title="{esc(", ".join(tw))}">(+{len(tw)} same holdings)</i>' if tw else "")
                      + "</span>" for i, (nm, tw) in enumerate(c["strategies"], 1))
    order = "total return since Oct 2" if c["by_return"] else "leaderboard rank"
    show_html(f'<div class="sa-card"><div class="sa-card-title">Top 5 consensus<span>the {len(c["stocks"])} stocks most held by the '
              f'{n} best strategies ({order}) · holdings at the {c["day"]:%a %b %-d} close</span></div>'
              f'<div class="sa-tier"><b>Stocks</b><div class="sa-tier-syms">{" ".join(chip(r) for r in c["stocks"].itertuples())}</div></div>'
              f'<div class="sa-tier"><b>Strategies</b><div class="sa-tier-syms">{strats}</div></div></div>')
    st.caption(("No strategy has a full week yet, so the 5 strategies are the best by total return since Oct 2; from the "
                "first full week they follow the leaderboard rank. " if c["by_return"] else "")
               + "Strategies with the same stocks at the same weights count once (the better-placed one). Order: most "
               "strategies, then summed weight, then latest rank. Green = held by 3+ of them, yellow = 2, red = ranked "
               f"worse than {EXIT_RANK} at the latest close. Display only: updated with the leaderboard, never traded.")


def render_forward_rules(p=None):
    """Strategy tab: the Top 5 consensus card, then each forward-test strategy's one-line rule and its holdings,
    ordered like the leaderboard above (rank, or total return since Oct 2 until the first full week)."""
    import forward_test as ft
    ranks = p.by_symbol["Strategy_Rank"].to_dict() if p is not None and "Strategy_Rank" in p.by_symbol else None
    render_top10(ranks)
    held, h = ft.holdings(), read_report_csv(ft.HOLDINGS_CSV)
    by_name = {c["name"]: c for c in ft.STRATEGIES}
    daily, values = read_report_csv(ft.DAILY_CSV), read_report_csv(ft.STRATEGIES_CSV)
    order = [c["name"] for c in ft.STRATEGIES]
    if values is not None and not values.empty:
        board = _board(daily, values)[0]
        names = board["Strategy"].str.replace(ft.LIVE_MARK, "", regex=False)
        order = [n for n in names if n in by_name] + [n for n in order if n not in set(names)]
    rules = pd.DataFrame([{"Strategy": n, "Rule": by_name[n]["rule"],
                           "Holdings now (target weight)": held.get(n, "cash")} for n in order])
    sty = toned(rules, ["Holdings now (target weight)"], fn=lambda v: MUTED if v == "cash" else INK)
    st.dataframe(live_row(sty, "Strategy", ft.LIVE), hide_index=True, width="stretch")
    if h is not None and len(h):
        st.caption(f"Holdings as of the {pd.Timestamp(h['Date'].max()):%a %b %-d} close (Reports/forward_strategies_holdings.csv "
                   "has every day). Ordered like the leaderboard above. Paper only: none of these is traded.")
