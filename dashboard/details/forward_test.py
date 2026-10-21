"""Strategy tab: the forward test (paper strategies vs the account, QQQ and SPY) and the paper strategies' rules."""
import pandas as pd
import streamlit as st

from dashboard.data import load_benchmarks, read_report_csv
from dashboard.style import INK, MUTED, drawdown_tone, esc, live_row, show_html, symbol_link, toned

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
    board = tidy_board(board, start[:-6])
    pct = [c for c in board.columns if c not in ("Rank", "Strategy", "Rule")]
    sty = toned(toned(board, [c for c in pct if c != "Max drawdown"]), ["Max drawdown"], fn=drawdown_tone)
    st.dataframe(highlight_rows(sty), hide_index=True, width="stretch",
                 height=35 * (len(board) + 1) + 3,                                                # every row, no scrolling
                 column_config={"Rank": st.column_config.NumberColumn(format="%d", width="small"),
                                "Strategy": st.column_config.TextColumn(width="large"), "Rule": st.column_config.TextColumn(),
                                **{c: st.column_config.NumberColumn(format="%+.2f%%") for c in pct}})
    acct = ""
    if daily is not None and len(daily):
        last = daily.iloc[-1]
        n, w, traded, cost = int(last["Closed_Picks"]), int(last["Winning_Picks"]), float(last["Traded_USD"]), float(last["Cost_USD"])
        acct = (f" Your account as of {last['Date']} {last['Time_CT']} CT: "
                + (f"closed picks {n}, win rate {w / n:.0%}" if n else "no closed picks yet")
                + f", traded \\${traded:,.0f}, cost vs the decision price {'-' if cost < 0 else ''}\\${abs(cost):,.2f}"
                + (f" ({cost / traded * 1e4:+.1f} bps; + = it cost money)" if traded else "") + ".")
    st.caption(f"{ft.RANK_RULE} Every strategy is paper only (never traded): it decides at the "
               "day's close, pays 0.1% per trade side, holds no stock above 20% (the live-rules strategy: the live cap, 15% from Oct 15), invests at most 99% and earns nothing on cash; each "
               f"starts at 1.0 on the {start} close. Your account = time-weighted (deposits and withdrawals are not returns); QQQ / SPY = closes, "
               "comparison only (not ranked). Week k = Friday close to Friday close (week 1 = Oct 2-9); '(so far)' = the week in "
               "progress, Friday's close to the latest close; later weeks are blank." + acct
               + marked + " Saved daily by the daily run's 3:05 PM CT step after the signal refresh (no orders); each strategy's rule and holdings are in the next section.")

HIGHLIGHT_BG = "background-color: #e2e8f0; font-weight: 700"   # subtle slate on the white table

def tidy_board(board, start):
    """Leaderboard for display: short column names (Total since <start>, Today, Max drawdown, Wk 1..), integer Rank
    (blank = not ranked), weeks that have not started dropped."""
    weeks = [c for c in board.columns if c.startswith("Week ") and board[c].notna().any()]
    b = board[["Rank", "Strategy", "Total return %", "Daily return %", *weeks[::-1], "Max drawdown %"]].copy()   # newest week first
    b["Rank"] = pd.array(pd.to_numeric(b["Rank"], errors="coerce"), dtype="Int64")
    names = {"Total return %": f"Total since {start}", "Daily return %": "Today", "Max drawdown %": "Max drawdown",
             **{w: w.replace("Week ", "Wk ").replace(" %", "") for w in weeks}}
    b.insert(2, "Rule", [rule_of(x) for x in b["Strategy"]])
    return b.rename(columns=names)

def rule_of(name):
    """The family a leaderboard row belongs to (forward_test.GROUP_LABEL), Live / Your account / Benchmark otherwise."""
    import forward_test as ft
    name = str(name)
    if name.startswith(ft.LIVE):
        return "Live strategy"
    if name == "My Alpaca account":
        return "Your account"
    g = ft.GROUP_OF.get(name.replace(ft.LIVE_MARK, ""))
    return ft.GROUP_LABEL.get(g, g) if g else "Benchmark"

def highlight_rows(sty):
    """Darker rows: your Alpaca account, the current (live) strategy, Low volatility in uptrend, Sector rotation."""
    import forward_test as ft
    keys = ("My Alpaca account", "Low volatility in uptrend", "Sector rotation")
    on = lambda name: str(name).startswith(ft.LIVE) or str(name) in keys
    return sty.apply(lambda r: [HIGHLIGHT_BG if on(r["Strategy"]) else ""] * len(r), axis=1)

CONSENSUS_GOOD, CONSENSUS_WARN, EXIT_RANK = 3, 2, 30   # chip colors: held by 3+ / 2 of the 5; ranked worse than 30

def render_group_legend():
    """The family colors used on strategy names (forward_test.GROUPS)."""
    import forward_test as ft
    chips = " · ".join(f'<span style="color:{ft.GROUPS[k]};font-weight:700">{esc(ft.GROUP_LABEL[k])}</span>' for k in ft.GROUPS)
    show_html(f'<div class="sa-note">Name color is the kind of rule: {chips}.</div>')

def paint_strategy(sty, extra=()):
    """Color the Strategy column (and `extra` columns) by the strategy's family."""
    import forward_test as ft
    cols = {"Strategy", *extra}

    def css(row):
        color = ft.strategy_color(row["Strategy"])
        return [f"color: {color}; font-weight: 600" if color and col in cols else "" for col in row.index]
    return sty.apply(css, axis=1)

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
    order = "total return since Oct 2" if c["by"] == "Total return %" else c["by"].replace(" %", " return")
    show_html(f'<div class="sa-card"><div class="sa-card-title">Top 5 consensus<span>the {len(c["stocks"])} stocks most held by the '
              f'{n} best strategies ({order}) · holdings at the {c["day"]:%a %b %-d} close</span></div>'
              f'<div class="sa-tier"><b>Stocks</b><div class="sa-tier-syms">{" ".join(chip(r) for r in c["stocks"].itertuples())}</div></div>'
              f'<div class="sa-tier"><b>Strategies</b><div class="sa-tier-syms">{strats}</div></div></div>')
    st.caption("The 5 strategies are the best by the most recent week's return: the week in progress (Friday's close to the "
               "latest close) once it has a day, else the latest finished week; a tie goes to the smaller drawdown. The "
               "leaderboard's Rank (total return since Oct 2) is not used here. "
               + "Strategies with the same stocks at the same weights count once (the better-placed one). Order: most "
               "strategies, then summed weight, then latest rank. Green = held by 3+ of them, yellow = 2, red = ranked "
               f"worse than {EXIT_RANK} at the latest close. Display only: updated with the leaderboard, never traded.")

def render_forward_rules(p=None):
    """Strategy tab: the Top 5 consensus card, then each forward-test strategy's one-line rule and its holdings,
    ordered like the leaderboard above (total return since Oct 2)."""
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
    render_group_legend()
    st.dataframe(live_row(paint_strategy(sty, extra=("Rule",)), "Strategy", ft.LIVE), hide_index=True, width="stretch")
    if h is not None and len(h):
        st.caption(f"Holdings as of the {pd.Timestamp(h['Date'].max()):%a %b %-d} close (Reports/forward_strategies_holdings.csv "
                   "has every day). Ordered like the leaderboard above. Paper only: none of these is traded.")

def render_strategy_guide():
    """Bottom of the Strategy tab: one or two lines for what each paper strategy holds."""
    import forward_test as ft
    guide = pd.DataFrame([{"Strategy": c["name"], "What it does": c["rule"]} for c in ft.STRATEGIES])
    st.caption("The stock count in each line is the book that strategy holds. A defensive sleeve, where a line names one, is held on top of that count.")
    st.dataframe(paint_strategy(guide.style, extra=("What it does",)), hide_index=True, width="stretch")
