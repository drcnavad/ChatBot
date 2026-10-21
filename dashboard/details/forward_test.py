"""Strategy tab: the forward test (paper strategies vs the account, QQQ and SPY) and the paper strategies' rules."""
import pandas as pd
import streamlit as st

from dashboard.data import load_benchmarks, read_report_csv
from dashboard.style import SPCT, drawdown_tone, esc, show_html, symbol_link, table, toned

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
        st.info(f"No forward-test rows yet (started {start}).")
        return
    raw, values = _board(daily, values)
    render_curve(raw, values)
    board = tidy_board(raw, start[:-6])
    f1, f2 = st.columns([2, 3])
    q = f1.text_input("Search", key="ft_search", placeholder="Search strategies…", label_visibility="collapsed")
    fams = f2.multiselect("Family", sorted(board["Rule"].unique()), key="ft_family", placeholder="All families",
                          label_visibility="collapsed")
    if q:
        board = board[board["Strategy"].str.contains(q, case=False, regex=False)]
    if fams:
        board = board[board["Rule"].isin(fams)]
    pct = [c for c in board.columns if c not in ("Rank", "Strategy", "Rule")]
    sty = toned(toned(board, [c for c in pct if c != "Max drawdown"]), ["Max drawdown"], fn=drawdown_tone)
    table(highlight_rows(sty), fit=True, cols={"Rank": RANK, "Strategy": st.column_config.TextColumn(width="large"),
                                               **{c: SPCT for c in pct},
                                               "Last EOD": st.column_config.NumberColumn(format="%+.2f%%", help="End-of-day close")})

def render_curve(board, values):
    """Interactive growth chart since FORWARD_START: the leaderboard's Rank 1-5, the live strategy, QQQ and SPY by default."""
    import plotly.graph_objects as go
    import forward_test as ft
    from backtest_engine import FORWARD_START
    from dashboard.style import CHART_FONT
    start = pd.Timestamp(FORWARD_START)
    v = values.assign(Date=pd.to_datetime(values["Date"]), Strategy=values["Strategy"].replace(ft.OLD_NAMES))
    v = v[v["Date"] >= start]
    series = {n: g.set_index("Date")["Value"].sort_index() for n, g in v.groupby("Strategy")}
    bench = load_benchmarks()
    if bench is not None:
        b = bench.loc[bench.index >= start]
        series.update({f"{x} (comparison)": b[x].dropna() for x in ("QQQ", "SPY") if x in b})
    names = board["Strategy"].str.replace(ft.LIVE_MARK, "", regex=False)
    ranked = names[board["Rank"].notna()].tolist()
    options = [n for n in names if n in series]
    default = [n for n in ranked[:5] + [ft.LIVE, "QQQ (comparison)", "SPY (comparison)"] if n in series]
    default = list(dict.fromkeys(default))
    pick = st.multiselect("Compare", options, default=default, key="ft_curve",
                          format_func=lambda n: CURRENT if n == ft.LIVE else n, label_visibility="collapsed")
    fig = go.Figure()
    for n in pick:
        y = series[n]
        if y.empty:
            continue
        y = (y / y.iloc[0] - 1) * 100
        live = n == ft.LIVE
        fig.add_trace(go.Scatter(x=y.index, y=y.values, mode="lines", name=CURRENT if live else n,
                                 line=dict(width=3.5 if live else 1.8, dash="dot" if "(comparison)" in n else None),
                                 hovertemplate="%{x|%b %-d}: %{y:+.2f}%<extra>%{fullData.name}</extra>"))
    fig.add_hline(y=0, line=dict(color="#cbd5e1", width=1))
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=90), font=dict(family=CHART_FONT, size=12),
                      plot_bgcolor="#fff", paper_bgcolor="#fff", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2, yanchor="top", x=0, xanchor="left", font=dict(size=11)),
                      yaxis=dict(ticksuffix="%", gridcolor="#f1f5f9", zeroline=False), xaxis=dict(gridcolor="#f1f5f9", tickformat="%b %-d"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

RANK = st.column_config.NumberColumn(format="%d", width="small",
                                    help="Forward-test rank: total return since Oct 2, smaller max drawdown breaks ties")
HIGHLIGHT_BG = "background-color: #e2e8f0; font-weight: 700"   # subtle slate on the white table

CURRENT = "Current Strategy (Live from Oct End)"   # display name of the live strategy's row (forward_test.LIVE stays its id)

def tidy_board(board, start):
    """Leaderboard for display: short column names (Total since <start>, Last EOD, Max drawdown, Wk 1..), integer Rank
    (blank = not ranked), weeks that have not started dropped."""
    import forward_test as ft
    weeks = [c for c in board.columns if c.startswith("Week ") and board[c].notna().any()]
    b = board[["Rank", "Strategy", "Total return %", "Daily return %", *weeks[::-1], "Max drawdown %"]].copy()   # newest week first
    b["Rank"] = pd.array(pd.to_numeric(b["Rank"], errors="coerce"), dtype="Int64")
    names = {"Total return %": f"Total since {start}", "Daily return %": "Last EOD", "Max drawdown %": "Max drawdown",
             **{w: w.replace("Week ", "Wk ").replace(" %", "") for w in weeks}}
    b.insert(2, "Rule", [rule_of(x) for x in b["Strategy"]])
    b["Strategy"] = [CURRENT if str(x).startswith(ft.LIVE) else x for x in b["Strategy"]]   # display name only
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
    on = lambda name: str(name).startswith(ft.LIVE) or str(name) in (*keys, CURRENT)
    return sty.apply(lambda r: [HIGHLIGHT_BG if on(r["Strategy"]) else ""] * len(r), axis=1)

CONSENSUS_GOOD, CONSENSUS_WARN, EXIT_RANK = 3, 2, 30   # chip colors: held by 3+ / 2 of the 5; ranked worse than 30

def render_top10(ranks=None):
    """Top 5 consensus card (display only): the 10 stocks most held by the 5 best forward-test strategies
    (forward_test.consensus): the leaderboard's Rank 1-5 above (total return since Oct 2, provisional close included)."""
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
    show_html(f'<div class="sa-card"><div class="sa-card-title">Top 5 consensus<span>forward-test Rank 1–5 (total return since Oct 2) · holdings at {c["day"]:%a %b %-d} close</span></div>'
              f'<div class="sa-tier"><b>Stocks</b><div class="sa-tier-syms">{" ".join(chip(r) for r in c["stocks"].itertuples())}</div></div>'
              f'<div class="sa-tier"><b>Strategies</b><div class="sa-tier-syms">{strats}</div></div></div>')

def render_forward_rules(p=None):
    """Strategy tab: the Top 5 consensus card, then each forward-test strategy's one-line rule and its holdings,
    ordered like the leaderboard above (total return since Oct 2)."""
    import forward_test as ft
    ranks = p.by_symbol["Strategy_Rank"].to_dict() if p is not None and "Strategy_Rank" in p.by_symbol else None
    render_top10(ranks)
    held = ft.holdings()
    by_name = {c["name"]: c for c in ft.STRATEGIES}
    daily, values = read_report_csv(ft.DAILY_CSV), read_report_csv(ft.STRATEGIES_CSV)
    order, rank = [c["name"] for c in ft.STRATEGIES], {}
    if values is not None and not values.empty:
        board = _board(daily, values)[0]                     # the same leaderboard (and Rank) as the table above
        names = board["Strategy"].str.replace(ft.LIVE_MARK, "", regex=False)
        rank = dict(zip(names, board["Rank"]))
        order = [n for n in names if n in by_name] + [n for n in order if n not in set(names)]
    rules = pd.DataFrame([{"Rank": rank.get(n, pd.NA), "Strategy": CURRENT if n == ft.LIVE else n,
                           "Rule": by_name[n]["rule"], "Holdings now": held.get(n, "cash")} for n in order])
    rules["Rank"] = pd.array(pd.to_numeric(rules["Rank"], errors="coerce"), dtype="Int64")
    q = st.text_input("Search rules", key="rules_search", placeholder="Search strategy, rule or holding…",
                      label_visibility="collapsed")
    if q:
        rules = rules[rules.apply(lambda r: q.lower() in " ".join(map(str, r.values)).lower(), axis=1)]
    table(rules, cols={"Rank": RANK, "Strategy": st.column_config.TextColumn(width="medium"),
                       "Rule": st.column_config.TextColumn(width="large"), "Holdings now": st.column_config.TextColumn(width="large")})
