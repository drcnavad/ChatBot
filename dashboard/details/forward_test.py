"""Strategy tab: the forward test (paper strategies vs the account, QQQ and SPY) and the paper strategies' rules."""
import pandas as pd
import streamlit as st

from dashboard.data import load_benchmarks, read_report_csv
from dashboard.style import drawdown_tone, esc, show_html, symbol_link, toned

@st.cache_data(ttl=900, max_entries=4, show_spinner=False)
def _board_cached(_daily, _values, key):
    import forward_test as ft
    bench = load_benchmarks()
    values, bench = ft.provisional(_values, bench.reset_index() if bench is not None else None)
    return ft.leaderboard(_daily, bench, values), values

def _forward_files():
    """(daily, values, (leaderboard, values with the provisional close rows) or None). The leaderboard is built once per
    file version and 5 minutes (the provisional closes count from 3:05 PM CT), shared by every panel of the tab."""
    import time
    import forward_test as ft
    from dashboard.data import mtime
    from dashboard.settings import BENCH_CSV
    daily, values = read_report_csv(ft.DAILY_CSV), read_report_csv(ft.STRATEGIES_CSV)
    if values is None or values.empty:
        return daily, values, None
    key = (*(mtime(f) for f in (ft.DAILY_CSV, ft.STRATEGIES_CSV, BENCH_CSV, ft.HOLDINGS_CSV, ft.FWD_BARS)), int(time.time() // 300))
    return daily, values, _board_cached(daily, values, key)

@st.fragment(run_every=300)      # re-reads its (mtime-cached) files: new forward-test days show up in place
def render_forward_test(p=None):
    """Strategy tab: the forward-test leaderboard from FORWARD_START (forward_test.py): every paper strategy
    (forward_test.STRATEGIES, Reports/forward_strategies.csv), the real account (Reports/forward_test_daily.csv), QQQ, SPY."""
    from backtest_engine import FORWARD_START
    start = f"{pd.Timestamp(FORWARD_START):%b %-d, %Y}"
    daily, values, built = _forward_files()
    if built is None:
        st.info(f"No forward-test rows yet (started {start}).")
        return
    raw, values = built
    render_top10(p.by_symbol["Strategy_Rank"].to_dict() if p is not None and "Strategy_Rank" in p.by_symbol else None)
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
    board = board.drop(columns="Rule")                   # family: filter only; rule + holdings show on hover
    pct = [c for c in board.columns if c not in ("Rank", "Strategy", "Rule")]
    sty = toned(toned(board, [c for c in pct if c != "Max drawdown"]), ["Max drawdown"], fn=drawdown_tone)
    board_html(highlight_rows(sty), board, pct)

TIP_CSS = ("visibility:hidden; position:absolute; z-index:9; left:6px; top:100%; width:520px; "
           "background:#0f172a; color:#f8fafc; font-weight:400; font-size:12.5px; line-height:1.45; padding:9px 11px; "
           "border-radius:8px; box-shadow:0 6px 18px rgba(15,23,42,.3)")
BOARD_CSS = """<style>
body{margin:0;font-family:Inter,system-ui,-apple-system,sans-serif;font-size:14px;color:#0f172a}
.wrap{border:1px solid #e2e8f0;border-radius:8px;overflow:visible}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th{position:sticky;top:0;background:#fff;color:#64748b;font-weight:400;text-align:left;cursor:pointer;user-select:none;white-space:nowrap}
th,td{height:34px;padding:0 9px;border-bottom:1px solid #f1f5f9;border-right:1px solid #f1f5f9;white-space:nowrap}
td{position:relative} td.num,th.num{text-align:right}
th.asc:after{content:" \\2191"} th.desc:after{content:" \\2193"}
td .pd-t{display:block;white-space:pre-line;__TIP__} td:hover .pd-t{visibility:visible} tr.up td .pd-t{top:auto;bottom:100%}
</style>""".replace("__TIP__", TIP_CSS)
SORT_JS = """<script>
const T=document.querySelector('table'),B=T.tBodies[0];
const fit=()=>{const h=document.body.scrollHeight;window.frameElement&&(window.frameElement.style.height=h+'px')};
T.querySelectorAll('thead th').forEach((th,i)=>th.onclick=()=>{const d=th.classList.contains('asc')?-1:1;
 T.querySelectorAll('th').forEach(x=>x.classList.remove('asc','desc'));th.classList.add(d>0?'asc':'desc');
 const v=r=>{const c=r.cells[i];const t=c.firstChild&&c.firstChild.nodeType==3?c.firstChild.data:c.innerText;
  const n=parseFloat(t.replace(/[%+,]/g,''));return isNaN(n)?(t.trim()?t:null):n};
 [...B.rows].sort((a,b)=>{const x=v(a),y=v(b);if(x===null)return 1;if(y===null)return -1;
  return (typeof x=='number'&&typeof y=='number'?x-y:String(x).localeCompare(String(y)))*d}).forEach(r=>B.appendChild(r));flip()});
const flip=()=>[...B.rows].forEach((r,k)=>r.classList.toggle('up',k>=B.rows.length-8&&k>8));flip();
</script>"""

def board_html(sty, board, pct):
    """The leaderboard as an HTML table (st.dataframe has no cell tooltips): the same tones and highlighted rows, click a
    header to sort, and hovering a strategy name shows its rule and current holdings (hover_tips); nothing else visible."""
    import streamlit.components.v1 as components
    tips = dict(zip(board["Strategy"], hover_tips(board)["Strategy"]))
    name = lambda v: esc(v) + (f'<span class="pd-t">{esc(tips[v])}</span>' if tips.get(v) else "")   # noqa: E731
    sty = (sty.format({"Rank": lambda v: "" if pd.isna(v) else f"{v:.0f}", "Strategy": name,
                       **{c: "{:+.2f}%" for c in pct}}, na_rep="")
           .set_td_classes(pd.DataFrame([["num" if c in pct or c == "Rank" else "" for c in board.columns]] * len(board),
                                        index=board.index, columns=board.columns))
           .hide(axis="index"))
    head = "".join(f'<th class="{"num" if c in pct or c == "Rank" else ""}">{esc(c)}</th>' for c in board.columns)
    body = sty.to_html()
    body = body[:body.index("<thead>")] + f"<thead><tr>{head}</tr></thead>" + body[body.index("</thead>") + 8:]
    components.html(f'{BOARD_CSS}<div class="wrap">{body}</div>{SORT_JS}', height=35 * (len(board) + 1) + 4)

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

def hover_tips(board):
    """Hover text per leaderboard row, on the Strategy cell only: the strategy's rule (forward_test.STRATEGIES) and the
    stocks it holds now (forward_test.holdings, the table below); blank for the account and benchmarks."""
    import forward_test as ft
    by_name, held = {c["name"]: c["rule"] for c in ft.STRATEGIES}, ft.holdings()
    def tip(name):
        n = ft.LIVE if name == CURRENT else str(name).replace(ft.LIVE_MARK, "")
        if n not in by_name:
            return ""
        return f"{by_name[n]}\n\nHolds now: {held.get(n, 'cash')}"
    t = pd.DataFrame("", index=board.index, columns=board.columns)
    t["Strategy"] = [tip(x) for x in board["Strategy"]]
    return t

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
    built = _forward_files()[2]
    c = ft.consensus(built[0], read_report_csv(ft.HOLDINGS_CSV), ranks) if built else None
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
