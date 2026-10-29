"""Strategy tab: the forward test (paper strategies vs the account, QQQ and SPY) and the paper strategies' rules."""
import pandas as pd
import streamlit as st

from dashboard.data import load_benchmarks, read_report_csv
from dashboard.style import BAD, GOOD, INK, drawdown_tone, esc, show_html, toned

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
    live = live_returns(values)
    board = apply_live_totals(board, live)
    board.insert(list(board.columns).index("Last EOD"), "Live", [live.get(x, float("nan")) for x in board["Strategy"]])
    pct = [c for c in board.columns if c not in ("Rank", "Strategy", "Rule")]
    sty = toned(toned(board, [c for c in pct if c != "Max drawdown"]), ["Max drawdown"], fn=drawdown_tone)
    board_html(highlight_rows(sty), board, pct)

def live_returns(values):
    """Display only: {leaderboard display name: live % vs its last EOD value}. A strategy = the cash + shares of its last
    saved day (forward_strategies*.csv, no trades) at live Alpaca prices vs its latest value (provisional close
    included); QQQ / SPY = live price vs their close that day; the account = Alpaca equity now vs its last saved equity."""
    import forward_test as ft
    from dashboard.data import index_quotes, load_benchmarks, read_report_csv
    hold, saved = read_report_csv(ft.HOLDINGS_CSV), read_report_csv(ft.STRATEGIES_CSV)
    if hold is None or saved is None or values is None or values.empty:
        return {}
    v = values.assign(Date=pd.to_datetime(values["Date"]), Strategy=values["Strategy"].replace(ft.OLD_NAMES))
    day = v["Date"].max()
    base = v.sort_values("Date").groupby("Strategy")["Value"].last()
    saved = saved.assign(Date=pd.to_datetime(saved["Date"]), Strategy=saved["Strategy"].replace(ft.OLD_NAMES))
    cash = saved.sort_values("Date").groupby("Strategy")["Cash"].last()
    last = saved.groupby("Strategy")["Date"].max()
    h = hold.assign(Date=pd.to_datetime(hold["Date"]), Strategy=hold["Strategy"].replace(ft.OLD_NAMES))
    h = h[h["Date"] == h["Strategy"].map(last)]          # holdings of the last saved day only (none = all cash that day)
    try:
        q = index_quotes(tuple(sorted(set(h["Symbol"]) | {"QQQ", "SPY"})))
    except Exception:
        q = {}
    out = {}
    for name, g in h.groupby("Strategy"):
        px = g["Symbol"].map(lambda s: q.get(s, (float("nan"),))[0])
        if name in base and name in cash and px.notna().all() and base[name]:
            out[name] = ((cash[name] + float((g["Shares"] * px).sum())) / base[name] - 1) * 100
    for name in base.index:                              # all cash on its last saved day: cash vs its latest value
        if name in cash and name not in set(h["Strategy"]) and base[name]:
            out[name] = (cash[name] / base[name] - 1) * 100
    bench = load_benchmarks()
    for x in ("QQQ", "SPY"):
        px, chg = q.get(x, (float("nan"), float("nan")))
        b = bench[x].dropna() if bench is not None and x in bench else pd.Series(dtype=float)
        ref = float(b[b.index <= day].iloc[-1]) if len(b[b.index <= day]) and b.index.max() >= day else px / (1 + chg / 100)
        out[f"{x} (comparison)"] = out[x] = (px / ref - 1) * 100
    from dashboard.details.live_holdings import live_holdings
    data, _ = live_holdings()
    daily = read_report_csv(ft.DAILY_CSV)
    if data and daily is not None and len(daily):
        out["My Alpaca account"] = (float(data["equity"]) / float(daily["Equity"].iloc[-1]) - 1) * 100
        out[CURRENT] = out["My Alpaca account"]   # live/Current row tracks the real account (Chirag, Oct 29)
    return out

def render_live_push():
    """The leaderboard's in-place Live updater, outside the 5-minute leaderboard fragment (nested timed fragments
    re-register their keys when both rerun together: StreamlitDuplicateElementKey, Oct 22)."""
    with st.container(key="ft_live_js"):
        _live_push()

@st.fragment(run_every=30)
def _live_push():
    """Every 30 s: write the Live column's new values into the leaderboard table in place (same-origin component
    iframe), so its sort order and hover stay; a hidden component (.st-key-ft_live_js)."""
    import json
    import streamlit.components.v1 as components
    built = _forward_files()[2]
    try:
        live = {k: round(v, 2) for k, v in live_returns(built[1]).items() if v == v} if built else {}
    except Exception:
        live = {}
    if not live:
        return
    components.html("""<script>
const L=%s;for(const f of window.parent.document.querySelectorAll('iframe')){let d;try{d=f.contentDocument}catch(e){continue}
 if(!d)continue;const T=d.querySelector('table');if(!T)continue;const hs=[...T.querySelectorAll('thead th')].map(h=>h.innerText.trim());
 const i=hs.indexOf('Live'),n=hs.indexOf('Strategy');if(i<0||n<0)continue;
 for(const r of T.tBodies[0].rows){const c=r.cells[n].firstChild;const k=c&&c.nodeType==3?c.data:'';if(!(k in L))continue;
  const v=L[k];r.cells[i].textContent=(v>0?'+':'')+v.toFixed(2)+'%%';r.cells[i].style.color=v>0?'%s':v<0?'%s':'%s'}}
</script>""" % (json.dumps(live), GOOD, BAD, INK), height=0)

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

CURRENT = "Live: RS short-term"   # display name of the live strategy's row (forward_test.LIVE stays its id)

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
    # Only the account-backed marked row (… LIVE_MARK) is shown as CURRENT; the paper RS short-term line keeps its name.
    b["Strategy"] = [CURRENT if ft.LIVE_MARK in str(x) else x for x in b["Strategy"]]
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

def live_components():
    """The forward-test lines the live strategy is built from (backtest_engine.WINNER['sleeves'] -> sleeves.STRATEGY)."""
    import backtest_engine as be
    import sleeves
    return tuple(sleeves.STRATEGY[h] for h in (be.WINNER.get("sleeves") or {}) if h in sleeves.STRATEGY)

def highlight_rows(sty):
    """Darker rows: your Alpaca account, the current (live) strategy and its component line (live_components)."""
    import forward_test as ft
    keys = ("My Alpaca account", CURRENT, *live_components())
    on = lambda name: name == CURRENT or ft.LIVE_MARK in str(name) or str(name).replace(ft.LIVE_MARK, "") in keys
    return sty.apply(lambda r: [HIGHLIGHT_BG if on(r["Strategy"]) else ""] * len(r), axis=1)


TILE_CSS = (
    '<style>.st-key-cs_grid{gap:4px!important;flex-wrap:wrap!important;justify-content:flex-start!important}'
    '.st-key-cs_grid > *{flex:0 0 84px!important;width:84px!important;min-width:0!important;max-width:84px!important}'
    '[class*="st-key-cs_b_"] button{width:84px;height:46px;min-height:46px;padding:0 2px;border:0;border-radius:3px}'
    '[class*="st-key-cs_b_"] button p{color:#fff!important;font:600 11px/1.2 Inter,system-ui,sans-serif;white-space:pre-line;text-align:center;margin:0}'
    '[class*="st-key-cs_b_"] button:hover{filter:brightness(1.25)}'
    '[class*="st-key-cs_t_"]{position:relative;gap:0!important;width:84px!important;flex:none!important}'
    '[class*="st-key-cs_t_"] [data-testid="stElementContainer"]:has(.sa-pop){position:absolute;top:0;left:0;height:0;margin:0;overflow:visible;pointer-events:none}'
    '[class*="st-key-cs_t_"] .sa-pop{top:52px}[class*="st-key-cs_t_"]:hover .sa-pop{display:block}'
    '[class*="st-key-cs_t_"]:hover{z-index:1000}</style>')


def apply_live_totals(board, live=None):
    """Compound Total since FORWARD_START (and the current week-so-far column) with today's live return; re-rank by the live
    total (smaller max drawdown breaks ties). Once today's close is saved, live ≈ after-hours only so this does not
    double-count. live keys are display names (CURRENT, strategy names, QQQ/SPY, My Alpaca account)."""
    import forward_test as ft
    live = live or {}
    b = board.copy()
    total_col = next((c for c in b.columns if c.startswith("Total since")), None)
    if total_col is None and "Total return %" in b.columns:
        total_col = "Total return %"
    wk_col = next((c for c in b.columns if ("so far" in c) and (c.startswith("Wk ") or c.startswith("Week "))), None)
    if total_col is None:
        return b
    lv = b["Strategy"].map(lambda n: live.get(n, float("nan")))
    # Also accept unmarked LIVE paper name / LIVE+mark raw forms
    def _lv(n):
        if n in live and live[n] == live[n]:
            return live[n]
        if n == CURRENT:
            return live.get(CURRENT, live.get("My Alpaca account", float("nan")))
        return live.get(n, float("nan"))
    lv = b["Strategy"].map(_lv)
    factor = (1 + lv.fillna(0) / 100)
    b[total_col] = ((1 + b[total_col] / 100) * factor - 1) * 100
    if wk_col is not None:
        b[wk_col] = ((1 + b[wk_col] / 100) * factor - 1) * 100
    # Re-rank strategies only
    ranked = pd.to_numeric(b["Rank"], errors="coerce").notna()
    dd = b["Max drawdown"] if "Max drawdown" in b.columns else b.get("Max drawdown %")
    order = b.loc[ranked].sort_values([total_col, dd], ascending=[False, False], na_position="last", kind="stable")
    new_rank = pd.Series(range(1, len(order) + 1), index=order.index, dtype="Int64")
    b["Rank"] = new_rank.reindex(b.index)
    return b.sort_values(["Rank", total_col], ascending=[True, False], na_position="last", kind="stable").reset_index(drop=True)

def by_this_week(board, live=None):
    """Display only (Chirag, Oct 22): the leaderboard re-ranked by the current week's return so far ("Week N (so far) %"),
    compounded with today's live return (live_returns) where there is one, ranked strategies only, so consensus()
    takes this week's top 5. forward_test's own Rank is untouched. Returns (board, week label like "Wk 3")."""
    import forward_test as ft
    cur = [c for c in board.columns if c.startswith("Week ") and "so far" in c and board[c].notna().any()]
    if not cur:
        return board, None
    wk, live = cur[0], live or {}
    def shown(n):
        return CURRENT if str(n).startswith(ft.LIVE) else str(n).replace(ft.LIVE_MARK, "")
    lv = board["Strategy"].map(lambda n: live.get(shown(n), float("nan")))
    ret = ((1 + board[wk] / 100) * (1 + lv.fillna(0) / 100) - 1) * 100
    ranked = pd.to_numeric(board["Rank"], errors="coerce").notna()
    r = ret.where(ranked).rank(ascending=False, method="first")
    return board.assign(Rank=r, _wk=ret), "Wk " + wk.split()[1]

def _cs_open(sym):
    st.session_state["_cs_open"] = sym

@st.fragment(run_every=30)
def render_top10(p=None):
    """Top 5 consensus (display only) as Stock Analysis-style tiles: every stock held by the leaderboard's Rank 1-5
    strategies by this week's return so far (by_this_week), colored by its live week-so-far return (the previous week's
    last close -> the Alpaca price now), labeled with how many of the 5 hold it; hover = names + today's sparkline; click = open in Stock Analysis.
    Sorted by holders, then this week's return. Not nested in the leaderboard fragment (duplicate keys). Its own fragment: refreshes silently every 30 s."""
    import forward_test as ft
    import sector_mapping as sm
    from dashboard.data import index_quotes, intraday, prior_week_close
    from dashboard.heatmap import _symbols
    ranks = p.by_symbol["Strategy_Rank"].to_dict() if p is not None and "Strategy_Rank" in p.by_symbol else None
    from dashboard.heatmap import tile_color
    from dashboard.page import sparkline
    if st.session_state.get("_cs_open"):
        st.session_state["_pending_ticker"] = st.session_state.pop("_cs_open")
        st.rerun(scope="app")
    built = _forward_files()[2]
    hold = read_report_csv(ft.HOLDINGS_CSV)
    try:
        live = live_returns(built[1]) if built else {}
    except Exception:
        live = {}
    board, wk = by_this_week(built[0], live) if built else (None, None)
    c = ft.consensus(board, hold, ranks, n_stocks=10 ** 4) if built else None
    if c is None or c["stocks"].empty:
        show_html('<div class="sa-card"><div class="sa-card-title">Top 5 consensus<span>no saved strategy holdings yet</span></div></div>')
        return
    h = hold.assign(Date=pd.to_datetime(hold["Date"]), Strategy=hold["Strategy"].replace(ft.OLD_NAMES))
    h = h[(h["Date"] == h.groupby("Strategy")["Date"].transform("max")) & (h["Weight"] > 0)]
    top = [nm for nm, _ in c["strategies"]]
    by = {s: [CURRENT if nm == ft.LIVE else nm for nm in top if nm in set(g["Strategy"])] for s, g in h.groupby("Symbol")}
    syms = tuple(c["stocks"]["Symbol"])
    # the Stock Analysis grid's own batched request (same symbol tuple -> the same 20 s / 60 s cache entries)
    grid = tuple(s for s in _symbols() if p is None or s in p.options)
    req = grid if set(syms) <= set(grid) else tuple(sorted(set(grid) | set(syms)))
    from dashboard.data import NY
    monday = (pd.Timestamp.now(tz=NY).normalize() - pd.Timedelta(days=pd.Timestamp.now(tz=NY).weekday())).tz_localize(None)
    good = st.session_state.setdefault("_cs_last", {"q": {}, "base": {}, "spark": {}})   # last good values per symbol
    for name, fn in (("q", lambda: index_quotes(req)), ("base", lambda: prior_week_close(syms, f"{monday:%Y-%m-%d}")),
                     ("spark", lambda: intraday(req))):
        try:
            good[name].update({k: v for k, v in (fn() or {}).items() if k in syms})
        except Exception:
            pass                                          # a hiccup: keep the last good values, show no error
    q, base, spark = good["q"], good["base"], good["spark"]
    def r5(s):
        px, b = q.get(s, (float("nan"),))[0], base.get(s, float("nan"))
        return (px / b - 1) * 100 if px == px and b == b and b else float("nan")
    rows = sorted(((s, len(by.get(s, [])), r5(s)) for s in syms), key=lambda x: (-x[1], -(x[2] if x[2] == x[2] else -1e9)))
    n = len(top)
    from datetime import datetime
    from dashboard.settings import CT
    now = datetime.now(CT)
    when = f"{now:%-I:%M %p} CT" + ("" if now.weekday() < 5 else f" {now:%a, %b %-d}")
    st.markdown(f'<div class="sa-card-title">Top 5 consensus<span>{len(rows)} stocks held by this week\'s top {n} strategies '
                f'({wk} so far). Colored by this week\'s return through {when}.</span></div>'
                + TILE_CSS + "".join(f'<style>.st-key-cs_b_{s.replace(".", "_")} button{{background:{tile_color(r / 1.5 if r == r else r)}!important}}</style>'
                                     for s, _, r in rows), unsafe_allow_html=True)
    with st.container(horizontal=True, gap=None, key="cs_grid"):
        for s, k, r in rows:
            key = s.replace(".", "_")
            with st.container(key=f"cs_t_{key}"):
                st.button(f"{s}  {k}/{n}\n" + (f"{r:+.2f}%" if r == r else "–"), key=f"cs_b_{key}", on_click=_cs_open, args=(s,))
                px = q.get(s, (float("nan"),))[0]
                head = (f"{esc(s)} · {esc(sm.symbol_name.get(s, s))}" + (f" · ${px:,.2f}" if px == px else "")
                        + (f" · {r:+.2f}% this week" if r == r else "") + f" · held by {k}/{n}: {esc(', '.join(by.get(s, [])))}")
                body = sparkline(*spark[s], ",.2f") if s in spark else '<div class="sa-pop-empty">No intraday data right now</div>'
                st.markdown(f'<div class="sa-pop"><div class="sa-pop-title">{head}</div>{body}</div>', unsafe_allow_html=True)


def current_holdings_table(positions, ranks=None):
    """Ticker, Weight % (of the holdings' market value + cash is not counted), RS short-term rank, Today % - from the
    live Alpaca positions (read-only)."""
    rows, tot = [], sum(float(x.get("market_value") or 0) for x in positions or []) or 1.0
    for x in positions or []:
        s, mv = x["symbol"], float(x.get("market_value") or 0)
        ch = x.get("change_today")
        rows.append({"Ticker": s, "Weight %": 100 * mv / tot, "Value $": mv,
                     "RS short-term rank": (ranks or {}).get(s), "Today %": float(ch) * 100 if ch not in (None, "") else None})
    return pd.DataFrame(rows, columns=["Ticker", "Weight %", "Value $", "RS short-term rank", "Today %"]).sort_values(
        "Weight %", ascending=False).reset_index(drop=True)

@st.fragment(run_every=60)
def render_current_holdings():
    """Current holdings of the live account (Alpaca positions, read-only) as the Top 5 consensus tiles, right below
    them: label = weight %, today %; color = today's change; hover = name, rank, value + today's sparkline; click = open
    in Stock Analysis (same as the consensus tiles)."""
    import sector_mapping as sm
    from dashboard.details.live_holdings import holdings_this_run
    from dashboard.heatmap import tile_color
    from dashboard.page import sparkline
    if st.session_state.get("_cs_open"):
        st.session_state["_pending_ticker"] = st.session_state.pop("_cs_open")
        st.rerun(scope="app")
    data, err = holdings_this_run()
    if data is None:
        show_html(f'<div class="sa-card"><div class="sa-card-title">Current holdings<span>{esc(str(err))}</span></div></div>')
        return
    try:
        import sleeves
        day = sleeves.inputs()["close"].index[-1]
        ranks = {s: i + 1 for i, s in enumerate(sleeves.ranking("rs_short", day).index)}
    except Exception:
        ranks = None
    t = current_holdings_table(data["positions"], ranks)
    if t.empty:
        show_html('<div class="sa-card"><div class="sa-card-title">Current holdings<span>no positions</span></div></div>')
        return
    spark = {}
    try:
        from dashboard.data import intraday
        spark = intraday(tuple(t["Ticker"])) or {}
    except Exception:
        pass
    # the Top 5 consensus tiles' own color logic: this week's return so far (the previous week's last close -> the
    # Alpaca price now), tile_color(r / 1.5); same cached batched requests
    from dashboard.data import NY, index_quotes, prior_week_close
    from dashboard.heatmap import _symbols
    syms = tuple(t["Ticker"])
    grid = tuple(_symbols())
    req = grid if set(syms) <= set(grid) else tuple(sorted(set(grid) | set(syms)))
    monday = (pd.Timestamp.now(tz=NY).normalize() - pd.Timedelta(days=pd.Timestamp.now(tz=NY).weekday())).tz_localize(None)
    good = st.session_state.setdefault("_ch_last", {"q": {}, "base": {}})
    for name, fn in (("q", lambda: index_quotes(req)), ("base", lambda: prior_week_close(syms, f"{monday:%Y-%m-%d}"))):
        try:
            good[name].update({k: v for k, v in (fn() or {}).items() if k in syms})
        except Exception:
            pass
    q, base = good["q"], good["base"]
    def r5(s):
        px, b = q.get(s, (float("nan"),))[0], base.get(s, float("nan"))
        return (px / b - 1) * 100 if px == px and b == b and b else float("nan")
    wk = {s: r5(s) for s in syms}
    eq = float(data.get("equity") or 0)
    rows = list(t.itertuples(index=False))
    css = TILE_CSS.replace("cs_grid", "ch_grid")
    st.markdown(f'<div class="sa-card-title">Current holdings<span>{len(rows)} stocks in the live account ({CURRENT})'
                + (f" · {100 * t['Value $'].sum() / eq:.1f}% invested" if eq else "")
                + ". Colored by this week's return, like the Top 5 consensus; label = weight, this week's return.</span></div>" + css
                + "".join(f'<style>.st-key-cs_b_h_{r[0].replace(".", "_")} button{{background:{tile_color(wk[r[0]] / 1.5 if wk[r[0]] == wk[r[0]] else wk[r[0]])}!important}}</style>'
                          for r in rows), unsafe_allow_html=True)
    with st.container(horizontal=True, gap=None, key="ch_grid"):
        for sym, w, val, rk, ch in rows:
            key = sym.replace(".", "_")
            ok = ch is not None and ch == ch
            with st.container(key=f"cs_t_h_{key}"):
                r = wk[sym]
                st.button(f"{sym}  {w:.1f}%\n" + (f"{r:+.2f}%" if r == r else "–"), key=f"cs_b_h_{key}", on_click=_cs_open, args=(sym,))
                head = (f"{esc(sym)} · {esc(sm.symbol_name.get(sym, sym))} · ${val:,.0f} ({w:.1f}%)"
                        + (f" · RS short-term rank {int(rk)}" if rk == rk and rk is not None else " · not ranked")
                        + (f" · {wk[sym]:+.2f}% this week" if wk[sym] == wk[sym] else "") + (f" · {ch:+.2f}% today" if ok else ""))
                body = sparkline(*spark[sym], ",.2f") if sym in spark else '<div class="sa-pop-empty">No intraday data right now</div>'
                st.markdown(f'<div class="sa-pop"><div class="sa-pop-title">{head}</div>{body}</div>', unsafe_allow_html=True)
