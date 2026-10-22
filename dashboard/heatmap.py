"""Stock Analysis: a Finviz-style heatmap of every stock in sector_mapping.py (display only). Tiles grouped by sector,
size = market cap (yfinance, read once a day), color = today's % change (Alpaca snapshots, SIP -> IEX, vs the previous
close; red -> green, centered at 0, clipped at +-3 %). Its own fragment: refreshes silently every 30 s (new prices
every 30 s in market hours, 5 min outside). Open a stock with the picker below."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

CLIP = 3.0

def _symbols():
    import sector_mapping as sm
    return [s for s in dict.fromkeys(sm.tradable_symbols)]

@st.cache_data(ttl=86400, show_spinner=False)
def market_caps(symbols: tuple, day: str):
    """{symbol: market cap $} from yfinance fast_info (shares x price), once per day; missing ones left out."""
    import yfinance as yf
    def one(s):
        try:
            fi = yf.Ticker(s).fast_info
            mc = fi.get("marketCap") or (fi.get("shares") or 0) * (fi.get("lastPrice") or 0)
            return s, float(mc) if mc else None
        except Exception:
            return s, None
    with ThreadPoolExecutor(max_workers=16) as pool:
        return {s: v for s, v in pool.map(one, symbols) if v}

def heatmap_figure(df):
    sectors = sorted(df["Sector"].unique())
    ids = sectors + list(df["Symbol"])
    parents = [""] * len(sectors) + list(df["Sector"])
    values = [0] * len(sectors) + list(df["Cap"])
    chg = [0.0] * len(sectors) + [max(-CLIP, min(CLIP, c)) if c == c else 0.0 for c in df["Chg"]]
    labels = sectors + [f"<b>{s}</b><br>{c:+.2f}%" if c == c else f"<b>{s}</b>" for s, c in zip(df["Symbol"], df["Chg"])]
    hover = [""] * len(sectors) + [f"<b>{s}</b> · {n}<br>${p:,.2f} · {c:+.2f}% today<br>Market cap ${m / 1e9:,.0f}B"
                                   if p == p else f"<b>{s}</b> · {n}" for s, n, p, c, m in
                                   zip(df["Symbol"], df["Name"], df["Price"], df["Chg"], df["Cap"])]
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values, branchvalues="remainder", customdata=hover,
        hovertemplate="%{customdata}<extra></extra>", textinfo="label", textposition="middle center",
        marker=dict(colors=chg, colorscale=[[0, "#b91c1c"], [0.35, "#7f1d1d"], [0.5, "#414554"], [0.65, "#14532d"], [1, "#16a34a"]],
                    cmin=-CLIP, cmax=CLIP, cmid=0, line=dict(color="#0f172a", width=1)),
        pathbar=dict(visible=False), tiling=dict(pad=1), maxdepth=2,
        textfont=dict(color="#fff", family="Inter, system-ui, sans-serif")))
    fig.update_traces(root_color="#0f172a", selector=dict(type="treemap"))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=460, paper_bgcolor="#0f172a", uirevision="heatmap")
    return fig

@st.fragment(run_every=30)
def render_heatmap(p):
    import sector_mapping as sm
    from dashboard.data import index_quotes
    syms = tuple(s for s in _symbols() if s in p.options)
    all_sectors = sorted({sm.symbol_sector.get(s, "Other") for s in syms})
    pick = st.multiselect("Sectors", all_sectors, key="hm_sectors", placeholder="All sectors", label_visibility="collapsed")
    caps = market_caps(syms, str(date.today()))
    try:
        q = index_quotes(syms)        # Alpaca snapshots, cached 30 s
    except Exception:
        q = {}
    rows = [{"Symbol": s, "Name": sm.symbol_name.get(s, ""), "Sector": sm.symbol_sector.get(s, "Other"),
             "Cap": caps.get(s, 1e9), "Price": q.get(s, (float("nan"), float("nan")))[0],
             "Chg": q.get(s, (float("nan"), float("nan")))[1]} for s in syms]
    df = pd.DataFrame(rows)
    if pick:
        df = df[df["Sector"].isin(pick)]
    if df.empty:
        st.info("No stocks in the chosen sectors.")
        return
    # tile clicks: Streamlit's selection events do not cover treemaps (checked Oct 22, 2026), so the picker below opens a stock
    st.plotly_chart(heatmap_figure(df), key="heatmap", config={"displayModeBar": False}, width="stretch")
