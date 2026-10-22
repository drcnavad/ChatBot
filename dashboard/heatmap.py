"""Stock Analysis: a Finviz-style heatmap of every stock in sector_mapping.py (display only): one grid of equal
small tiles in one compact grid, ordered Buy -> Sell by the strategy signal (then rank / score), colored by today's % change (Alpaca snapshots,
SIP -> IEX, vs the previous close; red -> green, centered at 0, clipped at +-3 %). Its own fragment: refreshes silently
every 30 s. A tile click opens the stock in the picker and chart below."""

import streamlit as st

CLIP = 3.0

def _symbols():
    import sector_mapping as sm
    return [s for s in dict.fromkeys(sm.tradable_symbols)]

def tile_color(c):
    """Red (-3 %) -> grey (0) -> green (+3 %), as Finviz."""
    if c != c:
        return "#414554"
    t = max(-CLIP, min(CLIP, c)) / CLIP
    lo, mid, hi = (185, 28, 28), (65, 69, 84), (22, 163, 74)
    end = hi if t > 0 else lo
    a = abs(t)
    return "#%02x%02x%02x" % tuple(round(m + (e - m) * a) for m, e in zip(mid, end))

def _open(sym):
    st.session_state["_hm_open"] = sym

SIGNAL_ORDER = {"Strong Buy": 0, "Buy": 1, "Hold": 2, "Watch": 3, "Not ranked": 4, "Score below 0": 5, "Sell": 6, "Sold": 7,
                "Strong Sell": 8}

def order(p, syms):
    """Buy first, then Hold, Watch, Not ranked, Score below 0, Sold; within a signal by rank, then score."""
    def key(s):
        sig = str(p.sig_off.get(s, "Not ranked"))
        r = p.by_symbol.loc[s] if s in p.by_symbol.index else {}
        rank, score = r.get("Strategy_Rank", None), r.get("combined_signal", None)
        return (SIGNAL_ORDER.get(sig, 4), rank if rank == rank and rank is not None else 1e9,
                -(score if score == score and score is not None else -1e9))
    return sorted(syms, key=key)

@st.fragment(run_every=30)
def render_heatmap(p):
    """One compact grid of small equal tiles, Buy -> Sell (the strategy signal, then rank / score), colored by today's %;
    a tile click opens the stock in the picker and chart below (a whole-page rerun; the page stays on Stock Analysis)."""
    import sector_mapping as sm
    from dashboard.data import index_quotes
    if st.session_state.get("_hm_open"):
        st.session_state["_pending_ticker"] = st.session_state.pop("_hm_open")
        st.rerun(scope="app")
    syms = [s for s in _symbols() if s in p.options]
    all_sectors = sorted({sm.symbol_sector.get(s, "Other") for s in syms})
    pick = st.multiselect("Sectors", all_sectors, key="hm_sectors", placeholder="All sectors", label_visibility="collapsed")
    from datetime import datetime
    from dashboard.settings import CT
    st.caption(f"Updated {datetime.now(CT):%I:%M:%S %p} CT · Alpaca prices · refreshes every 30 s")
    if pick:
        syms = [s for s in syms if sm.symbol_sector.get(s, "Other") in pick]
    try:
        q = index_quotes(tuple(syms))        # Alpaca snapshots (SIP -> IEX), cached 30 s
    except Exception:
        q = {}
    current = st.session_state.get("ticker_dropdown")
    css = ['<style>[class*="st-key-hm_"] button{width:62px;height:38px;min-height:38px;padding:0 2px;border:0;border-radius:3px}'
           '[class*="st-key-hm_"] button p{color:#fff!important;font:600 10.5px/1.15 Inter,system-ui,sans-serif;white-space:pre-line;text-align:center;margin:0}'
           '[class*="st-key-hm_"] button:hover{filter:brightness(1.25)}'
           '.st-key-hm_grid{gap:3px!important;flex-wrap:wrap!important;justify-content:flex-start!important}'
           '.st-key-hm_grid > *{flex:0 0 62px!important;width:62px!important;min-width:0!important;max-width:62px!important}'
           # hover popup (the Home cards' .sa-pop look): the tile's own container holds the button and the hidden popup
           '[class*="st-key-hmt_"]{position:relative;gap:0!important;width:62px!important;flex:none!important}'
           '[class*="st-key-hmt_"] [data-testid="stElementContainer"]:has(.sa-pop){position:absolute;top:0;left:0;height:0;margin:0;overflow:visible;pointer-events:none}'
           '[class*="st-key-hmt_"] .sa-pop{top:44px}'
           '[class*="st-key-hmt_"]:hover .sa-pop{display:block}'
           '[class*="st-key-hmt_"]:hover{z-index:1000}</style>']
    for s in syms:
        c = q.get(s, (float("nan"), float("nan")))[1]
        ring = ";box-shadow:inset 0 0 0 2px #fcd34d" if s == current else ""
        css.append(f'<style>.st-key-hm_{s.replace(".", "_")} button{{background:{tile_color(c)}!important{ring}}}</style>')
    st.markdown("".join(css), unsafe_allow_html=True)
    from dashboard.data import intraday
    from dashboard.page import sparkline
    from dashboard.style import esc
    try:
        spark = intraday(tuple(syms))     # one batched Alpaca 5-min bars request, cached 60 s (last session before the open)
    except Exception:
        spark = {}
    with st.container(horizontal=True, gap=None, key="hm_grid"):
        for s in order(p, syms):
            px, c = q.get(s, (float("nan"), float("nan")))
            k = s.replace('.', '_')
            with st.container(key=f"hmt_{k}"):
                st.button(f"{s}\n{c:+.2f}%" if c == c else s, key=f"hm_{k}", on_click=_open, args=(s,))
                head = (f"{esc(s)} · {esc(sm.symbol_name.get(s, s))} · {esc(str(p.sig_off.get(s, 'Not ranked')))}"
                        + (f" · ${px:,.2f} · {c:+.2f}% today" if px == px else ""))
                body = sparkline(*spark[s], ",.2f") if s in spark else '<div class="sa-pop-empty">No intraday data right now</div>'
                st.markdown(f'<div class="sa-pop"><div class="sa-pop-title">{head}</div>{body}</div>', unsafe_allow_html=True)
