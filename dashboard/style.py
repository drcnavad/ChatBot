"""Page setup (config + CSS, every run), chart colors, small HTML helpers (show_html, esc, stat tiles) and the
color rules (tone, toned), plus the text helpers the Details panels share (caption / warning / info text,
stat cards, usd, MONEY).

Markdown ends an HTML block at the first blank line and turns 4-space-indented lines into code, so every custom
HTML block goes through show_html(): one line, no indentation, dynamic text escaped with esc()."""
import html
import re

import pandas as pd
from pandas.io.formats.style import Styler
import streamlit as st

CSS = """
<style>
/* page selector (st.radio key="nav" in the title row) drawn as tabs */
[data-testid="stLayoutWrapper"]:has(.sa-brand) [role="radiogroup"] { gap: 0.25rem; flex-wrap: nowrap; justify-content: center; }
[data-testid="stLayoutWrapper"]:has(.sa-brand) [role="radiogroup"] label { padding: 0.35rem 0.7rem; border-radius: 8px; margin: 0; cursor: pointer; }
[data-testid="stLayoutWrapper"]:has(.sa-brand) [role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stLayoutWrapper"]:has(.sa-brand) [role="radiogroup"] label p { font-weight: 600; color: #64748b; white-space: nowrap; }
[data-testid="stLayoutWrapper"]:has(.sa-brand) [role="radiogroup"] label:has(input:checked) { box-shadow: inset 0 -3px 0 #0f766e; }
[data-testid="stLayoutWrapper"]:has(.sa-brand) [role="radiogroup"] label:has(input:checked) p { color: #0f766e; }
[data-testid="stStatusWidget"] { visibility: hidden; height: 0; }  /* no 'Running…' flash on 30 s fragment updates */
.stApp [data-stale="true"] { opacity: 1 !important; transition: none !important; }  /* no fade while a fragment reruns */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');
    :root { --sa-ink: #0f172a; --sa-muted: #64748b; --sa-line: #e2e8f0; --sa-brand: #0f766e; --sa-bg: #f6f8fb; }
    .stApp, .main { background: var(--sa-bg); }
    .main .block-container, [data-testid="stMainBlockContainer"] { padding: 0.75rem 2.25rem 3rem 2.25rem !important; max-width: 1480px; }
    .stApp > header, header[data-testid="stHeader"], [data-testid="stDecoration"] { display: none !important; height: 0 !important; }
    #MainMenu, footer, header { visibility: hidden; }
    html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, -apple-system, sans-serif; color: var(--sa-ink); }
    h1, h2, h3, h4, h5 { color: var(--sa-ink); font-weight: 700; letter-spacing: -0.02em; }
    /* brand bar */
    .sa-brand { display: flex; align-items: center; gap: 0.7rem; }
    .sa-logo { width: 2.1rem; height: 2.1rem; border-radius: 10px; background: linear-gradient(135deg, #0f766e, #14b8a6);
               color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.95rem;
               box-shadow: 0 4px 12px rgba(15,118,110,.25); }
    .sa-brand-name { font-size: 1.3rem; font-weight: 800; letter-spacing: -0.03em; color: var(--sa-ink); line-height: 1.1; }
    .sa-brand-sub { font-size: 0.75rem; color: var(--sa-muted); font-weight: 500; }
    /* held-stock ticker strip: the row is doubled and slides half its width, so the loop is seamless */
    .sa-ticker { overflow: hidden; background: #0f172a; border-radius: 10px; margin: 0 0 0.6rem; padding: 0.5rem 0;
        -webkit-mask-image: linear-gradient(90deg, transparent, #000 4%, #000 96%, transparent); }
    .sa-ticker-track { display: inline-flex; white-space: nowrap; animation: sa-scroll linear infinite; will-change: transform; }
    .sa-ticker:hover .sa-ticker-track { animation-play-state: paused; }
    @keyframes sa-scroll { from { transform: translateX(0); } to { transform: translateX(-50%); } }
    .sa-tick { display: inline-flex; gap: 0.45rem; align-items: baseline; padding: 0 1.3rem; text-decoration: none;
        font-size: 0.9rem; font-variant-numeric: tabular-nums; border-right: 1px solid #1e293b; }
    .sa-tick b { color: #f8fafc; font-weight: 700; } .sa-tick span { color: #cbd5e1; }
    .sa-tick .up { color: #4ade80; } .sa-tick .down { color: #f87171; }
    .st-key-ft_live_js, .st-key-sa_ticker_js { display: none; }  /* the hidden price updater of the ticker strip */
    /* stock card: compact label / value tables side by side */
    .sa-kv-grid { display: grid; grid-template-columns: 1fr 1fr 2fr; gap: 1.25rem; margin-top: 0.75rem; }
    @media (max-width: 1000px) { .sa-kv-grid { grid-template-columns: 1fr; } }
    .sa-kv table { width: 100%; border-collapse: collapse; margin-top: 0.3rem; font-size: 0.82rem; }
    .sa-kv td { padding: 0.28rem 0.35rem; border-bottom: 1px solid #f1f5f9; color: var(--sa-muted); }
    .sa-kv td:nth-child(even) { text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; padding-right: 0.9rem; }
    /* header cards (dashboard/page.py kpi_card): one grid, every card the same size; hover = today's intraday line */
    /* script-only frames (news voice) take no space */
    [data-testid="stElementContainer"]:has(> iframe[height="0"]), [data-testid="stElementContainer"]:has(iframe[data-testid="stCustomComponentV1"][height="0"]) { position: absolute; width: 0; height: 0; overflow: hidden; }
    .sa-kpis { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 0.6rem; margin-bottom: 0.6rem; }
    @media (max-width: 1100px) { .sa-kpis { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
    .sa-kpi { position: relative; border: 1px solid var(--sa-line); border-left: 4px solid #cbd5e1; border-radius: 14px;
        padding: 0.55rem 0.8rem 0.4rem; height: 118px; box-sizing: border-box; display: flex; flex-direction: column;
        align-items: center; text-align: center; background: linear-gradient(180deg, #fff 0%, #f8fafc 100%);
        box-shadow: 0 1px 2px rgba(15,23,42,.05); transition: box-shadow .18s ease, transform .18s ease; min-width: 0; }
    .sa-kpi.up { border-left-color: #16a34a; background: linear-gradient(180deg, #fff 0%, #f0fdf4 100%); }
    .sa-kpi.down { border-left-color: #dc2626; background: linear-gradient(180deg, #fff 0%, #fef2f2 100%); }
    .sa-kpi:hover { box-shadow: 0 10px 24px rgba(15,23,42,.12); transform: translateY(-2px); z-index: 50; }   /* its popup above the next cards / rows */
    .sa-kpi.has-pop { cursor: default; }
    .sa-kpi-badge { position: absolute; top: 6px; left: 8px; font-size: 0.6rem; font-weight: 700; letter-spacing: .04em;
        color: var(--sa-brand); background: rgba(37,99,235,.08); border-radius: 6px; padding: 1px 5px; }
    .sa-kpi-label { color: var(--sa-muted); font-weight: 600; font-size: 0.68rem; text-transform: uppercase; letter-spacing: .06em;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; margin-top: 0.85rem; }
    .sa-kpi-value { font-size: 1.25rem; font-weight: 700; color: var(--sa-ink); font-variant-numeric: tabular-nums; line-height: 1.25; letter-spacing: -.01em; }
    .sa-kpi-delta { font-size: 0.78rem; font-weight: 600; color: var(--sa-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; }
    .sa-kpi-delta.up { color: #15803d; } .sa-kpi-delta.down { color: #b91c1c; }
    .sa-kpi-spark { width: 100%; margin-top: auto; height: 22px; opacity: .85; }
    .sa-kpi-spark svg { width: 100%; height: 22px; }
    .sa-kpi:has(.sa-kpi-foot) { padding-bottom: 0.25rem; } .sa-kpi:has(.sa-kpi-foot) .sa-kpi-label { margin-top: 0.7rem; }
    .sa-kpi:has(.sa-kpi-foot) .sa-kpi-spark, .sa-kpi:has(.sa-kpi-foot) .sa-kpi-spark svg { height: 16px; }
    .sa-kpi-foot { width: 100%; border-top: 1px solid var(--sa-line); margin-top: 2px; padding-top: 2px; font-size: 10px;
        font-weight: 600; color: var(--sa-muted); letter-spacing: .02em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .sa-pop { display: none; position: absolute; top: calc(100% + 8px); left: 0; z-index: 1000; background: #fff;
        border: 1px solid var(--sa-line); border-radius: 12px; padding: 0.6rem 0.7rem; box-shadow: 0 12px 32px rgba(15,23,42,.16); }
    .sa-kpi.has-pop:hover .sa-pop { display: block; }
    .sa-kpi:nth-child(6n) .sa-pop, .sa-kpi:nth-child(6n-1) .sa-pop { left: auto; right: 0; }
    [data-testid="stElementContainer"]:has(.sa-kpis), [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .sa-kpis) {
        position: relative; z-index: 50; overflow: visible !important; }
    [data-testid="stMarkdownContainer"]:has(.sa-kpis), [data-testid="stMarkdown"]:has(.sa-kpis) { overflow: visible !important; }
    .sa-pop-title { font-size: 0.72rem; font-weight: 700; color: var(--sa-ink); margin-bottom: 0.3rem; }
    .sa-pop-foot { display: flex; justify-content: space-between; gap: 1rem; font-size: 0.72rem; color: var(--sa-muted); margin-top: 0.2rem; }
    .sa-pop-empty { font-size: 0.75rem; color: var(--sa-muted); width: 220px; }
    .sa-eq-head { display: flex; align-items: baseline; gap: 0.8rem; flex-wrap: wrap; margin: 0.2rem 0 0.3rem; }
    .sa-eq-val { font-size: 1.6rem; font-weight: 800; color: var(--sa-ink); font-variant-numeric: tabular-nums; }
    .sa-eq-chg { font-size: 0.95rem; font-weight: 700; } .sa-eq-sub { font-size: 0.78rem; color: var(--sa-muted); }
    /* KPI cards */
    [data-testid="stMetric"] { background: #fff; border: 1px solid var(--sa-line); border-radius: 14px; padding: 0.9rem 1.1rem;
        box-shadow: 0 1px 2px rgba(15,23,42,.04); transition: box-shadow .15s ease, transform .15s ease;
        height: 112px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: space-between; }
    [data-testid="stMetric"] > div { gap: 0.15rem; }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    [data-testid="stMetricDelta"] svg { width: 0.85rem; height: 0.85rem; }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width: 0; }
    [data-testid="stMetric"]:hover { box-shadow: 0 6px 18px rgba(15,23,42,.08); transform: translateY(-1px); }
    [data-testid="stMetricValue"] { color: var(--sa-ink); font-weight: 700; font-size: 1.45rem; line-height: 1.2; font-family: 'Inter', sans-serif;
        font-variant-numeric: tabular-nums; }
    [data-testid="stMetricLabel"] { color: var(--sa-muted); font-weight: 600; font-size: 0.7rem; text-transform: uppercase; letter-spacing: .05em; line-height: 1.2; }
    [data-testid="stMetricDelta"] { font-weight: 600; font-size: 0.8rem; }
    /* buttons */
    .stButton > button { background: var(--sa-brand); color: #fff; border: none; border-radius: 10px; font-weight: 600;
        padding: 0.45rem 1rem; box-shadow: 0 1px 2px rgba(15,23,42,.08); }
    .stButton > button:hover { background: #0d9488; color: #fff; }
    /* navigation tabs: a clean underline nav bar */
    [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid var(--sa-line); gap: 0.25rem; padding: 0; margin-bottom: 0.75rem; }
    [data-baseweb="tab"] { border-radius: 8px 8px 0 0; font-weight: 600; font-size: 0.95rem; color: var(--sa-muted);
        background: transparent; border: none; padding: 0.7rem 1.4rem; }
    [data-baseweb="tab"]:hover { color: var(--sa-brand); background: #f0fdfa; }
    [data-baseweb="tab"][aria-selected="true"] { color: var(--sa-brand); background: transparent; }
    [data-baseweb="tab-highlight"] { background: var(--sa-brand) !important; height: 3px !important; border-radius: 3px; }
    [data-baseweb="tab-border"] { display: none; }
    /* panels */
    [data-testid="stExpander"] { background: #fff; border: 1px solid var(--sa-line); border-radius: 14px;
        box-shadow: 0 1px 2px rgba(15,23,42,.04); margin-bottom: 0.9rem; }
    [data-testid="stExpander"] details { border: none; }
    [data-testid="stExpander"] summary { padding: 0.85rem 1.1rem; }
    [data-testid="stExpander"] summary p { font-weight: 700; color: var(--sa-ink); font-size: 1rem; }
    [data-testid="stDataFrame"] { border: 1px solid var(--sa-line); border-radius: 10px; overflow: hidden; }
    [data-testid="stCaptionContainer"] { color: var(--sa-muted); line-height: 1.45; font-size: 0.8rem; }
    [data-testid="stAlert"] { border-radius: 12px; }
    .js-plotly-plot { border-radius: 12px; background: #fff; border: 1px solid var(--sa-line); padding: 4px; }
    .symbol-link { color: #0f766e; text-decoration: none; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; }
    .symbol-link:hover { color: #0d9488; text-decoration: underline; }
    .sa-chip { font-size: 0.75rem; font-weight: 600; color: #0f766e; background: #ccfbf1; border: 1px solid #99f6e4;
               padding: 0.35rem 0.7rem; border-radius: 999px; white-space: nowrap; }
    .sa-section { font-weight: 700; color: #0f172a; margin: 0.6rem 0 0.2rem; font-size: 1.02rem; }
    .sa-chip-warn { color: #b45309; background: #fffbeb; border-color: #fde68a; }
    .sa-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 1rem 1.25rem; margin: 0.5rem 0 0.9rem;
                         box-shadow: 0 1px 3px rgba(15,23,42,.06); }
    .sa-card-title { font-size: 1.02rem; font-weight: 700; color: #0f172a; margin-bottom: 0.35rem; }
    .sa-card-title span { font-size: 0.8rem; font-weight: 500; color: #64748b; margin-left: 0.35rem; }
    .sa-tier { display: flex; gap: 0.75rem; align-items: baseline; padding: 0.45rem 0; border-top: 1px solid #f1f5f9; }
    .sa-tier > b { flex: 0 0 8.5rem; font-size: 0.7rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: #64748b; }
    .sa-tier-syms { display: flex; flex-wrap: wrap; gap: 0.3rem; font-size: 0.8rem; color: #64748b; align-items: center;
        flex: 1; min-width: 0; }
    .sa-tier-syms > * { flex-shrink: 0; }   /* chips stay whole; the row wraps onto more lines instead of scrolling */
    .sa-tier-syms .symbol-link { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 7px; padding: 1px 7px; font-size: 0.8rem; color: #334155; }
    .sa-tier-syms .symbol-link.t-good { color: #15803d; background: #f0fdf4; border-color: #bbf7d0; }
    .sa-tier-syms .symbol-link.t-warn { color: #b45309; background: #fffbeb; border-color: #fde68a; }
    .sa-tier-syms .symbol-link.t-bad { color: #b91c1c; background: #fef2f2; border-color: #fecaca; }
    .sa-tier-syms .symbol-link.sel { outline: 2px solid #0f766e; outline-offset: 1px; }
    .sa-ident { display: flex; align-items: center; gap: 0.85rem; flex-wrap: wrap; }
    .sa-sym { font-size: 1.75rem; font-weight: 700; color: #0f172a; font-family: 'JetBrains Mono', monospace; }
    .sa-price { font-size: 1.25rem; font-weight: 700; color: #334155; font-family: 'JetBrains Mono', monospace; }
    .sa-live { font-size: 0.85rem; font-weight: 600; color: #334155; font-family: 'JetBrains Mono', monospace; margin-top: 0.15rem; }
    .sa-live-dot { display: inline-block; width: 0.5rem; height: 0.5rem; border-radius: 9999px; background: #16a34a; margin-right: 0.3rem; }
    .sa-live-sub { font-weight: 400; color: #64748b; }
    .sa-badge { display: inline-block; padding: 0.35rem 0.8rem; border-radius: 999px; font-weight: 700; font-size: 0.8rem; }
    .sa-badge-bull { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .sa-badge-bear { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .sa-badge-hold { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
    .sa-badge-grey { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }
    .sa-why { font-size: 0.8rem; color: #64748b; margin-top: 0.35rem; }
    .sa-stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(9.5rem, 1fr)); gap: 0.5rem; margin-top: 0.8rem; }
    .sa-stat { background: #f8fafc; border: 1px solid #eef2f7; border-radius: 10px; padding: 0.45rem 0.7rem; }
    .sa-stats .sa-stat-label { white-space: normal; }
    .sa-group-title { font-size: 0.7rem; font-weight: 700; letter-spacing: .09em; color: #64748b; text-transform: uppercase; }
    .sa-stat-label { font-size: 0.66rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap; }
    .sa-stat-val { font-size: 0.95rem; font-weight: 700; color: #0f172a; font-family: 'JetBrains Mono', monospace; white-space: nowrap; }
/* breaking-news banner (dashboard/news.py): one headline at a time, CSS fade, paused on hover */
.sa-news{display:flex;align-items:flex-start;gap:14px;margin:10px 0 18px;padding:10px 16px;border-radius:8px;
  background:linear-gradient(90deg,#0b1f3a,#13315c);box-shadow:0 2px 6px rgba(15,23,42,.18);overflow:hidden}
.sa-news-tag{flex:none;margin-top:1px;display:inline-flex;align-items:center;gap:6px;font:700 11px Inter,system-ui,sans-serif;letter-spacing:.08em;
  color:#fff;background:#dc2626;border-radius:999px;padding:4px 10px;white-space:nowrap}
.sa-news-tag:before{content:"";width:7px;height:7px;border-radius:50%;background:#fff;animation:sa-blink 1.4s infinite}
@keyframes sa-blink{50%{opacity:.25}}
.sa-news-box{flex:1;min-width:0;display:grid}   /* pages stacked in one cell: the band fits the tallest */
.sa-news-page{grid-area:1/1;display:flex;flex-direction:column;gap:4px;opacity:0;visibility:hidden}
.sa-news-item{display:flex;align-items:baseline;gap:8px;text-decoration:none!important;color:#f8fafc!important;
  font-size:14.5px;font-weight:600;line-height:1.35}
.sa-news:hover .sa-news-page{animation-play-state:paused!important}
.sa-news-item:hover .sa-news-t{text-decoration:underline}
.sa-news-sym{flex:none;font:700 11px Inter,system-ui,sans-serif;color:#0b1f3a;background:#fcd34d;border-radius:4px;padding:2px 6px}
.sa-news-t{flex:1;min-width:0;white-space:normal}
.sa-news-voice{flex:none;margin-left:6px;width:30px;height:30px;border-radius:50%;border:1px solid rgba(255,255,255,.35);
  background:rgba(255,255,255,.08);color:#fff;font-size:15px;line-height:1;cursor:pointer;padding:0}
.sa-news-voice:hover{background:rgba(255,255,255,.18)}
.sa-news-voice.on{background:#f59e0b;border-color:#fcd34d;animation:sa-voice 1.2s ease-in-out infinite}
@keyframes sa-voice{0%,100%{box-shadow:0 0 0 0 rgba(252,211,77,.7)}50%{box-shadow:0 0 0 7px rgba(252,211,77,0)}}
.sa-news-reading .sa-news-page{animation:none!important;opacity:0!important;visibility:hidden!important}
.sa-news-reading .sa-news-page.sa-cur{opacity:1!important;visibility:visible!important}
.sa-news-item.sa-said .sa-news-t{color:#fde68a}
.sa-news-ago{flex:none;color:#94a3b8;font-size:11.5px;font-weight:500;white-space:nowrap}
/* one stock's news, above its chart */
.sa-snews{border:1px solid #e2e8f0;border-radius:8px;padding:8px 12px;margin:6px 0 8px;background:#fff}
.sa-snews-h{font-size:12px;font-weight:700;color:#334155;margin-bottom:4px}.sa-snews-h span{font-weight:500;color:#64748b}
.sa-snews a{display:flex;gap:8px;align-items:baseline;padding:3px 0;font-size:13.5px;color:#0f172a!important;text-decoration:none!important;border-top:1px solid #f1f5f9}
.sa-snews a:first-of-type{border-top:0}.sa-snews a:hover .t{text-decoration:underline}
.sa-snews .t{flex:1;font-weight:500}.sa-snews .s{color:#0f766e;font-size:11.5px;font-weight:600;white-space:nowrap}
.sa-snews .a{color:#64748b;font-size:11.5px;white-space:nowrap}
/* ---- phones and tablets ---- */
@media (hover: none) { .sa-pop { display: none !important; } .sa-kpi:hover { transform: none; } }   /* touch: no hover popups */
@media (max-width: 900px) {
  [data-testid="stMainBlockContainer"] { padding-left: 0.8rem !important; padding-right: 0.8rem !important; }
  /* title row: title, chip and Refresh wrap; the tab menu drops below it and scrolls sideways */
  [data-testid="stLayoutWrapper"]:has(.sa-brand) [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 0.5rem; }
  [data-testid="stLayoutWrapper"]:has(.sa-brand) [data-testid="stColumn"] { min-width: 0 !important; flex: 1 1 auto !important; width: auto !important; }
  [data-testid="stLayoutWrapper"]:has(.sa-brand) [data-testid="stColumn"]:nth-child(3) { display: none; }
  [data-testid="stLayoutWrapper"]:has(.sa-brand) [data-testid="stColumn"]:nth-child(2) { order: 5; flex: 1 1 100% !important; overflow-x: auto; }
  [data-testid="stLayoutWrapper"]:has(.sa-brand) [role="radiogroup"] { flex-wrap: nowrap; }
  [data-testid="stLayoutWrapper"]:has(.sa-brand) [role="radiogroup"] label { white-space: nowrap; }
  [data-baseweb="tab-list"] { flex-wrap: nowrap; }
  [data-baseweb="tab"] { white-space: nowrap; }
  .sa-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; gap: 0.5rem; }
  .sa-news { flex-wrap: wrap; padding: 8px 10px; gap: 8px; }
  .sa-news-box { flex-basis: 100%; order: 3; }
  .sa-news-voice { margin-left: auto; width: 36px; height: 36px; }
  .sa-news-item { font-size: 13.5px; flex-wrap: wrap; row-gap: 2px; }
  .sa-news-ago { flex-basis: 100%; }
  [data-testid="stDataFrame"], [data-testid="stTable"], .sa-card { max-width: 100%; overflow-x: auto; }
  .sa-kv-grid { min-width: 0; }
  iframe { max-width: 100%; }
  [data-testid="stPlotlyChart"] .modebar-container { display: none; }   /* Plotly charts: no toolbar on phones */
}
@media (max-width: 640px) {      /* rows of widgets wrap instead of squeezing (e.g. the portfolio chart's toggles) */
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; row-gap: 0.4rem; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width: min(100%, 120px) !important; flex: 1 1 auto !important; width: auto !important; }
}
@media (max-width: 480px) {
  .sa-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
  .sa-kpi { padding: 0.45rem 0.5rem 0.3rem; }
  .sa-kpi-value { font-size: 1.05rem; }
  .sa-kpi-label { font-size: 0.62rem; }
  .sa-kpi-foot { font-size: 9px; }
  .sa-brand-sub { display: none; }
  [data-testid="stLayoutWrapper"]:has(.sa-brand) [data-testid="stColumn"]:nth-child(1) { flex: 1 1 100% !important; }
  [data-testid="stLayoutWrapper"]:has(.sa-brand) [data-testid="stColumn"]:nth-child(3) { flex: 1 1 auto !important; }
  [data-testid="stLayoutWrapper"]:has(.sa-brand) [data-testid="stColumn"]:nth-child(4) { flex: 0 0 auto !important; }
  [data-testid="stLayoutWrapper"]:has(.sa-brand) .sa-chip { font-size: 0.68rem; }
  [data-testid="stLayoutWrapper"]:has(.sa-brand) button { padding: 0.25rem 0.8rem; min-height: 0; }
  [data-baseweb="tab"] { padding: 0.35rem 0.55rem !important; font-size: 0.85rem; }
}
</style>
"""

CHART_FONT = "Inter, system-ui, sans-serif"
PCT_COL = st.column_config.NumberColumn(format="%.2f%%")

MA_COLS = ['ma_10', 'ma_30', 'ma_50', 'ma_100', 'ma_200']
MA_COLORS = ['#0ea5e9', '#8b5cf6', '#f59e0b', '#a16207', '#94a3b8']   # chart lines: sky, violet, amber, brown, slate
TEAL = "#0f766e"
EARN_LINE = dict(color="rgba(234,88,12,0.45)", width=1, dash="dot")  # dotted vertical line at each earnings date

esc = html.escape

def show_html(markup):
    st.markdown("".join(line.strip() for line in str(markup).splitlines()), unsafe_allow_html=True)

def section(title):
    show_html(f'<div class="sa-section">{esc(title)}</div>')

def num(v):
    return float(v) if pd.notna(v) else None

def fmt(v, spec, prefix="", suffix=""):
    return f"{prefix}{v:{spec}}{suffix}" if v is not None else "—"

# One color rule for both tabs: green = good / Bullish / buy / positive, yellow = hold / neutral / caution / warning,
# red = bad / Bearish / sell / negative / error. Plain labels and headings stay slate (INK).
GOOD, CAUTION, BAD, INK = "#15803d", "#b45309", "#b91c1c", "#0f172a"
DD_RED = -10.0           # max drawdown: down to -10% yellow, worse than -10% red (0 = none yet, plain)
SCORE_GOOD = 40          # scores (strategy, technical, strength): above 40 green (about top-20 level), 0-40 yellow, below 0 red
TONE_WORDS = ((GOOD, ("buy", "bull", "fresh", "ok", "success", "add")),
              (BAD, ("sell", "sold", "bear", "score below", "missing", "error", "fail", "drop", "exit")),
              (CAUTION, ("hold", "keep", "watch", "neutral", "stale", "warn", "wait", "pending", "swap", "skip", "earn", "within")))

def tone(v, good=0.0, bad=0.0):
    """Color of a value. Text: by its first word (Buy / Hold / Sold, fresh / stale / missing...; other text stays plain).
    Number: above `good` green, below `bad` red, in between yellow (good == bad: exactly that value stays plain)."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return INK
    if isinstance(v, str):
        t = re.sub(r"^[^a-z]+", "", v.strip().lower())
        return next((c for c, words in TONE_WORDS if t.startswith(words)), INK)
    v = float(v)
    return GOOD if v > good else BAD if v < bad else (INK if good == bad else CAUTION)

def score_tone(v):
    return tone(v, SCORE_GOOD, 0.0)

def toned(df, cols, fn=None, **kw):
    """DataFrame (or Styler) -> Styler with `cols` colored by fn (default tone(v, **kw)); st.dataframe keeps the
    column_config formats."""
    sty = df if isinstance(df, Styler) else df.style
    fn = fn or (lambda v: tone(v, **kw))
    css = lambda v: "" if fn(v) == INK else f"color: {fn(v)}; font-weight: 600"
    return sty.map(css, subset=[c for c in cols if c in sty.data.columns])

def drawdown_tone(v):
    return INK if v is None or pd.isna(v) or v == 0 else tone(v, 0.0, DD_RED)

def live_row(sty, col, name):
    """Highlight the row whose `col` starts with `name` (the live rules) in the teal theme."""
    return sty.apply(lambda r: ["background-color: #f0fdfa; font-weight: 700" if str(r[col]).startswith(name) else ""] * len(r),
                     axis=1)

def md_tone(text, color):
    """Markdown text in the tone color (st.caption / st.markdown color syntax)."""
    return {GOOD: f":green[{text}]", CAUTION: f":orange[{text}]", BAD: f":red[{text}]"}.get(color, text)

def caption_text(text):
    """st.caption with literal dollar signs (Streamlit reads $...$ as math)."""
    st.caption(str(text).replace("$", "\\$"))

def warning_text(text):
    st.warning(str(text).replace("$", "\\$"))

def info_text(text):
    st.info(str(text).replace("$", "\\$"))

def stat_cards(items):
    """A row of card-style numbers: [(label, value text, color)]."""
    show_html('<div class="sa-stats">' + "".join(
        f'<div class="sa-stat"><div class="sa-stat-label">{esc(lbl)}</div><div class="sa-stat-val" style="color:{c}">{esc(v)}</div></div>'
        for lbl, v, c in items) + '</div>')

def usd(v, sign=True):
    if v is None or v != v:
        return "—"
    return f"{'+' if sign and v > 0.005 else '-' if v < -0.005 else ''}${abs(v):,.2f}"

MONEY = st.column_config.NumberColumn(format="dollar")
SPCT, INT = st.column_config.NumberColumn(format="%+.2f%%"), st.column_config.NumberColumn(format="%d")
SCORE, SHARES = st.column_config.NumberColumn(format="%.1f"), st.column_config.NumberColumn(format="%.4g")

def table(df, tone=(), fit=False, max_h=None, cols=None, **kw):
    """Every dashboard table: st.dataframe (sortable, searchable) with no index, full width, `tone` columns colored,
    height fitted to every row (fit) and/or capped at max_h; returns the selection event."""
    n = len(getattr(df, "data", df))
    if fit or max_h:
        kw["height"] = min(max_h or 10 ** 6, 35 * (n + 1) + 3)
    return st.dataframe(toned(df, list(tone)) if tone else df, hide_index=True, width="stretch", column_config=cols, **kw)

def setup_page():
    """The page config (the first Streamlit call) and the CSS, on every run."""
    st.set_page_config(page_title="Stock Analysis", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
