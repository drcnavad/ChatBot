"""Stock Analysis tab: the single-stock view (tile grid, stock picker, header, detail card)."""
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from dashboard.ai import ai_analysis_dialog
from dashboard.data import (company_metrics, last_next_earnings, live_quote, load_company,
    row_for)
from dashboard.settings import CT, HOLDINGS_CSV, symbol_sector
from dashboard.signals import PLAN_BADGE, SIGNAL_BADGE, live_stock_note
from dashboard.style import (BAD, GOOD, INK, MA_COLS, esc, fmt, num, score_tone,
    show_html, tone)

def ticker_label(p, s):
    """Picker label: "JPM (JP Morgan)" (company name from sector_mapping.symbol_name). Typing in the picker filters
    on this label, so a ticker or a company name both find the stock."""
    import sector_mapping as sm
    name = sm.symbol_name.get(s, "")
    return f"{s} ({name})" if name else s

def render_stock_picker(p, jumped):
    """Stock list (click a symbol to open it) + dropdown for manual typing + AI button.

    ?symbol=X, a table click, a tier click or the dropdown choose the stock."""
    pending = st.session_state.pop("_pending_ticker", None)
    if "ticker_dropdown" not in st.session_state and st.session_state.get("_ticker") in p.options:
        st.session_state.ticker_dropdown = st.session_state["_ticker"]       # back from another page: same stock
    if pending in p.options:
        st.session_state.ticker_dropdown = pending
    elif st.session_state.get("ticker_dropdown") not in p.options:
        st.session_state.ticker_dropdown = p.options[0]
    st.markdown('<div id="ticker-focus"></div>', unsafe_allow_html=True)
    if jumped or pending:  # opened from a link or a table: scroll the stock view into sight
        components.html("<script>const el = window.parent.document.getElementById('ticker-focus');"
                        "if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'});</script>", height=0)
    from dashboard.heatmap import render_heatmap
    render_heatmap(p)                          # replaced the stock list (Oct 22)
    pick_col, ai_col = st.columns([3, 1])
    ticker = pick_col.selectbox("Ticker", options=p.options, format_func=lambda s: ticker_label(p, s),
                                key="ticker_dropdown", label_visibility="collapsed",
                                placeholder="Search a ticker or company name…")
    st.session_state["_ticker"] = ticker
    tdata = p.df[p.df['Symbol'] == ticker]
    if ticker not in p.short.index and ai_col.button("Generate AI Analysis", type="primary", width="stretch", key="generate_ai_btn"):
        ai_analysis_dialog(ticker, tdata, p.sig_off.get(ticker, 'Not ranked'), p.why_off.get(ticker, ''))
    return ticker, tdata

@st.fragment(run_every=30)   # only the card reruns: Alpaca price new every 30 s open / 5 min closed
def render_stock_section(p, ticker, tdata):
    """One unified stock card: header (name, price, signal, stats) flowing into moving averages and fundamentals/news."""
    from dashboard.page import fresh
    if fresh(p) is not p:                                   # new end-of-day data since the page run: use it in place
        p = fresh(p)
        tdata = p.df[p.df['Symbol'] == ticker]
    latest = tdata.nlargest(1, 'Date').iloc[0]
    # Display-only live quote: shown under the official bar close, never used by the strategy.
    live_html = ""
    live = live_quote(ticker)
    if live is not None:
        lp, lts = live
        lts = pd.Timestamp(lts)
        # Only label it LIVE when the quote is from today's session; otherwise the bar close above is already the latest.
        if lts.tzinfo is not None and lts.tz_convert(CT).date() == datetime.now(tz=CT).date():
            bar_close = num(latest['Close'])
            chg = (lp / bar_close - 1) * 100 if bar_close else None
            live_html = (
                f'<div class="sa-live" title="Live quote from yfinance (display only — the strategy uses the bar close above)">'
                f'<span class="sa-live-dot"></span>LIVE {fmt(lp, ",.2f", "$")} '
                f'<span style="color:{tone(chg)};">{fmt(chg, "+.2f", suffix="%")}</span>'
                f' <span class="sa-live-sub">vs {pd.Timestamp(latest["Date"]):%a} close · '
                f'{lts.tz_convert(CT).strftime("%I:%M %p")} CT</span></div>'
            )
    score = num(latest.get('Strategy_Score'))
    weight = num(latest.get('Strategy_Weight')) or 0.0
    # Live Short-term relative strength note (not the Oct-16 pipeline strategy_changes board)
    note = live_stock_note(ticker, p)
    status, why = note["status"], note["why"]
    chips = list(note.get("chips") or [])
    if latest['final_trade'] == 'EARNING':
        chips.append(("Earnings within 5 sessions (no new buy)", "sa-badge-hold"))
    chips_html = "".join(
        f'<span class="sa-badge {c}" style="font-weight:600;font-size:0.72rem;">{esc(t)}</span>'
        for t, c in chips)
    next_ed = last_next_earnings([ticker])['Next ED'].iloc[0]
    badge = note["badge"]
    plan_html = ""
    if ticker in symbol_sector or ticker in p.plan:                      # tradable stocks only (not the QQQ benchmark)
        action = note["plan_action"]
        plan_html = (f'<span class="sa-badge {PLAN_BADGE.get(action, "sa-badge-grey")}" '
                     f'title="{esc(note["plan_tip"])}">'
                     f'{esc(note["plan_text"])}</span>')
    stats = [
        ("Strategy score", fmt(score, ".1f"), score_tone(score)),
        ("Technical", fmt(num(latest.get('Technical_Score')), ".1f"), score_tone(num(latest.get('Technical_Score')))),
        ("Strength vs sector/SPY", fmt(num(latest.get('RS_Score')), ".1f"), score_tone(num(latest.get('RS_Score')))),
        ("Next earnings", pd.Timestamp(next_ed).strftime("%b %d") if next_ed else "—", INK),
    ]
    hold = row_for(HOLDINGS_CSV, ticker) if weight > 0 else None
    if hold is not None:
        stats += [
            ("Held since", f"{pd.Timestamp(hold['Entry_Date']):%b %-d}" if pd.notna(hold['Entry_Date']) else "—", INK),
            ("P&L since entry", fmt(num(hold['PnL_%']), "+.1f", suffix="%"), tone(num(hold['PnL_%']))),
        ]
    close = num(latest['Close'])
    ma_stats = [(ma.upper().replace('_', ' '), fmt(num(latest[ma]), ",.2f", "$"),     # green = the close is above it
                 tone(close - num(latest[ma])) if close and num(latest[ma]) else INK) for ma in MA_COLS]
    # Context: latest-day values only, NOT part of the backtested rules
    company_df = load_company()
    comp = company_metrics(ticker, company_df)
    fv = comp.get('fair_value')
    upside = (fv / close - 1) * 100 if fv and close else None
    sentiment = num(latest['SentimentScore'])
    # RSI & MACD colored by day-over-day trend (green = rising, red = falling)
    rsi_now, macd_now = num(latest.get('RSI')), num(latest.get('macd'))
    prev_day = tdata.nlargest(2, 'Date').iloc[-1] if len(tdata) > 1 else latest
    rsi_prev, macd_prev = num(prev_day.get('RSI')), num(prev_day.get('macd'))
    def _trend(now, prev_):
        if now is None or prev_ is None or now == prev_:
            return INK
        return GOOD if now > prev_ else BAD
    fund_stats = [
        ("RSI", fmt(rsi_now, '.1f'), _trend(rsi_now, rsi_prev)),
        ("MACD", fmt(macd_now, '.3f'), _trend(macd_now, macd_prev)),
        ("Balance sheet", fmt(num(latest['Fundamental_Weight']), ".2f"), tone(num(latest['Fundamental_Weight']))),
        ("Sentiment", fmt(sentiment, ".2f"), tone(sentiment)),
        ("Fair value", fmt(fv, ",.2f", "$"), INK),
        ("Upside", fmt(upside, "+.1f", suffix="%"), tone(upside)),
        ("P/E", fmt(comp.get('pe_ratio'), ".1f"), INK),
        ("P/B", fmt(comp.get('pb_ratio'), ".2f"), INK),
        ("Rev YoY", fmt(comp.get('revenue_growth_yoy'), ".1f", suffix="%"), tone(comp.get('revenue_growth_yoy'))),
        ("ROE", fmt(comp.get('roe'), ".1f", suffix="%"), tone(comp.get('roe'))),
        ("Net margin", fmt(comp.get('net_margin'), ".1f", suffix="%"), tone(comp.get('net_margin'))),
        ("Debt/Eq", fmt(comp.get('debt_to_equity'), ".2f"), INK),
    ]
    groups = ('<div class="sa-kv-grid">' + kv_table("Score", stats) + kv_table("Moving averages", ma_stats, "Green = the close is above it")
              + kv_table("Fundamentals & news", fund_stats, cols=2) + "</div>")
    show_html(f"""
        <div class="sa-card">
          <div class="sa-ident">
            <div class="sa-sym">{esc(ticker)}</div>
            <div class="sa-price" title="Latest completed daily bar — the strategy's official price">{fmt(num(latest['Close']), ",.2f", "$")}</div>
            {live_html}
            <span class="sa-badge {SIGNAL_BADGE.get(status, 'sa-badge-grey')}" title="Live Short-term relative strength status (rank + next-run plan)">{esc(badge)}</span>
            {plan_html}
            {chips_html}
          </div>
          <div class="sa-why">{esc(why) if why else ""}</div>
          {groups}
        </div>""")

def kv_table(title, rows, tip="", cols=1):
    """A compact label / value table (rows = [(label, value, color)]), `cols` pairs per line."""
    cells = [f'<td>{esc(l)}</td><td style="color:{c}">{esc("—" if v is None else str(v))}</td>' for l, v, c in rows]
    body = "".join("<tr>" + "".join(cells[i:i + cols]) + "</tr>" for i in range(0, len(cells), cols))
    return (f'<div class="sa-kv"><div class="sa-group-title" title="{esc(tip)}">{esc(title)}</div>'
            f'<table>{body}</table></div>')
