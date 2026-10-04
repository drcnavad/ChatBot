"""Dashboard tab: the single-stock view (rank tiers, stock picker, header, detail card)."""
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from dashboard.ai import ai_analysis_dialog
from dashboard.data import (company_metrics, last_next_earnings, live_quote, load_benchmarks, load_company, load_news,
    row_for)
from dashboard.settings import CT, HOLDINGS_CSV, SECTOR_MAX, symbol_sector
from dashboard.signals import PLAN_BADGE, SIGNAL_BADGE, plan_text
from dashboard.style import (BAD, CAUTION, GOOD, INK, MA_COLS, SCORE_GOOD, TONE_CLASS, esc, fmt, num, score_tone,
    show_html, sign_color, stat_html, symbol_link, tone)


def ticker_label(p, s):
    if s in p.short.index:
        return f"{s}  ·  not traded yet (short history)  ·  rough {p.short.loc[s, 'Rough_Signal']} (less reliable)  ·  not ranked"
    r = p.by_symbol.loc[s]
    rank, score = r.get('Strategy_Rank'), r['combined_signal']
    parts = [s, f"{p.sig_off.get(s, 'Not ranked')} ({p.off_date:%b %-d})"]
    if pd.notna(rank):
        parts.append(f"rank #{rank:.0f}")
    parts.append(f"score {score:.0f}" if pd.notna(score) else "score —")
    return "  ·  ".join(parts)


def render_rank_tiers(p):
    """Horizontal clickable rank tiers: Rank 1 to 20, Rank 21 to 50, Rank 51+.

    Each symbol is a ?symbol= link, handled in main() exactly like a table click
    (opens the stock in the stock view below). The dropdown underneath stays for manual typing."""
    la = p.by_symbol
    ranked = la[la["Strategy_Rank"].notna()].sort_values("Strategy_Rank")
    tiers = [("Rank 1 to 20", ranked[ranked["Strategy_Rank"] <= 20]),
             ("Rank 21 to 50", ranked[(ranked["Strategy_Rank"] > 20) & (ranked["Strategy_Rank"] <= 50)]),
             ("Rank 51+", ranked[ranked["Strategy_Rank"] > 50])]
    current = st.session_state.get("ticker_dropdown")
    held = lambda s: TONE_CLASS.get(tone(p.sig_off.get(s)), "") if p.sig_off.get(s) in ("Buy", "Hold", "Sold") else ""
    rows = []
    for title, df in tiers:
        links = " ".join(symbol_link(s, held(s) + (" sel" if s == current else "")) for s in df.index)
        rows.append(f'<div class="sa-tier"><b>{title}:</b><div class="sa-tier-syms">{links or "\u2014"}</div></div>')
    if len(p.short):
        links = " ".join(f'{symbol_link(s)} <span style="color:{tone(r["Rough_Signal"])};">(rough {esc(str(r["Rough_Signal"]))}, '
                         'less reliable)</span>' for s, r in p.short.iterrows())
        rows.append(f'<div class="sa-tier"><b>Not traded yet (short history, not ranked):</b><div class="sa-tier-syms">{links}</div></div>')
    day = p.df["Date"].max()
    show_html(f'<div class="sa-card"><div class="sa-card-title">Stock list<span>ranks at the {day:%a %b %-d} close · '
              'click a stock to open it</span></div>' + "".join(rows)
              + f'<div class="sa-note">Color = the decision in force ({p.off_date:%b %-d}): <b style="color:{GOOD};">green Buy</b> · '
              f'<b style="color:{CAUTION};">yellow Hold</b> · <b style="color:{BAD};">red Sold</b> · grey = not in the portfolio. '
              'Teal outline = the stock shown below.</div></div>')


def render_stock_picker(p, jumped):
    """Rank tiers (click a symbol to open it) + dropdown for manual typing + AI button.

    ?symbol=X, a table click, a tier click or the dropdown choose the stock."""
    pending = st.session_state.pop("_pending_ticker", None)
    if pending in p.options:
        st.session_state.ticker_dropdown = pending
    elif st.session_state.get("ticker_dropdown") not in p.options:
        st.session_state.ticker_dropdown = p.options[0]
    st.markdown('<div id="ticker-focus"></div>', unsafe_allow_html=True)
    if jumped or pending:  # opened from a link or a table: scroll the stock view into sight
        components.html("<script>const el = window.parent.document.getElementById('ticker-focus');"
                        "if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'});</script>", height=0)
    render_rank_tiers(p)
    pick_col, ai_col = st.columns([3, 1])
    ticker = pick_col.selectbox("Ticker", options=p.options, format_func=lambda s: ticker_label(p, s),
                                key="ticker_dropdown", label_visibility="collapsed")
    tdata = p.df[p.df['Symbol'] == ticker]
    if ticker not in p.short.index and ai_col.button("Generate AI Analysis", type="primary", width="stretch", key="generate_ai_btn"):
        ai_analysis_dialog(ticker, tdata, p.sig_off.get(ticker, 'Not ranked'), p.why_off.get(ticker, ''))
    return ticker, tdata


def render_stock_header(p, ticker, tdata):
    """Name, price, official signal and the numbers that decide it (score, rank, slot, weight)."""
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
                f'<span style="color:{sign_color(chg)};">{fmt(chg, "+.2f", suffix="%")}</span>'
                f' <span class="sa-live-sub">vs {pd.Timestamp(latest["Date"]):%a} close · '
                f'{lts.tz_convert(CT).strftime("%I:%M %p")} CT</span></div>'
            )
    score, rank = num(latest.get('Strategy_Score')), num(latest.get('Strategy_Rank'))
    weight = num(latest.get('Strategy_Weight')) or 0.0
    n_ranked = int(p.latest['Strategy_Score'].notna().sum())
    slot = p.slot_off.get(ticker, "—")
    status, why = p.sig_off.get(ticker, "Not ranked"), p.why_off.get(ticker, "")
    chips = []
    if latest['final_trade'] == 'EARNING':
        chips.append(("Earnings within 2 sessions", "sa-badge-hold"))
    chips_html = "".join(f'<span class="sa-badge {c}" style="font-weight:600;font-size:0.72rem;">{esc(t)}</span>' for t, c in chips)
    next_ed = last_next_earnings([ticker])['Next ED'].iloc[0]
    day = pd.Timestamp(latest["Date"])
    rank_then = p.rank_off.get(ticker)
    badge = f"Strategy decision · {p.off_date:%a %b %-d}: {status}" + (f" ({p.tag_off[ticker]})" if p.tag_off.get(ticker) else "")
    plan_html = ""
    if ticker in symbol_sector or ticker in p.plan:                      # tradable stocks only (not the QQQ benchmark)
        action = p.plan.get(ticker, ("not picked",))[0]
        when = f"{p.plan_day:%a %b %-d}" if p.plan_day is not None else "Next rebalance"
        tip = (f"Full rebalance computed at the {p.plan_asof:%a %b %-d} close (strategy_picks.csv, the numbers the trade "
               f"step uses). Final at the {when} close; orders go out that day at 2:30 PM CT." if p.plan_asof is not None else "")
        plan_html = (f'<span class="sa-badge {PLAN_BADGE.get(action, "sa-badge-grey")}" title="{esc(tip)}">'
                     f'{esc(when + " plan: " + plan_text(p.plan, ticker))}</span>')
    stats = [
        stat_html("Strategy score", fmt(score, ".1f"), score_tone(score)),
        stat_html(f"Strategy rank · {day:%a %b %-d} close", f"#{rank:.0f} / {n_ranked}" if rank is not None else "—",
                  tip=f"Position by strategy score among the {n_ranked} ranked stocks at the {day:%a %b %-d} close (1 = best)"
                      + (f". At the {p.off_date:%a %b %-d} decision it was #{rank_then:.0f}." if rank_then is not None
                         and pd.notna(rank_then) else "")),
        stat_html(f"Portfolio slot · {p.off_date:%b %-d}", f"{slot} of 10" if slot != "—" else "— (not picked)",
                  tip="Position among the 10 stocks picked at the last decision, in rank order. The picks skip stocks "
                      f"whose sector already has {SECTOR_MAX}, so the slot can be smaller than the rank."),
        stat_html("Portfolio weight", fmt(weight * 100 if weight > 0 else None, ".1f", suffix="%")),
        stat_html("Technical", fmt(num(latest.get('Technical_Score')), ".1f"), score_tone(num(latest.get('Technical_Score')))),
        stat_html("Strength vs sector/SPY", fmt(num(latest.get('RS_Score')), ".1f"), score_tone(num(latest.get('RS_Score')))),
        stat_html("Next earnings", pd.Timestamp(next_ed).strftime("%b %d") if next_ed else "—"),
    ]
    hold = row_for(HOLDINGS_CSV, ticker) if weight > 0 else None
    if hold is not None:
        stats += [
            stat_html("Held since", f"{pd.Timestamp(hold['Entry_Date']):%b %-d} · {int(hold['Days_Held'])} sessions"
                      if pd.notna(hold['Entry_Date']) else "—"),
            stat_html("P&L since entry", fmt(num(hold['PnL_%']), "+.1f", suffix="%"), sign_color(num(hold['PnL_%']))),
        ]
    show_html(f"""
        <div class="sa-hero">
          <div class="sa-ident">
            <div class="sa-sym">{esc(ticker)}</div>
            <div class="sa-price" title="Latest completed daily bar — the strategy's official price">{fmt(num(latest['Close']), ",.2f", "$")}</div>
            {live_html}
            <span class="sa-badge {SIGNAL_BADGE.get(status, 'sa-badge-grey')}" title="The strategy's decision in force (last decision)">{esc(badge)}</span>
            {plan_html}
            {chips_html}
          </div>
          <div class="sa-why">{esc(status + ": " + why) if why else ""}</div>
          <div class="sa-stats">{"".join(stats)}</div>
          <div class="sa-note">Scores: <b style="color:{GOOD};">green above {SCORE_GOOD}</b> · <b style="color:{CAUTION};">yellow 0 to {SCORE_GOOD}</b> ·
            <b style="color:{BAD};">red below 0</b> (only stocks above 0 can be picked). Rank 1 = best. P&amp;L and returns: green up, red down.</div>
        </div>""")


def _detail_group(title, stats, note=""):
    """One titled row of stat tiles inside the stock detail card; stats = [(label, value, color)]."""
    return (f'<div class="sa-group"><div class="sa-group-title">{esc(title)}</div><div class="sa-stats" style="margin-top:0.5rem;">'
            + "".join(stat_html(l, "\u2014" if v is None else v, c) for l, v, c in stats) + "</div>"
            + (f'<div class="sa-why">{esc(note)}</div>' if note else "") + "</div>")


def render_stock_more(ticker, tdata, has_strategy):
    """Always-open detail card above the chart: moving averages, fundamentals/news, per-stock forward test."""
    latest = tdata.nlargest(1, 'Date').iloc[0]
    close = num(latest['Close'])
    ma_stats = [(ma.upper().replace('_', ' '), fmt(num(latest[ma]), ",.2f", "$"),     # green = the close is above it
                 tone(close - num(latest[ma])) if close and num(latest[ma]) else INK) for ma in MA_COLS]

    # Context: latest-day values only, NOT part of the backtested rules
    company_df = load_company()
    comp = company_metrics(ticker, company_df)
    fv = comp.get('fair_value')
    upside = (fv / close - 1) * 100 if fv and close else None
    sentiment = num(latest['SentimentScore'])
    comp_row = company_df[company_df['Symbol'].astype(str).str.upper() == ticker] if 'Symbol' in company_df else pd.DataFrame()
    fiscal = pd.to_datetime(comp_row['FiscalDateEnding'].iloc[0], errors='coerce') \
        if len(comp_row) and 'FiscalDateEnding' in comp_row else pd.NaT
    news = load_news()
    news_dates = pd.to_datetime(news.loc[news['symbol'] == ticker, 'date'], utc=True, format='mixed', errors='coerce')
    last_news = news_dates.max() if len(news_dates) else pd.NaT
    fund_stats = [
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
    fund_note = ("Context only (latest day, not part of the backtested rules) \u00b7 fundamentals: "
                 + (f"quarter ending {fiscal:%Y-%m-%d}" if pd.notna(fiscal) else "none on file")
                 + (f" \u00b7 newest relevant news {last_news:%b %-d}" if pd.notna(last_news)
                    else " \u00b7 no relevant news in the last 10 days"))

    fw_stats, fw_note = [], ""
    if has_strategy:            # per-stock forward test: only sessions from FORWARD_START on count
        from backtest_engine import FORWARD_START, forward_test
        s_ = tdata.sort_values('Date').set_index('Date')
        fw = forward_test(s_['Close'], s_['Strategy_Weight'], FORWARD_START)
        start_txt, since = f"{fw['Start']:%b %-d, %Y}", f"since {fw['Start']:%b %-d}"
        bench = load_benchmarks()
        q = bench["QQQ"].loc[fw["Start"]:].dropna() if bench is not None and "QQQ" in bench else pd.Series(dtype=float)
        qqq = (q.iloc[-1] / q.iloc[0] - 1) * 100 if len(q) else None
        n, ot, med, bnh = fw["Closed trades"], fw["Open trade"], fw["Median trade %"], fw["Buy & hold %"]
        fw_stats = [("Start", start_txt, INK), ("Closed trades", f"{n}", INK),
                    ("Open trade", f"{ot['Entry']:%b %-d} @ ${ot['Price']:,.2f} · {ot['Change %']:+.2f}%" if ot else "none",
                     sign_color(ot["Change %"]) if ot else INK)]
        if n:
            fw_stats += [("Win rate", fmt(fw["Win rate %"], ".0f", suffix="%"), tone(fw["Win rate %"], 50, 50)),
                         ("Median trade", fmt(med, "+.2f", suffix="%"), sign_color(med)),
                         ("Median hold", f"{fw['Median hold (sessions)']:.0f} sessions", INK)]
        fw_stats += [("Held % of sessions", fmt(fw["Held % of sessions"], ".0f", suffix="%"), INK),
                     (f"Buy & hold {since}", fmt(bnh, "+.2f", suffix="%"), sign_color(bnh)),
                     (f"QQQ {since}", fmt(qqq, "+.2f", suffix="%"), sign_color(qqq))]
        fw_note = (("" if n else f"Forward test started {start_txt}; no closed trades yet. ")
                   + f"Live rules from the {start_txt} close on: only sessions from then count (earlier history only "
                   "warms up the indicators); trades at the decision-day close, 0.1%/side. QQQ is not traded, "
                   "comparison only.")

    groups = _detail_group("Moving averages (green = the close is above it)", ma_stats)
    groups += _detail_group("Fundamentals & news", fund_stats, fund_note)
    if fw_stats:
        groups += _detail_group("Per-stock forward test", fw_stats, fw_note)
    show_html(f'<div class="sa-card"><div class="sa-card-title">More about {esc(ticker)}</div>' + groups + '</div>')
