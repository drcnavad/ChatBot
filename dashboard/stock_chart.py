"""Home tab: the single-stock price chart (strategy score, relative strength, RSI / MACD)."""
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.data import last_next_earnings, live_quote, load_decisions, row_for
from dashboard.history import (RS_COLORS, daily_status, event_hover,
    relative_strength_lines, strategy_events)
from dashboard.settings import CT, HOLDINGS_CSV, W_TECH
from dashboard.style import (BAD, CHART_FONT, EARN_LINE, GOOD, HOLD_SHADE, MA_COLORS, MA_COLS, TEAL, num)


def build_price_chart(ticker, tdata, show_strategy):
    """Last 12 months: price + moving averages + buy/sell markers, the score panel (when ranked), RSI/MACD and relative
    strength panels."""
    chart = tdata[tdata['Date'] >= tdata['Date'].max() - pd.Timedelta(days=365)].sort_values('Date').set_index('Date')
    x_start, x_end = chart.index[0], chart.index[-1]
    events, periods = strategy_events(tdata, load_decisions(), ticker)
    rs_lines = relative_strength_lines(chart, ticker)

    panels = ["price"] + (["score"] if show_strategy else []) + ["rsi", "macd"] + (["rs"] if rs_lines else [])
    height_of = {"price": 0.44, "score": 0.20, "rs": 0.18, "rsi": 0.11, "macd": 0.11}
    titles = {
        "price": "<b>Price</b>",
        "score": f"<b>Strategy score</b> (teal) = {W_TECH:g} × Technical (grey) + {1 - W_TECH:g} × Strength vs sector/SPY (violet)",
        "rs": ("<b>Performance since " + f"{x_start:%b %-d, %Y}" + "</b> (% price change): " + " · ".join(
            f'<span style="color:{RS_COLORS[k]};">━ {name}</span>' for k, (name, _s) in rs_lines.items())),
        "rsi": "<b>RSI</b> (above 70 = stretched up, below 30 = stretched down)", "macd": "<b>MACD</b> (indigo) vs signal (dashed)",
    }
    heights = [height_of[x] for x in panels]
    row_of = {x: i + 1 for i, x in enumerate(panels)}
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[h / sum(heights) for h in heights], subplot_titles=[titles[x] for x in panels])
    fig.update_annotations(font=dict(size=12, color='#334155', family=CHART_FONT), yshift=4)

    # Price line (hover shows the day's status), earnings dates, moving averages
    status_txt = []
    for w, r, s in zip(chart['Strategy_Weight'], chart['Strategy_Rank'], chart['Strategy_Score']):
        sig = daily_status(w, s)
        bits = [sig] + ([f"weight {w * 100:.1f}%"] if sig == "Hold" else []) \
            + ([f"rank #{r:.0f}"] if pd.notna(r) else []) + ([f"score {s:.1f}"] if pd.notna(s) else [])
        status_txt.append(" · ".join(bits))
    fig.add_trace(go.Scatter(x=chart.index, y=chart['Close'], name='Close', line=dict(color=TEAL, width=2.5), mode='lines',
                             customdata=status_txt, hovertemplate='<b>Close</b> $%{y:.2f}<br>%{customdata}<extra></extra>',
                             showlegend=False), row=1, col=1)
    earnings = chart[chart['is_earnings_date'] == 1]
    fig.add_trace(go.Scatter(x=earnings.index, y=earnings['Close'], name='Earnings date', mode='markers',
                             marker=dict(symbol='circle', size=9, color='#f97316'),
                             hovertemplate='<b>Earnings</b> %{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>',
                             showlegend=False), row=1, col=1)
    for d in earnings.index:
        fig.add_vline(x=d, line=EARN_LINE, row=1, col=1)
    fig.add_trace(go.Scatter(x=[x_start], y=[None], mode="lines", name="Earnings (dotted line; next one ahead)",
                             line=EARN_LINE, hoverinfo="skip", showlegend=False), row=1, col=1)
    for ma, color in zip(MA_COLS, MA_COLORS):
        name = ma.upper().replace('_', ' ')
        fig.add_trace(go.Scatter(x=chart.index, y=chart[ma], name=name, line=dict(color=color, width=1), mode='lines',
                                 hovertemplate=f'<b>{name}</b> $%{{y:.2f}}<extra></extra>'), row=1, col=1)
    if periods:
        fig.add_trace(go.Scatter(x=[x_start], y=[None], mode="markers", name="Hold (shaded period)", hoverinfo="skip",
                                 marker=dict(symbol="square", size=12, color="rgba(245,158,11,0.35)"),
                                 showlegend=False), row=1, col=1)

    # Entry / exit markers on the session after the decision (hollow = decided at the latest close, orders pending)
    shown = events[events['Fill'].fillna(x_end) >= x_start] if len(events) else events
    for kind, marker, color, name in (("entry", "triangle-up", GOOD, "Buy (strategy decision)"),
                                      ("exit", "triangle-down", BAD, "Sold (strategy decision)")):
        e = shown[shown['Kind'] == kind] if len(shown) else shown
        if e.empty:
            continue
        pending = e['Fill'].isna()
        fig.add_trace(go.Scatter(
            x=e['Fill'].fillna(x_end), y=e['Price'], mode='markers', name=name,
            marker=dict(symbol=[marker + ("-open" if x else "") for x in pending], size=13, color=color,
                        line=dict(width=1.5, color=color if pending.any() else "#ffffff")),
            hovertext=[event_hover(x) for x in e.itertuples()], hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False), row=1, col=1)

    # While held: entry price dotted line
    hold = row_for(HOLDINGS_CSV, ticker) if (num(chart['Strategy_Weight'].iloc[-1]) or 0) > 0 else None
    if hold is not None and pd.notna(hold['Entry_Price']):
        fig.add_hline(y=float(hold['Entry_Price']), line=dict(color='#64748b', width=1, dash='dot'), row=1, col=1,
                      annotation_text=f"entry ${hold['Entry_Price']:,.2f}", annotation_position="top left",
                      annotation_font=dict(size=10, color='#64748b'))

    # Next earnings date (extends the x-axis when it is within ~2 months)
    x_right = x_end
    ned = last_next_earnings([ticker])['Next ED'].iloc[0]
    if ned and pd.Timestamp(ned) - x_end <= pd.Timedelta(days=62):
        ned = pd.Timestamp(ned)
        x_right = max(x_end, ned) + pd.Timedelta(days=4)
        fig.add_vline(x=ned, line=EARN_LINE | dict(width=1.5), row="all", col=1)
        fig.add_annotation(x=ned, y=1, xref="x", yref="y domain", text=f"next earnings {ned:%b %-d}", showarrow=False,
                           yanchor="bottom", xanchor="right", font=dict(size=10, color='#c2410c'))

    # Live price marker (display only): a single dot at today's date so the price panel reaches the current
    # session. Bars, moving averages, and every strategy panel still end at the last completed bar.
    live = live_quote(ticker)
    if live is not None:
        lp, lts = live
        lts = pd.Timestamp(lts)
        if lts.tzinfo is not None and lts.tz_convert(CT).date() == datetime.now(tz=CT).date():
            live_day = lts.tz_convert(CT).normalize().tz_localize(None)
            fig.add_trace(go.Scatter(
                x=[live_day], y=[lp], name="Live", mode="markers",
                marker=dict(symbol="circle", size=8, color=TEAL, line=dict(width=2, color="#ffffff")),
                hovertemplate="<b>Live</b> $%{y:.2f}<br>%{x|%b %d, %Y} · intraday, display only<extra></extra>",
                showlegend=False),
                row=1, col=1)
            x_right = max(x_right, live_day) + pd.Timedelta(days=2)

    if "score" in row_of:  # score panel
        r = row_of["score"]
        for col, name, color, width, dash in (("Technical_Score", "Technical", "#9ca3af", 1.25, "dot"),
                                              ("RS_Score", "Strength vs sector/SPY", "#8b5cf6", 1.5, "solid"),
                                              ("Strategy_Score", "Strategy score", TEAL, 2, "solid")):
            fig.add_trace(go.Scatter(x=chart.index, y=chart[col], name=name, mode='lines', line=dict(color=color, width=width, dash=dash),
                                     showlegend=False, hovertemplate=f'<b>{name}</b> %{{y:.1f}}<extra></extra>'), row=r, col=1)
        fig.add_hline(y=0, line=dict(color='rgba(100,116,139,0.5)', width=1, dash='dot'), row=r, col=1)
    if "rs" in row_of:
        r = row_of["rs"]
        for key, (label, series) in rs_lines.items():
            fig.add_trace(go.Scatter(x=series.index, y=series, name=label, mode='lines',
                                     line=dict(color=RS_COLORS[key], width=2.25 if key == "stock" else 1.5),
                                     showlegend=False, hovertemplate=f'<b>{label}</b> %{{y:+.1f}}%<extra></extra>'), row=r, col=1)
        fig.add_hline(y=0, line=dict(color='rgba(100,116,139,0.5)', width=1, dash='dot'), row=r, col=1)
    if "rsi" in row_of:
        r = row_of["rsi"]
        fig.add_trace(go.Scatter(x=chart.index, y=chart['RSI'], name='RSI', line=dict(color='#0ea5e9', width=1.5), mode='lines',
                                 showlegend=False, hovertemplate='<b>RSI</b> %{y:.1f}<extra></extra>'), row=r, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(185,28,28,0.35)", row=r, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(21,128,61,0.35)", row=r, col=1)
        fig.update_yaxes(range=[0, 100], row=r, col=1)
        r = row_of["macd"]
        fig.add_trace(go.Scatter(x=chart.index, y=chart['macd'], name='MACD', line=dict(color='#4f46e5', width=1.5), mode='lines',
                                 showlegend=False, hovertemplate='<b>MACD</b> %{y:.3f}<extra></extra>'), row=r, col=1)
        fig.add_trace(go.Scatter(x=chart.index, y=chart['MACD Signal'], name='MACD signal', line=dict(color='#94a3b8', width=1.5, dash='dash'),
                                 mode='lines', showlegend=False, hovertemplate='<b>Signal</b> %{y:.3f}<extra></extra>'), row=r, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(128, 128, 128, 0.4)", row=r, col=1)

    # Held periods shaded on all panels (added last, with exclude_empty_subplots=False, or plotly drops them)
    for start, end in periods:
        if end >= x_start:
            fig.add_vrect(x0=max(start, x_start), x1=end, fillcolor=HOLD_SHADE, line_width=0, layer="below",
                          row="all", col=1, exclude_empty_subplots=False)

    grid = dict(showgrid=True, gridcolor='#eef2f7', showline=True, linecolor='#e2e8f0', tickfont=dict(size=10, color='#64748b'),
                zeroline=False)
    fig.update_layout(
        height=int(560 + 170 * (len(panels) - 1)), hovermode='x unified', margin=dict(l=50, r=30, t=90, b=64),
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', dragmode=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5, font=dict(size=10, color='#334155'),
                    title=dict(text="<b>Legend</b>  ", font=dict(size=10, color='#64748b')),
                    bgcolor='#f8fafc', bordercolor='#e2e8f0', borderwidth=1),
        font=dict(family=CHART_FONT, size=11, color='#334155'),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e2e8f0", font_size=11, font_family=CHART_FONT))
    fig.update_xaxes(type="date", range=[x_start, x_right], showspikes=True, spikemode="across", spikethickness=1, spikecolor="#94a3b8",
                     dtick=7 * 24 * 60 * 60 * 1000, tickformat='%b %d', tickangle=-45,
                     **(grid | dict(showgrid=False)))
    fig.update_yaxes(**grid)
    fig.update_yaxes(title_text="Price ($)", tickformat='$,.0f', row=1, col=1)
    return fig


def stock_chart_inputs(ticker, tdata):
    """The built figure (displayed later, below the detail block); the score panel only for a ranked stock."""
    return build_price_chart(ticker, tdata, bool(tdata['Strategy_Score'].notna().any()))


def render_stock_figure(fig, ticker):
    """The price chart itself, under the always-open detail block."""
    st.plotly_chart(fig, width="stretch", config={
        'displaylogo': False, 'scrollZoom': False, 'doubleClick': 'reset',
        'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'autoScale2d', 'zoomIn2d', 'zoomOut2d']})
