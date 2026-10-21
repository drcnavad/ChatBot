"""Home tab: the single-stock price chart (price, RSI / MACD, relative strength)."""
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.data import last_next_earnings, live_quote, load_decisions
from dashboard.history import (RS_COLORS, daily_status, event_hover,
    relative_strength_lines, strategy_events)
from dashboard.settings import CT
from dashboard.style import (BAD, CHART_FONT, EARN_LINE, GOOD, MA_COLORS, MA_COLS, TEAL)

WINDOW = pd.Timedelta(days=366 * 7)     # chart data: all cached daily bars, up to ~7 years (range buttons zoom inside)
RANGES = [dict(count=1, label="1M", step="month", stepmode="backward"), dict(count=3, label="3M", step="month", stepmode="backward"),
          dict(count=6, label="6M", step="month", stepmode="backward"), dict(count=1, label="1Y", step="year", stepmode="backward"),
          dict(count=2, label="2Y", step="year", stepmode="backward"), dict(count=5, label="5Y", step="year", stepmode="backward"),
          dict(step="all", label="ALL")]

def _with_bar_history(tdata, bars):
    """The signal rows (from Aug 2024) plus every older cached daily bar, with MAs / RSI / MACD worked out from the
    bars' closes where the signal file has none (same formulas: simple MAs, Wilder RSI 14, MACD 12/26/9)."""
    chart = tdata.sort_values('Date').drop_duplicates('Date').set_index('Date')
    if bars is None or bars.empty or "Close" not in bars:
        return chart
    chart = chart.reindex(bars.index.union(chart.index))
    close = chart['Close'].fillna(bars['Close'].reindex(chart.index)).astype(float)
    chart['Close'] = close
    for n in (10, 30, 50, 100, 200):
        chart[f'ma_{n}'] = chart[f'ma_{n}'].fillna(close.rolling(n).mean())
    d = close.diff()
    gain, loss = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(), (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    chart['RSI'] = chart['RSI'].fillna(100 - 100 / (1 + gain / loss))
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    chart['macd'] = chart['macd'].fillna(macd)
    chart['MACD Signal'] = chart['MACD Signal'].fillna(macd.ewm(span=9, adjust=False).mean())
    chart['Symbol'] = chart['Symbol'].fillna(tdata['Symbol'].iloc[0]) if len(tdata) else chart['Symbol']
    return chart

def build_price_chart(ticker, tdata, bars=None):
    """Candlesticks + moving averages + buy/sell markers, volume, RSI, MACD and performance panels on one shared,
    zoomable x-axis (range buttons 1M..All, unified hover). bars = daily OHLCV (data.ohlcv); missing days are drawn
    from the close alone."""
    chart = _with_bar_history(tdata, bars)
    chart = chart[chart.index >= chart.index.max() - WINDOW]
    bars = (bars if bars is not None else pd.DataFrame()).reindex(chart.index)
    for col in ("Open", "High", "Low", "Close"):
        bars[col] = bars[col].fillna(chart['Close']) if col in bars else chart['Close']
    x_start, x_end = chart.index[0], chart.index[-1]
    events, _ = strategy_events(tdata, load_decisions(), ticker)
    rs_lines = relative_strength_lines(chart[chart.index >= tdata["Date"].min()], ticker)   # where SPY / sector bars exist too

    panels = ["price", "vol", "rsi", "macd"] + (["rs"] if rs_lines else [])
    height_of = {"price": 0.46, "vol": 0.09, "rsi": 0.12, "macd": 0.12, "rs": 0.16}
    titles = {"price": "", "vol": "Volume", "rsi": "RSI (14)", "macd": "MACD vs signal",
              "rs": "Performance (% change) · " + " · ".join(
                  f'<span style="color:{RS_COLORS[k]};">━ {name}</span>' for k, (name, _s) in rs_lines.items())}
    heights = [height_of[x] for x in panels]
    row_of = {x: i + 1 for i, x in enumerate(panels)}
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.035,
                        row_heights=[h / sum(heights) for h in heights], subplot_titles=[titles[x] for x in panels])
    fig.update_annotations(font=dict(size=11, color='#64748b', family=CHART_FONT), xanchor="left", x=0, yshift=-2)

    # Price line (hover shows the day's status), earnings dates, moving averages
    status_txt = []
    for w, r, s in zip(chart['Strategy_Weight'], chart['Strategy_Rank'], chart['Strategy_Score']):
        sig = daily_status(w, s)
        bits = [sig] + ([f"weight {w * 100:.1f}%"] if sig == "Hold" else []) \
            + ([f"rank #{r:.0f}"] if pd.notna(r) else []) + ([f"score {s:.1f}"] if pd.notna(s) else [])
        status_txt.append(" · ".join(bits))
    fig.add_trace(go.Candlestick(x=chart.index, open=bars["Open"], high=bars["High"], low=bars["Low"], close=bars["Close"],
                                 name="Close", text=status_txt, increasing=dict(line=dict(color=GOOD, width=1), fillcolor=GOOD),
                                 decreasing=dict(line=dict(color=BAD, width=1), fillcolor=BAD), showlegend=False), row=1, col=1)
    if "Volume" in bars and bars["Volume"].notna().any():
        up = bars["Close"] >= bars["Open"]
        fig.add_trace(go.Bar(x=chart.index, y=bars["Volume"], name="Volume", showlegend=False,
                             marker=dict(color=[("rgba(21,128,61,.45)" if u else "rgba(185,28,28,.45)") for u in up], line_width=0),
                             hovertemplate="Vol %{y:.3s}<extra></extra>"), row=row_of["vol"], col=1)
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

    # Real account (Reports/live_trade_ledger.csv fills): green dashed entry line + green price, red dashed exit line + red price; dashed entry line from the entry date while held, dashed exit
    # line from the sell date after a full exit
    for kind, (day, price) in account_trade_lines(ticker).items():
        color = GOOD if kind == "Entry" else BAD
        fig.add_trace(go.Scatter(x=[max(pd.Timestamp(day), x_start), x_end], y=[price, price], mode="lines",
                                 line=dict(color=color, width=1.2, dash="dash"), name=f"${price:,.2f}",
                                 hovertemplate=f"${price:,.2f} ({day:%b %-d})<extra></extra>", showlegend=False),
                      row=1, col=1)
        fig.add_annotation(x=max(pd.Timestamp(day), x_start), y=price, xref="x", yref="y", showarrow=False,
                           text=f"${price:,.2f}", xanchor="left", yanchor="bottom",
                           font=dict(size=10, color=color))

    # Next earnings date (extends the x-axis when it is within ~2 months)
    x_right = x_end
    ned = last_next_earnings([ticker])['Next ED'].iloc[0]
    if ned and pd.Timestamp(ned) - x_end <= pd.Timedelta(days=62):
        ned = pd.Timestamp(ned)
        x_right = max(x_end, ned) + pd.Timedelta(days=4)
        fig.add_vline(x=ned, line=EARN_LINE | dict(width=1.5), row="all", col=1)
        fig.add_annotation(x=ned, y=0.99, xref="x", yref="y domain", text=f"next earnings {ned:%b %-d}", showarrow=False,
                           yanchor="top", xanchor="right", font=dict(size=10, color='#c2410c'))

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

    grid = dict(showgrid=True, gridcolor='#f1f5f9', showline=False, tickfont=dict(size=10, color='#64748b'), zeroline=False)
    view = chart.index[chart.index >= x_end - pd.Timedelta(days=365)]          # opens on 1 year
    lo, hi = bars.loc[view, "Low"].min(), bars.loc[view, "High"].max()
    fig.update_layout(
        height=int(620 + 120 * (len(panels) - 2)), hovermode='closest', margin=dict(l=10, r=70, t=44, b=24),
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', dragmode="zoom", bargap=0.15,
        legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="center", x=0.5, font=dict(size=10, color='#334155'),
                    bgcolor="rgba(0,0,0,0)"),
        font=dict(family=CHART_FONT, size=11, color='#334155'),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e2e8f0", font_size=11, font_family=CHART_FONT))
    fig.update_xaxes(type="date", range=[view[0], x_right], showspikes=False, rangeslider_visible=False,
                     rangebreaks=[dict(bounds=["sat", "mon"])], tickformat='%b %d', **(grid | dict(showgrid=False)))
    fig.update_xaxes(rangeselector=dict(buttons=RANGES, x=0, y=1.0, yanchor="bottom", bgcolor="#f1f5f9", activecolor="#5eead4",
                                        bordercolor="#cbd5e1", borderwidth=1, font=dict(size=12, color="#0f172a")), row=1, col=1)
    fig.update_yaxes(side="right", showticklabels=True, ticks="outside", ticklen=3, tickcolor="#e2e8f0", automargin=True,
                     showspikes=False,
                     title_font=dict(size=10, color="#64748b"), **grid)
    fig.update_yaxes(title_text="Price", tickformat='$,.2f', hoverformat='$,.2f', nticks=8,
                     range=[lo - (hi - lo) * 0.06, hi + (hi - lo) * 0.06], row=1, col=1)
    axis = {"vol": dict(title_text="Volume", tickformat="~s", nticks=3, showgrid=False),
            "rsi": dict(title_text="RSI", tickvals=[30, 50, 70]), "macd": dict(title_text="MACD", tickformat=".2f", nticks=4),
            "rs": dict(title_text="Perf. %", ticksuffix="%", tickformat="+.0f", nticks=5)}
    for name, kw in axis.items():
        if name in row_of:
            fig.update_yaxes(row=row_of[name], col=1, **kw)
    # No floating tooltip for the series: the chart component shows their values in a top-left legend per panel and
    # draws a dotted crosshair with the price at the cursor on the right axis (chart_component/index.html).
    legend_traces = {"Close", "Volume", "RSI", "MACD", "MACD signal"} | {n for n, _s in rs_lines.values()}
    fig.for_each_trace(lambda t: t.update(hoverinfo="none", hovertemplate=None)
                       if t.name in legend_traces or (t.name or "").startswith("MA ") else None)
    return fig

def account_trade_lines(ticker, ledger_csv=None):
    """{'Entry': (date, avg price of the shares still held)} while the account holds `ticker`, else
    {'Exit': (date, price)} of the sell that closed the position; {} when never traded. FIFO over the filled quantities
    of Reports/live_trade_ledger.csv (partial fills of expired / canceled orders count too)."""
    import os
    from dashboard.settings import REPORTS
    path = ledger_csv or os.path.join(REPORTS, "live_trade_ledger.csv")
    try:
        t = pd.read_csv(path, usecols=["Filled_CT", "Symbol", "Side", "Filled_Qty", "Fill_Price"])
    except (OSError, ValueError):
        return {}
    t = t[(t["Symbol"] == ticker) & (pd.to_numeric(t["Filled_Qty"], errors="coerce") > 0)].dropna(subset=["Filled_CT"])
    lots, last_sell = [], None
    for r in t.assign(When=pd.to_datetime(t["Filled_CT"], format="mixed")).sort_values("When").itertuples():
        q = float(r.Filled_Qty)
        if r.Side == "BUY":
            lots.append([r.When, q, float(r.Fill_Price)])
            continue
        last_sell = (r.When, float(r.Fill_Price))
        while q > 1e-9 and lots:
            used = min(q, lots[0][1])
            lots[0][1] -= used
            q -= used
            if lots[0][1] <= 1e-9:
                lots.pop(0)
    if lots:
        sh = sum(x[1] for x in lots)
        return {"Entry": (lots[0][0].normalize(), round(sum(x[1] * x[2] for x in lots) / sh, 2))}
    return {"Exit": (last_sell[0].normalize(), round(last_sell[1], 2))} if last_sell else {}

def render_stock_chart(ticker, tdata):
    """The stock chart (with your saved drawings, dashboard/drawings.py)."""
    from dashboard.data import ohlcv
    from dashboard.drawings import render_chart
    bars = ohlcv(ticker)
    fig = build_price_chart(ticker, tdata, bars)
    b = bars[bars.index >= tdata['Date'].max() - WINDOW]
    render_chart(fig, ticker, {"x": [f"{d:%Y-%m-%d}" for d in b.index], "low": b["Low"].tolist(), "high": b["High"].tolist()})
