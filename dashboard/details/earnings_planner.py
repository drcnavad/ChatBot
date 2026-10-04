"""Details tab: the earnings planner (earnings_planner.py; held stocks reporting in the next 14 days)."""
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard.details.live_holdings import holdings_this_run
from dashboard.settings import CT, EARNINGS_CSV, REPORTS
from dashboard.style import BAD, CAUTION, GOOD, INK, MONEY, caption_text, info_text, stat_cards, toned


@st.cache_data(ttl=3600, max_entries=2, show_spinner=False)
def _planner_bars(mtimes):
    """Cached daily bars (Reports/cache, no download), reloaded when a cache file changes."""
    import earnings_planner as ep
    return ep.cached_bars()


def _stop_distance_tone(v):
    return INK if v is None or v != v else GOOD if v >= 10 else CAUTION if v >= 5 else BAD


def render_earnings_planner():
    """Details tab: held live stocks with earnings in the next 14 days (else the next 3): date and time, days, stop
    window, current 3x ATR stop and distance, median past reaction from the cached bars. Read-only."""
    data, err = holdings_this_run()
    if err:
        caption_text("The earnings planner needs the live holdings, which are not available right now (see Live holdings above).")
        return
    import alpaca_paper as ap
    import backtest_engine as be
    import earnings_planner as ep
    import earnings_stop as es
    f = lambda v: float(v) if v not in (None, "") else float("nan")
    held = {str(p["symbol"]).upper(): f(p.get("current_price")) for p in data["positions"] if f(p.get("qty")) > 0}
    if not held:
        info_text("No open positions in the Alpaca account, so no earnings to plan for.")
        return
    cache = os.path.join(REPORTS, "cache")
    bars = _planner_bars(tuple(os.path.getmtime(os.path.join(cache, f)) if os.path.exists(os.path.join(cache, f)) else 0
                               for f in ep.CACHE_FILES))
    try:
        earnings = be.load_earnings(EARNINGS_CSV)
    except (OSError, ValueError) as e:
        info_text(f"Earnings calendar unavailable ({type(e).__name__}).")
        return
    today = pd.Timestamp(datetime.now(CT).date())
    table, in_range = ep.plan(held, ap.first_buy_dates(data["fills"]), earnings, bars, today)
    horizon_end = today + pd.Timedelta(days=ep.HORIZON_DAYS)
    if table.empty:
        info_text("No held stock has an upcoming earnings date on file (Reports/earnings_date.csv).")
        return
    soon = table if in_range else table.iloc[0:0]
    active = table[table["Stop window"].str.startswith("Active")]
    nearest = table.loc[table["To stop %"].idxmin()] if table["To stop %"].notna().any() else None
    first = table.iloc[0]
    stat_cards([("Reports in 14 days", f"{len(soon)} of {len(held)} held", CAUTION if len(soon) else INK),
                ("Stop windows open", str(len(active)), CAUTION if len(active) else INK),
                ("Next report", f"{first['Stock']} · {first['Earnings']:%a %b %-d}", INK),
                ("Days to next", str(first["Days"]), INK),
                ("Closest to stop", f"{nearest['Stock']} {nearest['To stop %']:.1f}%" if nearest is not None else "—",
                 _stop_distance_tone(nearest["To stop %"]) if nearest is not None else INK)])
    if not in_range:
        info_text(f"No held stock reports earnings in the next {ep.HORIZON_DAYS} days (through {horizon_end:%a %b %-d}). "
                f"The next {len(table)} upcoming reports:")
    shown = table.assign(Earnings=[f"{d:%a %b %-d, %Y}" for d in table["Earnings"]]).round(
        {"Stop": 2, "To stop %": 1, "Median gap %": 1, "Median 5-day %": 1})
    sty = toned(toned(shown, ["To stop %"], fn=_stop_distance_tone), ["Median gap %", "Median 5-day %"])
    sty = toned(sty, ["Stop window"], fn=lambda v: CAUTION if str(v).startswith("Active") else INK)
    pct1 = st.column_config.NumberColumn(format="%.1f%%")
    st.dataframe(sty, hide_index=True, width="stretch", height=35 * (len(shown) + 1) + 3,
                 column_config={"Last price": MONEY, "Stop": MONEY, "To stop %": pct1, "Median gap %": st.column_config.NumberColumn(format="%+.1f%%"),
                                "Median 5-day %": st.column_config.NumberColumn(format="%+.1f%%"),
                                "Days": st.column_config.NumberColumn(format="%d"), "Reports": st.column_config.NumberColumn(format="%d")})
    last_bar = bars["Date"].max() if len(bars) else None
    bars_lbl = f"the cached daily bars through {last_bar:%a %b %-d}" if last_bar is not None else "no cached bars"
    caption_text(f"As of {data['as_of']:%a %b %-d %I:%M %p} CT · last price from Alpaca · stops and past reactions from {bars_lbl}. "
               f"Stop = highest close since the first buy still held - {es.K_ATR:g} × ATR(14), the live stop job's formula; "
               f"To stop % = how far the last price can fall before the stop (green 10%+, yellow 5-10%, red under 5%). "
               f"Stop window = {es.ARM_DAYS} calendar days before the report through its reaction day (after close = the next "
               "trading day). Median gap % = the reaction day's open vs the close before; Median 5-day % = the 5th trading day's "
               "close vs that close; Reports = past reports with bars. Dates and times from Reports/earnings_date.csv.")
