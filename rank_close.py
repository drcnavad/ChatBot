"""Display-only 'Rank now' at the latest COMPLETED close (Chirag Nov 4, 2026, t295u).

Every weekday is a decision session, so the signal table keeps each day's 2:30 PM CT bar (keep_decision_bars) and
sleeves.live_ranks ranks on that, not on the final close. This module re-ranks the live score (RS short-term
10/21/63/63 = 1:1:2) on fresh Alpaca adjusted daily bars (market-data only, no paid quota, no orders), using only
sessions whose bar is final (after 4:05 PM ET). Same filters as sleeves.ranking: eligible, score > 0, not do_not_buy,
tradable. Never used by the trade path."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

WINDOWS = {"rs_short": (10, 21, 63), "rs_short_63": (10, 21, 63, 63)}


def close_ranks(now=None, inp=None, bars=None, all_stocks=False):
    """(as_of Timestamp, {symbol: (rank, score)}) on the latest final daily close."""
    import backtest_engine as be
    import sector_mapping as sm
    import sleeves
    inp = inp or sleeves.inputs()
    score = sleeves._cfg(sleeves.HALVES[0]).get("score", "rs_short_63")
    windows = WINDOWS[score]
    elig = inp["eligible"].iloc[-1]
    cols = [s for s in elig.index if elig[s]]
    etfs = sorted({sm.sector_etf_for(s) for s in cols} | {"SPY"})
    now = (now or datetime.now(be.EASTERN)).astimezone(be.EASTERN)
    if bars is None:
        bars = be.fetch_daily_bars(cols + etfs, start=now - timedelta(days=130))
    c = bars.pivot_table(index="Date", columns="Symbol", values="Close").sort_index()
    today = pd.Timestamp(now.date())
    if now.hour * 60 + now.minute < 16 * 60 + 5:          # today's bar not final yet
        c = c[c.index < today]
    rs = be.relative_strength(c, cols, windows=list(windows)).iloc[-1]
    rs = rs[~rs.index.isin(sm.do_not_buy) & rs.index.isin(set(sm.tradable_symbols))]
    pos = rs[rs > 0].sort_values(ascending=False, kind="stable")
    rs = pd.concat([pos, rs[~(rs > 0)].sort_values(ascending=False, kind="stable", na_position="last")]) if all_stocks else pos
    # all_stocks (dashboard only, Nov 5, 2026): score <= 0 continue the numbering after the positive ranks
    return c.index[-1], {s: (i + 1, float(v)) for i, (s, v) in enumerate(rs.items())}


if __name__ == "__main__":
    d, r = close_ranks()
    print(f"as of close {d:%a %b %-d}:", [(s, k, round(v, 1)) for s, (k, v) in list(r.items())[:20]])
    print("APA", r.get("APA"))
