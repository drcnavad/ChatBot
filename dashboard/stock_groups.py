"""Each half's ranking at the latest close (manual_trade.py uses _rankings; the old stock-list groups were removed).
Display only: nothing here feeds the strategy or orders."""

import pandas as pd
import db  # noqa: E402  tables live in SQLite (db.py), not CSV files

from dashboard.settings import PICKS_CSV

def _rankings():
    """(lv targets, sector targets, lv ranking, sector ranking) at the latest close; the rankings = each half's buy zone."""
    import sleeves
    inp = sleeves.sector_inputs()
    picks = db.read_csv(PICKS_CSV)
    day = pd.Timestamp(str(picks["As_Of"].iloc[0])[:10])
    if day not in inp["close"].index:
        day = inp["close"].index[-1]
    el = inp["eligible"].loc[day]
    lv = inp["scores"]["low_vol"].loc[day].where(el)
    sr = inp["scores"]["sector_rot"].loc[day].where(el)
    lv_zone = lv[lv > 0].sort_values(ascending=False, kind="stable")
    sr_zone = sr[sr > 0].sort_values(ascending=False, kind="stable")
    p = picks[picks["Provisional_Weight"].fillna(0) > 0].sort_values("Strategy_Rank")
    lv_t = [str(s).upper() for s in p["Symbol"]]
    try:
        w = sleeves.sector_weights(day, exclude=set(lv_t), inp=inp)
        sec_t = list(w[w > 0].sort_values(ascending=False, kind="stable").index)
    except Exception:
        sec_t = []
    import sector_mapping as sm                        # a stock commented out of the list is never shown (Oct 19)
    keep = set(sm.tradable_symbols)
    f = lambda xs: [x for x in xs if str(x).upper() in keep]
    return f(lv_t), f(sec_t), f(lv_zone.index), f(sr_zone.index)
