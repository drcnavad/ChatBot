"""Home tab stock list: every stock once, in three rows (Chirag, Oct 16, 2026).

  1. Holdings: the live Alpaca positions (read-only GET, the Trading Account tab's read); when Alpaca is not readable
     (tests, an outage) the planned shares in Reports/sleeve_ledger.json (both halves) stand in.
  2. Next in line (waiting list): up to 20 stocks the live 50/50 rules would pick next, not already held: first the next
     rebalance targets (low-volatility half = Provisional_Weight in strategy_picks.csv, sector half = sleeves.sector_weights
     without the low-volatility names), then the next best of each half's ranking, alternating (low volatility: 1 / 63-day
     volatility above the 200-day average; sector rotation: C6 score in the 3 sectors with the best 63-day ETF return).
     Ranked like any other stock (do_not_buy included).
  3. Remaining: every other stock of the stock list (sector_mapping.tradable_symbols) by Strategy_Rank, then any other
     symbol on the dashboard.
  sector_mapping.do_not_buy stocks stay in whichever row they land and get a small DNB tag (dnb_symbols); no separate
  buy-zone or do-not-buy row (Oct 16: they showed stocks twice).
Display only: nothing here feeds the strategy or orders."""

import pandas as pd

from dashboard.settings import PICKS_CSV

try:
    from sector_mapping import do_not_buy, tradable_symbols
except Exception:   # deployed without the pipeline modules
    tradable_symbols, do_not_buy = [], []

N_CANDIDATES = 20

def held_symbols():
    """(symbols, source): the live Alpaca positions (the Trading tab's cached read, at most 5 minutes old; Oct 19: the
    Home list was stale), else the sleeve ledger's planned shares."""
    try:
        from dashboard.details.live_holdings import holdings_this_run
        data, _err = holdings_this_run()
    except Exception:
        data = None
    if data is not None:
        pos = [str(p.get("symbol", "")).upper() for p in data.get("positions") or [] if float(p.get("qty") or 0) > 0]
        return sorted(s for s in pos if s), "live Alpaca positions"
    try:
        import sleeves
        led = sleeves.load_ledger()
    except Exception:
        led = None
    if led is None:
        return [], "no holdings source"
    syms = {s.upper() for h in sleeves.HALVES for s, q in (led["halves"][h].get("shares") or {}).items() if float(q) > 0}
    return sorted(syms), f"sleeve ledger of {led.get('day')} (Alpaca not readable here)"

def _rankings():
    """(lv targets, sector targets, lv ranking, sector ranking) at the latest close; the rankings = each half's buy zone."""
    import sleeves
    inp = sleeves.sector_inputs()
    picks = pd.read_csv(PICKS_CSV)
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

def _alternate(a, b):
    out = []
    for i in range(max(len(a), len(b))):
        out += a[i:i + 1] + b[i:i + 1]
    return out

def stock_groups(p):
    """[(title, symbols, note)] - the three rows, every symbol once."""
    held, src = held_symbols()
    try:
        lv_t, sec_t, lv_rk, sr_rk = _rankings()
    except Exception:
        lv_t = sec_t = lv_rk = sr_rk = []
    seen = set(held)
    cand = []
    for s in _alternate(lv_t, sec_t) + _alternate(lv_rk, sr_rk):
        if s not in seen and len(cand) < N_CANDIDATES:
            cand.append(s)
            seen.add(s)
    rank = p.by_symbol["Strategy_Rank"] if "Strategy_Rank" in p.by_symbol.columns else pd.Series(dtype=float)
    universe = list(tradable_symbols) + [s for s in p.options if s not in tradable_symbols and s != "QQQ"]
    rest = sorted({s for s in universe if s not in seen},
                  key=lambda s: (pd.isna(rank.get(s)), rank.get(s, 0) if pd.notna(rank.get(s)) else 0, s))
    return [("Holdings", held, src),
            ("Next in line (waiting list)", cand, "the next picks of the two halves, not held"),
            ("Remaining", rest, "")]

def dnb_symbols():
    """sector_mapping.do_not_buy, upper case: tagged DNB in the stock list."""
    return {str(s).upper() for s in do_not_buy}
