"""Home tab stock list: every stock once (Chirag, Oct 16, 2026). Oct 21: with the Next rebalance preview readable the
first row splits into Keep / Buy next / Sell next (last_decision.next_plan); rows 2-3 are now "Next in line" / "Others".

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
        import holdings
        pend = sorted(set(holdings.pending_symbols()) - set(pos))
        return sorted(s for s in pos if s), "live Alpaca positions" + (f" · Pending order: {', '.join(pend)}" if pend else "")
    return [], "Alpaca not readable right now (live positions are the only holdings source)"

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
    """[(title, symbols, note)] - every symbol once: Keep / Buy next / Sell next from the Next rebalance preview (Oct 21,
    2026), then Next in line and Others; Holdings / Next in line / Others when the preview cannot be read."""
    held, src = held_symbols()
    pl = {}
    try:
        from dashboard.details.last_decision import next_plan
        pl = (next_plan() or {}).get("plan", {})
    except Exception:
        pass
    pick = lambda acts: sorted((s for s, a in pl.items() if a[0] in acts), key=lambda s: -pl[s][1])
    try:
        lv_t, sec_t, lv_rk, sr_rk = _rankings()
    except Exception:
        lv_t = sec_t = lv_rk = sr_rk = []
    seen = set(held) | {s for s, a in pl.items() if a[0] != "Blocked"}
    cand = []
    for s in _alternate(lv_t, sec_t) + _alternate(lv_rk, sr_rk):
        if s not in seen and len(cand) < N_CANDIDATES:
            cand.append(s)
            seen.add(s)
    rank = p.by_symbol["Strategy_Rank"] if "Strategy_Rank" in p.by_symbol.columns else pd.Series(dtype=float)
    universe = list(tradable_symbols) + [s for s in p.options if s not in tradable_symbols and s != "QQQ"]
    rest = sorted({s for s in universe if s not in seen},
                  key=lambda s: (pd.isna(rank.get(s)), rank.get(s, 0) if pd.notna(rank.get(s)) else 0, s))
    nxt = ("Next in line", cand, "the next picks of the two halves")
    if not pl:
        return [("Holdings", held, src), nxt, ("Others", rest, "")]
    return [("Keep", pick(("Keep", "Take over")), "held, still a target"),
            ("Buy next", pick(("Buy",)), "targets not held yet"),
            ("Sell next", pick(("Sell",)), "held, no longer a target"), nxt, ("Others", rest, "")]

def dnb_symbols():
    """sector_mapping.do_not_buy, upper case: tagged DNB in the stock list."""
    return {str(s).upper() for s in do_not_buy}
