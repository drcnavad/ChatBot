"""The live account as one sleeve - backtest_engine.WINNER["sleeves"] = {"rs_short": 1.0}.

Live from Fri Nov 6, 2026 (Chirag; locked ~12 weeks, to ~Jan 29, 2027): 100% forward_test.py's "Short-term relative
strength, skip stretched buys" - relative strength over 10/21/63 days, top 10 equal weight, Friday rebalance only (the
week's last session; no Mon-Thu trades since Nov 4). A stock more than 8% above its 20-day average is not newly bought
(a held one in the top 10 is kept; held = this ledger). Blocked picks (earnings within 5 days, do_not_buy, sold_block)
are replaced by the next eligible, not stretched stock ranked 20 or better. Soft QQQ regime; do_not_sell never sold.

Sizing: target = (equity × 0.99) / 10 per stock (equal weight); leftover cash stays cash (Chirag Oct 31).

Ledger (Reports/sleeve_ledger.json, version 4): share count per symbol and cash after the planned trades. Next run
(reconcile): live positions are matched to the ledger; a position the ledger does not claim is sold (do_not_sell kept);
whatever the ledger does not explain (fills vs plan, dividends, a deposit) goes to the sleeve's cash. No v4 ledger (the
switch run): the sleeve = the whole equity, every holding unclaimed, and a held stock it targets is kept (no sell + rebuy).
The code stays general over HALVES (one entry now).
"""
import json
import math
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import db  # noqa: E402  tables live in SQLite (db.py), not CSV files
from tax_lots import _num as _float_or_nan   # one float parser for every account view (sleeves._num adapts it)

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "Reports")
LEDGER_JSON = os.environ.get("STOCK_ANALYSIS_SLEEVE_LEDGER") or os.path.join(REPORTS, "sleeve_ledger.json")
PLAN_CSV = os.environ.get("STOCK_ANALYSIS_SLEEVE_PLAN") or os.path.join(REPORTS, "sleeve_plan.csv")
HALVES = ("rs_short",)
STRATEGY = {"rs_short": "Short-term relative strength, skip stretched buys"}   # Fri Nov 6, 2026 (Chirag)
LABEL = STRATEGY
SHORT = STRATEGY  # alias
VERSION = 4
CT = ZoneInfo("America/Chicago")

def config():
    """{'rs_short': 1.0} (WINNER['sleeves']), else None."""
    import backtest_engine as be
    s = be.WINNER.get("sleeves")
    return dict(s) if s else None

def _cfg(h):
    import forward_test as ft
    return next(c for c in ft.STRATEGIES if c["name"] == STRATEGY[h])

def full_day(h, day):
    """True on half h's full rebalance session: the week's last session (Thursday when Friday is a holiday)."""
    import backtest_engine as be
    d = pd.Timestamp(day).normalize()
    return be.is_session(d) and be.next_sessions(d, 1)[0].isocalendar()[:2] != d.isocalendar()[:2]

def half_day(h, day):
    """True when half h may trade on session `day`: rebalance session only (Nov 4, 2026, Chirag: no Mon-Thu trades)."""
    import backtest_engine as be
    d = pd.Timestamp(day).normalize()
    if not be.is_session(d):
        return False
    return full_day(h, d)

def calendar(start, sessions=30):
    """[(date, halves that rebalance in full)] for the next sessions from `start` (inclusive); rs_short also checks Mon-Thu."""
    import backtest_engine as be
    d0 = pd.Timestamp(start).normalize()
    days = [d0] + list(be.next_sessions(d0, sessions)) if be.is_session(d0) else list(be.next_sessions(d0, sessions))
    return [(d, list(HALVES)) for d in days if full_day(HALVES[0], d)]

# ----------------------------------------------------------------------------- targets (forward_test's own code)
_INPUTS, _TARGETS = {}, {}

def inputs(signal_csv=None, bench_csv=None):
    """forward_test.strategy_inputs on the saved data (cached per table modification time)."""
    import forward_test as ft
    signal_csv, bench_csv = signal_csv or ft.SIGNAL_CSV, bench_csv or ft.BENCH_CSV
    key = (signal_csv, db.mtime(signal_csv), bench_csv, db.mtime(bench_csv))
    if key not in _INPUTS:
        sig = db.read_csv(signal_csv, usecols=ft.SIG_COLS, parse_dates=["Date"])
        _INPUTS.clear()
        _TARGETS.clear()
        _INPUTS[key] = ft.strategy_inputs(sig, None, None, ft._read(bench_csv), bars=[])   # no bar download
    return _INPUTS[key]

def targets(h, inp=None):
    """Date x Symbol live weights (fractions of the half, 99% invested) of half h = forward_test.strategy_targets."""
    import forward_test as ft
    inp = inp or inputs()
    key = (h, id(inp))
    if key not in _TARGETS:
        if "held" not in inp:                             # live holdings: a held stretched pick is kept (cool_buys)
            led = load_ledger()
            inp["held"] = set((led or {}).get("halves", {}).get(h, {}).get("shares", {}))
        _TARGETS[key] = ft.strategy_targets(_cfg(h), inp)
    return _TARGETS[key]

def half_weights(h, day, inp=None):
    """Half h's target *membership* on session `day` (Series; >0 = in the book). Live sizing ignores these values and uses equal invested/n."""
    import sector_mapping as sm
    inp = inp or inputs()
    w = targets(h, inp)
    d = pd.Timestamp(day).normalize()
    if d not in w.index:
        raise ValueError(f"no data for {d:%Y-%m-%d} in signal_analysis.csv - the {LABEL[h]} half cannot be ranked")
    s = w.loc[d]
    return s.where(s.index.isin(set(sm.tradable_symbols)), 0.0)

def ranking(h, day, inp=None):
    """Half h's score on `day`, best first (eligible, score above 0, do_not_buy out) - display / next in line."""
    import sector_mapping as sm
    inp = inp or inputs()
    sc = inp["scores"][_cfg(h).get("score", "live")]
    d = pd.Timestamp(day).normalize()
    if d not in sc.index:
        d = sc.index[sc.index <= d][-1]
    s = sc.loc[d].where(inp["eligible"].loc[d])
    s = s[(s > 0) & ~s.index.isin(sm.do_not_buy) & s.index.isin(set(sm.tradable_symbols))]
    return s.sort_values(ascending=False, kind="stable")

def live_ranks(day=None, inp=None):
    """{symbol: (rank, score)} of the live line (RS short-term) at `day` (default: the latest close) - display only. The
    pipeline's Strategy_Rank / Strategy_Score columns are a forward-test input with the old selection fields, so the
    dashboard and the chatbot show these instead (Oct 27, 2026)."""
    inp = inp or inputs()
    day = inp["close"].index[-1] if day is None else day
    r = ranking(HALVES[0], day, inp)
    return {s: (i + 1, float(v)) for i, (s, v) in enumerate(r.items())}

# ----------------------------------------------------------------------------- ledger
def load_ledger(path=None):
    """The v3 ledger, or None (none yet, or the pre-Oct 27 lowvol / sector ledger: the switch run starts fresh)."""
    path = path or LEDGER_JSON
    if not os.path.exists(path):
        return None
    with open(path) as f:
        led = json.load(f)
    if not isinstance(led, dict):
        raise ValueError(f"{path} is not a valid sleeve ledger - fix or remove it (no orders until then)")
    if int(led.get("version", 0)) < VERSION:
        return None
    if set(led.get("halves", {})) != set(HALVES):
        raise ValueError(f"{path} is not a valid sleeve ledger - fix or remove it (no orders until then)")
    return led

def save_ledger(led, path=None):
    path = path or LEDGER_JSON
    led = {**led, "saved_at": datetime.now(CT).strftime("%Y-%m-%d %H:%M:%S CT")}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(led, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return path

def write_strategy_picks(day, path=None, inp=None, led=None):
    """Rewrite Reports/strategy_picks.csv from the live RS short-term sleeve (top-n equal weight).

    The notebook still writes a WINNER-n=5 / low_vol picks file for display columns; live trading
    and trade_audit must follow the sleeve. Called after a successful ledger save.
    """
    import sector_mapping as sm
    path = path or os.path.join(REPORTS, "strategy_picks.csv")
    inp = inp or inputs()
    d = pd.Timestamp(day).normalize()
    w = half_weights(HALVES[0], d, inp)
    w = w[w > 0].sort_values(ascending=False)
    r = ranking(HALVES[0], d, inp)
    close = inp["close"]
    soon = inp.get("soon")
    tags = set((led or {}).get("halves", {}).get(HALVES[0], {}).get("tags") or [])
    rows = []
    for s, wt in w.items():
        rank = int(r.index.get_loc(s)) + 1 if s in r.index else None
        score = float(r.get(s)) if s in r.index else None
        px = float(close.at[d, s]) if s in close.columns and d in close.index and _num(close.at[d, s]) else None
        earn = None
        if soon is not None and d in soon.index and s in soon.columns and bool(soon.at[d, s]):
            earn = "within_5d"
        rows.append({
            "As_Of": d.date().isoformat(),
            "Last_Rebalance": (led or {}).get("day") or d.date().isoformat(),
            "Last_Decision": d.date().isoformat(),
            "Strategy": STRATEGY[HALVES[0]],
            "Regime_On": 1,
            "Symbol": s,
            "Sector": (sm.symbol_sector.get(s, "") if hasattr(sm, "symbol_sector") else ""),
            "Held": 1 if s in tags else 0,
            "Strategy_Weight": round(float(wt), 4),
            "Provisional_Weight": round(float(wt), 4),
            "Strategy_Score": None if score is None else round(score, 4),
            "Strategy_Rank": rank,
            "Technical_Score": None,
            "RS_Score": None if score is None else round(score, 4),
            "Close": None if px is None else round(px, 4),
            "Next_Earnings": earn,
        })
    out = pd.DataFrame(rows)
    out.pipe(db.to_csv, path, index=False)
    return path

def holders(led):
    """{SYMBOL: [halves claiming it]}."""
    out = {}
    for h in HALVES:
        for s, q in ((led or {}).get("halves", {}).get(h, {}).get("shares") or {}).items():
            if float(q or 0) > 0:
                out.setdefault(str(s).upper(), []).append(h)
    return out

def owners(led):
    """{SYMBOL: half} (the live RS short-term line)."""
    return {s: (hs[0] if len(hs) == 1 else "both") for s, hs in holders(led).items()}

def _num(x):
    """float(x) when it is finite and above zero, else None."""
    v = _float_or_nan(x)
    return v if v == v and v > 0 else None

def reconcile(led, positions, prices, equity, shares=None, fills=None, pending=()):
    """(state, unowned, info): state = {half: {'shares': {sym: qty}, 'cash': $}}; unowned = {sym: qty} no half claims.
    fills / pending: accepted for the old call signature (the residual split covers fills that differ from the plan)."""
    shares = shares or config() or {h: 1 / len(HALVES) for h in HALVES}
    pos = {str(k).upper(): float(v) for k, v in (positions or {}).items() if float(v or 0) > 0}
    if led is None:
        return ({h: {"shares": {}, "cash": equity * shares[h]} for h in HALVES}, dict(pos),
                {"init": True, "residual": 0.0, "dropped": [], "tags": {}})
    L = {h: {str(s).upper(): float(q or 0) for s, q in (led["halves"][h].get("shares") or {}).items() if float(q or 0) > 0}
         for h in HALVES}
    state = {h: {"shares": {}, "cash": float(led["halves"][h].get("cash", 0.0))} for h in HALVES}
    unowned, dropped = {}, []
    for s in set(pos) | {x for h in HALVES for x in L[h]}:
        tot = sum(L[h].get(s, 0.0) for h in HALVES)
        if s not in pos:
            dropped.append(s)
            continue
        if tot <= 0:
            unowned[s] = pos[s]
            continue
        for h in HALVES:
            if L[h].get(s):
                state[h]["shares"][s] = pos[s] * L[h][s] / tot
    val = {h: half_value(state[h], prices) for h in HALVES}
    resid = equity - sum(val.values()) - sum(q * (_num(prices.get(s)) or 0.0) for s, q in unowned.items())
    tot = sum(max(v, 0.0) for v in val.values())
    for h in HALVES:
        state[h]["cash"] += resid * (max(val[h], 0.0) / tot if tot > 0 else shares[h])
    return state, unowned, {"init": False, "residual": resid, "dropped": sorted(dropped), "tags": owners(led)}

def half_value(st, prices):
    return st["cash"] + sum(q * (_num(prices.get(s)) or 0.0) for s, q in st["shares"].items())

# ----------------------------------------------------------------------------- the plan

def exit_rank(h):
    """Half h's Mon-Thu exit / refill rank cap: forward_test config 'exit'[0] (20)."""
    try:
        ex = _cfg(h).get("exit")
        if isinstance(ex, (list, tuple)) and ex:
            return int(ex[0])
    except Exception:
        pass
    return 20

def replace_blocked_new_slots(h, d, names, held, blocked, px, inp, n_picks, invested):
    """`n_picks` slots at equal weight invested/n. Drop NEW blocked picks; fill from the live ranking.
    Held blocked names stay (build_orders freezes top-ups). Held replacements for blocked ideal
    picks (e.g. MA/V standing in for sold-blocked CRL/FDS) stay when still ranked <= exit_below
    (live Mon–Thu rule); they used to be marked taken and sold on the next plan."""
    held_set = set(held)
    tw = {s: 0.0 for s in names if s not in blocked or s in held_set}
    r = ranking(h, d, inp)
    # exit=(20, 10) on the live RS short-term strategy: keep a held stand-in while rank <= 20
    exit_below = exit_rank(h)
    rank_of = {s: i + 1 for i, s in enumerate(r.index)}
    for s in held_set:
        if s in tw or s in blocked or not _num(px.get(s)):
            continue
        if rank_of.get(s, 10**9) <= exit_below:
            tw[s] = 0.0
    taken = set(tw)
    soon = inp.get("soon")
    soon_d = soon.loc[d] if soon is not None and d in soon.index else pd.Series(dtype=bool)
    cool = inp["stretched"].loc[d] if _cfg(h).get("cool_buys") and d in inp["stretched"].index else None
    skipped = []
    for s in r.index:
        if len(tw) >= int(n_picks):
            break
        if rank_of[s] > exit_below:      # never refill with a stock ranked worse than exit_below (LIN #21, Nov 3, 2026)
            break
        if s in taken:
            continue
        if s in blocked:
            skipped.append(f"{s} #{rank_of[s]} ({blocked[s]})")
            continue
        if cool is not None and bool(cool.get(s, False)):   # skip stretched buys (live from Nov 6, 2026)
            skipped.append(f"{s} #{rank_of[s]} (more than 8% over its 20-day average)")
            continue
        if bool(soon_d.get(s, False)):   # earnings within 5 days: never a NEW buy (refills skipped it before Nov 3, 2026)
            skipped.append(f"{s} #{rank_of[s]} (earnings within 5 days)")
            continue
        if not _num(px.get(s)):
            skipped.append(f"{s} #{rank_of[s]} (no price)")
            continue
        tw[s] = 0.0
        taken.add(s)
    if skipped:
        print(f"  Refill candidates skipped (rank <= {exit_below}): " + "; ".join(skipped))
    if len(tw) < int(n_picks):
        print(f"  Only {len(tw)} of {int(n_picks)} slots filled with stocks ranked <= {exit_below}; the rest stays in cash")
    final = [s for s in tw if (s not in blocked or s in held_set) and (_num(px.get(s)) or s in held_set)]
    if not final:
        return {}
    # Prefer n_picks names: if we kept too many holds + ideal picks, drop worst-ranked extras
    # that are not needed (should be rare). Equal weight the final book.
    if len(final) > int(n_picks):
        final = sorted(final, key=lambda s: rank_of.get(s, 10**9))[:int(n_picks)]
    slot = float(invested) / len(final)
    return {s: slot for s in final}

def half_plan(h, d, st, px, value, init, inp, no_buy, no_sell, fractional, min_value):
    """One sizing rule on a trade day: each of the 10 book names gets invested/n (equal weight).
    Exits leave the book. Off-day: holds keep shares. No fill_idle / caps / scale-adds / renormalize."""
    import forward_test as ft
    import paper_trade as pt
    w = half_weights(h, d, inp)  # membership only — weight values are not used for sizing
    trade_today = init or half_day(h, d)
    held_v = {s: q * (_num(px.get(s)) or 0.0) for s, q in st["shares"].items()}
    held_w = pd.Series({s: v / value for s, v in held_v.items()}, dtype=float) if value > 0 else pd.Series(dtype=float)
    picks = sorted(w[w > 0].index.tolist())
    names = sorted(set(picks) | set(st["shares"]))
    hw = held_w.reindex(names).fillna(0.0)
    soon = inp.get("soon")
    soon_d = (soon.loc[d] if soon is not None and d in soon.index else pd.Series(False, index=w.index)).reindex(names).fillna(False)
    if not trade_today:
        tw = {s: float(hw[s]) for s in names if hw[s] > 0}
    else:
        tw = {s: 0.0 for s in picks}
    blocked = dict(no_buy or {})
    if trade_today:
        for s in names:
            if bool(soon_d.get(s, False)) and hw[s] > 0 and s in tw:
                blocked.setdefault(s, "earnings within 5 days (held: not topped up)")
    for s in list(tw):
        if float(hw.get(s, 0.0) or 0.0) <= 0 and not _num(px.get(s)):
            blocked.setdefault(s, "no live price")
    if trade_today:
        n_picks = int((_cfg(h).get("select") or {}).get("n") or ft.PIN.get("n") or 10)
        invested = float(ft.PIN.get("invested") or 0.99)
        held_set = {s for s in names if hw[s] > 0}
        tw = replace_blocked_new_slots(h, d, list(tw), held_set, blocked, px, inp, n_picks, invested)
    tg = pd.DataFrame({"Symbol": list(tw), "Weight": [tw[s] for s in tw], "Price": [px.get(s) for s in tw]})
    stat = {s: ("hold" if s in st["shares"] else "add") for s in tw}
    o = pt.build_orders(tg, value, st["shares"], px, min_value=min_value, fractional=fractional, statuses=stat,
                        blocked=blocked, keep=no_sell)
    return o, dict(tw), bool(trade_today and len(tw) > 0)


def plan(day, positions, equity, prices, no_buy=None, no_sell=None, fractional=True, ledger="load", min_value=1.0,
         inp=None, fills=None, pending=(), **_old):
    """Orders for session `day` (2:30 PM run): one netted order per symbol. Returns (orders, info); info['ledger'] = the
    ledger to save after the orders are sent."""
    import paper_trade as pt
    shares = config()
    if not shares:
        raise ValueError("WINNER['sleeves'] is off")
    led = load_ledger() if ledger == "load" else ledger
    d = pd.Timestamp(day).normalize()
    if led is not None and str(led.get("day", "")) > d.date().isoformat():
        raise ValueError(f"the sleeve ledger is from {led.get('day')}, after {d:%Y-%m-%d} - an old decision is not traded")
    inp = inp or inputs()
    px = {str(k).upper(): v for k, v in (prices or {}).items()}
    close = inp["close"]
    for h in HALVES:                                        # a target / refill candidate with no live price: the saved close
        cands = set(half_weights(h, d, inp).pipe(lambda x: x[x > 0]).index)
        cands |= set(ranking(h, d, inp).index[:exit_rank(h)])   # Mon-Thu refills (rank <= 20) are priced too (Nov 3, 2026)
        for s in sorted(cands):
            if not _num(px.get(s)) and s in close.columns and d in close.index and _num(close.at[d, s]):
                px[s] = float(close.at[d, s])
    state, unowned, info = reconcile(led, positions, px, equity, shares, fills=fills, pending=pending)
    init = info["init"]
    no_sell = no_sell or {}
    values = {h: half_value(state[h], px) for h in HALVES}
    # Unowned names (not do_not_sell) are sold this session — fold their value into the
    # sleeve so buys size against that cash. Otherwise sell proceeds sit idle (Chirag).
    unowned_mv = sum(q * (_num(px.get(s)) or 0.0) for s, q in unowned.items() if s not in no_sell)
    if unowned_mv > 0:
        for h in HALVES:
            values[h] += unowned_mv * float(shares.get(h, 0.0) or 0.0)
    parts, tgts, traded = {}, {}, {}
    for h in HALVES:
        if values[h] <= 0:
            raise ValueError(f"the {LABEL[h]} half is worth ${values[h]:,.2f} - refusing to size its orders")
        parts[h], tgts[h], traded[h] = half_plan(h, d, state[h], px, values[h], init, inp, no_buy, no_sell, fractional, min_value)
    # ---- net the halves into one order per symbol
    pos = {str(k).upper(): float(v) for k, v in (positions or {}).items() if float(v or 0) > 0}
    tgt_sh = {}
    half_sh = {h: {} for h in HALVES}
    for h in HALVES:
        for r in parts[h].itertuples():
            t = r.Target_Shares if r.Target_Shares is not None and not pd.isna(r.Target_Shares) else r.Current_Shares
            t = float(t or 0)
            if t > 0:
                half_sh[h][r.Symbol] = t
            tgt_sh[r.Symbol] = tgt_sh.get(r.Symbol, 0.0) + t
    for s, q in unowned.items():                            # claimed by no half: sold (do_not_sell kept)
        if s in no_sell:
            tgt_sh[s] = tgt_sh.get(s, 0.0) + q
        else:
            tgt_sh.setdefault(s, 0.0)
    rows = []
    for s in sorted(set(tgt_sh) | set(pos)):
        cur, tgt, p = pos.get(s, 0.0), tgt_sh.get(s, 0.0), _num(px.get(s))
        if s in no_sell and tgt < cur:
            tgt = cur
        diff = tgt - cur
        if fractional:
            q = math.floor(abs(diff) * 100 + 1e-6) / 100
        else:
            q = math.floor(abs(diff) + 1e-9)
        if tgt <= 0 and cur > 0:
            q = cur                                         # a full exit sells every share
        side = "HOLD"
        if p and q > 0 and q * p >= min_value:
            side = "BUY" if diff > 0 else "SELL"
        elif not p and q > 0:
            side = "SKIP (no price)"
        tv = tgt * p if p else None
        rows.append({"Symbol": s, "Side": side, "Shares": q if side in ("BUY", "SELL") else 0, "Price": p,
                     "Est_Value": round(q * p, 2) if side in ("BUY", "SELL") else 0.0, "Current_Shares": cur,
                     "Target_Shares": round(tgt, 4) if side in ("BUY", "SELL", "HOLD") else None,
                     "Target_Weight_%": round(100 * tv / equity, 2) if tv is not None and equity else None,
                     "Target_Value": round(tv, 2) if tv is not None else None,
                     "Half": "+".join(h for h in HALVES if s in half_sh[h]) or "none"})
    orders = pd.DataFrame(rows, columns=pt.ORDER_COLUMNS + ["Half"])
    orders["_o"] = orders["Side"].map({"SELL": 0}).fillna(1)
    orders = orders.sort_values(["_o", "Symbol"], kind="stable").drop(columns="_o").reset_index(drop=True)
    new = {"version": VERSION, "day": d.date().isoformat(), "cash_at": datetime.now(ZoneInfo("UTC")).isoformat(timespec="seconds"),
           "halves": {}}
    for h in HALVES:
        hv = sum(q * (_num(px.get(s)) or 0.0) for s, q in half_sh[h].items())
        new["halves"][h] = {"shares": {s: round(q, 4) for s, q in sorted(half_sh[h].items())},
                            "tags": sorted(half_sh[h]), "cash": round(values[h] - hv, 2), "value": round(values[h], 2),
                            "rebalanced": bool(traded[h])}
    info.update(day=d.date().isoformat(), values=values, targets=tgts, traded=traded, unowned=unowned, ledger=new,
                orders=orders, half_orders=parts, half_shares=half_sh, transfers=[])
    return orders, info

def plan_table(info, orders=None):
    """One row per symbol per half: the half's planned shares, price, value, % of the half and of the account."""
    rows, total = [], sum(info["values"].values()) or 1.0
    for h in HALVES:
        hv = info["values"][h]
        for r in info["half_orders"][h].itertuples():
            t = r.Target_Shares if r.Target_Shares is not None and not pd.isna(r.Target_Shares) else r.Current_Shares
            v = float(t or 0) * float(r.Price or 0)
            if v <= 0:
                continue
            rows.append({"Half": LABEL[h], "Symbol": r.Symbol, "Held_Shares": r.Current_Shares, "Target_Shares": t,
                         "Price": r.Price, "Target_Value": round(v, 2), "Target_in_half_%": round(100 * v / hv, 2) if hv else None,
                         "Target_of_account_%": round(100 * v / total, 2)})
    return pd.DataFrame(rows, columns=["Half", "Symbol", "Held_Shares", "Target_Shares", "Price", "Target_Value",
                                       "Target_in_half_%", "Target_of_account_%"])

def summary_text(info):
    v = info["values"]
    t = ", ".join(f"{LABEL[h]} ${v[h]:,.0f} ({'traded' if info['traded'][h] else 'held'})" for h in HALVES)
    if info.get("init"):
        t += "; first run: the sleeve starts at the whole equity"
    kept = set().union(*(info["half_shares"][h] for h in HALVES))
    gone = sorted(set(info.get("unowned") or {}) - kept)
    if gone:
        t += "; sold (not a target): " + ", ".join(gone)
    if abs(info.get("residual", 0.0)) >= 1:
        t += f"; ${info['residual']:,.2f} not explained by the ledger split pro rata"
    return t
