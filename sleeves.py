"""Two independent halves in one Alpaca account (Chirag, Oct 15, 2026; live from Fri Oct 16) - backtest_engine.WINNER["sleeves"].

  lowvol  50%: the monthly low-volatility rules (the pipeline's Provisional_Weight in strategy_picks.csv), rebalanced on
               the last trading day of each month (+ the Oct 16 switch).
  sector  50%: forward_test.py's "Sector rotation" (top 3 sector ETFs by 63-day return, then their top 10 by the C6 score,
               inverse-volatility weights, max 4 per sector, 20% cap, QQQ halving, do_not_buy), full rebalance of this half
               every Friday (the week's last session). No overlap: it never buys a name the low-volatility half holds or
               targets; when the low-volatility half buys a name this half holds, the shares move to the low-volatility half
               at the current price (no trade) and this half rebalances the same day without it.

Each half is sized from its OWN value (its shares at today's prices + its own cash) and trades with paper_trade.build_orders
as if it were its own account (1-point band of the half, 99% invested, trims, rounding) - the backtested '50/50 + SR weekly,
no overlap' (two sleeves, each with its own cash). A half that does not rebalance today is left alone.

Ledger (Reports/sleeve_ledger.json), written after each trade run: per half the PLANNED shares after the run's orders, the
prices used and the half's cash after them. Next run, before planning (reconcile):
  - shares owned = the account's actual shares of each symbol in that half's ledger;
  - an order that did not fill (actual != planned) gives its planned dollars back to its own half's cash;
  - whatever is left (fill slippage, dividends, interest, a deposit, an account position no half owns) is split between the
    halves pro rata to their value; a position no half owns is sold at the next trade run.
First run (no ledger, the Oct 16 switch): each half = 50% of equity in cash; a held stock that one half targets is taken
over by that half at the current price, every other holding is sold.
"""
import json
import math
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "Reports")
LEDGER_JSON = os.environ.get("STOCK_ANALYSIS_SLEEVE_LEDGER") or os.path.join(REPORTS, "sleeve_ledger.json")
PLAN_CSV = os.environ.get("STOCK_ANALYSIS_SLEEVE_PLAN") or os.path.join(REPORTS, "sleeve_plan.csv")
HALVES = ("lowvol", "sector")
LABEL = {"lowvol": "Low volatility (monthly)", "sector": "Sector rotation (weekly)"}
SECTOR_STRATEGY = "Sector rotation"          # forward_test.STRATEGIES entry this half trades
CT = ZoneInfo("America/Chicago")

def config():
    """{'lowvol': 0.5, 'sector': 0.5} when the two halves are live, else None."""
    import backtest_engine as be
    s = be.WINNER.get("sleeves")
    return dict(s) if s else None

def is_week_end(day):
    """True when session `day` is the week's last session (Friday, or Thursday before a Friday holiday)."""
    import backtest_engine as be
    d = pd.Timestamp(day).normalize()
    return be.is_session(d) and be.next_sessions(d, 1)[0].isocalendar()[:2] != d.isocalendar()[:2]

def lowvol_day(day):
    """True when session `day` is a low-volatility rebalance (last trading day of the month, or a switch day)."""
    import backtest_engine as be
    d = pd.Timestamp(day).normalize()
    return be.is_session(d) and bool(be.winner_rebalance_days(pd.DatetimeIndex([d])).iloc[0])

def calendar(start, sessions=30):
    """[(date, halves trading)] for the next decision sessions from `start` (inclusive)."""
    import backtest_engine as be
    d0 = pd.Timestamp(start).normalize()
    days = [d0] + list(be.next_sessions(d0, sessions)) if be.is_session(d0) else list(be.next_sessions(d0, sessions))
    out = []
    for d in days:
        h = (["lowvol"] if lowvol_day(d) else []) + (["sector"] if is_week_end(d) else [])
        if h:
            out.append((d, h))
    return out

# ----------------------------------------------------------------------------- the sector-rotation half's targets
_INPUTS = {}

def sector_inputs(signal_csv=None, bench_csv=None):
    """forward_test.strategy_inputs on the saved data (cached per file modification time)."""
    import forward_test as ft
    signal_csv, bench_csv = signal_csv or ft.SIGNAL_CSV, bench_csv or ft.BENCH_CSV
    key = (signal_csv, os.path.getmtime(signal_csv), bench_csv, os.path.getmtime(bench_csv))
    if key not in _INPUTS:
        sig = pd.read_csv(signal_csv, usecols=ft.SIG_COLS, parse_dates=["Date"])
        _INPUTS.clear()
        _INPUTS[key] = ft.strategy_inputs(sig, None, None, ft._read(bench_csv), bars=[])   # no bar download
    return _INPUTS[key]

def sector_args(inp):
    """rank_targets settings of forward_test's Sector rotation (as forward_test.strategy_targets passes them), 10 stocks."""
    import backtest_engine as be
    import forward_test as ft
    cfg = next(c for c in ft.STRATEGIES if c["name"] == SECTOR_STRATEGY)
    a = be.winner_rank_args(inp["regime"])
    a.update({"max_weight": ft.MAX_WEIGHT, **ft.PIN, **(cfg.get("select") or {}), "sleeve": ()})
    a["n"] = 10
    return a, cfg

def sector_weights(day, exclude=(), held=None, inp=None):
    """The sector half's target weights (fractions of the half, 99% invested) on session `day`, as a Series by symbol:
    forward_test's Sector rotation selection for that one day, `exclude` (the low-volatility half's names) not
    eligible, do_not_buy never bought, `held` {symbol: weight in the half} = the half's holdings before the rebalance."""
    import backtest_engine as be
    import forward_test as ft
    import sector_mapping as sm
    inp = inp or sector_inputs()
    d = pd.Timestamp(day).normalize()
    idx = inp["close"].index
    if d not in idx:
        raise ValueError(f"no data for {d:%Y-%m-%d} in signal_analysis.csv - the sector half cannot be ranked")
    t = idx.get_loc(d)
    cols = list(inp["close"].columns)
    args, cfg = sector_args(inp)
    score = inp["scores"][cfg.get("score", "live")]
    el = inp["eligible"].copy()
    ex = [x for x in exclude if x in el.columns]
    if ex:
        el.loc[idx[[t]], ex] = False
    block = pd.DataFrame(np.nan, index=idx[[t]], columns=cols)
    if cfg.get("earnings", ft.EARN_DAYS):          # (Sector rotation has no earnings rule)
        block = be.earnings_days_ahead(idx[[t]], cols, inp["earnings"], cfg.get("earnings", ft.EARN_DAYS)).reindex(columns=cols)
    block.loc[:, [x for x in sm.do_not_buy if x in cols]] = 999.0
    hw = np.array([float((held or {}).get(c, 0.0)) for c in cols])
    w = be.rank_targets(score.iloc[[t]], el, inp["vol"], rebalance_days=pd.Series(True, index=idx[[t]]),
                        tiebreak_w=inp["scores"].get(cfg.get("tiebreak", "rs")), buy_block=block, start_holdings=hw,
                        **args).iloc[0]
    return be.live_weights(w.clip(upper=ft.MAX_WEIGHT)).reindex(cols).fillna(0.0)

# ----------------------------------------------------------------------------- ledger
def load_ledger(path=None):
    path = path or LEDGER_JSON
    if not os.path.exists(path):
        return None
    with open(path) as f:
        led = json.load(f)
    if not isinstance(led, dict) or set(led.get("halves", {})) != set(HALVES):
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

def _num(x):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None

def reconcile(led, positions, prices, equity, shares=None):
    """(state, unowned, info): state = {half: {'shares': {sym: actual qty}, 'cash': $}} after the reconciliation in the
    module doc; unowned = {sym: qty} the account holds that no half owns. No ledger -> each half = its share of equity in cash."""
    shares = shares or config() or {"lowvol": 0.5, "sector": 0.5}
    pos = {str(k).upper(): float(v) for k, v in (positions or {}).items() if float(v or 0) > 0}
    if led is None:
        return ({h: {"shares": {}, "cash": equity * shares[h]} for h in HALVES}, dict(pos),
                {"init": True, "residual": 0.0, "unfilled": {}})
    state, owner, unfilled = {}, {}, {}
    for h in HALVES:
        L = led["halves"][h]
        cash = float(L.get("cash", 0.0))
        own = {}
        for sym, planned in (L.get("shares") or {}).items():
            sym = sym.upper()
            if sym in owner:            # (never by construction) first half keeps it
                continue
            owner[sym] = h
            actual = pos.get(sym, 0.0)
            p = _num((L.get("prices") or {}).get(sym)) or _num(prices.get(sym)) or 0.0
            if abs(actual - float(planned)) > 1e-6:
                cash += (float(planned) - actual) * p          # unfilled buy -> its dollars back; unfilled sell -> none received
                unfilled[sym] = (h, float(planned), actual)
            if actual > 0:
                own[sym] = actual
        state[h] = {"shares": own, "cash": cash}
    unowned = {s: q for s, q in pos.items() if s not in owner}
    val = {h: state[h]["cash"] + sum(q * (_num(prices.get(s)) or 0.0) for s, q in state[h]["shares"].items()) for h in HALVES}
    resid = equity - sum(val.values())
    tot = sum(max(v, 0.0) for v in val.values())
    for h in HALVES:
        state[h]["cash"] += resid * (max(val[h], 0.0) / tot if tot > 0 else shares[h])
    return state, unowned, {"init": False, "residual": resid, "unfilled": unfilled}

def half_value(st, prices):
    return st["cash"] + sum(q * (_num(prices.get(s)) or 0.0) for s, q in st["shares"].items())

def _take(state, h, sym, qty, price, frm=None):
    """Half h takes `qty` shares of sym at `price` (from half `frm`, or an unowned holding) - no trade."""
    v = qty * price
    state[h]["shares"][sym] = state[h]["shares"].get(sym, 0.0) + qty
    state[h]["cash"] -= v
    if frm is not None:
        state[frm]["shares"].pop(sym, None)
        state[frm]["cash"] += v

def plan(day, positions, equity, prices, lv_targets=None, lv_statuses=None, no_buy=None, fractional=True, ledger="load",
         min_value=1.0, inp=None):
    """Orders for session `day` (2:30 PM run). lv_targets: DataFrame Symbol, Weight (fractions of the low-volatility half,
    = Provisional_Weight), Price - needed when the low-volatility half rebalances. prices {sym: price} for every held / target
    symbol. Returns (orders, info); info['ledger'] = the ledger to save after the orders are sent."""
    import paper_trade as pt
    shares = config()
    if not shares:
        raise ValueError("WINNER['sleeves'] is off")
    led = load_ledger() if ledger == "load" else ledger
    d = pd.Timestamp(day).normalize()
    do_lv, do_sec = lowvol_day(d), is_week_end(d)
    if led is None and not do_lv:
        raise ValueError("no sleeve ledger (Reports/sleeve_ledger.json) and today is not a low-volatility rebalance - "
                         "the two halves start at a full rebalance; no orders")
    if led is not None and str(led.get("day", "")) > d.date().isoformat():
        raise ValueError(f"the sleeve ledger is from {led.get('day')}, after {d:%Y-%m-%d} - an old decision is not traded")
    px = {str(k).upper(): v for k, v in (prices or {}).items()}
    state, unowned, info = reconcile(led, positions, px, equity, shares)
    if info["init"]:
        do_sec = True                                   # the switch: both halves are bought
    transfers = []
    lv_w = {}
    if do_lv:
        if lv_targets is None:
            raise ValueError("the low-volatility half rebalances today but its targets are missing")
        lv_w = {str(s).upper(): float(w) for s, w in zip(lv_targets["Symbol"], lv_targets["Weight"]) if float(w) > 0}
        for s, p in zip(lv_targets["Symbol"], lv_targets["Price"]):
            px.setdefault(str(s).upper(), p)
        for sym in lv_w:
            p = _num(px.get(sym))
            if sym in state["sector"]["shares"] and p:
                transfers.append((sym, "sector", "lowvol", state["sector"]["shares"][sym], p))
                _take(state, "lowvol", sym, state["sector"]["shares"][sym], p, frm="sector")
                do_sec = True                           # the sector half rebalances without it (backtest: a clash)
            elif sym in unowned and p:
                transfers.append((sym, None, "lowvol", unowned[sym], p))
                _take(state, "lowvol", sym, unowned.pop(sym), p)
    sec_w = {}
    if do_sec:
        lv_names = set(state["lowvol"]["shares"]) | set(lv_w)
        vs = half_value(state["sector"], px)
        held = {s: q * (_num(px.get(s)) or 0.0) / vs for s, q in state["sector"]["shares"].items()} if vs > 0 else {}
        w = sector_weights(d, exclude=lv_names, held=held, inp=inp)
        sec_w = {s: float(x) for s, x in w.items() if x > 0}
        missing = [s for s in sec_w if not _num(px.get(s))]
        if missing:
            ip = sector_inputs()["close"]
            for s in missing:
                if s in ip.columns and d in ip.index and _num(ip.at[d, s]):
                    px[s] = float(ip.at[d, s])
        for sym in sec_w:
            if sym in unowned and _num(px.get(sym)):
                transfers.append((sym, None, "sector", unowned[sym], px[sym]))
                _take(state, "sector", sym, unowned.pop(sym), px[sym])
    no_buy = no_buy or {}
    parts, values = [], {h: half_value(state[h], px) for h in HALVES}
    for h, on, wts, stat in (("lowvol", do_lv, lv_w, lv_statuses), ("sector", do_sec, sec_w, None)):
        st = state[h]
        if on:
            if values[h] <= 0:
                raise ValueError(f"the {LABEL[h]} half is worth ${values[h]:,.2f} - refusing to size its orders")
            tg = pd.DataFrame({"Symbol": list(wts), "Weight": [wts[s] for s in wts], "Price": [px.get(s) for s in wts]})
            if stat is None:                            # the sector half's own picks: bought / kept (no signal file)
                stat = {s: ("hold" if s in st["shares"] else "add") for s in wts}
            o = pt.build_orders(tg, values[h], st["shares"], px, min_value=min_value, fractional=fractional,
                                statuses=stat, blocked=no_buy)
        else:
            o = pt.build_hold_orders(st["shares"], px)
        parts.append(o.assign(Half=h))
    if unowned:                                         # held by no half: sold in full
        o = pt.build_orders(pd.DataFrame(columns=["Symbol", "Weight", "Price"]), equity, unowned, px,
                            min_value=0.0, fractional=fractional)
        parts.append(o.assign(Half="none"))
    orders = pd.concat(parts, ignore_index=True)
    trade = orders[orders["Side"].isin(["BUY", "SELL"])]
    dup = trade["Symbol"][trade["Symbol"].duplicated()]
    if len(dup):
        raise ValueError(f"both halves trade {sorted(set(dup))} - refusing (no-overlap broken)")
    orders["_o"] = orders["Side"].map({"SELL": 0}).fillna(1)
    orders = orders.sort_values(["_o"], kind="stable").drop(columns="_o").reset_index(drop=True)
    # the ledger after this run's orders (planned shares at the plan's prices)
    new = {"version": 1, "day": d.date().isoformat(), "halves": {}}
    for h in HALVES:
        st = state[h]
        sh = dict(st["shares"])
        for r in orders[(orders["Half"] == h) & orders["Side"].isin(["BUY", "SELL"])].itertuples():
            q = float(r.Shares) * (1 if r.Side == "BUY" else -1)
            sh[r.Symbol] = round(sh.get(r.Symbol, 0.0) + q, 6)
        sh = {s: (q if q > 1e-6 else 0.0) for s, q in sh.items()}
        cash = values[h] - sum(q * (_num(px.get(s)) or 0.0) for s, q in sh.items())
        new["halves"][h] = {"shares": sh, "prices": {s: float(_num(px.get(s)) or 0.0) for s in sh}, "cash": round(cash, 2),
                            "value": round(values[h], 2), "rebalanced": bool(do_lv if h == "lowvol" else do_sec)}
    info.update(day=d.date().isoformat(), lowvol=do_lv, sector=do_sec, transfers=transfers, values=values,
                lv_targets=lv_w, sector_targets=sec_w, unowned=unowned, ledger=new)
    return orders, info

def plan_table(info, prices=None):
    """One row per symbol of each half after the plan: half, planned shares, value, weight in the half and in the account."""
    rows = []
    led = info["ledger"]["halves"]
    total = sum(info["values"].values()) or 1.0
    for h in HALVES:
        L = led[h]
        for s, q in sorted(L["shares"].items()):
            if q <= 0:
                continue
            v = q * L["prices"].get(s, 0.0)
            tw = (info["lv_targets"] if h == "lowvol" else info["sector_targets"]).get(s)
            rows.append({"Half": LABEL[h], "Symbol": s, "Shares": q, "Price": L["prices"].get(s), "Value": round(v, 2),
                         "Weight_in_half_%": round(100 * v / info["values"][h], 2) if info["values"][h] else None,
                         "Target_in_half_%": round(100 * tw, 2) if tw is not None else None,
                         "Weight_of_account_%": round(100 * v / total, 2)})
        rows.append({"Half": LABEL[h], "Symbol": "(cash)", "Shares": None, "Price": None, "Value": L["cash"],
                     "Weight_in_half_%": round(100 * L["cash"] / info["values"][h], 2) if info["values"][h] else None,
                     "Target_in_half_%": None, "Weight_of_account_%": round(100 * L["cash"] / total, 2)})
    return pd.DataFrame(rows)

def summary_text(info):
    v = info["values"]
    t = (f"halves: {LABEL['lowvol']} ${v['lowvol']:,.0f} ({'rebalanced' if info['lowvol'] else 'held'}), "
         f"{LABEL['sector']} ${v['sector']:,.0f} ({'rebalanced' if info['sector'] else 'held'})")
    if info.get("init"):
        t += "; first run: each half starts at 50% of equity"
    if info.get("transfers"):
        t += "; moved without a trade: " + ", ".join(f"{s} {q:g} sh {a or 'unowned'} -> {b}" for s, a, b, q, _ in info["transfers"])
    if info.get("unowned"):
        t += "; sold (no half owns them): " + ", ".join(sorted(info["unowned"]))
    if abs(info.get("residual", 0.0)) >= 1:
        t += f"; ${info['residual']:,.2f} not explained by the ledger split pro rata"
    return t
