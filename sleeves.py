"""The live account as one sleeve - backtest_engine.WINNER["sleeves"] = {"rs_short": 1.0}.

Live from Fri Nov 6, 2026 (Chirag; locked ~12 weeks, to ~Jan 29, 2027): 100% forward_test.py's "Short-term relative
strength, skip stretched buys" - relative strength over 10/21/63 days (63-day x2 from Nov 6), top 10 equal weight (max 4 per sector), Friday rebalance only (the
week's last session; no Mon-Thu trades since Nov 4). A stock more than 15% above its 20-day average is not newly bought
(a held one in the top 10 is kept; held = this ledger). Blocked picks (earnings within 5 days, do_not_buy)
are replaced by the next eligible, not stretched stock ranked 20 or better. Soft QQQ regime; do_not_sell never sold.

Sizing: equal weight (equity × 0.99 × regime) / n per stock, then capped at 15% of equity (backtest_engine.MAX_WEIGHT);
regime = 0.5 when QQQ closes at or below its 200-day average on the decision day, else 1 (Chirag, Nov 8, 2026).
Excess over 15% and unfilled slots stay cash.

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
SKIP_CSV = os.environ.get("STOCK_ANALYSIS_DECISION_SKIPS") or os.path.join(REPORTS, "decision_skips.csv")
LEDGER_HISTORY = os.environ.get("STOCK_ANALYSIS_LEDGER_HISTORY") or os.path.join(REPORTS, "ledger_history")
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
    return full_day(h, day)

def calendar(start, sessions=30):
    """[(date, halves that rebalance in full)] for the next sessions from `start` (inclusive)."""
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

def rebalance_inputs(day=None, inp=None):
    """(day, inputs) with `day` (default: the latest close) treated as a Friday rebalance - the dashboard preview of the next
    Friday through the same plan() / strategy_targets path (read-only; the live run never uses this)."""
    inp = inp or inputs()
    d = pd.Timestamp(day if day is not None else inp["close"].index[-1]).normalize()
    targets(HALVES[0], inp)                               # sets inp["held"] (the ledger book) first
    wk = inp["weekly"].copy()
    wk.loc[d] = True
    return d, {**inp, "weekly": wk}

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

def display_ranking(h, day, inp=None):
    """DISPLAY ONLY (Chirag, Nov 5, 2026: rank every stock): ranking(h, day) first - the exact trading order, score
    above 0 - then every other buyable tradable stock by score, best first (score <= 0, ineligible, no score last),
    continuing the numbering. Dashboard views only; trading, news, chatbot and forward tests keep ranking()."""
    import sector_mapping as sm
    inp = inp or inputs()
    r = ranking(h, day, inp)
    sc = inp["scores"][_cfg(h).get("score", "live")]
    d = pd.Timestamp(day).normalize()
    if d not in sc.index:
        d = sc.index[sc.index <= d][-1]
    row = sc.loc[d]
    dnb = set(sm.do_not_buy)
    rest = [s for s in sm.tradable_symbols if s not in dnb and s not in r.index]
    rs = pd.Series({s: (float(row[s]) if s in row.index else float("nan")) for s in rest}, dtype=float)
    rs = rs.sort_values(ascending=False, kind="stable", na_position="last")
    return pd.concat([r.astype(float), rs]) if len(rs) else r

def live_ranks(day=None, inp=None, all_stocks=False):
    """{symbol: (rank, score)} of the live line (RS short-term) at `day` (default: the latest close) - display only.
    The pipeline's Strategy_Rank / Strategy_Score columns are the same live short-term RS rules (WINNER["score"])."""
    inp = inp or inputs()
    day = inp["close"].index[-1] if day is None else day
    r = (display_ranking if all_stocks else ranking)(HALVES[0], day, inp)   # all_stocks: dashboard only (score <= 0 too)
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

def save_ledger(led, path=None, skipped=None):
    path = path or LEDGER_JSON
    led = {**led, "saved_at": datetime.now(CT).strftime("%Y-%m-%d %H:%M:%S CT")}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(led, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    try:  # record keeping only: a failure here must never break a trade
        import trade_records as tr
        tr.after_ledger_save(led, path, skipped=skipped)
    except Exception as e:
        print(f"  decision records not saved (display only): {type(e).__name__}: {e}")
    return path

def ledger_book_weights(led, close):
    """{symbol: weight} of the last completed decision, from the live ledger (shares x that day's close / sleeve value).
    These are what was actually traded, not a later preview that may have been rescaled (e.g. a QQQ-filter rewrite)."""
    if not led:
        return {}
    st = ((led.get("halves") or {}).get(HALVES[0]) or {})
    shares = st.get("shares") or {}
    value = float(st.get("value") or 0.0)
    if value <= 0 or not shares:
        return {}
    d_led = pd.Timestamp(led["day"]).normalize()
    out = {}
    for s, q in shares.items():
        q = float(q or 0.0)
        if q <= 0:
            continue
        px = None
        if close is not None and s in getattr(close, "columns", []) and d_led in close.index:
            px = _num(close.at[d_led, s])
        if px:
            out[s] = q * px / value
    return out


def _completed_orders(day):
    """({bought}, {sold outright}) from the live order log on `day` (completed rows only)."""
    path = os.path.join(REPORTS, "live_orders_log.csv")
    if not db.exists(path):
        return set(), set()
    try:
        log = db.read_csv(path)
    except Exception:
        return set(), set()
    if log is None or not len(log):
        return set(), set()
    col = "As_Of" if "As_Of" in log.columns else ("Date" if "Date" in log.columns else None)
    if not col:
        return set(), set()
    day_s = pd.Timestamp(day).date().isoformat()
    asof = pd.to_datetime(log[col], errors="coerce").dt.strftime("%Y-%m-%d")
    rows = log[asof.eq(day_s)]
    if "Status" in rows.columns:
        rows = rows[rows["Status"].astype(str).str.contains("COMPLETED", case=False, na=False)]
    if not len(rows) or "Side" not in rows.columns:
        return set(), set()
    bought = set(rows.loc[rows["Side"].astype(str).str.upper().eq("BUY"), "Symbol"].astype(str))
    sold = set(rows.loc[rows["Side"].astype(str).str.upper().eq("SELL"), "Symbol"].astype(str))
    return bought, sold


def write_midweek_summary(day, path=None, inp=None, led=None):
    """Write Reports/strategy_midweek_check.csv: last completed decision from the live ledger (and order log), plus a
    labeled preview of the next Friday. Does not rewrite the traded book as a later target list."""
    import backtest_engine as be
    path = path or os.path.join(REPORTS, "strategy_midweek_check.csv")
    inp = inp or inputs()
    d = pd.Timestamp(day).normalize()
    H = HALVES[0]
    close = inp.get("close")
    led_w = ledger_book_weights(led, close)
    last = pd.Timestamp(led["day"]).normalize() if led and led.get("day") else None
    names = ", ".join(sorted(led_w) or ((led or {}).get("halves") or {}).get(H, {}).get("tags") or [])
    invested = sum(led_w.values()) if led_w else 0.0
    bought, sold = _completed_orders(last) if last is not None else (set(), set())
    held = set(led_w) or set(((led or {}).get("halves") or {}).get(H, {}).get("shares") or {})
    sold_out = sorted(sold - held)
    bought = sorted(bought)

    def _day(x):
        t = pd.Timestamp(x)
        return f"{t:%a %b} {t.day}"

    if last is not None:
        f0 = be.next_sessions(last, 1)[0]
        trade_bit = ""
        if sold_out or bought:
            bits = []
            if sold_out:
                bits.append("sold " + ", ".join(sold_out))
            if bought:
                bits.append("bought " + ", ".join(bought))
            trade_bit = " Traded that day: " + "; ".join(bits) + "."
        msg = (f"{STRATEGY[H]}: last completed decision {_day(last)} (from the live ledger): "
               f"held {names} ({len(led_w) or len(held)} names, {invested:.1%} invested).{trade_bit} "
               f"These are the names and weights that were actually traded, not a later preview.")
        event_date, applies = last, f0
        action = "REBALANCE"
        sell, buy = ", ".join(sold_out), ", ".join(bought)
        nxt_d, nxt_kind, nxt_fill = be.next_decision(last)
    else:
        nxt_d, nxt_kind, nxt_fill = be.next_decision(d)
        msg = f"{STRATEGY[H]}: no sleeve ledger yet; nothing to record as a completed decision."
        event_date, applies = d, d
        action, sell, buy = "NONE", "", ""

    preview_for, _, preview_fill = (nxt_d, nxt_kind, nxt_fill)
    _, p_inp = rebalance_inputs(d, inp)
    pw = half_weights(H, d, p_inp)
    pw = pw[pw > 0]
    prev_names = ", ".join(sorted(pw.index))
    prev_inv = float(pw.sum()) if len(pw) else 0.0
    next_msg = (f"Preview for next Friday {_day(preview_for)} {be.slot_text(preview_for)} "
                f"(not yet traded; ranks as of {_day(d)}): {prev_names} ({len(pw)} names, {prev_inv:.1%} invested). "
                f"Orders would go out {_day(preview_for)} at {be.slot_text(preview_for)}, any rest "
                f"{_day(preview_fill)} at 9 AM CT.")
    import backtest_engine as be2
    row = {
        "As_Of": d.date().isoformat(),
        "Event": "full rebalance" if last is not None else "none",
        "Event_Date": pd.Timestamp(event_date).date().isoformat(),
        "Applies_To_Open": pd.Timestamp(applies).date().isoformat(),
        "Action": action,
        "Sell": sell,
        "Buy": buy,
        "Message": msg,
        "Is_Latest": 1,
        "Next_Message": next_msg,
        "Rules": be2.WINNER.get("tag", ""),
        "Preview_For": pd.Timestamp(preview_for).date().isoformat(),
    }
    pd.DataFrame([row]).pipe(db.to_csv, path, index=False)
    return path


def write_strategy_picks(day, path=None, inp=None, led=None):
    """Write Reports/strategy_picks.csv. Strategy_Weight = the last completed decision from the live ledger (what was
    actually traded). Provisional_Weight = a preview of the next Friday, through the same finalize_book path, labeled
    Preview_For. As_Of is the latest session (the trade step's freshness check reads it). Written by
    main_signal_analysis.ipynb every run and again by paper_trade after a successful trade."""
    import backtest_engine as be
    import sector_mapping as sm
    path = path or os.path.join(REPORTS, "strategy_picks.csv")
    inp = inp or inputs()
    d = pd.Timestamp(day).normalize()
    close = inp["close"]
    soon = inp.get("soon")
    r = ranking(HALVES[0], d, inp)
    tags = set((led or {}).get("halves", {}).get(HALVES[0], {}).get("tags") or [])
    shares = set(((led or {}).get("halves") or {}).get(HALVES[0], {}).get("shares") or [])
    held = tags | shares
    led_w = ledger_book_weights(led, close)
    last_reb = pd.Timestamp(led["day"]).date().isoformat() if led and led.get("day") else d.date().isoformat()
    if led and led.get("day"):
        preview_for = be.next_decision(pd.Timestamp(led["day"]).normalize())[0]
    else:
        preview_for = be.next_decision(d)[0]
    _, p_inp = rebalance_inputs(d, inp)
    pw = half_weights(HALVES[0], d, p_inp)
    pw = pw[pw > 0]
    reg = inp.get("regime")
    regime_on = int(bool(reg.get(d, False))) if reg is not None else 0
    symbols = list(dict.fromkeys(list(led_w) + list(pw.index)))
    rows = []
    for s in symbols:
        wt = float(led_w.get(s, 0.0) or 0.0)
        pwt = float(pw.get(s, 0.0) or 0.0)
        rank = int(r.index.get_loc(s)) + 1 if s in r.index else None
        score = float(r.get(s)) if s in r.index else None
        px = float(close.at[d, s]) if s in close.columns and d in close.index and _num(close.at[d, s]) else None
        earn = None
        if soon is not None and d in soon.index and s in soon.columns and bool(soon.at[d, s]):
            earn = "within_5d"
        rows.append({
            "As_Of": d.date().isoformat(),
            "Last_Rebalance": last_reb,
            "Last_Decision": last_reb,
            "Preview_For": pd.Timestamp(preview_for).date().isoformat(),
            "Strategy": STRATEGY[HALVES[0]],
            "Regime_On": regime_on,
            "Symbol": s,
            "Sector": (sm.symbol_sector.get(s, "") if hasattr(sm, "symbol_sector") else ""),
            "Held": 1 if s in held else 0,
            "Strategy_Weight": round(wt, 4),
            "Provisional_Weight": round(pwt, 4),
            "Strategy_Score": None if score is None else round(score, 4),
            "Strategy_Rank": rank,
            "Technical_Score": None,
            "RS_Score": None if score is None else round(score, 4),
            "Close": None if px is None else round(px, 4),
            "Next_Earnings": earn,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["Strategy_Weight", "Provisional_Weight"], ascending=False)
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
    if led is None:                                     # first run: every position is unowned and plan() folds its value
        held = sum(q * (_num(prices.get(s)) or 0.0) for s, q in pos.items())   # back in, so cash = equity - holdings
        return ({h: {"shares": {}, "cash": (equity - held) * shares[h]} for h in HALVES}, dict(pos),
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

def _sector_key(s):
    """Sector for the cap. A missing mapping is unique per symbol so fake test tickers are not clumped as 'Other'."""
    import sector_mapping as sm
    sec = sm.symbol_sector.get(s) if hasattr(sm, "symbol_sector") else None
    return sec if sec else f"_{s}"


def finalize_book(names, held, rank_of, blocked, stretched, soon, priced, n_picks, invested,
                  max_rank=20, sector_cap=0.4, max_weight=None, skipped_out=None):
    """Shared membership + sizing for live replace_blocked_new_slots and forward_test.cool_buys.

    Keep a held name even if it is stretched or ranked worse than max_rank. Drop an unheld starting pick if it is
    ranked worse than max_rank, stretched, blocked, soon to report earnings, unpriced, or would break the sector cap
    (floor(sector_cap * n_picks); missing sector = unique per symbol). Fill remaining slots from rank order with the
    same gates. If the book is longer than n_picks, keep the best-ranked n (a held name worse than max_rank stays when
    the book is already exactly n; it drops only if more than n names are present and n better-ranked ones exist).
    Each slot is min(invested / len(book), max_weight); leftover cash stays cash.
    """
    import backtest_engine as be
    n_picks = int(n_picks)
    max_rank = int(max_rank) if max_rank else 10 ** 9
    cap = float(sector_cap) if sector_cap is not None else 0.4
    cap_n = n_picks if cap >= 1 else max(1, int(math.floor(cap * n_picks + 1e-9)))
    max_w = float(max_weight if max_weight is not None else getattr(be, "MAX_WEIGHT", 0.15))
    held_set, blocked, stretched, soon, priced = set(held or ()), dict(blocked or {}), set(stretched or ()), set(soon or ()), set(priced or ())
    rank_of = dict(rank_of or {})
    skipped = []

    def note(s, why):
        """Print/log a skip; also append a structured row when the caller asked for one."""
        rank = rank_of.get(s)
        skipped.append(f"{s} #{rank if rank is not None else '?'} ({why})")
        if skipped_out is not None:
            skipped_out.append({"Symbol": s, "Rank": None if rank is None else int(rank), "Reason": why})

    def reason_unheld(s):
        if rank_of.get(s, 10 ** 9) > max_rank:
            return f"rank {rank_of.get(s, '?')} > {max_rank}"
        if s in blocked:
            return str(blocked[s])
        if s in stretched:
            return "stretched"
        if s in soon:
            return "earnings within 5 days"
        if s not in priced:
            return "no price"
        return None

    book, seen = [], set()

    def add(s):
        if s not in seen:
            book.append(s)
            seen.add(s)

    # Starting names, best rank first so a sector-cap drop prefers the worse unheld name
    start = list(names or [])
    start.sort(key=lambda s: rank_of.get(s, 10 ** 9))
    counts = {}
    for s in start:
        if s in held_set:
            add(s)
            counts[_sector_key(s)] = counts.get(_sector_key(s), 0) + 1
            continue
        why = reason_unheld(s)
        if why:
            note(s, why)
            continue
        sec = _sector_key(s)
        if counts.get(sec, 0) >= cap_n:
            note(s, f"sector cap {cap_n}")
            continue
        add(s)
        counts[sec] = counts.get(sec, 0) + 1
    for s in held_set:
        if s in seen:
            continue
        if s in priced or s in start:
            add(s)
            counts[_sector_key(s)] = counts.get(_sector_key(s), 0) + 1
    ranked = sorted(rank_of, key=lambda s: rank_of[s])
    for s in ranked:
        if len(book) >= n_picks:
            break
        if rank_of[s] > max_rank:
            break
        if s in seen:
            continue
        why = reason_unheld(s)
        if why:
            note(s, why)
            continue
        sec = _sector_key(s)
        if counts.get(sec, 0) >= cap_n:
            note(s, f"sector cap {cap_n}")
            continue
        add(s)
        counts[sec] = counts.get(sec, 0) + 1
    if skipped and (book or names or held_set):
        print(f"  Refill candidates skipped (rank <= {max_rank}, max {cap_n}/sector): " + "; ".join(skipped))
    if len(book) < n_picks and (book or names or held_set):
        print(f"  Only {len(book)} of {n_picks} slots filled with stocks ranked <= {max_rank}; the rest stays in cash")
    if len(book) > n_picks:
        book = sorted(book, key=lambda s: rank_of.get(s, 10 ** 9))[:n_picks]
    if not book:
        return {}
    slot = min(float(invested) / len(book), max_w)
    return {s: slot for s in book}


def replace_blocked_new_slots(h, d, names, held, blocked, px, inp, n_picks, invested, skipped_out=None):
    """`n_picks` slots via finalize_book (sector cap, rank-20 new buys, 15% weight cap). Held names stay even if
    stretched or ranked worse than 20; new / swapped-in names must pass the same gates as the forward test."""
    import backtest_engine as be
    r = ranking(h, d, inp)
    rank_of = {s: i + 1 for i, s in enumerate(r.index)}
    d = pd.Timestamp(d).normalize()
    soon = inp.get("soon") if inp is not None else None
    soon_d = soon.loc[d] if soon is not None and d in soon.index else pd.Series(dtype=bool)
    try:
        sk = _cfg(h).get("stretch", "stretched")
        cool = inp[sk].loc[d] if inp is not None and _cfg(h).get("cool_buys") and sk in inp and d in inp[sk].index else None
    except Exception:
        cool = None
    stretched = {s for s in (cool.index if cool is not None else []) if bool(cool.get(s, False))}
    soon_set = {s for s in soon_d.index if bool(soon_d.get(s, False))}
    priced = {s for s, v in (px or {}).items() if _num(v)}
    try:
        sel = (_cfg(h).get("select") or {})
        sector_cap = sel.get("sector_cap", be.WINNER.get("sector_cap", 0.4))
    except Exception:
        sector_cap = be.WINNER.get("sector_cap", 0.4)
    return finalize_book(names, held, rank_of, blocked, stretched, soon_set, priced, n_picks, invested,
                         max_rank=exit_rank(h), sector_cap=sector_cap, max_weight=getattr(be, "MAX_WEIGHT", 0.15),
                         skipped_out=skipped_out)

def regime_scale(d, inp):
    """Soft QQQ market filter, as the forward test applies it (backtest_engine.rank_targets): WINNER['regime_scale'] (0.5)
    when QQQ closed at or below its 200-day average on session d (or the day is missing), else 1.0. Live from Nov 13, 2026
    (Chirag, Nov 8): a regime-off Friday invests 99% x 0.5 = 49.5% (equal slots of 4.95%); the rest stays cash."""
    import backtest_engine as be
    W = be.WINNER
    if not W.get("use_regime") or W.get("regime_scale") is None:
        return 1.0
    r = inp.get("regime")
    on = bool(r.get(pd.Timestamp(d).normalize(), False)) if r is not None else False
    return 1.0 if on else float(W["regime_scale"])

def half_plan(h, d, st, px, value, init, inp, no_buy, no_sell, fractional, min_value, rebalance=False):
    """One sizing rule on a trade day: finalize_book (equal weight invested/n, then the 15% per-stock cap).
    Exits leave the book. Off-day: holds keep shares. No fill_idle / scale-adds / renormalize."""
    import forward_test as ft
    import paper_trade as pt
    w = half_weights(h, d, inp)  # membership only — weight values are not used for sizing
    trade_today = init or rebalance or half_day(h, d)
    held_v = {s: q * (_num(px.get(s)) or 0.0) for s, q in st["shares"].items()}
    held_w = pd.Series({s: v / value for s, v in held_v.items()}, dtype=float) if value > 0 else pd.Series(dtype=float)
    picks = sorted(w[w > 0].index.tolist())
    names = sorted(set(picks) | set(st["shares"]))
    hw = held_w.reindex(names).fillna(0.0)
    soon = inp.get("soon")
    soon_d = (soon.loc[d] if soon is not None and d in soon.index else pd.Series(False, index=w.index)).reindex(names).fillna(False)
    skipped = []
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
        invested = float(ft.PIN.get("invested") or 0.99) * regime_scale(d, inp)   # QQQ below its 200-day: half invested
        held_set = {s for s in names if hw[s] > 0}
        skipped = []
        tw = replace_blocked_new_slots(h, d, list(tw), held_set, blocked, px, inp, n_picks, invested,
                                       skipped_out=skipped)
    tg = pd.DataFrame({"Symbol": list(tw), "Weight": [tw[s] for s in tw], "Price": [px.get(s) for s in tw]})
    stat = {s: ("hold" if s in st["shares"] else "add") for s in tw}
    o = pt.build_orders(tg, value, st["shares"], px, min_value=min_value, fractional=fractional, statuses=stat,
                        blocked=blocked, keep=no_sell)
    return o, dict(tw), bool(trade_today and len(tw) > 0), skipped


def plan(day, positions, equity, prices, no_buy=None, no_sell=None, fractional=True, ledger="load", min_value=1.0,
         inp=None, fills=None, pending=(), rebalance=False, **_old):
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
    parts, tgts, traded, skipped = {}, {}, {}, []
    for h in HALVES:
        if values[h] <= 0:
            raise ValueError(f"the {LABEL[h]} half is worth ${values[h]:,.2f} - refusing to size its orders")
        parts[h], tgts[h], traded[h], skipped_h = half_plan(h, d, state[h], px, values[h], init, inp, no_buy, no_sell, fractional, min_value, rebalance)
        skipped.extend(skipped_h)
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
    tgt_val = sum(q * (_num(px.get(s)) or 0.0) for h in HALVES for s, q in half_sh[h].items())
    has_buys = any(r.Side == "BUY" for r in orders.itertuples())
    cap = equity * (0.995 if has_buys else 1.0) + 1.0   # off-day / hold-only holds may drift up to the whole equity
    if tgt_val > cap:
        raise ValueError(f"sleeve targets ${tgt_val:,.2f} exceed ${cap:,.2f} of equity ${equity:,.2f} - no orders")
    new = {"version": VERSION, "day": d.date().isoformat(), "cash_at": datetime.now(ZoneInfo("UTC")).isoformat(timespec="seconds"),
           "halves": {}}
    for h in HALVES:
        hv = sum(q * (_num(px.get(s)) or 0.0) for s, q in half_sh[h].items())
        new["halves"][h] = {"shares": {s: round(q, 4) for s, q in sorted(half_sh[h].items())},
                            "tags": sorted(half_sh[h]), "cash": round(values[h] - hv, 2), "value": round(values[h], 2),
                            "rebalanced": bool(traded[h])}
    info.update(day=d.date().isoformat(), values=values, targets=tgts, traded=traded, unowned=unowned, ledger=new,
                orders=orders, half_orders=parts, half_shares=half_sh, transfers=[], skipped=skipped)
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
