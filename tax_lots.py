"""Tax lots of the LIVE Alpaca account for the dashboard's Trading Account tab (approved by Chirag, Sun Oct 4, 2026).

ESTIMATE, NOT TAX ADVICE. Alpaca's Form 1099-B / 1099-DIV / 1099-INT are the official record; this module only
re-creates the lots from the account's activity history so the dashboard can show where the year stands.

Inputs (read-only GETs through alpaca_paper.PaperAccount, the allow-listed live endpoints): every account activity
(/account/activities: FILL incl. partial fills and fractional shares, DIV*, INT*, FEE (REG / TAF / CAT), SSP stock
splits, ACATS / JNLS / FOPT transfers, other corporate actions), the open positions (/positions) and the orders
(/orders, only to tell the bot's own "live-..." orders apart). Nothing here sends anything.

Rules used (US individual, IRS Pub 550 / Form 8949 conventions):
  - Lots: FIFO by trade date (Alpaca's default); HIFO and LIFO are computed only as a comparison.
  - Trade date = the fill's date in New York time. Fees: REG / TAF (sell-side) lower that day's sale proceeds pro rata;
    CAT fees are spread over that day's fills (sales: lower proceeds, buys: added to the basis).
  - Holding period: long-term when sold after the one-year anniversary of the acquisition (held more than one year,
    counted from the day after the purchase); a Feb 29 purchase turns long-term after Feb 28 of the next year.
  - Wash sales (both directions): a sale at a loss with identical shares bought within 30 days before or after it
    (other shares of the same purchase count, the shares sold do not). Replacement shares are matched oldest purchase
    first, each share at most once, partial share counts allowed. The disallowed loss is added to the replacement
    shares' basis and the sold shares' holding period is added to theirs (Form 8949 code W, adjustment = the
    disallowed amount). Only this account is seen (not IRAs or other brokers).
  - Splits (SSP): open lots' shares x ratio, basis per share / ratio, dates unchanged.
  - Dividends: qualified is ESTIMATED with the 61-day holding test (more than 60 days in the 121-day window around the
    ex-date; ex-date taken = the record date, the T+1 rule since May 2024); the 1099-DIV decides.

Fresh start (Chirag, Sun Oct 4, 2026): only activity on or after TAX_START (a New York trade date) counts. Older fills,
dividends, interest and fees are ignored, and so is wash-sale matching against them. Shares held from before TAX_START
are left out of the lots (their basis and purchase date live in the older history; Alpaca's /positions has only an
average cost): they are worked out as Alpaca's position minus the shares bought plus the shares sold since TAX_START,
sold first (they are the oldest shares), and listed in report["pre_open"] / report["pre_sales"] instead of the gains.
build(..., start=None) uses the full history (the engine tests do).
"""
import copy
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")
TAX_START = date(2026, 10, 2)    # fresh start: the tax view counts only activity on/after this date (change it here)
PRE_START = "before start"       # Lot.source of the shares held from before TAX_START (left out of the view)
EPS = 1e-9
WASH_DAYS = 30
NIIT_RATE = 0.038
DISCLAIMER = ("Estimate, not tax advice. Rebuilt from your Alpaca account history; Alpaca's Form 1099 is the official record, "
              "and wash sales across other accounts (IRAs, a spouse, other brokers) are not seen here. Check with a tax professional.")
NOTE_1099B = ("1099-B check: Alpaca reports these sales as covered (Form 8949 box A short-term / box D long-term, basis reported "
              "to the IRS). Compare the totals by term with the 1099-B when it arrives (usually mid-February); if they differ, the "
              "1099-B wins unless it is wrong, and differences usually come from fees, wash-sale matching or corporate actions.")
METHODS = ("FIFO", "HIFO", "LIFO")
DIV_TYPES = {"DIV": "Dividend", "DIVCGL": "Capital gain distribution (long-term)", "DIVCGS": "Capital gain distribution (short-term)",
             "CGD": "Capital gain distribution", "DIVROC": "Return of capital", "DIVTXEX": "Tax-exempt dividend",
             "DIVFT": "Foreign tax withheld", "DIVNRA": "NRA tax withheld", "DIVTW": "Tax withheld (dividend)",
             "DIVFEE": "Dividend fee"}
INT_SUB = {"SWP": "Sweep interest", "MGN": "Margin interest paid", "FPSL": "Securities lending income",
           "CDT": "Interest on cash"}
TRANSFER_TYPES = {"ACATS", "JNLS", "FOPT"}
REVIEW_TYPES = {"MA": "merger / acquisition", "REORG": "reorganization", "SSO": "spin-off", "SC": "symbol change",
                "NC": "name change", "OPASN": "option assignment", "OPEXP": "option expiration", "OPXRC": "option exercise",
                "OPTRD": "option trade", "OPCA": "option corporate action", "OPCSH": "option cash"}
CASH_TYPES = {"CSD", "CSW", "JNLC", "JNL", "TRANS", "ACATC", "PTC", "PTR", "CFEE"}

# 2026 federal brackets (IRS, Rev. Proc. 2025-32 as amended by the One Big Beautiful Bill): upper bounds of each rate.
ORDINARY_2026 = {
    "single": [(12400, .10), (50400, .12), (105700, .22), (201775, .24), (256225, .32), (640600, .35), (math.inf, .37)],
    "mfj": [(24800, .10), (100800, .12), (211400, .22), (403550, .24), (512450, .32), (768700, .35), (math.inf, .37)],
    "mfs": [(12400, .10), (50400, .12), (105700, .22), (201775, .24), (256225, .32), (384350, .35), (math.inf, .37)],
    "hoh": [(17700, .10), (67450, .12), (105700, .22), (201775, .24), (256200, .32), (640600, .35), (math.inf, .37)],
}
LTCG_2026 = {"single": [(49450, 0.0), (545500, .15), (math.inf, .20)], "mfj": [(98900, 0.0), (613700, .15), (math.inf, .20)],
             "mfs": [(49450, 0.0), (306850, .15), (math.inf, .20)], "hoh": [(66200, 0.0), (579600, .15), (math.inf, .20)]}
STANDARD_DEDUCTION_2026 = {"single": 16100, "mfj": 32200, "mfs": 16100, "hoh": 24150}
NIIT_THRESHOLD = {"single": 200000, "mfj": 250000, "mfs": 125000, "hoh": 200000}     # not inflation-indexed
LOSS_LIMIT = {"single": 3000, "mfj": 3000, "mfs": 1500, "hoh": 3000}
FILING = {"single": "Single", "mfj": "Married filing jointly", "mfs": "Married filing separately", "hoh": "Head of household"}

# ----------------------------------------------------------------------------- dates
def one_year_after(d):
    try:
        return d.replace(year=d.year + 1)
    except ValueError:                       # Feb 29 -> Feb 28
        return d.replace(year=d.year + 1, day=28)

def is_long_term(acquired, sold):
    """Held more than one year: sold after the one-year anniversary of the acquisition."""
    return sold > one_year_after(acquired)

def long_term_on(acquired):
    """First sale date that counts as long-term."""
    return one_year_after(acquired) + timedelta(days=1)

def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan

def _trade_time(a):
    """(UTC timestamp, New York date) of an activity: fills have transaction_time, the others a date."""
    t = a.get("transaction_time")
    if t:
        ts = pd.Timestamp(t)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return ts, ts.tz_convert(ET).date()
    d = pd.Timestamp(str(a.get("date"))[:10]).date()
    return pd.Timestamp(datetime(d.year, d.month, d.day), tz=ET).tz_convert("UTC"), d

# ----------------------------------------------------------------------------- read-only fetch (cached by the app)
def _activity_date(a):
    """New York date an activity counts on (fees: the trade date in their description); None if it has none."""
    try:
        return _fee_date(a) if a.get("activity_type") == "FEE" else _trade_time(a)[1]
    except Exception:
        return None

def fetch_inputs(acct, max_pages=200, start=TAX_START):
    """Account activities (oldest first; paging stops once it reaches dates before `start`, None = all) and
    {order id: client order id}. GET only (the positions come from the dashboard's live holdings read)."""
    def before(d):
        return start is not None and d is not None and d < start
    acts, params = [], {"page_size": 100, "direction": "desc"}
    for _ in range(max_pages):
        page = acct._get("/account/activities", params)
        acts += page
        if len(page) < 100 or (page and before(_activity_date(page[-1]))):
            break
        params = {**params, "page_token": page[-1]["id"]}
    orders, params = [], {"status": "all", "limit": 500, "direction": "desc"}
    for _ in range(40):
        page = acct._get("/orders", params)
        orders += page
        if len(page) < 500 or (page and page[-1].get("submitted_at")
                               and before(pd.Timestamp(page[-1]["submitted_at"]).tz_convert(ET).date())):
            break
        params = {**params, "until": page[-1]["submitted_at"]}
    return {"activities": acts[::-1], "client_ids": {o["id"]: o.get("client_order_id") or "" for o in orders}}

# ----------------------------------------------------------------------------- lots
@dataclass
class Lot:
    lot_id: str
    symbol: str
    time: pd.Timestamp
    trade_date: date
    qty0: float                 # shares this lot (piece) was bought with
    cost_ps: float              # basis per share: price + fees + wash adjustments, after splits
    acq: date                   # holding-period start (earlier than trade_date after a wash sale)
    order_id: str = ""
    source: str = "buy"         # buy / transfer
    qty: float = 0.0            # shares still open (0 until bought)
    opened: bool = False
    replacement: bool = False   # already used as wash-sale replacement shares
    wash_adj_ps: float = 0.0
    wash_note: str = ""
    buy_id: str = ""            # the fill that bought it (pieces split off for wash sales share it)

class _Book:
    def __init__(self, method, bot_orders):
        self.method, self.bot_orders = method, bot_orders
        self.lots, self.disp, self.washes, self.issues, self.n = {}, [], [], [], 0
        self.pre_sales = []          # sales of shares held from before the start (left out of the gains)

    def issue(self, kind, text):
        self.issues.append({"Check": kind, "Detail": text})

    def avail(self, l):
        return l.qty if l.opened else l.qty0

    def carve(self, l, m):
        """The lot holding exactly m of l's available shares (l itself, or a new piece split off right after it)."""
        if m >= self.avail(l) - EPS:
            return l
        self.n += 1
        new = copy.copy(l)
        new.lot_id, new.qty0 = f"{l.lot_id}.{self.n}", m
        new.qty = m if l.opened else 0.0
        l.qty0 -= m
        if l.opened:
            l.qty -= m
        q = self.lots[l.symbol]
        q.insert(q.index(l) + 1, new)
        return new

    def sell(self, ev):
        sym, q, sold = ev["symbol"], ev["qty"], ev["date"]
        open_lots = [l for l in self.lots.get(sym, []) if l.opened and l.qty > EPS]
        key = {"FIFO": lambda l: (l.time, l.lot_id), "LIFO": lambda l: (-l.time.value, l.lot_id),
               "HIFO": lambda l: (-(l.cost_ps if l.cost_ps == l.cost_ps else -1e18), l.time)}[self.method]
        first = lambda l: (l.source != PRE_START, key(l))         # shares from before the start go first (oldest)
        pps = (q * ev["price"] - ev["fee"]) / q if q > EPS else math.nan
        left, pieces = q, []
        for l in sorted(open_lots, key=first):
            if left <= EPS:
                break
            m = min(left, l.qty)
            l.qty -= m
            left -= m
            pieces.append({"Symbol": sym, "Lot": l.lot_id, "Shares": m, "Acquired": l.acq, "Bought": l.trade_date,
                           "Sold": sold, "Proceeds": m * pps, "Basis": m * l.cost_ps, "Wash adj": 0.0,
                           "Order": ev["order_id"], "_lot": l})
        if left > 1e-6:
            self.issue("Unmatched sell", f"{sym}: sold {left:g} more shares on {sold} than the history shows as held "
                                         "(bought before the history starts, a transfer, or a short sale) - no basis for them.")
            pieces.append({"Symbol": sym, "Lot": "?", "Shares": left, "Acquired": None, "Bought": None, "Sold": sold,
                           "Proceeds": left * pps, "Basis": math.nan, "Wash adj": 0.0, "Order": ev["order_id"], "_lot": None})
        for p in pieces:
            lot = p.pop("_lot")
            if lot is not None and lot.source == PRE_START:      # out of scope: no gain, no wash-sale matching
                self.pre_sales.append({"Symbol": sym, "Shares": p["Shares"], "Sold": sold, "Proceeds": p["Proceeds"]})
                continue
            if p["Basis"] == p["Basis"] and p["Proceeds"] - p["Basis"] < -0.005:
                self.wash(p)
            self.disp.append(p)

    def wash(self, p):
        sym, sold, n = p["Symbol"], p["Sold"], p["Shares"]
        loss_ps = (p["Basis"] - p["Proceeds"]) / n
        lo, hi = sold - timedelta(days=WASH_DAYS), sold + timedelta(days=WASH_DAYS)
        cands = [l for l in self.lots.get(sym, []) if l.source == "buy" and not l.replacement
                 and lo <= l.trade_date <= hi and self.avail(l) > EPS]
        need = n
        for l in sorted(cands, key=lambda l: (l.time, l.lot_id)):
            if need <= EPS:
                break
            m = min(need, self.avail(l))
            r = self.carve(l, m)
            r.replacement = True
            r.cost_ps += loss_ps
            r.wash_adj_ps += loss_ps
            r.acq = r.acq - (sold - p["Acquired"])          # the sold shares' holding period carries over
            r.wash_note = f"wash sale: +${loss_ps * m:,.2f} loss from the {sold:%b %-d, %Y} sale"
            bot = self.bot_orders.get(r.order_id, "").startswith("live-")
            self.washes.append({"Symbol": sym, "Loss sale": sold, "Shares": m, "Disallowed loss": loss_ps * m,
                                "Replacement bought": r.trade_date, "Replacement lot": r.lot_id,
                                "Replacement order": "bot rebuy (live- order)" if bot else "other order (manual / earlier tools)",
                                "Before / after": "",
                                "Holding period carried (days)": (sold - p["Acquired"]).days})
            need -= m
        p["Wash adj"] = (n - need) * loss_ps

def _fee_date(a):
    m = re.search(r"on (\d{4}-\d{2}-\d{2})", str(a.get("description", "")))
    return pd.Timestamp(m.group(1)).date() if m else pd.Timestamp(str(a.get("date"))[:10]).date()

def _split_ratio(a, held):
    desc = str(a.get("description", ""))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-\s*)?(?:for|:)\s*(?:-\s*)?(\d+(?:\.\d+)?)", desc, re.I)
    if m and float(m.group(2)) > 0:
        return float(m.group(1)) / float(m.group(2))
    q = _num(a.get("qty"))
    if q == q and held > EPS and held + q > EPS:
        return (held + q) / held                  # qty = shares added (negative: reverse split)
    return None

def build(activities, positions=None, method="FIFO", bot_orders=None, today=None, start=TAX_START):
    """Rebuild lots, sales, wash sales, income and data checks from the activities (oldest first). start = the first
    trade date that counts (default TAX_START; None = the full history)."""
    bot_orders = bot_orders or {}
    today = today or datetime.now(ET).date()
    b = _Book(method, bot_orders)
    n_all = len(activities)
    if start is not None:
        activities = [a for a in activities if (_activity_date(a) or start) >= start]
    dropped = n_all - len(activities)
    fills, other = [], []
    for a in activities:
        (fills if a.get("activity_type") == "FILL" else other).append(a)
    # fills -> events with fees
    evs = []
    for a in fills:
        ts, d = _trade_time(a)
        qty, price = _num(a.get("qty")), _num(a.get("price"))
        side = str(a.get("side", "")).lower()
        if side.startswith("sell"):
            side = "sell"
        if not (qty > 0):
            b.issue("Bad fill", f"{a.get('symbol')}: fill on {d} with quantity {a.get('qty')!r} skipped.")
            continue
        evs.append({"kind": side, "symbol": str(a.get("symbol", "")).upper(), "qty": qty, "price": price, "time": ts,
                    "date": d, "fee": 0.0, "order_id": a.get("order_id") or "", "id": a.get("id") or f"f{len(evs)}"})
    # fees -> that trade date's fills
    fee_rows, unallocated = [], 0.0
    for a in other:
        if a.get("activity_type") != "FEE" or a.get("status") == "canceled":
            continue
        amt, d, sub = -_num(a.get("net_amount")), _fee_date(a), str(a.get("activity_sub_type") or "")
        if amt != amt:
            continue
        fee_rows.append({"Date": d, "Type": f"{sub or 'Other'} fee", "Amount": amt})
        sells = [e for e in evs if e["date"] == d and e["kind"] == "sell"]
        pool = sells if sub in ("REG", "TAF") else [e for e in evs if e["date"] == d]
        notional = sum(e["qty"] * e["price"] for e in pool if e["price"] == e["price"])
        if not pool or notional <= 0:
            unallocated += amt
            continue
        for e in pool:
            if e["price"] == e["price"]:
                e["fee"] += amt * e["qty"] * e["price"] / notional
    if unallocated > 0.005:
        b.issue("Fees not matched", f"${unallocated:,.2f} of fees had no fill on their trade date; left out of basis/proceeds.")
    # lots for every buy (created up front so wash sales can reach purchases after a sale)
    for e in evs:
        if e["kind"] != "buy":
            continue
        cps = (e["qty"] * e["price"] + e["fee"]) / e["qty"] if e["price"] == e["price"] and e["price"] > 0 else math.nan
        if cps != cps:
            b.issue("Missing basis", f"{e['symbol']}: buy of {e['qty']:g} shares on {e['date']} has no price.")
        lot = Lot(e["id"], e["symbol"], e["time"], e["date"], e["qty"], cps, e["date"], e["order_id"], buy_id=e["id"])
        b.lots.setdefault(e["symbol"], []).append(lot)
    # shares held from before the start = Alpaca's position - bought + sold since the start (left out, sold first)
    if start is not None and positions is not None:
        net = {}
        for e in evs:
            net[e["symbol"]] = net.get(e["symbol"], 0.0) + (e["qty"] if e["kind"] == "buy" else -e["qty"])
        cur = {str(p["symbol"]).upper(): _num(p.get("qty")) for p in positions}
        t0 = pd.Timestamp(datetime(start.year, start.month, start.day), tz=ET).tz_convert("UTC") - pd.Timedelta(microseconds=1)
        for sym in sorted(set(net) | set(cur)):
            x = (cur.get(sym, 0.0) if cur.get(sym, 0.0) == cur.get(sym, 0.0) else 0.0) - net.get(sym, 0.0)
            if x > 1e-6:
                d0 = start - timedelta(days=1)
                b.lots.setdefault(sym, []).insert(0, Lot(f"pre-{sym}", sym, t0, d0, x, math.nan, d0, "", PRE_START, x, True))
    # non-trade events that change shares
    income, review = [], {}
    for a in other:
        t, ts_d = a.get("activity_type"), _trade_time(a)
        if a.get("status") == "canceled":
            continue
        sym = str(a.get("symbol") or "").upper()
        if t == "SSP":
            evs.append({"kind": "split", "symbol": sym, "time": ts_d[0], "date": ts_d[1], "act": a})
        elif t in TRANSFER_TYPES:
            evs.append({"kind": "transfer", "symbol": sym, "time": ts_d[0], "date": ts_d[1], "act": a})
        elif t in DIV_TYPES or t == "INT" or t in ("INTNRA", "INTTW"):
            income.append(a)
        elif t in REVIEW_TYPES:
            review.setdefault(REVIEW_TYPES[t], []).append(f"{sym or '?'} {ts_d[1]}")
        elif t not in CASH_TYPES and t not in ("FEE",):
            review.setdefault(f"activity type {t}", []).append(f"{sym or '?'} {ts_d[1]}")
    for what, items in review.items():
        b.issue("Needs review", f"{what}: {', '.join(items[:6])}{' ...' if len(items) > 6 else ''} - not applied to the lots; "
                                "check the basis against the 1099-B.")
    order = {"split": 0, "transfer": 1, "buy": 2, "sell": 3}
    for e in sorted(evs, key=lambda e: (e["time"], order[e["kind"]])):
        if e["kind"] == "buy":                      # the lot and any piece split off it by a wash sale before the buy
            for l in b.lots[e["symbol"]]:
                if l.buy_id == e["id"] and not l.opened:
                    l.opened, l.qty = True, l.qty0
        elif e["kind"] == "sell":
            b.sell(e)
        elif e["kind"] == "split":
            held = sum(l.qty for l in b.lots.get(e["symbol"], []) if l.opened)
            r = _split_ratio(e["act"], held)
            if not r:
                b.issue("Needs review", f"{e['symbol']} split on {e['date']}: ratio unknown, lots not adjusted.")
                continue
            for l in b.lots.get(e["symbol"], []):
                if l.opened:
                    l.qty, l.qty0, l.cost_ps, l.wash_adj_ps = l.qty * r, l.qty0 * r, l.cost_ps / r, l.wash_adj_ps / r
        elif e["kind"] == "transfer":
            a, q = e["act"], _num(e["act"].get("qty"))
            if not q == q or abs(q) <= EPS:
                b.issue("Needs review", f"{e['symbol']} transfer on {e['date']} without a share count.")
            elif q > 0:
                basis = _num(a.get("cost_basis"))
                cps = basis / q if basis == basis else _num(a.get("price"))
                acq = pd.Timestamp(a["acquired_date"]).date() if a.get("acquired_date") else e["date"]
                if cps != cps:
                    b.issue("Missing basis", f"{e['symbol']}: {q:g} shares transferred in on {e['date']} without a cost basis.")
                if not a.get("acquired_date"):
                    b.issue("Needs review", f"{e['symbol']}: transfer-in on {e['date']} has no original purchase date "
                                            "(holding period starts at the transfer here).")
                lot = Lot(f"t-{a.get('id', e['date'])}", e["symbol"], e["time"], acq, q, cps, acq, "", "transfer", q, True)
                b.lots.setdefault(e["symbol"], []).append(lot)
            else:                                   # transfer out: shares leave without a sale (FIFO)
                left = -q
                for l in [l for l in b.lots.get(e["symbol"], []) if l.opened and l.qty > EPS]:
                    m = min(left, l.qty)
                    l.qty -= m
                    left -= m
                if left > 1e-6:
                    b.issue("Unmatched transfer", f"{e['symbol']}: {left:g} shares transferred out on {e['date']} were not held.")
    return {**_finish(b, positions or [], income, fee_rows, today), "start": start, "dropped": dropped}

def _segments(disp, lots):
    """(symbol, shares, bought, sold or None) for every share the history saw (dividend holding test)."""
    seg = [(d["Symbol"], d["Shares"], d["Bought"], d["Sold"]) for d in disp if d["Bought"] is not None]
    return seg + [(l.symbol, l.qty, l.trade_date, None) for q in lots.values() for l in q if l.opened and l.qty > EPS]

def _income(income, segs, today):
    rows = []
    for a in income:
        t, sub = a.get("activity_type"), str(a.get("activity_sub_type") or "")
        amt, d, sym = _num(a.get("net_amount")), _trade_time(a)[1], str(a.get("symbol") or "").upper()
        note, qual = "", math.nan
        if t == "INT":
            kind = INT_SUB.get(sub, "Interest")
            if sub == "SWP" and abs(amt) < EPS and _num(a.get("qty")) > 0:
                amt, note = _num(a.get("qty")), "sweep: Alpaca shows $0 net; the qty field is taken as the interest"
            if sub == "MGN":
                note = "investment interest expense (deductible only if you itemize, Form 4952)"
        elif t in ("INTNRA", "INTTW"):
            kind = "Tax withheld (interest)"
        else:
            kind = DIV_TYPES.get(t, t)
            if t == "DIV":
                m = re.search(r"Rec Date: (\d{4}-\d{2}-\d{2})", str(a.get("description", "")))
                ex = pd.Timestamp(m.group(1)).date() if m else d
                lo, hi = ex - timedelta(days=60), ex + timedelta(days=60)
                held = [(q, b, s) for s_, q, b, s in segs if s_ == sym and b < ex and (s is None or s >= ex)]
                tot = sum(q for q, _, _ in held)
                ok = sum(q for q, b, s in held
                         if ((min(s or hi, hi) - max(b + timedelta(days=1), lo)).days + 1) > 60)
                pending = hi > today and any(s is None for _, _, s in held)
                qual = amt * ok / tot if tot > EPS else 0.0
                note = ("qualified estimated: 61-day holding test, ex-date taken as the record date"
                        + ("; the test window is still open" if pending else ""))
                if tot <= EPS:
                    note = "no shares found at the ex-date in the history: counted as ordinary"
        rows.append({"Date": d, "Year": d.year, "Type": kind, "Symbol": sym, "Amount": amt, "Qualified (est.)": qual,
                     "Note": note})
    return pd.DataFrame(rows, columns=["Date", "Year", "Type", "Symbol", "Amount", "Qualified (est.)", "Note"])

def _finish(b, positions, income, fee_rows, today):
    disp = pd.DataFrame(b.disp, columns=["Symbol", "Lot", "Shares", "Acquired", "Bought", "Sold", "Proceeds", "Basis",
                                         "Wash adj", "Order"])
    if len(disp):
        disp["Gain"] = disp["Proceeds"] - disp["Basis"] + disp["Wash adj"]
        disp["Term"] = [("Long" if is_long_term(a, s) else "Short") if a is not None else "Unknown"
                        for a, s in zip(disp["Acquired"], disp["Sold"])]
        disp["Year"] = [s.year for s in disp["Sold"]]
    else:
        disp = disp.assign(Gain=[], Term=[], Year=[])
    price = {str(p["symbol"]).upper(): _num(p.get("current_price")) for p in positions}
    pos = {str(p["symbol"]).upper(): p for p in positions}
    rows, pre_open = [], []
    for sym, q in b.lots.items():
        for l in q:
            if l.qty < -1e-9:
                b.issue("Negative lot", f"{sym}: lot {l.lot_id} has {l.qty:g} shares.")
            if not l.opened or l.qty <= 1e-9:
                continue
            if l.source == PRE_START:
                pq, pc = _num(pos.get(sym, {}).get("qty")), _num(pos.get(sym, {}).get("cost_basis"))
                pre_open.append({"Symbol": sym, "Shares": l.qty, "Alpaca avg cost": pc / pq * l.qty if pq > EPS else math.nan,
                                 "Value": l.qty * price.get(sym, math.nan)})
                continue
            px = price.get(sym, math.nan)
            lt = long_term_on(l.acq)
            rows.append({"Symbol": sym, "Bought": l.trade_date, "Holding from": l.acq, "Shares": l.qty,
                         "Basis / share": l.cost_ps, "Basis": l.qty * l.cost_ps, "Price": px, "Value": l.qty * px,
                         "Gain": l.qty * (px - l.cost_ps), "Term": "Long" if is_long_term(l.acq, today) else "Short",
                         "Long-term on": lt, "Days to long-term": max(0, (lt - today).days),
                         "Wash adj": l.qty * l.wash_adj_ps, "Note": l.wash_note, "Source": l.source})
    lots = pd.DataFrame(rows, columns=["Symbol", "Bought", "Holding from", "Shares", "Basis / share", "Basis", "Price", "Value",
                                       "Gain", "Term", "Long-term on", "Days to long-term", "Wash adj", "Note", "Source"])
    # reconcile with Alpaca's positions
    held = lots.groupby("Symbol")["Shares"].sum().to_dict() if len(lots) else {}
    for r in pre_open:                                   # the left-out shares still count for the position check
        held[r["Symbol"]] = held.get(r["Symbol"], 0.0) + r["Shares"]
    for p in positions:
        s, q = str(p["symbol"]).upper(), _num(p.get("qty"))
        if q < -EPS:
            b.issue("Negative lot", f"{s}: Alpaca shows a short position ({q:g} shares).")
        if abs(held.get(s, 0.0) - q) > 1e-4:
            b.issue("Position mismatch", f"{s}: Alpaca holds {q:g} shares, the rebuilt lots {held.get(s, 0.0):g}.")
    for s in set(held) - {str(p["symbol"]).upper() for p in positions}:
        if positions:
            b.issue("Position mismatch", f"{s}: the rebuilt lots hold {held[s]:g} shares, Alpaca shows none.")
    if len(lots) and lots["Price"].isna().any():
        b.issue("Missing price", "no current price for " + ", ".join(sorted(set(lots.loc[lots["Price"].isna(), "Symbol"]))))
    if len(lots) and lots["Basis"].isna().any():
        b.issue("Missing basis", "open lots without a basis: " + ", ".join(sorted(set(lots.loc[lots["Basis"].isna(), "Symbol"]))))
    segs = _segments(b.disp, b.lots)
    inc = _income(income, segs, today)
    washes = pd.DataFrame(b.washes, columns=["Symbol", "Loss sale", "Shares", "Disallowed loss", "Replacement bought",
                                             "Replacement lot", "Replacement order", "Before / after",
                                             "Holding period carried (days)"])
    if len(washes):
        washes["Before / after"] = ["bought before the sale" if r <= s else "bought after the sale"
                                    for r, s in zip(washes["Replacement bought"], washes["Loss sale"])]
    issues = pd.DataFrame(b.issues, columns=["Check", "Detail"]).drop_duplicates()
    return {"sales": disp, "lots": lots, "washes": washes, "income": inc, "fees": pd.DataFrame(fee_rows, columns=["Date", "Type", "Amount"]),
            "issues": issues, "method": b.method, "today": today,
            "pre_open": pd.DataFrame(pre_open, columns=["Symbol", "Shares", "Alpaca avg cost", "Value"]),
            "pre_sales": pd.DataFrame(b.pre_sales, columns=["Symbol", "Shares", "Sold", "Proceeds"])}

# ----------------------------------------------------------------------------- summaries
def realized_by_year(sales):
    """Per tax year: short-term and long-term gain (after wash adjustments), disallowed losses, proceeds, basis."""
    cols = ["Year", "Short-term gain", "Long-term gain", "Total gain", "Wash sale disallowed", "Proceeds", "Basis", "Lot sales"]
    if sales is None or sales.empty:
        return pd.DataFrame(columns=cols)
    g = sales.groupby("Year")
    term = sales.pivot_table(index="Year", columns="Term", values="Gain", aggfunc="sum").reindex(columns=["Short", "Long"]).fillna(0.0)
    out = pd.DataFrame({"Short-term gain": term["Short"], "Long-term gain": term["Long"],
                        "Wash sale disallowed": g["Wash adj"].sum(), "Proceeds": g["Proceeds"].sum(),
                        "Basis": g["Basis"].sum(min_count=1), "Lot sales": g.size()}).reset_index()
    out["Total gain"] = out["Short-term gain"] + out["Long-term gain"]
    return out[cols].sort_values("Year", ascending=False).reset_index(drop=True)

def ytd(report, year=None):
    """Card numbers for the tax year."""
    year = year or report["today"].year
    s, l, inc = report["sales"], report["lots"], report["income"]
    sy = s[s["Year"] == year] if len(s) else s
    iy = inc[inc["Year"] == year] if len(inc) else inc
    fees = report["fees"]
    fy = fees[[d.year == year for d in fees["Date"]]] if len(fees) else fees
    div = iy[iy["Type"] == "Dividend"]
    interest = iy[iy["Type"].isin(["Sweep interest", "Interest on cash", "Interest", "Securities lending income"])]
    return {"year": year,
            "realized_st": float(sy.loc[sy["Term"] == "Short", "Gain"].sum()) if len(sy) else 0.0,
            "realized_lt": float(sy.loc[sy["Term"] == "Long", "Gain"].sum()) if len(sy) else 0.0,
            "wash_disallowed": float(sy["Wash adj"].sum()) if len(sy) else 0.0,
            "unrealized_st": float(l.loc[l["Term"] == "Short", "Gain"].sum()) if len(l) else 0.0,
            "unrealized_lt": float(l.loc[l["Term"] == "Long", "Gain"].sum()) if len(l) else 0.0,
            "dividends": float(div["Amount"].sum()), "qualified": float(div["Qualified (est.)"].sum()),
            "interest": float(interest["Amount"].sum()),
            "margin_interest": float(-iy.loc[iy["Type"] == "Margin interest paid", "Amount"].sum()),
            "fees": float(fy["Amount"].sum()) if len(fy) else 0.0, "sales": int(len(sy))}

def harvest(report, planned_buys=(), today=None):
    """(candidates, excluded) for selling at a loss now. Excluded: a purchase of the stock in the last 30 days (it would
    be the replacement) or a planned bot buy / hold (the bot would buy it back within 30 days: wash sale)."""
    today = today or report["today"]
    l, planned = report["lots"], set(planned_buys)
    cols = ["Symbol", "Shares at a loss", "Loss", "Short-term loss", "Long-term loss", "Next long-term date", "Why"]
    if l.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)
    loss = l[l["Gain"] < -0.005]
    recent = {}
    for sym, q in l.groupby("Symbol"):
        recent[sym] = q.loc[[(today - d).days <= WASH_DAYS for d in q["Bought"]], "Bought"].max()
    rows_ok, rows_ex = [], []
    for sym, q in loss.groupby("Symbol"):
        st = q.loc[q["Term"] == "Short"]
        nxt = st.loc[st["Days to long-term"] <= 60, "Long-term on"].min() if len(st) else None
        row = {"Symbol": sym, "Shares at a loss": q["Shares"].sum(), "Loss": q["Gain"].sum(),
               "Short-term loss": st["Gain"].sum(), "Long-term loss": q.loc[q["Term"] == "Long", "Gain"].sum(),
               "Next long-term date": nxt if nxt == nxt else None}
        why = []
        r = recent.get(sym)
        if r is not None and r == r:
            why.append(f"bought {r:%b %-d} (within 30 days): selling now is a wash sale")
        if sym in planned:
            why.append("the bot plans to hold or buy it: a rebuy within 30 days would make it a wash sale")
        if why:
            rows_ex.append({**row, "Why": "; ".join(why)})
        else:
            rows_ok.append({**row, "Why": "no purchase in the last 30 days and not in the bot's plan: don't rebuy for 31 days"
                                         + (f"; harvest before {nxt:%b %-d} to keep it short-term" if row["Next long-term date"] else "")})
    mk = lambda r: pd.DataFrame(r, columns=cols).sort_values("Loss").reset_index(drop=True)
    return mk(rows_ok), mk(rows_ex)

def turning_long_term(report, days=60):
    """Open lots that turn long-term within `days` days."""
    l = report["lots"]
    if l.empty:
        return l
    t = l[(l["Term"] == "Short") & (l["Days to long-term"] <= days)].copy()
    t["Window"] = ["within 30 days" if d <= 30 else "within 60 days" for d in t["Days to long-term"]]
    t["Hint"] = ["gain: waiting makes it long-term (lower rate)" if g > 0 else
                 "loss: selling before then keeps it short-term" for g in t["Gain"]]
    return t.sort_values("Days to long-term").reset_index(drop=True)

def form_8949(sales, year=None):
    """Form 8949-style rows: description, acquired, sold, proceeds, basis, adjustment code W + amount, gain, term, box."""
    s = sales if year is None else sales[sales["Year"] == year]
    rows = []
    for _, r in s.iterrows():
        adj, acq = float(r["Wash adj"]), r["Acquired"]
        rows.append({"Description": f"{r['Shares']:.6g} sh {r['Symbol']}",
                     "Date acquired": f"{acq:%m/%d/%Y}" if acq is not None and acq == acq else "VARIOUS",
                     "Date sold": f"{r['Sold']:%m/%d/%Y}", "Proceeds": round(r["Proceeds"], 2),
                     "Cost basis": round(r["Basis"], 2) if r["Basis"] == r["Basis"] else None,
                     "Adjustment code": "W" if adj > 0.005 else "", "Adjustment amount": round(adj, 2) if adj > 0.005 else 0.0,
                     "Gain or loss": round(r["Gain"], 2) if r["Gain"] == r["Gain"] else None, "Term": r["Term"],
                     "Box": {"Short": "A", "Long": "D"}.get(r["Term"], "?")})
    return pd.DataFrame(rows, columns=["Description", "Date acquired", "Date sold", "Proceeds", "Cost basis", "Adjustment code",
                                       "Adjustment amount", "Gain or loss", "Term", "Box"])

# ----------------------------------------------------------------------------- estimated tax
def bracket_tax(income, table):
    tax, lo = 0.0, 0.0
    for hi, rate in table:
        if income > lo:
            tax += (min(income, hi) - lo) * rate
        lo = hi
    return tax

def _ltcg_tax(ordinary, pref, table):
    """Tax on `pref` (long-term gains + qualified dividends) stacked on top of `ordinary` taxable income."""
    tax, lo, start, end = 0.0, 0.0, ordinary, ordinary + pref
    for hi, rate in table:
        a, b = max(start, lo), min(end, hi)
        if b > a:
            tax += (b - a) * rate
        lo = hi
    return tax

def net_capital(st, lt, carry_st=0.0, carry_lt=0.0, filing="single"):
    """Schedule D netting: (net short, net long, deductible loss, carryforward short, carryforward long)."""
    st, lt = st - carry_st, lt - carry_lt
    if st < 0 < lt or lt < 0 < st:
        x = st + lt
        st, lt = (x, 0.0) if (x < 0 and st < 0) or (x >= 0 and st > 0) else (0.0, x)
    net = st + lt
    if net >= 0:
        return st, lt, 0.0, 0.0, 0.0
    ded = min(-net, LOSS_LIMIT[filing])
    st_loss, lt_loss = -min(st, 0.0), -min(lt, 0.0)
    use_st = min(ded, st_loss)
    use_lt = ded - use_st
    return st, lt, ded, st_loss - use_st, lt_loss - use_lt

def estimate_tax(st, lt, qualified_div=0.0, ordinary_div=0.0, interest=0.0, filing="single", other_agi=150000.0,
                 deduction=None, state_rate=0.0495, carry_st=0.0, carry_lt=0.0):
    """Extra tax caused by this account for the year (2026 brackets), as a dict. other_agi = your income outside this
    account (adjusted gross income); deduction defaults to the 2026 standard deduction."""
    deduction = STANDARD_DEDUCTION_2026[filing] if deduction is None else deduction
    nst, nlt, ded, cf_st, cf_lt = net_capital(st, lt, carry_st, carry_lt, filing)
    base_taxable = max(0.0, other_agi - deduction)
    acct_ordinary = max(nst, 0.0) + ordinary_div + interest - ded
    pref = max(nlt, 0.0) + qualified_div
    ordinary = max(0.0, base_taxable + acct_ordinary)
    room = max(0.0, base_taxable + acct_ordinary + pref) - ordinary      # deduction can absorb part of pref
    fed_with = bracket_tax(ordinary, ORDINARY_2026[filing]) + _ltcg_tax(ordinary, min(pref, room), LTCG_2026[filing])
    fed_without = bracket_tax(base_taxable, ORDINARY_2026[filing])
    nii = max(0.0, max(nst, 0.0) + max(nlt, 0.0) + qualified_div + ordinary_div + interest)
    magi = other_agi + acct_ordinary + pref
    niit = NIIT_RATE * max(0.0, min(nii, magi - NIIT_THRESHOLD[filing]))
    state_base = acct_ordinary + pref
    state = state_rate * state_base
    return {"federal": fed_with - fed_without, "niit": niit, "state": state, "total": fed_with - fed_without + niit + state,
            "net_short": nst, "net_long": nlt, "loss_deducted": ded, "carry_short": cf_st, "carry_long": cf_lt,
            "magi": magi, "niit_threshold": NIIT_THRESHOLD[filing], "deduction": deduction}

# ----------------------------------------------------------------------------- display helpers
def open_lots_view(lots):
    """Open lots for display: pieces of the same purchase with the same basis and holding period merged."""
    cols = ["Symbol", "Bought", "Holding from", "Shares", "Basis / share", "Basis", "Price", "Value", "Gain", "Term",
            "Long-term on", "Days to long-term", "Note"]
    if lots is None or lots.empty:
        return pd.DataFrame(columns=cols)
    l = lots.assign(Note=lots["Note"].fillna(""), _bps=lots["Basis / share"].round(4))
    keys = ["Symbol", "Bought", "Holding from", "Term", "Long-term on", "Days to long-term", "Note", "_bps"]
    g = l.groupby(keys, dropna=False, sort=False).agg(Shares=("Shares", "sum"), Basis=("Basis", "sum"), Value=("Value", "sum"),
                                                       Gain=("Gain", "sum"), Price=("Price", "first")).reset_index()
    g["Basis / share"] = g["Basis"] / g["Shares"]
    return g[cols].sort_values(["Symbol", "Bought"]).reset_index(drop=True)

def wash_summary(washes, year):
    """Wash sales of loss sales in `year`, per stock: loss sales matched, shares, disallowed loss, bot rebuys."""
    cols = ["Symbol", "Wash sales", "Shares", "Disallowed loss", "Bot rebuys", "Last loss sale"]
    w = washes[[d.year == year for d in washes["Loss sale"]]] if len(washes) else washes
    if w.empty:
        return pd.DataFrame(columns=cols), w
    g = w.groupby("Symbol").agg(**{"Wash sales": ("Loss sale", "size"), "Shares": ("Shares", "sum"),
                                   "Disallowed loss": ("Disallowed loss", "sum"),
                                   "Bot rebuys": ("Replacement order", lambda s: int(s.str.startswith("bot").sum())),
                                   "Last loss sale": ("Loss sale", "max")}).reset_index()
    return g.sort_values("Disallowed loss", ascending=False).reset_index(drop=True)[cols], w
