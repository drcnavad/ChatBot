"""Band-hold audit of the live rebalance math (paper_trade.build_orders + apply_buying_power_guard, the exact path of
auto_trade) and fake-broker runs of Fri 2026-10-02 (full rebalance) and Mon 2026-10-05 (mid-week check). Zero broker calls.

A held pick within the 1-point no-trade band keeps its old weight (e.g. TWLO 9.14% vs a new 8.56% target) and new buys are
sized at their own targets. Trim fix (2026-10-01): when new buys do not fit (the plan would end above 99%, or above the cash +
planned sells less the 1% cushion), band-held names ABOVE target are trimmed toward (never below) their target, pro rata to
their excess, so each new pick gets close to its full weight; the buying-power guard then only absorbs a small residual. So the
total stays <= 99% with new buys, each band-held name is <= 1 point off, and a below-target band hold leaves that shortfall in
cash. Same rule in the backtest (simulate(band=...): sells first, the same trim, buys scaled to cash).
Run: PYTHONPATH=. python tests/test_band_hold.py"""
import json
import os
import sys
import tempfile
import types
from enum import Enum as _Enum

os.environ.setdefault("STOCK_ANALYSIS_RUN_LOG", os.path.join(tempfile.gettempdir(), "sa_test_run_log.csv"))  # never the real log
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# --- stub alpaca.trading + dotenv (no keys, no network) -----------------------------------------------------------
class OrderSide(_Enum):
    BUY = "buy"
    SELL = "sell"

class TimeInForce(_Enum):
    DAY = "day"

class QueryOrderStatus(_Enum):
    ALL = "all"
    OPEN = "open"
    CLOSED = "closed"

class Sort(_Enum):
    DESC = "desc"
    ASC = "asc"

class _Req:
    def __init__(self, **kw):
        self.__dict__.update(kw)

class MarketOrderRequest(_Req): pass
class LimitOrderRequest(_Req): pass
class GetOrdersRequest(_Req): pass

_mods = {n: types.ModuleType(n) for n in ("alpaca", "alpaca.trading", "alpaca.trading.client", "alpaca.trading.enums",
                                          "alpaca.trading.requests", "alpaca.common", "alpaca.common.enums", "dotenv")}
_mods["alpaca.trading.enums"].__dict__.update(OrderSide=OrderSide, TimeInForce=TimeInForce, QueryOrderStatus=QueryOrderStatus)
_mods["alpaca.common.enums"].Sort = Sort
_mods["alpaca.trading.requests"].__dict__.update(MarketOrderRequest=MarketOrderRequest, LimitOrderRequest=LimitOrderRequest,
                                                  GetOrdersRequest=GetOrdersRequest)
_mods["alpaca.trading.client"].TradingClient = type("TradingClient", (), {})
_mods["dotenv"].load_dotenv = lambda *a, **k: None
sys.modules.update(_mods)

import pandas as pd  # noqa: E402

import paper_trade as pt  # noqa: E402

FAIL = []

def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)

# ================================================================= A. the rebalance math (auto_trade's own two steps)
def friday(targets, held, prices, cash, statuses=None, blocked=None):
    """targets {sym: weight}, held {sym: shares}. Plans like auto_trade (equity = holdings + cash; build_orders with
    2-decimal shares; buying-power guard at cash + planned sells) and fills every order at the plan price.
    Returns (orders, end weights {sym: %}, end cash %, equity)."""
    equity = cash + sum(q * prices[s] for s, q in held.items())
    t = pd.DataFrame([(s, w, prices[s]) for s, w in targets.items()], columns=["Symbol", "Weight", "Price"])
    st = statuses or {s: ("hold" if s in held else "add") for s in targets}
    o = pt.build_orders(t, equity, held, prices, fractional=True, statuses=st, blocked=blocked)
    sells = o.loc[o.Side == "SELL", "Est_Value"].sum()
    o = pt.apply_buying_power_guard(o, cash + sells, fractional=True)
    end = dict(held)
    for r in o.itertuples():
        if r.Side == "BUY":
            end[r.Symbol] = end.get(r.Symbol, 0) + r.Shares
        elif r.Side == "SELL":
            end[r.Symbol] = end.get(r.Symbol, 0) - r.Shares
    spent = sum(r.Shares * r.Price for r in o.itertuples() if r.Side == "BUY")
    got = sum(r.Shares * r.Price for r in o.itertuples() if r.Side == "SELL")
    w = {s: q * prices[s] / equity * 100 for s, q in end.items() if q > 1e-9}
    return o, w, (cash + got - spent) / equity * 100, equity

def side(o, s):
    return o.set_index("Symbol").at[s, "Side"]

P = {f"S{i}": 100.0 for i in range(12)}
T10 = {f"S{i}": 0.099 for i in range(10)}                       # 10 picks x 9.9% = the live 99%

def shares_for(pcts):
    return {s: round(p * 1000 / 100, 4) for s, p in pcts.items()}  # equity ~ $100,000, price $100

# A1. several band-held names ABOVE target (+0.9 point each) and 5 new buys -> trim fix: the 5 are trimmed to target
held = shares_for({f"S{i}": 10.8 for i in range(5)})
o, w, c, eq = friday(T10, held, P, cash=100000 - 54000)
check("above (trim fix): the 5 band-held names are trimmed (SELL) to ~9.9%, never below",
      all(side(o, f"S{i}") == "SELL" and 9.9 - 1e-6 <= w[f"S{i}"] < 9.92 for i in range(5)), w)
check("above (trim fix): the new buys get their full 9.9% (not scaled)", all(abs(w[f"S{i}"] - 9.9) < 0.02 for i in range(5, 10)), w)
check("above: total invested <= 99% and cash never negative", sum(w.values()) <= 99 + 1e-6 and c >= 0, (sum(w.values()), c))
check("above: the 1% cash is kept (cash >= ~1% of equity)", c >= 0.99, c)
check("above: every stock within 1 point of its target", all(abs(w[s] - 9.9) <= 1.0 for s in T10), w)
check("above: SELLs before BUYs", list(o.Side[o.Side.isin(["SELL", "BUY"])]) == ["SELL"] * 5 + ["BUY"] * 5)
print(f"   above: total {sum(w.values()):.2f}%  cash {c:.2f}%  new buys {[round(w[f'S{i}'], 2) for i in range(5, 10)]}")

# A2. several band-held names BELOW target (-0.9 point each)
held = shares_for({f"S{i}": 9.0 for i in range(5)})
o, w, c, eq = friday(T10, held, P, cash=100000 - 45000)
check("below: band-held names not topped up (HOLD at 9.0%)", all(side(o, f"S{i}") == "HOLD" for i in range(5)))
check("below: new buys get their full 9.9% (nothing scaled)", all(abs(w[f"S{i}"] - 9.9) < 0.01 for i in range(5, 10)), w)
check("below: the shortfall (5 x 0.9 points) stays in cash: total 94.5%", abs(sum(w.values()) - 94.5) < 0.02, sum(w.values()))

# A3. mixed: 3 above, 2 below
held = shares_for({"S0": 10.8, "S1": 10.8, "S2": 10.8, "S3": 9.0, "S4": 9.0})
o, w, c, eq = friday(T10, held, P, cash=100000 - 50400)
check("mixed (trim fix): the 2 below-target names HOLD (not topped up)", all(side(o, f"S{i}") == "HOLD" for i in (3, 4)))
check("mixed (trim fix): the 3 above-target names trimmed only part way (to ~10.5%: just enough to end at 99%)",
      all(side(o, f"S{i}") == "SELL" and 10.45 < w[f"S{i}"] < 10.55 for i in range(3)), w)
check("mixed (trim fix): new buys at their full 9.9%", all(abs(w[f"S{i}"] - 9.9) < 0.02 for i in range(5, 10)), w)
check("mixed: total <= 99%, cash >= 0, every stock within 1 point", sum(w.values()) <= 99 + 1e-6 and c >= 0
      and all(abs(w[s] - 9.9) <= 1.0 for s in T10), (sum(w.values()), c))

# A4. the TWLO example: 9.14% held vs a new 8.56% target (0.58 point, inside the band)
px = {**P, "TWLO": 301.27, "S10": 100.0, "S11": 100.0}
held = {"TWLO": round(9140 / 301.27, 2)}
tg = {"TWLO": 0.0856, **{f"S{i}": 0.0905 for i in range(1, 10)}}          # the new buys fit -> TWLO left alone
o, w, c, eq = friday(tg, held, px, cash=100000 - held["TWLO"] * 301.27)
check("TWLO 9.14% vs 8.56%, buys fit: HOLD, not trimmed (band kept)", side(o, "TWLO") == "HOLD" and abs(w["TWLO"] - 9.14) < 0.01,
      w.get("TWLO"))
tg = {"TWLO": 0.0856, **{f"S{i}": 0.0905 for i in range(1, 11)}}          # 10 new buys at 9.05% do not fit next to 9.14%
o, w, c, eq = friday(tg, held, px, cash=100000 - held["TWLO"] * 301.27)
check("TWLO case, buys do not fit (trim fix): TWLO trimmed toward 8.56%, every new buy at its full 9.05%",
      side(o, "TWLO") == "SELL" and 8.56 - 1e-6 <= w["TWLO"] < 9.14 and all(abs(w[f"S{i}"] - 9.05) < 0.02 for i in range(1, 11)), w)
check("TWLO case: total <= the targets' sum (99.06%) + 0.01 point of share rounding, cash >= 0",
      sum(w.values()) <= 99.06 + 0.01 and c >= 0, (sum(w.values()), c))

# A5. worst case: 9 band-held names at +0.99 point and one new buy (before the trim fix the buy got only 1.97%)
held = shares_for({f"S{i}": 10.89 for i in range(9)})
o, w, c, eq = friday(T10, held, P, cash=100000 - 98010)
check("worst case: never over 99%, cash >= 0 (no margin, no insufficient-buying-power reject)",
      sum(w.values()) <= 99 + 1e-6 and c >= 0, (sum(w.values()), c))
check("worst case (trim fix): the new buy gets its full 9.9% (was 1.97%)", abs(w.get("S9", 0) - 9.9) < 0.02, w.get("S9"))
check("worst case (trim fix): the 9 band-held names are trimmed to ~9.9%, never below",
      all(side(o, f"S{i}") == "SELL" and 9.9 - 1e-6 <= w[f"S{i}"] < 9.92 for i in range(9)), w)
check("worst case: SELLs are planned before the BUY", list(o.Side[o.Side.isin(["SELL", "BUY"])]) == ["SELL"] * 9 + ["BUY"])
print(f"   worst case: total {sum(w.values()):.2f}%  new buy S9 {w.get('S9', 0):.2f}% (target 9.9%)  cash {c:.2f}%")

# A5b. no new buy -> nothing is trimmed: band-held names above target stay as they are (the band is unchanged)
held = shares_for({f"S{i}": 10.89 for i in range(9)})
o, w, c, eq = friday({f"S{i}": 0.099 for i in range(9)}, held, P, cash=100000 - 98010)
check("no new buy: the band holds are not trimmed (all HOLD)", all(side(o, f"S{i}") == "HOLD" for i in range(9)), list(o.Side))
# A5c. a new buy that fits next to an above-target band hold -> no trim
held = shares_for({"S0": 10.5})
o, w, c, eq = friday({"S0": 0.099, "S1": 0.099}, held, P, cash=100000 - 10500)
check("buy fits: the above-target band hold is not trimmed", side(o, "S0") == "HOLD" and abs(w["S1"] - 9.9) < 0.02, (side(o, "S0"), w))
# A5d. whole-share mode: the trim is whole shares and still never below target
t = pd.DataFrame([(s, 0.099, 100.0) for s in T10], columns=["Symbol", "Weight", "Price"])
o = pt.build_orders(t, 100000, {f"S{i}": 108.9 for i in range(9)}, P, statuses={s: ("hold" if s != "S9" else "add") for s in T10})
tr = o[o.Side == "SELL"]
check("whole shares: trims are whole shares and leave >= target", len(tr) == 9 and all(float(q).is_integer() for q in tr.Shares)
      and all(ts * 100 >= 9900 - 1e-6 for ts in tr.Target_Shares), tr[["Symbol", "Shares", "Target_Shares"]].to_dict("records"))
check("trim cap == backtest_engine.LIVE_INVESTED (99%)", pt.INVESTED_CAP == __import__("backtest_engine").LIVE_INVESTED)

# A5e. the backtest mirrors it: simulate(band=0.01) with 9 band-held names at ~+0.95 point and one new add
import backtest_engine as be  # noqa: E402
dts = pd.bdate_range("2026-01-05", periods=8)
cols = [f"S{i}" for i in range(10)]
op = pd.DataFrame(100.0, index=dts, columns=cols)
op.loc[dts[3]:, cols[:9]] = 100.0 * 5.085                       # the 9 holdings rise -> each ~10.85% of the book
tw = pd.DataFrame(0.0, index=dts, columns=cols)
tw.loc[:, cols[:9]] = 0.099
tw.loc[dts[4]:, "S9"] = 0.099                                    # decided at the 5th close (a weekly rebalance), filled next open
reb = pd.Series(False, index=dts)
reb[dts[4]] = True
res = be.simulate(op, op, tw, dts[1], rebalance=reb, cost=be.COST, band=0.01)
trims = res["trades"][res["trades"].Kind == "trim"]
r5 = be.simulate(op, op, tw, dts[1], end=dts[5], rebalance=reb, cost=be.COST, band=0.01)
print(f"   simulate trim fix: exposure after the rebalance {r5['exposure'].iloc[-1] * 100:.2f}%, trims {len(trims)}")
check("simulate (trim fix): the 9 band-held names are trimmed on the rebalance day", len(trims) == 9, len(trims))
check("simulate (trim fix): exposure ends at ~99% (the new add got its full weight, not the ~2% cash left)",
      0.985 <= r5["exposure"].iloc[-1] <= 0.991, r5["exposure"].iloc[-1])           # 99% at the open, measured at the close
r_old = be.simulate(op, op, tw.drop(columns="S9").assign(S9=0.0), dts[1], end=dts[5], rebalance=reb, cost=be.COST, band=0.01)
check("simulate: with no new add nothing is trimmed (band unchanged)", (r_old["trades"].Kind != "trim").all())

# A6. regime-off week: weights halved (10 x 4.95%), names held at ~9.9% are trimmed back
held = shares_for({f"S{i}": 9.9 for i in range(10)})
half = {s: 0.0495 for s in T10}
o, w, c, eq = friday(half, held, P, cash=100000 - 99000)
check("regime off: every pick trimmed (SELL) to ~4.95%", all(side(o, s) == "SELL" and abs(w[s] - 4.95) < 0.02 for s in T10), w)
check("regime off: total ~49.5%, rest cash", abs(sum(w.values()) - 49.5) < 0.1 and c > 50, (sum(w.values()), c))

# A7. earnings rule (E5): a new pick with earnings within 5 days is not bought; its cash stays idle (nobody is upsized)
held = shares_for({f"S{i}": 9.9 for i in range(9)})
tg = {**{f"S{i}": 0.099 for i in range(9)}, "S10": 0.099}
o, w, c, eq = friday(tg, held, P, cash=100000 - 89100, blocked={"S10": "earnings Mon Oct 5, in 3 days"})
check("E5: the blocked new pick is SKIPPED", str(side(o, "S10")).startswith("SKIP (earnings"), side(o, "S10"))
check("E5: its 9.9% stays in cash, no other pick is bought up", abs(c - 10.9) < 0.02 and all(abs(w[f"S{i}"] - 9.9) < 0.01 for i in range(9)),
      (c, w))
held = shares_for({"S0": 7.0, "S1": 12.0})
o, w, c, eq = friday({"S0": 0.099, "S1": 0.099}, held, P, cash=100000 - 19000, blocked={"S0": "e", "S1": "e"})
check("E5: an owned pick under target is NOT topped up before earnings", side(o, "S0") == "HOLD", side(o, "S0"))
check("E5: an owned pick over target is still trimmed", side(o, "S1") == "SELL" and abs(w["S1"] - 9.9) < 0.02, w)

# A8. sector cap: the planner trades exactly the target names (the cap lives in the engine's targets); per-sector weights
# end at the target's per-sector weights (here 4 Technology + 6 others, as the cap allows)
sector = {f"S{i}": ("Technology" if i < 4 else "Energy" if i < 7 else "Financials") for i in range(12)}
held = shares_for({"S0": 10.5, "S1": 9.4, "S10": 9.9, "S11": 9.9})        # S10/S11 = not picked (e.g. cap) -> sold
o, w, c, eq = friday(T10, held, P, cash=100000 - 39700)
check("sector cap: names outside the targets are sold completely", "S10" not in w and "S11" not in w, w)
check("sector cap: nothing outside the targets is bought", set(w) <= set(T10), set(w) - set(T10))
by = lambda d: pd.Series(d).groupby(pd.Series(sector)).sum()     # noqa: E731
diff = (by(w) - by({s: 9.9 for s in T10})).abs()
check("sector cap: per-sector weight within 1 point x names of target", (diff <= 1.0 * 4 + 1e-9).all(), diff.to_dict())

# A9. small calculation checks
o, w, c, eq = friday({"X": 0.0999}, {}, {"X": 139.66}, cash=100000)
check("rounding: a buy is rounded DOWN to 2-decimal shares, never above its weight", w["X"] <= 9.99 and 9.99 - w["X"] < 0.01, w)
check("99% applied once: the plan buys weight x equity (no second 0.99)", abs(o.at[0, "Target_Value"] - 9990) < 0.01, o.at[0, "Target_Value"])
o, w, c, eq = friday({"S1": 0.099}, {"S0": 99.0}, P, cash=0.0)
check("sells fund buys: zero cash, the planned sale pays for the buy", side(o, "S1") == "BUY" and c >= 0, (side(o, "S1"), c))
check("orders list SELLs before BUYs", list(o.Side[o.Side.isin(["SELL", "BUY"])]) == ["SELL", "BUY"])
o = pt.build_orders(pd.DataFrame({"Symbol": ["S0"], "Weight": [0.1], "Price": [100.0]}), 100000, {"S0": 10.0}, P,
                    statuses={"S0": "hold"}, fractional=False)
check("whole shares (fractional=False): a buy is a whole number of shares", float(o.at[0, "Shares"]).is_integer(), o.at[0, "Shares"])
orders = pd.DataFrame({"Symbol": ["A", "B"], "Side": ["HOLD", "BUY"], "Shares": [0, 5], "Price": [10.0, 20.0],
                       "Current_Shares": [7.0, 0.0], "Target_Shares": [None, 5.0]})
check("dry-run summary counts a HOLD row without Target_Shares at its held shares", pt.invested_after(orders) == 170.0,
      pt.invested_after(orders))

# ================================================================= B. fake-broker runs: Fri 10/2 rebalance, Mon 10/5 check
class FakeOrder:
    def __init__(self, **kw):
        self.__dict__.update(kw)

class FakeBroker:
    """Fills every order at once at its limit price; updates cash and positions, so the re-read buying power is real."""

    def __init__(self, positions, cash):
        self.pos, self.cash, self.submitted, self.orders = dict(positions), cash, [], {}

    def get_account(self):
        return types.SimpleNamespace(equity=self.cash + sum(q * PX[s] for s, q in self.pos.items()), cash=self.cash,
                                     buying_power=self.cash)

    def get_all_positions(self):
        return [types.SimpleNamespace(symbol=s, qty=q) for s, q in self.pos.items() if q > 0]

    def submit_order(self, req):
        q, s = float(req.qty), req.symbol
        sign = 1 if req.side == OrderSide.BUY else -1
        px = float(getattr(req, "limit_price", None) or PX[s])
        self.pos[s] = self.pos.get(s, 0) + sign * q
        self.cash -= sign * q * px
        o = FakeOrder(id=f"o{len(self.submitted) + 1}", status="filled", filled_qty=q, symbol=s,
                      client_order_id=getattr(req, "client_order_id", ""))
        self.submitted.append(req)
        self.orders[o.id] = o
        return o

    def get_order_by_id(self, oid):
        return self.orders[oid]

    def get_orders(self, *a, **k):
        return []

    def cancel_order_by_id(self, oid):
        pass

# Fake portfolio (no live Reports files): 7 kept picks (most inside the 1-point band, some above target -> the trim fix;
# SLB 1.5 points under -> bought up), 3 new picks, 3 dropped holdings. Old weights and targets both sum to 99%.
#        symbol  sector         close  old weight  target
BOOK = [("FTNT", "Technology", 180.95, 0.125, 0.120), ("SNOW", "Technology", 341.04, 0.115, 0.110),
        ("NVDA", "Technology", 187.62, 0.100, 0.105), ("COF", "Financials", 210.40, 0.108, 0.100),
        ("SLB", "Energy", 34.25, 0.085, 0.100), ("LMT", "Industrials", 498.10, 0.100, 0.095),
        ("MRK", "Health Care", 85.30, 0.090, 0.095), ("NFLX", "Communication Services", 1189.50, 0.0, 0.090),
        ("AMZN", "Consumer Discretionary", 222.40, 0.0, 0.090), ("GOOGL", "Communication Services", 245.10, 0.0, 0.085),
        ("AAPL", "Technology", 255.45, 0.090, 0.0), ("MSFT", "Technology", 517.90, 0.090, 0.0),
        ("META", "Communication Services", 727.05, 0.087, 0.0)]
picks = pd.DataFrame([{"Strategy": "test", "Regime_On": 1, "Symbol": s, "Sector": sec, "Held": int(w > 0), "Strategy_Weight": w,
                       "Provisional_Weight": t, "Strategy_Score": 50.0 - i, "Strategy_Rank": float(i + 1), "Technical_Score": 50.0,
                       "RS_Score": 50.0, "Close": c, "Next_Earnings": "2026-11-20"} for i, (s, sec, c, w, t) in enumerate(BOOK)])
PX = dict(zip(picks.Symbol, picks.Close))
import fake_quotes  # noqa: E402
fake_quotes.install(pt, prices=PX)   # each stock quoted with the ask at its price (buy limit = ask + 0.05%)
tmp = tempfile.mkdtemp()
SIGNAL_CSV = os.path.join(tmp, "signal_analysis.csv")     # prices for held non-picks
EARNINGS_CSV = os.path.join(tmp, "earnings_date.csv")     # no earnings within 5 days
# 10/1 closes + the Mon 10/5 ranking (a Mon/Wed check day: Midweek_Check 1) for the spare-cash rule
pd.concat([picks.assign(Date="2026-10-01")[["Date", "Symbol", "Close"]],
           picks.assign(Date="2026-10-05", Regime_On=1, Midweek_Check=1)[
               ["Date", "Symbol", "Close", "Strategy_Score", "Strategy_Rank", "Provisional_Weight", "Regime_On", "Midweek_Check"]]]
          ).to_csv(SIGNAL_CSV, index=False)
pd.DataFrame({"Symbol": picks.Symbol, "Earnings Date": "2026-11-20", "Time": "PM"}).to_csv(EARNINGS_CSV, index=False)

def run_day(day, last_reb, positions, cash, changes_rows=None, midweek_rows=None):
    """auto_trade() on a fake broker with the pipeline files as they would be after `day`'s 2:30 PM run."""
    pk = picks.copy()
    pk["As_Of"], pk["Last_Rebalance"], pk["Last_Decision"] = day, last_reb, day
    paths = {k: os.path.join(tmp, f"{k}_{day}.csv") for k in ("picks", "changes", "midweek")}
    pk.to_csv(paths["picks"], index=False)
    pd.DataFrame(changes_rows or [], columns=["Date", "Symbol", "Status", "Reason"]).to_csv(paths["changes"], index=False)
    pd.DataFrame(midweek_rows or [], columns=["As_Of", "Event", "Event_Date", "Action", "Sell", "Buy", "Message", "Sell_Rank",
                                               "Buy_Rank", "Weight_%"]).to_csv(paths["midweek"], index=False)
    broker = FakeBroker(positions, cash)
    plan0, status0, prices0, earn0 = pt.plan_orders, pt.latest_signal_status, pt.latest_prices, pt.earnings_blocked
    saved = {k: getattr(pt, k) for k in ("check_signal_freshness", "get_live_positions_and_equity", "_past_evening_cutoff",
                                         "paper_trading_client", "_todays_recorded_orders", "PENDING_ORDERS_JSON",
                                         "plan_orders", "latest_signal_status", "SELL_SETTLE_POLL_SECS", "latest_prices",
                                         "earnings_blocked")}
    acct = broker.get_account()
    try:
        pt.check_signal_freshness = lambda **k: day
        pt.get_live_positions_and_equity = lambda: (dict(broker.pos), float(acct.equity), broker.cash, broker.cash)
        pt._past_evening_cutoff = lambda: False
        pt.paper_trading_client = lambda: broker
        pt._todays_recorded_orders = lambda *a, **k: set()
        pt.PENDING_ORDERS_JSON = os.path.join(tmp, f"pending_{day}.json")
        pt.SELL_SETTLE_POLL_SECS = 0
        pt.plan_orders = lambda *a, **k: plan0(*a, picks_csv=paths["picks"], midweek_csv=paths["midweek"],
                                               signal_csv=SIGNAL_CSV, **k)
        pt.latest_signal_status = lambda changes_csv=None, as_of=None: status0(paths["changes"], as_of)
        pt.latest_prices = lambda symbols, signal_csv=None: prices0(symbols, SIGNAL_CSV)
        pt.earnings_blocked = lambda symbols, as_of, earnings_csv=None: earn0(symbols, as_of, EARNINGS_CSV)
        orders, meta, results = pt.auto_trade(log_csv=None)
    finally:
        for k, v in saved.items():
            setattr(pt, k, v)
    return orders, meta, results, broker, float(acct.equity)

EQ = 100000.0
pos = {r.Symbol: int(r.Strategy_Weight * EQ / r.Close) for r in picks.itertuples() if r.Held}   # whole shares at the old weights
cash0 = EQ - sum(q * PX[s] for s, q in pos.items())

# Fri 10/2: the full rebalance (decision = the Thursday-close plan; Friday's real ranks can differ)
rows = [{"Date": "2026-10-02", "Symbol": r.Symbol, "Status": ("hold" if r.Held else "add") if r.Provisional_Weight > 0 else "drop",
         "Reason": "test"} for r in picks.itertuples() if r.Provisional_Weight > 0 or r.Held]
orders, meta, results, broker, eq = run_day("2026-10-02", "2026-10-02", pos, cash0, rows)
sides = [("SELL" if r.side == OrderSide.SELL else "BUY") for r in broker.submitted]
check("Fri 10/2: source = provisional (full rebalance)", meta["source"] == "provisional", meta["source"])
check("Fri 10/2: every SELL is sent before any BUY", sides == sorted(sides, key=lambda x: x != "SELL"), sides)
want_out = set(picks.loc[(picks.Held == 1) & (picks.Provisional_Weight == 0), "Symbol"])
want_in = set(picks.loc[(picks.Held == 0) & (picks.Provisional_Weight > 0), "Symbol"])
sold = {r.symbol for r in broker.submitted if r.side == OrderSide.SELL}
bought = {r.symbol for r in broker.submitted if r.side == OrderSide.BUY}
check("Fri 10/2: sells exactly the dropped names", want_out <= sold, (want_out, sold))
check("Fri 10/2: buys the new names", want_in <= bought, (want_in, bought))
check("Fri 10/2: dropped names fully sold", all(broker.pos.get(s, 0) == 0 for s in want_out), {s: broker.pos.get(s) for s in want_out})
check("Fri 10/2: cash never negative after the evening orders", broker.cash >= 0, broker.cash)
inv = sum(q * PX[s] for s, q in broker.pos.items()) / eq * 100
names = sorted(s for s, q in broker.pos.items() if q > 0)
end_pct = {s: broker.pos[s] * PX[s] / eq * 100 for s in names}
tgt_pct = dict(zip(picks.Symbol, picks.Provisional_Weight * 100))
# evening orders are whole shares; the 9 AM check buys the fractional rest -> each name ends within 1 point + one share
gap = {s: round(end_pct[s] - tgt_pct.get(s, 0), 2) for s in names}
check("Fri 10/2: holds exactly the 10 target names after the evening", len(names) == 10 and set(names) == set(picks.Symbol[picks.Provisional_Weight > 0]),
      names)
check("Fri 10/2: every name within 1 point (+ the fractional rest) of its target", all(abs(g) <= 1.0 + PX[s] / eq * 100 for s, g in gap.items()), gap)
check("Fri 10/2: total invested <= 99% + band excess, never > 100%", inv <= 100.0, inv)
check("Fri 10/2 (trim fix): total invested <= 99%", inv <= 99.0 + 1e-6, inv)
under = {s: round(tgt_pct[s] - end_pct[s], 2) for s in want_in}
check("Fri 10/2 (trim fix): each new buy within one share of its full target after the evening",
      all(g <= PX[s] / eq * 100 + 0.02 for s, g in under.items()), under)
pend = json.load(open(os.path.join(tmp, "pending_2026-10-02.json")))
print(f"   Fri 10/2: {len(sold)} sells, {len(bought)} buys, invested {inv:.2f}% after the evening, cash {broker.cash / eq * 100:.2f}%; "
      f"{len(pend.get('orders', []))} rows recorded for the 9 AM fractional completion; per-name gap vs target {gap}")

# Fri 10/2 with a held stock that was REMOVED from sector_mapping.py (here MSTR): it has no rows in signal_analysis.csv, is
# priced from market data (current_prices, patched here - no network) and sold completely like any non-pick
PX["MSTR"] = 150.0
cp0 = pt.current_prices
pt.current_prices = lambda syms: {s: PX[s] for s in syms if s in PX}
try:
    import sector_mapping as _sm
    assert "MSTR" not in _sm.tradable_symbols
    _, _, _, broker_rm, _ = run_day("2026-10-02", "2026-10-02", {**pos, "MSTR": 7.0}, cash0 - 7 * 150.0, rows)
finally:
    pt.current_prices = cp0
check("Fri 10/2: a held stock removed from the list (MSTR) is sold completely at the rebalance",
      broker_rm.pos.get("MSTR", 0) == 0 and any(r.symbol == "MSTR" and r.side == OrderSide.SELL for r in broker_rm.submitted),
      broker_rm.pos.get("MSTR"))

# Mon 10/5: a quiet mid-week check -> nothing is traded
pos_mon = dict(broker.pos)
orders, meta, results, broker2, _ = run_day("2026-10-05", "2026-10-02", pos_mon, broker.cash)
check("Mon 10/5 quiet check: source = hold, no order sent", meta["source"] == "hold" and not broker2.submitted, (meta["source"], len(broker2.submitted)))

# Mon 10/5: a cash exit (no top-10 refill) -> the exited name is sold, then the Mon/Wed spare-cash rule (2026-10-06; leftover
# rule 2026-10-07): no top-10 stock is left unheld (the exited AMZN is not bought back), so the cash tops up rank 1 (FTNT)
# to the 19.8% cap, then rank 2 (SNOW); AAPL (rank 11) is not bought; the other held names get no order
victim = names[0]
exit_row = [{"As_Of": "2026-10-05", "Event": "mid-week check", "Event_Date": "2026-10-05", "Action": "SELL", "Sell": victim,
             "Buy": None, "Message": f"exit {victim}", "Sell_Rank": 35, "Buy_Rank": None, "Weight_%": round(end_pct[victim], 2)}]
orders, meta, results, broker3, eq3 = run_day("2026-10-05", "2026-10-02", pos_mon, broker.cash, midweek_rows=exit_row)
sent = [(r.symbol, "SELL" if r.side == OrderSide.SELL else "BUY") for r in broker3.submitted]
ftnt = broker3.pos.get("FTNT", 0) * PX["FTNT"] / eq3 * 100
check("Mon 10/5 exit: the exited name is sold first, then only ranks 1-2 (FTNT, SNOW) are topped up; AAPL (rank 11) not "
      "bought; FTNT ends at the 19.8% cap", sent and sent[0] == (victim, "SELL") and {x for x in sent[1:]} == {("FTNT", "BUY"),
      ("SNOW", "BUY")} and meta["source"] == "midweek+cash" and 19.0 <= ftnt <= 19.8 + 1e-6, (sent, meta["source"], ftnt))
inv3 = sum(q * PX[s] for s, q in broker3.pos.items()) / eq3 * 100
check("Mon 10/5 exit: the account ends ~99% invested (<= 99%, >= 97%; the fractional rest goes at 9 AM)",
      97.0 <= inv3 <= 99.0 + 1e-6 and broker3.cash >= 0, (inv3, broker3.cash))

# do-not-sell stock outside the targets: its planned sale is skipped BEFORE the trim fix, so no cash is counted from it
tg = pd.DataFrame({"Symbol": ["P1", "P2", "P3", "P4", "P5"], "Weight": [0.178] * 5, "Price": 100.0})
pos = {"SNOW": 100, **{f"P{i}": 185 for i in range(1, 5)}}                      # SNOW $10k kept, P1-P4 $18.5k each, $16k cash
o = pt.build_orders(tg, 100_000, pos, {"SNOW": 100.0}, fractional=True, statuses={f"P{i}": "hold" for i in range(1, 6)},
                    keep={"SNOW": "on your do-not-sell list"}).set_index("Symbol")
check("do-not-sell SNOW is a SKIP row, and P1-P4 are trimmed so the P5 buy fits the real cash",
      o.Side["SNOW"] == "SKIP (on your do-not-sell list)" and all(o.Side[f"P{i}"] == "SELL" for i in range(1, 5))
      and o.loc[o.Side == "BUY", "Est_Value"].sum() * 1.01 <= 16_000 + o.loc[o.Side == "SELL", "Est_Value"].sum() + 1,
      o[["Side", "Shares", "Est_Value"]].to_dict("index"))

print("\nBAND-HOLD AUDIT OK" if not FAIL else f"\nBAND-HOLD AUDIT FAILURES ({len(FAIL)}): {FAIL}")
sys.exit(1 if FAIL else 0)
