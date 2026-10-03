"""Forward testing from backtest_engine.FORWARD_START (2026-10-02). 1) The dashboard's per-stock forward test (forward_test): only sessions on/after FORWARD_START count (a trade
before it is ignored, a position held into it starts at the start close), trades fill at the decision-day close with COST
per side, and the empty state is None (the panel then says no closed trades yet). 2) forward_test.py (the live account):
picks start from a flat position on/after the start, trading cost vs the decision price, deposits are not returns, weekly
median / drawdown, and record() saves one row per trading day (idle on holidays) through a FAKE account - no requests,
temporary files only."""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import backtest_engine as be
import forward_test as ft

FAIL = []


def check(what, ok, got=""):
    print(("ok   " if ok else "FAIL ") + what + ("" if ok else f"  got {got!r}"))
    if not ok:
        FAIL.append(what)


check("one start-date constant: 2026-10-02", be.FORWARD_START == "2026-10-02", be.FORWARD_START)
d = pd.bdate_range("2026-09-21", "2026-10-16")
close = pd.Series(range(100, 100 + len(d)), index=d, dtype=float)
w = pd.Series(0.0, index=d)
w["2026-09-22":"2026-09-25"] = 0.1          # a whole trade before the start: must not count
w["2026-09-30":"2026-10-06"] = 0.1          # held into the start: counts from the Oct 2 close only
w["2026-10-12":] = 0.1                      # still open at the end
f = be.forward_test(close, w)
c = (1 - be.COST) / (1 + be.COST)
check("cutoff: only the trade from the start counts", f["Closed trades"] == 1, f["Closed trades"])
check("cutoff: entry at the start close, exit at the decision-day close",
      abs(f["Median trade %"] - (close["2026-10-07"] * c / close["2026-10-02"] - 1) * 100) < 1e-9, f["Median trade %"])
check("median hold in sessions (Oct 2 -> Oct 7)", f["Median hold (sessions)"] == 3, f["Median hold (sessions)"])
check("win rate", f["Win rate %"] == 100, f["Win rate %"])
check("held % of sessions counts only sessions from the start", abs(f["Held % of sessions"] - 100 * 8 / 11) < 1e-9,
      f["Held % of sessions"])
check("buy & hold from the start close", abs(f["Buy & hold %"] - (close.iloc[-1] / close["2026-10-02"] - 1) * 100) < 1e-9,
      f["Buy & hold %"])
ot = f["Open trade"]
check("open trade: entry date, price and % change", ot and ot["Entry"] == pd.Timestamp("2026-10-12")
      and ot["Price"] == close["2026-10-12"] and abs(ot["Change %"] - (close.iloc[-1] / close["2026-10-12"] - 1) * 100) < 1e-9, ot)

e = be.forward_test(close[:"2026-10-02"], w[:"2026-10-02"] * 0)        # start day only, nothing held
check("empty state: no trades, no open trade, stats None",
      e["Closed trades"] == 0 and e["Open trade"] is None and e["Win rate %"] is None and e["Median trade %"] is None
      and e["Median hold (sessions)"] is None and e["Held % of sessions"] == 0 and e["Buy & hold %"] == 0, e)
n = be.forward_test(close[:"2026-10-01"], w[:"2026-10-01"])           # no bar on/after the start yet
check("no bars after the start: everything None / 0",
      n["Sessions"] == 0 and n["Closed trades"] == 0 and n["Held % of sessions"] is None and n["Buy & hold %"] is None, n)

# ---------------------------------------------------------------- 2) forward_test.py (account level)
def fill(t, sym, side, qty, px):
    return {"transaction_time": t, "symbol": sym, "side": side, "qty": str(qty), "price": str(px), "id": f"{sym}{t}{side}"}


FILLS = [fill("2026-09-03T19:00:00Z", "OLD", "buy", 10, 50),          # held before the start: never a pick
         fill("2026-10-02T19:31:00Z", "OLD", "sell", 10, 55),
         fill("2026-10-02T19:31:00Z", "AAA", "buy", 10, 101),         # pick 1: win
         fill("2026-10-05T19:31:00Z", "AAA", "sell", 4, 110),
         fill("2026-10-07T19:31:00Z", "AAA", "sell", 6, 108),
         fill("2026-10-02T19:31:00Z", "BBB", "buy", 5, 200),          # pick 2: loss
         fill("2026-10-09T19:31:00Z", "BBB", "sell", 5, 190),
         fill("2026-10-09T19:32:00Z", "CCC", "buy", 2, 30)]           # still open
pk = ft.picks(FILLS)
check("picks: only round trips opened on/after the start, closed when flat", list(pk["Symbol"]) == ["AAA", "BBB"], pk)
check("picks: P/L and win/loss", abs(pk["P/L $"].iloc[0] - (4 * 110 + 6 * 108 - 1010)) < 1e-9 and pk["P/L $"].iloc[1] == -50, pk)
orders = pd.DataFrame({"Symbol": ["AAA", "OLD", "BBB", "AAA"], "Side": ["BUY", "SELL", "BUY", "SELL"],
                       "Submitted_At_CT": ["2026-10-02 14:31:00", "2026-10-02 14:31:00", "2026-10-02 14:31:00", "2026-10-05 14:31:00"],
                       "As_Of": ["2026-10-02", "2026-10-02", "2026-10-02", "2026-10-05"]})
closes = {(pd.Timestamp("2026-10-02"), "AAA"): 100.0, (pd.Timestamp("2026-10-02"), "OLD"): 56.0,
          (pd.Timestamp("2026-10-02"), "BBB"): 200.0, (pd.Timestamp("2026-10-05"), "AAA"): 111.0}
traded, cost = ft.trade_cost(FILLS, orders, closes)
# AAA buy 10 @101 vs 100 = +10; OLD sell 10 @55 vs 56 = +10; BBB buy @200 vs 200 = 0; AAA sell 4 @110 vs 111 = +4; the
# Oct 7 AAA sell maps to the Oct 5 order (latest sent before it): 6 @108 vs 111 = +18; BBB sell / CCC buy have no order row
check("trade cost vs the decision price (+ = cost), fills without an order row left out",
      abs(cost - 42) < 1e-9 and abs(traded - (1010 + 550 + 1000 + 440 + 648)) < 1e-9, (traded, cost))
check("trade cost: nothing before the start", ft.trade_cost([FILLS[0]], orders, closes) == (0.0, 0.0))

days = pd.bdate_range("2026-10-02", "2026-10-16")
eq = pd.Series(100000.0, index=days)
eq["2026-10-06":] += 5000                      # a $5,000 deposit on Oct 6 ...
eq["2026-10-09"] += 3000                       # ... and a real gain that partly reverses
eq["2026-10-16"] += 6000
daily = pd.DataFrame({"Date": days, "Equity": eq.values, "Net_Deposits": [0.0] * 2 + [5000.0] * (len(days) - 2)})
bench = pd.DataFrame({"Date": days, "QQQ": [100.0] * 5 + [110.0] * 5 + [99.0] * 1, "SPY": 100.0})
sm = ft.summary(daily, bench).set_index("Series")
st_ = sm.loc["Strategy (live account)"]
check("summary: deposits are not returns (total +6%)", abs(st_["Total return %"] - 6) < 1e-9, st_.to_dict())
check("summary: median weekly return Friday to Friday (+3%, +2.91%)",
      abs(st_["Median weekly return %"] - (3 + (106 / 103 - 1) * 100) / 2) < 1e-9, st_["Median weekly return %"])
check("summary: max drawdown (103k -> 100k on Oct 12)", abs(st_["Max drawdown %"] - (100 / 103 - 1) * 100) < 1e-9,
      st_["Max drawdown %"])
q = sm.loc["QQQ (comparison, not traded)"]
check("summary: QQQ from the start close", abs(q["Total return %"] + 1) < 1e-9 and abs(q["Max drawdown %"] - (99 / 110 - 1) * 100) < 1e-9,
      q.to_dict())
check("summary: nothing saved yet -> only the comparison rows", list(ft.summary(daily.iloc[:0], bench)["Series"].str.split().str[0]) == ["QQQ", "SPY"])


class FakeAccount:
    calls = 0

    def account_summary(self):
        FakeAccount.calls += 1
        return {"Equity": 58000.0 + FakeAccount.calls, "Cash": 1200.0}

    def fills(self, after=None):
        return FILLS

    def net_deposits(self):
        return 57000.0

    def position_dicts(self):
        return [{"symbol": "CCC"}]


import tempfile  # noqa: E402
from datetime import datetime  # noqa: E402
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "daily.csv")
    at = lambda s: datetime.fromisoformat(s).replace(tzinfo=ft.CT)
    check("record: idle on a holiday / weekend / before the start",
          ft.record(FakeAccount(), at("2026-10-03 16:15"), path) is None and ft.record(FakeAccount(), at("2026-11-26 16:15"), path) is None
          and ft.record(FakeAccount(), at("2026-10-01 16:15"), path) is None and not os.path.exists(path))
    r1 = ft.record(FakeAccount(), at("2026-10-05 16:15"), path)
    ft.record(FakeAccount(), at("2026-10-05 16:40"), path)         # same day again: replaces the row
    ft.record(FakeAccount(), at("2026-10-06 16:15"), path)
    saved = pd.read_csv(path)
    check("record: one row per trading day, a re-run replaces it", list(saved["Date"]) == ["2026-10-05", "2026-10-06"]
          and saved["Time_CT"].iloc[0] == "16:40" and list(saved.columns) == ft.DAILY_COLS, saved)
    check("record: picks, winners and positions saved", r1["Closed_Picks"] == 2 and r1["Winning_Picks"] == 1 and r1["Positions"] == 1, r1)
# shadow (no orders): the live rules minus T20 / cap_soft and E5, passed as overrides only; closes, 0.1% per side
sd = pd.bdate_range("2026-09-28", "2026-10-09")
sig = pd.DataFrame([{"Date": d, "Symbol": sym, "Close": px, "RS_Score": 0.0, "Strategy_Score": 1.0, "Regime_On": 1}
                    for i, d in enumerate(sd) for sym, px in (("AAA", 100.0 + (i >= 5) * 10), ("BBB", 50.0))])
seen_kw, w0 = {}, dict(be.WINNER)
real_wt = ft.be.winner_targets


def fake_targets(score, *a, **k):
    seen_kw.update(k)
    t = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    t.loc[:"2026-10-05", "AAA"] = 1.0                          # all in AAA from the start (99% live) ...
    t.loc["2026-10-06":, "BBB"] = 1.0                          # ... switched to BBB at the Oct 6 close
    return t, None


try:
    ft.be.winner_targets = fake_targets
    v = ft.shadow_values(sig)
finally:
    ft.be.winner_targets = real_wt
check("shadow: only the overrides (no rank-20 limit, strict sector cap, no earnings skip), WINNER untouched",
      seen_kw.get("selection") == {"max_pick_rank": None, "cap_soft": False} and seen_kw.get("earnings_block_days") is None
      and be.WINNER == w0, seen_kw)
# Oct 2 = 1.0 after its buys; AAA +10% on Oct 5 (99% invested); Oct 6 switch to BBB pays 0.1% on both legs; then flat
a5 = 0.01 + 0.99 * 1.1
after = a5 - be.COST * (0.99 * 1.1 + 0.99 * a5)      # sell all AAA + buy 99% BBB
check("shadow: starts at 1.0 on the start close, follows the closes, pays 0.1% per side on a change",
      v.index[0] == pd.Timestamp("2026-10-02") and v.iloc[0] == 1.0 and abs(v["2026-10-05"] - a5) < 1e-9
      and abs(v["2026-10-06"] - after) < 1e-9 and abs(v.iloc[-1] - after) < 1e-9, v.round(6).to_dict())
dsh = daily.assign(Shadow_Value=[1.0] * len(daily))
check("summary: the shadow row sits right after the live strategy",
      list(ft.summary(dsh, bench)["Series"])[:2] == ["Strategy (live account)", ft.SHADOW])
src = open(os.path.join(ROOT, "forward_test.py"), encoding="utf-8").read()
check("forward_test.py has no order code (read-only)",
      not any(w in src for w in ("submit_order", "cancel_order", "OrderRequest", "TradingClient", "import paper_trade", "requests.post")))
print(f"\n{len(FAIL)} failed" if FAIL else "\nFORWARD TEST OK")
sys.exit(1 if FAIL else 0)
