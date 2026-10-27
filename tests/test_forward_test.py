"""Forward testing from backtest_engine.FORWARD_START (2026-10-02). 1) The dashboard's per-stock forward test (forward_test): only sessions on/after FORWARD_START count (a trade
before it is ignored, a position held into it starts at the start close), trades fill at the decision-day close with COST
per side, and the empty state is None (the panel then says no closed trades yet). 2) forward_test.py (the live account):
picks start from a flat position on/after the start, trading cost vs the decision price, deposits are not returns, weekly
columns / drawdown, and record() saves one row per trading day (idle on holidays) through a FAKE account - no requests,
temporary files only. 3) The paper strategies on fake bars / benchmarks / earnings (never a download)."""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import db  # noqa: E402  tables live in SQLite (db.py), not CSV files
import c6_rules  # noqa: E402,F401  the Oct 15 weekly rules this test checks (live: monthly low volatility)
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
sm = ft.leaderboard(daily, bench, values=pd.DataFrame(columns=["Date", "Strategy", "Value"])).set_index("Strategy")
st_ = sm.loc[ft.ACCOUNT]
# time-weighted: the deposit day is flat (105k on 100k + 5k), then 105k -> 108k -> 105k -> 111k
check("leaderboard: deposits are not returns, time-weighted (total 111/105 = +5.71%)",
      abs(st_["Total return %"] - (111 / 105 - 1) * 100) < 1e-9, st_.to_dict())
check("leaderboard: one column per week Friday to Friday after the daily return (Week 1 % = Oct 2-9 = 108/105, Week 2 % = "
      "Oct 9-16 = 111/108), every row; no median column",
      [c for c in sm.columns if c.startswith("Week ")] == [f"Week {k} %" for k in range(1, 13)]
      and list(sm.columns).index("Week 1 %") == list(sm.columns).index("Daily return %") + 1
      and sm.filter(like="Week ").iloc[:, 2:].isna().all().all() and not any(c in sm for c in ("Weeks", "Median weekly return %"))
      and abs(st_["Week 1 %"] - (108 / 105 - 1) * 100) < 1e-9 and abs(st_["Week 2 %"] - (111 / 108 - 1) * 100) < 1e-9
      and abs(sm.at["QQQ (comparison)", "Week 1 %"] - 10) < 1e-9
      and sm.loc[[ft.ACCOUNT, "QQQ (comparison)", "SPY (comparison)"], ["Week 1 %", "Week 2 %"]].notna().all().all(),
      sm.filter(like="Week ").to_dict())
_wk = ft.leaderboard(daily.iloc[:5], bench.iloc[:5], pd.DataFrame(columns=["Date", "Strategy", "Value"])).set_index("Strategy")
check("leaderboard: the unfinished week is its own 'so far' column (Fri Oct 2 close -> Thu Oct 8), not a finished week",
      [c for c in _wk.columns if c.startswith("Week ")][:2] == ["Week 1 (so far) %", "Week 2 %"]
      and abs(_wk.at[ft.ACCOUNT, "Week 1 (so far) %"] - 0.0) < 1e-9
      and "Week 1 %" in ft.leaderboard(daily.iloc[:1], bench.iloc[:1], pd.DataFrame(columns=["Date", "Strategy", "Value"])).columns,
      _wk.filter(like="Week").to_dict())
check("leaderboard: max drawdown (108k -> 105k on Oct 12)", abs(st_["Max drawdown %"] - (105 / 108 - 1) * 100) < 1e-9,
      st_["Max drawdown %"])
q = sm.loc["QQQ (comparison)"]
check("leaderboard: QQQ from the start close", abs(q["Total return %"] + 1) < 1e-9 and abs(q["Max drawdown %"] - (99 / 110 - 1) * 100) < 1e-9,
      q.to_dict())
check("leaderboard: the account and QQQ / SPY are not ranked", sm.loc[[ft.ACCOUNT, "QQQ (comparison)", "SPY (comparison)"], "Rank"].isna().all())
vals = pd.DataFrame([{"Date": d, "Strategy": c["name"], "Value": 1.0} for d in days for c in ft.STRATEGIES])
vals.loc[(vals["Strategy"] == "Quant Score top 5") & (vals["Date"] >= pd.Timestamp("2026-10-09")), "Value"] = 1.02   # +2% then flat
vals.loc[(vals["Strategy"] == "Quant Score top 10") & (vals["Date"] >= pd.Timestamp("2026-10-09")), "Value"] = 1.02
vals.loc[(vals["Strategy"] == "Quant Score top 10") & (vals["Date"] == pd.Timestamp("2026-10-05")), "Value"] = 0.95  # same total, deeper drop
lb = ft.leaderboard(daily, bench, vals)
N_RANKED = sum(bool(c.get("ranked", True)) for c in ft.STRATEGIES)   # the Oct 16 live halves are shown, not ranked
check("leaderboard: ranked by total return since Oct 2, a tie goes to the smaller drawdown; Rank counts strategies only",
      list(lb.dropna(subset=["Rank"])["Strategy"][:2]) == ["Quant Score top 5", "Quant Score top 10"] and list(lb.dropna(subset=["Rank"])["Rank"][:2]) == [1, 2]
      and sorted(lb["Rank"].dropna()) == list(range(1, N_RANKED + 1)), lb.head(4).to_dict("records"))
check("leaderboard: the live row is marked", (lb["Strategy"] == ft.MARKED + ft.LIVE_MARK).sum() == 1, list(lb["Strategy"]))
_early = ft.leaderboard(daily.iloc[:3], bench.iloc[:3], vals[vals["Date"] <= "2026-10-06"])
check("leaderboard: ranked from day one (no wait for a full week)",
      sorted(_early["Rank"].dropna()) == list(range(1, N_RANKED + 1)) and _early.dropna(subset=["Rank"])["Strategy"].iloc[-1] == "Quant Score top 10",
      _early.head(3).to_dict("records"))
check("verdict: too early before 12 weeks", ft.verdict(lb) == "Week 2 of 12: too early to name a winner.", ft.verdict(lb))
long_days = pd.bdate_range("2026-10-02", periods=70)
lv = pd.DataFrame([{"Date": d, "Strategy": c["name"], "Value": 1.0 + i * (0.002 if c["name"] == "Quant Score top 5" else 0.0)}
                   for i, d in enumerate(long_days) for c in ft.STRATEGIES])
lbq = lambda q_end: ft.leaderboard(daily.iloc[:0], pd.DataFrame({"Date": long_days, "QQQ": 100.0, "SPY": 100.0}).assign(
    QQQ=lambda x: [100.0] * 69 + [q_end]), lv)
check("verdict: after 12 weeks the top strategy wins only if it beats live and QQQ",
      "the winner is Quant Score top 5" in ft.verdict(lbq(100.0)) and "no winner" in ft.verdict(lbq(200.0)),
      (ft.verdict(lbq(100.0)), ft.verdict(lbq(200.0))))

class FakeAccount:
    calls = 0

    def account_summary(self):
        FakeAccount.calls += 1
        return {"Equity": 58000.0 + FakeAccount.calls, "Cash": 1200.0}

    def fills(self, after=None):
        return FILLS

    def cash_flows(self):
        return [{"date": "2026-09-30", "time": "2026-09-30T21:15:00Z", "amount": 57000.0}]

    def position_dicts(self):
        return [{"symbol": "CCC", "qty": "5"}]

import tempfile  # noqa: E402
from datetime import datetime  # noqa: E402
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "daily.csv")
    at = lambda s: datetime.fromisoformat(s).replace(tzinfo=ft.CT)
    check("record: idle on a holiday / weekend / before the start",
          ft.record(FakeAccount(), at("2026-10-03 16:15"), path) is None and ft.record(FakeAccount(), at("2026-11-26 16:15"), path) is None
          and ft.record(FakeAccount(), at("2026-10-01 16:15"), path) is None and not db.exists(path))
    r1 = ft.record(FakeAccount(), at("2026-10-05 16:15"), path)
    ft.record(FakeAccount(), at("2026-10-05 16:40"), path)         # same day again: replaces the row
    ft.record(FakeAccount(), at("2026-10-06 16:15"), path)
    saved = db.read_csv(path)
    check("record: one row per trading day, a re-run replaces it", list(saved["Date"]) == ["2026-10-05", "2026-10-06"]
          and saved["Time_CT"].iloc[0] == "16:40" and list(saved.columns) == ft.DAILY_COLS, saved)
    check("record: picks, winners and positions saved", r1["Closed_Picks"] == 2 and r1["Winning_Picks"] == 1 and r1["Positions"] == 1, r1)

    class LateDeposit(FakeAccount):            # a $3,000 deposit booked Oct 6, 4:15 PM CT, already in the balance
        def account_summary(self):
            return {"Equity": 61000.0, "Cash": 4200.0}

        def cash_flows(self):
            return super().cash_flows() + [{"date": "2026-10-06", "time": "2026-10-06T21:15:22Z", "amount": 3000.0},
                                           {"date": "2099-01-02", "time": "2099-01-02T21:15:00Z", "amount": 9.0}]
    r2 = ft.record(LateDeposit(), at("2026-10-06 16:21"), path)
    check("record after the 4:15 PM deposit booking: the row leaves that day's deposit out of equity, cash and net "
          "deposits (it counts from the next session); a deposit booked after the balance read is not counted",
          (r2["Equity"], r2["Cash"], r2["Net_Deposits"]) == (58000.0, 1200.0, 57000.0), r2)
# ---------------------------------------------------------------- 3) paper strategies (forward_test.STRATEGIES, no orders)
import numpy as np  # noqa: E402
import sector_mapping  # noqa: E402

rng = np.random.default_rng(7)
SYMS = list(sector_mapping.tradable_symbols)[:30]
sdays = pd.DatetimeIndex([d for d in pd.bdate_range("2025-07-01", "2026-10-16") if be.is_session(d)])

def fake_sig(days):
    """signal_analysis.csv-like rows: random-walk closes and every column the strategies read."""
    out = []
    for k, sym in enumerate(SYMS):
        c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005 * (k % 5 - 2), 0.02, len(days)))), index=days)
        f = pd.DataFrame({"Date": days, "Symbol": sym, "Close": c.values, "RSI": rng.uniform(25, 75, len(days)),
                          "macd": rng.normal(0, 1, len(days)), "MACD Signal": rng.normal(0, 1, len(days)),
                          "Technical_Score": rng.uniform(-60, 90, len(days)), "RS_Score": rng.uniform(-100, 100, len(days)),
                          "Regime_On": 1, "final_trade": rng.choice(["BUY", "SELL", "HOLD", "HOLD"], len(days))})
        for n in (10, 30, 50, 100, 200):
            f[f"ma_{n}"] = c.rolling(n, min_periods=1).mean().values
        out.append(f)
    s = pd.concat(out, ignore_index=True)
    return s.assign(Strategy_Score=0.5 * s["Technical_Score"] + 0.5 * s["RS_Score"])[ft.SIG_COLS]

SIG = fake_sig(sdays)
FACTS = pd.DataFrame({"as_of": "2026-09-01 15:31", "bar_date": "2026-09-01", "Symbol": SYMS,
                      "SentimentScore": rng.normal(1, 3, len(SYMS)), "Fundamental_Weight": rng.uniform(-3, 8, len(SYMS))})
NO_EARN = pd.DataFrame(columns=["Symbol", "Earnings Date"])
EARN = pd.DataFrame({"Symbol": SYMS * 2, "Time": ["AM", "PM"] * 30,     # reports spread over a month, twice
                     "Earnings Date": [pd.Timestamp(m) + pd.Timedelta(days=k) for m in ("2026-07-20", "2026-09-01") for k in range(30)]})
BENCH = pd.DataFrame({"Date": sdays, **{s: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(sdays))))
                                       for s in ["SPY", "QQQ", *be.SECTOR_ETFS]}})
_c = SIG.pivot(index="Date", columns="Symbol", values="Close").stack().rename("Close").reset_index()
BARS = [_c.assign(Open=_c["Close"], High=_c["Close"] * 1.01, Low=_c["Close"] * 0.985,
                  Volume=rng.lognormal(13, 0.5, len(_c)))[["Symbol", "Date", *be.BAR_COLS]]]
w0 = dict(be.WINNER)
inp = ft.strategy_inputs(SIG, FACTS, EARN, BENCH, BARS)
full = {c["name"]: ft.strategy_targets(c, inp) for c in ft.STRATEGIES}
tg = {k: t.loc["2026-10-02":] for k, t in full.items()}
check(f"registry: {len(ft.STRATEGIES)} strategies with unique names, each with a one-line rule and a color",
      len({c["name"] for c in ft.STRATEGIES}) == len(ft.STRATEGIES) >= 28
      and all(c.get("rule") and ft.strategy_color(c["name"]) for c in ft.STRATEGIES)
      and any(c["name"] == "Quant Score top 10" for c in ft.STRATEGIES)
      and any(c["name"] == "ATR dip buy (top 10)" for c in ft.STRATEGIES))
_combo = {c["name"] for c in ft.STRATEGIES if c.get("combo")}   # forward_combos lines (tests/test_forward_combos.py)
check("registry: every strategy picks something, weights 0..19.8% (20% max x 99%), at most 99% invested",
      all((t.to_numpy() >= 0).all() and t.to_numpy().max() <= (0.198 + 1e-12 if k not in _combo else 0.25)
          and t.sum(axis=1).max() <= (0.99 + 1e-9 if k not in _combo else 1.0 + 1e-9)   # combo: drifted weights, no margin
          and (t.sum(axis=1).max() > 0 or k == "EMA 9/21 + ADX>25") for k, t in tg.items()),   # ADX>25: none on the fake data
      {k: (round(t.to_numpy().max(), 4), round(t.sum(axis=1).max(), 4)) for k, t in tg.items()})
check("registry: the strategies are not copies of each other (distinct weights over the fake history)",
      len({t.round(4).to_numpy().tobytes() for k, t in full.items() if k not in _combo}) >= len(ft.STRATEGIES) - len(_combo) - 5)   # the small fake history makes a few
#                                                    coincide; the sleeve stocks are not in it (no sleeve = live now)
check("registry: WINNER untouched", be.WINNER == w0)
# Forward-test safety (t187u): the strategies keep their own Mon/Wed rules (ft.MIDWEEK / EXIT_BELOW / EXIT_TO_TOP, the Oct 2
# rules); a change of the live WINNER Mon/Wed keys must not move any of them.
FT_PINS = dict(midweek=ft.MIDWEEK, exit_all_below=ft.EXIT_BELOW, exit_to_top=ft.EXIT_TO_TOP,
               selection={"sleeve": ()})
check("forward-test pins = the Oct 2 mid-week rules (top-3 swap below 15, exit below 30, refill from the top 10), Mon-Thu",
      ft.MIDWEEK == {"enter_top": 3, "exit_below": 15, "days": ["Mon", "Tue", "Wed", "Thu"]} and (ft.EXIT_BELOW, ft.EXIT_TO_TOP) == (30, 10)
      and ft.NOW["midweek"] == {"enter_top": None, "exit_below": None, "days": ft.MON_THU} and ft.NOW["exit"] == (20, 10))
_keys = ("midweek_swap", "midweek_exit_below", "midweek_exit_to_top")
_saved = {k: be.WINNER.get(k) for k in _keys}
try:
    be.WINNER.update(midweek_swap={"enter_top": 5, "exit_below": 12, "days": ["Mon", "Wed"]}, midweek_exit_below=40,
                     midweek_exit_to_top=None)
    full_alt = {c["name"]: ft.strategy_targets(c, inp) for c in ft.STRATEGIES}
finally:
    be.WINNER.update(_saved)
check("forward-test safety: other WINNER mid-week settings change no strategy (the live 50/50 has no mid-week trading)",
      all(full_alt[k].equals(full[k]) for k in full)
      and be.WINNER == w0, [k for k in full if not full_alt[k].equals(full[k])])
live_raw, _ = be.winner_targets(inp["scores"]["live"], inp["eligible"], inp["vol"], inp["regime"], inp["weekly"],
                                tiebreak_w=inp["scores"]["rs"], earnings=EARN, **FT_PINS)
check("ATR stops: with no stop the day-by-day replay = be.winner_targets exactly (every day, both windows)",
      all(ft.atr_stop_targets(inp, **{**ft.STOPS[m], "k": np.inf})[0].equals(live_raw) for m in ft.STOPS))
# the "Now" strategies (live rules of Oct 15): do_not_buy never held, checks also on Tue / Thu, one change each
now = {c["name"]: full[c["name"]] for c in ft.STRATEGIES if c.get("score") in ("rs", "rs_short", "rs_long", "tech30", "rs_calm")
       and c.get("exit")}
_nb = [s_ for s_ in sector_mapping.do_not_buy if s_ in full[ft.LIVE]]
_chg = lambda t: set(t.index[(t.diff().abs().sum(axis=1) > 1e-12).to_numpy()].dayofweek)
check("Now strategies: 6 (Oct 19: near-duplicates removed), do_not_buy never held, trades on Tue / Thu checks too, all different",
      len(now) == 6 and all((t[_nb].to_numpy() == 0).all() for t in now.values())
      and {1, 3} <= set().union(*(_chg(t) for t in now.values()))
      and len({t.round(6).to_numpy().tobytes() for t in now.values()}) >= 5,     # fake data: no sleeve stocks
      (_nb, sorted(set().union(*(_chg(t) for t in now.values())))))
# stop scenarios: each stock held over a Friday reports (after the close) 2 sessions after that Friday
L, fridays = live_raw.to_numpy(), np.where(inp["weekly"].to_numpy(bool))[0]
EARN2 = pd.DataFrame([{"Symbol": s_, "Earnings Date": live_raw.index[f + 2], "Time": "PM"} for j_, s_ in enumerate(live_raw.columns)
                      for f in fridays[(fridays > 0) & (fridays < len(L) - 6)] if L[f - 1, j_] > 0 and (L[f:f + 6, j_] > 0).all()])
inp2 = dict(inp, earnings=EARN2)
live_raw, _ = be.winner_targets(inp2["scores"]["live"], inp2["eligible"], inp2["vol"], inp2["regime"], inp2["weekly"],
                                tiebreak_w=inp2["scores"]["rs"], earnings=EARN2, **FT_PINS)
cols_ = list(live_raw.columns)
ev = ft.earnings_events(live_raw.index, cols_, EARN2)
react = np.zeros(live_raw.shape, bool)
for j, _, r in ev:
    react[r, j] = True

def in_window(m, t, j):
    """Stop window of column j on row t: post = reaction day .. +10 sessions; pre = earnings within 7 days of the last
    Friday rebalance, through its reaction day (and the stock was held before that Friday)."""
    if m == "post":
        return any(r <= t <= r + 10 for jj, _, r in ev if jj == j)
    f = fridays[fridays <= t].max()
    return any(jj == j and live_raw.index[f] < day <= live_raw.index[f] + pd.Timedelta(days=7) and r >= t
               for jj, day, r in ev)

stops = {}
for m, kk in (("pre", 0.5), ("post", 0.5)):           # small k: the fake random walks rarely fall 3.5 ATR in a few days
    st_w, st_open = stops[m] = ft.atr_stop_targets(inp2, **{**ft.STOPS[m], "k": kk})
    diff = ~np.isclose(st_w, live_raw).all(axis=1)
    t = int(np.argmax(diff)) if diff.any() else None
    ok = t is not None
    if ok:                                            # the first day it differs from live: a stop inside the window
        now, live_now = st_w.iloc[t].to_numpy(), live_raw.iloc[t].to_numpy()     # same holdings as live until yesterday
        sold, bought = np.where((live_now > 0) & (now == 0))[0], np.where((live_now == 0) & (now > 0))[0]
        c_, a_ = inp2["close"].to_numpy(), inp2["atr"].reindex(columns=cols_).to_numpy()
        held_from = [t - np.argmax(st_w.iloc[:t, j].to_numpy()[::-1] == 0) for j in sold]   # the day it was bought
        hit = [c_[t, j] <= c_[h:t + 1, j].max() - kk * a_[t, j] or st_open.iat[t, j] == st_open.iat[t, j]
               for j, h in zip(sold, held_from)]
        ok = len(sold) >= 1 and all(in_window(m, t, j) for j in sold) and all(hit) and np.isclose(now.sum(), live_now.sum()) \
            and set(bought).isdisjoint(sold) and len(bought) == len(sold)
    check(f"ATR stop ({m}-earnings, {kk}x): the first change vs live is a holding in its window at/below its "
          "peak - k ATR, sold in full and replaced at the same weight", ok, t)
o = stops["pre"][1].notna().to_numpy()
check(f"ATR stop (pre-earnings): open sales only on reaction days, at the open price ({len(EARN2)} reports)",
      len(EARN2) >= 4 and react[o].all() and np.allclose(stops["pre"][1].to_numpy()[o], inp2["open"].reindex(columns=cols_).to_numpy()[o]), int(o.sum()))

import filecmp  # noqa: E402
with tempfile.TemporaryDirectory() as tmp:
    P = lambda n: dict(path=os.path.join(tmp, n + ".csv"), hold_path=os.path.join(tmp, n + "_h.csv"))
    n1 = ft.update_strategies(SIG, FACTS, NO_EARN, BENCH, BARS, **P("a"))
    v = db.read_csv(P("a")["path"], parse_dates=["Date"])
    check("no margin: no strategy's cash is ever below 0", v["Cash"].min() >= -1e-12, v.loc[v["Cash"].idxmin()].to_dict())
    _st = [pd.Timestamp((c.get("combo") or {}).get("start") or be.FORWARD_START) for c in ft.STRATEGIES]
    n_rows = lambda lo, hi=None: sum(int(((sdays >= max(lo, s0)) & (sdays <= (hi or sdays.max()))).sum()) for s0 in _st)
    n_oct2 = sum(s0 == pd.Timestamp(be.FORWARD_START) for s0 in _st)
    check("update: one row per strategy and trading day from its start close (Oct 2; the live halves Oct 16), 1.0 on Oct 2",
          n1 == n_rows(pd.Timestamp(be.FORWARD_START)) and (v.loc[v["Date"] == "2026-10-02", "Value"] == 1.0).sum() == n_oct2, n1)
    import shutil  # noqa: E402
    shutil.copy(P("a")["path"], os.path.join(tmp, "a_copy.csv"))
    ft.update_strategies(SIG[SIG["Date"] <= "2026-10-05"], FACTS, NO_EARN, BENCH, BARS, **P("r"))
    first_rows = open(P("r")["path"]).read() + open(P("r")["hold_path"]).read()
    ft.update_strategies(SIG, FACTS, NO_EARN, BENCH, BARS, **P("r"))                       # a later run rewrites the files
    check("a later run keeps every saved row byte-identical (exact float parsing)",
          (open(P("r")["path"]).read() + open(P("r")["hold_path"]).read()).count("\n") > first_rows.count("\n")
          and all(l in (open(P("r")["path"]).read() + open(P("r")["hold_path"]).read()).splitlines()
                  for l in first_rows.splitlines()))
    check("idempotent: a re-run the same day adds nothing and leaves the files unchanged",
          ft.update_strategies(SIG, FACTS, NO_EARN, BENCH, BARS, **P("a")) == 0 and filecmp.cmp(P("a")["path"], os.path.join(tmp, "a_copy.csv"), shallow=False))
    early = [BARS[0][BARS[0]["Date"] <= "2026-10-05"]]                                  # bars only through Oct 5
    n_early = ft.update_strategies(SIG[SIG["Date"] <= "2026-10-07"], FACTS, NO_EARN, BENCH, early, **P("b"))
    check("update: days after the last volume bar wait for the bars", n_early == n_rows(pd.Timestamp(be.FORWARD_START), pd.Timestamp("2026-10-05")), n_early)
    ft.update_strategies(SIG[SIG["Date"] <= "2026-10-07"], FACTS, NO_EARN, BENCH, BARS, **P("b"))  # missed days ...
    nb = ft.update_strategies(SIG, FACTS, NO_EARN, BENCH, BARS, **P("b"))                          # ... caught up later
    a_, b_ = db.read_csv(P("a")["path"]), db.read_csv(P("b")["path"])
    ha, hb = db.read_csv(P("a")["hold_path"]), db.read_csv(P("b")["hold_path"])
    check("catch-up: missed days are added from the saved bars, same values and holdings as running every day",
          nb == n_rows(pd.Timestamp("2026-10-08")) and len(a_) == len(b_)
          and np.allclose(a_["Value"], b_["Value"], rtol=0, atol=1e-12) and ha[["Date", "Strategy", "Symbol"]].equals(hb[["Date", "Strategy", "Symbol"]]),
          (nb, len(a_), len(b_), sorted(set(a_.loc[(a_["Value"] - b_["Value"]).abs() > 1e-12, "Strategy"]))))
    # A later pipeline run rewrote the past (e.g. the 8f57015 rule change: Oct 2 BE -> MRVL). The saved holdings must stay
    # put on a day with no decision (Tue Oct 6) instead of trading the retroactive difference.
    rd = lambda f: db.read_csv(f, dtype={"Date": str}, float_precision="round_trip")
    for k in ("path", "hold_path"):
        d_ = rd(P("a")[k]); d_[d_["Date"] <= "2026-10-05"].pipe(db.to_csv, P("c")[k], index=False)
    hc = rd(P("c")["hold_path"])
    FO = "Now: Friday only"                                               # no decision on Tue Oct 6 (Mon-Thu checks would)
    L = (hc["Strategy"] == FO) & (hc["Date"] == "2026-10-05") & (hc["Shares"] > 0)
    alien = next(s for s in sorted(SIG["Symbol"].unique()) if s not in set(hc.loc[hc["Strategy"] == FO, "Symbol"]))
    gone = hc.loc[L, "Symbol"].iloc[0]
    hc.loc[L & (hc["Symbol"] == gone), "Symbol"] = alien
    hc.pipe(db.to_csv, P("c")["hold_path"], index=False)
    ft.update_strategies(SIG[SIG["Date"] <= "2026-10-06"], FACTS, NO_EARN, BENCH, BARS, **P("c"))
    h6 = rd(P("c")["hold_path"]); h6 = h6[(h6["Strategy"] == FO) & (h6["Shares"] > 0)]
    s5, s6 = (h6[h6["Date"] == x].set_index("Symbol")["Shares"].sort_index() for x in ("2026-10-05", "2026-10-06"))
    ha_ = ha[(ha["Strategy"] == FO) & (ha["Shares"] > 0)]
    pre = set(ha_.loc[ha_["Date"] == "2026-10-05", "Symbol"]) == set(ha_.loc[ha_["Date"] == "2026-10-06", "Symbol"])
    check("rewritten past: a saved holding the new targets no longer have is kept on a day without a decision (no trade)",
          pre and alien in s6.index and gone not in s6.index and s5.equals(s6), (alien, gone, list(s6.index)))
    hv = ft.holdings(P("a")["hold_path"])
    check("holdings: each strategy's latest holdings for the dashboard", set(hv) <= {c["name"] for c in ft.STRATEGIES} and len(hv) >= 15, list(hv))

    # hand-checked accounting: all in AAA from the start (99% live), +10% on Oct 5, switch to BBB at the Oct 6 close
    hd = pd.bdate_range("2026-09-28", "2026-10-09")
    hs = pd.DataFrame([{"Date": d, "Symbol": sym, "Close": px} for i, d in enumerate(hd)
                       for sym, px in (("AAA", 100.0 + (i >= 5) * 10), ("BBB", 50.0))])
    real_inputs, real_targets, real_list = ft.strategy_inputs, ft.strategy_targets, ft.STRATEGIES

    def fake_targets(cfg, inp_):
        t = pd.DataFrame(0.0, index=inp_["close"].index, columns=inp_["close"].columns)
        t.loc[:"2026-10-05", "AAA"], t.loc["2026-10-06":, "BBB"] = 0.99, 0.99
        if cfg["name"] == "Z":                        # half / half; Oct 6 only BBB changes; Oct 7 AAA's target moves < 1 point
            t[:] = 0.0
            t["AAA"], t["BBB"] = 0.495, 0.495
            t.loc["2026-10-06":, "BBB"] = 0.3
            t.loc["2026-10-07":, "AAA"] = 0.52
        if cfg["name"] == "V":                        # fixed 50 / 49 targets; AAA's +10% on Oct 5 is rebalanced on Friday Oct 9
            t[:] = 0.0
            t["AAA"], t["BBB"] = 0.5, 0.49
        if cfg["name"] == "T":                        # Oct 7: BBB 49.5 -> 49% needs more cash than the 1% left; AAA 51% in band
            t[:] = 0.0
            t["AAA"], t["BBB"] = 0.495, 0.495
            t.loc["2026-10-07":, ["AAA", "BBB"]] = [0.51, 0.49]
        if cfg["name"] == "Y":                        # the same switch, but AAA sold at a 95 open on Oct 6 (a stop)
            inp_.setdefault("open_sells", {})["Y"] = t * np.nan
            inp_["open_sells"]["Y"].loc["2026-10-06", "AAA"] = 95.0
        return t
    try:
        def fake_inputs(sig, *a, **k):
            c = sig.pivot(index="Date", columns="Symbol", values="Close")
            soon = pd.DataFrame(False, index=c.index, columns=c.columns)
            soon.loc["2026-10-08":, "BBB"] = True                                      # BBB reports within 5 days
            return {"close": c, "weekly": pd.Series(c.index.dayofweek == 4, index=c.index), "soon": soon}
        ft.strategy_inputs = fake_inputs
        ft.strategy_targets, ft.STRATEGIES = fake_targets, [{"name": n, "rule": n} for n in "XYZVT"]
        ft.update_strategies(hs[hs["Date"] <= "2026-10-05"], None, NO_EARN, **P("h"))   # resume across the switch
        ft.update_strategies(hs, None, NO_EARN, **P("h"))
    finally:
        ft.strategy_inputs, ft.strategy_targets, ft.STRATEGIES = real_inputs, real_targets, real_list
    hy = db.read_csv(P("h")["path"], parse_dates=["Date"])
    ht = hy[hy["Strategy"] == "T"].set_index("Date")
    hv, hy, hz, hw = (hy[hy["Strategy"] == x].set_index("Date")["Value"] for x in "XYZV")
    a5 = 0.01 + 0.99 * 1.1
    after = a5 - be.COST * (0.99 * 1.1 + 0.99 * a5) / (1 + 0.99 * be.COST)   # sell all AAA + buy 99% of what is left
    check("accounting: 1.0 at the start close, follows the closes, pays 0.1% per side on a change, cash earns 0",
          hv.index[0] == pd.Timestamp("2026-10-02") and hv.iloc[0] == 1.0 and abs(hv["2026-10-05"] - a5) < 1e-12
          and abs(hv["2026-10-06"] - after) < 1e-12 and abs(hv.iloc[-1] - after) < 1e-12, hv.round(6).to_dict())
    at_open = (0.01 + 0.0099 * 95 * (1 - be.COST)) / (1 + 0.99 * be.COST)   # AAA sold at the open, then 99% BBB at the close
    check("accounting: a stop sale at the open gets the open price (0.1% cost), the refill buys at the close",
          abs(hy["2026-10-05"] - a5) < 1e-12 and abs(hy["2026-10-06"] - at_open) < 1e-12, hy.round(6).to_dict())
    zh = db.read_csv(P("h")["hold_path"], parse_dates=["Date"]).query("Strategy == 'Z' and Symbol == 'AAA'").set_index("Date")
    z6 = 1.0495 - be.COST * (0.495 - 0.3 * 1.0495) / (1 - 0.3 * be.COST)  # Oct 5: .01 + .00495 x 110 + .0099 x 50; Oct 6: only BBB sold down to 30%
    check("accounting: only names whose target changed trade (AAA keeps its shares); a target within 1 point is left alone",
          abs(hz["2026-10-05"] - 1.0495) < 1e-12 and abs(hz["2026-10-06"] - z6) < 1e-12 and abs(hz["2026-10-07"] - z6) < 1e-12
          and np.allclose(zh["Shares"], 0.00495, rtol=0, atol=1e-15), (hz.round(8).to_dict(), zh["Shares"].tolist()))
    w9 = 1.05 - be.COST * (0.55 - 0.5 * 1.05) / (1 - 0.5 * be.COST)   # Friday: AAA 52.4% -> 50%; BBB (46.7%, earnings soon) is not bought up
    check("accounting: drift is left alone mid-week and brought back on the Friday rebalance; earnings soon: not bought up",
          np.allclose(hw["2026-10-05":"2026-10-08"], 1.05, rtol=0, atol=1e-12) and abs(hw["2026-10-09"] - w9) < 1e-12,
          hw.round(8).to_dict())
    t7 = 1.0495 - be.COST * (0.0495 - 0.02 * 1.0495) / (1 - 0.02 * be.COST)   # BBB bought up, AAA trimmed, 0.1% of both
    check("accounting: buys that need more than the cash first trim a band hold above its target (no negative cash)",
          abs(ht.at[pd.Timestamp("2026-10-07"), "Value"] - t7) < 1e-12 and (ht["Cash"] >= -1e-12).all(), ht.round(8).to_dict())
    rd = {c["name"]: ft.rebalance_days(c, {"weekly": "W", "monthly": "M"}) for c in ft.STRATEGIES}
    check("rebalance days: the calendar of each rule (Fridays, month ends, never for the dip and threshold rules)",
          rd[ft.LIVE] == rd["Now: Friday only"] == "W" and rd["Monthly"] == "M"
          and rd["Buy the dip"] is None and set(rd.values()) <= {"W", "M", None}, rd)

# 200-day breakout family (Oct 19, 2026): 3 closes above the 200-day after one below, bought on the 3rd; TP / stop / 200-day exit
_bd = pd.bdate_range("2026-10-01", periods=12)
_c = pd.DataFrame({"A": [99, 101, 102, 103, 104, 110, 125, 120, 120, 120, 120, 120.0],      # breaks out, +20% TP on day 6
                   "B": [99, 101, 102, 103, 100, 99.5, 98, 98, 98, 98, 98, 98.0],          # stop -3% / 1% below the 200-day
                   "C": [101] * 12}, index=_bd, dtype=float)                                # always above: never a breakout
_m = pd.DataFrame(100.0, index=_bd, columns=_c.columns)
_e = pd.DataFrame(True, index=_bd, columns=_c.columns)
_v = pd.DataFrame({"A": 0.2, "B": 0.1, "C": 0.3}, index=_bd)
wa = ft.breakout200_weights(_c, _m, _e, _v, tp=0.20, start="2026-10-01")
check("breakout: bought on the 3rd close above (10% each), sold at +20% from the entry close",
      wa["A"].tolist() == [0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0] and wa["C"].eq(0).all(), wa.to_dict())
check("breakout: plain exit = the close below the 200-day; no stop", wa["B"].tolist()[:6] == [0, 0, 0, 0.1, 0.1, 0], wa["B"].tolist())
wc = ft.breakout200_weights(_c, _m, _e, _v, tp=0.20, stop=0.03, start="2026-10-01")
check("breakout: -3% stop from the entry (103 -> 99.91) sells at 99.5", wc["B"].tolist()[:6] == [0, 0, 0, 0.1, 0.1, 0], wc["B"].tolist())
wg = ft.breakout200_weights(_c, _m, _e, _v, tp=0.20, buffer=0.99, start="2026-10-01")
check("breakout g: held at 99.5 (within 1% of the 200-day), sold at 98 (more than 1% below)",
      wg["B"].tolist()[:7] == [0, 0, 0, 0.1, 0.1, 0.1, 0], wg["B"].tolist())
w1 = ft.breakout200_weights(_c, _m, _e, _v, tp=0.20, n=1, start="2026-10-01")
check("breakout: more signals than slots -> the calmest first (B before A); unused cash stays cash",
      w1.loc[_bd[3], "B"] == 1.0 and w1.loc[_bd[3], "A"] == 0, w1.iloc[3].to_dict())
w5 = {d: ft.breakout200_weights(_c, _m, _e, _v, tp=0.20, days=d, start="2026-10-01")["A"].tolist() for d in (1, 2, 4)}
check("breakout N days: bought on the Nth close above (A crosses on day 2)", w5[1][1] == 0.1 and w5[1][0] == 0
      and w5[2][1:3] == [0, 0.1] and w5[4][3:5] == [0, 0.1], w5)
w98 = ft.breakout200_weights(_c, _m, _e, _v, tp=0.20, buffer=0.98, start="2026-10-01")["B"].tolist()
check("breakout 2% exit: held at 99.5 and 98 (not 2%+ below)... sold below 98", w98[3:7] == [0.1, 0.1, 0.1, 0.1], w98)
check("breakout: 2 variants (N = 2, 4), exit 2% below the 200-day, no calendar rebalance", [c["b200"].get("days") for c in ft.STRATEGIES if c.get("b200")] == [2, 4] and all(c["b200"]["buffer"] == 0.98 for c in ft.STRATEGIES if c.get("b200"))
      and all(ft.rebalance_days(c, {"weekly": "W"}) is None for c in ft.STRATEGIES if c.get("b200")))

# event lines (Oct 19, 2026): fresh crosses only, TP / exit signal, same slots as the breakouts
_x = pd.DataFrame(False, index=_bd, columns=_c.columns)
_en = _x.copy(); _en.loc[_bd[1], ["A", "B"]] = True
_ex = _x.copy(); _ex.loc[_bd[4], "B"] = True
we = ft.event_slot_weights(_c, _en, _ex, _e, _v, tp=0.20, start="2026-10-01")
check("event lines: bought at the entry close, B sold on its exit signal, A at +20% (101 -> 125)",
      we["B"].tolist()[:6] == [0, 0.1, 0.1, 0.1, 0, 0] and we["A"].tolist()[:7] == [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0], we.to_dict())
_f = lambda v: pd.DataFrame({"A": v}, index=_bd[:len(v)], dtype=float)
_i = {"ma10": _f([1, 1, 3, 3, 3, 1]), "ma30": _f([2, 2, 2, 2, 2, 2]), "ma50": _f([0] * 6), "ma200": _f([0] * 6),
      "eligible": _f([1] * 6).astype(bool), "scores": {"tech": _f([0] * 6)}}
_en3, _ex3 = ft.event_signals(_i, "ma10_30")
check("MA10 x MA30: entry on the 2nd close after a fresh cross only; exit when MA10 <= 0.98 x MA30",
      _en3["A"].tolist() == [False, False, False, True, False, False] and _ex3["A"].tolist()[-1], _en3["A"].tolist())
_i["ma10"] = _f([3] * 6)
check("MA10 x MA30: already above from the start is not a fresh cross", not ft.event_signals(_i, "ma10_30")[0]["A"].any())
wt = ft.event_slot_weights(_c, _en, _x, _e, _v, max_days=3, start="2026-10-01")["B"].tolist()
ws = ft.event_slot_weights(_c, _en, _x, _e, _v, stop=0.05, start="2026-10-01")["B"].tolist()
check("event lines: time exit after 3 sessions; -5% stop from entry not hit at 98 (stop 95.95)",
      wt[:6] == [0, 0.1, 0.1, 0.1, 0, 0] and ws[1:7] == [0.1] * 6, (wt, ws))
_ii = {**_i, "close": _c[["A"]].iloc[:6]}
check("event signals: every kind builds", all(ft.event_signals({**_ii, "ma10": _f([1] * 6)}, k)[0].shape == (6, 1) for k in
      ("tech80", "tech70", "ma10_30", "golden", "macd", "donchian", "bollinger", "ema", "ema_adx", "minervini", "high52",
      "ma_rsi", "golden_macd")))
_dn = sorted(set(sector_mapping.do_not_buy) & set(full))
check("do-not-buy: no paper strategy (all but the live rules) ever holds one",
      all((full[k][_dn].to_numpy() == 0).all() for k in full if not k.startswith("Live") and k != ft.LIVE), _dn)
check("event lines: 13 named lines, no calendar rebalance",
      [c["name"] for c in ft.STRATEGIES if c.get("event")] == ["Tech score 80/60 + TP20", "Tech score 70/60/90", "MA10 x MA30",
       "MA50 x MA200 (golden cross)", "MACD cross", "Donchian 20-day breakout", "Bollinger mean reversion", "EMA 9/21 cross",
       "EMA 9/21 + ADX>25", "Minervini trend template", "52-week high momentum", "Dual MA + RSI filter", "Golden cross + MACD confirm"]
      and all(ft.rebalance_days(c, {"weekly": "W"}) is None for c in ft.STRATEGIES if c.get("event")))

# the rules that are not rankings, on tiny hand-made data
ud = pd.bdate_range("2026-01-05", periods=8)
uc = pd.DataFrame({"A": [100, 91, 90, 95, 100, 101, 90, 89.0], "B": [100, 100, 100, 100, 100, 100, 100, 91.0]}, index=ud)
um = pd.DataFrame(100.0, index=ud, columns=uc.columns)
dw = ft.dip_weights(uc, um, uc.notna(), n=2)
check("buy the dip: in at 8%+ below the 50-day, out at the average, back in on a new dip, 1/n each",
      list(dw["A"]) == [0, 0.5, 0.5, 0.5, 0, 0, 0.5, 0.5] and list(dw["B"]) == [0] * 7 + [0.5], dw.to_dict("list"))
check("buy the dip: sold after max_hold sessions, not bought back the same day (still 10% below)",
      list(ft.dip_weights(uc, um, uc.notna(), max_hold=1, n=2)["A"][:4]) == [0, 0.5, 0, 0],
      ft.dip_weights(uc, um, uc.notna(), max_hold=1, n=2)["A"].tolist())
spy = pd.Series(100.0, index=ud)
spy.iloc[3] = 101.0
ed = ft.earnings_drift(uc, spy, pd.DataFrame({"Symbol": ["A", "B"], "Earnings Date": [ud[2], ud[2]], "Time": ["PM", "AM"]}), hold=2)
check("earnings drift: the reaction day (next session after a PM report, same day for AM) vs SPY, kept for `hold` sessions",
      ed["A"].iloc[3:5].round(6).tolist() == [round((95 / 90 - 101 / 100) * 100, 6)] * 2 and ed["A"].iloc[[2, 5]].isna().all()
      and ed["B"].iloc[2:4].tolist() == [0.0, 0.0] and ed["B"].iloc[4:].isna().all(), ed.to_dict("list"))
ds = pd.DataFrame(False, index=pd.bdate_range("2026-01-05", periods=6), columns=list("ABC"))
ds.iloc[0, 0] = ds.iloc[1, 1] = ds.iloc[1, 2] = ds.iloc[3, 0] = True
dp = pd.DataFrame([[1.0, 2.0, 3.0]] * 6, index=ds.index, columns=ds.columns)           # C ranks best, then B
d1, d2 = ft.atr_dip_weights(ds, dp, hold=3, n=2), ft.atr_dip_weights(ds, dp, hold=3, n=2, room=np.array([2, 2, 1, 2, 2, 2]))
check("ATR dip buy: 1/n each, best rank first, sold after `hold` sessions (bought again on a new signal)",
      d1["A"].tolist() == [0.5] * 6 and d1["B"].sum() == 0 and d1["C"].tolist() == [0, 0.5, 0.5, 0.5, 0, 0], d1.to_dict("list"))
dsell = pd.DataFrame(False, index=ds.index, columns=ds.columns)
dsell.iloc[[2, 4], 2] = dsell.iloc[5, 0] = True                      # C: sell on days 2 and 4; A: sell on day 5
d3 = ft.atr_dip_weights(ds.assign(C=ds["C"] | (ds.index == ds.index[4])), dp, hold=None, n=2, sell=dsell)   # C dips again on 4
check("ATR dip buy, sell on signal: no fixed hold, sold on its sell day, not bought on a sell day",
      d3["A"].tolist() == [0.5] * 5 + [0] and d3["C"].tolist() == [0, 0.5, 0, 0, 0, 0] and d3["B"].sum() == 0, d3.to_dict("list"))
check("ATR dip buy: fewer rooms than positions -> the oldest is sold",
      d2["A"].tolist() == [0.5, 0.5, 0, 0.5, 0.5, 0.5] and d2["C"].tolist() == [0, 0.5, 0.5, 0.5, 0, 0], d2.to_dict("list"))
vb = pd.DataFrame({"Symbol": "A", "Date": pd.bdate_range("2026-01-05", periods=20), "Open": 1.0, "High": 12.0, "Low": 8.0,
                   "Close": [10.0] * 19 + [10.5], "Volume": [100.0] * 19 + [300.0]})
vf = ft.bar_features([vb], pd.DatetimeIndex(vb["Date"]), ["A"])
vw = (10 * 100 * 19 + (12 + 8 + 10.5) / 3 * 300) / (100 * 19 + 300)
check("Quant Score VWAP part: close / 20-day VWAP of the typical price", abs(vf["vwap_ratio"]["A"].iloc[-1] - 10.5 / vw) < 1e-12
      and vf["vwap_ratio"]["A"].iloc[:-1].isna().all(), vf["vwap_ratio"]["A"].tail(2).tolist())

# the real saved data (read-only; output to temporary files): the live rules recomputed = the live Strategy_Weight
LIVE_FILES = [os.path.join(ROOT, "Reports", f) for f in ("signal_analysis.csv", "strategy_picks.csv", "run_state.json",
                                                         "live_pending_orders.json", "live_orders_log.csv", "factor_history.csv")]

def digest():
    import hashlib
    return {f: hashlib.sha256((db.table_text(db.table_name(f)) or open(f, "rb").read().decode()).encode()).hexdigest() for f in LIVE_FILES if db.exists(f)}

if db.exists(ft.SIGNAL_CSV):
    before = digest()
    real_inp = ft.strategy_inputs(db.read_csv(ft.SIGNAL_CSV, usecols=ft.SIG_COLS, parse_dates=["Date"]),
                                  ft._read(ft.FACTOR_CSV), None, ft._read(ft.BENCH_CSV), bars=[])   # no bar download
    real = db.read_csv(ft.SIGNAL_CSV, usecols=["Date", "Symbol", "Strategy_Weight"], parse_dates=["Date"])
    d0 = pd.Timestamp(be.FORWARD_START)
    live = real[(real["Date"] == d0) & (real["Strategy_Weight"] > 0)].set_index("Symbol")["Strategy_Weight"].sort_index()
    real_live, _ = be.winner_targets(real_inp["scores"]["live"], real_inp["eligible"], real_inp["vol"], real_inp["regime"],
                                     real_inp["weekly"], tiebreak_w=real_inp["scores"]["rs"], earnings=real_inp["earnings"],
                                     **FT_PINS)
    check("ATR stops on the real data: with no stop the replay = the (pinned Oct 2) live targets on every saved day",
          all(ft.atr_stop_targets(real_inp, **{**ft.STOPS[m], "k": np.inf})[0].equals(real_live) for m in ft.STOPS))
    check("exit replaced by top-N on the real data: with N = the pinned EXIT_TO_TOP the replay = the pinned live targets",
          ft.atr_stop_targets(real_inp, k=np.inf, exit_to_top=ft.EXIT_TO_TOP)[0].equals(real_live))
    # the live strategy's rule weights (before do_not_buy and the spare cash, order-time rules) = the pipeline's own
    # Strategy_Weight on every forward-test day (a file written before Oct 15, 2:30 PM still has Mon/Wed checks)
    pipe = real.pivot(index="Date", columns="Symbol", values="Strategy_Weight").fillna(0.0).loc[d0:]
    fits = [float((be.live_weights(be.winner_targets(real_inp["scores"]["now"], real_inp["eligible"], real_inp["vol"],
                   real_inp["regime"], real_inp["weekly"], tiebreak_w=real_inp["scores"]["rs"], earnings=real_inp["earnings"],
                   midweek={**be.WINNER["midweek_swap"], "days": dd})[0])
                   .reindex(index=pipe.index, columns=pipe.columns).fillna(0.0) - pipe).abs().max().max())
            for dd in (be.WINNER["midweek_swap"]["days"], ["Mon", "Wed"])]
    mw = os.path.join(os.path.dirname(ft.SIGNAL_CSV), "strategy_midweek_check.csv")
    file_tag = str(db.read_csv(mw)["Rules"].iloc[0]) if db.exists(mw) else ""
    if file_tag.startswith("LV-"):   # reports written by the live monthly low-vol rules (from Oct 16): checked in test_low_vol_monthly / test_rank_audit
        print(f"skip  live strategy = the pipeline's Strategy_Weight (reports carry {file_tag}, not the pinned Oct 15 rules)")
    else:
        check("live strategy = the pipeline's Strategy_Weight on every day since Oct 2 (same be.WINNER rules)", min(fits) < 1e-9, fits)
    check("no live file changed (signal_analysis, picks, run_state, pending orders, orders log); WINNER untouched",
          digest() == before and be.WINNER == w0)
# provisional marks (display only): Tue / Thu have no signal rows until the next decision run
with tempfile.TemporaryDirectory() as pt:
    PN = ft.STRATEGIES[1]["name"]                                   # "P" = a real strategy name (the board lists those)
    pv = pd.DataFrame({"Date": ["2026-10-02", "2026-10-05", "2026-10-02", "2026-10-05"], "Strategy": [PN, PN, "Q", "Q"],
                       "Value": [1.0, 1.02, 1.0, 0.99], "Cash": [0.01, 0.01, 0.5, 0.5]})
    ph = pd.DataFrame({"Date": ["2026-10-05"] * 3, "Strategy": [PN, PN, "Q"], "Symbol": ["AAA", "BBB", "AAA"],
                       "Weight": [0.5, 0.49, 0.49], "Shares": [0.005, 0.01, 0.004], "Price": [102.0, 50.0, 122.5]})
    ph.pipe(db.to_csv, os.path.join(pt, "h.csv"), index=False)
    pb = pd.DataFrame([{"Date": pd.Timestamp(d), "Symbol": s, "Open": px, "High": px, "Low": px, "Close": px, "Volume": 1e6}
                       for d, s, px in (("2026-10-05", "AAA", 102.0), ("2026-10-06", "AAA", 110.0),
                                        ("2026-10-05", "QQQ", 600.0), ("2026-10-06", "QQQ", 606.0))])   # no BBB bar Oct 6
    pb.to_pickle(os.path.join(pt, "b.pkl"))
    before_pv = pv.copy()
    pvals, pbench = ft.provisional(pv, pd.DataFrame({"Date": ["2026-10-02", "2026-10-05"], "QQQ": [590.0, 600.0]}),
                                   hold_path=os.path.join(pt, "h.csv"), bars_path=os.path.join(pt, "b.pkl"))
    p6 = pvals[pvals["Date"] == "2026-10-06"].set_index("Strategy")
    check("provisional: a day with bars but no saved row = saved shares + cash at that close (no trades), not saved",
          len(p6) == 2 and abs(p6.at[PN, "Value"] - (0.01 + 0.005 * 110 + 0.01 * 50)) < 1e-12
          and abs(p6.at["Q", "Value"] - (0.5 + 0.004 * 110)) < 1e-12 and p6["Provisional"].all()
          and not pvals.loc[pvals["Date"] <= "2026-10-05", "Provisional"].astype(bool).any() and pv.equals(before_pv)
          and list(pbench["QQQ"]) == [590.0, 600.0, 606.0], (p6.to_dict(), pbench.to_dict()))
    lbp = ft.leaderboard(pd.DataFrame(), pbench, pvals).set_index("Strategy")
    check("provisional: the board moves with the marked day (P +2% -> +6%, QQQ to Oct 6)",
          abs(lbp.at[PN, "Total return %"] - 6.0) < 1e-9 and abs(lbp.at[PN, "Daily return %"] - (106 / 102 - 1) * 100) < 1e-9
          and abs(lbp.at["QQQ (comparison)", "Total return %"] - (606 / 590 - 1) * 100) < 1e-9, lbp.head(3).to_dict())
    check("provisional: nothing after the last saved day -> unchanged",
          ft.provisional(pv[pv["Date"] <= "2026-10-05"], None, hold_path=os.path.join(pt, "h.csv"),
                         bars_path=os.path.join(pt, "nope.pkl"))[0].equals(pv))
# --- Top 5 consensus (display only): 5 best strategies, 10 most-held stocks; twins once
_N = [c["name"] for c in ft.STRATEGIES]
_bd = pd.DataFrame({"Rank": pd.array([pd.NA] * 6, dtype="Int64"), "Strategy": [_N[0] + ft.LIVE_MARK, _N[1], _N[2], _N[3], "QQQ (comparison)", _N[4]],
                    "Total return %": [1.0, 3.0, 2.0, 2.5, 9.0, 0.5], "Max drawdown %": [0.0] * 6})
_hd = pd.DataFrame([("2026-10-05", _N[1], "OLD", 0.5),                                   # an older day: not used
                    ("2026-10-06", _N[1], "AAA", 0.3), ("2026-10-06", _N[1], "BBB", 0.2),
                    ("2026-10-06", _N[3], "AAA", 0.3), ("2026-10-06", _N[3], "BBB", 0.2),  # twin of _N[1] (same weights)
                    ("2026-10-06", _N[2], "BBB", 0.1), ("2026-10-06", _N[2], "CCC", 0.4), ("2026-10-06", _N[2], "ZZZ", 0.0),
                    ("2026-10-06", _N[0], "AAA", 0.2), ("2026-10-06", _N[0], "DDD", 0.2),
                    ("2026-10-06", _N[4], "EEE", 0.5)], columns=["Date", "Strategy", "Symbol", "Weight"])
_c = ft.consensus(_bd, _hd, {"AAA": 5, "BBB": 1, "CCC": 2, "DDD": 3}, n_strategies=3)
check("Top 5 consensus: no week column -> by total return; benchmarks out; twins once (better-placed kept); n best only",
      _c["by"] == "Total return %" and [n for n, _ in _c["strategies"]] == [_N[1], _N[2], _N[0]] and _c["strategies"][0][1] == [_N[3]],
      _c["strategies"])
check("Top 5 consensus: stocks by count, then summed weight, then latest rank; weight 0 and older days ignored",
      list(_c["stocks"]["Symbol"]) == ["AAA", "BBB", "CCC", "DDD"] and list(_c["stocks"]["Count"]) == [2, 2, 1, 1],
      _c["stocks"].to_dict("list"))
_c2 = ft.consensus(_bd.assign(Rank=pd.array([2, 1, 3, pd.NA, pd.NA, 4], dtype="Int64"), **{"Week 1 %": [5.0, 1.0, 1.0, 1.0, 9.0, 0.0],
                                "Week 2 (so far) %": [0.5, 0.1, 2.0, 3.0, 9.0, 0.0]}), _hd, None, n_strategies=2)
check("Top 5 consensus: the leaderboard's Rank order (not the latest week); unranked after; twins once",
      _c2["by"] == "Total return %" and [n for n, _ in _c2["strategies"]] == [_N[1], _N[0]] and _c2["strategies"][0][1] == [_N[3]],
      _c2["strategies"])
import inspect
_sig = inspect.signature(ft.consensus)
check("Top 5 consensus: defaults are 5 strategies and 10 stocks",
      _sig.parameters["n_strategies"].default == 5 and _sig.parameters["n_stocks"].default == 10)
src = open(os.path.join(ROOT, "forward_test.py"), encoding="utf-8").read()
_src_wo_get = (src.replace("be.WINNER.get(", "")
               .replace("WINNER.get(", ""))  # allow .get; ban WINNER[ writes/reads
check("forward_test.py has no order code and never writes the live files or WINNER",
      not any(w in src for w in ("submit_order", "cancel_order", "OrderRequest", "TradingClient", "import paper_trade",
                                 "requests.post", "run_state", "live_pending_orders", "WINNER.update"))
      and "WINNER[" not in _src_wo_get)
print(f"\n{len(FAIL)} failed" if FAIL else "\nFORWARD TEST OK")
sys.exit(1 if FAIL else 0)
