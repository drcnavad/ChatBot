"""Forward testing from backtest_engine.FORWARD_START (2026-10-02). 1) The dashboard's per-stock forward test (forward_test): only sessions on/after FORWARD_START count (a trade
before it is ignored, a position held into it starts at the start close), trades fill at the decision-day close with COST
per side, and the empty state is None (the panel then says no closed trades yet). 2) forward_test.py (the live account):
picks start from a flat position on/after the start, trading cost vs the decision price, deposits are not returns, weekly
median / drawdown, and record() saves one row per trading day (idle on holidays) through a FAKE account - no requests,
temporary files only. 3) The paper strategies on fake bars / benchmarks / earnings (never a download)."""
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
sm = ft.leaderboard(daily, bench, values=pd.DataFrame(columns=["Date", "Strategy", "Value"])).set_index("Strategy")
st_ = sm.loc[ft.ACCOUNT]
check("leaderboard: deposits are not returns (total +6%)", abs(st_["Total return %"] - 6) < 1e-9, st_.to_dict())
check("leaderboard: median weekly return Friday to Friday (+3%, +2.91%), 2 weeks",
      abs(st_["Median weekly return %"] - (3 + (106 / 103 - 1) * 100) / 2) < 1e-9 and st_["Weeks"] == 2, st_.to_dict())
check("leaderboard: max drawdown (103k -> 100k on Oct 12)", abs(st_["Max drawdown %"] - (100 / 103 - 1) * 100) < 1e-9,
      st_["Max drawdown %"])
q = sm.loc["QQQ (comparison)"]
check("leaderboard: QQQ from the start close", abs(q["Total return %"] + 1) < 1e-9 and abs(q["Max drawdown %"] - (99 / 110 - 1) * 100) < 1e-9,
      q.to_dict())
check("leaderboard: the account and QQQ / SPY are not ranked", sm.loc[[ft.ACCOUNT, "QQQ (comparison)", "SPY (comparison)"], "Rank"].isna().all())
vals = pd.DataFrame([{"Date": d, "Strategy": c["name"], "Value": 1.0} for d in days for c in ft.STRATEGIES])
vals.loc[(vals["Strategy"] == "Top 5") & (vals["Date"] >= pd.Timestamp("2026-10-09")), "Value"] = 1.02   # +2% then flat
vals.loc[(vals["Strategy"] == "Top 20") & (vals["Date"] >= pd.Timestamp("2026-10-09")), "Value"] = 1.02
vals.loc[(vals["Strategy"] == "Top 20") & (vals["Date"] == pd.Timestamp("2026-10-05")), "Value"] = 0.95  # same median, deeper drop
lb = ft.leaderboard(daily, bench, vals)
check("leaderboard: ranked by median weekly return, a tie goes to the smaller drawdown; Rank counts strategies only",
      list(lb.dropna(subset=["Rank"])["Strategy"][:2]) == ["Top 5", "Top 20"] and list(lb.dropna(subset=["Rank"])["Rank"][:2]) == [1, 2]
      and sorted(lb["Rank"].dropna()) == list(range(1, len(ft.STRATEGIES) + 1)), lb.head(4).to_dict("records"))
check("leaderboard: the live row is marked", (lb["Strategy"] == ft.LIVE + ft.LIVE_MARK).sum() == 1, list(lb["Strategy"]))
check("leaderboard: no rank before the first full week", ft.leaderboard(daily.iloc[:1], bench, vals[vals["Date"] == "2026-10-02"])["Rank"].isna().all())
check("verdict: too early before 12 weeks", ft.verdict(lb) == "Week 2 of 12: too early to name a winner.", ft.verdict(lb))
long_days = pd.bdate_range("2026-10-02", periods=70)
lv = pd.DataFrame([{"Date": d, "Strategy": c["name"], "Value": 1.0 + i * (0.002 if c["name"] == "Top 5" else 0.0)}
                   for i, d in enumerate(long_days) for c in ft.STRATEGIES])
lbq = lambda q_end: ft.leaderboard(daily.iloc[:0], pd.DataFrame({"Date": long_days, "QQQ": 100.0, "SPY": 100.0}).assign(
    QQQ=lambda x: [100.0] * 69 + [q_end]), lv)
check("verdict: after 12 weeks the top strategy wins only if it beats live and QQQ",
      "the winner is Top 5" in ft.verdict(lbq(100.0)) and "no winner" in ft.verdict(lbq(200.0)),
      (ft.verdict(lbq(100.0)), ft.verdict(lbq(200.0))))


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
EARN = pd.DataFrame({"Symbol": SYMS * 2, "Earnings Date": pd.to_datetime(["2026-08-04"] * 30 + ["2026-09-15"] * 30),
                     "Time": ["AM", "PM"] * 30})
BENCH = pd.DataFrame({"Date": sdays, **{s: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(sdays))))
                                       for s in ["SPY", "QQQ", *be.SECTOR_ETFS]}})
_c = SIG.pivot(index="Date", columns="Symbol", values="Close").stack().rename("Close").reset_index()
BARS = [_c.assign(Open=_c["Close"], High=_c["Close"] * 1.01, Low=_c["Close"] * 0.985,
                  Volume=rng.lognormal(13, 0.5, len(_c)))[["Symbol", "Date", *be.BAR_COLS]]]
w0 = dict(be.WINNER)
inp = ft.strategy_inputs(SIG, FACTS, EARN, BENCH, BARS)
tg = {c["name"]: ft.strategy_targets(c, inp).loc["2026-10-02":] for c in ft.STRATEGIES}
check(f"registry: {len(ft.STRATEGIES)} strategies with unique names, each with a one-line rule",
      len({c["name"] for c in ft.STRATEGIES}) == len(ft.STRATEGIES) >= 28 and all(c.get("rule") for c in ft.STRATEGIES))
check("registry: every strategy picks something, weights 0..19.8% (20% max x 99%), at most 99% invested",
      all((t.to_numpy() >= 0).all() and t.to_numpy().max() <= 0.198 + 1e-12 and t.sum(axis=1).max() <= 0.99 + 1e-9
          and t.sum(axis=1).max() > 0 for t in tg.values()),
      {k: (round(t.to_numpy().max(), 4), round(t.sum(axis=1).max(), 4)) for k, t in tg.items()})
check("registry: the strategies are not copies of each other (distinct weights over the test)",
      len({t.round(4).to_numpy().tobytes() for t in tg.values()}) >= len(ft.STRATEGIES) - 3)
check("registry: WINNER untouched", be.WINNER == w0)

import filecmp  # noqa: E402
with tempfile.TemporaryDirectory() as tmp:
    P = lambda n: dict(path=os.path.join(tmp, n + ".csv"), hold_path=os.path.join(tmp, n + "_h.csv"))
    n1 = ft.update_strategies(SIG, FACTS, NO_EARN, BENCH, BARS, **P("a"))
    v = pd.read_csv(P("a")["path"], parse_dates=["Date"])
    n_days = (sdays >= pd.Timestamp(be.FORWARD_START)).sum()
    check("update: one row per strategy and trading day from the Oct 2 close, each at 1.0 on Oct 2",
          n1 == len(ft.STRATEGIES) * n_days and (v.loc[v["Date"] == "2026-10-02", "Value"] == 1.0).sum() == len(ft.STRATEGIES), n1)
    import shutil  # noqa: E402
    shutil.copy(P("a")["path"], os.path.join(tmp, "a_copy.csv"))
    check("idempotent: a re-run the same day adds nothing and leaves the files unchanged",
          ft.update_strategies(SIG, FACTS, NO_EARN, BENCH, BARS, **P("a")) == 0 and filecmp.cmp(P("a")["path"], os.path.join(tmp, "a_copy.csv"), shallow=False))
    early = [BARS[0][BARS[0]["Date"] <= "2026-10-05"]]                                  # bars only through Oct 5
    n_early = ft.update_strategies(SIG[SIG["Date"] <= "2026-10-07"], FACTS, NO_EARN, BENCH, early, **P("b"))
    check("update: days after the last volume bar wait for the bars", n_early == len(ft.STRATEGIES) * 2, n_early)
    ft.update_strategies(SIG[SIG["Date"] <= "2026-10-07"], FACTS, NO_EARN, BENCH, BARS, **P("b"))  # missed days ...
    nb = ft.update_strategies(SIG, FACTS, NO_EARN, BENCH, BARS, **P("b"))                          # ... caught up later
    a_, b_ = pd.read_csv(P("a")["path"]), pd.read_csv(P("b")["path"])
    ha, hb = pd.read_csv(P("a")["hold_path"]), pd.read_csv(P("b")["hold_path"])
    check("catch-up: missed days are added from the saved bars, same values and holdings as running every day",
          nb == len(ft.STRATEGIES) * (sdays > pd.Timestamp("2026-10-07")).sum() and len(a_) == len(b_)
          and np.allclose(a_["Value"], b_["Value"], rtol=0, atol=1e-12) and ha[["Date", "Strategy", "Symbol"]].equals(hb[["Date", "Strategy", "Symbol"]]),
          (nb, len(a_), len(b_)))
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
        return t
    try:
        ft.strategy_inputs = lambda sig, *a, **k: {"close": sig.pivot(index="Date", columns="Symbol", values="Close")}
        ft.strategy_targets, ft.STRATEGIES = fake_targets, [{"name": "X", "rule": "x"}]
        ft.update_strategies(hs[hs["Date"] <= "2026-10-05"], None, NO_EARN, **P("h"))   # resume across the switch
        ft.update_strategies(hs, None, NO_EARN, **P("h"))
    finally:
        ft.strategy_inputs, ft.strategy_targets, ft.STRATEGIES = real_inputs, real_targets, real_list
    hv = pd.read_csv(P("h")["path"], parse_dates=["Date"]).set_index("Date")["Value"]
    a5 = 0.01 + 0.99 * 1.1
    after = a5 - be.COST * (0.99 * 1.1 + 0.99 * a5)      # sell all AAA + buy 99% BBB
    check("accounting: 1.0 at the start close, follows the closes, pays 0.1% per side on a change, cash earns 0",
          hv.index[0] == pd.Timestamp("2026-10-02") and hv.iloc[0] == 1.0 and abs(hv["2026-10-05"] - a5) < 1e-12
          and abs(hv["2026-10-06"] - after) < 1e-12 and abs(hv.iloc[-1] - after) < 1e-12, hv.round(6).to_dict())

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
r = rng.normal(0, 1, (80, 3)) * [0.01, 0.02, 0.04]
rc = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=pd.bdate_range("2026-01-01", periods=80), columns=list("XYZ"))
rt = pd.DataFrame(0.3, index=rc.index, columns=rc.columns)
rp = ft.risk_parity(rt, rc, pd.Series(rc.index == rc.index[-1], index=rc.index)).iloc[-1]   # recomputed on the last day
cov = rc.pct_change().iloc[-63:].cov().to_numpy()
contrib = rp.to_numpy() * (cov @ rp.to_numpy())
check("risk parity: same total, equal risk contributions, the calmest stock gets the most",
      abs(rp.sum() - 0.9) < 1e-12 and np.allclose(contrib, contrib.mean(), rtol=1e-6) and rp["X"] > rp["Y"] > rp["Z"], rp.to_dict())
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
    return {f: hashlib.sha256(open(f, "rb").read()).hexdigest() for f in LIVE_FILES if os.path.exists(f)}


if os.path.exists(ft.SIGNAL_CSV):
    before = digest()
    real_inp = ft.strategy_inputs(pd.read_csv(ft.SIGNAL_CSV, usecols=ft.SIG_COLS, parse_dates=["Date"]),
                                  ft._read(ft.FACTOR_CSV), None, ft._read(ft.BENCH_CSV), bars=[])   # no bar download
    real = pd.read_csv(ft.SIGNAL_CSV, usecols=["Date", "Symbol", "Strategy_Weight"], parse_dates=["Date"])
    d0 = pd.Timestamp(be.FORWARD_START)
    live = real[(real["Date"] == d0) & (real["Strategy_Weight"] > 0)].set_index("Symbol")["Strategy_Weight"].sort_index()
    mine = ft.strategy_targets(ft.STRATEGIES[0], real_inp).loc[d0]
    mine = mine[mine > 0].sort_index()
    check("live rules recomputed from the saved files = the live Strategy_Weight of Oct 2 (apples to apples)",
          list(live.index) == list(mine.index) and np.allclose(live, mine, atol=1e-9), (live.to_dict(), mine.to_dict()))
    check("no live file changed (signal_analysis, picks, run_state, pending orders, orders log); WINNER untouched",
          digest() == before and be.WINNER == w0)
src = open(os.path.join(ROOT, "forward_test.py"), encoding="utf-8").read()
check("forward_test.py has no order code and never writes the live files or WINNER",
      not any(w in src for w in ("submit_order", "cancel_order", "OrderRequest", "TradingClient", "import paper_trade",
                                 "requests.post", "run_state", "live_pending_orders", "WINNER.update", "WINNER[")))
print(f"\n{len(FAIL)} failed" if FAIL else "\nFORWARD TEST OK")
sys.exit(1 if FAIL else 0)
