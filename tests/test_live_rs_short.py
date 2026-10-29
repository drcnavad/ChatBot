"""Live rules from Tue Oct 27, 2026 (Chirag): 100% "RS short-term (weekly + Mon–Thu sells)", exactly its forward_test line
(+ the no-idle-cash fill). Mocked: no broker calls, the ledger is a temp file."""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ["STOCK_ANALYSIS_SLEEVE_LEDGER"] = os.path.join(tempfile.mkdtemp(), "sleeve_ledger.json")
import pandas as pd  # noqa: E402

import backtest_engine as be  # noqa: E402
import forward_test as ft  # noqa: E402
import paper_trade as pt  # noqa: E402
import sleeves  # noqa: E402

FAIL = []
def check(ok, what, got=""):
    print(("PASS " if ok else "FAIL ") + what + ("" if ok else f"  ({got})"))
    if not ok:
        FAIL.append(what)

W = be.WINNER
check(W.get("halves") and W.get("sleeves") == {"rs_short": 1.0}, "WINNER = one sleeve, RS short-term 100%", W.get("sleeves"))
check(be.winner_label(90)[0] == "RS-S-U90" and be.winner_label(90)[1].startswith("Live: RS short-term"), "tag / name", be.winner_label(90))
names = {c["name"] for c in ft.STRATEGIES}
check(ft.LIVE == sleeves.STRATEGY["rs_short"] and {ft.LIVE, "RS standard (Fridays only)", ft.BLEND} <= names,
      "the live line is RS short-term; RS Fridays and the 50/50 stay as paper lines")
check(pt.MAX_ORDER_PCT == 0.20, "order backstop = 20% of equity")
check(sleeves.LEDGER_JSON.startswith(tempfile.gettempdir()) or "tmp" in sleeves.LEDGER_JSON, "test ledger is a temp file")

days = pd.date_range("2026-10-26", "2026-12-04")
kinds = {d.date().isoformat(): be.decision_kind(d) for d in days if be.decision_kind(d)}
fulls = sorted(k for k, v in kinds.items() if v == "full rebalance")
check(fulls[:6] == ["2026-10-30", "2026-11-06", "2026-11-13", "2026-11-20", "2026-11-27", "2026-12-04"],
      "full rebalance every Friday (Thanksgiving week: Fri Nov 27)", fulls)
check(kinds.get("2026-10-27") == "mid-week check" and "2026-10-31" not in kinds, "Mon-Thu = exit check, weekends nothing", kinds)

inp = sleeves.inputs()
d = inp["close"].index[-1]
px = {s: float(inp["close"].at[d, s]) for s in inp["close"].columns if pd.notna(inp["close"].at[d, s])}
orders, info = sleeves.plan(d, {}, 100000.0, px, ledger=None, inp=inp)
check(abs(info["values"]["rs_short"] - 100000) < 1, "switch: the sleeve = the whole equity", info["values"])
w = sleeves.half_weights("rs_short", d, inp)
# Oct 29, 2026: live line has no defensive sleeve — top 10 only (~9.9% each after 99% scale)
import sector_mapping as sm
cfg = next(c for c in ft.STRATEGIES if c["name"] == ft.LIVE)
check(cfg.get("sleeve") is False, "live RS short-term sleeve=False (no defensive)", cfg.get("sleeve"))
forced = [s for s in sm.defensive if s in w.index and float(w[s]) > 0]
# a defensive name may still appear if it ranks in the top 10 on its own
r = sleeves.ranking("rs_short", d, inp)
top10 = set(r.index[:10])
check(all(s in top10 for s in forced), "defensive names in targets only if they rank top 10", forced)
check(len(w[w > 0]) <= 10 and abs(float(w[w > 0].sum()) - 0.99) < 0.02,
      "at most 10 targets, ~99% invested", (len(w[w > 0]), float(w[w > 0].sum())))
buys = orders[orders["Side"] == "BUY"].set_index("Symbol")
check(set(buys.index) == set(w[w > 0].index) and all(abs(buys.at[s, "Est_Value"] - 100000 * w[s]) < 0.02 * 100000 * 0.15
      for s in buys.index), "buys = the forward-test weights x equity", buys["Est_Value"].to_dict())
led = info["ledger"]
check(led["version"] == 4 and list(led["halves"]) == ["rs_short"] and led["halves"]["rs_short"]["cash"] >= -1, "v4 ledger, not overspent", led["halves"])
# do_not_sell / do_not_buy
o2, _ = sleeves.plan(d, {"ZZZ": 3.0}, 100000.0, {**px, "ZZZ": 10.0}, ledger=None, inp=inp, no_sell={"ZZZ": "x"})
check(o2.set_index("Symbol").at["ZZZ", "Side"] == "HOLD", "do_not_sell: an unclaimed holding is kept")
blk = next(iter(buys.index))
o3, _ = sleeves.plan(d, {}, 100000.0, px, ledger=None, inp=inp, no_buy={blk: "x"})
check(blk not in set(o3[o3["Side"] == "BUY"]["Symbol"]), "do_not_buy: never bought")

# no idle cash (Oct 26, 2026): a blocked target's weight goes to the half's other buys, pro rata; total = the strategy's own
o4, i4 = sleeves.plan(d, {}, 100000.0, px, ledger=None, inp=inp, no_buy={blk: "x"})
b4 = o4[o4["Side"] == "BUY"]["Est_Value"].sum()
check(abs(b4 - orders[orders["Side"] == "BUY"]["Est_Value"].sum()) < 300, "blocked pick: its cash is spread over the other buys", b4)
tw = sleeves.fill_idle({"A": 0.5, "B": 0.25, "C": 0.24}, {}, {"A": True, "B": True, "C": True}, {"C": "x"}, {"A": 1, "B": 1, "C": 1}, 0.99)
check(abs(tw["A"] - 0.66) < 1e-9 and abs(tw["B"] - 0.33) < 1e-9 and "C" not in tw, "fill_idle: pro rata, blocked dropped", tw)
tw = sleeves.fill_idle({"A": 0.2, "B": 0.2}, {}, {"A": True, "B": True}, {}, {"A": 1, "B": 1}, 0.495)
check(abs(sum(tw.values()) - 0.495) < 1e-9, "fill_idle: never above the strategy's own total (QQQ halving kept)", tw)

print("LIVE RS SHORT-TERM OK" if not FAIL else f"LIVE RS SHORT-TERM FAILURES ({len(FAIL)}): {FAIL}")
sys.exit(1 if FAIL else 0)
