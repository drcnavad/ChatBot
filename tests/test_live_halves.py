"""Live rules from Tue Oct 27, 2026 (Chirag): 50% "RS short-term (weekly + Mon–Thu sells)" + 50% "RS standard (Fridays only)",
each exactly its forward_test line; a stock in both halves = one combined position (no overlap rule). Mocked: no broker
calls, the ledger is a temp file."""
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
check(W.get("halves") and W.get("sleeves") == {"rs_short": 0.5, "friday": 0.5}, "WINNER = two 50% halves", W.get("sleeves"))
check(be.winner_label(90)[0] == "RS-H-U90" and "RS short-term + RS Fridays" in be.winner_label(90)[1], "tag / name", be.winner_label(90))
names = {c["name"] for c in ft.STRATEGIES}
check(set(sleeves.STRATEGY.values()) <= names and ft.LIVE in names, "both halves and the live blend are forward-test lines")
check(pt.MAX_ORDER_PCT == 0.20, "order backstop = 20% of equity")
check(sleeves.LEDGER_JSON.startswith(tempfile.gettempdir()) or "tmp" in sleeves.LEDGER_JSON, "test ledger is a temp file")

# calendar: every session decides; the week's last session is the full rebalance of both halves
days = pd.date_range("2026-10-26", "2026-12-04")
kinds = {d.date().isoformat(): be.decision_kind(d) for d in days if be.decision_kind(d)}
fulls = sorted(k for k, v in kinds.items() if v == "full rebalance")
check(fulls[:6] == ["2026-10-30", "2026-11-06", "2026-11-13", "2026-11-20", "2026-11-27", "2026-12-04"],
      "full rebalance every Friday (Thanksgiving week: Fri Nov 27)", fulls)
check(kinds.get("2026-10-27") == "mid-week check" and "2026-10-31" not in kinds, "Mon-Thu = mid-week check, weekends nothing", kinds)
check(sleeves.half_day("rs_short", "2026-10-28") and not sleeves.half_day("friday", "2026-10-28")
      and sleeves.half_day("friday", "2026-10-30"), "RS Fridays decides only on Fridays; RS short-term every session")

# plan: switch run from cash, synthetic inputs
inp = sleeves.inputs()
d = inp["close"].index[-1]
px = {s: float(inp["close"].at[d, s]) for s in inp["close"].columns if pd.notna(inp["close"].at[d, s])}
orders, info = sleeves.plan(d, {}, 100000.0, px, ledger=None, inp=inp)
check(abs(info["values"]["rs_short"] - 50000) < 1 and abs(info["values"]["friday"] - 50000) < 1, "switch: each half = 50%", info["values"])
w = {h: sleeves.half_weights(h, d, inp) for h in sleeves.HALVES}
both = set(w["rs_short"][w["rs_short"] > 0].index) & set(w["friday"][w["friday"] > 0].index)
buys = orders[orders["Side"] == "BUY"].set_index("Symbol")
ok = all(abs(buys.at[s, "Est_Value"] - 50000 * (w["rs_short"].get(s, 0) + w["friday"].get(s, 0))) < 0.02 * 50000 * 0.15
         for s in buys.index)
check(ok and set(buys.index) == set(w["rs_short"][w["rs_short"] > 0].index) | set(w["friday"][w["friday"] > 0].index),
      "buys = sum of each half's forward-test weight x its half (shared stock combined)", buys["Est_Value"].to_dict())
check(not both or all(buys.at[s, "Half"] == "rs_short+friday" for s in both if s in buys.index), "shared stocks tagged to both")
led = info["ledger"]
check(led["version"] == 3 and all(led["halves"][h]["cash"] >= -1 for h in sleeves.HALVES), "v3 ledger, no half overspent", led["halves"])
# do_not_sell / do_not_buy
o2, _ = sleeves.plan(d, {"ZZZ": 3.0}, 100000.0, {**px, "ZZZ": 10.0}, ledger=None, inp=inp, no_sell={"ZZZ": "x"})
check(o2.set_index("Symbol").at["ZZZ", "Side"] == "HOLD", "do_not_sell: an unclaimed holding is kept")
blk = next(iter(buys.index))
o3, _ = sleeves.plan(d, {}, 100000.0, px, ledger=None, inp=inp, no_buy={blk: "x"})
check(blk not in set(o3[o3["Side"] == "BUY"]["Symbol"]), "do_not_buy: never bought")

print("LIVE HALVES OK" if not FAIL else f"LIVE HALVES FAILURES ({len(FAIL)}): {FAIL}")
sys.exit(1 if FAIL else 0)
