"""App regression (Streamlit AppTest, no browser, no network): the page renders without exceptions, HTML in markdown renders
as HTML (no escaped tags / code blocks / unbalanced tags), the price chart draws Close + all MAs on a date axis, the displayed
rank == Strategy_Rank and the portfolio slot == position among the picks, the Strategy tab widgets work, captions name the live
rules, and the holdings alert renders (strategy holdings + a temporary positions file, deleted afterwards).
Run: python tests/run_tests.py  (or python tests/test_app.py)"""
import base64
import io
import json
import os
import re
import sys
import tempfile
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
from markdown_it import MarkdownIt
from streamlit.testing.v1 import AppTest

import backtest_engine as be
import holdings_alert
import sector_mapping as sm

FAIL = []
MD = MarkdownIt("commonmark", {"html": True})
VOID = {"br", "img", "hr", "meta", "link", "input"}
MA_NAMES = ["MA 10", "MA 30", "MA 50", "MA 100", "MA 200"]
TICKERS = ["MRK", "AAPL", "QQQ", "APA", "TRGP", "CRDO", "RBRK", "COF"]   # original 78, U91 and U96 names + the benchmark
SIG = pd.read_csv("Reports/signal_analysis.csv", parse_dates=["Date"])
DEC = pd.read_csv("Reports/strategy_decisions.csv", parse_dates=["Date"])


def expect(ok, what):
    if not ok:
        FAIL.append(what)
        print("   !! FAIL:", what)


class Balance(HTMLParser):
    def __init__(self):
        super().__init__(); self.stack, self.errors = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}>")
            while tag in self.stack and self.stack.pop() != tag:
                pass
        else:
            self.stack.pop()


def html_problems(src):
    if "<" not in src or not re.search(r"<(div|span|table|tr|td|a|p|h1|style)\b", src):
        return []
    out, probs = MD.render(src), []
    if re.search(r"&lt;/?(div|span|table|tr|td|th|a|p|h1)\b", out):
        probs.append("HTML tag rendered as text")
    if "<pre><code>" in out and "<pre" not in src:
        probs.append("indented HTML became a code block")
    if "<style" not in src:
        lines = src.strip().splitlines()
        if any(not l.strip() for l in lines):
            probs.append("blank line inside HTML")
        b = Balance(); b.feed(src); b.close()
        if b.errors or b.stack:
            probs.append(f"unbalanced tags {b.errors[:2]} open={b.stack[:3]}")
    return probs


def page_ok(at, label):
    exc = [str(e.value)[:300] for e in at.exception]
    bad = [(p, m.value[:80]) for m in list(at.markdown) + list(at.caption) for p in html_problems(m.value)]
    print(f"[{label}] exceptions={len(exc)} html_problems={len(bad)} tabs={len(at.tabs)} dataframes={len(at.dataframe)}")
    for x in exc + bad[:3]:
        print("   ", x)
    expect(not exc and not bad, f"{label}: exceptions/HTML problems")


def _arr(v):
    if isinstance(v, dict) and "bdata" in v:
        return np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v["dtype"])).astype(float)
    return np.array([np.nan if x is None else x for x in (v or [])], dtype=object)


def _finite(v):
    a = _arr(v)
    return int(sum(1 for x in a if x is not None and not (isinstance(x, float) and np.isnan(x))))


def price_chart_ok(at, sym):
    specs = [json.loads(c.proto.spec) for c in at.get("plotly_chart")]
    price = [f for f in specs if any(t.get("name") == "Close" for t in f["data"])]
    if not price:
        return expect(False, f"{sym}: no price chart")
    f = price[0]
    on_x = {t.get("name"): t for t in f["data"] if t.get("xaxis", "x") == "x" and t.get("yaxis", "y") == "y"}
    probs = [n for n in ["Close"] + MA_NAMES if n not in on_x or _finite(on_x[n].get("y")) == 0 or _finite(on_x[n].get("x")) == 0]
    if f["layout"].get("xaxis", {}).get("type") != "date":
        probs.append("x-axis not a date axis")
    expect(not probs, f"{sym}: price chart problems {probs}")


# ---------------------------------------------------------------- page, tickers, charts, displayed rank / slot
latest = SIG[SIG.Date == SIG.Date.max()].set_index("Symbol")
off_day = DEC.Date.max()
sel = DEC[(DEC.Date == off_day) & DEC.Status.isin(["add", "hold"])].sort_values("Rank")
exp_slot = {s: i + 1 for i, s in enumerate(sel.Symbol)}
at = AppTest.from_file("app.py", default_timeout=180).run()
page_ok(at, "initial")
opts = at.selectbox(key="ticker_dropdown").options
expect(len(opts) == len(sm.tradable_symbols) + 1, f"dropdown has {len(opts)} options")
for sym in TICKERS:
    at = at.selectbox(key="ticker_dropdown").set_value(sym).run()
    page_ok(at, sym)
    price_chart_ok(at, sym)
    hero = next((m.value for m in at.markdown if 'class="sa-hero"' in m.value), "")
    rk = re.search(r"Rank today.*?sa-stat-val[^>]*>([^<]*)<", hero)
    sl = re.search(r"Portfolio slot.*?sa-stat-val[^>]*>([^<]*)<", hero)
    r = latest.loc[sym, "Strategy_Rank"]
    want_rk = f"#{r:.0f} / {int(latest['Strategy_Rank'].notna().sum())}" if pd.notna(r) else "—"
    want_sl = f"{exp_slot[sym]} of 10" if sym in exp_slot else "— (not picked)"
    expect(rk and sl and rk.group(1) == want_rk and sl.group(1) == want_sl,
           f"{sym}: header rank/slot {rk and rk.group(1)!r}/{sl and sl.group(1)!r}, expected {want_rk!r}/{want_sl!r}")

# Rank tab: newest column == Strategy_Rank; slots == picks
html_tbl = next((m.value for m in at.markdown if "<thead>" in m.value and "Trend" in m.value), None)
if html_tbl:
    rt = pd.read_html(io.StringIO(html_tbl))[0]
    col = SIG.Date.max().strftime("%m/%d")
    x = rt.set_index("Symbol")[col].dropna().astype(int)
    expect((x - latest["Strategy_Rank"].reindex(x.index)).abs().sum() == 0, "Rank tab column != Strategy_Rank")
    ts = rt[rt.Slot.astype(str) != "—"].set_index("Symbol").Slot.astype(int).to_dict()
    expect(ts == exp_slot, f"Rank tab slots {ts} != {exp_slot}")
else:
    expect(False, "Rank tab table not found")

# ---------------------------------------------------------------- Strategy tab widgets and captions
# the "if rebalanced at latest close" / "if the week ended today" preview views were removed from the app
# (the pipeline runs on decision days only, so there is nothing hypothetical left to show)
expect(not any(r.key == "changes_view" for r in at.radio), "stale changes_view preview radio still present")
expect(not any(r.key == "signals_view" for r in at.radio), "stale signals_view preview radio still present")
at = at.checkbox(key="changes_all").check().run(); page_ok(at, "changes_all")
at = at.text_area(key="order_positions").input("MRK,10\nAMD,5\nBAD LINE").run(); page_ok(at, "order preview positions")
for tgt in ("midweek", "auto"):
    at = at.selectbox(key="order_target").set_value(tgt).run(); page_ok(at, f"order target {tgt}")
# the Backtest results expander (and its bt_segment selectbox) was removed from the app
# (live Alpaca P&L is the real number now; the backtest was a research artifact)
expect(not any(s.key == "bt_segment" for s in at.selectbox), "stale bt_segment selectbox still present")
expect(any(m.label == "Last mid-week check" for m in at.metric), "metric 'Last mid-week check' missing")
blob = "\n".join(str(t.value) for t in list(at.markdown) + list(at.caption))
mwc = pd.read_csv("Reports/strategy_midweek_check.csv")
expect("run_pipeline" not in blob, "old command 'run_pipeline' still shown in the app")

# ---------------------------------------------------------------- holdings alert (strategy + temporary positions file)
box = [m.value for m in at.markdown if "sa-alert " in m.value]
expect(len(box) == 1, f"alert boxes on the page: {len(box)}")
a = holdings_alert.build_alert(use_positions=False)
print("alert (strategy):", " | ".join(holdings_alert.alert_text(a, urls=False)))
held = latest[latest.Strategy_Weight.fillna(0) > 0].index.tolist()
with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
    f.write("Symbol,Shares\n" + "\n".join(f"{s},10" for s in held[:-1] + ["PLTR"]) + "\n")
try:
    b = holdings_alert.build_alert(positions_path=f.name)
    print("alert (temp positions, last holding replaced by off-universe PLTR):", " | ".join(holdings_alert.alert_text(b, urls=False)))
    expect(b["uses_positions"] and b["lines"], "positions-file alert empty")
    if holdings_alert.exit_below():               # the off-universe PLTR is never ranked -> the latest check calls for selling it
        expect("PLTR" in " ".join(holdings_alert.alert_text(b, urls=False)), "rank-exit alert for an unranked holding missing")
finally:
    os.remove(f.name)

# ---------------------------------------------------------------- alert on a historical check day with a rank exit
if holdings_alert.exit_below():
    dec = pd.read_csv("Reports/strategy_decisions.csv", parse_dates=["Date"])
    ex = dec[dec.Reason.astype(str).str.startswith("mid-week exit")]
    sig_all = pd.read_csv("Reports/signal_analysis.csv", usecols=["Date", "Symbol", "Close", "Strategy_Score", "Strategy_Rank",
                                                                  "Strategy_Weight", "Rebalance_Day", "Midweek_Check"], parse_dates=["Date"])
    for D in sorted(ex.Date.unique())[-2:]:
        c = holdings_alert.build_alert(sig=sig_all[sig_all.Date <= D], use_positions=False, now=pd.Timestamp(D) + pd.Timedelta(hours=20))
        txt = " | ".join(holdings_alert.alert_text(c, urls=False))
        print(f"alert as of the {pd.Timestamp(D).date()} check:", txt)
        want = sorted(ex.loc[ex.Date == D, "Symbol"])
        expect(c["level"] == "red" and all(w in txt for w in want) and "hold cash until" in txt,
               f"exit alert on {pd.Timestamp(D).date()} should sell {want}: {txt}")

# ---------------------------------------------------------------- earnings rule in the alert (fake dates, in memory only)
if be.WINNER.get("earnings_block_days"):
    sig_d = pd.read_csv("Reports/signal_analysis.csv", usecols=["Date", "Symbol", "Close", "Strategy_Score", "Strategy_Rank",
                                                                "Strategy_Weight"], parse_dates=["Date"])
    day_df = sig_d[sig_d.Date == sig_d.Date.max()]
    d0 = holdings_alert._Day(day_df, sm.tradable_symbols)
    top3 = [d0.symbols[j] for j in d0.order[:3]]
    D0 = pd.Timestamp(day_df.Date.iloc[0])
    fake = pd.DataFrame({"Symbol": top3, "Earnings Date": [D0 + pd.Timedelta(days=k) for k in (1, 5, 6)]})
    d1 = holdings_alert._Day(day_df, sm.tradable_symbols, fake)
    weak = {d1.symbols[j]: 0.1 for j in d1.order[40:42]}          # two holdings ranked far below 15
    swaps = holdings_alert.evaluate(d1, weak)[0]
    blocked = holdings_alert.blocked_entrants(d1, weak)
    print("earnings rule (fake dates +1/+5/+6 days for the top 3):", swaps, blocked)
    expect(sorted(d1.symbols[j] for j in d1.earn_days) == sorted(top3[:2]), "earnings window should be d < E <= d + 5")
    expect(all(buy not in top3[:2] for _, buy, _ in swaps), "blocked name swapped in")
    expect(len(blocked) == 2 and "not bought: earnings in 1 day" in blocked[0], f"blocked-entrant text: {blocked}")

print("\nAPP TESTS OK" if not FAIL else f"\nAPP TEST FAILURES ({len(FAIL)}): {FAIL}")
sys.exit(1 if FAIL else 0)
