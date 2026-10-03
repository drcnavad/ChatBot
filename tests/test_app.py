"""App regression (Streamlit AppTest, no browser, no network): the page renders without exceptions, HTML in markdown renders
as HTML (no escaped tags / code blocks / unbalanced tags), the price chart draws Close + all MAs on a date axis, the displayed
rank == Strategy_Rank and the portfolio slot == position among the picks, the dated decision badge / "Strategy rank ·
<latest close>" / "<rebalance day> plan" chip agree with strategy_changes / signal_analysis / strategy_picks (and with the
"Latest signals" and "Last decision" tables in Details), the Strategy tab widgets work, the rules text matches
backtest_engine.WINNER, removed sections stay gone (incl. the alert banner, the Summary section and the Strategy Health tab), and the rank tiers render with a ?symbol= link
per ranked stock.
Run: python tests/run_tests.py  (or python tests/test_app.py)"""
import base64
import json
import os
import re
import sys
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.environ["STOCK_ANALYSIS_LIVE_HOLDINGS"] = "off"   # never call Alpaca here (tests/test_live_holdings.py fakes it)
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
from markdown_it import MarkdownIt
from streamlit.testing.v1 import AppTest

import backtest_engine as be
import sector_mapping as sm

FAIL = []
MD = MarkdownIt("commonmark", {"html": True})
VOID = {"br", "img", "hr", "meta", "link", "input"}
MA_NAMES = ["MA 10", "MA 30", "MA 50", "MA 100", "MA 200"]
_FIRST_PER_SECTOR = {}
for _s in sm.tradable_symbols:                 # from the stock list: the first stock of each sector (up to 7) + the benchmark
    _FIRST_PER_SECTOR.setdefault(sm.symbol_sector.get(_s), _s)
TICKERS = list(_FIRST_PER_SECTOR.values())[:7] + ["QQQ"]
SIG = pd.read_csv("Reports/signal_analysis.csv", parse_dates=["Date"])
_SH = "Reports/short_history_reference.csv"     # stocks in the list with < 200 days (reference only, never ranked)
SHORT = pd.read_csv(_SH) if os.path.exists(_SH) else pd.DataFrame(columns=["Symbol", "As_Of"])
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
opts = [o.split("  ·  ")[0] for o in at.selectbox(key="ticker_dropdown").options]  # labels are "SYM · rank · score"; compare raw symbols
_short_now = set(SHORT["Symbol"]) & set(sm.tradable_symbols)
expect(len(opts) == len(sm.tradable_symbols) + 1, f"dropdown has {len(opts)} options")   # ranked stocks + QQQ + short-history ones
expect(opts[-len(SHORT):] == SHORT["Symbol"].tolist() if len(SHORT) else True, "short-history stocks should close the dropdown")
expect(not (_short_now & set(SIG.Symbol)), f"short-history stocks must never be scored/ranked: {_short_now & set(SIG.Symbol)}")
for sym in TICKERS:
    at = at.selectbox(key="ticker_dropdown").set_value(sym).run()
    page_ok(at, sym)
    price_chart_ok(at, sym)
    hero = next((m.value for m in at.markdown if 'class="sa-hero"' in m.value), "")
    rk = re.search(r"Strategy rank.*?sa-stat-val[^>]*>([^<]*)<", hero)
    sl = re.search(r"Portfolio slot.*?sa-stat-val[^>]*>([^<]*)<", hero)
    r = latest.loc[sym, "Strategy_Rank"]
    want_rk = f"#{r:.0f} / {int(latest['Strategy_Rank'].notna().sum())}" if pd.notna(r) else "—"
    want_sl = f"{exp_slot[sym]} of 10" if sym in exp_slot else "— (not picked)"
    expect(rk and sl and rk.group(1) == want_rk and sl.group(1) == want_sl,
           f"{sym}: header rank/slot {rk and rk.group(1)!r}/{sl and sl.group(1)!r}, expected {want_rk!r}/{want_sl!r}")

# ---------------------------------------------------------------- decision badge / rank label / plan chip / Details tables agree
# decision in force = strategy_changes.csv (what the app reads); today's rank = latest signal_analysis; plan = strategy_picks
CH = pd.read_csv("Reports/strategy_changes.csv", parse_dates=["Date"])
CH = CH[CH.Symbol.notna()]
dec_day = CH.Date.iloc[0]
PK = pd.read_csv("Reports/strategy_picks.csv", parse_dates=["As_Of", "Last_Rebalance"])
as_of = PK.As_Of.iloc[0]
plan_day = as_of if as_of == PK.Last_Rebalance.iloc[0] else None
if plan_day is None:
    d = as_of
    for _ in range(6):
        d, kind, _f = be.next_decision(d)
        if kind == "full rebalance":
            plan_day = d; break
prev = SIG[SIG.Date < plan_day]; prev = prev[prev.Date == prev.Date.max()]
held_into = set(prev.loc[prev.Strategy_Weight.fillna(0) > 0, "Symbol"])
pw = PK.set_index("Symbol").Provisional_Weight.fillna(0)
def want_plan(sym):
    w = pw.get(sym, 0.0) * 100
    return (f"Keep {w:.2f}%" if sym in held_into else f"Buy {w:.2f}%") if w > 0 else ("Sell" if sym in held_into else "not picked")
WORD = {"add": "Buy", "hold": "Hold", "drop": "Sold"}
st_ = CH.set_index("Symbol").Status
picks_for = {"drop": CH[CH.Status == "drop"].Symbol.tolist(), "hold": CH[CH.Status == "hold"].Symbol.tolist()}
cases = (["FTNT"] if "FTNT" in st_.index else []) + picks_for["drop"][:1] + picks_for["hold"][:1] \
    + [s for s in pw.index if pw[s] > 0 and s not in held_into][:1]
fmt = lambda d: f"{d:%a %b} {d.day}"
for sym in dict.fromkeys(cases):
    at = at.selectbox(key="ticker_dropdown").set_value(sym).run()
    hero = next((m.value for m in at.markdown if 'class="sa-hero"' in m.value), "")
    badges = [re.sub("<[^>]+>", "", b) for b in re.findall(r'<span class="sa-badge[^"]*"[^>]*>[^<]*</span>', hero)]
    word = WORD.get(st_.get(sym), None)
    want_badge = f"Strategy decision · {fmt(dec_day)}: " + (word or "")
    expect(bool(badges) and badges[0].startswith(want_badge), f"{sym}: badge {badges[:1]} should start {want_badge!r}")
    want_chip = f"{fmt(plan_day)} plan: {want_plan(sym)}"
    expect(want_chip in badges, f"{sym}: plan chip {badges[1:]} != {want_chip!r}")
    rk = re.search(r"Strategy rank · ([^<ⓘ]*?)\s*ⓘ?</div><div class=\"sa-stat-val\"[^>]*>([^<]*)<", hero)
    r = latest.loc[sym, "Strategy_Rank"]
    expect(rk and rk.group(1) == f"{fmt(SIG.Date.max())} close" and rk.group(2).startswith(f"#{r:.0f} /"),
           f"{sym}: rank label {rk and rk.groups()} vs latest {fmt(SIG.Date.max())} rank {r}")
    tabs = {tuple(d.value.columns): d.value for d in at.dataframe}
    lt = next((v for c, v in tabs.items() if "Signal today" in c), None)
    ld = next((v for c, v in tabs.items() if any(x.startswith("Rank at ") for x in c)), None)
    expect(lt is not None and ld is not None, f"{sym}: Latest signals / Last decision tables missing")
    if lt is not None and ld is not None:
        a = lt.set_index("Symbol").loc[sym]
        expect(int(a["Rank today"]) == int(r) and a[f"{fmt(plan_day)} plan"] == want_plan(sym)
               and abs(a["Score today"] - latest.loc[sym, "Strategy_Score"]) < 0.06,
               f"{sym}: Latest signals row {a.to_dict()} disagrees")
        rank_then = f"Rank at {dec_day:%b} {dec_day.day} decision"
        ld = ld.set_index("Symbol")
        b = ld.loc[sym] if sym in ld.index else None          # the default filter shows the portfolio & changes only
        expect(b is None and word is None or b is not None and int(b["Rank today"]) == int(r) and b["Next rebalance plan"] == want_plan(sym) and b["Signal"] == word
               and int(b[rank_then]) == int(CH.set_index("Symbol").Rank[sym]),
               f"{sym}: Last decision row {b.to_dict()} disagrees")
    print(f"   {sym}: {badges} · rank {rk and rk.groups()} OK")
exp_titles = ["Live holdings (Alpaca account)", f"Latest signals · {fmt(SIG.Date.max())} close", f"Last decision · {fmt(dec_day)}"]
labels = [e.label for e in at.expander]
expect(labels[:2] == exp_titles[:2] and any(l.startswith(exp_titles[2]) for l in labels),
       f"Details expanders {labels[:3]} should start with {exp_titles}")
# short history: no Details section any more; the stocks sit in the clickable stock list (see the rank tiers test)
expect(not any(l.startswith("Reference only") for l in labels), f"stale short-history Details section: {labels}")
lt_all = next((d.value for d in at.dataframe if "Signal today" in d.value.columns), pd.DataFrame())
expect(set(lt_all.get("Symbol", [])) == set(latest.index), "Latest signals should list every stock of the latest close")

# ---------------------------------------------------------------- Strategy tab widgets and captions
# the "if rebalanced at latest close" / "if the week ended today" preview views were removed from the app
# (the pipeline runs on decision days only, so there is nothing hypothetical left to show)
expect(not any(r.key == "changes_view" for r in at.radio), "stale changes_view preview radio still present")
expect(not any(r.key == "signals_view" for r in at.radio), "stale signals_view preview radio still present")
at = at.radio(key="signals_filter").set_value("All stocks").run(); page_ok(at, "signals_filter all")
# ONE decision view: the old "What changed at the latest decision" table (rules area) is merged into "Last decision"
expect(not any(c.key == "changes_all" for c in at.checkbox), "stale 'What changed' checkbox still present")
expect(not any("What changed at the latest decision" in m.value for m in at.markdown), "duplicate 'What changed' view is back")
why = [d.value for d in at.dataframe if "Why" in d.value.columns]
expect(len(why) == 1 and {"Weight before %", "Portfolio weight %", "Next rebalance plan"} <= set(why[0].columns)
       and len(why[0]) == len(latest), f"one decision table with every stock and before/after weights: {[list(w.columns) for w in why]}")
dec_labels = [e.label for e in at.expander if "decision" in e.label.lower()]
expect(len(dec_labels) == 1 and dec_labels[0].startswith(f"Last decision · {fmt(dec_day)}"), f"decision expanders: {dec_labels}")
expect([e.label for e in at.expander][-1] == "Strategy rules", f"the rules must be the last Details section: {[e.label for e in at.expander]}")
# the Holdings risk and Order preview expanders were removed from the Details tab
expect(not any(t.key == "order_positions" for t in at.text_area), "stale order preview positions box still present")
expect(not any(s.key == "order_target" for s in at.selectbox), "stale order preview target selectbox still present")
expect(not any(e.label in ("Holdings risk", "Order preview (nothing is sent)") for e in at.expander),
       "stale Holdings risk / Order preview expander still present")
# the Backtest results expander (and its bt_segment selectbox) was removed from the app
# (live Alpaca P&L is the real number now; the backtest was a research artifact)
expect(not any(s.key == "bt_segment" for s in at.selectbox), "stale bt_segment selectbox still present")
expect(any(m.label == "Last mid-week check" for m in at.metric), "metric 'Last mid-week check' missing")
blob = "\n".join(str(t.value) for t in list(at.markdown) + list(at.caption))
mwc = pd.read_csv("Reports/strategy_midweek_check.csv")
expect("run_pipeline" not in blob, "old command 'run_pipeline' still shown in the app")
# the Rank history expander was removed
expect(not any(e.label.startswith("Rank history") for e in at.expander), "stale Rank history expander still present")
# the rules shown match the live config (backtest_engine.WINNER), in one place
W, mw = be.WINNER, be.WINNER["midweek_swap"]
rules = next((m.value for m in at.markdown if m.value.startswith("**Strategy rules")), "")
for want in (f"Strategy rules ({W['tag']})", f"{W['w_tech']:g} × Technical", f"at least {be.MIN_BARS} trading days",
             f"the {W['n']} best-ranked stocks from ranks 1–{W['max_pick_rank']}", f"max {be.winner_max_per_sector()} per sector",
             f"scaled to {be.LIVE_INVESTED:.0%} invested", f"{W['regime_symbol']} at or below its 200-day average",
             "every weight is halved" if W["regime_scale"] == 0.5 else "every weight is multiplied",
             f"within {W['rebalance_band'] * 100:g} percentage point", f"top {mw['enter_top']}", f"below rank {mw['exit_below']}",
             f"worse than {W['midweek_exit_below']}", f"within {W['earnings_block_days']} calendar days", "after-hours limit orders",
             "9 AM CT", "2-decimal shares", "never runs twice"):
    expect(want in rules, f"rules text is missing {want!r}")
expect("next open" not in blob, "outdated 'next open' wording (orders go out the same evening)")

# ---------------------------------------------------------------- removed: alert banner, Summary section, Strategy Health tab
box = [m.value for m in at.markdown if "sa-alert " in m.value]
expect(len(box) == 0, f"the alert banner was removed from the dashboard, found {len(box)}")
expect([t.label for t in at.tabs] == ["Dashboard", "Details"], f"tabs: {[t.label for t in at.tabs]} (Strategy Health removed)")
expect(not [m for m in at.markdown if '<div class="sa-section">Summary</div>' in m.value], "the Summary section was removed")
expect(any("Live holdings are turned off here" in i.value for i in at.info), "holdings expander: off message in tests")

# ---------------------------------------------------------------- rank tiers: horizontal clickable lists by rank
tier_md = [m.value for m in at.markdown if "Rank 1 to 20" in m.value and "?symbol=" in m.value]
expect(len(tier_md) == 1, f"rank tier block found: {len(tier_md)}")
blob = tier_md[0] if tier_md else ""
parts = re.split(r"<b>(Rank 1 to 20|Rank 21 to 50|Rank 51\+|Not traded yet \(short history, not ranked\)):</b>", blob)
expect(len(parts) == 7 + 2 * bool(len(SHORT)), f"tier headers/bodies: {len(parts)} parts")
bodies = dict(zip(parts[1::2], parts[2::2]))
def want_tier(sym):
    r = latest.loc[sym, "Strategy_Rank"]
    return "Rank 1 to 20" if r <= 20 else ("Rank 21 to 50" if r <= 50 else "Rank 51+")
for sym in latest[latest.Strategy_Rank.notna()].index:
    syms = set(re.findall(r"\?symbol=([A-Z0-9.]+)", bodies[want_tier(sym)]))
    expect(sym in syms, f"{sym} (rank {latest.loc[sym, 'Strategy_Rank']:.0f}) not in its tier {want_tier(sym)!r}")
all_linked = set(re.findall(r"\?symbol=([A-Z0-9.]+)", blob))
expect(all_linked <= set(opts), f"tier links outside the dropdown: {sorted(all_linked - set(opts))}")

# short-history stocks: listed with their rough signal, and a click (?symbol=) opens a stock view that renders
if len(SHORT):
    sh_body = bodies.get("Not traded yet (short history, not ranked)", "")
    for _, r in SHORT.iterrows():
        expect(f"?symbol={r.Symbol}" in sh_body and f"(rough {r.Rough_Signal}, less reliable)" in sh_body,
               f"{r.Symbol} missing from the short-history row: {sh_body[:200]}")
        at2 = AppTest.from_file("app.py", default_timeout=180)
        at2.query_params["symbol"] = r.Symbol
        at2.run()
        page_ok(at2, f"click {r.Symbol}")
        expect(at2.selectbox(key="ticker_dropdown").value == r.Symbol, f"{r.Symbol}: click did not open the stock")
        warn = " ".join(w.value for w in at2.warning)
        expect(f"Not traded yet (short history): {int(r.Days_Of_History)} of" in warn, f"{r.Symbol}: short-history note missing")
        mets = {m.label: m.value for m in at2.metric}
        expect(mets.get("Rough signal (less reliable)") == r.Rough_Signal and "RSI 14" in mets and "MA 50" in mets,
               f"{r.Symbol}: stock view metrics {mets}")
        expect(not any(d for d in at2.dataframe if "Symbol" in d.value.columns and r.Symbol in set(d.value["Symbol"])),
               f"{r.Symbol} shows up in a table (it must appear in the stock list only)")
        charts = [json.loads(c.proto.spec) for c in at2.get("plotly_chart")]
        print(f"   {r.Symbol} click OK · metrics {mets} · price chart {'yes' if any(any(t.get('name') == 'Close' for t in f['data']) for f in charts) else 'unavailable (yfinance)'}")

print("\nAPP TESTS OK" if not FAIL else f"\nAPP TEST FAILURES ({len(FAIL)}): {FAIL}")
sys.exit(1 if FAIL else 0)
