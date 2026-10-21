"""App regression (Streamlit AppTest, no browser, no network): the page renders without exceptions, HTML in markdown renders
as HTML (no escaped tags / code blocks / unbalanced tags), the price chart draws Close + all MAs on a date axis, the displayed
rank == Strategy_Rank and the portfolio slot == position among the picks, the dated decision badge / "Strategy rank ·
<latest close>" / "<rebalance day> plan" chip agree with strategy_changes / signal_analysis / strategy_picks (and with the
Strategy tab's Last decision table), the Strategy tab widgets work, the rules text matches
backtest_engine.WINNER, removed sections stay gone (incl. the alert banner, the Summary section and the Strategy Health tab), and the stock list renders
rows (Keep / Buy next / Sell next from the preview, else Holdings; Next in line, Others) with a ?symbol= link per stock, each once, do-not-buy
stocks tagged DNB inline; no strategy
score chart panel.
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
import c6_rules  # noqa: E402,F401  the Oct 15 weekly rules this test checks (live: monthly low volatility)
import numpy as np
import pandas as pd
from markdown_it import MarkdownIt
from streamlit.testing.v1 import AppTest

import backtest_engine as be
import sector_mapping as sm
import sleeves

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
    names = {t.get("name") for t in f["data"]}
    expect(not names & {"Strategy score", "Technical", "Strength vs sector/SPY"},
           f"{sym}: the strategy score panel was removed from the Home tab (Oct 16, 2026): {sorted(names & {'Strategy score'})}")

# ---------------------------------------------------------------- page, tickers, charts, displayed rank / slot
latest = SIG[SIG.Date == SIG.Date.max()].set_index("Symbol")
off_day = DEC.Date.max()
sel = DEC[(DEC.Date == off_day) & DEC.Status.isin(["add", "hold"])].sort_values("Rank")
exp_slot = {s: i + 1 for i, s in enumerate(sel.Symbol)}
at = AppTest.from_file("app.py", default_timeout=180).run()
page_ok(at, "initial")
opts = [o.split("  ·  ")[0] for o in at.selectbox(key="ticker_dropdown").options]  # labels are "SYM · rank · score"; compare raw symbols
_short_now = set(SHORT["Symbol"]) & set(sm.tradable_symbols)
_pending = [s for s in sm.tradable_symbols if s not in set(SIG.Symbol) | set(SHORT["Symbol"])]   # added after the last pipeline run
if _pending:
    print(f"   note: {_pending} join the dropdown on the next pipeline run (not in signal_analysis.csv yet)")
expect(len(opts) == len(sm.tradable_symbols) + 1 - len(_pending), f"dropdown has {len(opts)} options")   # ranked + QQQ + short-history
expect(opts[-len(SHORT):] == SHORT["Symbol"].tolist() if len(SHORT) else True, "short-history stocks should close the dropdown")
expect(not (_short_now & set(SIG.Symbol)), f"short-history stocks must never be scored/ranked: {_short_now & set(SIG.Symbol)}")
for sym in TICKERS:
    at = at.selectbox(key="ticker_dropdown").set_value(sym).run()
    page_ok(at, sym)
    price_chart_ok(at, sym)
    # 4ad4ef4 simplified the stock card: score + weight tiles (no rank/slot tiles); the rank sits in the dropdown label
    hero = next((m.value for m in at.markdown if 'class="sa-ident"' in m.value), "")
    sc = re.search(r'Strategy score</div><div class="sa-stat-val"[^>]*>([^<]*)<', hero)
    s_ = latest.loc[sym, "Strategy_Score"]
    want_sc = f"{s_:.1f}" if pd.notna(s_) else "—"
    expect(sc and sc.group(1) == want_sc and "Portfolio weight" not in hero, f"{sym}: header score {sc and sc.group(1)!r}, expected {want_sc!r}; no weight tile")
    r = latest.loc[sym, "Strategy_Rank"]
    lab = next((o for o in at.selectbox(key="ticker_dropdown").options if o.split("  ·  ")[0] == sym), "")
    expect((f"rank #{r:.0f}  ·" in lab) if pd.notna(r) else "rank #" not in lab, f"{sym}: dropdown label {lab!r} vs rank {r}")
    # 4ad4ef4 folded "More about" into the stock card and dropped the per-stock forward test panel (the forward test lives
    # in Details only); the card keeps the moving-average and fundamentals groups, never a backtest panel
    if sym in sm.tradable_symbols:
        expect("Moving averages" in hero and "Fundamentals &amp; news" in hero and "Per-stock backtest" not in hero,
               f"{sym}: stock card detail groups missing")

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
    hero = next((m.value for m in at.markdown if 'class="sa-ident"' in m.value), "")
    badges = [re.sub("<[^>]+>", "", b) for b in re.findall(r'<span class="sa-badge[^"]*"[^>]*>[^<]*</span>', hero)]
    word = WORD.get(st_.get(sym), None)
    want_badge = f"Strategy decision · {fmt(dec_day)}: " + (word or "")
    expect(bool(badges) and badges[0].startswith(want_badge), f"{sym}: badge {badges[:1]} should start {want_badge!r}")
    want_chip = f"{fmt(plan_day)} plan: {want_plan(sym)}"
    expect(want_chip in badges, f"{sym}: plan chip {badges[1:]} != {want_chip!r}")
    r = latest.loc[sym, "Strategy_Rank"]
    lab = next((o for o in at.selectbox(key="ticker_dropdown").options if o.split("  ·  ")[0] == sym), "")
    tiers = next((m.value for m in at.markdown if "Stock list<span>" in m.value), "")
    rk = re.search(r"rank #(\d+)", lab)
    expect(rk and int(rk.group(1)) == int(r) and f"ranks at the {fmt(SIG.Date.max())} close" in tiers,
           f"{sym}: rank label {lab!r} / stock list date vs latest {fmt(SIG.Date.max())} rank {r}")
    rk = (fmt(SIG.Date.max()), rk and rk.group(1))
    print(f"   {sym}: {badges} · rank {rk} OK")
import forward_test as ft  # noqa: E402
# 4ad4ef4 order: Last decision, Forward test, Strategy rules and holdings first; live holdings and latest signals further down
exp_titles = ["Last decision · portfolio and both strategies' ranks",   # Oct 21: no date (each half shows its own)
              f"Forward test · {len(ft.STRATEGIES)} strategies vs your account, QQQ and SPY since Oct 2, 2026",
              "Strategy rules and holdings (paper strategies)",
              "Live holdings (Alpaca account)"]
labels = [e.label for e in at.expander if e.label != "My notes"]   # the notes table sits on top (daf6f89)
expect(len(labels) >= 3 and labels[0] == exp_titles[0] and labels[1:3] == exp_titles[1:3]
       and set(exp_titles[3:]) <= set(labels), f"Details expanders {labels[:6]} should hold {exp_titles}")
expect(sum("forward test" in l.lower() for l in labels) == 1, f"the forward test is in one place only: {labels}")
_ft = [d.value for d in at.dataframe if "Total since Oct 2" in d.value.columns]
expect(len(_ft) == 1 and {"QQQ", "SPY"} <= {x.split()[0] for x in _ft[0]["Strategy"]} if _ft else
       any("Forward test started Oct 2, 2026" in i.value for i in at.info), "forward test leaderboard (QQQ + SPY rows) or empty state")
_wk = [c for c in _ft[0].columns if c.startswith("Wk ")] if _ft else []
def _ranked_ok(b):
    """Ranked rows = the ranked strategies (cfg "ranked" not False) that have a total return; the rest show '–'."""
    rk = {c["name"] for c in ft.STRATEGIES if c.get("ranked", True)}
    names = b["Strategy"].replace({"Current Strategy (Live from Oct End)": ft.MARKED}).str.replace(ft.LIVE_MARK, "", regex=False)
    want = names.isin(rk) & b["Total since Oct 2"].notna()
    return bool((b["Rank"].notna() == want).all()) or not b["Rank"].notna().any()
expect(not _ft or (list(_ft[0].columns) == ["Rank", "Strategy", "Rule", "Total since Oct 2", "Last EOD", *_wk, "Max drawdown"] and [w.replace(" (so far)", "") for w in _wk] == [f"Wk {k}" for k in range(len(_wk), 0, -1)]
                   and (_ft[0]["Strategy"] == "Current Strategy (Live from Oct End)").sum() == 1 and _ranked_ok(_ft[0])),
       f"leaderboard: columns, one marked live row, every ranked strategy with data ranked (the Oct 16 live halves not): {_ft[0].to_dict('list') if _ft else None}")
expect(not _ft or sum(ft.RANK_RULE in c.value for c in at.caption) == 1, "the ranking rule is written once in the caption")
_rules = [d.value for d in at.dataframe if "Rule" in d.value.columns and "Total since Oct 2" not in d.value.columns]
_shown = [ft.MARKED if s == "Current Strategy (Live from Oct End)" else s.replace(ft.LIVE_MARK, "") for s in _ft[0]["Strategy"]] if _ft else []
_board_names = [s for s in _shown if s in {c["name"] for c in ft.STRATEGIES}] if _ft else [c["name"] for c in ft.STRATEGIES]
expect(len(_rules) == 1 and set(_rules[0]["Strategy"]) == {c["name"] for c in ft.STRATEGIES}
       and list(_rules[0]["Strategy"]) == _board_names,
       f"rules table: every strategy once, same order as the leaderboard: {_rules[0]['Strategy'][:5] if _rules else None}")
_top = [m.value for m in at.markdown if 'sa-card-title">Top 5 consensus<' in m.value]
_top_syms = re.findall(r'\?symbol=([A-Z0-9.]+)"[^>]*>[^<]*</a> (\d+)/(\d+) strategies', _top[0]) if _top else []
expect(len(_top) == 1 and len(_top_syms) == 10 and len({s for s, _, _ in _top_syms}) == 10
       and [int(c) for _, c, _ in _top_syms] == sorted((int(c) for _, c, _ in _top_syms), reverse=True)
       and all(int(c) <= int(n) == 5 for _, c, n in _top_syms),
       f"Top 5 consensus: 10 linked stocks from 5 strategies, most held first: {_top_syms}")
expect(any("flex-wrap: wrap" in m.value and "sa-tier-syms" in m.value for m in at.markdown),
       "stock-list rows (sa-tier-syms) must wrap onto more lines, not scroll")
_rule = "Don't change the strategy until 12+ weeks of forward results (from Oct 2, 2026) compare against QQQ."
for _rule in (_rule,):
    expect(sum(_rule in m.value for m in at.markdown) + sum(_rule in c.value for c in at.caption) == 1, f"shown once: {_rule}")
# short history: no Details section any more; the stocks sit in the clickable stock list (see the rank tiers test)
expect(not any(l.startswith("Reference only") for l in labels), f"stale short-history Details section: {labels}")
# Oct 16, 2026: ONE portfolio view (Strategy tab, Last decision); the Two halves, Targets and Latest signals views are gone
expect(not any(l.startswith(("Latest signals", "Two halves")) for l in labels), f"stale duplicate views: {labels}")
expect(not any("Signal today" in d.value.columns or "Weight_in_half_%" in d.value.columns for d in at.dataframe),
       "stale Latest signals / halves targets tables")
# Oct 21, 2026: Last decision = one top-20 ranking table per half (tests/test_strategy_tab.py has the math); the old
# "Show" radio (decision_filter) and the all-stocks table are gone
expect(not any(r.key == "decision_filter" for r in at.radio), "stale decision_filter radio (old Last decision view)")

# ---------------------------------------------------------------- Strategy tab widgets and captions
# the "if rebalanced at latest close" / "if the week ended today" preview views were removed from the app
# (the pipeline runs on decision days only, so there is nothing hypothetical left to show)
expect(not any(r.key == "changes_view" for r in at.radio), "stale changes_view preview radio still present")
expect(not any(r.key == "signals_view" for r in at.radio), "stale signals_view preview radio still present")
# Oct 16, 2026: Last decision = bought / sold / held from the account's fills (tests/test_strategy_tab.py has the math);
# Alpaca is off here, so it and Live vs benchmark show the plain off message
expect(not any(r.key == "signals_filter" for r in at.radio), "stale signals_filter radio (old Last decision view)")
_infos = [i.value for i in at.info]
expect(any(x.startswith("Last decision:") and "turned off" in x or x.startswith("No trade run yet") for x in _infos)
       and not any(x.startswith("Live vs benchmark") for x in _infos),
       f"Last decision: the plain message when Alpaca is not read (Live vs benchmark removed Oct 21): {_infos}")
# ONE decision view: the old "What changed at the latest decision" table (rules area) is merged into "Last decision"
expect(not any(c.key == "changes_all" for c in at.checkbox), "stale 'What changed' checkbox still present")
expect(not any("What changed at the latest decision" in m.value for m in at.markdown), "duplicate 'What changed' view is back")
old_cols = {"Weight before %", "Next rebalance plan", "Rank change"}
expect(not any(old_cols & set(d.value.columns) for d in at.dataframe), "stale Last decision columns still shown")
dec_labels = [e.label for e in at.expander if "decision" in e.label.lower()]
expect(dec_labels == [exp_titles[0]], f"decision expanders: {dec_labels}")
expect([e.label for e in at.expander][-1] == "Strategy rules", f"the rules must be the last Details section: {[e.label for e in at.expander]}")
# the Holdings risk and Order preview expanders were removed from the Details tab
expect(not any(t.key == "order_positions" for t in at.text_area), "stale order preview positions box still present")
expect(not any(s.key == "order_target" for s in at.selectbox), "stale order preview target selectbox still present")
expect(not any(e.label in ("Holdings risk", "Order preview (nothing is sent)") for e in at.expander),
       "stale Holdings risk / Order preview expander still present")
# the Backtest results expander (and its bt_segment selectbox) was removed from the app
# (live Alpaca P&L is the real number now; the backtest was a research artifact)
expect(not any(s.key == "bt_segment" for s in at.selectbox), "stale bt_segment selectbox still present")
expect(not any(m.label == "Last mid-week check" for m in at.metric), "stale 'Last mid-week check' metric")
blob = "\n".join(str(t.value) for t in list(at.markdown) + list(at.caption))
mwc = pd.read_csv("Reports/strategy_midweek_check.csv")
expect("run_pipeline" not in blob, "old command 'run_pipeline' still shown in the app")
# the Rank history expander was removed
expect(not any(e.label.startswith("Rank history") for e in at.expander), "stale Rank history expander still present")
# the rules shown match the live config (backtest_engine.WINNER), in one place
W = be.WINNER
rules = next((m.value for m in at.markdown if m.value.startswith("**Strategy rules")), "")
_friday = f"the {W['n']} best-ranked stocks by rank (no sector limit)"
_wants = [f"Strategy rules ({W['tag']})", f"{W['w_tech']:g} × Technical", f"at least {be.MIN_BARS} trading days",
          _friday, f"scaled to {be.LIVE_INVESTED:.0%} invested", f"{W['regime_symbol']} at or below its 200-day average",
          "every weight is halved" if W["regime_scale"] == 0.5 else "every weight is multiplied",
          f"within {W['rebalance_band'] * 100:g} percentage point", f"within {W['earnings_block_days']} calendar days", "after-hours limit orders",
          "9 AM CT", "2-decimal shares", "never runs twice"]
for want in _wants:
    expect(want in rules, f"rules text is missing {want!r}")
expect("swap" not in rules.lower() and "None" not in rules, "rules text still mentions a swap (removed 2026-10-07)")
expect("next open" not in blob, "outdated 'next open' wording (orders go out the same evening)")

# ---------------------------------------------------------------- removed: alert banner, Summary section, Strategy Health tab
box = [m.value for m in at.markdown if "sa-alert " in m.value]
expect(len(box) == 0, f"the alert banner was removed from the dashboard, found {len(box)}")
expect([t.label for t in at.tabs][:3] == ["Home", "Strategy", "Trading Account"],
       f"tabs: {[t.label for t in at.tabs]} (4ad4ef4: Home / Strategy / Trading Account; Strategy Health removed)")
expect(not [m for m in at.markdown if '<div class="sa-section">Summary</div>' in m.value], "the Summary section was removed")
expect(any("Live holdings are turned off here" in i.value for i in at.info), "holdings expander: off message in tests")

# ---------------------------------------------------------------- stock list: three rows, every stock once (Oct 16, 2026)
ROWS = ["Holdings", "Next in line", "Others"]   # Alpaca off: no preview, so no Keep / Buy next / Sell next
tier_md = [m.value for m in at.markdown if "Stock list<span>" in m.value and "?symbol=" in m.value]
expect(len(tier_md) == 1, f"stock list block found: {len(tier_md)}")
blob = tier_md[0] if tier_md else ""
parts = re.split(r'<b(?: title="[^"]*")?>(Keep|Buy next|Sell next|Holdings|Next in line|Others) \((\d+)\):</b>', blob)
expect([parts[i] for i in range(1, len(parts), 3)] == ROWS, f"stock list rows: {[parts[i] for i in range(1, len(parts), 3)]}")
bodies = {parts[i]: re.findall(r"\?symbol=([A-Z0-9.]+)", parts[i + 2]) for i in range(1, len(parts), 3)}
counts = {parts[i]: int(parts[i + 1]) for i in range(1, len(parts), 3)}
for name, syms in bodies.items():
    expect(len(syms) == counts[name], f"{name}: header count {counts[name]} != {len(syms)} links")
every = [s for syms in bodies.values() for s in syms]
_dnb = re.findall(r"\?symbol=([A-Z0-9.]+)[^<]*</a><sup[^>]*>DNB</sup>", blob)
expect(sorted(_dnb) == sorted(s for s in map(str.upper, sm.do_not_buy) if s in every) and _dnb,
       f"do-not-buy stocks tagged DNB inline once: {_dnb}")
expect(len(every) == len(set(every)), f"stock list: a stock shown twice: {sorted({s for s in every if every.count(s) > 1})}")
expect(set(sm.tradable_symbols) <= set(every), f"stock list misses {sorted(set(sm.tradable_symbols) - set(every))}")
expect(len(bodies.get("Next in line", [])) <= 20, "at most 20 candidates")
_led = sleeves.load_ledger()
if _led is not None:      # Alpaca is off in tests: the holdings row = the sleeve ledger's planned shares
    _held = {s for h in sleeves.HALVES for s, q in _led["halves"][h]["shares"].items() if q > 0}
    expect(set(bodies.get("Holdings", [])) == _held, f"holdings row {bodies.get('Holdings')} != ledger {sorted(_held)}")
print(f"   stock list rows: {counts}")
_sec = [d.value for d in at.dataframe if "1 month %" in d.value.columns and "Sector" in d.value.columns]
expect(len(_sec) == 1 and list(_sec[0].columns) == ["Sector", "Stocks", "14 days %", "1 month %", "2 months %", "3 months %", "6 months %", "1 year %"]
       and _sec[0]["Stocks"].sum() == len(sm.tradable_symbols) and _sec[0]["1 month %"].is_monotonic_decreasing,
       f"Home tab sector performance table: {_sec[0].columns.tolist() if _sec else None}")
all_linked = set(every)
expect(all_linked <= set(opts), f"stock list links outside the dropdown: {sorted(all_linked - set(opts))}")

# short-history stocks: listed with their rough signal, and a click (?symbol=) opens a stock view that renders
if len(SHORT):
    for _, r in SHORT.iterrows():   # Oct 16, 2026: in the stock list rows (Remaining unless held / a candidate)
        expect(r.Symbol in all_linked, f"{r.Symbol} missing from the stock list")
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
