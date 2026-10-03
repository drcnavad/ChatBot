"""sector_mapping.py is the ONLY place the stock list lives (Chirag, 2026-10-01: "I will keep adding and removing stocks in
sector mapping ... all files should use that updated list"). Fails when any other file has its own universe:

  1. a list / set / tuple literal holding >= 5 tickers of the stock list (sector_mapping.tradable_symbols +
     fundamentals_symbols, plus the names removed on 2026-10-01) - in every .py file and notebook code cell, tests included;
  2. a dict literal keyed by >= 20 such tickers (a second symbol map), except the allowlisted per-stock metadata maps below;
  3. a strategy-tag literal like "C6-U91" in production code (the U-count must come from len(tradable_symbols));
  4. a literal stock count like "91 stocks" (50..300) in a production string (counts must be computed);
and checks that the derived lists really are the sector_mapping ones (engine, autofetch, company-report scoring,
news/earnings call counts).
Docstrings and comments are documentation and are not scanned. No network calls.
Run: PYTHONPATH=. python tests/test_single_universe.py"""
import ast
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("STOCK_ANALYSIS_RUN_LOG", "/tmp/sa_test_run_log.csv")

import sector_mapping as sm  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


REMOVED_2026_10_01 = {"ADBE", "AFRM", "MU", "SOFI", "MDB", "MSTR"}       # must not reappear as a hidden list either
TICKERS = set(sm.tradable_symbols) | set(sm.fundamentals_symbols) | REMOVED_2026_10_01
# per-stock METADATA (keyed by symbol, not a universe: a symbol missing here falls back to sector_mapping), allowed by name
ALLOWED_NAMES = {
    ("sentiment_analysis.ipynb", "AMBIGUOUS_TICKERS"),     # tickers that are also English words (news search hygiene)
    ("sentiment_analysis.ipynb", "ALIASES"),               # curated news search names; fallback = sector_mapping.symbol_name
}
SKIP_FILES = {"sector_mapping.py", os.path.join("tests", "test_single_universe.py")}


def sources():
    """(relative path, python source, is_production) for every .py file and notebook in the project (no docs/, no hidden)."""
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if not x.startswith(".") and x not in {"docs", "Reports", "__pycache__", "logs"}]
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(d, f), ROOT)
            if rel in SKIP_FILES:
                continue
            prod = not rel.startswith("tests" + os.sep)
            if f.endswith(".py"):
                yield rel, open(os.path.join(d, f), encoding="utf-8").read(), prod
            elif f.endswith(".ipynb"):
                nb = json.load(open(os.path.join(d, f), encoding="utf-8"))
                for i, c in enumerate(nb.get("cells", [])):
                    if c.get("cell_type") == "code":
                        src = "".join(c.get("source", []))
                        src = "\n".join("" if ln.lstrip().startswith(("%", "!")) else ln for ln in src.splitlines())
                        yield f"{rel}#cell{i}", src, prod


def docstring_ids(tree):
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                ids.add(id(first.value))
    return ids


def assigned_name(tree):
    """{id(value node): target name} for simple assignments."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out[id(node.value)] = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            out[id(node.value)] = node.target.id
    return out


lists, maps, tags, counts, unparsable, n_files = [], [], [], [], [], 0
COUNT_RE = re.compile(r"\b(\d{2,3})[- ]stocks?\b", re.I)
TAG_RE = re.compile(r"\bC6-U\d+")
for rel, src, prod in sources():
    n_files += 1
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        unparsable.append(f"{rel}: {e}")
        continue
    docs, names = docstring_ids(tree), assigned_name(tree)
    base = rel.split("#")[0]
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
            hits = [e.value for e in node.elts if isinstance(e, ast.Constant) and e.value in TICKERS]
            if len(hits) >= 5 and (base, names.get(id(node))) not in ALLOWED_NAMES:
                lists.append(f"{rel}:{node.lineno} {names.get(id(node), '')} {hits[:6]}...({len(hits)})")
        elif isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant) and k.value in TICKERS]
            if len(keys) >= 20 and (base, names.get(id(node))) not in ALLOWED_NAMES:
                maps.append(f"{rel}:{node.lineno} {names.get(id(node), '')} ({len(keys)} ticker keys)")
        elif prod and isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            if TAG_RE.search(node.value):
                tags.append(f"{rel}:{node.lineno} {TAG_RE.search(node.value).group(0)!r}")
            m = COUNT_RE.search(node.value)
            if m and 50 <= int(m.group(1)) <= 300:
                counts.append(f"{rel}:{node.lineno} {m.group(0)!r}")

print(f"scanned {n_files} files / notebook cells; stock list = {len(sm.tradable_symbols)} tradable, "
      f"{len(sm.fundamentals_symbols)} fundamentals")
check("every file and notebook cell parses (else it could hide a list)", not unparsable, unparsable)
check("no hardcoded stock list outside sector_mapping.py (>= 5 tickers in one literal)", not lists, lists)
check("no second symbol map outside sector_mapping.py (dict with >= 20 ticker keys)", not maps, maps)
check("no hardcoded strategy tag (C6-U<n>) in production code - the U-count comes from the list", not tags, tags)
check("no hardcoded stock count ('NN stocks') in production strings", not counts, counts)

# --- the derived lists really are the sector_mapping ones ------------------------------------------------------------
import backtest_engine as be  # noqa: E402
import run_all  # noqa: E402

check("backtest/live engine universe == sector_mapping.tradable_symbols", be.TRADABLE == list(sm.tradable_symbols))
check("strategy tag U-count == stocks scored (the list minus short-history stocks)",
      be.WINNER["tag"].startswith(f"C6-U{be.scored_stock_count()}-"), be.WINNER["tag"])
# company_report_autofetch.py is checked from its source (importing it would load .env for the Alpha Vantage key)
_af = ast.parse(open("company_report_autofetch.py", encoding="utf-8").read())
_all = [ast.unparse(n.value) for n in ast.walk(_af) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "ALL_SYMBOLS" for t in n.targets)]
check("fundamentals autofetch list is sector_mapping.fundamentals_symbols (every tradable stock + extras)",
      _all == ["list(sector_mapping.fundamentals_symbols)"] and set(sm.tradable_symbols) <= set(sm.fundamentals_symbols), _all)
check("removed stocks are gone from the stock list and the fundamentals list",
      not (REMOVED_2026_10_01 & (set(sm.stock_symbols) | set(sm.fundamentals_symbols))),
      REMOVED_2026_10_01 & set(sm.fundamentals_symbols))
check("fundamentals extras are only names outside the stock list", not (set(sm.fundamentals_extra) & set(sm.stock_symbols)),
      set(sm.fundamentals_extra) & set(sm.stock_symbols))
check("news / earnings call counts come from the list", run_all.N_CALLS == len(sm.stock_symbols)
      and f"NewsAPI {len(sm.stock_symbols)}" in run_all.EXPECTED_CALLS["sentiment"], run_all.EXPECTED_CALLS)
check("every tradable stock has a sector and a name in sector_mapping",
      all(sm.symbol_sector.get(s) and sm.symbol_name.get(s) for s in sm.tradable_symbols),
      [s for s in sm.tradable_symbols if not (sm.symbol_sector.get(s) and sm.symbol_name.get(s))])
# company_report_processing.ipynb scores only the stocks in sector_mapping.py: the first cell is run on a temp balance_sheet.csv
# holding a listed stock, a fundamentals extra, a removed stock and an unknown one (no network: that cell only reads the csv)
import tempfile  # noqa: E402
_nb = json.load(open("company_report_processing.ipynb", encoding="utf-8"))
_cell = "".join(next(c for c in _nb["cells"] if c.get("cell_type") == "code")["source"])
_keep = [sm.tradable_symbols[0], sm.fundamentals_extra[0]]
_syms = _keep + ["MU", "ZZZZ"]
_old_dir = sm.REPORTS_DIR
with tempfile.TemporaryDirectory() as _d:
    be.pd.DataFrame({"Symbol": _syms, "FiscalDateEnding": "2026-06-30", "Date": "2026-07-01", "DateAdded": "2026-07-01",
                     **{c: 1.0 for c in ["TotalAssets", "TotalLiabilities", "TotalShareholderEquity",
                                         "CommonStockSharesOutstanding", "TotalDebt", "CashAndEquivalents"]}}
                    ).to_csv(os.path.join(_d, "balance_sheet.csv"), index=False)
    sm.REPORTS_DIR = _d
    try:
        _ns = {}
        exec(compile(_cell, "company_report_processing.ipynb#cell1", "exec"), _ns)
    finally:
        sm.REPORTS_DIR = _old_dir
check("company reports score only sector_mapping.py stocks (removed / unknown symbols in balance_sheet.csv dropped)",
      sorted(_ns["df"]["Symbol"]) == sorted(_keep), sorted(_ns["df"]["Symbol"]))
_cache = be.CACHE_DIR / be.LONG_CACHE
miss = sorted(set(be.TRADABLE + be.BENCHMARKS + be.SECTOR_ETFS) - set(be.pd.read_pickle(_cache)["Symbol"])) if _cache.exists() else []
check("backtest bar cache has every stock of the list (a new stock is fetched by be.ensure_long_cache on the next run)",
      not miss, f"missing {miss}: run the pipeline or python -c \"import backtest_engine as be; be.ensure_long_cache()\"")

print("\nSINGLE UNIVERSE OK" if not FAIL else f"\nSINGLE UNIVERSE FAILURES ({len(FAIL)}): {FAIL}")
sys.exit(1 if FAIL else 0)
