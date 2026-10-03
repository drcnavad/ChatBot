"""The REAL alpaca-py (no fakes, no network): every alpaca import and enum member the code uses exists, and the
exact request objects it sends can be built. The other tests replace alpaca with fakes, so a renamed library name
(e.g. SortDirection -> alpaca.common.enums.Sort) would only show up here.

Run: cd <folder> && python3 tests/test_alpaca_api_names.py"""
import ast
import glob
import importlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f" ({detail})" if detail and not cond else ""))


def sources():
    """(file, python source) for every .py file and every notebook code cell."""
    for f in sorted(glob.glob("*.py")):
        yield f, open(f).read()
    for f in sorted(glob.glob("*.ipynb")):
        for c in json.load(open(f))["cells"]:
            if c["cell_type"] == "code":
                yield f, "\n".join(l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("%", "!")))


imported = {}                                              # name -> real object
for f, src in sources():
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("alpaca"):
            mod = importlib.import_module(node.module)
            for a in node.names:
                ok = hasattr(mod, a.name)
                check(f"{f}: from {node.module} import {a.name}", ok)
                if ok:
                    imported[a.asname or a.name] = getattr(mod, a.name)

# every Enum.MEMBER the code uses (e.g. Sort.DESC, QueryOrderStatus.ALL, TimeInForce.DAY) exists
for f, src in sources():
    for cls, member in set(re.findall(r"\b([A-Z][A-Za-z]+)\.([A-Z][A-Za-z_]*)\b", src)):
        if cls in imported and isinstance(imported[cls], type):
            check(f"{f}: {cls}.{member}", hasattr(imported[cls], member))

# the exact request objects paper_trade / backtest_engine build (pydantic validates the fields; nothing is sent)
from datetime import datetime

from alpaca.common.enums import Sort
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest

builds = {
    "GetOrdersRequest(ALL, 500, Sort.DESC) - broker reconciliation":
        lambda: GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500, direction=Sort.DESC),
    "GetOrdersRequest(OPEN, symbols) - open orders per symbol":
        lambda: GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=["ANET"]),
    "LimitOrderRequest extended hours, whole shares":
        lambda: LimitOrderRequest(symbol="ANET", qty=3, limit_price=139.66, time_in_force=TimeInForce.DAY,
                                  extended_hours=True, side=OrderSide.BUY, client_order_id="live-20261002-BUY-ANET-3-1"),
    "LimitOrderRequest market hours, 2-decimal shares, DAY (smart limit)":
        lambda: LimitOrderRequest(symbol="ANET", qty=1.58, limit_price=139.73, time_in_force=TimeInForce.DAY,
                                  side=OrderSide.BUY, client_order_id="live-fill-20261002-BUY-ANET-1-13966"),
    "LimitOrderRequest under $1, 4 decimals": lambda: LimitOrderRequest(
        symbol="ABCD", qty=12.5, limit_price=0.5126, time_in_force=TimeInForce.DAY, side=OrderSide.SELL),
    "StockLatestQuoteRequest SIP feed (the plan check)":
        lambda: StockLatestQuoteRequest(symbol_or_symbols="SPY", feed=DataFeed.SIP),
    "StockLatestQuoteRequest feed from its name (sip / iex)":
        lambda: [StockLatestQuoteRequest(symbol_or_symbols="ANET", feed=DataFeed(f)) for f in ("sip", "iex")],
    "MarketOrderRequest 2-decimal shares":
        lambda: MarketOrderRequest(symbol="ANET", qty=0.58, time_in_force=TimeInForce.DAY, side=OrderSide.SELL,
                                   client_order_id="live-fill-20261002-SELL-ANET-0.58-1"),
    "StockBarsRequest daily, adjusted":
        lambda: StockBarsRequest(symbol_or_symbols=["AAPL"], timeframe=TimeFrame.Day, start=datetime(2026, 1, 1),
                                 end=datetime(2026, 2, 1), adjustment=Adjustment.ALL),
    "StockLatestTradeRequest": lambda: StockLatestTradeRequest(symbol_or_symbols=["AAPL"]),
}
for name, build in builds.items():
    try:
        build()
        check(f"builds {name}", True)
    except Exception as e:
        check(f"builds {name}", False, e)

# Every broker / market-data method the code calls exists on the real clients (the fakes in the other tests could hide a
# wrong name: client.get_order / cancel_order do not exist - the real ones are get_order_by_id / cancel_order_by_id).
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
for f, src in sources():
    if f not in ("paper_trade.py", "alpaca_paper.py", "run_all.py", "live_trade.ipynb", "backtest_engine.py"):
        continue
    for name in sorted(set(re.findall(r"\b(?:client|data_client|market_data_client\(\))\.(\w+)\(", src))):
        check(f"{f}: client.{name}() exists on the real Alpaca client",
              hasattr(TradingClient, name) or hasattr(StockHistoricalDataClient, name))

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
