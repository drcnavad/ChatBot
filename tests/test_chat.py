"""Dashboard chatbot (chatbot/core.py) — read-only: parse, refuse trades, balance/help.
Alpaca mocked where needed. Run: python tests/run_tests.py
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ["STOCK_ANALYSIS_CHAT_PIN"] = os.path.join(tempfile.mkdtemp(), "pin.json")
import pandas as pd
from chatbot import core as chat

FAIL = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        FAIL.append(name)

class FakeAcct:
    def positions(self):
        return pd.DataFrame([{"Symbol": "KO", "Qty": 10.0, "Avg entry": 50.0, "Current price": 55.0,
                              "Market value": 550.0, "Unrealized P/L": 50.0, "Unrealized P/L %": 10.0, "Side": "long"}])
    def account_summary(self):
        return {"Equity": 10000.0, "Cash": 1000.0, "Buying power": 1000.0, "Change today": -12.0}
    def _get(self, path, params=None):
        return {"last_equity": 10012.0} if path == "/account" else []
    def open_orders(self):
        return pd.DataFrame()
    def recent_orders(self, n=5):
        return pd.DataFrame()

check("parse cash -> balance", chat.parse("what's my cash")["kind"] == "balance")
check("parse holdings", chat.parse("what do I hold")["kind"] == "positions")
check("parse trade plan monday", chat.parse("what trades happen Monday")["kind"] == "trade_plan")
check("parse why APA", chat.parse("why is APA underweight") == {"kind": "explain", "symbol": "APA"})
check("parse strategy", chat.parse("how does the live strategy work")["kind"] == "strategy")
check("withdraw refused", chat.parse("withdraw $500 to my bank")["kind"] == "refuse")
check("buy refused", chat.parse("buy 5 aapl limit 200")["kind"] == "refuse")
check("sell refused", chat.parse("sell all KO")["kind"] == "refuse")
check("cancel refused", chat.parse("cancel all")["kind"] == "refuse")
check("close refused", chat.parse("close KO")["kind"] == "refuse")
check("'each' is not ACH", chat.parse("positions each")["kind"] == "positions")
p, msg = chat.plan({"kind": "order", "side": "buy", "symbol": "AAPL", "qty": 1})
check("plan never returns a trade", p is None and "read-only" in msg.lower())
ok, msg = chat.execute({"kind": "order", "side": "buy", "symbol": "AAPL", "qty": 1}, "482913")
check("execute hard-refuses", not ok and "read-only" in msg.lower())
A = FakeAcct()
bal = chat.answer({"kind": "balance"}, A)
check("balance answer", "Equity" in bal and "$1,000.00" in bal)
print("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
