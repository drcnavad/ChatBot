"""Dashboard chatbot (chatbot/core.py + chatbot/pin.py) with Alpaca MOCKED - never a real order: command parsing,
withdrawals / security refused, do-not-sell refused, PIN hashed (no plaintext), wrong PIN sends nothing, lockout,
no PIN set = refused, right PIN sends exactly the summarized order. Temp PIN file only.
Run: python tests/run_tests.py  (or python tests/test_chat.py)"""
import os
import sys
import tempfile
from types import SimpleNamespace as NS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ["STOCK_ANALYSIS_CHAT_PIN"] = os.path.join(tempfile.mkdtemp(), "pin.json")
import pandas as pd
from chatbot import pin as chat_pin
from chatbot import core as chat

FAIL = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        FAIL.append(name)

class FakeAcct:
    def positions(self):
        return pd.DataFrame([{"Symbol": "KO", "Qty": 10.0}, {"Symbol": "XOM", "Qty": 4.0}])
    def account_summary(self):
        return {"Buying power": 5000.0}
    def _get(self, path, params=None):
        return [{"id": "abcd1234-0000", "symbol": "KO", "side": "buy", "qty": "1", "type": "limit"}]

class FakeClient:
    def __init__(self):
        self.sent = []
    def submit_order(self, req):
        self.sent.append(req); return NS(id="order-1", status="accepted")
    def cancel_order_by_id(self, oid):
        self.sent.append(("cancel", oid))
    def close_position(self, sym):
        self.sent.append(("close", sym)); return NS(id="c-1", status="accepted")

chat.no_sell = lambda: {"XOM": "on your do-not-sell list"}
q = lambda s: (99.0, 100.0)
A = FakeAcct()

check("parse buy limit", chat.parse("buy 5 aapl limit 200") == {"kind": "order", "side": "buy", "symbol": "AAPL", "limit": 200.0, "qty": 5.0})
check("withdraw refused", chat.parse("withdraw $500 to my bank")["kind"] == "refuse")
check("password refused", chat.parse("change my password")["kind"] == "refuse")
check("'each' is not ACH", chat.parse("positions each")["kind"] == "positions")
check("do-not-sell refused", chat.plan(chat.parse("sell all XOM"), A, q)[0] is None)
check("close do-not-sell refused", chat.plan(chat.parse("close XOM"), A, q)[0] is None)
check("sell more than held refused", chat.plan(chat.parse("sell 50 KO"), A, q)[0] is None)
check("over buying power refused", chat.plan(chat.parse("buy 100 AAPL"), A, q)[0] is None)
p, s = chat.plan(chat.parse("buy 5 AAPL limit 98"), A, q)
check("buy plan summary", p and "BUY 5 shares of AAPL" in s and "LIMIT $98.00" in s)

c = FakeClient()
ok, msg = chat.execute(p, "1234", client=c)
check("no PIN set: refused, nothing sent", not ok and not c.sent)
try:
    chat_pin.set_pin("12ab"); check("non-digit PIN rejected", False)
except ValueError:
    check("non-digit PIN rejected", True)
chat_pin.set_pin("482913")
check("PIN not stored in plaintext", "482913" not in open(chat_pin.PIN_FILE).read())
ok, msg = chat.execute(p, "000000", client=c)
check("wrong PIN: nothing sent", not ok and not c.sent)
ok, msg = chat.execute(p, "482913", client=c)
r = c.sent[-1] if c.sent else None
check("right PIN sends the exact order", ok and r is not None and r.symbol == "AAPL" and float(r.qty) == 5 and float(r.limit_price) == 98)
for _ in range(chat_pin.MAX_TRIES):
    chat_pin.verify("1")
n = len(c.sent)
ok, msg = chat.execute(p, "482913", client=c)
check("locked out after wrong tries, even with the right PIN", not ok and "locked" in msg.lower() and len(c.sent) == n)
chat_pin.set_pin("482913")
cp, _ = chat.plan(chat.parse("cancel abcd"), A, q)
ok, _ = chat.execute(cp, "482913", client=c)
check("cancel by id prefix", ok and c.sent[-1] == ("cancel", "abcd1234-0000"))
ok, _ = chat.execute({"kind": "close", "side": "sell", "symbol": "XOM"}, "482913", client=c)
check("execute re-checks do-not-sell", not ok and c.sent[-1] != ("close", "XOM"))
print("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED")
sys.exit(1 if FAIL else 0)
