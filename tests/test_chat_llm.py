"""dashboard/chat_llm.py with the LLM and Alpaca MOCKED (no network, never an order): tool calls become validated
commands for chat.plan; bad args rejected; security refused before the model; fallback chain; no PIN wording.
Run: python3 tests/test_chat_llm.py"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from dashboard import chat, chat_llm as L

FAIL = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        FAIL.append(name)

chat.trading_client = lambda: (_ for _ in ()).throw(AssertionError("trade client touched"))
L.providers = lambda: [("p1", "x", None, "m1", 1), ("p2", "x", None, "m2", 1)]
def tc(name, **a):
    return {"tool_calls": [{"id": "1", "function": {"name": name, "arguments": json.dumps(a)}}]}
def script(*replies, fail=()):
    seq, seen = list(replies), []
    def call(prov, msgs, first=False):
        seen.append((prov[0], msgs))
        if prov[0] in fail:
            raise RuntimeError("down")
        return seq.pop(0)
    return call, seen
plans = []
def plan(c):
    plans.append(c); return {"kind": c["kind"], **c}, "**SUMMARY**"

call, _ = script(tc("propose_order", side="buy", symbol="umac", qty=10, limit_price=15))
r = L.respond([], "Set a Buy order for 10 stocks of UMAC at Limit of 15", call=call, plan=plan)
check("NL buy -> plan buy 10 UMAC limit 15", plans[-1] == {"kind": "order", "side": "buy", "symbol": "UMAC", "limit": 15.0, "qty": 10.0}
      and r["plan"] and "Confirm below, or Cancel." in r["text"])
n = len(plans)
for bad in (dict(side="buy", symbol="AAPL"), dict(side="buy", symbol="AAPL", qty=-3), dict(side="buy", symbol="AAPL; rm", qty=1),
            dict(side="buy", symbol="AAPL", qty=1, notional=100), dict(side="buy", symbol="AAPL", notional=100, limit_price=5),
            dict(side="buy", symbol="AAPL", all=True), dict(side="short", symbol="AAPL", qty=1)):
    call, seen = script(tc("propose_order", **bad), {"content": "Need more detail."})
    r = L.respond([], "x", call=call, plan=plan)
    check(f"bad args rejected {bad}", len(plans) == n and r["plan"] is None and "Invalid" in seen[-1][1][-1]["content"])
call, _ = script(tc("propose_cancel", order_id="'; drop"), {"content": "?"})
check("bad cancel id rejected", L.respond([], "x", call=call, plan=plan)["plan"] is None and len(plans) == n)
call, seen = script()
r = L.respond([], "withdraw $500 to my bank", call=call, plan=plan)
check("withdrawal refused without calling the model", not seen and "Alpaca app" in r["text"] and r["plan"] is None)
L.run_read = lambda name, a, account=None: f"{name}:{a.get('symbol')}"
call, seen = script(tc("get_news", symbol="LMT"), {"content": "LMT news: upgrade."})
r = L.respond([], "What is LMT news today", call=call, plan=plan)
check("news tool called for LMT", seen[-1][1][-1]["content"] == "get_news:LMT" and r["text"] == "LMT news: upgrade.")
call, seen = script(tc("stock_info", symbol="LMT"), {"content": "LMT ranks 18."}, fail=("p1",))
r = L.respond([("user", "hi"), ("assistant", "hello")], "Tell me more about Lockheed martin stock", call=call, plan=plan)
check("falls back to next provider, keeps history", r["model"] == "m2" and seen[-1][1][1]["content"] == "hi")
call, _ = script(fail=("p1", "p2"))
check("all providers down -> None (keyword parser)", L.respond([], "balance", call=call, plan=plan) is None)
call, _ = script({"content": "Sure. Enter your PIN to confirm. Done soon."})
check("PIN wording stripped", "pin" not in L.respond([], "x", call=call, plan=plan)["text"].lower())
call, _ = script({"content": "**SELL 50 shares of KO** at market. Confirm below, or Cancel."})
r = L.respond([], "sell half KO", call=call, plan=plan)
check("model-written fake order summary not shown", r["plan"] is None and "Confirm below" not in r["text"])
print("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED")
sys.exit(1 if FAIL else 0)
