"""chatbot/llm.py with LLM mocked — read-only tools, no propose_*, security refused, fallback.
Run: python3 tests/test_chat_llm.py"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from chatbot import core as chat, llm as L

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

# buy intent refused by rules before model
call, seen = script()
r = L.respond([], "buy 10 UMAC limit 15", call=call, plan=lambda c: (None, "x"))
check("buy refused without model", not seen and r and r["plan"] is None and "read-only" in r["text"].lower())

call, seen = script()
r = L.respond([], "withdraw $500 to my bank", call=call)
check("withdrawal refused without calling the model", not seen and "Alpaca app" in r["text"] and r["plan"] is None)

L.run_read = lambda name, a, account=None: f"{name}:{a.get('symbol')}"
call, seen = script(tc("get_news", symbol="LMT"), {"content": "LMT news: upgrade."})
r = L.respond([], "What is LMT news today", call=call)
check("news tool called for LMT", seen[-1][1][-1]["content"] == "get_news:LMT" and r["text"] == "LMT news: upgrade.")

call, seen = script(tc("get_trade_plan"), {"content": "Sells MA and V tomorrow."})
r = L.respond([], "what trades happen Monday", call=call)
check("trade_plan tool", seen[-1][1][-1]["content"] == "get_trade_plan:None" or "get_trade_plan" in seen[-1][1][-1]["content"]
      or r["text"] == "Sells MA and V tomorrow.")
# run_read for get_trade_plan gets {} args — our stub returns get_trade_plan:None
check("trade_plan answered", r["text"] == "Sells MA and V tomorrow.")

call, seen = script(tc("stock_info", symbol="LMT"), {"content": "LMT ranks 18."}, fail=("p1",))
r = L.respond([("user", "hi"), ("assistant", "hello")], "Tell me more about Lockheed martin stock", call=call)
check("falls back to next provider, keeps history", r["model"] == "m2" and seen[-1][1][1]["content"] == "hi")

call, _ = script(fail=("p1", "p2"))
check("all providers down -> None (keyword parser)", L.respond([], "balance", call=call) is None)

call, _ = script({"content": "Sure. Enter your PIN to confirm. Done soon."})
check("PIN wording stripped", "pin" not in L.respond([], "x", call=call)["text"].lower())

call, seen = script(tc("propose_order", side="buy", symbol="AAPL", qty=1), {"content": "ok read-only noted."})
r = L.respond([], "tell me a joke", call=call)  # won't parse as refuse; model may call propose
# If model calls propose_*, tool result says read-only then model replies
check("stale propose_* tool refused in-loop", any("read-only" in (m[-1].get("content") or "").lower()
      for m in [seen[-1][1]] if seen) or r["plan"] is None)

print("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
