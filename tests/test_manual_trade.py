"""manual_trade.py with Alpaca MOCKED (a fake TradingClient + fake quote) - no network, never a real order:
market-closed refusal, bid/ask DAY limit orders, default buy size, --dry-run sends nothing, sell drops the symbol from the
sleeve ledger and adds a buy block, blocks feed paper_trade.manual_list('buy'), the after-close update sets / removes
blocks, every action logged. All files are temp files.
Run: python tests/run_tests.py  (or python tests/test_manual_trade.py)"""
import json
import os
import sys
import tempfile
from types import SimpleNamespace as NS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
_tmp = tempfile.mkdtemp()
for k, n in (("ORDERS", "manual_orders.json"), ("BLOCK", "manual_block.json")):
    os.environ[f"STOCK_ANALYSIS_MANUAL_{k}"] = os.path.join(_tmp, n)
os.environ["STOCK_ANALYSIS_SLEEVE_LEDGER"] = os.path.join(_tmp, "sleeve_ledger.json")
os.environ["STOCK_ANALYSIS_LIVE_HOLDINGS"] = "off"
import manual_trade as mt
import paper_trade as pt
import sleeves

FAIL = []

def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info and not ok else ""))
    if not ok:
        FAIL.append(name)

class FakeClient:
    def __init__(self, is_open=True):
        self.is_open, self.sent = is_open, []
    def get_clock(self):
        return NS(is_open=self.is_open)
    def get_all_positions(self):
        return [NS(symbol="KO", qty="98.5")]
    def get_account(self):
        return NS(equity="200000")
    def submit_order(self, req):
        self.sent.append(req)
        return NS(id="abc", status=NS(value="accepted"))

quote = lambda sym, side: {"bid": 60.0, "ask": 60.1, "limit": 59.97 if side == "SELL" else 60.13, "skip": None}
sleeves.save_ledger({"day": "2026-10-16", "halves": {"lowvol": {"shares": {"KO": 98.5, "PFE": 10}, "prices": {"KO": 60, "PFE": 25},
                                                                 "cash": 0}, "sector": {"shares": {"OXY": 5}, "prices": {"OXY": 50}, "cash": 0}}})

c = FakeClient(is_open=False)
e = mt.run("SELL", "KO", block_drop=10, client=c, quote=quote)
check("market closed: refused with a clear message, nothing sent", e["status"].startswith("refused: market closed") and not c.sent, e)

c = FakeClient()
e = mt.run("SELL", "ko", block_drop=10, dry_run=True, client=c, quote=quote)
check("--dry-run: planned and logged, nothing sent, ledger and blocks untouched",
      not c.sent and e["status"].startswith("dry run") and e["qty"] == 98.5 and "KO" in sleeves.load_ledger()["halves"]["lowvol"]["shares"]
      and mt.load_blocks() == {}, e)

e = mt.run("SELL", "KO", block_drop=10, client=c, quote=quote, wait=lambda cl, oid: (98.5, 60.0),
           rankings={"lowvol": ["KO", "PFE", "JCI", "NUE"], "sector": ["OXY"]})
r = c.sent[0] if c.sent else None
check("sell: whole position, DAY limit at the bid side price", r is not None and r.symbol == "KO" and float(r.qty) == 98.5
      and float(r.limit_price) == 59.97 and r.side.value == "sell" and r.time_in_force.value == "day"
      and r.client_order_id.startswith("manual-"), r)
led = sleeves.load_ledger()
check("sell: KO removed from the sleeve ledger (shares and prices), the rest kept",
      "KO" not in led["halves"]["lowvol"]["shares"] and "KO" not in led["halves"]["lowvol"]["prices"]
      and led["halves"]["lowvol"]["shares"].get("PFE") == 10 and e.get("ledger_half_removed") == "lowvol"
      and abs(led["halves"]["lowvol"]["cash"] - (98.5 * 60.0 - 98.33 * 60.1)) < 0.01, led)
rb = c.sent[1] if len(c.sent) > 1 else None
check("sell filled -> replacement: next-in-line of the same half (KO sold, PFE held -> JCI), limit <= ask, cash = proceeds",
      rb is not None and rb.symbol == "JCI" and rb.side.value == "buy" and float(rb.limit_price) == 60.1
      and float(rb.qty) == 98.33 and e["replacement"]["symbol"] == "JCI" and e["replacement"]["half"] == "lowvol", rb)
check("replacement joins the low-volatility half of the ledger", led["halves"]["lowvol"]["shares"].get("JCI") == 98.33, led)
b = mt.load_blocks()
check("sell --block-until-drop 10: block waiting for today's close", b.get("KO", {}).get("reference_close") is None
      and b["KO"]["drop_pct"] == 10, b)
check("the pipeline's do-not-buy list includes the block", "KO" in pt.manual_list("buy"), pt.manual_list("buy"))
e = mt.run("BUY", "KO", amount=1000, client=c, quote=quote)
check("buy of a blocked symbol: refused, nothing sent", e["status"].startswith("refused: manual block") and len(c.sent) == 2, e)

day = mt.load_blocks()["KO"]["added"]
mt.update_blocks({"KO": [("2020-01-01", 70.0), (day, 60.0)]})
b = mt.load_blocks()["KO"]
check("after close: reference close = that day's close, block_until_price = close x 0.9",
      b["reference_close"] == 60.0 and abs(b["block_until_price"] - 54.0) < 1e-9, b)
mt.update_blocks({"KO": [(day, 60.0), ("2099-01-02", 55.0)]})
check("price above block_until_price: block stays", "KO" in mt.load_blocks())
_, gone = mt.update_blocks({"KO": [("2099-01-02", 55.0), ("2099-01-03", 54.0)]})
check("price at block_until_price: block removed", gone == ["KO"] and "KO" not in mt.load_blocks())

c = FakeClient()
e = mt.run("BUY", "XYZ", amount=1000, client=c, quote=quote)
check("buy --amount 1000: floor(1000 / ask limit) to 0.01 shares, buy side", c.sent and float(c.sent[0].qty) == 16.63
      and c.sent[0].side.value == "buy" and float(c.sent[0].limit_price) == 60.13, c.sent)
c = FakeClient()
mt.run("BUY", "XYZ", client=c, quote=quote)
check("buy default: 5% of equity ($10,000 of $200,000)", c.sent and float(c.sent[0].qty) == 166.3, c.sent)
sleeves.save_ledger({"day": "2026-10-16", "halves": {"lowvol": {"shares": {"PFE": 10}, "prices": {"PFE": 25}, "cash": 0},
                                                     "sector": {"shares": {"KO": 98.5}, "prices": {"KO": 60}, "cash": 0}}})
c = FakeClient()
mt.run("SELL", "KO", client=c, quote=quote, replace=False, wait=lambda cl, oid: (98.5, 60.0))
check("--no-replace: the sell only, the cash stays in the half", len(c.sent) == 1
      and sleeves.load_ledger()["halves"]["sector"]["cash"] == 5910.0, c.sent)
sleeves.save_ledger({"day": "2026-10-16", "halves": {"lowvol": {"shares": {}, "prices": {}, "cash": 0},
                                                     "sector": {"shares": {"KO": 98.5}, "prices": {"KO": 60}, "cash": 0}}})
c = FakeClient()
mt.run("SELL", "KO", client=c, quote=quote, wait=lambda cl, oid: None, rankings={"sector": ["OXY"]})
check("sell not filled in time: no replacement buy (logged)", len(c.sent) == 1
      and json.load(open(mt.ORDERS_JSON))[-1]["status"].startswith("no replacement: the sell was not filled"), c.sent)
c = FakeClient()
e = mt.run("SELL", "AAPL", client=c, quote=quote)
check("sell of a stock not held: refused", e["status"] == "refused: not held" and not c.sent, e)
e = mt.run("BUY", "XYZ", client=FakeClient(), quote=lambda s, d: {"skip": "stale quote"})
check("bad quote: refused, nothing sent", e["status"] == "refused: stale quote", e)

rows = json.load(open(mt.ORDERS_JSON))
check("every action logged to the manual orders file (also refusals and dry runs)", len(rows) == 12
      and all({"time", "action", "symbol", "status"} <= set(r) for r in rows), len(rows))
check("tests never touched the real files", mt.ORDERS_JSON.startswith(_tmp) and mt.BLOCK_JSON.startswith(_tmp))
print(f"\n{len(FAIL)} failed" if FAIL else "\nall passed")
sys.exit(1 if FAIL else 0)
