"""Chatbot brain for the dashboard's bottom-right chat button (UI: chatbot/ui.py). No AI model and no agent quota:
plain commands are parsed here in code.

Read-only (no PIN): balance, positions, orders, P&L, quote SYM, news [SYM], help.
Money-moving (summary first, then the PIN every time, checked in code by chat_pin.verify before anything is sent):
  buy 5 AAPL | buy 5 AAPL limit 200 | buy $1000 AAPL | sell 3 AAPL | sell all AAPL | close AAPL
  cancel <order id start> | cancel all
Symbols on sector_mapping.do_not_buy are never bought from here; symbols on sector_mapping.do_not_sell are never sold or closed from here. Withdrawals, transfers and account / security
settings are refused (the trading API keys cannot do them; use the Alpaca app)."""
import re

SECURITY_WORDS = ("withdraw", "transfer", "wire", " ach ", "deposit", "bank", "password", "2fa", "mfa", "api key",
                  "secret", "email", "phone", "address", "setting", "configur", "margin", "options level", "journal")
HELP = ("Try **balance**, **positions**, **orders**, **P&L**, **quote AAPL** or **news AAPL**.")
SYM = r"([A-Za-z]{1,5}(?:[.\-/][A-Za-z]{1,2})?)"

def _money(x):
    return f"${float(x):,.2f}"

def parse(text):
    """-> dict(kind=..., ...). Kinds: help, balance, positions, orders, pnl, quote, news, refuse, order, close, cancel, unknown."""
    t = " ".join(str(text).strip().split())
    low = t.lower()
    padded = f" {low} "
    if not t:
        return {"kind": "help"}
    if any(w in padded for w in SECURITY_WORDS):
        return {"kind": "refuse", "msg": "I can't do withdrawals, transfers, deposits or account / security settings: "
                "the trading API keys this dashboard uses have no access to them. Please do that in the Alpaca app."}
    m = re.fullmatch(r"(buy|sell)\s+(all|\$?[\d,]*\.?\d+)\s+(?:shares?\s+(?:of\s+)?)?" + SYM +
                     r"(?:\s+(?:limit|lmt|at|@)\s*\$?([\d,]*\.?\d+))?", low)
    if m:
        side, amt, sym, lim = m.group(1), m.group(2), m.group(3).upper(), m.group(4)
        o = {"kind": "order", "side": side, "symbol": sym, "limit": float(lim.replace(",", "")) if lim else None}
        if amt == "all":
            if side == "buy":
                return {"kind": "unknown"}
            o["all"] = True
        elif amt.startswith("$"):
            o["notional"] = float(amt[1:].replace(",", ""))
            if o["limit"]:
                return {"kind": "error", "msg": "Dollar-amount orders are market orders only. Use a share count for a limit order."}
        else:
            o["qty"] = float(amt.replace(",", ""))
        if (o.get("qty") or o.get("notional") or o.get("all")) and (o["limit"] is None or o["limit"] > 0):
            return o
        return {"kind": "error", "msg": "The amount and price have to be above zero."}
    m = re.fullmatch(r"close\s+(?:position\s+)?" + SYM, low)
    if m and m.group(1) not in ("all",):
        return {"kind": "close", "symbol": m.group(1).upper()}
    if re.fullmatch(r"close\s+all(\s+positions)?|liquidate.*|sell everything", low):
        return {"kind": "refuse", "msg": "Closing every position at once isn't allowed from the chat. Close them one at a time."}
    m = re.fullmatch(r"cancel\s+(?:order\s+)?(all|[0-9a-f\-]{4,36})(?:\s+orders?)?", low)
    if m:
        return {"kind": "cancel", "target": m.group(1)}
    m = re.fullmatch(r"(?:quote|price)\s+(?:of\s+|for\s+)?" + SYM, low)
    if m:
        return {"kind": "quote", "symbol": m.group(1).upper()}
    m = re.fullmatch(r"news(?:\s+(?:on|for|about))?(?:\s+" + SYM + r")?", low)
    if m:
        return {"kind": "news", "symbol": (m.group(1) or "").upper() or None}
    if re.search(r"\b(modify|replace|change|edit)\b.*\border", low):
        return {"kind": "refuse", "msg": "To change an order: **cancel** it, then place the new one ."}
    for kind, words in (("balance", ("balance", "buying power", "cash", "equity", "account")),
                        ("pnl", ("p&l", "pnl", "p/l", "profit", "return", "gain", "loss", "today")),
                        ("positions", ("position", "portfolio", "holding")),
                        ("orders", ("order",)),
                        ("help", ("help", "what can you", "commands", "hi", "hello"))):
        if any(w in low for w in words):
            return {"kind": kind}
    return {"kind": "unknown"}

def no_buy():
    """SYMBOL -> note for Chirag's do-not-buy list (sector_mapping via paper_trade.manual_list)."""
    import paper_trade as pt
    return pt.manual_list("buy")


def no_sell():
    """SYMBOL -> note for Chirag's do-not-sell list (sector_mapping via paper_trade.manual_list)."""
    import paper_trade as pt
    return pt.manual_list("sell")

def _account():
    import alpaca_paper as ap
    return ap.PaperAccount()

def trading_client():
    import paper_trade as pt
    return pt.paper_trading_client()

def latest_quote(sym):
    """(bid, ask) from Alpaca market data (display / summary only)."""
    from alpaca.data.requests import StockLatestQuoteRequest
    from dashboard.alpaca_data import sip_then_iex
    import backtest_engine as be
    c = be.market_data_client()
    q, _ = sip_then_iex(lambda feed: c.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=sym, feed=feed))[sym])
    return float(q.bid_price or 0), float(q.ask_price or 0)

# ---- read-only answers -------------------------------------------------------------------------------------------------
def _day_moves(syms):
    """{sym: (today's % change, $ change per share)}: latest daily close vs the close before it (Alpaca daily bars)."""
    try:
        import pandas as pd
        from dashboard.data import NY, alpaca_bars
        bars = alpaca_bars(tuple(syms), None, pd.Timestamp.now(tz=NY) - pd.Timedelta(days=10))
        out = {}
        for s in syms:
            c = bars.get(s, {}).get("Close") if s in bars else None
            if c is not None and len(c.dropna()) >= 2:
                c = c.dropna()
                out[s] = ((c.iloc[-1] / c.iloc[-2] - 1) * 100, c.iloc[-1] - c.iloc[-2])
        return out
    except Exception:
        return {}

def answer(cmd, account=None):
    """Markdown answer for a read-only command."""
    k = cmd["kind"]
    if k in ("help", "unknown"):
        return ("I didn't get that. " if k == "unknown" else "") + HELP
    if k in ("refuse", "error"):
        return cmd["msg"]
    acct = account or _account()
    if k == "balance":
        s = acct.account_summary()
        return (f"**Equity** {_money(s['Equity'])} · **Cash** {_money(s['Cash'])} · **Buying power** "
                f"{_money(s['Buying power'])} · **Today** {s['Change today']:+,.2f} USD")
    if k == "pnl":
        s, pos = acct.account_summary(), acct.positions()
        a = acct._get("/account")
        last = float(a.get("last_equity") or 0)
        pct = (s["Change today"] / last * 100) if last else 0
        un = float(pos["Unrealized P/L"].sum()) if len(pos) else 0.0
        return (f"**Today** {s['Change today']:+,.2f} USD ({pct:+.2f}%) vs yesterday's close {_money(last)}.\n\n"
                f"**Open positions, unrealized** {un:+,.2f} USD.")
    if k == "positions":
        pos = acct.positions()
        if pos.empty:
            return "No open positions."
        day = _day_moves(tuple(pos["Symbol"]))
        rows = []
        for r in pos.itertuples():
            pct, chg = day.get(r.Symbol, (None, None))
            rows.append((pct if pct is not None else 0.0, f"| {r.Symbol} | {r.Qty:g} | {_money(r._5)} | "
                         + (f"{pct:+.2f}% ({chg * r.Qty:+,.2f} USD)" if pct is not None else "n/a")
                         + f" | {r._7:+.1f}% |"))
        rows.sort(key=lambda x: x[0])                      # biggest drop today first
        return ("| Symbol | Qty | Value | Today (vs prior close) | Total P/L since bought |\n|---|---|---|---|---|\n"
                + "\n".join(t for _, t in rows))
    if k == "orders":
        o = acct.open_orders()
        r = acct.recent_orders(5)
        txt = "**Open orders:** none." if o.empty else "**Open orders:**\n" + "\n".join(
            f"- {x.Symbol} {x.Side} {x.Qty} {x.Type} ({x.Status}), {x[1]}" for x in o.itertuples())
        ids = [f"- `{d['id'][:8]}` {d['symbol']} {d['side']} {d.get('qty') or d.get('notional')} {d['type']}"
               + (f" @ {d['limit_price']}" if d.get("limit_price") else "")
               for d in acct._get("/orders", {"status": "open", "limit": 50})]
        if ids:
            txt += "\n\nIDs to cancel:\n" + "\n".join(ids)
        if not r.empty:
            txt += "\n\n**Last 5 orders:**\n" + "\n".join(f"- {x[1]} {x.Symbol} {x.Side} {x.Qty} ({x.Status})" for x in r.itertuples())
        return txt
    if k == "quote":
        bid, ask = latest_quote(cmd["symbol"])
        return f"**{cmd['symbol']}** bid {_money(bid)} · ask {_money(ask)}"
    if k == "news":
        from dashboard import news as nw
        items = nw.stock_news(cmd["symbol"], 6)[0] if cmd["symbol"] else nw.headlines()[:8]
        if not items:
            return "No fresh headlines right now."
        return "\n".join(f"- [{i['title']}]({i.get('link') or i.get('url') or '#'}) · {i.get('source', '')}" for i in items)
    return HELP

# ---- money-moving actions: plan (no PIN) then execute (PIN checked first) -------------------------------------------
def plan(cmd, account=None, quote=latest_quote):
    """(plan dict, summary markdown) for an order / close / cancel, or (None, refusal text). Sends nothing."""
    k = cmd["kind"]
    acct = account or _account()
    if k in ("order", "close"):
        sym = cmd["symbol"]
        side = "sell" if k == "close" else cmd["side"]
        if side == "sell" and sym in no_sell():
            return None, f"**{sym}** is on your do-not-sell list, so I won't sell it from the chat."
        if side == "buy" and sym in no_buy():
            return None, f"**{sym}** is on your do-not-buy list, so I won't buy it from the chat."
        held = {r.Symbol: r.Qty for r in acct.positions().itertuples()}
        p = {"kind": k, "side": side, "symbol": sym, "limit": cmd.get("limit"), "qty": cmd.get("qty"), "notional": cmd.get("notional")}
        if side == "sell":
            have = held.get(sym, 0)
            if have <= 0:
                return None, f"You don't hold {sym}."
            if k == "close" or cmd.get("all"):
                p["qty"] = have
            if p["qty"] and p["qty"] > have + 1e-9:
                return None, f"You hold {have:g} {sym}; can't sell {p['qty']:g} (no short selling from the chat)."
            if p.get("notional"):
                return None, "Sell by share count (e.g. sell 3 AAPL or sell all AAPL)."
        if k == "close":
            summary = f"**CLOSE position {sym}**: sell all {p['qty']:g} shares at market (Alpaca close-position)."
        else:
            try:
                bid, ask = quote(sym)
            except Exception:
                bid = ask = 0
            ref = p["limit"] or (ask if side == "buy" else bid)
            size = f"{_money(p['notional'])} of" if p.get("notional") else f"{p['qty']:g} shares of"
            typ = f"LIMIT {_money(p['limit'])}" if p["limit"] else "MARKET"
            est = p["notional"] or (p["qty"] * ref if ref else None)
            summary = (f"**{side.upper()} {size} {sym}**, {typ}, DAY order.\n\nBid {_money(bid)} / ask {_money(ask)}"
                       + (f" · about **{_money(est)}**" if est else ""))
            if side == "buy" and est:
                bp = acct.account_summary()["Buying power"]
                if est > bp:
                    return None, f"That's about {_money(est)} but buying power is {_money(bp)}."
        return p, summary
    if k == "cancel":
        opens = acct._get("/orders", {"status": "open", "limit": 100})
        if cmd["target"] == "all":
            if not opens:
                return None, "No open orders to cancel."
            return {"kind": "cancel", "ids": [o["id"] for o in opens]}, f"**CANCEL all {len(opens)} open orders.**"
        hit = [o for o in opens if o["id"].startswith(cmd["target"])]
        if len(hit) != 1:
            return None, "No open order matches that ID." if not hit else "That ID start matches several orders; type more of it."
        o = hit[0]
        return ({"kind": "cancel", "ids": [o["id"]]},
                f"**CANCEL** order `{o['id'][:8]}`: {o['symbol']} {o['side']} {o.get('qty') or o.get('notional')} {o['type']}.")
    return None, "That isn't a trade."

def execute(p, pin, client=None, verify=None):
    """PIN first, then do_not_sell again, then send. Returns a message. Nothing is sent unless the PIN checks out."""
    from chatbot import pin as chat_pin
    ok, msg = (verify or chat_pin.verify)(pin)
    if not ok:
        return False, msg
    if p.get("side") == "sell" and p["symbol"] in no_sell():
        return False, f"{p['symbol']} is on your do-not-sell list; not sent."
    if p.get("side") == "buy" and p["symbol"] in no_buy():
        return False, f"{p['symbol']} is on your do-not-buy list; not sent."
    c = client or trading_client()
    try:
        if p["kind"] == "cancel":
            for oid in p["ids"]:
                c.cancel_order_by_id(oid)
            return True, f"Cancel sent for {len(p['ids'])} order(s)."
        if p["kind"] == "close":
            o = c.close_position(p["symbol"])
            return True, f"Close sent for {p['symbol']} (order `{str(getattr(o, 'id', ''))[:8]}`, {getattr(getattr(o, 'status', ''), 'value', '')})."
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
        kw = dict(symbol=p["symbol"], side=OrderSide.BUY if p["side"] == "buy" else OrderSide.SELL, time_in_force=TimeInForce.DAY)
        if p.get("notional"):
            kw["notional"] = round(p["notional"], 2)
        else:
            kw["qty"] = p["qty"]
        req = LimitOrderRequest(limit_price=round(p["limit"], 2), **kw) if p.get("limit") else MarketOrderRequest(**kw)
        o = c.submit_order(req)
        return True, f"Order sent: `{str(getattr(o, 'id', ''))[:8]}` ({getattr(getattr(o, 'status', ''), 'value', getattr(o, 'status', ''))})."
    except Exception as e:
        return False, f"Alpaca refused it: {str(e)[:200]}"
