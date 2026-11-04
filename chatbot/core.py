"""Chatbot brain for the dashboard's bottom-right chat (UI: chatbot/ui.py).

Read-only (no PIN): balance, positions, orders, P&L, quote, news, strategy, trade plan, explain SYM, help.
Money-moving (summary first, then PIN every time via chat_pin.verify before anything is sent):
  buy 5 AAPL | buy 5 AAPL limit 200 | buy $1000 AAPL | sell 3 AAPL | sell all AAPL | close AAPL
  cancel <order id start> | cancel all
do_not_buy / do_not_sell respected. Withdrawals / transfers / account settings refused (use the Alpaca app)."""
import re

SECURITY_WORDS = ("withdraw", "transfer", "wire", " ach ", "deposit", "bank", "password", "2fa", "mfa", "api key",
                  "secret", "email", "phone", "address", "setting", "configur", "margin", "options level", "journal")
HELP = ("Try **balance**, **positions**, **cash**, **P&L**, **orders**, **strategy**, **trade plan**, "
        "**why is APA underweight**, **quote AAPL**, **news AAPL**, or e.g. **buy 5 AAPL limit 200** "
        "(orders need PIN confirmation).")
SYM = r"([A-Za-z]{1,5}(?:[.\-/][A-Za-z]{1,2})?)"


def _money(x):
    return f"${float(x):,.2f}"


def parse(text):
    """-> dict(kind=...). Kinds include order/close/cancel (money-moving) plus read-only kinds."""
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
    m = re.fullmatch(r"(?:why\s+(?:is\s+)?|explain\s+)" + SYM + r"(?:\s+\w+){0,3}", low)
    if m:
        return {"kind": "explain", "symbol": m.group(1).upper()}
    if any(w in low for w in ("trade plan", "planned trade", "next trade", "what trade", "monday", "tomorrow's trade",
                              "what happens monday", "next rebalance", "dry run", "dry-run")):
        return {"kind": "trade_plan"}
    if any(w in low for w in ("strategy", "how does", "how do you", "rules", "equal weight", "rebalance schedule")):
        return {"kind": "strategy"}
    if re.search(r"\b(modify|replace|change|edit)\b.*\border", low):
        return {"kind": "refuse", "msg": "To change an order: **cancel** it, then place the new one."}
    for kind, words in (("balance", ("balance", "buying power", "cash", "equity", "account")),
                        ("pnl", ("p&l", "pnl", "p/l", "profit", "return", "gain", "loss", "today")),
                        ("positions", ("position", "portfolio", "holding", "what do i hold", "what do i own")),
                        ("orders", ("order",)),
                        ("help", ("help", "what can you", "commands", "hi", "hello"))):
        if any(w in low for w in words):
            return {"kind": kind}
    return {"kind": "unknown"}


def no_buy():
    import paper_trade as pt
    return pt.manual_list("buy")


def no_sell():
    import paper_trade as pt
    return pt.manual_list("sell")


def _account():
    import alpaca_paper as ap
    return ap.PaperAccount()


def trading_client():
    import paper_trade as pt
    return pt.paper_trading_client()


def latest_quote(sym):
    from alpaca.data.requests import StockLatestQuoteRequest
    from dashboard.alpaca_data import sip_then_iex
    import backtest_engine as be
    c = be.market_data_client()
    q, _ = sip_then_iex(lambda feed: c.get_stock_latest_quote(
        StockLatestQuoteRequest(symbol_or_symbols=sym, feed=feed))[sym])
    return float(q.bid_price or 0), float(q.ask_price or 0)


def _day_moves(syms):
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


def _strategy_text():
    from dashboard.details.last_decision import next_run
    import sleeves as sl
    import sector_mapping as sm
    led = sl.load_ledger()
    slot, kind = next_run(ledger=led)
    nxt = f"{slot:%a %b %-d, %-I:%M %p} CT · {kind}" if slot is not None else "none scheduled"
    dns = ", ".join(sorted(str(x).upper() for x in sm.do_not_sell)) or "(empty)"
    dnb = ", ".join(sorted(str(x).upper() for x in sm.do_not_buy)) or "(empty)"
    return (
        "**Live: Short-term relative strength, skip top 2 sectors (1-year) and stretched buys** (from Fri Nov 6, 2026) — one sleeve = the whole account.\n"
        "- Stocks in the **2 sectors with the best 1-year return** (Home tab sector table) are never bought or kept; a stock more than **8% above its 20-day average** is not newly bought (held ones kept); the next ranks fill in.\n"
        "- Top **10** stocks, **equal weight**: each targets `(equity × 0.99) / 10` (~9.9%). No per-name weight cap; no defensive sleeve.\n"
        "- **Friday 2:30 PM CT**: full rebalance. No Mon–Thu trades and drop_guard off (Nov 4, 2026).\n"
        "- Earnings within **5 days**: not newly bought; a held name is **not topped up** (can look underweight).\n"
        f"- **do_not_sell** (never sold): {dns}\n"
        f"- **do_not_buy** (never newly bought): {dnb}\n"
        f"- **Next run:** {nxt}"
    )


def _trade_plan_text():
    from dashboard.details.last_decision import next_run, plan_frames
    from dashboard.details.live_holdings import account_core
    import sleeves as sl
    import paper_trade as pt
    import sold_block as sb
    data, err = account_core()
    if data is None:
        return f"Trade plan unavailable: {err or 'no account data'}"
    led = sl.load_ledger()
    slot, kind = next_run(ledger=led)
    head = (f"**Next run: {slot:%a %b %-d, %-I:%M %p} CT** · {kind}\n"
            if slot is not None else "**Next run:** none scheduled\n")
    try:
        o, t, cash = plan_frames(data["positions"], data["equity"])
    except Exception as e:
        return head + f"Planned trades unavailable: {type(e).__name__}: {str(e)[:120]}"
    lines = [head.strip(), f"Cash after plan ≈ {_money(cash)} (preview; not submitted)."]
    if len(o):
        for r in o.itertuples(index=False):
            lines.append(f"- **{r.Side} {r.Symbol}** {r.Shares:g} sh · ≈ {_money(r[4])} "
                         f"(now {r[5]:g} → target {r[6]:g})")
    else:
        lines.append("- No buy/sell rows in the dry-run (all holds).")
    try:
        day = sl.inputs()["close"].index[-1]
        blocked = {**pt.earnings_blocked([p["symbol"] for p in data["positions"]], day),
                   **pt.earnings_stop_blocked(day), **dict(sb.active_blocks()), **no_buy()}
        held = {p["symbol"] for p in data["positions"] if float(p.get("qty") or 0) > 0}
        tgt_syms = set(t["Symbol"]) if len(t) else set()
        notes = [f"{s}: {blocked[s]}" for s in sorted(set(blocked) & (held | tgt_syms))]
        if notes:
            lines.append("**Blockers on held/target names:** " + "; ".join(notes[:12]))
    except Exception:
        pass
    if len(t):
        lines.append("**Target weights (preview):** " + ", ".join(
            f"{r.Symbol} {r[1]:.1f}%" for r in t.itertuples(index=False)))
    return "\n".join(lines)


def _explain_text(sym, account=None):
    acct = account or _account()
    import sleeves as sl
    import paper_trade as pt
    import sold_block as sb
    import forward_test as ft
    import backtest_engine as be
    import pandas as pd
    s = acct.account_summary()
    eq = float(s["Equity"])
    pdf = acct.positions()
    held_qty = held_val = 0.0
    if not pdf.empty and sym in set(pdf["Symbol"]):
        row = pdf[pdf["Symbol"] == sym].iloc[0]
        held_qty = float(row["Qty"])
        held_val = float(row["Market value"])
    pct = 100.0 * held_val / eq if eq and held_val else 0.0
    equal = 100.0 * 0.99 / 10.0
    lines = [f"**{sym}** · weight now **{pct:.2f}%** of equity {_money(eq)} · equal-weight slot **{equal:.1f}%** "
             f"(`(equity×0.99)/10`)."]
    if held_qty:
        lines.append(f"Held **{held_qty:g}** shares ≈ {_money(held_val)}.")
    else:
        lines.append("Not currently held.")
    try:
        lr = sl.live_ranks().get(sym)
        lines.append(f"Live RS rank: {lr[0]} (score {lr[1]:.2f}) — rank does not change the equal-weight slot for book members."
                    if lr else "Not in the live RS ranking.")
    except Exception:
        pass
    notes = []
    if sym in no_sell():
        notes.append(f"on do-not-sell ({no_sell()[sym]})")
    if sym in no_buy():
        notes.append(f"on do-not-buy ({no_buy()[sym]})")
    try:
        blk = sb.active_blocks()
        if sym in blk:
            notes.append(blk[sym])
    except Exception:
        pass
    try:
        day = sl.inputs()["close"].index[-1]
        earn = pt.earnings_blocked([sym], day)
        if sym in earn:
            notes.append(earn[sym])
        else:
            a = be.earnings_days_ahead([day], [sym], sl.inputs()["earnings"], ft.EARN_DAYS).iloc[0]
            if sym in a.index and pd.notna(a[sym]):
                notes.append(f"earnings in ~{int(a[sym])} session(s) (within 5-day no-buy/top-up window)")
    except Exception:
        pass
    try:
        from dashboard.details.last_decision import plan_frames
        from dashboard.details.live_holdings import account_core
        data, _ = account_core()
        if data:
            o, tgt, _cash = plan_frames(data["positions"], data["equity"])
            if len(o) and sym in set(o["Symbol"]):
                r = o[o["Symbol"] == sym].iloc[0]
                lines.append(f"Next dry-run row: **{r['Side']}** {float(r['Shares']):g} sh "
                             f"(now {float(r['Now shares']):g} → target {float(r['Target shares']):g}).")
            if len(tgt) and sym in set(tgt["Symbol"]):
                lines.append(f"Preview target weight: {float(tgt[tgt['Symbol']==sym]['Target %'].iloc[0]):.2f}%.")
            elif held_qty:
                lines.append("Not in the next dry-run target book (likely exiting or already out).")
    except Exception:
        pass
    if notes:
        lines.append("Notes: " + "; ".join(notes) + ".")
    elif held_qty and abs(pct - equal) > 0.4:
        lines.append("No earnings/sold_block/do_not_* note found; weight drift is from price moves since the last fill "
                     "or from cash that other blocked names could not absorb.")
    elif not held_qty:
        lines.append("Equal-weight only applies to the 10 names in the live book after the next trade.")
    return "\n".join(lines)


def answer(cmd, account=None):
    """Markdown answer for a read-only command."""
    k = cmd["kind"]
    if k in ("help", "unknown"):
        return ("I didn't get that. " if k == "unknown" else "") + HELP
    if k in ("refuse", "error"):
        return cmd["msg"]
    if k == "strategy":
        return _strategy_text()
    if k == "trade_plan":
        return _trade_plan_text()
    if k == "explain":
        return _explain_text(cmd["symbol"], account)
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
        s = acct.account_summary()
        eq = float(s["Equity"]) or 1.0
        rows = []
        for _, r in pos.iterrows():
            sym = r["Symbol"]
            qty = float(r["Qty"])
            val = float(r["Market value"])
            upct = float(r["Unrealized P/L %"])
            pct, chg = day.get(sym, (None, None))
            rows.append((pct if pct is not None else 0.0,
                         f"| {sym} | {qty:g} | {_money(val)} | {100 * val / eq:.2f}% | "
                         + (f"{pct:+.2f}% ({chg * qty:+,.2f} USD)" if pct is not None else "n/a")
                         + f" | {upct:+.1f}% |"))
        rows.sort(key=lambda x: x[0])
        return ("| Symbol | Qty | Value | Weight | Today (vs prior close) | Total P/L since bought |\n"
                "|---|---|---|---|---|---|\n" + "\n".join(line for _, line in rows)
                + f"\n\nEqual-weight slot ≈ **{100 * 0.99 / 10:.1f}%** each (`(equity×0.99)/10`). Equity {_money(eq)}.")
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
            txt += "\n\n**Last 5 orders:**\n" + "\n".join(
                f"- {x[1]} {x.Symbol} {x.Side} {x.Qty} ({x.Status})" for x in r.itertuples())
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

    def _status_val(o):
        st = getattr(o, "status", "")
        return (st.value if hasattr(st, "value") else str(st)).lower()

    def _record_chatbot_sell(sym, o):
        """sold_block only when the sell actually filled (shares left)."""
        try:
            filled = float(getattr(o, "filled_qty", 0) or 0)
        except (TypeError, ValueError):
            filled = 0.0
        if filled <= 0 and _status_val(o) != "filled":
            return False
        try:
            import sold_block
            sold_block.record_sell(str(sym).upper(), source="chatbot",
                                   note=f"chatbot fill {filled:g}",
                                   order_id=str(getattr(o, "id", "") or ""))
        except Exception:
            pass
        return True

    c = client or trading_client()
    try:
        if p["kind"] == "cancel":
            for oid in p["ids"]:
                c.cancel_order_by_id(oid)
            return True, f"Cancel sent for {len(p['ids'])} order(s)."
        if p["kind"] == "close":
            o = c.close_position(p["symbol"])
            _record_chatbot_sell(p["symbol"], o)
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
        if p.get("side") == "sell":
            _record_chatbot_sell(p["symbol"], o)
        return True, f"Order sent: `{str(getattr(o, 'id', ''))[:8]}` ({getattr(getattr(o, 'status', ''), 'value', getattr(o, 'status', ''))})."
    except Exception as e:
        return False, f"Alpaca refused it: {str(e)[:200]}"
