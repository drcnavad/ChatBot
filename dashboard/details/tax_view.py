"""Trading Account tab: the tax view (tax_lots.py, an estimate, counted from tax_lots.TAX_START)."""
from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard.details.live_holdings import holdings_this_run
from dashboard.settings import CT
from dashboard.style import (BAD, CAUTION, GOOD, INK, MONEY, caption_text, esc, info_text, show_html, stat_cards, tone,
    toned, usd, warning_text)

# ---------------------------------------------------------------------------- tax view (estimate, read-only)
@st.cache_data(ttl=3600, max_entries=2, show_spinner=False)
def _read_tax_inputs(hour_key):
    """Every account activity + order client ids from the live account (GET only), once per hour (hour_key)."""
    import alpaca_paper as ap
    import tax_lots as tl
    return {**tl.fetch_inputs(ap.PaperAccount()), "as_of": datetime.now(CT)}

@st.cache_data(ttl=3600, max_entries=12, show_spinner=False)
def _tax_report(hour_key, method, positions):
    import tax_lots as tl
    d = _read_tax_inputs(hour_key)
    return tl.build(d["activities"], list(positions), method, d["client_ids"], today=datetime.now(CT).date())

HARVEST_COLS = {"Loss": MONEY, "Short-term loss": MONEY, "Long-term loss": MONEY, "Why": st.column_config.TextColumn(width="large"),
                "Shares at a loss": st.column_config.NumberColumn(format="%.4g")}

def _harvest_view(t):
    return (t.round({"Loss": 2, "Short-term loss": 2, "Long-term loss": 2})
            .assign(**{"Next long-term date": [f"{d:%b %-d, %Y}" if d is not None and d == d else "—" for d in t["Next long-term date"]]}))

def render_tax_view(p):
    """Trading Account tab: realized / unrealized gains (short vs long term), wash sales, harvest list, estimated tax, Form 8949
    export, from the live account's activity since tax_lots.TAX_START (read-only, hourly). An estimate, not tax advice."""
    import tax_lots as tl
    show_html(f'<div class="sa-note"><b>Estimate, not tax advice.</b> {esc(tl.DISCLAIMER.split(". ", 1)[1])}</div>')
    data, err = holdings_this_run()
    if err:
        caption_text("The tax view needs the live holdings, which are not available right now (see Live holdings above).")
        return
    key = f"{datetime.now(CT):%Y-%m-%d %H}"
    positions = tuple(data["positions"])
    try:
        inputs = _read_tax_inputs(key)
        rep = _tax_report(key, "FIFO", positions)
    except Exception as e:
        info_text(f"Tax view unavailable right now ({type(e).__name__}: {str(e)[:200]}). It tries again next hour.")
        return
    y = tl.ytd(rep)
    year = y["year"]
    start = rep.get("start")
    scope = f"since {start:%b %-d}" if start is not None and start.year == year else str(year)     # card labels
    when = f"since {start:%a %b %-d, %Y}" if start is not None else f"in {year}"
    pre, pre_sales = rep.get("pre_open", pd.DataFrame()), rep.get("pre_sales", pd.DataFrame())
    if start is not None:
        note = (f"Fresh start: only activity on or after {start:%a %b %-d, %Y} counts (TAX_START in tax_lots.py). "
                "Earlier account activity (trades, dividends, interest, fees, prior tax years) is not counted, and neither is "
                "wash-sale matching against older trades.")
        if len(pre):
            note += (" Left out (bought before the start; Alpaca's positions have no purchase date, so they are not carried "
                     "in): " + ", ".join(f"{r['Symbol']} {r['Shares']:.4g} sh (Alpaca cost {usd(r['Alpaca avg cost'], sign=False)})"
                                         for _, r in pre.iterrows()) + ".")
        if len(pre_sales):
            note += (f" {len(pre_sales)} sale(s) {when} sold shares bought before it ({', '.join(sorted(set(pre_sales['Symbol'])))}; "
                     f"proceeds {usd(pre_sales['Proceeds'].sum(), sign=False)}): not counted.")
        info_text(note + " Alpaca's 1099-B uses the full history, so its numbers can differ.")
    issues = rep["issues"]
    if len(issues):
        warning_text(f"Data check: {len(issues)} issue(s) found, so some numbers below may be off. Details in the table.")
        st.dataframe(issues, hide_index=True, width="stretch", height=min(400, 35 * (len(issues) + 1) + 3))
    stat_cards([(f"Realized {scope} short-term", usd(y["realized_st"]), tone(y["realized_st"])),
                (f"Realized {scope} long-term", usd(y["realized_lt"]), tone(y["realized_lt"])),
                ("Unrealized short-term", usd(y["unrealized_st"]), tone(y["unrealized_st"])),
                ("Unrealized long-term", usd(y["unrealized_lt"]), tone(y["unrealized_lt"])),
                (f"Wash-sale loss deferred {scope}", usd(y["wash_disallowed"], sign=False), CAUTION if y["wash_disallowed"] > 0.005 else INK),
                (f"Dividends {scope} (qualified est.)", f"{usd(y['dividends'], sign=False)} ({usd(y['qualified'], sign=False)})", INK),
                (f"Interest {scope}", usd(y["interest"], sign=False), INK),
                (f"Fees {scope}", usd(y["fees"], sign=False), INK)])
    counted = (f"counted from {start:%b %-d, %Y} (New York trade dates)" if start is not None
               else f"{year} = trade dates in {year} (New York time)")
    held_txt = f"the lots plus the {len(pre)} left-out holding(s)" if len(pre) else "the lots"
    caption_text(f"As of {inputs['as_of']:%a %b %-d %I:%M %p} CT (account history read from Alpaca hourly, read-only; prices from the "
               f"live holdings read) · lot method FIFO (Alpaca's default) · {counted} · "
               "realized gains include the wash-sale adjustments; unrealized = open lots at the latest price."
               + ("" if len(issues) else f" Data checks passed: every sale matched to a purchase, no missing basis, no negative "
                                         f"lots, and {held_txt} equal Alpaca's {len(positions)} positions."))
    tabs = st.tabs(["By year", "Open lots", "Tax-loss harvest", "Wash sales", "Estimated tax", "Dividends, interest, fees",
                    "Lot method", "Form 8949 export"])
    sales = rep["sales"]
    with tabs[0]:
        by = tl.realized_by_year(sales)
        if by.empty:
            info_text(f"No sales of lots bought {when} yet, so nothing is realized.")
        st.dataframe(toned(by.round(2), ["Short-term gain", "Long-term gain", "Total gain"]), hide_index=True, width="stretch",
                     column_config={c: MONEY for c in by.columns if c not in ("Year", "Lot sales")} | {"Year": st.column_config.NumberColumn(format="%d")})
        caption_text(f"Realized gains by tax year, as of {inputs['as_of']:%a %b %-d, %Y}. Gain = proceeds (after REG/TAF/CAT fees) - "
                   "basis + wash-sale adjustment; short-term = held one year or less, long-term = more than one year. Wash sale "
                   "disallowed = losses not deductible that year (moved into the replacement shares' basis).")
    with tabs[1]:
        lots = tl.open_lots_view(rep["lots"])
        if lots.empty:
            info_text("No open lots.")
        else:
            shown = lots.assign(**{c: [f"{d:%b %-d, %Y}" for d in lots[c]] for c in ("Bought", "Holding from", "Long-term on")})
            shown["Gain"] = shown["Gain"].round(2) + 0.0                        # no "-$0.00"
            st.dataframe(toned(shown, ["Gain"]), hide_index=True, width="stretch",
                         column_config={"Basis / share": MONEY, "Basis": MONEY, "Price": MONEY, "Value": MONEY, "Gain": MONEY,
                                        "Shares": st.column_config.NumberColumn(format="%.4g"), "Note": st.column_config.TextColumn(width="medium")})
            soon = tl.turning_long_term(rep, 60)
            if len(soon):
                warning_text(f"{len(soon)} lot(s) turn long-term within 60 days: "
                           + "; ".join(f"{r['Symbol']} {r['Shares']:g} sh on {r['Long-term on']:%b %-d} ({r['Window']}; {r['Hint']})"
                                       for _, r in soon.iterrows()))
            else:
                oldest = rep["lots"].sort_values("Holding from").iloc[0]
                caption_text(f"No open lot turns long-term within 60 days (the oldest, {oldest['Symbol']} held from "
                           f"{oldest['Holding from']:%b %-d, %Y}, turns long-term on {oldest['Long-term on']:%b %-d, %Y}).")
            caption_text(f"Open lots as of {inputs['as_of']:%a %b %-d %I:%M %p} CT (FIFO). Holding from = the purchase date, earlier "
                       "after a wash sale (the sold shares' holding period carries over); Long-term on = the first day a sale "
                       "counts as long-term. Basis includes fees and wash-sale adjustments (Alpaca's position basis does not "
                       "include wash-sale adjustments, so it can be lower).")
    with tabs[2]:
        planned = {s for s, (a, w) in (p.plan or {}).items() if a in ("Buy", "Keep")}
        ok, excluded = tl.harvest(rep, planned)
        if ok.empty:
            st.success("No position to harvest right now: " + ("nothing is at a loss." if excluded.empty else
                                                               "every position at a loss is excluded (wash-sale risk, below)."))
        else:
            st.dataframe(toned(_harvest_view(ok), ["Loss", "Short-term loss", "Long-term loss"]), hide_index=True, width="stretch",
                         column_config=HARVEST_COLS)
        if len(excluded):
            caption_text("Excluded (selling now would likely be a wash sale):")
            st.dataframe(toned(_harvest_view(excluded), ["Loss"]), hide_index=True, width="stretch", column_config=HARVEST_COLS)
        plan_lbl = f"The bot's {p.plan_day:%a %b %-d} plan" if p.plan_day is not None else "The bot's next plan"
        caption_text(f"Possible tax-loss sales as of {inputs['as_of']:%a %b %-d %I:%M %p} CT. Wash-sale rule: buying the same stock "
                   "within 30 days before or after selling it at a loss cancels the loss for now (it moves into the new shares' "
                   f"basis). {plan_lbl} holds or buys {', '.join(sorted(planned)) or 'nothing'}, so a sale of those would likely be "
                   "bought back within 30 days. Nothing here sells anything; a sale is your decision.")
    with tabs[3]:
        summ, w = tl.wash_summary(rep["washes"], year)
        if summ.empty:
            st.success(f"No wash sales on sales {when}.")
        else:
            st.dataframe(summ.round(2).assign(**{"Last loss sale": [f"{d:%b %-d, %Y}" for d in summ["Last loss sale"]]}),
                         hide_index=True, width="stretch", height=min(420, 35 * (len(summ) + 1) + 3),
                         column_config={"Disallowed loss": MONEY, "Shares": st.column_config.NumberColumn(format="%.4g")})
            caption_text(f"Wash sales on {year} loss sales: {len(w)} matches, {usd(w['Disallowed loss'].sum(), sign=False)} of losses "
                       f"deferred; {int(w['Replacement order'].str.startswith('bot').sum())} replacement buy(s) were the bot's own "
                       "orders (live- client ids: weekly rebalance / mid-week rebuys), the rest other orders.")
            detail = w.assign(**{c: [f"{d:%b %-d, %Y}" for d in w[c]] for c in ("Loss sale", "Replacement bought")}).drop(columns="Replacement lot")
            st.dataframe(detail.round(2).iloc[::-1], hide_index=True, width="stretch", height=300,
                         column_config={"Disallowed loss": MONEY, "Shares": st.column_config.NumberColumn(format="%.4g")})
        caption_text("Matching: replacement shares bought within 30 days before or after the loss sale (other shares of the same "
                   "purchase count), oldest purchase first, each share once, partial shares allowed. The deferred loss is added "
                   "to the replacement shares' basis and the sold shares' holding period carries over (Form 8949 code W).")
    with tabs[4]:
        c = st.columns(3)
        filing = c[0].selectbox("Filing status (default: Single)", list(tl.FILING), format_func=tl.FILING.get, key="tax_filing")
        agi = c[1].number_input("Other income outside this account, AGI (default $150,000, edit)", min_value=0.0,
                                value=150000.0, step=5000.0, key="tax_agi")
        ded = c[2].number_input(f"Deduction (default: 2026 standard, ${tl.STANDARD_DEDUCTION_2026[filing]:,})", min_value=0.0,
                                value=float(tl.STANDARD_DEDUCTION_2026[filing]), step=500.0, key=f"tax_ded_{filing}")
        c = st.columns(3)
        state = c[0].number_input("State tax rate % (default 4.95% = Illinois flat, edit)", min_value=0.0, max_value=15.0,
                                  value=4.95, step=0.25, key="tax_state")
        cst = c[1].number_input("Short-term loss carryover from last year (default $0)", min_value=0.0, value=0.0, step=100.0, key="tax_cst")
        clt = c[2].number_input("Long-term loss carryover from last year (default $0)", min_value=0.0, value=0.0, step=100.0, key="tax_clt")
        ordinary_div = max(0.0, y["dividends"] - y["qualified"])
        est = tl.estimate_tax(y["realized_st"], y["realized_lt"], y["qualified"], ordinary_div, y["interest"], filing, agi, ded,
                              state / 100, cst, clt)
        stat_cards([("Federal income tax (extra)", usd(est["federal"]), BAD if est["federal"] > 0.005 else GOOD if est["federal"] < -0.005 else INK),
                    ("NIIT 3.8%", usd(est["niit"]), BAD if est["niit"] > 0.005 else INK),
                    ("State", usd(est["state"]), BAD if est["state"] > 0.005 else GOOD if est["state"] < -0.005 else INK),
                    ("Total estimate", usd(est["total"]), BAD if est["total"] > 0.005 else GOOD if est["total"] < -0.005 else INK),
                    ("Net short-term", usd(est["net_short"]), tone(est["net_short"])),
                    ("Net long-term", usd(est["net_long"]), tone(est["net_long"])),
                    ("Loss deducted vs income", usd(est["loss_deducted"], sign=False), INK),
                    ("Carryforward to next year", usd(est["carry_short"] + est["carry_long"], sign=False), INK)])
        caption_text(f"{year} estimate as of {inputs['as_of']:%a %b %-d, %Y}, from the realized gains, dividends and interest so far "
                   "(unrealized gains are not taxed until sold). Extra tax = tax with this account - tax without it, 2026 federal "
                   "brackets; short-term gains, ordinary dividends and interest at ordinary rates, long-term gains and qualified "
                   f"dividends at 0/15/20% stacked on top. NIIT: 3.8% on investment income above ${est['niit_threshold']:,} of "
                   f"modified AGI (here ${est['magi']:,.0f}). Net capital loss: up to ${tl.LOSS_LIMIT[filing]:,} a year offsets "
                   "other income, the rest carries forward (short-term used first). State = the rate × this account's taxable "
                   "income. Margin interest and other itemized items are not deducted. Negative = tax saved.")
    with tabs[5]:
        inc = rep["income"]
        if inc.empty and rep["fees"].empty:
            info_text(f"No dividends, interest or fees {when}.")
        else:
            by = inc.groupby(["Year", "Type"], as_index=False).agg(Amount=("Amount", "sum"), **{"Qualified (est.)": ("Qualified (est.)", "sum")})
            fees = rep["fees"].assign(Year=[d.year for d in rep["fees"]["Date"]]).groupby(["Year", "Type"], as_index=False)["Amount"].sum()
            fees["Amount"] = -fees["Amount"]
            table = pd.concat([by, fees], ignore_index=True).sort_values(["Year", "Type"], ascending=[False, True])
            st.dataframe(toned(table.round(2), ["Amount"]), hide_index=True, width="stretch",
                         column_config={"Amount": MONEY, "Qualified (est.)": MONEY, "Year": st.column_config.NumberColumn(format="%d")})
            caption_text(f"By year, as of {inputs['as_of']:%a %b %-d, %Y}. Qualified (est.) = dividends on shares held more than 60 days "
                       "in the 121-day window around the ex-date (ex-date taken as the record date); the 1099-DIV decides. Fees "
                       "(negative) are already in the gains: REG / TAF lower that day's sale proceeds, CAT is spread over that "
                       "day's fills. Margin interest is investment interest expense (deductible only if you itemize, Form 4952); "
                       "securities lending income is reported on a 1099-MISC.")
            st.dataframe(inc.assign(Date=[f"{d:%b %-d, %Y}" for d in inc["Date"]]).iloc[::-1].round(2), hide_index=True,
                         width="stretch", height=min(300, 35 * (len(inc) + 1) + 3),
                         column_config={"Amount": MONEY, "Qualified (est.)": MONEY, "Year": st.column_config.NumberColumn(format="%d"),
                                        "Note": st.column_config.TextColumn(width="large")})
    with tabs[6]:
        rows = []
        for m in tl.METHODS:
            r = rep if m == "FIFO" else _tax_report(key, m, positions)
            v = tl.ytd(r)
            rows.append({"Method": m + (" (Alpaca default, used above)" if m == "FIFO" else ""),
                         f"{year} short-term": v["realized_st"], f"{year} long-term": v["realized_lt"],
                         f"{year} total": v["realized_st"] + v["realized_lt"], "Wash sale disallowed": v["wash_disallowed"],
                         "Unrealized": v["unrealized_st"] + v["unrealized_lt"]})
        cmp_ = pd.DataFrame(rows)
        st.dataframe(toned(cmp_.round(2), [c for c in cmp_.columns if c.startswith(str(year))]), hide_index=True, width="stretch",
                     column_config={c: MONEY for c in cmp_.columns if c != "Method"})
        caption_text(f"Same trades, {year} realized as of {inputs['as_of']:%a %b %-d, %Y}, under other lot methods: HIFO sells the "
                   "highest-cost shares first, LIFO the newest. Comparison only: your broker's method decides, and a different "
                   "method (or picking specific lots) must be set at Alpaca before a sale; it never changes past sales.")
    with tabs[7]:
        years = sorted({int(v) for v in sales["Year"]}, reverse=True) if len(sales) else [year]
        fy = st.selectbox("Tax year", years, key="tax_8949_year")
        form = tl.form_8949(sales, fy)
        st.dataframe(form.head(200), hide_index=True, width="stretch", height=300,
                     column_config={"Proceeds": MONEY, "Cost basis": MONEY, "Adjustment amount": MONEY, "Gain or loss": MONEY})
        st.download_button(f"Download Form 8949-style CSV ({fy}, {len(form)} rows)", form.to_csv(index=False).encode(),
                           file_name=f"form8949_estimate_{fy}.csv", mime="text/csv", key="tax_8949_dl")
        caption_text(f"{fy}: {len(form)} sale rows, as of {inputs['as_of']:%a %b %-d, %Y}"
                   + (" (first 200 shown; the CSV has all)" if len(form) > 200 else "") + ". Date acquired = the holding-period "
                   "start (earlier after a wash sale); code W = wash sale, adjustment = the disallowed loss added back. "
                   + tl.NOTE_1099B)
