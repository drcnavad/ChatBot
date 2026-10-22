"""Trading Account tab: the tax view (tax_lots.py, an estimate, counted from tax_lots.TAX_START)."""
from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard.details.live_holdings import holdings_this_run
from dashboard.settings import CT
from dashboard.style import (BAD, CAUTION, GOOD, INK, INT, MONEY, SHARES, caption_text, info_text, stat_cards, table, tone, usd,
    warning_text)

# ---------------------------------------------------------------------------- tax view (estimate, read-only)
@st.cache_data(ttl=3600, max_entries=2, show_spinner=False)
def _read_tax_inputs(hour_key):
    """Every account activity + order client ids from the live account (GET only), once per hour_key (the hour + the
    number of fills, so a new fill is read at once and never shows as a pre-start holding)."""
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

@st.fragment(run_every=60)      # unrealized gains follow the live prices
def render_tax_view(p):
    """Trading Account tab: realized / unrealized gains (short vs long term), wash sales, harvest list, estimated tax, Form 8949
    export, from the live account's activity since tax_lots.TAX_START (read-only, hourly). An estimate, not tax advice."""
    import tax_lots as tl
    data, err = holdings_this_run()
    if err:
        caption_text("The tax view needs the live holdings, which are not available right now (see Live holdings above).")
        return
    key = f"{datetime.now(CT):%Y-%m-%d %H}-{len(data['fills'])}"
    positions = tuple(data["positions"])
    try:
        rep = _tax_report(key, "FIFO", positions)
    except Exception as e:
        info_text(f"Tax view unavailable right now ({type(e).__name__}: {str(e)[:200]}). It tries again next hour.")
        return
    y = tl.ytd(rep)
    year = y["year"]
    start = rep.get("start")
    scope = f"since {start:%b %-d}" if start is not None and start.year == year else str(year)     # card labels
    when = f"since {start:%a %b %-d, %Y}" if start is not None else f"in {year}"
    issues = rep["issues"]
    if len(issues):
        warning_text(f"Data check: {len(issues)} issue(s) found, so some numbers below may be off. Details in the table.")
        table(issues, max_h=400)
    stat_cards([(f"Realized {scope} short-term", usd(y["realized_st"]), tone(y["realized_st"])),
                (f"Realized {scope} long-term", usd(y["realized_lt"]), tone(y["realized_lt"])),
                ("Unrealized short-term", usd(y["unrealized_st"]), tone(y["unrealized_st"])),
                ("Unrealized long-term", usd(y["unrealized_lt"]), tone(y["unrealized_lt"])),
                (f"Wash-sale loss deferred {scope}", usd(y["wash_disallowed"], sign=False), CAUTION if y["wash_disallowed"] > 0.005 else INK),
                (f"Dividends {scope} (qualified est.)", f"{usd(y['dividends'], sign=False)} ({usd(y['qualified'], sign=False)})", INK),
                (f"Interest {scope}", usd(y["interest"], sign=False), INK),
                (f"Fees {scope}", usd(y["fees"], sign=False), INK)])
    tabs = st.tabs(["By year", "Open lots", "Tax-loss harvest", "Wash sales", "Estimated tax", "Dividends, interest, fees",
                    "Lot method", "Form 8949 export"])
    sales = rep["sales"]
    with tabs[0]:
        by = tl.realized_by_year(sales)
        if by.empty:
            info_text(f"No sales of lots bought {when} yet, so nothing is realized.")
        table(by.round(2), ["Short-term gain", "Long-term gain", "Total gain"],
              cols={c: MONEY for c in by.columns if c not in ("Year", "Lot sales")} | {"Year": INT})
    with tabs[1]:
        lots = tl.open_lots_view(rep["lots"])
        if lots.empty:
            info_text("No open lots.")
        else:
            shown = lots.assign(**{c: [f"{d:%b %-d, %Y}" for d in lots[c]] for c in ("Bought", "Holding from", "Long-term on")})
            shown["Gain"] = shown["Gain"].round(2) + 0.0                        # no "-$0.00"
            table(shown, ["Gain"], cols={**dict.fromkeys(("Basis / share", "Basis", "Price", "Value", "Gain"), MONEY),
                                         "Shares": SHARES, "Note": st.column_config.TextColumn(width="medium")})
            soon = tl.turning_long_term(rep, 60)
            if len(soon):
                warning_text(f"{len(soon)} lot(s) turn long-term within 60 days: "
                           + "; ".join(f"{r['Symbol']} {r['Shares']:g} sh on {r['Long-term on']:%b %-d} ({r['Window']}; {r['Hint']})"
                                       for _, r in soon.iterrows()))
    with tabs[2]:
        planned = {s for s, a in (p.plan or {}).items() if a[0] in ("Buy", "Keep", "Take over")}
        ok, excluded = tl.harvest(rep, planned)
        if ok.empty:
            st.success("No position to harvest right now: " + ("nothing is at a loss." if excluded.empty else
                                                               "every position at a loss is excluded (wash-sale risk, below)."))
        else:
            table(_harvest_view(ok), ["Loss", "Short-term loss", "Long-term loss"], cols=HARVEST_COLS)
        if len(excluded):
            caption_text("Excluded (selling now would likely be a wash sale):")
            table(_harvest_view(excluded), ["Loss"], cols=HARVEST_COLS)
    with tabs[3]:
        summ, w = tl.wash_summary(rep["washes"], year)
        if summ.empty:
            st.success(f"No wash sales on sales {when}.")
        else:
            wash = {"Disallowed loss": MONEY, "Shares": SHARES}
            table(summ.round(2).assign(**{"Last loss sale": [f"{d:%b %-d, %Y}" for d in summ["Last loss sale"]]}), max_h=420, cols=wash)
            detail = w.assign(**{c: [f"{d:%b %-d, %Y}" for d in w[c]] for c in ("Loss sale", "Replacement bought")}).drop(columns="Replacement lot")
            table(detail.round(2).iloc[::-1], height=300, cols=wash)
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
    with tabs[5]:
        inc = rep["income"]
        if inc.empty and rep["fees"].empty:
            info_text(f"No dividends, interest or fees {when}.")
        else:
            by = inc.groupby(["Year", "Type"], as_index=False).agg(Amount=("Amount", "sum"), **{"Qualified (est.)": ("Qualified (est.)", "sum")})
            fees = rep["fees"].assign(Year=[d.year for d in rep["fees"]["Date"]]).groupby(["Year", "Type"], as_index=False)["Amount"].sum()
            fees["Amount"] = -fees["Amount"]
            income = pd.concat([by, fees], ignore_index=True).sort_values(["Year", "Type"], ascending=[False, True])
            money = {"Amount": MONEY, "Qualified (est.)": MONEY, "Year": INT}
            table(income.round(2), ["Amount"], cols=money)
            table(inc.assign(Date=[f"{d:%b %-d, %Y}" for d in inc["Date"]]).iloc[::-1].round(2), max_h=300,
                  cols={**money, "Note": st.column_config.TextColumn(width="large")})
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
        table(cmp_.round(2), [c for c in cmp_.columns if c.startswith(str(year))], cols={c: MONEY for c in cmp_.columns if c != "Method"})
    with tabs[7]:
        years = sorted({int(v) for v in sales["Year"]}, reverse=True) if len(sales) else [year]
        fy = st.selectbox("Tax year", years, key="tax_8949_year")
        form = tl.form_8949(sales, fy)
        table(form.head(200), height=300, cols=dict.fromkeys(("Proceeds", "Cost basis", "Adjustment amount", "Gain or loss"), MONEY))
        st.download_button(f"Download Form 8949-style CSV ({fy}, {len(form)} rows)", form.to_csv(index=False).encode(),
                           file_name=f"form8949_estimate_{fy}.csv", mime="text/csv", key="tax_8949_dl")
