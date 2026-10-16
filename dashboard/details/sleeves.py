"""Trading Account tab: the two halves of the live account (WINNER['sleeves']): low volatility (monthly) + sector rotation
(weekly), from Reports/sleeve_ledger.json (written by each trade run) at the latest closes, and the halves' next targets."""
import os

import pandas as pd
import streamlit as st

from dashboard.settings import SIGNAL_CSV, WINNER

def sleeves_title():
    s = WINNER.get("sleeves") or {}
    return (f"Two halves · {s.get('lowvol', 0):.0%} Low volatility (monthly) + {s.get('sector', 0):.0%} Sector rotation "
            "(weekly, no overlap)")

@st.cache_data(show_spinner=False)
def _next_targets(mtime, picks_mtime):
    import sleeves
    picks = pd.read_csv(os.path.join(os.path.dirname(SIGNAL_CSV), "strategy_picks.csv"))
    day = str(picks["As_Of"].iloc[0])[:10]
    lv = picks.loc[picks["Provisional_Weight"].fillna(0) > 0, ["Symbol", "Provisional_Weight", "Close"]]
    sec = sleeves.sector_weights(day, exclude=set(lv["Symbol"]))
    sec = sec[sec > 0]
    close = sleeves.sector_inputs()["close"].loc[pd.Timestamp(day)]
    rows = [{"Half": sleeves.LABEL["lowvol"], "Symbol": s, "Weight_in_half_%": round(w * 100, 2), "Close": c}
            for s, w, c in lv.itertuples(index=False)]
    rows += [{"Half": sleeves.LABEL["sector"], "Symbol": s, "Weight_in_half_%": round(w * 100, 2), "Close": close.get(s)}
             for s, w in sec.items()]
    return day, pd.DataFrame(rows)

def render_sleeves(p):
    import sleeves
    led = None
    try:
        led = sleeves.load_ledger()
    except Exception as e:
        st.error(f"Sleeve ledger unreadable: {e}")
    close = p.by_symbol["Close"] if "Close" in p.by_symbol.columns else pd.Series(dtype=float)
    if led is None:
        st.info("The two halves start at the Fri Oct 16, 2026 rebalance (2:30 PM CT); the ledger is written by that run.")
    else:
        rows = []
        for h in sleeves.HALVES:
            L = led["halves"][h]
            for s, q in sorted(L["shares"].items()):
                if q > 0:
                    px = float(close.get(s, L["prices"].get(s, 0.0)))
                    rows.append({"Half": sleeves.LABEL[h], "Symbol": s, "Shares": q, "Close": px, "Value": q * px})
            rows.append({"Half": sleeves.LABEL[h], "Symbol": "(cash)", "Shares": None, "Close": None, "Value": float(L["cash"])})
        t = pd.DataFrame(rows)
        tot = t.groupby("Half")["Value"].transform("sum")
        t["Weight in half %"] = (100 * t["Value"] / tot).round(2)
        t["Weight of account %"] = (100 * t["Value"] / t["Value"].sum()).round(2)
        t["Value"] = t["Value"].round(2)
        st.caption(f"Ledger of the {led.get('day')} run (saved {led.get('saved_at', '?')}), valued at the latest closes. "
                   "Planned shares; the next trade run reconciles them with the account.")
        for h in sleeves.HALVES:
            part = t[t["Half"] == sleeves.LABEL[h]]
            st.markdown(f"**{sleeves.LABEL[h]}** · ${part['Value'].sum():,.0f} "
                        f"({100 * part['Value'].sum() / max(t['Value'].sum(), 1):.1f}% of the account)")
            st.dataframe(part.drop(columns="Half"), hide_index=True, width="stretch")
    try:
        day, nt = _next_targets(os.path.getmtime(SIGNAL_CSV),
                                os.path.getmtime(os.path.join(os.path.dirname(SIGNAL_CSV), "strategy_picks.csv")))
        st.markdown(f"**Targets at the {day} data** (the low-volatility half trades them only on its monthly rebalance; the "
                    "sector half every Friday, without the low-volatility names)")
        st.dataframe(nt, hide_index=True, width="stretch")
    except Exception as e:
        st.caption(f"Targets not available: {e}")
    cal = sleeves.calendar(pd.Timestamp.now(tz="America/Chicago").date(), 30)[:6]
    st.caption("Next trading days: " + "; ".join(f"{d:%a %b %-d}: " + " + ".join(
        "low volatility" if h == "lowvol" else "sector" for h in hs) for d, hs in cal))
