"""Trading Account tab: intraday drop-guard decisions and hit rate (read-only)."""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from dashboard.settings import CT, REPORTS
from dashboard.style import MONEY, SPCT, caption_text, info_text, table

LOG = os.path.join(REPORTS, "drop_guard_log.csv")
CFG = os.path.join(REPORTS, "drop_guard_config.json")


def _cfg():
    try:
        with open(CFG) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


@st.cache_data(ttl=120, max_entries=2, show_spinner=False)
def _read_log(_mtime):
    import db
    if not db.exists(LOG):
        return pd.DataFrame()
    df = db.read_csv(LOG)
    return df if df is not None else pd.DataFrame()


def render_drop_guard():
    """Show recent drop-guard evaluations, AI decisions, sells/replacements, and a simple hit rate."""
    cfg = _cfg()
    enabled = cfg.get("enabled", True)
    dry = cfg.get("dry_run", True)
    drop = float(cfg.get("drop_pct", 0.05)) * 100
    conf = float(cfg.get("min_confidence", 0.75))
    status = ("LIVE" if enabled and not dry else "DRY-RUN" if dry else "DISABLED")
    info_text(
        f"Drop guard · {status}: every 15 min ~8:30 AM–2:15 PM CT, any held stock down ≥{drop:g}% vs prior close "
        f"is reviewed by the LLM (sell if SELL and confidence ≥{conf:g}). After a fill, proceeds buy the top-ranked "
        f"eligible replacement (do_not_buy, ≤{float(cfg.get('max_weight', 0.15))*100:g}% weight, above SMA200, "
        f"not sold today, not itself down ≥{drop:g}%). Cash waits for 2:30 if nothing qualifies."
    )
    try:
        import db
        mtime = db.mtime(LOG) if db.exists(LOG) else 0
    except Exception:
        mtime = 0
    try:
        log = _read_log(mtime)
    except Exception as e:
        info_text(f"Drop-guard log unavailable ({type(e).__name__}: {str(e)[:120]}).")
        return
    if log is None or log.empty:
        caption_text("No drop-guard checks logged yet.")
        return
    log = log.copy()
    # hit rate: among SOLD / WOULD_SELL with a next-day close, was Next_Close below Sell_Limit? (sale was right)
    sold = log[log["Action"].astype(str).str.contains("SOLD|WOULD_SELL", na=False)].copy()
    hit = ""
    if len(sold):
        sc = pd.to_numeric(sold.get("Sell_Limit"), errors="coerce")
        nc = pd.to_numeric(sold.get("Next_Close"), errors="coerce")
        ok = sc.notna() & nc.notna() & (sc > 0)
        if ok.any():
            wins = int((nc[ok] < sc[ok]).sum())
            n = int(ok.sum())
            hit = f" Hit rate (next close < sell limit): {wins}/{n} = {wins/n:.0%}."
        else:
            hit = " Hit rate pending next-day closes."
    caption_text(
        f"{len(log)} evaluations · {int((log['Triggered'].astype(str) == 'yes').sum())} triggered · "
        f"{len(sold)} sell actions.{hit}"
    )
    show = log.sort_values("Checked_At_CT", ascending=False).head(80)
    cols = [c for c in (
        "Checked_At_CT", "Symbol", "Vs_Prev_%", "Triggered", "AI_Action", "AI_Confidence",
        "Action", "Sell_Limit", "Sell_Proceeds", "Replace_Symbol", "Replace_Qty",
        "Replace_Note", "Close_Price", "Next_Close", "Dry_Run", "AI_Reasons", "Note",
    ) if c in show.columns]
    cfg_fmt = {"Sell_Limit": MONEY, "Sell_Proceeds": MONEY, "Close_Price": MONEY,
               "Next_Close": MONEY, "Replace_Limit": MONEY, "AI_Confidence": SPCT,
               "Vs_Prev_%": SPCT, "Close_Vs_Sell_%": SPCT, "Next_Vs_Sell_%": SPCT}
    # Vs_Prev_% already stored as percent points (e.g. -5.2); SPCT may expect fraction — show raw
    table(show[cols], max_h=360, cols={k: v for k, v in cfg_fmt.items() if k in cols and k not in (
        "Vs_Prev_%", "AI_Confidence", "Close_Vs_Sell_%", "Next_Vs_Sell_%")}, key="drop_guard_log")
