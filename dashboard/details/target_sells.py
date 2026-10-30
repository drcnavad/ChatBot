"""Dashboard panel: target limit sells + sold_block (no-rebuy after any live sell)."""
import json
import os

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS = os.path.join(ROOT, "Reports")
CFG = os.path.join(REPORTS, "target_sells_config.json")
STATE = os.path.join(REPORTS, "target_sells_state.json")
BLOCK = os.path.join(REPORTS, "sold_block.json")
TS_LOG = os.path.join(REPORTS, "target_sells_log.csv")
SB_LOG = os.path.join(REPORTS, "sold_block_log.csv")


def _load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def render_target_sells():
    import db
    cfg = _load(CFG, {})
    state = _load(STATE, {})
    blocks = _load(BLOCK, {})
    enabled = cfg.get("enabled", False)
    dry = cfg.get("dry_run", True)
    status = ("LIVE" if enabled and not dry else "DRY-RUN" if enabled else "OFF")
    targets = cfg.get("targets") or {}
    start = cfg.get("window_start_ct", [3, 0])
    end = cfg.get("window_end_ct", [19, 0])
    st.markdown(
        f"**Target sells · {status}:** every 15 min "
        f"~{int(start[0]):02d}:{int(start[1]):02d}–{int(end[0]):02d}:{int(end[1]):02d} CT on session days, "
        f"places/refreshes a DAY+extended_hours limit sell for each configured name "
        f"(Alpaca has no GTC+extended; the job renews each day). "
        f"drop_guard / 2:30 cancel the target before selling the same name. "
        f"**Sold block:** after any live sell, that symbol is blocked from all buys for "
        f"{blocks.get('business_days', 5)} business days (spare-cash, replacements, rebalance)."
    )
    if targets:
        rows = []
        orders = (state.get("orders") or {})
        for sym, e in targets.items():
            o = orders.get(str(sym).upper(), {})
            rows.append({
                "Symbol": sym, "Limit": e.get("limit"), "Qty": e.get("qty", "full"),
                "Open order": o.get("order_id", ""), "Day": o.get("day", ""),
                "Last": o.get("at", ""),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.caption("No target sells configured.")

    active = blocks.get("blocks") or {}
    st.subheader("Sold block (no rebuy)")
    if active:
        brows = []
        for sym, e in sorted(active.items()):
            brows.append({
                "Symbol": sym, "Sold on": e.get("sold_on"), "Blocked through": e.get("through"),
                "Source": e.get("source"), "Note": e.get("note"),
            })
        st.dataframe(pd.DataFrame(brows), hide_index=True, use_container_width=True)
    else:
        st.caption("No active sold blocks.")

    if db.exists(TS_LOG):
        log = db.read_csv(TS_LOG)
        if log is not None and not log.empty:
            st.subheader("Target sell log")
            st.dataframe(log.tail(30), hide_index=True, use_container_width=True)
    if db.exists(SB_LOG):
        slog = db.read_csv(SB_LOG)
        if slog is not None and not slog.empty:
            st.subheader("Sold-block log")
            st.dataframe(slog.tail(30), hide_index=True, use_container_width=True)
