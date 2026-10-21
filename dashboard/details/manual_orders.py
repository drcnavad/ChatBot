"""Strategy tab: manual orders (manual_trade.py, Reports/manual_orders.json) and the active buy blocks
(Reports/manual_block.json). Display only."""
import pandas as pd
import streamlit as st

def render_manual_orders():
    import manual_trade as mt
    blocks = mt.load_blocks()
    if blocks:
        st.markdown("**Active buy blocks** (the pipeline does not buy these)")
        st.dataframe(pd.DataFrame([{"Symbol": s, "Reference close": e.get("reference_close"),
                                    "Blocked until price": e.get("block_until_price"), "Drop %": e.get("drop_pct"),
                                    "Added": e.get("added")} for s, e in sorted(blocks.items())]),
                     hide_index=True, width="stretch")
    else:
        st.caption("No active buy blocks.")
    rows = mt._read(mt.ORDERS_JSON, [])
    if rows:
        t = pd.DataFrame(rows[::-1][:20])
        cols = [c for c in ("time", "action", "symbol", "qty", "limit", "value", "status", "dry_run") if c in t.columns]
        st.markdown("**Manual orders** (newest first, last 20)")
        st.dataframe(t[cols], hide_index=True, width="stretch")
    else:
        st.caption("No manual orders yet.")
