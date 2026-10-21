"""Stock Analysis tab: your notes, an editable table saved to Reports/my_notes.csv on every change.
Date (created) fills itself in; a note is deleted at midnight when its End Date has passed (blank End Date = kept)."""
from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard.settings import CT, NOTES_CSV

ACTIONS = ["Buy", "Sell", "Hold", "Watch"]
COLS = ["Stock", "Date", "End Date", "Note", "Action", "Price"]

def tidy(n):
    """Columns in order with the types the table edits (new rows come back from the table as text)."""
    n = n.reindex(columns=COLS)
    return n.assign(Stock=n["Stock"].astype("string").str.strip().str.upper(), Date=pd.to_datetime(n["Date"]).dt.date,
                    **{"End Date": pd.to_datetime(n["End Date"]).dt.date}, Price=pd.to_numeric(n["Price"]))

def read_notes(today=None):
    """Reports/my_notes.csv without expired notes (End Date before today); expired rows are removed from the file."""
    n = tidy(pd.read_csv(NOTES_CSV, dtype={"Stock": str, "Note": str, "Action": str}))
    today = today or datetime.now(CT).date()
    keep = n[~(pd.to_datetime(n["End Date"]) < pd.Timestamp(today))].reset_index(drop=True)
    if len(keep) != len(n) or "Time" in pd.read_csv(NOTES_CSV, nrows=0).columns:
        keep.to_csv(NOTES_CSV, index=False)
    return keep

@st.fragment
def render_notes():
    """The notes table: add a row at the bottom, edit any cell, select rows and delete them. Every change is saved at
    once and the table reloads from the file (a fragment: only this section reruns)."""
    notes = read_notes()
    # No per-run defaults in column_config: a changing default changes the editor's identity and resets it mid-edit.
    edited = tidy(st.data_editor(
        notes, key="notes_editor", num_rows="dynamic", hide_index=True, width="stretch",
        column_config={
            "Stock": st.column_config.TextColumn("Stock", max_chars=10),
            "Date": st.column_config.DateColumn("Date", format="MMM D, YYYY", disabled=True, help="Filled in when the note is created"),
            "End Date": st.column_config.DateColumn("End Date", format="MMM D, YYYY", help="Deleted at midnight after this day; blank keeps it"),
            "Note": st.column_config.TextColumn("Note", width="large"),
            "Action": st.column_config.SelectboxColumn("Action", options=ACTIONS),
            "Price": st.column_config.NumberColumn("Price", format="$%.2f", min_value=0.0),
        }))
    edited["Date"] = edited["Date"].where(edited["Date"].notna(), datetime.now(CT).date())
    if edited.to_csv(index=False) != notes.to_csv(index=False):
        edited.to_csv(NOTES_CSV, index=False)
        del st.session_state["notes_editor"]
        st.rerun(scope="fragment")
