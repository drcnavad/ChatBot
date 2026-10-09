"""Home tab: your notes, an editable table saved to Reports/my_notes.csv on every change."""
from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard.settings import CT, NOTES_CSV

ACTIONS = ["Buy", "Sell", "Hold", "Watch"]

def tidy(n):
    """Stock (upper case) / Date / Time / Price with the types the table edits (new rows come back from the table as text)."""
    return n.assign(Stock=n["Stock"].astype("string").str.strip().str.upper(), Date=pd.to_datetime(n["Date"]).dt.date,
                    Time=pd.to_datetime(n["Time"].astype("string"), format="mixed").dt.time,
                    Price=pd.to_numeric(n["Price"]))

def read_notes():
    """Reports/my_notes.csv: Stock, Date, Time, Note, Action, Price."""
    return tidy(pd.read_csv(NOTES_CSV, dtype={"Stock": str, "Note": str, "Action": str}))

@st.fragment
def render_notes():
    """The notes table: add a row at the bottom, edit any cell, select rows and delete them. Every change is saved at
    once and the table reloads from the file (a fragment: only this section reruns)."""
    notes = read_notes()
    now = datetime.now(CT)
    edited = tidy(st.data_editor(
        notes, key="notes_editor", num_rows="dynamic", hide_index=True, width="stretch",
        column_config={
            "Stock": st.column_config.TextColumn("Stock", max_chars=10),
            "Date": st.column_config.DateColumn("Date", format="MMM D, YYYY", default=now.date()),
            "Time": st.column_config.TimeColumn("Time", format="h:mm a", step=60, default=now.time().replace(second=0, microsecond=0)),
            "Note": st.column_config.TextColumn("Note", width="large"),
            "Action": st.column_config.SelectboxColumn("Action", options=ACTIONS),
            "Price": st.column_config.NumberColumn("Price", format="$%.2f", min_value=0.0),
        }))
    if edited.to_csv(index=False) != notes.to_csv(index=False):
        edited.to_csv(NOTES_CSV, index=False)
        del st.session_state["notes_editor"]
        st.rerun(scope="fragment")
    st.caption("Click the empty bottom row (or + above the table) to add a note (today's date and time are filled in), click a cell to edit it, "
               "tick rows on the left and press the trash icon to delete them. Saved automatically to Reports/my_notes.csv.")
