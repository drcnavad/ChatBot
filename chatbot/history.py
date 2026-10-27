"""Chat history in SQLite (db.py's data/stock_analysis.db, table chat_history), kept 1 year then deleted automatically.
Only the visible messages are saved; the PIN is never part of them."""
import time

import db

KEEP_SECS = 365 * 24 * 3600

def _con():
    con = db.connect()
    con.execute("CREATE TABLE IF NOT EXISTS chat_history (ts REAL NOT NULL, role TEXT NOT NULL, text TEXT NOT NULL)")
    con.execute("CREATE INDEX IF NOT EXISTS chat_history_ts ON chat_history (ts)")
    return con

def add(role, text):
    try:
        con = _con()
        with con:
            con.execute("INSERT INTO chat_history VALUES (?, ?, ?)", (time.time(), role, str(text)))
            con.execute("DELETE FROM chat_history WHERE ts < ?", (time.time() - KEEP_SECS,))
        con.close()
    except Exception:
        pass                                 # history is best-effort: never break the chat

def recent(n=20):
    try:
        con = _con()
        rows = con.execute("SELECT role, text FROM chat_history ORDER BY ts DESC, rowid DESC LIMIT ?", (int(n),)).fetchall()
        con.close()
        return [(r, t) for r, t in reversed(rows)]
    except Exception:
        return []
