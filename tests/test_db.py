"""db.py (SQLite storage that replaced the Reports CSV files) on a temp database: save/load round trip keeps values,
dtypes, dates and the index; read_csv options work; mode='a' appends like a file; exists / mtime / remove / glob; a
leftover file is read until the first write imports it and deletes it; non-project paths stay plain files; WAL mode.
Run: python tests/run_tests.py  (or python tests/test_db.py)"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["STOCK_ANALYSIS_DB"] = os.path.join(tempfile.mkdtemp(), "t.db")
import pandas as pd
import db

FAIL = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        FAIL.append(name)

P = os.path.join(ROOT, "Reports", "zz_dbtest_tmp.csv")
df = pd.DataFrame({"Date": pd.to_datetime(["2026-10-23", "2026-10-26"]), "Symbol": ["KO", "A,B \"q\""],
                   "Close": [61.123456789, float("nan")], "N": [1, 2]})
try:
    df.pipe(db.to_csv, P, index=False)
    check("no file written for a project path", not os.path.exists(P))
    back = db.read_csv(P, parse_dates=["Date"])
    check("round trip equal (values, NaN, quotes, dtypes)", back.equals(df) and (back.dtypes == df.dtypes).all())
    check("float_precision round_trip", db.read_csv(P, float_precision="round_trip")["Close"][0] == 61.123456789)
    check("usecols / nrows", list(db.read_csv(P, usecols=["Symbol"], nrows=1).columns) == ["Symbol"])
    t1 = db.mtime(P)
    df.head(1).pipe(db.to_csv, P, mode="a", header=False, index=False)
    check("append adds rows", len(db.read_csv(P)) == 3 and db.mtime(P) > t1)
    ix = df.set_index("Symbol")
    ix.pipe(db.to_csv, P)
    check("index kept", db.read_csv(P, index_col=0, parse_dates=["Date"]).equals(ix))
    check("load_table name", db.load_table("zz_dbtest_tmp").shape == (2, 4))
    check("glob finds the table", P in db.glob(os.path.join(ROOT, "Reports", "zz_dbtest_*.csv")))
    db.remove(P)
    check("remove", not db.exists(P))
    df.to_csv(P, index=False)                     # a leftover real file
    check("leftover file is read", db.read_csv(P).shape == (2, 4))
    df.pipe(db.to_csv, P, index=False)
    check("first write imports it and deletes the file", not os.path.exists(P) and db.exists(P))
    out = os.path.join(tempfile.mkdtemp(), "x.csv")
    df.pipe(db.to_csv, out, index=False)
    check("outside the project: a plain file", os.path.exists(out) and not db.has_table(db.table_name(out)))
    with db.connect() as con:
        check("WAL mode", con.execute("PRAGMA journal_mode").fetchone()[0] == "wal")
    pd.DataFrame(columns=["A", "B"]).pipe(db.to_csv, P, index=False)
    e = db.read_csv(P)
    check("header-only table", list(e.columns) == ["A", "B"] and e.empty)
finally:
    db.remove(P)
print("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED")
sys.exit(1 if FAIL else 0)
