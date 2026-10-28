"""SQLite storage for every table the project used to keep as a CSV file (Chirag, Oct 26, 2026).

One file: data/stock_analysis.db (override: STOCK_ANALYSIS_DB; the tests use a temp one). WAL mode + a 30 s busy
timeout, so the dashboard, the launchd jobs and the notebooks can read while one of them writes; each write is one
transaction (readers see the old table or the new one, never half of it).

Two ways in:
  save_table(name, df, **to_csv_kwargs) / load_table(name, **read_csv_kwargs)   e.g. load_table("signal_analysis")
  read_csv(path, ...) / to_csv(df, path, ...) / exists(path) / mtime(path) / remove(path) / glob(pattern)
      drop-in for pd.read_csv / df.to_csv / os.path.exists / os.path.getmtime on the old "*.csv" paths: a path is
      just the table's address (Reports/signal_analysis.csv -> table "signal_analysis"; Reports/cache/x.csv ->
      "cache/x"; data/x.csv -> ":data/x"). Anything that is not a .csv path inside the project (URLs, buffers,
      downloads, a CSV you export or pass in from elsewhere, the tests' temp files) goes straight to pandas / the file.

Values are stored exactly as the CSV text had them (one TEXT column per CSV column, header kept in _tables) and are
read back through pandas' own CSV parser, so every read_csv option (parse_dates, dtype, usecols, index_col, nrows,
float_precision ...) behaves as before and numbers / dates / index round-trip exactly. mode="a" appends rows like a
file append did. A .csv file still on disk with no table yet is read (and then imported on the next write).

Daily OHLCV lives in the typed table bars_daily (PRIMARY KEY Symbol, Date): load_bars / append_bars.
Refreshes are append-only (INSERT OR IGNORE for history; INSERT OR REPLACE for the newest session).
benchmark_prices stays wide (unique Date) and is written with append_benchmark_prices."""
import csv
import fnmatch
import io
import json
import os
import sqlite3
import threading
import time

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "Reports")
BUSY_MS = 30_000
# Typed daily OHLCV table (see append_bars / load_bars at bottom). Name checked early in save_table/load_table.
BARS_DAILY = "bars_daily"
BARS_COLS = ("Symbol", "Date", "Open", "High", "Low", "Close", "Volume")
BARS_OVERLAP_DAYS = 5
_BARS_META = "_bars_meta"
_lock = threading.Lock()

def db_path():
    return os.environ.get("STOCK_ANALYSIS_DB") or os.path.join(ROOT, "data", "stock_analysis.db")

_ready = set()

def connect():
    path = db_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    con = sqlite3.connect(path, timeout=BUSY_MS / 1000, isolation_level=None, check_same_thread=False)
    con.execute(f"PRAGMA busy_timeout={BUSY_MS}")
    if path not in _ready:
        with _lock:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("CREATE TABLE IF NOT EXISTS _tables (name TEXT PRIMARY KEY, columns TEXT NOT NULL, "
                        "rows INTEGER NOT NULL, updated_at REAL NOT NULL)")
            _ready.add(path)
    con.execute("PRAGMA synchronous=NORMAL")
    return con

# ---- path <-> table name ------------------------------------------------------------------------------------------
def is_csv_path(p):
    if isinstance(p, os.PathLike):
        p = os.fspath(p)
    return (isinstance(p, str) and p.lower().endswith(".csv") and "://" not in p
            and os.path.abspath(p).startswith(ROOT + os.sep))         # files elsewhere (exports, test temp files) stay files

def table_name(p):
    """Reports/x.csv -> 'x'; Reports/cache/y.csv -> 'cache/y'; <project>/data/z.csv -> ':data/z'; else the absolute path."""
    a = os.path.abspath(os.fspath(p))[:-4]
    for base, prefix in ((REPORTS, ""), (ROOT, ":")):
        if a.startswith(base + os.sep):
            return prefix + os.path.relpath(a, base).replace(os.sep, "/")
    return a

def path_of(name):
    """The old file path of table `name` (inverse of table_name)."""
    if os.path.isabs(name):
        return name + ".csv"
    if name.startswith(":"):
        return os.path.join(ROOT, *name[1:].split("/")) + ".csv"
    return os.path.join(REPORTS, *name.split("/")) + ".csv"

def _q(ident):
    return '"' + str(ident).replace('"', '""') + '"'

def _sql_cols(header):
    """Real column names where usable, else c0, c1 ... (blank / duplicate CSV headers)."""
    seen, out = set(), []
    for i, h in enumerate(header):
        k = str(h).strip().lower()
        out.append(h if h and k not in seen and not k.startswith("_") else f"_c{i}")
        seen.add(k)
    return out

# ---- the table API ------------------------------------------------------------------------------------------------
def _meta(con, name):
    r = con.execute("SELECT columns, rows, updated_at FROM _tables WHERE name=?", (name,)).fetchone()
    return (json.loads(r[0]), r[1], r[2]) if r else None

def has_table(name):
    with connect() as con:
        return _meta(con, name) is not None

def updated_at(name):
    with connect() as con:
        m = _meta(con, name)
    return m[2] if m else None

def tables():
    with connect() as con:
        return [r[0] for r in con.execute("SELECT name FROM _tables ORDER BY name")]

def _write_lines(name, lines, append=False):
    """lines = list of CSV records (lists of strings); the first is the header unless appending to an existing table."""
    with connect() as con:
        con.execute("BEGIN IMMEDIATE")
        try:
            m = _meta(con, name) if append else None
            if m:
                header, rows = m[0], lines
            else:
                header, rows = (lines[0] if lines else []), lines[1:]
            width = max([len(header)] + [len(r) for r in rows])
            if width > len(header):                     # a wider row (as a file append would allow): extra blank columns
                header = header + [""] * (width - len(header))
            cols = _sql_cols(header)
            if m:
                have = [r[1] for r in con.execute(f"PRAGMA table_info({_q(name)})")]
                for c in cols[len(have):]:
                    con.execute(f"ALTER TABLE {_q(name)} ADD COLUMN {_q(c)} TEXT")
            else:
                con.execute(f"DROP TABLE IF EXISTS {_q(name)}")
                con.execute(f"CREATE TABLE {_q(name)} (" + ", ".join(f"{_q(c)} TEXT" for c in cols) + ")" if cols
                            else f"CREATE TABLE {_q(name)} (_empty TEXT)")
            if rows and cols:
                ph = ",".join("?" * len(cols))
                con.executemany(f"INSERT INTO {_q(name)} ({','.join(_q(c) for c in cols)}) VALUES ({ph})",
                                [list(r) + [""] * (len(cols) - len(r)) for r in rows])
            n = con.execute(f"SELECT COUNT(*) FROM {_q(name)}").fetchone()[0] if cols else 0
            now = max(time.time(), ((m[2] if m else 0) or 0) + 1e-6)
            con.execute("INSERT OR REPLACE INTO _tables VALUES (?,?,?,?)", (name, json.dumps(header), n, now))
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise

def table_text(name):
    """The table as the CSV text it was saved from (None if there is no such table)."""
    with connect() as con:
        m = _meta(con, name)
        if m is None:
            return None
        header = m[0]
        rows = con.execute(f"SELECT * FROM {_q(name)} ORDER BY rowid").fetchall() if header else []
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    if header:
        w.writerow(header)
    w.writerows([r[:len(header)] for r in rows])
    return buf.getvalue()

def save_table(name, df, mode="w", header=True, **to_csv_kwargs):
    """Store a DataFrame as table `name` (same options as DataFrame.to_csv; mode='a' appends rows)."""
    if name == BARS_DAILY:
        raise ValueError("bars_daily is append-only: use db.append_bars(df), never save_table/replace")
    append = mode.startswith("a") and has_table(name)
    if not append and (header is False or header is None):
        raise ValueError(f"header=False on new table {name!r} would store its first data row as the header")
    text = df.to_csv(header=header, **to_csv_kwargs)
    lines = list(csv.reader(io.StringIO(text)))
    if append and header is not False and header is not None and lines:
        pass                                         # a header written in append mode is a data row in a file too
    _write_lines(name, lines, append=append)

def load_table(name, **read_csv_kwargs):
    """Table `name` as a DataFrame (same options as pd.read_csv). FileNotFoundError if there is no such table."""
    if name == BARS_DAILY:                                 # typed OHLCV table: never through the CSV text path
        return load_bars()
    text = table_text(name)
    if text is None:
        raise FileNotFoundError(f"no table {name!r} in {db_path()}")
    if not text.strip():
        raise pd.errors.EmptyDataError("No columns to parse from file")
    return pd.read_csv(io.StringIO(text), **read_csv_kwargs)

def drop_table(name):
    with connect() as con:
        con.execute("BEGIN IMMEDIATE")
        con.execute(f"DROP TABLE IF EXISTS {_q(name)}")
        con.execute("DELETE FROM _tables WHERE name=?", (name,))
        con.execute("COMMIT")

def import_csv_file(path, name=None):
    """Copy one CSV file into its table, line for line."""
    with open(path, newline="", encoding="utf-8") as f:
        lines = list(csv.reader(f))
    _write_lines(name or table_name(path), lines)
    return name or table_name(path)

# ---- drop-ins for the old file calls -----------------------------------------------------------------------------
def read_csv(path, *args, **kw):
    if not is_csv_path(path):
        return pd.read_csv(path, *args, **kw)
    name = table_name(path)
    if has_table(name):
        return load_table(name, *args, **kw) if not args else pd.read_csv(io.StringIO(table_text(name)), *args, **kw)
    return pd.read_csv(path, *args, **kw)          # not imported yet (or a test fixture file): the file itself

def to_csv(df, path=None, *args, mode="w", header=True, **kw):
    if path is None or not is_csv_path(path):
        return df.to_csv(path, *args, mode=mode, header=header, **kw)
    name = table_name(path)
    if mode.startswith("a") and not has_table(name) and os.path.exists(os.fspath(path)):
        import_csv_file(path, name)                   # appending to a not-yet-imported file: bring its rows first
    save_table(name, df, mode=mode, header=header, **kw)
    if os.path.exists(os.fspath(path)):                # the table is the copy now; no stale file left beside it
        os.remove(os.fspath(path))

def exists(path):
    if not is_csv_path(path):
        return os.path.exists(path)
    return has_table(table_name(path)) or os.path.exists(os.fspath(path))

def mtime(path):
    """When the table was last written (the file's mtime for a file not imported yet). OSError when neither exists."""
    if is_csv_path(path):
        t = updated_at(table_name(path))
        if t is not None:
            return t
    return os.path.getmtime(path)

def remove(path):
    if is_csv_path(path):
        drop_table(table_name(path))
    if os.path.exists(os.fspath(path)):
        os.remove(os.fspath(path))

def glob(pattern):
    """Paths of the tables (and leftover files) matching a .csv path pattern, sorted."""
    import glob as _g
    found = set(_g.glob(os.fspath(pattern)))
    if is_csv_path(pattern):
        pat = table_name(pattern)
        for t in tables():
            if fnmatch.fnmatchcase(t, pat):
                found.add(path_of(t))
    return sorted(found)

def append_rows(path, rows, columns):
    """csv.writer-style append of plain rows (a header first when the table is new)."""
    if not is_csv_path(path):                          # a file outside the project: a plain file append
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(columns)
            w.writerows(rows)
        return
    to_csv(pd.DataFrame(rows, columns=columns), path, mode="a", header=not exists(path), index=False)


# ---- daily OHLCV bars (typed table; append-only) --------------------------------------------------------------
# Single source of truth for every stock / ETF daily bar. Not a CSV-emulation table: REAL columns + PRIMARY KEY
# (Symbol, Date) so refreshes can INSERT OR IGNORE / INSERT OR REPLACE without rebuilding. Intraday bars are never
# stored here (dashboard fetches 5-/15-minute bars live for display only).
# (BARS_DAILY / BARS_COLS / BARS_OVERLAP_DAYS / _BARS_META defined near top)

def ensure_bars_daily():
    """Create bars_daily + its Date index + the mtime stamp table if they are missing. Never drops existing rows."""
    with connect() as con:
        con.execute("BEGIN IMMEDIATE")
        try:
            con.execute(
                f"CREATE TABLE IF NOT EXISTS {_q(BARS_DAILY)} ("
                "Symbol TEXT NOT NULL, Date TEXT NOT NULL, "
                "Open REAL, High REAL, Low REAL, Close REAL, Volume REAL, "
                "PRIMARY KEY (Symbol, Date))"
            )
            con.execute(f"CREATE INDEX IF NOT EXISTS bars_daily_date ON {_q(BARS_DAILY)}(Date)")
            con.execute(f"CREATE TABLE IF NOT EXISTS {_q(_BARS_META)} "
                        "(key TEXT PRIMARY KEY, value REAL NOT NULL)")
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise

def bars_mtime():
    """When bars_daily was last appended to (0.0 if the table is empty / missing)."""
    ensure_bars_daily()
    with connect() as con:
        r = con.execute(f"SELECT value FROM {_q(_BARS_META)} WHERE key='updated_at'").fetchone()
    return float(r[0]) if r else 0.0

def has_bars():
    ensure_bars_daily()
    with connect() as con:
        return con.execute(f"SELECT 1 FROM {_q(BARS_DAILY)} LIMIT 1").fetchone() is not None

def bars_symbol_max_dates(symbols=None):
    """{Symbol: 'YYYY-MM-DD'} of the latest stored date per symbol (empty dict when none)."""
    ensure_bars_daily()
    with connect() as con:
        if symbols is None:
            rows = con.execute(f"SELECT Symbol, MAX(Date) FROM {_q(BARS_DAILY)} GROUP BY Symbol").fetchall()
        else:
            syms = list(dict.fromkeys(symbols))
            if not syms:
                return {}
            ph = ",".join("?" * len(syms))
            rows = con.execute(
                f"SELECT Symbol, MAX(Date) FROM {_q(BARS_DAILY)} WHERE Symbol IN ({ph}) GROUP BY Symbol",
                syms).fetchall()
    return {s: d for s, d in rows if d}

def load_bars(symbols=None, start=None, end=None):
    """Daily OHLCV as a DataFrame (Date = datetime64, sorted by Symbol, Date). Empty frame when nothing matches."""
    ensure_bars_daily()
    where, args = [], []
    if symbols is not None:
        syms = list(dict.fromkeys(symbols))
        if not syms:
            return pd.DataFrame(columns=list(BARS_COLS))
        where.append(f"Symbol IN ({','.join('?' * len(syms))})")
        args.extend(syms)
    if start is not None:
        where.append("Date >= ?")
        args.append(pd.Timestamp(start).strftime("%Y-%m-%d"))
    if end is not None:
        where.append("Date <= ?")
        args.append(pd.Timestamp(end).strftime("%Y-%m-%d"))
    sql = f"SELECT {', '.join(BARS_COLS)} FROM {_q(BARS_DAILY)}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY Symbol, Date"
    with connect() as con:
        df = pd.read_sql_query(sql, con, params=args)
    if len(df):
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        df = pd.DataFrame(columns=list(BARS_COLS))
        df["Date"] = pd.to_datetime(df["Date"])
    return df

def append_bars(df, replace_from=None):
    """Append-only write into bars_daily. Never drops or rebuilds the table.

    Choice: INSERT OR IGNORE for settled history; INSERT OR REPLACE for dates on/after `replace_from`
    (default = the max Date in `df`, i.e. only the newest session). That lets today's possibly-partial
    bar become the final close without rewriting older days, and blocks accidental history rewrites.
    Pass replace_from=min_date_of_overlap to also refresh a short correction window after an incremental fetch.
    Returns the number of rows presented for write (not the number newly inserted)."""
    ensure_bars_daily()
    if df is None or not len(df):
        return 0
    out = df.loc[:, list(BARS_COLS)].copy()
    out["Symbol"] = out["Symbol"].astype(str)
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    for c in ("Open", "High", "Low", "Close", "Volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["Symbol", "Date"]).drop_duplicates(["Symbol", "Date"], keep="last")
    if not len(out):
        return 0
    if replace_from is None:
        replace_from = out["Date"].max()
    else:
        replace_from = pd.Timestamp(replace_from).strftime("%Y-%m-%d")
    settle = out[out["Date"] < replace_from]
    refresh = out[out["Date"] >= replace_from]
    rows_settle = [tuple(r) for r in settle.itertuples(index=False, name=None)]
    rows_refresh = [tuple(r) for r in refresh.itertuples(index=False, name=None)]
    cols_sql = ", ".join(_q(c) for c in BARS_COLS)
    ph = ",".join("?" * len(BARS_COLS))
    with connect() as con:
        con.execute("BEGIN IMMEDIATE")
        try:
            if rows_settle:
                con.executemany(
                    f"INSERT OR IGNORE INTO {_q(BARS_DAILY)} ({cols_sql}) VALUES ({ph})", rows_settle)
            if rows_refresh:
                con.executemany(
                    f"INSERT OR REPLACE INTO {_q(BARS_DAILY)} ({cols_sql}) VALUES ({ph})", rows_refresh)
            n = con.execute(f"SELECT COUNT(*) FROM {_q(BARS_DAILY)}").fetchone()[0]
            now = time.time()
            con.execute(f"INSERT OR REPLACE INTO {_q(_BARS_META)} VALUES ('updated_at', ?)", (now,))
            # discoverable in tables() without going through the CSV text path
            con.execute("INSERT OR REPLACE INTO _tables VALUES (?,?,?,?)",
                        (BARS_DAILY, json.dumps(list(BARS_COLS)), n, now))
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise
    return len(out)

def append_benchmark_prices(df, path=None):
    """Append-only write of the wide benchmark_prices table (one row per Date; columns = ETF closes).

    Unique key is Date (wide layout kept so dashboard / forward_test / the notebook keep working). New dates
    are INSERT OR IGNORE; the newest date in `df` is INSERT OR REPLACE so a same-day refresh can fill blanks.
    Never drops older history. `path` defaults to Reports/benchmark_prices.csv (the table address)."""
    path = path or os.path.join(REPORTS, "benchmark_prices.csv")
    name = table_name(path) if is_csv_path(path) else "benchmark_prices"
    if df is None or not len(df):
        return 0
    out = df.copy()
    if "Date" not in out.columns:
        raise ValueError("benchmark_prices frame needs a Date column")
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out = out.drop_duplicates("Date", keep="last").sort_values("Date")
    # first write ever: create the table from this frame
    if not has_table(name):
        save_table(name, out, index=False)
        with connect() as con:
            con.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{name.replace('/', '_')}_date "
                        f"ON {_q(name)}({_q('Date')})")
        return len(out)
    # ensure unique index (safe if it already exists)
    with connect() as con:
        con.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{name.replace('/', '_')}_date "
                    f"ON {_q(name)}({_q('Date')})")
        header, _, _ = _meta(con, name)
    # align columns to the existing header (extra cols in df get added; missing become "")
    for h in header:
        if h not in out.columns:
            out[h] = ""
    # widen table if df brings new ETF columns
    extra = [c for c in out.columns if c not in header]
    if extra:
        new_header = header + extra
        with connect() as con:
            con.execute("BEGIN IMMEDIATE")
            try:
                for c in extra:
                    con.execute(f"ALTER TABLE {_q(name)} ADD COLUMN {_q(c)} TEXT")
                n = con.execute(f"SELECT COUNT(*) FROM {_q(name)}").fetchone()[0]
                con.execute("INSERT OR REPLACE INTO _tables VALUES (?,?,?,?)",
                            (name, json.dumps(new_header), n, time.time()))
                con.execute("COMMIT")
                header = new_header
            except BaseException:
                con.execute("ROLLBACK")
                raise
    latest = out["Date"].max()
    cols_sql = ", ".join(_q(c) for c in header)
    ph = ",".join("?" * len(header))
    def row_tuple(r):
        return tuple("" if (v is None or (isinstance(v, float) and pd.isna(v))) else
                     (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in (r[c] for c in header))
    settle = [row_tuple(r) for _, r in out[out["Date"] < latest].iterrows()]
    refresh = [row_tuple(r) for _, r in out[out["Date"] >= latest].iterrows()]
    with connect() as con:
        con.execute("BEGIN IMMEDIATE")
        try:
            if settle:
                con.executemany(
                    f"INSERT OR IGNORE INTO {_q(name)} ({cols_sql}) VALUES ({ph})", settle)
            if refresh:
                con.executemany(
                    f"INSERT OR REPLACE INTO {_q(name)} ({cols_sql}) VALUES ({ph})", refresh)
            n = con.execute(f"SELECT COUNT(*) FROM {_q(name)}").fetchone()[0]
            con.execute("INSERT OR REPLACE INTO _tables VALUES (?,?,?,?)",
                        (name, json.dumps(header), n, time.time()))
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise
    return len(out)
