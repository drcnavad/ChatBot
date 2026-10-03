"""
company_report_autofetch.py
---------------------------
Fetches quarterly income statement + balance sheet data from Alpha Vantage and saves
the merged rows (amounts in millions) to Reports/balance_sheet.csv, the input for
company_report_processing.ipynb.

Daily rotation (default):
  - Up to MAX_STOCKS symbols whose last successful fetch is older than STALE_DAYS
  - Symbols attempted within the cooldown are skipped so the whole list rotates first
  - Empty API responses are logged + alerted; existing rows are never wiped

Manual: set MANUAL_SYMBOL = "TICKER" to fetch one stock right now (skips rotation).

Limits: Alpha Vantage free tier = 25 calls/day, 2 calls per stock, 13s between calls.

Usage:
  python company_report_autofetch.py
Schedule with cron (daily at 7am):
  0 7 * * * cd "/path/to/Stock Analysis" && python company_report_autofetch.py
"""

import os
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

import sector_mapping

# --- CONFIG ------------------------------------------------------------------

MANUAL_SYMBOL = ""  # e.g. "UPST" -> fetch only this stock now; "" -> daily rotation

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
ALPHA_VANTAGE_API = os.getenv("ALPHAVANTAGE_API_KEY")
if not ALPHA_VANTAGE_API:
    raise EnvironmentError("Missing ALPHAVANTAGE_API_KEY in .env - add it to run the fundamentals step")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "Reports")
BALANCE_CSV = os.path.join(REPORTS_DIR, "balance_sheet.csv")
RUN_LOG_CSV = os.path.join(REPORTS_DIR, "fetch_run_log.csv")  # Symbol | RunDate | IncomeStatus | BalanceStatus

STALE_DAYS = 90      # re-fetch when the last successful fetch is older than this
MAX_STOCKS = 12      # 25 API calls/day / 2 calls per stock
CALL_DELAY_SEC = 13  # free tier = 5 calls/min
YEARS_TO_KEEP = 5
COOLDOWN_DAYS = None  # None = ceil(len(ALL_SYMBOLS) / MAX_STOCKS), one full pass before repeats

# Fundamentals universe: from sector_mapping.py only (every tradable stock + its fundamentals_extra watchlist), so a stock
# added to / removed from sector_mapping is fetched / dropped automatically. No list of its own here.
ALL_SYMBOLS = list(sector_mapping.fundamentals_symbols)

# --- COLUMNS -----------------------------------------------------------------

INCOME_FIELDS = {
    "TotalRevenue": "totalRevenue",
    "GrossProfit": "grossProfit",
    "NetIncome": "netIncome",
    "OperatingIncome": "operatingIncome",
}
BALANCE_FIELDS = {
    "TotalAssets": "totalAssets",
    "TotalLiabilities": "totalLiabilities",
    "CommonStockSharesOutstanding": "commonStockSharesOutstanding",
    "TotalShareholderEquity": "totalShareholderEquity",
    "TotalDebt": "shortLongTermDebtTotal",
    "CashAndEquivalents": "cashAndCashEquivalentsAtCarryingValue",
}
MILLIONS_COLS = [*INCOME_FIELDS, *BALANCE_FIELDS]
OUTPUT_COLUMNS = [
    "Symbol", "FiscalDateEnding", *INCOME_FIELDS, "OperatingMargin",
    *BALANCE_FIELDS, "BVPS", "Debt_to_Equity", "Date", "DateAdded",
]

# --- FETCH -------------------------------------------------------------------


def fetch_statement(function, symbol, fields):
    """Quarterly reports as a DataFrame; None on rate limit, empty on no data/error."""
    try:
        data = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": function, "symbol": symbol, "apikey": ALPHA_VANTAGE_API},
            timeout=30,
        ).json()
    except Exception as e:
        print(f"  [!] {function} request failed for {symbol}: {e}")
        return pd.DataFrame()
    if "Note" in data or "Information" in data:
        return None
    return pd.DataFrame([
        {"Symbol": symbol, "FiscalDateEnding": r.get("fiscalDateEnding"),
         **{col: r.get(key) for col, key in fields.items()}}
        for r in data.get("quarterlyReports", [])
    ])


# --- TRANSFORM + SAVE --------------------------------------------------------


def build_fundamentals(income, balance):
    """Merge raw income + balance rows into the balance_sheet.csv format."""
    income, balance = (
        d.assign(FiscalDateEnding=pd.to_datetime(d["FiscalDateEnding"], errors="coerce"))
        for d in (income, balance)
    )
    df = income.merge(balance, on=["Symbol", "FiscalDateEnding"], how="outer")
    df = df[df["FiscalDateEnding"] >= pd.Timestamp.today() - pd.DateOffset(years=YEARS_TO_KEEP)].copy()
    df[MILLIONS_COLS] = df[MILLIONS_COLS].apply(pd.to_numeric, errors="coerce")
    df["OperatingMargin"] = (df["OperatingIncome"] / df["TotalRevenue"].replace(0, np.nan) * 100).round(2)
    df["BVPS"] = (df["TotalShareholderEquity"] / df["CommonStockSharesOutstanding"].replace(0, np.nan)).round(2)
    df["Debt_to_Equity"] = (df["TotalLiabilities"] / df["TotalShareholderEquity"]).round(2)
    df[MILLIONS_COLS] = df[MILLIONS_COLS] / 1_000_000
    df = df.replace([np.inf, -np.inf], np.nan)
    df["Date"] = df["DateAdded"] = pd.Timestamp.today().normalize()
    # No per-company outlier clipping: clamping a company's own quarters to Q1/Q3 +- 1.5 IQR flattened the newest
    # quarters of every fast grower (e.g. NVDA's last two quarters showed the same capped net income and assets).
    return df.sort_values(["Symbol", "FiscalDateEnding"])[OUTPUT_COLUMNS]


def save_fundamentals(new_df):
    """Replace these symbols' rows in balance_sheet.csv; other symbols stay untouched."""
    existing = pd.read_csv(BALANCE_CSV, parse_dates=["FiscalDateEnding"])
    result = pd.concat([existing[~existing["Symbol"].isin(new_df["Symbol"])], new_df], ignore_index=True)
    result = result.sort_values(["Symbol", "FiscalDateEnding"], ascending=[True, False])
    result[OUTPUT_COLUMNS].to_csv(BALANCE_CSV, index=False)


# --- ROTATION ----------------------------------------------------------------


def load_run_log():
    if not os.path.exists(RUN_LOG_CSV):
        return pd.DataFrame(columns=["Symbol", "RunDate", "IncomeStatus", "BalanceStatus"])
    log = pd.read_csv(RUN_LOG_CSV)
    log["RunDate"] = pd.to_datetime(log["RunDate"], errors="coerce").dt.date
    return log.dropna(subset=["RunDate"])


def log_run(symbol, income_status, balance_status):
    pd.DataFrame([[symbol, datetime.now().date(), income_status, balance_status]]).to_csv(
        RUN_LOG_CSV, mode="a", header=not os.path.exists(RUN_LOG_CSV), index=False
    )


def get_stale_symbols():
    """Symbols needing a fetch and outside cooldown: never-attempted first, then oldest attempt."""
    today = datetime.now().date()
    cool_for = COOLDOWN_DAYS or -(-len(ALL_SYMBOLS) // MAX_STOCKS)
    log = load_run_log()
    attempted = log[(log["IncomeStatus"] != "rate_limit") & (log["BalanceStatus"] != "rate_limit")]
    last_any = attempted.groupby("Symbol")["RunDate"].max()
    ok = log[(log["IncomeStatus"] == "ok") & (log["BalanceStatus"] == "ok")]
    last_ok = ok.groupby("Symbol")["RunDate"].max()

    stale_cut = today - timedelta(days=STALE_DAYS)
    cool_cut = today - timedelta(days=cool_for)
    in_cooldown = sorted(last_any[last_any >= cool_cut].index)
    stale = [
        s for s in ALL_SYMBOLS
        if last_ok.get(s, date.min) < stale_cut and s not in in_cooldown
    ]
    stale.sort(key=lambda s: (s in last_any, last_any.get(s, date.min)))
    return stale, cool_for, in_cooldown


# --- MAIN --------------------------------------------------------------------


def fetch_symbol(symbol):
    """Fetch + save one symbol. Returns False when the API rate limit is hit."""
    income = fetch_statement("INCOME_STATEMENT", symbol, INCOME_FIELDS)
    if income is None:
        log_run(symbol, "rate_limit", "skipped")
        return False
    time.sleep(CALL_DELAY_SEC)
    balance = fetch_statement("BALANCE_SHEET", symbol, BALANCE_FIELDS)
    if balance is None:
        log_run(symbol, "ok" if len(income) else "empty", "rate_limit")
        return False

    income_status = "ok" if len(income) else "empty"
    balance_status = "ok" if len(balance) else "empty"
    if income.empty or balance.empty:
        print(f"  [ALERT] income {income_status}, balance {balance_status} -> balance_sheet.csv unchanged")
    else:
        new_df = build_fundamentals(income, balance)
        save_fundamentals(new_df)
        print(f"  [OK] {len(new_df)} quarters saved -> balance_sheet.csv")
    log_run(symbol, income_status, balance_status)
    return True


def main():
    print("=" * 60)
    print(f"  Auto Fetch Fundamentals  |  {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 60)

    manual = MANUAL_SYMBOL.strip().upper()
    if manual:
        to_process = [manual]
        print(f"\n>> Manual run: {manual}")
        if manual not in ALL_SYMBOLS:
            print("   Not in the rotation - add it to sector_mapping.stock_symbols to keep it refreshed.")
    else:
        stale, cool_for, in_cooldown = get_stale_symbols()
        print(f"\n>> Cooldown {cool_for} days, skipped ({len(in_cooldown)}): {', '.join(in_cooldown) or 'none'}")
        print(f">> {len(stale)} eligible (no ok fetch in {STALE_DAYS}d): {', '.join(stale) or 'none'}")
        to_process = stale[:MAX_STOCKS]
        if len(stale) > MAX_STOCKS:
            print(f">> Deferring {len(stale) - MAX_STOCKS} symbols to future runs")

    for i, symbol in enumerate(to_process, 1):
        print(f"\n[{i}/{len(to_process)}] {symbol} " + "-" * 38)
        if not fetch_symbol(symbol):
            print("  RATE LIMIT reached - stopping run early.")
            break
        if i < len(to_process):
            time.sleep(CALL_DELAY_SEC)

    log = load_run_log()
    today_log = log[log["RunDate"] == datetime.now().date()]
    print(f"\n  fetch_run_log.csv - today's entries ({len(today_log)}):")
    print(today_log.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
