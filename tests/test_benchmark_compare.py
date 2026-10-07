"""Trading Account tab: the account vs QQQ / SPY / IWM / DIA (alpaca_paper.account_vs_etfs, snapshot, daily_history):
  - one source: Alpaca's daily account history + one live read (balance first, then only the deposits already in it)
  - each ETF gets the same money on the same days: the start balance at the start close, every later deposit (withdrawal)
    bought (sold) at the close of its date (next session's close on a weekend, the latest price when not closed yet)
  - Return % is time-weighted: deposits book after the close (about 4:15 PM CT), so a day's growth leaves out that day's
    deposit, and a deposit never changes an ETF's or the account's return
  - deposits read from Alpaca's CSD / CSW / JNLC activities (paged, canceled left out)
Pure functions and a fake client: no network. Run: python tests/run_tests.py  (or python tests/test_benchmark_compare.py)"""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pandas as pd

import alpaca_paper as ap

FAIL = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info and not ok else ""))
    if not ok:
        FAIL.append(name)


def flow(day, utc_time, amount):
    return {"date": day, "time": f"{day}T{utc_time}Z", "amount": amount}


def snap(equity, flows, at="2026-10-07 16:30"):
    return {"equity": equity, "flows": flows, "at": pd.Timestamp(at, tz=ap.CT)}


# Alpaca's daily bars (end of day, that day's deposit included): Oct 1 (before the start), Oct 2 start $1,000 incl. its
# $300 deposit, Oct 5 $1,600 incl. a $500 deposit (so $1,100 before it), Oct 6 $1,650.
HIST = pd.DataFrame({"Date": pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06"]),
                     "Equity": [690.0, 1000.0, 1600.0, 1650.0], "Cashflow": [0.0, 300.0, 500.0, 0.0]})
FLOWS = [flow("2026-10-05", "21:15:38", 500.0), flow("2026-10-02", "21:15:23", 300.0), flow("2026-09-30", "15:00:00", 700.0)]
CLOSES = pd.DataFrame({"QQQ": [99.0, 100.0, 110.0, 105.0], "SPY": [49.0, 50.0, 50.0, 55.0], "DIA": [1.0, 2.0, 2.0, 2.0]},
                      index=pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06"]))
NOW = {"QQQ": 120.0, "SPY": 60.0, "IWM": 30.0, "DIA": 2.5}
EQ_NOW = 1700.0
START = "2026-10-02"
ACCT = 1100 / 1000 * 1650 / 1600 * EQ_NOW / 1650 - 1        # Oct 5 growth before its deposit, Oct 6, today (live)

check("four index ETFs: QQQ, SPY, IWM, DIA", list(ap.BENCHMARK_ETFS) == ["QQQ", "SPY", "IWM", "DIA"])
t, start, put_in = ap.account_vs_etfs(HIST, snap(EQ_NOW, FLOWS), CLOSES, NOW, START)
t = t.set_index("Compared with")
check("start = the start day's end balance (its deposit in); money put in = that + later deposits",
      str(start.date()) == START and math.isclose(put_in, 1500), (start, put_in))
check("rows: the account, then each ETF with closes (IWM has none: left out)",
      list(t.index) == ["Your account", "QQQ (Nasdaq 100)", "SPY (S&P 500)", "DIA (Dow Jones)"], list(t.index))
q = t.loc["QQQ (Nasdaq 100)"]
units = 1000 / 100 + 500 / 110                         # start close Oct 2, the deposit at the Oct 5 close
check("QQQ: start balance at the Oct 2 close, the Oct 5 deposit at the Oct 5 close, valued at the latest price",
      math.isclose(q["Value now"], units * 120) and math.isclose(q["Gain $"], units * 120 - 1500), q.to_dict())
check("QQQ return = its price change since the start close (time-weighted)", math.isclose(q["Return %"], 20), q["Return %"])
a = t.loc["Your account"]
check("account return: each day's growth leaves out that day's after-close deposit; live step from the last bar",
      math.isclose(a["Return %"], ACCT * 100) and math.isclose(a["Gain $"], EQ_NOW - 1500) and a["Value now"] == EQ_NOW, a.to_dict())
check("account ahead by = account return - ETF return (pts)", math.isclose(q["Account ahead by (pts)"], ACCT * 100 - 20)
      and math.isclose(t.loc["SPY (S&P 500)", "Account ahead by (pts)"], ACCT * 100 - 20) and pd.isna(a["Account ahead by (pts)"]))

# A deposit changes no return: one big extra deposit on Oct 5 that only adds cash.
h2 = HIST.assign(Equity=[690.0, 1000.0, 11600.0, 11650.0], Cashflow=[0.0, 300.0, 10500.0, 0.0])
big = [flow("2026-10-05", "21:15:38", 10500.0)] + FLOWS[1:]
t2 = ap.account_vs_etfs(h2, snap(11700.0, big), CLOSES, NOW, START)[0].set_index("Compared with")
check("deposit: every ETF's return unchanged", (t2["Return %"].drop("Your account") == t["Return %"].drop("Your account")).all())
check("deposit: its value adds exactly deposit x the ETF's growth since that day's close",
      math.isclose(t2.loc["QQQ (Nasdaq 100)", "Value now"] - q["Value now"], 10000 * 120 / 110))
flat = HIST.assign(Equity=[690.0, 1000.0, 1500.0, 1500.0])                  # $500 deposited, nothing earned
check("account: a deposit alone is no return (flat account = 0%)",
      math.isclose(ap.account_vs_etfs(flat, snap(1500.0, FLOWS), CLOSES, NOW, START)[0]["Return %"].iloc[0], 0, abs_tol=1e-12))

# Today's deposit (Oct 7, 4:15 PM CT): counted only once the balance has it (snapshot), and then it is no return.
today = flow("2026-10-07", "21:15:13", 3000.0)
before = ap.account_vs_etfs(HIST, snap(EQ_NOW, FLOWS, "2026-10-07 16:10"), CLOSES, NOW, START)
after = ap.account_vs_etfs(HIST, snap(EQ_NOW + 3000.0, [today] + FLOWS, "2026-10-07 16:20"), CLOSES, NOW, START)
check("today's after-close deposit: same account return before and after it books, put-in grows by it",
      math.isclose(before[0]["Return %"].iloc[0], after[0]["Return %"].iloc[0]) and math.isclose(after[2] - before[2], 3000),
      (before[0]["Return %"].iloc[0], after[0]["Return %"].iloc[0]))
check("today's deposit buys each ETF at the latest price (no close yet): its value adds the deposit exactly",
      math.isclose(after[0].set_index("Compared with").loc["QQQ (Nasdaq 100)", "Value now"] - q["Value now"], 3000))
# A bar for today (partial / before the deposit booked) is left out: the live read replaces it.
h3 = pd.concat([HIST, pd.DataFrame({"Date": [pd.Timestamp("2026-10-07")], "Equity": [1690.0], "Cashflow": [0.0]})])
check("a daily bar for today is ignored (the live read is today's step)",
      math.isclose(ap.account_vs_etfs(h3, snap(EQ_NOW, FLOWS), CLOSES, NOW, START)[0]["Return %"].iloc[0], ACCT * 100))
# History lag: yesterday's bar not out yet -> yesterday's deposit was in today's session (start-of-step money).
lag = HIST.iloc[:3]
r_lag = ap.account_vs_etfs(lag, snap(EQ_NOW, FLOWS + [flow("2026-10-06", "21:15:00", 50.0)]), CLOSES, NOW, START)
check("missing bar: a deposit dated after the last bar but before today is start money for the live step",
      math.isclose(r_lag[0]["Return %"].iloc[0], (1100 / 1000 * EQ_NOW / (1600 + 50) - 1) * 100) and math.isclose(r_lag[2], 1550))
check("history not reaching the start -> None (the table is left out)",
      ap.account_vs_etfs(HIST.iloc[2:], snap(EQ_NOW, FLOWS), CLOSES, NOW, START) is None)
check("a flow without a time counts at the end of its date",
      ap._flow_time({"date": "2026-10-02", "time": "", "amount": 1}) == pd.Timestamp("2026-10-02 23:59", tz=ap.CT))
flat_rows = pd.DataFrame({"Date": ["2026-10-02", "2026-10-05", "2026-10-06"], "Equity": [1000.0, 1000.0, 1500.0],
                          "Net_Deposits": [1000.0, 1000.0, 1500.0]})
check("account TWR index (forward-test rows): 1.0 at the start, a deposit day is flat",
      list(ap.account_twr_index(flat_rows)) == [1.0, 1.0, 1.0])


class FakeAccount(ap.PaperAccount):
    """No network: 150 deposit / withdrawal activities over two pages (one canceled)."""
    def __init__(self):
        acts = [{"id": f"a{i}", "activity_type": "CSD", "date": "2026-10-05", "created_at": "2026-10-05T21:15:38Z",
                 "net_amount": "10", "status": "executed"} for i in range(148)]
        acts += [{"id": "w", "activity_type": "CSW", "date": "2026-10-06", "net_amount": "-30", "status": "executed"},
                 {"id": "c", "activity_type": "CSD", "date": "2026-10-06", "net_amount": "999", "status": "canceled"}]
        self.acts, self.calls = acts, []

    def _get(self, path, params=None):
        assert path in ap.ALLOWED_PATHS, path
        if path == "/account":
            return {"equity": "1480"}
        if path == "/account/portfolio/history":
            self.calls.append(dict(params))
            return {"timestamp": [1790985600, 1791244800], "equity": [1000.0, None], "cashflow": {"CSD": [300.0, 0.0], "JNLC": [-5.0]}}
        assert params["activity_types"] == "CSD,CSW,JNLC", params
        self.calls.append(dict(params))
        i = 0 if "page_token" not in params else [a["id"] for a in self.acts].index(params["page_token"]) + 1
        return self.acts[i:i + params["page_size"]]


fa = FakeAccount()
cf = fa.cash_flows()
check("cash_flows: every page, canceled left out, withdrawals negative, date / time / amount",
      len(cf) == 149 and len(fa.calls) == 2 and cf[-1] == {"date": "2026-10-06", "time": "", "amount": -30.0}
      and cf[0] == {"date": "2026-10-05", "time": "2026-10-05T21:15:38Z", "amount": 10.0}, (len(cf), fa.calls[-1:], cf[-1]))
check("net_deposits = the sum of cash_flows", math.isclose(fa.net_deposits(), 148 * 10 - 30))
fa.acts.append({"id": "late", "activity_type": "CSD", "date": "2099-01-02", "created_at": "2099-01-02T21:15:00Z",
                "net_amount": "5000", "status": "executed"})
sn = fa.snapshot()
check("snapshot: balance read first; a deposit booked after the read is left out until the next read",
      sn["equity"] == 1480.0 and len(sn["flows"]) == 149 and sum(f["amount"] for f in sn["flows"]) == 148 * 10 - 30, sn["flows"][:1])
fa.calls = []
dh = fa.daily_history("2026-10-02")
check("daily_history: one GET of the 1D history from the start with the deposit types; ET trading days; cash flows summed; "
      "a day without equity dropped",
      fa.calls == [{"timeframe": "1D", "start": "2026-10-02", "pnl_reset": "no_reset", "cashflow_types": "CSD,CSW,JNLC"}]
      and list(dh["Date"]) == [pd.Timestamp("2026-10-02")] and dh["Equity"].tolist() == [1000.0] and dh["Cashflow"].tolist() == [295.0],
      (fa.calls, dh))
print(f"\n{len(FAIL)} failed" if FAIL else "\nBENCHMARK COMPARE OK")
sys.exit(1 if FAIL else 0)
