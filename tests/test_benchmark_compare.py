"""Trading Account tab: the account vs QQQ / SPY / IWM / DIA (alpaca_paper.benchmark_table, account_twr, cash_flows):
  - each ETF gets the same money on the same days: the start balance at the start close, every later deposit (withdrawal)
    bought (sold) at the close of its date (next session's close on a weekend, the latest price when not closed yet)
  - Return % is time-weighted: a mid-period deposit changes neither an ETF's return nor the account's
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


# Oct 2 row (10:15 PM CT) holds $1,000 incl. that day's deposit; $500 arrives Oct 5 after the 3:15 PM row; Oct 6 row.
DAILY = pd.DataFrame({"Date": ["2026-10-02", "2026-10-05", "2026-10-06"], "Time_CT": ["22:15", "15:15", "16:15"],
                      "Equity": [1000.0, 1100.0, 1600.0], "Net_Deposits": [1000.0, 1000.0, 1500.0]})
FLOWS = [flow("2026-10-05", "21:15:38", 500.0), flow("2026-10-02", "21:15:23", 300.0), flow("2026-09-30", "15:00:00", 700.0)]
CLOSES = pd.DataFrame({"QQQ": [99.0, 100.0, 110.0, 105.0], "SPY": [49.0, 50.0, 50.0, 55.0], "DIA": [1.0, 2.0, 2.0, 2.0]},
                      index=pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06"]))
NOW = {"QQQ": 120.0, "SPY": 60.0, "IWM": 30.0, "DIA": 2.5}
EQ_NOW = 1650.0

check("four index ETFs: QQQ, SPY, IWM, DIA", list(ap.BENCHMARK_ETFS) == ["QQQ", "SPY", "IWM", "DIA"])
t, start, put_in = ap.benchmark_table(DAILY, EQ_NOW, FLOWS, CLOSES, NOW)
t = t.set_index("Compared with")
check("start = first daily row; money put in = its balance + later deposits (earlier ones are in that balance)",
      str(start.date()) == "2026-10-02" and math.isclose(put_in, 1500), (start, put_in))
check("rows: the account, then each ETF with closes (IWM has none: left out)",
      list(t.index) == ["Your account", "QQQ (Nasdaq 100)", "SPY (S&P 500)", "DIA (Dow Jones)"], list(t.index))
q = t.loc["QQQ (Nasdaq 100)"]
units = 1000 / 100 + 500 / 110                         # start close Oct 2, the deposit at the Oct 5 close
check("QQQ: start balance at the Oct 2 close, the Oct 5 deposit at the Oct 5 close, valued at the latest price",
      math.isclose(q["Value now"], units * 120) and math.isclose(q["Gain $"], units * 120 - 1500), q.to_dict())
check("QQQ return = its price change since the start close (time-weighted)", math.isclose(q["Return %"], 20), q["Return %"])
acct = 1100 / 1000 * 1600 / (1100 + 500) * EQ_NOW / 1600 - 1
a = t.loc["Your account"]
check("account return: each period's growth on its starting money incl. that period's deposits",
      math.isclose(a["Return %"], acct * 100) and math.isclose(a["Gain $"], EQ_NOW - 1500) and a["Value now"] == EQ_NOW, a.to_dict())
check("account ahead by = account return - ETF return (pts)", math.isclose(q["Account ahead by (pts)"], acct * 100 - 20)
      and math.isclose(t.loc["SPY (S&P 500)", "Account ahead by (pts)"], acct * 100 - 20) and pd.isna(a["Account ahead by (pts)"]))

# A mid-period deposit changes no return: same prices, one big extra deposit that only adds cash.
big = [flow("2026-10-05", "21:15:38", 500.0 + 10000.0)] + FLOWS[1:]
d2 = DAILY.assign(Equity=[1000.0, 1100.0, 11600.0], Net_Deposits=[1000.0, 1000.0, 11500.0])
t2 = ap.benchmark_table(d2, EQ_NOW + 10000, big, CLOSES, NOW)[0].set_index("Compared with")
check("mid-period deposit: every ETF's return unchanged", (t2["Return %"].drop("Your account") == t["Return %"].drop("Your account")).all())
check("mid-period deposit: its value adds exactly deposit x the ETF's growth since that day's close",
      math.isclose(t2.loc["QQQ (Nasdaq 100)", "Value now"] - q["Value now"], 10000 * 120 / 110))
flat = DAILY.assign(Equity=[1000.0, 1000.0, 1500.0])                  # $500 deposited, nothing earned
check("account: a deposit alone is no return (flat account = 0%)",
      math.isclose(ap.account_twr(flat, 1500.0, 1500.0), 0, abs_tol=1e-12))
check("account TWR index: 1.0 at the start, deposit day flat", list(ap.account_twr_index(flat)) == [1.0, 1.0, 1.0])

# Withdrawal today (no close yet: the latest price) and a weekend deposit (next session's close).
more = FLOWS + [flow("2026-10-07", "15:00:00", -200.0), flow("2026-10-03", "15:00:00", 100.0)]
t3, _, put3 = ap.benchmark_table(DAILY, EQ_NOW, more, CLOSES, NOW)
q3 = t3.set_index("Compared with").loc["QQQ (Nasdaq 100)"]
check("withdrawal sells at the latest price when its day has no close; Saturday deposit buys at Monday's close",
      math.isclose(q3["Value now"], (units - 200 / 120 + 100 / 110) * 120) and math.isclose(put3, 1400), (q3.to_dict(), put3))
check("live step uses deposits since the last row (lifetime net deposits now)",
      math.isclose(t3.iloc[0]["Return %"], (1100 / 1000 * 1600 / 1600 * EQ_NOW / (1600 - 100) - 1) * 100))
check("a flow without a time counts at the end of its date",
      ap._flow_time({"date": "2026-10-02", "time": "", "amount": 1}) == pd.Timestamp("2026-10-02 23:59", tz=ap.CT))


class FakeAccount(ap.PaperAccount):
    """No network: 150 deposit / withdrawal activities over two pages (one canceled)."""
    def __init__(self):
        acts = [{"id": f"a{i}", "activity_type": "CSD", "date": "2026-10-05", "created_at": "2026-10-05T21:15:38Z",
                 "net_amount": "10", "status": "executed"} for i in range(148)]
        acts += [{"id": "w", "activity_type": "CSW", "date": "2026-10-06", "net_amount": "-30", "status": "executed"},
                 {"id": "c", "activity_type": "CSD", "date": "2026-10-06", "net_amount": "999", "status": "canceled"}]
        self.acts, self.calls = acts, []

    def _get(self, path, params=None):
        assert path in ap.ALLOWED_PATHS and params["activity_types"] == "CSD,CSW,JNLC", (path, params)
        self.calls.append(dict(params))
        i = 0 if "page_token" not in params else [a["id"] for a in self.acts].index(params["page_token"]) + 1
        return self.acts[i:i + params["page_size"]]


fa = FakeAccount()
cf = fa.cash_flows()
check("cash_flows: every page, canceled left out, withdrawals negative, date / time / amount",
      len(cf) == 149 and len(fa.calls) == 2 and cf[-1] == {"date": "2026-10-06", "time": "", "amount": -30.0}
      and cf[0] == {"date": "2026-10-05", "time": "2026-10-05T21:15:38Z", "amount": 10.0}, (len(cf), fa.calls[-1:], cf[-1]))
check("net_deposits = the sum of cash_flows", math.isclose(fa.net_deposits(), 148 * 10 - 30))
print(f"\n{len(FAIL)} failed" if FAIL else "\nBENCHMARK COMPARE OK")
sys.exit(1 if FAIL else 0)
