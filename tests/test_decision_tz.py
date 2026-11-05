"""Regression (Oct 21, 2026 launchd_support.err: 'Cannot compare tz-naive and tz-aware timestamps' in the old
is_sector_day, from run_all.finished_text -> decision_kind -> next_decision): the decision calendar must accept naive,
America/Chicago and UTC timestamps and agree on the day kind."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import pandas as pd
import backtest_engine as b

for day in ["2026-10-19", "2026-10-21", "2026-10-23", "2026-11-05", "2026-11-06", "2026-11-30"]:
    kinds = set()
    for d in (pd.Timestamp(day), pd.Timestamp(day + " 14:30", tz="America/Chicago"), pd.Timestamp(day, tz="UTC")):
        kinds.add(b.decision_kind(d))
        nd = b.next_decision(d)
        assert pd.Timestamp(nd[0]).normalize().tz_localize(None) > pd.Timestamp(day), (day, nd)
    assert len(kinds) == 1, (day, kinds)
print("PASS decision_kind / next_decision: naive, Chicago and UTC timestamps agree, no tz TypeError")
