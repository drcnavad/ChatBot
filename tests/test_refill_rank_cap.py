"""Mon-Thu refill (sleeves.replace_blocked_new_slots): never a stock ranked worse than exit_below (20), skips
earnings-soon and unpriced candidates with a logged reason (Nov 3, 2026: LIN #21 bought while WMT #8 had no quote)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import sleeves

def main():
    d = pd.Timestamp("2026-11-03")
    syms = [f"S{i}" for i in range(1, 26)]
    scores = pd.DataFrame([[100 - i for i in range(25)]], index=[d], columns=syms)
    inp = {"scores": {"rs_short": scores}, "eligible": scores > 0, "close": scores * 0 + 10.0,
           "soon": pd.DataFrame([[s == "S3" for s in syms]], index=[d], columns=syms)}
    orig_rank, orig_cfg = sleeves.ranking, sleeves._cfg
    sleeves.ranking = lambda h, day, inp=None: scores.loc[d].sort_values(ascending=False)
    sleeves._cfg = lambda h: {"exit": (20, 10)}
    try:
        held = {f"S{i}" for i in range(11, 20)}                       # 9 held, ranks 11-19
        px = {s: 10.0 for s in syms if s not in ("S1", "S2")}         # S1/S2 unpriced
        blocked = {f"S{i}": "sold-block" for i in range(4, 11)}       # S4-S10 blocked, S3 earnings soon
        tw = sleeves.replace_blocked_new_slots("rs_short", d, [], held, blocked, px, inp, 10, 0.99)
        assert "S20" in tw and not any(s in tw for s in ("S21", "S22", "S3", "S1")), sorted(tw)
        blocked["S20"] = "sold-block"
        tw = sleeves.replace_blocked_new_slots("rs_short", d, [], held, blocked, px, inp, 10, 0.99)
        assert set(tw) == held, sorted(tw)                            # nothing past #20: the slot stays cash
        print("PASS refill capped at rank 20, earnings-soon / unpriced skipped")
    finally:
        sleeves.ranking, sleeves._cfg = orig_rank, orig_cfg

if __name__ == "__main__":
    main()
