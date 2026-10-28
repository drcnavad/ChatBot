"""Test fixture: pins backtest_engine.WINNER to the live rules of Oct 15, 2026 (weekly C6 rebalance, relative strength,
defensive sleeve, Mon-Thu checks, E5, 15% cap) for the tests of that machinery, which stays in the code (forward-test
strategies, a switch back). The live rules from Oct 27, 2026 (RS short-term) are checked in test_live_rs_short.py.
Import it before paper_trade / run_all / dashboard."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest_engine as be  # noqa: E402

be.WINNER.clear()
be.WINNER.update({k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
                  for k, v in be.C6_RULES_OCT15.items()})
be.WINNER["tag"], be.WINNER["name"] = be.winner_label(be.scored_stock_count())
import paper_trade  # noqa: E402
paper_trade.MAX_ORDER_PCT = 0.15   # the Oct 15 order backstop
