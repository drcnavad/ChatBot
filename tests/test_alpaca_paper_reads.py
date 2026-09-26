"""Mocked tests for the new alpaca_paper.py read-only views (open_orders, market_clock,
portfolio_history): parsing is correct and the module stays read-only (no order placement,
paper endpoint allowlist intact). Zero network calls.
Run from the project root:  python tests/test_alpaca_paper_reads.py
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import alpaca_paper as ap


class FakeAccount(ap.PaperAccount):
    """Stub _get so no network is touched; asserts the path is allowlisted."""

    def __init__(self, payloads):
        self._payloads = payloads
        self.seen = []

    def _get(self, path, params=None):
        assert path in ap.ALLOWED_PATHS, f"path not allowlisted: {path}"
        self.seen.append((path, params))
        return self._payloads[path]


ORDER = {"submitted_at": "2026-09-26T20:30:00Z", "symbol": "NVDA", "side": "buy", "qty": "10",
         "filled_qty": "0", "type": "limit", "status": "new", "filled_avg_price": None}


def make_fake():
    return FakeAccount({
        "/orders": [dict(ORDER), dict(ORDER, symbol="AAPL", status="held", side="sell")],
        "/clock": {"is_open": False, "next_open": "2026-09-28T13:30:00Z",
                   "next_close": "2026-09-28T20:00:00Z", "timestamp": "2026-09-26T16:00:00Z"},
        "/account/portfolio/history": {"timestamp": [1758844800, 1758931200],
                                       "equity": [100000.0, 100500.25],
                                       "profit_loss": [0.0, 500.25],
                                       "profit_loss_pct": [0.0, 0.005]},
    })


class TestNewReads(unittest.TestCase):
    def test_open_orders(self):
        oo = make_fake().open_orders()
        self.assertEqual(len(oo), 2)
        self.assertEqual(oo["Symbol"].tolist(), ["NVDA", "AAPL"])
        self.assertEqual(oo["Status"].tolist(), ["new", "held"])
        # asked for open orders only
        f = make_fake()
        f.open_orders()
        self.assertEqual(f.seen[0][0], "/orders")
        self.assertEqual(f.seen[0][1]["status"], "open")

    def test_market_clock(self):
        mc = make_fake().market_clock()
        self.assertFalse(mc["Is open"])
        self.assertIn("2026-09-28", mc["Next open (CT)"])
        self.assertIn("2026-09-28", mc["Next close (CT)"])

    def test_portfolio_history(self):
        ph = make_fake().portfolio_history()
        self.assertEqual(list(ph.columns), ["As_Of (CT)", "Equity", "P/L $", "P/L %"])
        self.assertEqual(len(ph), 2)
        self.assertAlmostEqual(ph["Equity"].iloc[1], 100500.25)
        self.assertAlmostEqual(ph["P/L %"].iloc[1], 0.5)

    def test_portfolio_history_ragged(self):
        f = FakeAccount({"/account/portfolio/history": {"timestamp": [1], "equity": [5.0]}})
        ph = f.portfolio_history()
        self.assertEqual(len(ph), 1)
        self.assertEqual(ph["P/L $"].iloc[0], 0.0)      # missing series degrade gracefully

    def test_recent_orders_unchanged(self):
        ro = make_fake().recent_orders()
        self.assertEqual(len(ro), 2)
        f = make_fake()
        f.recent_orders()
        self.assertEqual(f.seen[0][1]["status"], "all")


class TestReadOnlyGuardrails(unittest.TestCase):
    def test_new_paths_allowlisted(self):
        self.assertIn("/clock", ap.ALLOWED_PATHS)
        self.assertIn("/account/portfolio/history", ap.ALLOWED_PATHS)

    def test_dangerous_paths_refused_without_network(self):
        acct = ap.PaperAccount.__new__(ap.PaperAccount)
        acct.base_url = ap.PAPER_BASE_URL
        for bad in ("/orders/123/cancel", "/orders", "/positions/XYZ"):
            if bad in ap.ALLOWED_PATHS:
                continue
            with self.assertRaises(ap.PaperAccountError, msg=bad):
                ap.PaperAccount._get(acct, bad)

    def test_no_order_placing_code(self):
        src = open(ap.__file__).read().lower()
        for needle in ("submit_order", "cancel_order", "replace_order", "close_position",
                       "method=\"post\"", "method='post'", '"post"', "'post'"):
            self.assertNotIn(needle, src, needle)


if __name__ == "__main__":
    unittest.main(verbosity=1)
