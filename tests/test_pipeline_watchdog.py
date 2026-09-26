"""Mocked tests for pipeline_watchdog.py: classification, resume logic, retry policy, diagnose-only contract.

Zero subprocess runs of the real pipeline, zero network calls, zero LLM calls (all stubbed).
Run from the project root:  python tests/test_pipeline_watchdog.py
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import run_all
import pipeline_watchdog as wd


class TestClassify(unittest.TestCase):
    def test_transient_samples(self):
        for sample in [
            "ERROR: 429 Too Many Requests",
            "NewsAPI rate limit exceeded, retry later",
            "requests.exceptions.ConnectionError: connection reset by peer",
            "urllib.error.URLError: timed out",
            "HTTP 503 Service Unavailable from finnhub",
            "alpaca_trade_api.rest.APIError: 500 Internal Server Error",
            "SSLError: EOF occurred in violation of protocol",
        ]:
            kind, detail = wd.classify_failure(sample)
            self.assertEqual((kind, detail), ("transient", None), sample)

    def test_unknown_samples(self):
        for sample in [
            "Cell In[12], line 5: KeyError: 'Strategy_Weight'",
            "NameError: name 'foo' is not defined",
            "ValueError: cannot reindex on an axis with duplicate labels",
            "Traceback (most recent call last):\n  File \"x.py\", line 1\nTypeError: bad",
        ]:
            self.assertEqual(wd.classify_failure(sample), ("unknown", None), sample)

    def test_missing_upstream_file(self):
        kind, step = wd.classify_failure(
            "FileNotFoundError: [Errno 2] No such file or directory: 'Reports/weighted_sentiment.csv'")
        self.assertEqual((kind, step), ("missing_upstream", "sentiment"))
        kind, step = wd.classify_failure("OSError: [Errno 2] No such file: 'Reports/balance_sheet_weights.csv'")
        self.assertEqual((kind, step), ("missing_upstream", "scoring"))
        # a missing file with no known producer is not auto-resumed
        self.assertEqual(wd.classify_failure("FileNotFoundError: 'Reports/mystery.csv'"), ("unknown", None))

    def test_empty_is_unknown(self):
        self.assertEqual(wd.classify_failure(""), ("unknown", None))
        self.assertEqual(wd.classify_failure(None), ("unknown", None))


class TestResume(unittest.TestCase):
    def test_failed_step_wins(self):
        cp = {"phase": "steps", "failed_step": "sentiment", "steps_done": ["fundamentals"]}
        self.assertEqual(wd.resume_step(cp), "sentiment")

    def test_after_last_done(self):
        cp = {"phase": "steps", "failed_step": None, "steps_done": ["fundamentals", "processing"]}
        self.assertEqual(wd.resume_step(cp), "scoring")

    def test_no_checkpoint(self):
        self.assertIsNone(wd.resume_step({}))
        self.assertIsNone(wd.resume_step(None))

    def test_build_resume_argv(self):
        self.assertEqual(wd.build_resume_argv(["--trade"], "main"), ["--trade", "--from", "main"])
        self.assertEqual(wd.build_resume_argv(["--trade", "--from", "main"], "scoring"),
                         ["--trade", "--from", "main"])          # already scoped: untouched
        self.assertEqual(wd.build_resume_argv(["--only", "main"], "scoring"), ["--only", "main"])
        self.assertEqual(wd.build_resume_argv(["--trade"], None), ["--trade"])


class TestCheckpointRoundtrip(unittest.TestCase):
    def test_write_read_clear(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "cp.json")
            run_all.write_checkpoint("r1", ["--trade"], "full", ["fundamentals"], failed_step="sentiment",
                                     phase="steps", path=p)
            cp = run_all.read_checkpoint(path=p)
            self.assertEqual(cp["failed_step"], "sentiment")
            self.assertEqual(cp["argv"], ["--trade"])
            self.assertEqual(cp["steps_done"], ["fundamentals"])
            run_all.clear_checkpoint(path=p)
            self.assertEqual(run_all.read_checkpoint(path=p), {})

    def test_read_corrupt_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "cp.json")
            with open(p, "w") as f:
                f.write("not json{{{")
            self.assertEqual(run_all.read_checkpoint(path=p), {})


class TestMainFlow(unittest.TestCase):
    CP = {"run_id": "r1", "argv": ["--trade"], "mode": "full", "phase": "steps",
          "failed_step": "sentiment", "steps_done": ["fundamentals", "processing", "scoring"]}

    def test_transient_retries_then_succeeds(self):
        calls = []

        def fake_run(argv):
            calls.append(list(argv))
            if len(calls) < 3:
                return 1, "urllib.error.HTTPError: HTTP 429 Too Many Requests"
            return 0, "ok"

        with mock.patch.object(wd, "run_pipeline", side_effect=fake_run), \
             mock.patch.object(wd.run_all, "read_checkpoint", return_value=dict(self.CP)), \
             mock.patch("time.sleep") as slp:
            rc = wd.main(["--trade"])
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0], ["--trade"])                                   # first: as-is
        self.assertEqual(calls[1], ["--trade", "--from", "sentiment"])            # resume
        self.assertEqual(calls[2], ["--trade", "--from", "sentiment"])
        self.assertEqual(slp.call_count, 2)                                       # backoff between tries

    def test_retries_exhausted_diagnoses(self):
        with mock.patch.object(wd, "run_pipeline", return_value=(1, "HTTP 503 Service Unavailable")) as rp, \
             mock.patch.object(wd.run_all, "read_checkpoint", return_value=dict(self.CP)), \
             mock.patch("time.sleep"), \
             mock.patch.object(wd, "alpaca_snapshot", return_value={}) as snap, \
             mock.patch.object(wd, "diagnose_with_llm", return_value={"diagnosis": "d"}) as diag, \
             mock.patch.object(wd, "write_diagnosis_report", return_value="/tmp/d.md") as rep, \
             mock.patch.object(wd.run_all, "_notify") as notif:
            rc = wd.main(["--trade"])
        self.assertEqual(rc, 1)
        self.assertEqual(rp.call_count, wd.MAX_TRIES)
        diag.assert_called_once()
        rep.assert_called_once()
        notif.assert_called_once()
        self.assertIn("retries exhausted", notif.call_args[0][0])

    def test_trade_phase_never_retried(self):
        cp = dict(self.CP, phase="trade", failed_step=None)
        with mock.patch.object(wd, "run_pipeline", return_value=(1, "trade failed: boom")) as rp, \
             mock.patch.object(wd.run_all, "read_checkpoint", return_value=cp), \
             mock.patch.object(wd, "alpaca_snapshot", return_value={"account": {"Equity": 1}}), \
             mock.patch.object(wd, "diagnose_with_llm", return_value={"diagnosis": "d"}) as diag, \
             mock.patch.object(wd, "write_diagnosis_report", return_value="/tmp/d.md"), \
             mock.patch.object(wd.run_all, "_notify") as notif:
            rc = wd.main(["--trade"])
        self.assertEqual(rc, 1)
        rp.assert_called_once()                       # exactly one attempt - no auto-retry
        diag.assert_called_once()                     # diagnosed, not fixed
        notif.assert_called_once()
        self.assertIn("trade failed", notif.call_args[0][0])

    def test_unknown_error_diagnosed_not_retried(self):
        with mock.patch.object(wd, "run_pipeline", return_value=(1, "KeyError: 'Strategy_Weight'")) as rp, \
             mock.patch.object(wd.run_all, "read_checkpoint", return_value=dict(self.CP)), \
             mock.patch.object(wd, "alpaca_snapshot", return_value={}), \
             mock.patch.object(wd, "diagnose_with_llm", return_value={"diagnosis": "d"}), \
             mock.patch.object(wd, "write_diagnosis_report", return_value="/tmp/d.md"), \
             mock.patch.object(wd.run_all, "_notify") as notif:
            rc = wd.main(["--trade"])
        self.assertEqual(rc, 1)
        rp.assert_called_once()                       # code errors are never retried
        notif.assert_called_once()

    def test_missing_upstream_resumes_once(self):
        calls = []

        def fake_run(argv):
            calls.append(list(argv))
            return 1, "FileNotFoundError: 'Reports/weighted_sentiment.csv'"

        with mock.patch.object(wd, "run_pipeline", side_effect=fake_run), \
             mock.patch.object(wd.run_all, "read_checkpoint", return_value=dict(self.CP)), \
             mock.patch.object(wd, "alpaca_snapshot", return_value={}), \
             mock.patch.object(wd, "diagnose_with_llm", return_value={"diagnosis": "d"}), \
             mock.patch.object(wd, "write_diagnosis_report", return_value="/tmp/d.md"), \
             mock.patch.object(wd.run_all, "_notify"):
            rc = wd.main(["--trade"])
        self.assertEqual(rc, 1)
        self.assertEqual(calls[0], ["--trade"])
        self.assertEqual(calls[1], ["--trade", "--from", "sentiment"])   # one upstream resume...
        self.assertEqual(len(calls), 2)                                  # ...then give up + diagnose


class TestDiagnoseLLM(unittest.TestCase):
    def test_no_key_returns_none(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(wd.diagnose_with_llm("main", "steps", "boom", {}))

    def test_parses_json_reply(self):
        payload = {"content": [{"type": "text",
                                "text": '{"diagnosis": "bad key", "confidence": "high"}'}]}

        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return json.dumps(payload).encode()

        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}, clear=True), \
             mock.patch("urllib.request.urlopen", return_value=FakeResp()):
            d = wd.diagnose_with_llm("main", "steps", "KeyError", {})
        self.assertEqual(d["diagnosis"], "bad key")
        self.assertEqual(d["confidence"], "high")

    def test_http_error_degrades_to_none(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}, clear=True), \
             mock.patch("urllib.request.urlopen", side_effect=Exception("nope")):
            self.assertIsNone(wd.diagnose_with_llm("main", "steps", "boom", {}))


class TestLLMProviders(unittest.TestCase):
    """Gemini / Groq provider selection and request shapes (all HTTP stubbed)."""

    class _FakeResp:
        def __init__(self, payload):
            self._b = json.dumps(payload).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._b

    def test_provider_priority(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "g", "GROQ_API_KEY": "q",
                                           "ANTHROPIC_API_KEY": "a"}, clear=True):
            self.assertEqual(wd._llm_provider(), ("gemini", "gemini-2.0-flash", "g"))
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "q",
                                           "ANTHROPIC_API_KEY": "a"}, clear=True):
            self.assertEqual(wd._llm_provider(), ("groq", "llama-3.3-70b-versatile", "q"))
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "a"}, clear=True):
            self.assertEqual(wd._llm_provider(), ("anthropic", "claude-3-5-haiku-latest", "a"))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(wd._llm_provider(), (None, None, None))

    def test_model_override(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "g",
                                           "PIPELINE_LLM_MODEL": "gemini-2.5-flash"}, clear=True):
            self.assertEqual(wd._llm_provider(), ("gemini", "gemini-2.5-flash", "g"))

    def test_gemini_request_and_parse(self):
        payload = {"candidates": [{"content": {"parts": [{"text": '{"diagnosis": "gem blew it"}'}]}}]}
        seen = {}

        def fake_urlopen(req, timeout=None):
            seen["url"] = req.full_url
            seen["body"] = json.loads(req.data.decode())
            return self._FakeResp(payload)

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "g-key"}, clear=True), \
             mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            d = wd.diagnose_with_llm("main", "steps", "KeyError", {})
        self.assertIn("generativelanguage.googleapis.com", seen["url"])
        self.assertIn("gemini-2.0-flash", seen["url"])
        self.assertIn("contents", seen["body"])
        self.assertEqual(seen["body"]["generationConfig"]["maxOutputTokens"], 1500)
        self.assertEqual(d["diagnosis"], "gem blew it")

    def test_groq_request_and_parse(self):
        payload = {"choices": [{"message": {"content": '{"diagnosis": "llama says hi",'
                                                        '"confidence": "high"}'}}]}
        seen = {}

        def fake_urlopen(req, timeout=None):
            seen["url"] = req.full_url
            seen["auth"] = req.get_header("Authorization")
            seen["body"] = json.loads(req.data.decode())
            return self._FakeResp(payload)

        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "q-key"}, clear=True), \
             mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            d = wd.diagnose_with_llm("sentiment", "steps", "429", {})
        self.assertEqual(seen["url"], "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(seen["auth"], "Bearer q-key")
        self.assertEqual(seen["body"]["model"], "llama-3.3-70b-versatile")
        self.assertEqual(seen["body"]["messages"][0]["role"], "user")
        self.assertNotIn("q-key", json.dumps(seen["body"]))      # key never in body/prompt
        self.assertEqual(d["diagnosis"], "llama says hi")
        self.assertEqual(d["confidence"], "high")

    def test_non_json_reply_falls_back_to_text(self):
        payload = {"candidates": [{"content": {"parts": [{"text": "plain words, no json"}]}}]}
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "g"}, clear=True), \
             mock.patch("urllib.request.urlopen", return_value=self._FakeResp(payload)):
            d = wd.diagnose_with_llm("main", "steps", "boom", {})
        self.assertEqual(d, {"diagnosis": "plain words, no json"})


if __name__ == "__main__":
    unittest.main(verbosity=1)
