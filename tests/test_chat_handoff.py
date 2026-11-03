"""chatbot/ui.send_handoff — webhook POST mocked; no network. Run: python tests/run_tests.py"""
import io
import json
import os
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from chatbot import ui as chat_ui

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f" ({detail})" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


# missing env
os.environ.pop("HANDOFF_WEBHOOK_URL", None)
os.environ.pop("HANDOFF_WEBHOOK_KEY", None)
check("missing env", chat_ui.send_handoff([("user", "hi")]) == "Handoff not set up yet")

os.environ["HANDOFF_WEBHOOK_URL"] = "https://example.invalid/handoff"
os.environ["HANDOFF_WEBHOOK_KEY"] = "test-key-do-not-use"
captured = {}


class FakeResp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return b"ok"


def fake_urlopen(req, timeout=None):
    captured["url"] = req.full_url
    captured["method"] = req.get_method()
    captured["timeout"] = timeout
    captured["headers"] = {k: v for k, v in req.header_items()}
    captured["body"] = json.loads(req.data.decode())
    return FakeResp()


_orig = urllib.request.urlopen
urllib.request.urlopen = fake_urlopen
try:
    msgs = [("user", "why APA?"), ("assistant", "earnings"), ("user", "transfer to you")]
    out = chat_ui.send_handoff(msgs)
finally:
    urllib.request.urlopen = _orig

check("success toast", out == "Sent, reply will come in Grok Bot")
check("POST", captured.get("method") == "POST")
check("timeout 10", captured.get("timeout") == 10)
check("auth bearer", captured.get("headers", {}).get("Authorization") == "Bearer test-key-do-not-use")
check("x-webhook-key", captured.get("headers", {}).get("X-webhook-key") == "test-key-do-not-use"
      or captured.get("headers", {}).get("X-Webhook-Key") == "test-key-do-not-use")
body = captured.get("body") or {}
check("question latest user", body.get("question") == "transfer to you")
check("transcript len", isinstance(body.get("transcript"), list) and len(body["transcript"]) == 3)
check("page", body.get("page") == "dashboard chat")
check("sent_at", isinstance(body.get("sent_at"), str) and "T" in body["sent_at"])

# HTTP error
def boom(req, timeout=None):
    raise urllib.error.HTTPError(req.full_url, 503, "no", hdrs=None, fp=io.BytesIO(b""))

urllib.request.urlopen = boom
try:
    out2 = chat_ui.send_handoff([("user", "x")])
finally:
    urllib.request.urlopen = _orig
check("http error status", out2 == "HTTP 503")

print("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
