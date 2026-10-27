"""PIN gate for the dashboard chatbot's money-moving actions (chatbot/core.py).

The PIN is never stored: only a salted PBKDF2-SHA256 hash in .chat_pin.json (gitignored, file mode 600).
5 wrong tries lock trading for 15 minutes. No PIN file = every trade is refused.
Set or change it yourself:  python chatbot/set_pin.py
"""
import hashlib
import hmac
import json
import os
import secrets
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root: .chat_pin.json stays there
PIN_FILE = os.getenv("STOCK_ANALYSIS_CHAT_PIN", os.path.join(ROOT, ".chat_pin.json"))
ITERATIONS = 300_000
MAX_TRIES = 5
LOCK_SECS = 15 * 60

def _hash(pin, salt, iters=ITERATIONS):
    return hashlib.pbkdf2_hmac("sha256", str(pin).encode(), bytes.fromhex(salt), iters).hex()

def _read(path=None):
    try:
        with open(path or PIN_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None

def _write(data, path=None):
    path = path or PIN_FILE
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    os.chmod(path, 0o600)

def is_set(path=None):
    d = _read(path)
    return bool(d and d.get("hash") and d.get("salt"))

def set_pin(pin, path=None):
    pin = str(pin)
    if not (pin.isdigit() and 4 <= len(pin) <= 12):
        raise ValueError("PIN must be 4-12 digits")
    salt = secrets.token_hex(16)
    _write({"salt": salt, "hash": _hash(pin, salt), "iters": ITERATIONS, "fails": 0, "locked_until": 0}, path)

def locked_for(path=None, now=None):
    """Seconds left on the lockout (0 = not locked)."""
    d = _read(path) or {}
    return max(0, int(float(d.get("locked_until", 0)) - (now or time.time())))

def verify(pin, path=None, now=None):
    """(ok, message). Counts wrong tries and locks after MAX_TRIES."""
    now = now or time.time()
    d = _read(path)
    if not d or not d.get("hash"):
        return False, "Not available right now."
    left = locked_for(path, now)
    if left:
        return False, f"Locked. Try again in {left // 60 + 1} min."
    good = hmac.compare_digest(_hash(pin, d["salt"], d.get("iters", ITERATIONS)), d["hash"])
    if good:
        d.update(fails=0, locked_until=0)
        _write(d, path)
        return True, "OK"
    d["fails"] = int(d.get("fails", 0)) + 1
    if d["fails"] >= MAX_TRIES:
        d.update(fails=0, locked_until=now + LOCK_SECS)
        _write(d, path)
        return False, "Incorrect."
    _write(d, path)
    return False, f"Wrong PIN ({MAX_TRIES - d['fails']} tries left before a {LOCK_SECS // 60}-min lock)."
