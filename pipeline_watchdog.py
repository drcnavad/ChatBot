"""Self-healing wrapper around run_all.py: retry transient failures and resume the pipeline.

Usage (drop-in replacement for run_all.py - every argument is passed straight through):

    python pipeline_watchdog.py --trade          # evening run (Mon/Wed/Fri after 3:15 PM CT)
    python pipeline_watchdog.py --fill-check     # morning run (next trading day from 9:00 AM CT)

What it does on a failed run (exit code != 0):
  * Reads Reports/.pipeline_checkpoint.json (written by run_all.py after every step) to find
    where the pipeline stopped.
  * TRANSIENT errors (API rate limits, timeouts, connection drops, HTTP 5xx): waits with
    backoff (60s, 300s, 900s) and re-runs run_all.py --from <failed step>, up to 3 attempts.
  * MISSING upstream files (e.g. main fails because weighted_sentiment.csv is gone): re-runs
    once from the step that produces that file, then resumes.
  * ANYTHING ELSE (code errors, strategy bugs, ...): does NOT touch the code. It optionally
    asks an LLM for a diagnosis + proposed fix (see below), writes
    Reports/pipeline_diagnosis_<timestamp>.md, sends a macOS notification, and stops.
    A human (Chirag) reviews the diagnosis and applies the fix.

What it NEVER does (diagnose-only contract):
  * Never edits paper_trade.py, any notebook, or any other pipeline code.
  * Never places, cancels, or retries orders itself. Failures in the --trade or --fill-check
    phases are never auto-retried: they go straight to diagnosis + notification, because
    order submission is handled by the existing idempotent guards in paper_trade.py and a
    human should decide the next step.
  * Never invents pipeline steps or arguments; a resume is always `run_all.py --from <step>`
    with the original arguments.

The LLM diagnosis layer is optional and needs ONE free API key - put it in the
project's .env file (loaded automatically) or the environment:

    GEMINI_API_KEY=...      # Google AI Studio (aistudio.google.com) - recommended free tier
    GROQ_API_KEY=...        # console.groq.com - free tier, serves Llama 3.3 70B
    ANTHROPIC_API_KEY=...   # console.anthropic.com - paid

  The first key found wins (Gemini > Groq > Anthropic). Without any key the watchdog
  still retries transient failures; it just skips the LLM diagnosis and notifies with
  the raw log excerpt instead. Model override: PIPELINE_LLM_MODEL
  (defaults: gemini-2.0-flash / llama-3.3-70b-versatile / claude-3-5-haiku-latest).

  Diagnose-only contract: the LLM explains the failure and proposes a fix for human
  review. It is never applied automatically.

"Continuous" monitoring: the pipeline only runs on its schedule (launchd), so the watchdog
rides along with each scheduled run - between runs the pipeline is idle and there is nothing
to watch. To use it, point the launchd jobs at pipeline_watchdog.py instead of run_all.py.
"""
import collections
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN_ALL = os.path.join(ROOT, "run_all.py")
REPORTS = os.path.join(ROOT, "Reports")
CT = ZoneInfo("America/Chicago")

sys.path.insert(0, ROOT)
import run_all                                                        # noqa: E402  (checkpoints, _notify, STEPS)

MAX_TRIES = 3                       # total attempts for transient failures (1 initial + 2 retries)
RETRY_BACKOFF_S = [60, 300, 900]    # wait between transient retries
LOG_TAIL_LINES = 4000               # bounded buffer kept for error classification
LLM_TIMEOUT_S = 120


def _load_env():
    """Load the project's .env so scheduled (launchd) runs see the LLM/API keys."""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"))
    except Exception:
        pass                                            # dotenv missing: env vars still work


_load_env()

# Free/paid LLM providers for the diagnose-only layer: (env key, default model, id).
# The first configured key wins.
LLM_PROVIDERS = (
    ("GEMINI_API_KEY", "gemini-2.0-flash", "gemini"),
    ("GROQ_API_KEY", "llama-3.3-70b-versatile", "groq"),
    ("ANTHROPIC_API_KEY", "claude-3-5-haiku-latest", "anthropic"),
)


def _llm_provider():
    """(provider id, model, key) for the first configured provider, else (None, None, None)."""
    override = os.environ.get("PIPELINE_LLM_MODEL", "").strip()
    for env_key, default_model, provider in LLM_PROVIDERS:
        key = os.environ.get(env_key, "").strip()
        if key:
            return provider, (override or default_model), key
    return None, None, None

# Phases where money moves (or may have moved): never auto-retry, diagnose + notify only.
NO_RETRY_PHASES = ("trade", "fill-check")

# Upstream file -> the STEPS entry that produces it (for the missing-file resume).
UPSTREAM_STEP = {
    "weighted_sentiment.csv": "sentiment",
    "news_cleaned_df.csv": "sentiment",
    "balance_sheet_weights.csv": "scoring",
    "earnings_date.csv": "earnings",
    "signal_analysis.csv": "main",
    "strategy_picks.csv": "main",
    "strategy_changes.csv": "main",
    "strategy_holdings.csv": "main",
}

_TRANSIENT_RES = [
    r"\b429\b", r"rate[\s_-]?limit", r"too many requests",
    r"timeout", r"timed out", r"connection (reset|aborted|refused)", r"temporary failure",
    r"name resolution", r"\bssl\b", r"eof occurred", r"urlerror",
    r"HTTP (500|502|503|504)", r"internal server error", r"service unavailable", r"bad gateway",
]
_MISSING_FILE_RES = [
    r"FileNotFoundError.*?([\w\-]+\.csv)",
    r"No such file[^\n]*?([\w\-]+\.csv)",
]


def log(msg, *args):
    """Watchdog's own log line (goes to stdout -> the launchd log)."""
    print(f"[watchdog {datetime.now(CT):%H:%M:%S}] " + (msg % args if args else msg), flush=True)


def classify_failure(text):
    """('transient', None) | ('missing_upstream', step) | ('unknown', None) for the log tail.

    Pure: safe to unit-test. A missing upstream file is checked first (more specific). """
    text = text or ""
    for rx in _MISSING_FILE_RES:
        m = re.search(rx, text, re.IGNORECASE)
        if m:
            step = UPSTREAM_STEP.get(m.group(1))
            if step:
                return "missing_upstream", step
    low = text.lower()
    if any(re.search(rx, low) for rx in _TRANSIENT_RES):
        return "transient", None
    return "unknown", None


def resume_step(checkpoint):
    """The STEPS name to resume from: the failed step, else the step after the last completed one."""
    if not checkpoint:
        return None
    if checkpoint.get("failed_step"):
        return checkpoint["failed_step"]
    done = checkpoint.get("steps_done") or []
    names = [s[0] for s in run_all.STEPS]
    for n in reversed(names):
        if n in done:
            i = names.index(n)
            return names[i + 1] if i + 1 < len(names) else None
    return names[0] if names else None


def build_resume_argv(argv, step):
    """Original argv + '--from <step>', unless the caller already scoped the run."""
    if step is None or "--from" in argv or "--only" in argv:
        return list(argv)
    return list(argv) + ["--from", step]


def run_pipeline(argv):
    """Run run_all.py with argv; stream its output live and keep a bounded tail.

    Returns (exit code, tail text). """
    buf = collections.deque(maxlen=LOG_TAIL_LINES)
    proc = subprocess.Popen([sys.executable, RUN_ALL] + list(argv), cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        buf.append(line)
        print(line, end="", flush=True)
    return proc.wait(), "".join(buf)


def alpaca_snapshot():
    """Best-effort read-only Alpaca context for a diagnosis (never raises, never leaks keys)."""
    snap = {}
    try:
        import alpaca_paper
        acct = alpaca_paper.PaperAccount()
        s = acct.account_summary()
        snap["account"] = {k: s[k] for k in ("Equity", "Cash", "Buying power", "Status") if k in s}
        try:
            oo = acct.open_orders(limit=20)
            snap["open_orders"] = [{"Symbol": r["Symbol"], "Side": r["Side"], "Qty": str(r["Qty"]),
                                   "Status": r["Status"]} for r in oo.to_dict("records")]
        except Exception:
            snap["open_orders"] = "unreadable"
        try:
            snap["market"] = acct.market_clock()
        except Exception:
            pass
    except Exception as e:
        snap["error"] = f"{type(e).__name__}: {e}"
    return snap


def _llm_complete(provider, model, key, prompt):
    """One completion call. Returns the raw reply text. Raises on any failure.

    Keys are never logged or included in the prompt - Gemini takes its key in the
    HTTPS query string per Google's API contract; Groq/Anthropic use headers.
    """
    headers = {"content-type": "application/json"}
    if provider == "gemini":
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
               f":generateContent?key={key}")
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 1500, "temperature": 0.2}}
    elif provider == "groq":
        # OpenAI-compatible chat completions.
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers["Authorization"] = f"Bearer {key}"
        body = {"model": model, "max_tokens": 1500, "temperature": 0.2,
                "messages": [{"role": "user", "content": prompt}]}
    else:  # anthropic
        url = "https://api.anthropic.com/v1/messages"
        headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
        body = {"model": model, "max_tokens": 1500,
                "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_S) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if provider == "gemini":
        parts = payload["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    if provider == "groq":
        return payload["choices"][0]["message"]["content"]
    return "".join(b.get("text", "") for b in payload.get("content", [])
                   if b.get("type") == "text")


def diagnose_with_llm(step, phase, log_tail, snapshot):
    """Ask the LLM for a diagnosis + proposed fix. Returns a dict, or None when unavailable.

    Diagnose-only: the result is written to a report for human review - it is never applied.
    Never raises (any failure degrades to None so the raw notification still goes out). """
    provider, model, key = _llm_provider()
    if not provider:
        return None
    prompt = (
        "You are diagnosing a failed step of an automated stock-analysis pipeline that paper-trades "
        "stocks on Alpaca (paper account only). DIAGNOSE ONLY - do not apply anything.\n\n"
        f"Failed step: {step} (phase: {phase})\n"
        f"Alpaca snapshot (read-only): {json.dumps(snapshot)[:1500]}\n\n"
        "Log tail (most recent lines first are the most relevant):\n"
        f"{log_tail[-6000:]}\n\n"
        "Reply with JSON ONLY, no markdown fences, with these keys:\n"
        '{"diagnosis": "one-paragraph plain-English explanation of what broke",\n'
        ' "root_cause": "the most likely root cause",\n'
        ' "proposed_fix": "concrete fix: file path + what to change (a unified diff if short)",\n'
        ' "resume_step": "which pipeline step to resume from after the fix, or null",\n'
        ' "confidence": "high|medium|low"}'
    )
    try:
        text = _llm_complete(provider, model, key, prompt)
    except Exception as e:
        log("LLM diagnosis failed (%s: %s) - falling back to raw notification", type(e).__name__, e)
        return None
    try:
        d = json.loads(text)
        return d if isinstance(d, dict) else {"diagnosis": text}
    except ValueError:
        return {"diagnosis": text.strip()[:2000]}


def write_diagnosis_report(step, phase, diagnosis, log_tail, snapshot):
    """Write the human-review report. Returns the path. Never raises."""
    os.makedirs(REPORTS, exist_ok=True)
    path = os.path.join(REPORTS, f"pipeline_diagnosis_{datetime.now(CT):%Y%m%d_%H%M%S}.md")
    try:
        with open(path, "w") as f:
            f.write(f"# Pipeline diagnosis - {datetime.now(CT):%Y-%m-%d %I:%M %p CT}\n\n")
            f.write(f"**Failed step:** {step}  |  **Phase:** {phase}\n\n")
            if diagnosis:
                f.write(f"**Diagnosis:** {diagnosis.get('diagnosis', 'n/a')}\n\n")
                f.write(f"**Likely root cause:** {diagnosis.get('root_cause', 'n/a')}\n\n")
                f.write(f"**Proposed fix:**\n\n{diagnosis.get('proposed_fix', 'n/a')}\n\n")
                f.write(f"**Resume from:** {diagnosis.get('resume_step') or step}  "
                        f"|  **Confidence:** {diagnosis.get('confidence', 'n/a')}\n\n")
                f.write("> Diagnose-only: nothing was changed. Review the fix, apply it yourself\n"
                        "> (or ask Muse), then resume with:\n>\n"
                        f"> `python run_all.py --from {diagnosis.get('resume_step') or step}`\n\n")
            else:
                f.write("**Diagnosis:** LLM unavailable (no GEMINI_API_KEY / GROQ_API_KEY / ANTHROPIC_API_KEY) - raw log excerpt below.\n\n")
                f.write(f"> Resume manually with: `python run_all.py --from {step}`\n\n")
            if snapshot:
                f.write(f"**Alpaca snapshot:** `{json.dumps(snapshot)[:800]}`\n\n")
            f.write("<details><summary>Log tail</summary>\n\n```\n")
            f.write(log_tail[-6000:])
            f.write("\n```\n</details>\n")
    except OSError as e:
        log("could not write diagnosis report: %s", e)
        return ""
    return path


def handle_failure(argv, tail, checkpoint):
    """A run failed: retry what is safe, diagnose the rest. Returns the process exit code."""
    phase = checkpoint.get("phase", "steps") if checkpoint else "steps"
    step = resume_step(checkpoint)

    if phase in NO_RETRY_PHASES:
        # Money may have moved (or been about to): never auto-retry order-adjacent phases.
        log("FAILED in '%s' phase - no auto-retry (order-adjacent); diagnosing", phase)
        snapshot = alpaca_snapshot()
        diagnosis = diagnose_with_llm(step or phase, phase, tail, snapshot)
        report = write_diagnosis_report(step or phase, phase, diagnosis, tail, snapshot)
        run_all._notify(f"Pipeline needs attention: {phase} failed",
                        f"{(diagnosis or {}).get('diagnosis', 'see the run log')[:220]}"
                        + (f" - report: {os.path.basename(report)}" if report else ""))
        return 1

    kind, detail = classify_failure(tail)
    log("FAILED at step '%s' (phase %s) - classified as: %s%s", step, phase, kind,
        f" -> {detail}" if detail else "")

    if kind == "transient":
        return "retry_transient", step
    if kind == "missing_upstream":
        return "retry_upstream", detail
    snapshot = alpaca_snapshot()
    diagnosis = diagnose_with_llm(step or "unknown", phase, tail, snapshot)
    report = write_diagnosis_report(step or "unknown", phase, diagnosis, tail, snapshot)
    run_all._notify(f"Pipeline needs attention: step '{step}' failed",
                    f"{(diagnosis or {}).get('diagnosis', 'see the run log')[:220]}"
                    + (f" - report: {os.path.basename(report)}" if report else ""))
    return 1


def main(argv=None):
    """Entry point: run the pipeline, heal what is safe to heal, diagnose the rest."""
    args = list(argv) if argv is not None else list(sys.argv[1:])
    if "--help" in args or "-h" in args:
        print(__doc__)
        return 0

    tries, retried_upstream = 0, False
    cur_argv = list(args)
    while True:
        tries += 1
        log("attempt %d: run_all.py %s", tries, " ".join(cur_argv) or "(no args)")
        rc, tail = run_pipeline(cur_argv)
        if rc == 0:
            log("pipeline finished cleanly on attempt %d", tries)
            return 0
        checkpoint = run_all.read_checkpoint()
        action = handle_failure(args, tail, checkpoint)
        if action == 1:
            return 1
        kind, target = action
        if kind == "retry_transient" and tries < MAX_TRIES:
            wait = RETRY_BACKOFF_S[min(tries - 1, len(RETRY_BACKOFF_S) - 1)]
            log("transient failure - retrying from step '%s' in %ds (attempt %d/%d)",
                target, wait, tries + 1, MAX_TRIES)
            time.sleep(wait)
            cur_argv = build_resume_argv(args, target)
            continue
        if kind == "retry_upstream" and not retried_upstream:
            retried_upstream = True
            log("missing upstream file - re-running from producing step '%s' once", target)
            cur_argv = build_resume_argv(args, target)
            continue
        # Retries exhausted (or a retry itself failed non-transiently): diagnose and stop.
        log("no safe retry left - diagnosing for human review")
        snapshot = alpaca_snapshot()
        diagnosis = diagnose_with_llm(target or "unknown", checkpoint.get("phase", "steps"), tail, snapshot)
        report = write_diagnosis_report(target or "unknown", checkpoint.get("phase", "steps"),
                                        diagnosis, tail, snapshot)
        run_all._notify("Pipeline needs attention: retries exhausted",
                        f"step '{target}' still failing"
                        + (f" - report: {os.path.basename(report)}" if report else ""))
        return 1


if __name__ == "__main__":
    sys.exit(main())
