"""Tests for run_all.py helpers (no notebooks run, no broker calls).

Covers: _notebook_error_summary
extracts cell/line/error from nbconvert output (with fallbacks), run_cmd returns
(exit code, captured output).
Run: PYTHONPATH=. python tests/test_run_all_helpers.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import run_all as r

FAIL = []


def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what)
    if not ok:
        FAIL.append(what)


def test_notebook_error_summary_traceback():
    out = ("[NbConvertApp] Converting notebook main_signal_analysis.ipynb to notebook\n"
           "Traceback (most recent call last):\n"
           '  File "<frozen>", line 1, in <module>\n'
           "Cell In[12], line 5\n"
           '----> 5 df["bad"]\n'
           "KeyError: 'bad'\n")
    s = r._notebook_error_summary(out, "main_signal_analysis.ipynb")
    check("cell 12, line 5" in s and "KeyError: 'bad'" in s,
          f"error summary extracts cell/line/error (got {s!r})")


def test_notebook_error_summary_nbconvert_error():
    out = "[NbConvertApp] ERROR | Notebook JSON is invalid: blah\n"
    s = r._notebook_error_summary(out, "x.ipynb")
    check("unknown location" in s and "Notebook JSON is invalid" in s,
          f"error summary falls back to the nbconvert ERROR line (got {s!r})")


def test_notebook_error_summary_empty():
    s = r._notebook_error_summary("", "x.ipynb")
    check("see the run log" in s, f"error summary on empty output points at the log (got {s!r})")
    check("see the run log" in r._notebook_error_summary(None, "x.ipynb"),
          "error summary handles None output")


def test_run_cmd_returns_rc_and_output():
    env = dict(os.environ)
    rc, out = r.run_cmd([sys.executable, "-c", "print('hello-nb')"], env)
    check(rc == 0 and "hello-nb" in out, "run_cmd returns (0, output) on success")
    rc, out = r.run_cmd([sys.executable, "-c", "import sys; print('bye'); sys.exit(3)"], env)
    check(rc == 3 and "bye" in out, "run_cmd returns (nonzero, output) on failure")


if __name__ == "__main__":
    test_notebook_error_summary_traceback()
    test_notebook_error_summary_nbconvert_error()
    test_notebook_error_summary_empty()
    test_run_cmd_returns_rc_and_output()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILURES:")
        for f in FAIL:
            print(" -", f)
        sys.exit(1)
    print("PASS test_run_all_helpers.py")
