"""sentiment_analysis.ipynb fetch cell under an API rate limit (fake NewsAPI / Finnhub, no network): the first 429 /
rateLimited skips the remaining symbols with a log line, is never retried, and the step does not raise (no articles ->
re-score offline). Also: pipeline_watchdog classifies a quota error as 'quota' (never retried)."""
import io, json, os, sys, tempfile, types, contextlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)

def cells():
    nb = json.load(open(os.path.join(ROOT, "sentiment_analysis.ipynb")))
    code = [''.join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    return code[0], code[1]

def run(news_err, fh_err):
    calls = {"news": 0, "fh": 0}
    class NewsApiClient:
        def __init__(self, **k): pass
        def get_everything(self, **k):
            calls["news"] += 1
            raise news_err
    class Client:
        def __init__(self, **k): pass
        def company_news(self, *a, **k):
            calls["fh"] += 1
            if fh_err: raise fh_err
            return []
    sys.modules["newsapi"] = types.SimpleNamespace(NewsApiClient=NewsApiClient)
    sys.modules["finnhub"] = types.SimpleNamespace(Client=Client)
    os.environ.update(PIPELINE_SENTIMENT_MODE="online", PIPELINE_NEWS_APPROVED="1")
    os.environ.pop("PIPELINE_CALLS_FILE", None)
    setup, fetch = cells()
    g = {"__name__": "__nb__"}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(setup, g)
        g["CACHE_DIR"] = tempfile.mkdtemp()
        g["time"].sleep = lambda s: None
        exec(fetch, g)
    return g, calls, out.getvalue()

def main():
    g, calls, out = run(Exception("{'status': 'error', 'code': 'rateLimited', 'message': 'You have made too many requests'}"),
                        Exception("FinnhubAPIException(status_code: 429): API limit reached"))
    assert calls["news"] == 1 and calls["fh"] == 1, calls          # no retry, rest skipped
    assert "skipping the remaining" in out and g["MODE"] == "offline", out[-400:]
    print("PASS NewsAPI + Finnhub limit: 1 call each, rest skipped, no raise, re-score offline")
    g, calls, out = run(Exception("HTTP 429 Too Many Requests"), None)
    assert calls["news"] == 1 and calls["fh"] == len(g["SYMBOLS"]), calls
    print("PASS NewsAPI 429: skipped after 1 call, Finnhub still covers every symbol")
    import pipeline_watchdog as pw
    assert pw.classify_failure("newsapi rateLimited: daily limit")[0] == "quota"
    assert pw.classify_failure("HTTP 503 Service Unavailable")[0] == "transient"
    print("PASS watchdog: quota never retried, 503 still transient")

if __name__ == "__main__":
    main()
