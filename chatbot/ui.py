"""The floating chat button (bottom-right, every tab, phone too) and its panel.
Logic: chatbot/core.py + chatbot/llm.py. A fragment, so chatting never reruns the rest of the page.
READ-ONLY: no order confirm / PIN UI. Compact "Send to Mac Mini 4" handoff posts chat context to a webhook."""
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone

import streamlit as st

from chatbot import core as chat

CSS = """<style>
.st-key-sa_chat_panel [data-testid="InputInstructions"] { display: none !important; }
.st-key-sa_chat_fab { position: fixed; right: 1.1rem; bottom: 1.1rem; z-index: 999990; width: auto !important; }
.st-key-sa_chat_fab button { border-radius: 999px; padding: .55rem 1rem; box-shadow: 0 4px 14px rgba(0,0,0,.35); }
div[data-testid="stPopoverBody"]:has(.st-key-sa_chat_panel) { width: min(430px, 94vw); max-height: 75vh; overflow-y: auto; }
.st-key-sa_chat_panel .st-key-sa_handoff button { padding: .15rem .45rem; min-height: 1.6rem; font-size: .75rem; }
</style>"""


def _say(role, text):
    from chatbot import history
    _messages().append((role, text))
    history.add(role, text)


def _messages():
    if "sa_chat" not in st.session_state:
        from chatbot import history
        st.session_state["sa_chat"] = history.recent(20)
    return st.session_state["sa_chat"]


def send_handoff(messages=None):
    """POST last ~10 chat turns to HANDOFF_WEBHOOK_URL. Returns a short status string (no secrets)."""
    url = (os.getenv("HANDOFF_WEBHOOK_URL") or "").strip()
    key = (os.getenv("HANDOFF_WEBHOOK_KEY") or "").strip()
    if not url or not key:
        return "Handoff not set up yet"
    msgs = list(messages if messages is not None else _messages())[-10:]
    question = next((t for r, t in reversed(msgs) if r == "user"), "")
    body = json.dumps({
        "question": question,
        "transcript": [{"role": r, "content": t} for r, t in msgs],
        "page": "dashboard chat",
        "sent_at": datetime.now(timezone.utc).isoformat(),
    }).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}",
                 "X-Webhook-Key": key},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return "Sent, reply will come in Grok Bot" if 200 <= getattr(resp, "status", 200) < 300 else f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"
    except Exception as e:
        return f"fail: {type(e).__name__}"


def _handle(text):
    hist = list(_messages())
    _say("user", text)
    try:
        from chatbot import llm as chat_llm
        r = chat_llm.respond(hist, text)
    except Exception:
        r = None
    if r is not None:
        _say("assistant", r["text"])
        return
    cmd = chat.parse(text)
    try:
        _say("assistant", chat.answer(cmd))
    except Exception as e:
        _say("assistant", f"Couldn't reach Alpaca: {type(e).__name__}: {str(e)[:150]}")


def _rerun():
    try:
        st.rerun(scope="fragment")
    except st.errors.StreamlitAPIException:
        st.rerun()


@st.fragment
def render_chat():
    st.markdown(CSS, unsafe_allow_html=True)
    with st.container(key="sa_chat_fab"):
        with st.popover("💬 Chat"):
            with st.container(key="sa_chat_panel"):
                top = st.columns([5, 1])
                with top[1]:
                    with st.container(key="sa_handoff"):
                        if st.button("📤", help="Send to Mac Mini 4", key="sa_handoff_btn"):
                            st.session_state["sa_handoff_status"] = send_handoff()
                status = st.session_state.pop("sa_handoff_status", None)
                if status:
                    st.caption(status)
                for role, text in _messages()[-20:]:
                    with st.chat_message(role):
                        st.markdown(text.replace("$", r"\$"))
                with st.form("sa_chat_form", clear_on_submit=True):
                    text = st.text_input("Message", placeholder="", label_visibility="collapsed")
                    if st.form_submit_button("Send") and text.strip():
                        with st.spinner("..."):
                            _handle(text.strip())
                        _rerun()
