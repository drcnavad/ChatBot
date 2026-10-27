"""The floating chat button (bottom-right, every tab, phone too) and its panel. Logic and the PIN gate: chatbot/core.py,
chatbot/pin.py. A fragment, so chatting never reruns the rest of the page."""
import streamlit as st

from chatbot import core as chat

CSS = """<style>
.st-key-sa_chat_panel [data-testid="InputInstructions"] { display: none !important; }
.st-key-sa_chat_fab { position: fixed; right: 1.1rem; bottom: 1.1rem; z-index: 999990; width: auto !important; }
.st-key-sa_chat_fab button { border-radius: 999px; padding: .55rem 1rem; box-shadow: 0 4px 14px rgba(0,0,0,.35); }
div[data-testid="stPopoverBody"]:has(.st-key-sa_chat_panel) { width: min(430px, 94vw); max-height: 75vh; overflow-y: auto; }
</style>"""

def _say(role, text):
    from chatbot import history
    _messages().append((role, text))
    history.add(role, text)                      # saved in SQLite for 1 year

def _messages():
    if "sa_chat" not in st.session_state:        # a new page load picks up where the saved chat left off
        from chatbot import history
        st.session_state["sa_chat"] = history.recent(20)
    return st.session_state["sa_chat"]

def _handle(text):
    hist = list(_messages())
    _say("user", text)
    try:
        from chatbot import llm as chat_llm
        r = chat_llm.respond(hist, text)
    except Exception:
        r = None
    if r is not None:
        st.session_state["sa_chat_pending"] = r["plan"]
        _say("assistant", r["text"])
        return
    cmd = chat.parse(text)          # no model reachable: keyword commands
    try:
        if cmd["kind"] in ("order", "close", "cancel"):
            p, msg = chat.plan(cmd)
            st.session_state["sa_chat_pending"] = p
            _say("assistant", msg + ("\n\nConfirm below, or Cancel." if p else ""))
        else:
            _say("assistant", chat.answer(cmd))
    except Exception as e:
        _say("assistant", f"Couldn't reach Alpaca: {type(e).__name__}: {str(e)[:150]}")

def _rerun():
    try:
        st.rerun(scope="fragment")
    except st.errors.StreamlitAPIException:     # a full-page run (first load): rerun the page
        st.rerun()

@st.fragment
def render_chat():
    st.markdown(CSS, unsafe_allow_html=True)
    with st.container(key="sa_chat_fab"):
        with st.popover("💬 Chat"):
            with st.container(key="sa_chat_panel"):
                for role, text in _messages()[-20:]:
                    with st.chat_message(role):
                        st.markdown(text.replace("$", "\\$"))   # $...$ would render as math
                pending = st.session_state.get("sa_chat_pending")
                if pending:
                    from chatbot import pin as chat_pin
                    if not chat_pin.is_set():
                        st.error("Not available right now.")
                    with st.form("sa_chat_pin_form", clear_on_submit=True):
                        pin = st.text_input("Code", type="password", max_chars=12, label_visibility="collapsed")
                        c1, c2 = st.columns(2)
                        go = c1.form_submit_button("Confirm & send", type="primary")
                        stop = c2.form_submit_button("Cancel")
                    if stop:
                        st.session_state.pop("sa_chat_pending", None)
                        _say("assistant", "Cancelled. Nothing was sent.")
                        _rerun()
                    if go:
                        ok, msg = chat.execute(pending, pin)
                        if ok or "locked" in msg.lower() or "not available" in msg.lower() or "do-not-sell" in msg:
                            st.session_state.pop("sa_chat_pending", None)
                        _say("assistant", ("✅ " if ok else "⛔ ") + msg)
                        _rerun()
                else:
                    with st.form("sa_chat_form", clear_on_submit=True):
                        text = st.text_input("Message", placeholder="",
                                             label_visibility="collapsed")
                        if st.form_submit_button("Send") and text.strip():
                            with st.spinner("..."):
                                _handle(text.strip())
                            _rerun()
