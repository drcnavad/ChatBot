"""The floating chat button (bottom-right, every tab, phone too) and its panel.
Logic: chatbot/core.py + chatbot/llm.py. A fragment, so chatting never reruns the rest of the page.
READ-ONLY: no order confirm / PIN UI."""
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
    history.add(role, text)


def _messages():
    if "sa_chat" not in st.session_state:
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
                for role, text in _messages()[-20:]:
                    with st.chat_message(role):
                        st.markdown(text.replace("$", "\\$"))
                with st.form("sa_chat_form", clear_on_submit=True):
                    text = st.text_input("Message", placeholder="", label_visibility="collapsed")
                    if st.form_submit_button("Send") and text.strip():
                        with st.spinner("..."):
                            _handle(text.strip())
                        _rerun()
