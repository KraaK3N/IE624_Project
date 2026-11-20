import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from src.bot import DepartmentChatbot  # noqa: E402

st.set_page_config(page_title="SysCon Assistant", page_icon="🎓", layout="centered")
st.title("🎓 SysCon Assistant")
st.caption("Centre for Systems and Control, IIT Bombay · answers from the department website, with sources")


@st.cache_resource(show_spinner="Loading models (first time only)...")
def load_bot() -> DepartmentChatbot:
    return DepartmentChatbot()


bot = load_bot()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! Ask me about SysCon courses, faculty, admissions, forms or department rules.",
    }]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask a question about SysCon..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        stream = bot.respond_stream(prompt, history=st.session_state.messages[1:-1])
        with st.spinner("Thinking..."):
            first = next(stream, "")

        def rest():
            yield first
            yield from stream

        reply = st.write_stream(rest())
    st.session_state.messages.append({"role": "assistant", "content": reply})
