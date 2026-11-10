import streamlit as st
from src.bot import DepartmentChatbot

st.set_page_config(
    page_title="Department AI Chatbot",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Department Assistant")
st.caption("Course Project for IE624 - Interactive Department Q&A Chatbot")

# Initialize chatbot and session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = DepartmentChatbot()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Department Assistant. Ask me about courses, faculty, schedules, or department policies."}
    ]

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the department..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate bot response
    with st.chat_message("assistant"):
        response = st.session_state.chatbot.respond(
            query=prompt,
            history=st.session_state.messages[:-1]
        )
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
