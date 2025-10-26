import streamlit as st
from streamlit.components.v1 import html
from backend import ChatConversationMemory, chatbot, system_prompt, chatbot_stream
from copy import deepcopy

# ---------------------------------------------
# 1. Page Setup
# ---------------------------------------------
st.set_page_config(page_title="WriterAI", page_icon="✨", layout="wide")

# ---------------------------------------------
# 2. Simple Login System
# ---------------------------------------------
USER_CREDENTIALS = {
    "admin": "734238@admin",
    "aditi": "aditi@aditi",
    "ayush": "ayu@1012",
    "suman": "mummytummy"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatConversationMemory(system_prompt)
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

if not st.session_state.logged_in:
    st.title("🔐 Login to WriterAI")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.chat_history = ChatConversationMemory(system_prompt)
            st.success("Login successful! Redirecting to chat...")
            st.experimental_rerun()
        else:
            st.error("Invalid credentials.")
    st.stop()

# ---------------------------------------------
# 3. Chat Interface
# ---------------------------------------------
# st.title()

st.markdown(
    f"""
    <h1 style='text-align: center; color: #3a3a3a;'>
        Hi {st.session_state.username}! How can I help you today?
    </h1>
    """,
    unsafe_allow_html=True
)

# Editable Text Area
# --------------------------------------------------------
st.subheader("Your Document")
st.session_state.full_text = st.text_area(
    "Paste or write your text here:",
    value=st.session_state.get("full_text", ""),
    height=500,
    key="editor_text"
)
#
# st.markdown("✂️ *Optional:* Paste or type a section you want to focus on below:")
# selected_text = st.text_input("Selected portion (optional):")
# if selected_text:
#     st.session_state.selected_text = selected_text

# ------------------------------------------------------------
# Control Buttons Row (Open / Close / Clear Chat / Clear Text)
# ------------------------------------------------------------
_, col1, col2, col3, col4 = st.columns([5, 1, 1, 1, 1])

with col1:
    if st.button("💬 Open Chat", use_container_width=True):
        st.session_state.chat_open = True

with col2:
    if st.button("❌ Close Chat", use_container_width=True):
        st.session_state.chat_open = False

with col3:
    if st.button("🧹 Clear Chat", use_container_width=True):
        # Reset chat memory (preserve system prompt)
        st.session_state.chat_history = deepcopy(ChatConversationMemory(system_prompt))
        st.success("Chat cleared!")

with col4:
    if st.button("✏️ Clear Text", use_container_width=True):
        st.session_state.full_text = ""
        st.experimental_rerun()


# ------------------------------------------------------------
# Sidebar "Popup" Chat (ChatGPT-like sidebar behavior)
# ------------------------------------------------------------
if st.session_state.chat_open:
    with st.sidebar:
        # ---- CSS (full height, scrollable chat, fixed input)
        st.markdown("""
        <style>
        div[data-testid="stSidebarContent"] {
            display: flex;
            flex-direction: column;
            height: 100vh;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }
        .chat-wrap {
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        .chat-scroll {
            flex: 1 1 auto;
            overflow-y: auto;
            margin-bottom: 8px;
            padding-right: 6px;
        }
        .chat-input {
            position: sticky;
            bottom: 0;
            background-color: white;
            border-top: 1px solid #e5e7eb;
            padding-top: 8px;
            padding-bottom: 4px;
        }
        </style>
        """, unsafe_allow_html=True)

        # ---- Structure
        st.markdown("<div class='chat-wrap'>", unsafe_allow_html=True)

        # 👇 if no chat yet: show only title
        if len(st.session_state.chat_history.get()) <= 1:  # only system prompt
            st.subheader("💬 Chat with WriterAI")

        # ---- Scrollable chat messages area
        st.markdown("<div class='chat-scroll' id='chat-scroll'>", unsafe_allow_html=True)

        if len(st.session_state.chat_history.get()) > 1:  # messages exist
            for msg in st.session_state.chat_history.get()[1:]:
                if msg["role"] == "assistant":
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(msg["content"])
                elif msg["role"] == "user":
                    with st.chat_message("user", avatar="🧑‍💻"):
                        st.markdown(msg["content"])

        st.markdown("</div>", unsafe_allow_html=True)  # close .chat-scroll

        # ---- Fixed input area
        st.markdown("<div class='chat-input'>", unsafe_allow_html=True)
        with st.form("writerai_chat_form", clear_on_submit=True):
            user_input = st.text_input("Type your message here:", key="chat_box")
            send = st.form_submit_button("Send")
        st.markdown("</div>", unsafe_allow_html=True)  # close .chat-input

        st.markdown("</div>", unsafe_allow_html=True)  # close .chat-wrap

        # ---- Handle send event
        if send and user_input.strip():
            # Display user's message immediately
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(user_input)

            # Stream assistant response
            with st.chat_message("assistant", avatar="🤖"):
                placeholder = st.empty()
                full_reply = ""
                text_to_analyse = (
                    st.session_state.selected_text
                    if "selected_text" in st.session_state and st.session_state.selected_text
                    else (st.session_state.full_text or "")
                )
                with st.spinner("WriterAI is typing..."):
                    for token in chatbot_stream(user_input, st.session_state.chat_history, text_to_analyse):
                        full_reply += token
                        placeholder.markdown(full_reply)

            st.rerun()

        # ---- Autoscroll
        html("""
        <script>
        const box = window.parent.document.getElementById('chat-scroll');
        if (box) { box.scrollTop = box.scrollHeight; }
        </script>
        """, height=0)