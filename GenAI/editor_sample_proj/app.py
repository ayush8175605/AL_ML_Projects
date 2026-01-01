import streamlit as st
from streamlit.components.v1 import html

from AL_ML_Projects.GenAI.editor_sample_proj.backend import stream_text_streamlit
from AL_ML_Projects.GenAI.editor_sample_proj.frontend import auth_page
from backend import ChatConversationMemory, chatbot, chatbot_stream, proofreader, summarizer, detect_styles, \
    rewrite_in_style, reader_reactions_func, expert_review, on_text_change, writing_agent
from copy import deepcopy
from prompts import main_prompt

# ---------------------------------------------
# 1. Page Setup
# ---------------------------------------------
st.set_page_config(page_title="WriterAI", page_icon="✨", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatConversationMemory(main_prompt)
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "page_mode" not in st.session_state:
    st.session_state.page_mode = "login"  # or "signup"
if "username" not in st.session_state:
    st.session_state.username = ""
if "proofread_result" not in st.session_state:
    st.session_state.proofread_result = {}
if "pending_replace_text" not in st.session_state:
    st.session_state.pending_replace_text = None
if "full_text" not in st.session_state:
    st.session_state.full_text = ""
if "summary_result" not in st.session_state:
    st.session_state.summary_result = {}
if "rewrite_result" not in st.session_state:
    st.session_state.rewrite_result = ""
if "rewrite_styles" not in st.session_state:
    st.session_state.rewrite_styles = {}
if "rewrite_style_selected" not in st.session_state:
    st.session_state.rewrite_style_selected = ""
if "show_rewrite" not in st.session_state:
    st.session_state.show_rewrite = False
if "reader_reaction" not in st.session_state:
    st.session_state.reader_reaction = {}
if "expert_review" not in st.session_state:
    st.session_state.expert_review = {}
if "editor_text" not in st.session_state:
    st.session_state.editor_text = ""

# 2. LOGIN / SIGNUP SYSTEM
if not st.session_state.logged_in:
    auth_page()

# ---------------------------------------------
# 3. Chat Interface
# ---------------------------------------------
# st.title()
# if st.session_state.logged_in:
st.markdown(
    f"""
    <h1 style='text-align: center; color: #3a3a3a;'>
        Hi, there! How can I help you today?
    </h1>
    """,
    unsafe_allow_html=True
)

# Editable Text Area
# --------------------------------------------------------
st.subheader("Your Document")
# --- Toolbar (Proofread, Rewrite, Summarize etc.)
col1, col2, col3, col4, col5, col6 = st.columns([1.2, 1, 1, 1, 1, 1])
with col1:
    st.markdown("### ✍️ Tools:")
with col2:
    proofread_clicked = st.button("🧠 Proofread Text", use_container_width=True)
with col3:
    summarize_clicked = st.button("📝 Summarize", use_container_width=True)
with col4:
    rewrite_clicked = st.button("🔄 Rewrite", use_container_width=True)
with col5:
    readers_reaction = st.button("🎭 Reader Reactions", use_container_width=True)
with col6:
    experts_review = st.button("🧩 Experts Review", use_container_width=True)

CHAR_LIMIT = 9500

if st.session_state.pending_replace_text is None:
    st.text_area(
        "Paste or write your text here:",
        value=st.session_state.get("full_text", ""),
        height=500,
        key="editor_text",
        on_change=on_text_change(CHAR_LIMIT),
    )

    char_count = st.session_state.get("char_count", 0)
    st.markdown(f"**Character count (excluding spaces):** {char_count}/{CHAR_LIMIT}")

    if char_count >= CHAR_LIMIT:
        st.error("⚠️ Character limit reached!")
    elif char_count >= int(CHAR_LIMIT * 0.8):
        st.warning("⚠️ You’re nearing the limit.")


# --- Proofread Logic
if proofread_clicked and st.session_state.full_text.strip():
    with st.spinner("Analysing your text for spelling, grammar, tone, and clarity..."):
        proofreading_reply = proofreader(st.session_state.full_text)
        st.session_state.proofread_result = proofreading_reply

    if "suggestions" in st.session_state.proofread_result and 'correctly_written_text' in st.session_state.proofread_result:
        suggestions = st.session_state.proofread_result["suggestions"]
        corrected_text = st.session_state.proofread_result["correctly_written_text"]
        st.markdown("### 🧠 Proofreading Results")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ✏️ Your Original Text")
            user_text = st.text_area(
                "Original",
                value=st.session_state.full_text,
                height=400,
                key="user_text_area"
            )
            if "original_score" in st.session_state.proofread_result:
                st.markdown(f"Score for your original text: **{st.session_state.proofread_result['original_score']}/10**")

        with col2:
            st.markdown("#### 🤖 AI-Corrected Version")
            ai_text = st.text_area(
                "AI Corrected",
                value=corrected_text,
                height=400,
                key="ai_text_area"
            )
            if "ai_version_score" in st.session_state.proofread_result:
                st.markdown(f"Score for the improved text: **{st.session_state.proofread_result['ai_version_score']}/10**")

        st.markdown("---")
        st.markdown("### 💡 Suggestions & Notes")
        st.markdown(suggestions)

    st.markdown("Click on the button below to close the proofreading session, or click on the other widgets to enjoy different functionalities!")
    if st.button("End Session!"):
        proofread_clicked=False
        st.rerun()

if summarize_clicked and st.session_state.full_text.strip():
    with st.spinner("Understanding and summarizing your text..."):
        summary_result = summarizer(st.session_state.full_text)
        st.session_state.summary_result = summary_result

    data = st.session_state.summary_result

    if "type_detected" in data:
        st.markdown(f"### 🧩 Detected Type: **{data['type_detected']}**")

    if "summary" in data:
        st.markdown(f"**Summary:** {data['summary']}")

    if "key_elements" in data:
        st.markdown(f"### Key Elements:")
        for k, v in data["key_elements"].items():
            st.markdown(f"**{k.capitalize()}:** {v}")

    if "reader_impressions" in data:
        st.markdown(f"**Reader Impression:** {data['reader_impression']}")

    st.markdown(
        "Click on the button below to clear this summary, or click on the other widgets to enjoy different functionalities!")
    if st.button("End Session!"):
        summarized_clicked = False
        st.rerun()

# --- Rewrite Logic ---

# 1. Check if the 'Rewrite' tool was just clicked OR if the rewrite UI is already open
if rewrite_clicked and st.session_state.full_text.strip():
    # If the main button is clicked, open the rewrite UI
    st.session_state.show_rewrite = True
    # Clear previous results to start fresh (optional but good practice)
    st.session_state.rewrite_result = ""
    st.session_state.rewrite_style_selected = ""

    # Process the initial style detection only once
    with st.spinner("Analysing your text for possible rewrite styles..."):
        # The key line where the styles are detected
        style_options = detect_styles(st.session_state.full_text)
        st.session_state.rewrite_styles = style_options

# 2. Display the rewrite UI if the state is set to show it
if st.session_state.show_rewrite:
    # Use st.session_state.rewrite_styles which was populated above or in a previous run
    if st.session_state.rewrite_styles:
        st.markdown("### ✨ Choose Your Rewrite Style")
        if st.session_state.rewrite_styles:
            styles_data = st.session_state.rewrite_styles.get("possible_styles", [])

            if not styles_data:
                st.warning("No style options were returned. You can provide your own style.")
                st.session_state.rewrite_style_selected = st.text_input("Choose your rewrite style")
                custom_description = st.text_area("Describe your desired style", height=100)
            else:
                st.markdown("#### Choose Your Style:")
                # Show styles with descriptions
                for i, style in enumerate(styles_data):
                    st.markdown(
                        f"**{i + 1}. {style['name']}** \n"
                        f"<span style='color:gray;'>{style['description']}</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown("---")

                # Selection (persist in session_state)
                style_names = [s["name"] for s in styles_data]
                # Add "Custom Style" option - FIXED: Use consistent name
                style_names.append("🎨 Custom Style")

                # Find the initial index based on the previously selected style, if any
                try:
                    default_index = style_names.index(st.session_state.rewrite_style_selected)
                except ValueError:
                    default_index = 0

                st.session_state.rewrite_style_selected = st.selectbox(
                    "Pick one:", style_names, index=default_index, key="rewrite_style_select"
                )

                # Show custom input fields if "Custom Style" is selected
                custom_description = ""
                custom_style_name = ""
                if st.session_state.rewrite_style_selected == "🎨 Custom Style":
                    st.markdown("#### Define Your Custom Style:")
                    custom_style_name = st.text_input(
                        "Style Name (optional)",
                        placeholder="e.g., Shakespearean, Technical, Casual",
                        key="custom_style_name"
                    )
                    custom_description = st.text_area(
                        "Style Description*",
                        placeholder="Describe how you want the text to be rewritten...",
                        height=100,
                        key="custom_style_desc"
                    )

            # Trigger rewrite
            rewrite_button_disabled = False
            if st.session_state.rewrite_style_selected == "🎨 Custom Style" and not custom_description:
                st.warning("⚠️ Please provide a style description for your custom style.")
                rewrite_button_disabled = True

            if st.button("🔄 Rewrite in this Style", key="rewrite_go", use_container_width=True,
                         disabled=rewrite_button_disabled):
                # Determine which style and description to use
                if st.session_state.rewrite_style_selected == "🎨 Custom Style" or not styles_data:
                    # Use custom style
                    style_name = custom_style_name if custom_style_name else "Custom"
                    sel = style_name + " style description: " + custom_description
                    # Update the display name if custom name was provided
                    if custom_style_name:
                        display_style = custom_style_name
                    else:
                        display_style = "Custom Style"
                else:
                    # Find the selected style's description from styles_data
                    selected_style_obj = next(
                        (s for s in styles_data if s["name"] == st.session_state.rewrite_style_selected), None)
                    if selected_style_obj:
                        sel = st.session_state.rewrite_style_selected + " style description: " + selected_style_obj[
                            "description"]
                        display_style = st.session_state.rewrite_style_selected
                    else:
                        sel = st.session_state.rewrite_style_selected
                        display_style = st.session_state.rewrite_style_selected

                with st.spinner(f"Rewriting text in '{display_style}' style..."):
                    # Call the backend function
                    result_dict = rewrite_in_style(st.session_state.full_text, sel)

                    # Assuming rewrite_in_style returns a dictionary like {"rewritten_text": "..."}
                    st.session_state.rewrite_result = result_dict
                    # Store the display style name for the header
                    st.session_state.display_style_name = display_style

                # No st.rerun() here, just update the state and let the display logic below run

            # If we have a result, show comparison
            if st.session_state.rewrite_result:
                display_name = st.session_state.get('display_style_name', st.session_state.rewrite_style_selected)
                st.markdown(f"### ✏️ Rewritten in *{display_name}* Style")
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📝 Original")
                    st.text_area("Original Text", value=st.session_state.full_text, height=400,
                                 key="original_rewrite", label_visibility = "collapsed")

                with col2:
                    st.subheader("✨ Rewritten")
                    st.text_area(
                        "Rewritten Text",
                        value=st.session_state.rewrite_result,
                        height=400,
                        key="rewritten_text_output",
                        label_visibility = "collapsed",
                    )

                # --- Accept & Replace Button ---
                # # You can add a button here if you want to replace the main text area content
                # if st.button("✅ Accept & Replace Text", key="accept_rewrite_replace"):
                #     st.session_state.full_text = st.session_state.rewrite_result
                #     st.session_state.show_rewrite = False  # Close the panel after replacement
                # st.rerun()  # Rerun to update the main text area

        st.markdown("---")
        st.markdown(
            "Click on the button below to close this rewrite session, or use the main text area."
        )
        # End Session button now explicitly controls the 'show_rewrite' state
        if st.button("End Rewrite Session!", key="close_rewrite_session", use_container_width=True):
            st.session_state.show_rewrite = False
            st.session_state.rewrite_styles = {}
            st.session_state.rewrite_result = ""
            st.rerun()  # Rerun to clear the panel
    # if st.button("Accept & Replace Text"):
    #         st.session_state.pending_replace_text = corrected_text  # defer update
    #         # Optional: clear the proofread panel now
    #         if "proofread_result" in st.session_state:
    #             del st.session_state["proofread_result"]
    #         st.success("Text will be updated…")
    #         st.rerun()

if readers_reaction and st.session_state.full_text.strip():
    with st.spinner("Analyzing your text for reader reactions..."):
        reaction_reply = reader_reactions_func(st.session_state.full_text)
        st.session_state.reader_reaction = reaction_reply

    st.markdown("# 🎭 Reader Reactions")

    # i# Reader Summary
    if "reader_summary" in st.session_state.reader_reaction:
        st.markdown("### 🧠 **Overall Reader Sentiment**")
        st.info(st.session_state.reader_reaction["reader_summary"], icon="💬")

    # Reader Questions
    if "questions_readers" in st.session_state.reader_reaction:
        st.markdown("### ❓ **Questions Readers Might Have**")
        st.markdown(
            f"""
            <div style="background-color:#f9f9f9; padding:12px; border-radius:10px; border:1px solid #e0e0e0;">
                💭 {st.session_state.reader_reaction["questions_readers"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    # Reader Reactions
    if "reactions_readers" in st.session_state.reader_reaction:
        st.markdown("### 🪶 **Reactions & Feelings of Readers**")
        st.markdown(
            f"""
            <div style="background-color:#f0f7ff; padding:12px; border-radius:10px; border:1px solid #cce0ff;">
                🎢 {st.session_state.reader_reaction["reactions_readers"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    if "readability" in st.session_state.reader_reaction:
        st.markdown("### ✨ **Readability**")
        st.markdown(
            f"""
            <div style="background-color:#f7f0ff; padding:12px; border-radius:10px; border:1px solid #cce0ff;">
                📖 {st.session_state.reader_reaction["readability"]}
            </div>
            """,
            unsafe_allow_html=True
        )
    # Overall Rating
    if "overall_rating" in st.session_state.reader_reaction:
        rating_value = float(st.session_state.reader_reaction["overall_rating"])
        st.markdown("### ⭐ **Overall Reader Rating**")
        st.metric(label="Average Reader Score", value=f"{rating_value:.1f} / 10")
        st.progress(min(rating_value / 10, 1.0))

    st.markdown(
        "Click on the button below to clear this reader reactions section, or click on the other widgets to enjoy different functionalities!")
    if st.button("End Session!"):
        readers_reaction = False
        st.rerun()

if experts_review and st.session_state.full_text.strip():
    with st.spinner("Analyzing your text for expert reviews..."):
        expert_reply = expert_review(st.session_state.full_text)
        st.session_state.expert_review = expert_reply

    experts = st.session_state.expert_review

    st.title("🧠 Expert Review Panel")
    st.divider()

    for _, expert in experts.items():
        with st.container():

            col1, col2 = st.columns([3,1])

            with col1:
                st.subheader(f"👤 {expert['name']}")
            with col2:
                st.caption(f"{len(expert['corrections'])} suggestions")

            st.markdown("**📝 Overview**")
            st.write(expert['explanation'])

            for idx, correction in enumerate(expert['corrections'], start=1):

                with st.expander(f"✏️ Suggestion {idx}", expanded=(idx == 1)):

                    col1, col2 = st.columns(2, gap="large")

                    with col1:
                        st.markdown("#### 🧾 Original")
                        st.info(correction['original_section'])

                    with col2:
                        st.markdown("#### ✅ Improved Version")
                        st.success(correction['corrected_section'])

    # # -----------------------------------------------
    # # Show rewritten text
    # # -----------------------------------------------
    # st.markdown("---")
    # st.markdown("<h2 style='text-align:center;'>✍️ Rewritten Chapter (Incorporating Expert Feedback)</h2>",
    #             unsafe_allow_html=True)
    #
    # col1, col2 = st.columns(2)
    #
    # with col1:
    #     st.markdown("### 🧾 Original Text")
    #     st.markdown(
    #         f"""
    #         <div style="
    #             background-color:#f8f9fa;
    #             border:1px solid #ddd;
    #             border-radius:10px;
    #             padding:20px;
    #             height:500px;
    #             overflow-y:auto;
    #         ">
    #             <p style="text-align:justify; line-height:1.6;">{st.session_state.full_text}</p>
    #         </div>
    #         """,
    #         unsafe_allow_html=True
    #     )
    #
    # with col2:
    #     st.markdown("### ✍️ Rewritten Text")
    #     st.markdown(
    #         f"""
    #         <div style="
    #             background-color:#fffbea;
    #             border:1px solid #e1b000;
    #             border-radius:10px;
    #             padding:20px;
    #             height:500px;
    #             overflow-y:auto;
    #         ">
    #             <p style="text-align:justify; line-height:1.6;">{rewrite_text}</p>
    #         </div>
    #         """,
    #         unsafe_allow_html=True
    #     )

    st.markdown(
        "Click on the button below to clear this experts review section, or click on the other widgets to enjoy different functionalities!")
    if st.button("End Session!"):
        readers_reaction = False
        st.rerun()
# st.markdown("✂️ *Optional:* Paste or type a section you want to focus on below:")
# selected_text = st.text_input("Selected portion (optional):")
# if selected_text:
#     st.session_state.selected_text = selected_text
st.markdown("For doing an **in-depth** analysis, try out our **chatbot** feature, where you can ask **any question** about your text!")
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
        st.session_state.chat_history = deepcopy(ChatConversationMemory(main_prompt))
        st.success("Chat cleared!")

with col4:
    if st.button("✏️ Clear Text", use_container_width=True):
        st.session_state.full_text = ""
        st.rerun()

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

        # if no chat yet: show only title
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
                if len(st.session_state.chat_history.get())>1:
                    st.session_state.chat_history.messages[1]={"role":"system", "content": f"The text to analyse: {st.session_state.full_text}"}
                else:
                    st.session_state.chat_history.add_text(f"The text to analyse: {st.session_state.full_text}")
                with st.spinner("WriterAI is typing..."):
                    # for token in chatbot_stream(user_input, st.session_state.chat_history):
                    #     full_reply += token
                    #     placeholder.markdown(full_reply)
                    st.session_state.chat_history.add_user(user_input)
                    full_reply = writing_agent(st.session_state.chat_history)

                stream_text_streamlit(full_reply, 0.01)
                st.session_state.chat_history.add_assistant(full_reply)
                # st.session_state.chat_history.add_user(user_input)
                # full_reply = writing_agent(st.session_state.chat_history)
                # placeholder.markdown(full_reply)
                # st.session_state.chat_history.add_assistant(full_reply)

            st.rerun()

        # ---- Autoscroll
        html("""
        <script>
        const box = window.parent.document.getElementById('chat-scroll');
        if (box) { box.scrollTop = box.scrollHeight; }
        </script>
        """, height=0)

st.markdown("<hr>", unsafe_allow_html=True)
logout_col = st.columns([9, 1])[1] 
with logout_col:
    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.page_mode = "login"
        st.session_state.full_text = ""
        st.rerun()