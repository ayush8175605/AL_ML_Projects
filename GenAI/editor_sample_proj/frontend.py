import random
import time
import streamlit as st
from backend import ChatConversationMemory
from db_utils import verify_user, check_username_exists, check_email_exists, \
    add_user_to_system, send_invitation_email, get_user_by_email, send_otp, update_password
from prompts import main_prompt


def auth_page():

    # ---- LOGIN PAGE ----
    if st.session_state.page_mode == "login":
        st.markdown("<h2>🔐 Login to WriterAI</h2>", unsafe_allow_html=True)
        username = st.text_input("Username/Email", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", use_container_width=True):
            if verify_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.chat_history = ChatConversationMemory(main_prompt)
                st.success("Login successful! Redirecting...")
                time.sleep(2.5)
                st.rerun()
            else:
                st.error("Invalid username or password!")

        # Inline helper message and buttons side by side
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown(
                "**Don't have an account?** Create one below.",
                unsafe_allow_html=True
            )
            if st.button("🆕 Create Account", use_container_width=True):
                st.session_state.page_mode = "signup"
                st.rerun()

        with col_right:
            st.markdown(
                "**Forgot your password?** Reset below.",
                unsafe_allow_html=True
            )
            if st.button("🔒 Forgot Password", use_container_width=True):
                st.session_state.page_mode = "forgot"
                st.rerun()

    # ---- SIGNUP PAGE ----
    elif st.session_state.page_mode == "signup":
        st.markdown("<h2>🆕 Create a New Account</h2>", unsafe_allow_html=True)
        new_username = st.text_input("Choose a username", key="signup_username")
        new_email = st.text_input("Enter your email", key="signup_email")
        new_password = st.text_input("Choose a password", type="password", key="signup_password")

        if st.button("Create Account", use_container_width=True):
            if check_username_exists(new_username):
                st.warning("Username already taken!")
            elif check_email_exists(new_email):
                st.warning("Email already registered!")
            elif len(new_password) < 5:
                st.warning("Password too short!")
            else:
                success = add_user_to_system(new_username, new_password, new_email)
                if success:
                    send_invitation_email(new_email, new_username)
                    st.success("Account created successfully! Redirecting...")
                    time.sleep(2.5)
                    st.session_state.page_mode = "login"
                    st.rerun()
                else:
                    st.error("Something went wrong. Please try again!")

        # 👉 Inline “Back to Login” link
        st.markdown(
            """
            <p style='margin-top: 20px;'>
            🔙 Already have an account? <b>Login using the button below.</b></a>
            </p>
            """,
            unsafe_allow_html=True
        )

        if st.button("🔑 Back to Login", key="inline_login", use_container_width=True):
            st.session_state.page_mode = "login"
            st.rerun()

    # ---- FORGOT PASSWORD PAGE ----
    elif st.session_state.page_mode == "forgot":
        st.markdown("<h2>🔑 Reset Your Password</h2>", unsafe_allow_html=True)
        fp_email = st.text_input("Enter your registered email")

        if "otp_sent" not in st.session_state:
            st.session_state.otp_sent = False

        if not st.session_state.otp_sent:
            if st.button("Send OTP"):
                user = get_user_by_email(fp_email)
                if user:
                    otp = str(random.randint(100000, 999999))
                    send_otp(fp_email, otp)
                    st.session_state.sent_otp = otp
                    st.session_state.otp_sent = True
                    st.session_state.fp_email = fp_email
                    st.success(f"OTP successfully sent to {fp_email}!")
                    time.sleep(2.5)
                    st.rerun()
                else:
                    st.error("Email not found!")
        else:
            entered_otp = st.text_input("Enter OTP")
            new_pass = st.text_input("Enter new password", type="password")
            confirm_pass = st.text_input("Confirm new password", type="password")

            if st.button("Verify and Reset"):
                if entered_otp == st.session_state.sent_otp:
                    if new_pass == confirm_pass and len(new_pass) >= 5:
                        update_password(get_user_by_email(st.session_state.fp_email)[0], new_pass)
                        st.success("Password reset successful! Please login again.")
                        time.sleep(2.5)
                        # Cleanup
                        del st.session_state.otp_sent
                        del st.session_state.sent_otp
                        st.session_state.page_mode = "login"
                        st.rerun()
                    else:
                        st.error("Passwords don't match or too short!")
                else:
                    st.error("Invalid OTP!")

        if st.button("🔙 Back to Login"):
            st.session_state.page_mode = "login"
            st.rerun()
    st.stop()

