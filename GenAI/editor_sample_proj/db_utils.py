import os
import hashlib
import secrets
import psycopg2
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib, ssl
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        dbname=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
        port=int(os.getenv("DB_PORT", 5432))
    )

def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${hashed}"

def _verify_password(password: str, stored: str) -> bool:
    if "$" not in stored:
        # Legacy plain-text — force password reset flow
        return False
    salt, hashed = stored.split("$", 1)
    return hashlib.sha256((salt + password).encode()).hexdigest() == hashed

def add_user_to_system(username, password, email):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO writerai.users (username, password, email) VALUES (%s, %s, %s)",
            (username, _hash_password(password), email)
        )
        conn.commit()
        return True
    except psycopg2.IntegrityError:
        conn.rollback()
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT password FROM writerai.users WHERE (username=%s OR email=%s)",
        (username, username)
    )
    result = cursor.fetchone()
    conn.close()
    if not result:
        return False
    return _verify_password(password, result[0])

def check_username_exists(username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM writerai.users WHERE username=%s", (username,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def check_email_exists(email):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM writerai.users WHERE email=%s", (email,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def update_password(username, new_password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE writerai.users SET password=%s WHERE username=%s",
        (_hash_password(new_password), username)
    )
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0

def _send_email(to_email: str, subject: str, html: str):
    sender = os.getenv("GMAIL_SENDER")
    password = os.getenv("GMAIL_APP_PASSWORD")
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email
    msg.attach(MIMEText(html, "html"))
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender, password)
        server.sendmail(sender, to_email, msg.as_string())

def send_otp(email, otp):
    html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; color: #333;">
            <p>Hi there 👋,</p>
            <p>We received a request to reset your <b>WriterAI</b> account password.</p>
            <p>Your One-Time Password (OTP) is:</p>
            <h2 style="color: #0073e6; font-size: 24px;">{otp}</h2>
            <p>Please do not share it with anyone.</p>
            <p>If you did not request a password reset, you can safely ignore this email.</p>
            <br>
            <p>—<br><b>WriterAI Team</b><br>
            ThinkChat Technologies<br>
            <a href="mailto:{os.getenv('GMAIL_SENDER')}">{os.getenv('GMAIL_SENDER')}</a></p>
        </body>
        </html>
        """
    _send_email(email, "WriterAI Password Reset Verification Code", html)

def send_invitation_email(email, username):
    sender = os.getenv("GMAIL_SENDER")
    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; color: #333; background-color: #fafafa; padding: 20px;">
        <div style="max-width: 600px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);">
            <div style="text-align: center;">
                <h1 style="color: #0073e6; margin-bottom: 0;">✨ Welcome to WriterAI ✨</h1>
                <p style="font-size: 16px; color: #555;">by ThinkChat Technologies</p>
                <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
            </div>

            <p>Hi <b>{username}</b>, 👋</p>

            <p>Welcome to <b>WriterAI</b> — your new creative companion powered by <b>ThinkChat</b>!</p>
            <p>We're thrilled to have you on board. You can now explore WriterAI to:</p>

            <ul>
                <li>💡 Write, edit, and refine your text instantly</li>
                <li>🧠 Get feedback on tone, grammar, and clarity</li>
                <li>📚 Brainstorm ideas, blog posts, and stories</li>
                <li>✍️ Collaborate naturally with AI like never before</li>
            </ul>

            <p>We're constantly improving WriterAI to help you write better and faster. Stay tuned for upcoming features and updates.</p>

            <p>If you have any questions or suggestions, just reply to this email — we'd love to hear from you!</p>

            <br>
            <p>Warm regards,<br><b>The WriterAI Team</b><br>
            ThinkChat Technologies</p>

            <hr style="border: none; border-top: 1px solid #eee; margin-top: 30px;">
            <p style="font-size: 12px; color: #777; text-align: center;">
                You're receiving this email because you recently created an account on WriterAI.<br>
                If this wasn't you, please ignore this message.<br><br>
                ✉️ <a href="mailto:{sender}" style="color:#0073e6;">{sender}</a>
            </p>
        </div>
    </body>
    </html>
    """
    _send_email(email, "🎉 Welcome to WriterAI!", html)

def get_user_by_email(email):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM writerai.users WHERE email=%s", (email,))
    user = cursor.fetchone()
    conn.close()
    return user
