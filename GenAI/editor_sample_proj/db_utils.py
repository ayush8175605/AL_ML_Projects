import psycopg2
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib, ssl

def get_connection():
    # 🔹 Update with your actual credentials
    return psycopg2.connect(
        host="localhost",
        dbname="postgres",
        user="postgres",
        password="ayush1012",
        port=5432
    )

def add_user_to_system(username, password, email):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO writerai.users (username, password, email) VALUES (%s, %s, %s)", (username, password, email))
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
    cursor.execute("SELECT * FROM writerai.users WHERE (username=%s OR email=%s) AND password=%s", (username, username, password))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def check_username_exists(username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM writerai.users WHERE username=%s", (username,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def check_email_exists(email):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM writerai.users WHERE email=%s", (email,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def update_password(username, new_password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE writerai.users SET password=%s WHERE username=%s", (new_password, username))
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0

def send_otp(email, otp):
    sender = "noreply.writerai.thinkchat@gmail.com"
    password = "twjd wkim xypg tfqf"  # Use Gmail App Password
    subject = "WriterAI Password Reset Verification Code"
    # HTML version of email
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
            <a href="mailto:noreply.writerai.thinkchat@gmail.com">noreply.writerai.thinkchat@gmail.com</a></p>
        </body>
        </html>
        """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = email
    msg.attach(MIMEText(html, "html"))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender, password)
        server.sendmail(sender, email, msg.as_string())

def send_invitation_email(email, username):
    sender = "noreply.writerai.thinkchat@gmail.com"
    password = "twjd wkim xypg tfqf"  # Gmail App Password
    subject = "🎉 Welcome to WriterAI!"

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
            <p>We’re thrilled to have you on board. You can now explore WriterAI to:</p>

            <ul>
                <li>💡 Write, edit, and refine your text instantly</li>
                <li>🧠 Get feedback on tone, grammar, and clarity</li>
                <li>📚 Brainstorm ideas, blog posts, and stories</li>
                <li>✍️ Collaborate naturally with AI like never before</li>
            </ul>

            <p>We’re constantly improving WriterAI to help you write better and faster. Stay tuned for upcoming features and updates.</p>

            <p>If you have any questions or suggestions, just reply to this email — we’d love to hear from you!</p>

            <br>
            <p>Warm regards,<br><b>The WriterAI Team</b><br>
            ThinkChat Technologies</p>

            <hr style="border: none; border-top: 1px solid #eee; margin-top: 30px;">
            <p style="font-size: 12px; color: #777; text-align: center;">
                You’re receiving this email because you recently created an account on WriterAI.<br>
                If this wasn’t you, please ignore this message.<br><br>
                ✉️ <a href="mailto:noreply.writerai.thinkchat@gmail.com" style="color:#0073e6;">noreply.writerai.thinkchat@gmail.com</a>
            </p>
        </div>
    </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] =sender
    msg["To"] = email

    msg.attach(MIMEText(html, "html"))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender, password)
        server.sendmail(sender, email, msg.as_string())


def get_user_by_email(email):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM writerai.users WHERE email=%s", (email,))
    user = cursor.fetchone()
    conn.close()
    return user