import os
import emails
from emails.template import JinjaTemplate
from dotenv import load_dotenv

load_dotenv()

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAILS_FROM_EMAIL = os.getenv("EMAILS_FROM_EMAIL")
EMAILS_FROM_NAME = os.getenv("EMAILS_FROM_NAME")

EMAIL_TEMPLATE = """
<html>
<body>
    <div style="font-family: Arial, sans-serif; padding: 20px; border: 1px solid #ddd;">
        <h2 style="color: #2e6c80;">Thông báo điểm danh: {{ student_name }}</h2>
        <p>Xin chào phụ huynh,</p>
        <p>Hệ thống Vision Guard xin thông báo trạng thái của học sinh <strong>{{ student_name }}</strong>:</p>
        <ul>
            <li><strong>Trạng thái:</strong> {{ status }}</li>
            <li><strong>Thời gian:</strong> {{ time }}</li>
        </ul>
        <p>Vui lòng liên hệ với nhà trường nếu có bất kỳ thắc mắc nào.</p>
        <hr>
        <p style="font-size: 0.8em; color: #888;">Đây là email tự động, vui lòng không phản hồi.</p>
    </div>
</body>
</html>
"""

def send_attendance_email(to_email: str, student_name: str, status: str, time: str):
    # Mapping status to Vietnamese
    status_map = {
        "on_bus": "Đã lên xe",
        "off_bus": "Đã xuống xe"
    }
    status_vn = status_map.get(status, status)

    message = emails.Message(
        html=JinjaTemplate(EMAIL_TEMPLATE),
        subject=f"[Vision Guard] Thông báo điểm danh: {student_name}",
        mail_from=(EMAILS_FROM_NAME, EMAILS_FROM_EMAIL),
    )

    try:
        response = message.send(
            to=to_email,
            render={
                "student_name": student_name,
                "status": status_vn,
                "time": time
            },
            smtp={
                "host": SMTP_HOST,
                "port": SMTP_PORT,
                "ssl": True if SMTP_PORT == 465 else False,
                "tls": True if SMTP_PORT == 587 else False,
                "user": SMTP_USER,
                "password": SMTP_PASSWORD
            }
        )
        if response.status_code != 250:
            print(f"Email delivery failed with status {response.status_code}: {response.error}")
        return response.status_code == 250
    except Exception as e:
        print(f"SMTP Exception: {e}")
        return False
