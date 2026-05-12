import os
import smtplib
from email.mime.text import MIMEText

from utils.logger import get_logger

logger = get_logger(__name__)


def send_alert(subject: str, body: str, recipient: str | None = None) -> None:
    smtp_host = os.environ.get("ALERT_SMTP_HOST")
    smtp_user = os.environ.get("ALERT_SMTP_USER")
    smtp_pass = os.environ.get("ALERT_SMTP_PASS")

    if not all([smtp_host, smtp_user, smtp_pass, recipient]):
        logger.warning("Alert not sent — SMTP not configured. Subject: %s", subject)
        return

    msg = MIMEText(body)
    msg["Subject"] = f"[YT-AutoPilot] {subject}"
    msg["From"] = smtp_user  # type: ignore[assignment]
    msg["To"] = recipient  # type: ignore[assignment]

    try:
        with smtplib.SMTP_SSL(smtp_host, 465) as server:  # type: ignore[arg-type]
            server.login(smtp_user, smtp_pass)  # type: ignore[arg-type]
            server.sendmail(smtp_user, [recipient], msg.as_string())  # type: ignore[list-item]
        logger.info("Alert email sent: %s", subject)
    except Exception as exc:
        logger.error("Failed to send alert email: %s", exc)
