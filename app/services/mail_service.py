from __future__ import annotations

import smtplib
from email.message import EmailMessage

from app.core.config import get_settings


def send_verification_email(to_email: str, token: str) -> None:
    settings = get_settings()
    subject = "Verify your email - American Board System"
    base = settings.BASE_URL.rstrip("/")
    verify_url = f"{base}/api/v1/auth/verify?token={token}"
    body = (
        f"Welcome to {settings.APP_NAME}!\n\n"
        f"Please verify your email by visiting the following link:\n"
        f"{verify_url}\n\n"
        "If you did not sign up for this account, you can ignore this message."
    )

    msg = EmailMessage()
    msg["Subject"] = subject
    from_addr = settings.MAIL_FROM or settings.MAIL_USERNAME or ""
    msg["From"] = f"{settings.MAIL_FROM_NAME} <{from_addr}>"
    msg["To"] = to_email
    msg.set_content(body)

    if not (settings.MAIL_USERNAME and settings.MAIL_PASSWORD):
        return

    with smtplib.SMTP(settings.MAIL_HOST, settings.MAIL_PORT) as server:
        use_tls = settings.MAIL_USE_TLS or (str(settings.MAIL_ENCRYPTION or "").lower() in {"tls", "starttls"})
        if use_tls:
            server.starttls()
        server.login(settings.MAIL_USERNAME, settings.MAIL_PASSWORD)
        server.send_message(msg)


def send_verification_code_email(to_email: str, code: str) -> None:
    settings = get_settings()
    subject = "Your verification code - American Board System"
    body = (
        f"Welcome to {settings.APP_NAME}!\n\n"
        f"Your email verification code is: {code}\n"
        f"This code expires in {settings.EMAIL_VERIFICATION_CODE_EXPIRE_MINUTES} minutes.\n\n"
        "If you did not sign up for this account, you can ignore this message."
    )

    msg = EmailMessage()
    msg["Subject"] = subject
    from_addr = settings.MAIL_FROM or settings.MAIL_USERNAME or ""
    msg["From"] = f"{settings.MAIL_FROM_NAME} <{from_addr}>"
    msg["To"] = to_email
    msg.set_content(body)

    if not (settings.MAIL_USERNAME and settings.MAIL_PASSWORD):
        return

    with smtplib.SMTP(settings.MAIL_HOST, settings.MAIL_PORT) as server:
        use_tls = settings.MAIL_USE_TLS or (str(settings.MAIL_ENCRYPTION or "").lower() in {"tls", "starttls"})
        if use_tls:
            server.starttls()
        server.login(settings.MAIL_USERNAME, settings.MAIL_PASSWORD)
        server.send_message(msg)


def send_password_reset_email(to_email: str, token: str) -> None:
    settings = get_settings()
    subject = "Reset your password - American Board System"
    base = settings.BASE_URL.rstrip("/")
    reset_url = f"{base}/reset-password?token={token}"
    body = (
        f"We received a request to reset your password for {settings.APP_NAME}.\n\n"
        f"Use the following link to reset your password (valid for {settings.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES} minutes):\n"
        f"{reset_url}\n\n"
        "If you did not request a password reset, you can ignore this message."
    )

    msg = EmailMessage()
    msg["Subject"] = subject
    from_addr = settings.MAIL_FROM or settings.MAIL_USERNAME or ""
    msg["From"] = f"{settings.MAIL_FROM_NAME} <{from_addr}>"
    msg["To"] = to_email
    msg.set_content(body)

    if not (settings.MAIL_USERNAME and settings.MAIL_PASSWORD):
        return

    with smtplib.SMTP(settings.MAIL_HOST, settings.MAIL_PORT) as server:
        use_tls = settings.MAIL_USE_TLS or (str(settings.MAIL_ENCRYPTION or "").lower() in {"tls", "starttls"})
        if use_tls:
            server.starttls()
        server.login(settings.MAIL_USERNAME, settings.MAIL_PASSWORD)
        server.send_message(msg)
