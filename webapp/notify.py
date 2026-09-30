"""Email the admins when someone asks for access. SMTP settings come from the environment (Secret Manager on Cloud
Run); without them only the in-app count of pending requests shows. The message carries the requester's email and a
link, never player names or stats."""

import os
import smtplib
from email.message import EmailMessage


def smtp_config() -> dict | None:
    host, user, password = (os.environ.get(k) for k in ("NOTIFY_SMTP_HOST", "NOTIFY_SMTP_USER", "NOTIFY_SMTP_PASSWORD"))
    if not (host and user and password):
        return None
    return dict(
        host=host,
        port=int(os.environ.get("NOTIFY_SMTP_PORT", "465")),
        user=user,
        password=password,
        sender=os.environ.get("NOTIFY_FROM", user),
    )


def request_message(admins: list, requester: str, link: str, sender: str) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = "Coaching site: new access request"
    msg["From"] = sender
    msg["To"] = ", ".join(admins)
    msg.set_content(
        f"{requester} asked for access to the coaching site.\n\nReview it here (sign-in required): {link}\n"
    )
    return msg


def send_request_notice(admins: list, requester: str, link: str, send=None) -> bool:
    """True if an email went out. send: a replacement for SMTP delivery (tests)."""
    if not admins:
        return False
    cfg = smtp_config()
    if send is None and cfg is None:
        return False
    msg = request_message(admins, requester, link, (cfg or {}).get("sender", "coaching-site"))
    if send is not None:
        send(msg)
        return True
    with smtplib.SMTP_SSL(cfg["host"], cfg["port"], timeout=15) as s:
        s.login(cfg["user"], cfg["password"])
        s.send_message(msg)
    return True
