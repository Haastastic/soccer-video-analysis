"""Email the admins when someone asks for access, and the requester when an admin approves or denies it. SMTP
settings come from the environment (Secret Manager on Cloud Run); without them only the in-app count of pending
requests shows. Messages carry an email address, the outcome and a link, never player names or stats."""

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


def decision_message(requester: str, approved: bool, link: str, sender: str) -> EmailMessage:
    """No player names or stats: the requester only learns the outcome and where to sign in."""
    msg = EmailMessage()
    msg["Subject"] = "Coaching site: access approved" if approved else "Coaching site: access request not approved"
    msg["From"] = sender
    msg["To"] = requester
    if approved:
        body = (
            f"Your request for access to the coaching site was approved.\n\nSign in with this Google account: {link}\n"
        )
    else:
        body = (
            "Your request for access to the coaching site was not approved.\n\n"
            "If you think this is a mistake, contact the team's site administrator.\n"
        )
    msg.set_content(body)
    return msg


def send_decision_notice(requester: str, approved: bool, link: str, send=None) -> bool:
    """True if an email went out. send: a replacement for SMTP delivery (tests)."""
    cfg = smtp_config()
    if send is None and cfg is None:
        return False
    return deliver(decision_message(requester, approved, link, (cfg or {}).get("sender", "coaching-site")), cfg, send)


def send_request_notice(admins: list, requester: str, link: str, send=None) -> bool:
    """True if an email went out. send: a replacement for SMTP delivery (tests)."""
    if not admins:
        return False
    cfg = smtp_config()
    if send is None and cfg is None:
        return False
    msg = request_message(admins, requester, link, (cfg or {}).get("sender", "coaching-site"))
    return deliver(msg, cfg, send)


def deliver(msg: EmailMessage, cfg: dict | None, send=None) -> bool:
    if send is not None:
        send(msg)
        return True
    with smtplib.SMTP_SSL(cfg["host"], cfg["port"], timeout=15) as s:
        s.login(cfg["user"], cfg["password"])
        s.send_message(msg)
    return True
