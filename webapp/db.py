"""Users, grants, access requests, settings and the audit log.

Two backends with one interface: Firestore on Cloud Run, and an in-memory store (optionally saved to a JSON file)
for tests and local development. Emails are keys, always lower case.
"""

import base64
import json
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

ACTIVITY_DAYS = 180  # activity entries expire after this (Firestore TTL on expire_at; trimmed in memory)
ACTIVITY_MEMORY_MAX = 5000


def norm_email(email: str) -> str:
    return (email or "").strip().lower()


class MemoryDB:
    """Everything in dicts; with a path, saved as JSON after each write (bytes as base64)."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self.lock = threading.Lock()
        self.data = {"users": {}, "settings": {}, "games": {}, "requests": {}, "audit": [], "activity": []}
        if self.path and self.path.exists():
            self.data = json.loads(self.path.read_text(), object_hook=_decode)
            self.data.setdefault("activity", [])  # files saved before the activity log

    def _save(self) -> None:
        if self.path:
            self.path.write_text(json.dumps(self.data, default=_encode, indent=1))

    def get_user(self, email: str) -> dict | None:
        u = self.data["users"].get(norm_email(email))
        return dict(u) if u else None

    def put_user(self, email: str, user: dict) -> None:
        with self.lock:
            self.data["users"][norm_email(email)] = dict(user, email=norm_email(email))
            self._save()

    def list_users(self) -> list:
        return sorted((dict(u) for u in self.data["users"].values()), key=lambda u: u["email"])

    def any_admin(self) -> bool:
        return any(
            (u.get("admin") or u.get("role") == "admin") and u.get("active", True) for u in self.data["users"].values()
        )

    def get_setting(self, key: str) -> dict:
        return dict(self.data["settings"].get(key, {}))

    def put_setting(self, key: str, value: dict) -> None:
        with self.lock:
            self.data["settings"][key] = dict(value)
            self._save()

    def get_game(self, game_id: str) -> dict:
        return dict(self.data["games"].get(game_id, {}))

    def put_game(self, game_id: str, value: dict) -> None:
        with self.lock:
            self.data["games"][game_id] = dict(value)
            self._save()

    def add_request(self, req: dict) -> str:
        rid = uuid.uuid4().hex
        with self.lock:
            self.data["requests"][rid] = dict(req, id=rid)
            self._save()
        return rid

    def get_request(self, rid: str) -> dict | None:
        r = self.data["requests"].get(rid)
        return dict(r) if r else None

    def list_requests(self, email: str | None = None, status: str | None = None) -> list:
        out = [
            dict(r)
            for r in self.data["requests"].values()
            if (email is None or r["email"] == norm_email(email)) and (status is None or r["status"] == status)
        ]
        return sorted(out, key=lambda r: r["created"])

    def update_request(self, rid: str, fields: dict) -> None:
        with self.lock:
            self.data["requests"][rid].update(fields)
            self._save()

    def audit(self, actor: str, action: str, detail: dict) -> None:
        with self.lock:
            self.data["audit"].append(dict(at=time.time(), actor=actor, action=action, detail=detail))
            self._save()

    def list_audit(self, limit: int = 100) -> list:
        return list(reversed(self.data["audit"][-limit:]))

    def log_activity(self, entry: dict) -> None:
        with self.lock:
            log = self.data["activity"]
            log.append(dict(entry, at=time.time()))
            cutoff = time.time() - ACTIVITY_DAYS * 86400
            while log and (log[0]["at"] < cutoff or len(log) > ACTIVITY_MEMORY_MAX):
                log.pop(0)
            self._save()

    def list_activity(self, since: float, limit: int = 2000) -> list:
        """Newest first, at or after since (epoch seconds)."""
        return [dict(e) for e in reversed(self.data["activity"]) if e["at"] >= since][:limit]


def _encode(o):
    if isinstance(o, bytes):
        return {"__b64__": base64.b64encode(o).decode()}
    raise TypeError(type(o))


def _decode(d):
    return base64.b64decode(d["__b64__"]) if set(d) == {"__b64__"} else d


class FirestoreDB:
    """The same interface on Firestore (native mode). Collections: users, settings, games, access_requests, audit."""

    def __init__(self, project: str | None = None):
        from google.cloud import firestore

        self.fs = firestore
        self.db = firestore.Client(project=project)

    def get_user(self, email: str) -> dict | None:
        doc = self.db.collection("users").document(norm_email(email)).get()
        return doc.to_dict() if doc.exists else None

    def put_user(self, email: str, user: dict) -> None:
        self.db.collection("users").document(norm_email(email)).set(dict(user, email=norm_email(email)))

    def list_users(self) -> list:
        return sorted((d.to_dict() for d in self.db.collection("users").stream()), key=lambda u: u["email"])

    def any_admin(self) -> bool:
        # app.save_user keeps role == "admin" in step with the admin flag, so one indexed filter finds them all
        q = self.db.collection("users").where(filter=self.fs.FieldFilter("role", "==", "admin"))
        return any(d.to_dict().get("active", True) for d in q.stream())

    def get_setting(self, key: str) -> dict:
        doc = self.db.collection("settings").document(key).get()
        return doc.to_dict() if doc.exists else {}

    def put_setting(self, key: str, value: dict) -> None:
        self.db.collection("settings").document(key).set(value)

    def get_game(self, game_id: str) -> dict:
        doc = self.db.collection("games").document(game_id).get()
        return doc.to_dict() if doc.exists else {}

    def put_game(self, game_id: str, value: dict) -> None:
        self.db.collection("games").document(game_id).set(value)

    def add_request(self, req: dict) -> str:
        ref = self.db.collection("access_requests").document()
        ref.set(dict(req, id=ref.id))
        return ref.id

    def get_request(self, rid: str) -> dict | None:
        doc = self.db.collection("access_requests").document(rid).get()
        return doc.to_dict() if doc.exists else None

    def list_requests(self, email: str | None = None, status: str | None = None) -> list:
        # single-field equality filters only (automatic indexes); the rest is filtered here
        q = self.db.collection("access_requests")
        if email is not None:
            q = q.where(filter=self.fs.FieldFilter("email", "==", norm_email(email)))
        elif status is not None:
            q = q.where(filter=self.fs.FieldFilter("status", "==", status))
        out = [d.to_dict() for d in q.stream()]
        if status is not None:
            out = [r for r in out if r["status"] == status]
        return sorted(out, key=lambda r: r["created"])

    def update_request(self, rid: str, fields: dict) -> None:
        self.db.collection("access_requests").document(rid).update(fields)

    def audit(self, actor: str, action: str, detail: dict) -> None:
        self.db.collection("audit").add(dict(at=time.time(), actor=actor, action=action, detail=detail))

    def list_audit(self, limit: int = 100) -> list:
        q = self.db.collection("audit").order_by("at", direction=self.fs.Query.DESCENDING).limit(limit)
        return [d.to_dict() for d in q.stream()]

    def log_activity(self, entry: dict) -> None:
        # expire_at drives the collection's TTL policy (webapp/README.md); a single-field query needs no index
        expire = datetime.now(UTC) + timedelta(days=ACTIVITY_DAYS)
        self.db.collection("activity").add(dict(entry, at=time.time(), expire_at=expire))

    def list_activity(self, since: float, limit: int = 2000) -> list:
        q = (self.db.collection("activity").where(filter=self.fs.FieldFilter("at", ">=", since))
             .order_by("at", direction=self.fs.Query.DESCENDING).limit(limit))  # fmt: skip
        return [{k: v for k, v in d.to_dict().items() if k != "expire_at"} for d in q.stream()]
