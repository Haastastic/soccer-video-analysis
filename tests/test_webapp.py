"""The coaching site: who sees what, CSRF, access requests, the first admin. Synthetic players only."""

import json
import re
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "webapp"), str(Path(__file__).resolve().parents[1])]

from app import create_app, process_logo  # noqa: E402
from db import MemoryDB  # noqa: E402
from store import LocalStore  # noqa: E402

PLAYERS = {2: "Alder Test", 4: "Birch Test", 6: "Cedar Test", 11: "Dogwood Test"}  # 11: goalkeeper
ADMIN = "owner@example.com"
GAMES = ["2026-01-02", "2026-01-09"]


def metrics(rng) -> pd.DataFrame:
    rows = []
    for j, name in PLAYERS.items():
        r = dict(jersey=j, name=name, role="goalkeeper" if j == 11 else ["defender", "midfielder", "forward"][j % 3],
                 goalkeeper=j == 11, minutes=rng.uniform(6, 30), min_h1=5.0, min_h2=5.0, windows=3, m_per_min=80.0,
                 work_rel=rng.uniform(0.8, 1.2), work_k=3, rel_sd=0.15, pct_fast=2.0, pct_sprint=0.5, from_goal=40.0,
                 depth=1.0, abs_y=10.0, roam=20.0, ball_seen_min=5.0, near_ball_pct=20.0, ball_dist_med=15.0,
                 touches=6.0, touches_per_min=0.4, exp_touch=6.0, touch_rel=1.0, on_ball_pct=2.0)  # fmt: skip
        for ph in ["h1_early", "h1_late", "h2_early", "h2_late", "h1", "h2"]:
            r.update({f"{ph}_min": 3.0, f"{ph}_rel": 1.0, f"{ph}_k": 2})
        rows.append(r)
    m = pd.DataFrame(rows)
    return m.assign(conf="medium")


def write_release(root: Path) -> None:
    rng = np.random.default_rng(0)
    rel = root / "releases" / "r1"
    (rel / "season").mkdir(parents=True)
    meds = pd.DataFrame([dict(jersey=j, pct_fast=2.0, from_goal=40.0, depth=0.0, near_ball_pct=20.0, touch_rel=1.0,
                              m_per_min=80.0) for j in PLAYERS])  # fmt: skip
    games = []
    for k in GAMES:
        d = rel / "games" / k
        d.mkdir(parents=True)
        metrics(rng).to_csv(d / "metrics.csv", index=False)
        meds.to_csv(d / "meds.csv", index=False)
        (d / "tips.json").write_text(json.dumps({"2": [["Work rate (strength)", "Covers more ground.", "evidence"]]}))
        smp = pd.DataFrame(dict(jersey=np.repeat(list(PLAYERS), 50), from_goal=rng.uniform(0, 100, 200),
                                y_team=rng.uniform(-30, 30, 200)))  # fmt: skip
        smp.to_csv(d / "samples.csv.gz", index=False)
        games.append(dict(id=k, label=k, n_windows=3, length=100.0, width=64.0))
    metrics(rng).to_csv(rel / "season" / "metrics.csv", index=False)
    meds.to_csv(rel / "season" / "meds.csv", index=False)
    tagged = {"2": [dict(kind="Work rate (strength)", text="Covers more ground.", evidence=[["Pooled", "ev"]],
                         games=GAMES, pooled=True, status="every")]}  # fmt: skip
    (rel / "season" / "tagged.json").write_text(json.dumps(tagged))
    (rel / "manifest.json").write_text(json.dumps(dict(exported="now", games=games)))
    (rel / "photos").mkdir()
    (rel / "photos" / "player_02.jpg").write_bytes(b"\xff\xd8photo3")
    (rel / "photos" / "player_04.jpg").write_bytes(b"\xff\xd8photo5")
    (root / "current.json").write_text(json.dumps({"release": "releases/r1"}))


class Site:
    def __init__(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ADMIN_EMAIL", ADMIN)
        monkeypatch.setenv("DEV_LOGIN", "1")
        monkeypatch.delenv("K_SERVICE", raising=False)
        monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
        write_release(tmp_path / "bucket")
        self.db = MemoryDB()
        self.mail = []
        self.app = create_app(self.db, LocalStore(tmp_path / "bucket"), send_mail=self.mail.append,
                              config={"TESTING": True})  # fmt: skip
        self.client = self.app.test_client()

    def csrf(self) -> str:
        with self.client.session_transaction() as s:
            return s.get("csrf", "")

    def login(self, email: str) -> None:
        self.client.get("/login")
        r = self.client.post("/dev-login", data={"email": email, "csrf": self.csrf()})
        assert r.status_code == 302

    def post(self, path: str, data: dict | None = None, **kw):
        data = dict(data or {})
        data.setdefault("csrf", self.csrf())
        return self.client.post(path, data=data, **kw)


@pytest.fixture
def site(tmp_path, monkeypatch):
    return Site(tmp_path, monkeypatch)


def text(r) -> str:
    return r.get_data(as_text=True)


def test_first_admin_only_when_none_exists(site):
    site.login(ADMIN)
    assert site.db.get_user(ADMIN)["role"] == "admin"
    assert site.client.get("/admin").status_code == 200
    # demoting the only admin is refused
    r = site.post("/admin/users", {"email": ADMIN, "role": "coach", "active": "1"})
    assert r.status_code == 400 and site.db.get_user(ADMIN)["role"] == "admin"


def test_admin_email_not_promoted_when_an_admin_exists(site):
    site.db.put_user("other@example.com", dict(role="admin", players=[], active=True))
    site.login(ADMIN)
    assert site.db.get_user(ADMIN) is None
    assert site.client.get("/").status_code == 302  # to the request page


def test_signed_out_redirects_to_login(site):
    for path in ["/", "/games", "/players", "/players/2", "/photo/2", "/admin", "/logo/team"]:
        r = site.client.get(path)
        assert r.status_code == 302 and "/login" in r.headers["Location"], path


def test_uninvited_sees_only_request_page(site):
    site.login("stranger@example.com")
    for path in ["/", "/games", "/players", "/players/2", f"/games/{GAMES[0]}", "/admin"]:
        r = site.client.get(path)
        assert r.status_code == 302 and r.headers["Location"].endswith("/request"), path
    for path in ["/photo/2", "/logo/team", f"/logo/game/{GAMES[0]}"]:
        assert site.client.get(path).status_code == 403, path
    page = text(site.client.get("/request"))
    assert not any(n in page for n in PLAYERS.values())


def test_parent_sees_only_their_players(site):
    site.db.put_user("parent@example.com", dict(role="parent", players=[2], active=True))
    site.login("parent@example.com")
    for path in ["/", f"/games/{GAMES[0]}", "/players", "/games"]:
        page = text(site.client.get(path))
        assert PLAYERS[2] in page or path == "/games", path
        for j, n in PLAYERS.items():
            if j != 2:
                assert n not in page, (path, n)
                assert f"/photo/{j}" not in page and f"/players/{j}" not in page, (path, j)
    assert "Showing the 1 player" in text(site.client.get("/"))
    assert site.client.get("/players/2").status_code == 200
    assert site.client.get(f"/games/{GAMES[1]}/players/2").status_code == 200
    assert site.client.get("/players/4").status_code == 403
    assert site.client.get(f"/games/{GAMES[0]}/players/4").status_code == 403
    assert site.client.get("/photo/2").status_code == 200
    assert site.client.get("/photo/4").status_code == 403
    assert site.client.get("/admin").status_code == 403


def test_coach_sees_everyone_but_not_admin(site):
    site.db.put_user("coach@example.com", dict(role="coach", players=[], active=True))
    site.login("coach@example.com")
    page = text(site.client.get("/"))
    assert all(n in page for n in PLAYERS.values())
    assert site.client.get("/players/6").status_code == 200
    assert site.client.get("/photo/6").status_code == 404  # no photo chosen
    assert site.client.get("/admin").status_code == 403


def test_inactive_user_loses_access(site):
    site.db.put_user("coach@example.com", dict(role="coach", players=[], active=False))
    site.login("coach@example.com")
    assert site.client.get("/").headers["Location"].endswith("/request")


def test_csrf_required(site):
    site.login(ADMIN)
    r = site.client.post("/admin/users", data={"email": "x@example.com", "role": "coach", "active": "1"})
    assert r.status_code == 400 and site.db.get_user("x@example.com") is None
    r = site.client.post("/admin/users", data={"email": "x@example.com", "role": "coach", "csrf": "wrong"})
    assert r.status_code == 400


def test_access_request_flow(site):
    site.login(ADMIN)
    site.client.post("/logout", data={"csrf": site.csrf()})
    site.login("parent@example.com")
    r = site.post("/request", {"role": "parent", "players": "Alder", "note": "hi"})
    assert r.status_code == 302
    reqs = site.db.list_requests(status="pending")
    assert len(reqs) == 1 and reqs[0]["email"] == "parent@example.com"
    assert len(site.mail) == 1
    msg = site.mail[0]
    assert msg["To"] == ADMIN and "parent@example.com" in msg.get_content()
    assert not any(n in msg.get_content() for n in PLAYERS.values())
    # one pending request at a time
    site.post("/request", {"role": "parent", "players": "Birch"})
    assert len(site.db.list_requests(email="parent@example.com")) == 1
    # admin approves with a chosen player
    site.client.post("/logout", data={"csrf": site.csrf()})
    site.login(ADMIN)
    assert 'Admin<span class="badge">1</span>' in text(site.client.get("/admin"))
    rid = reqs[0]["id"]
    r = site.post(f"/admin/requests/{rid}", {"decision": "approve", "role": "parent", "players": ["2", "99"]})
    assert r.status_code == 302
    u = site.db.get_user("parent@example.com")
    assert u["role"] == "parent" and u["players"] == [2]  # 99 is not a player
    assert site.db.get_request(rid)["status"] == "approved"
    assert any(a["action"] == "request_approve" for a in site.db.list_audit())


def test_denied_request_gives_no_access_and_rate_limit(site):
    site.login(ADMIN)
    site.client.post("/logout", data={"csrf": site.csrf()})
    site.login("someone@example.com")
    for i in range(4):
        site.post("/request", {"role": "coach", "note": str(i)})
        for r in site.db.list_requests(status="pending"):
            site.db.update_request(r["id"], {"status": "denied"})
    assert len(site.db.list_requests(email="someone@example.com")) == 3  # 3 per day
    assert site.db.get_user("someone@example.com") is None
    assert "not approved" in text(site.client.get("/request"))


def test_no_mail_without_smtp(monkeypatch):
    import notify

    for k in ["NOTIFY_SMTP_HOST", "NOTIFY_SMTP_USER", "NOTIFY_SMTP_PASSWORD"]:
        monkeypatch.delenv(k, raising=False)
    assert notify.send_request_notice([ADMIN], "x@example.com", "https://x/admin/requests") is False


def test_headers(site):
    site.login(ADMIN)
    r = site.client.get("/")
    csp = r.headers["Content-Security-Policy"]
    nonce = re.search(r"'nonce-([^']+)'", csp).group(1)
    assert f'<script nonce="{nonce}">' in text(r)
    assert "<script>" not in text(r)  # every script carries the nonce
    assert r.headers["X-Robots-Tag"].startswith("noindex")
    assert "no-store" in r.headers["Cache-Control"]


def test_logo_upload_reencoded_and_game_opponent(site):
    from PIL import Image

    site.login(ADMIN)
    buf = BytesIO()
    Image.new("RGB", (800, 400), (10, 20, 30)).save(buf, "JPEG")
    r = site.post(f"/admin/games/{GAMES[0]}", {"opponent": "Rival Test", "logo": (BytesIO(buf.getvalue()), "l.jpg")},
                  content_type="multipart/form-data")  # fmt: skip
    assert r.status_code == 302
    logo = site.db.get_game(GAMES[0])["logo"]
    assert logo.startswith(b"\x89PNG") and max(Image.open(BytesIO(logo)).size) <= 256
    assert "vs Rival Test" in text(site.client.get(f"/games/{GAMES[0]}"))
    r = site.post("/admin/team", {"name": "School Test", "logo": (BytesIO(b"GIF89a..."), "x.gif")},
                  content_type="multipart/form-data")  # fmt: skip
    assert r.status_code == 400
    with pytest.raises(ValueError):
        process_logo(b"not an image")


def test_dev_login_refused_on_cloud_run(tmp_path, monkeypatch):
    monkeypatch.setenv("DEV_LOGIN", "1")
    monkeypatch.setenv("K_SERVICE", "coaching")
    with pytest.raises(SystemExit):
        create_app(MemoryDB(), LocalStore(tmp_path))


def test_safe_next():
    from app import safe_next

    assert safe_next("/players/2") == "/players/2"
    assert safe_next("//evil.example") == "/"
    assert safe_next("https://evil.example") == "/"
    assert safe_next(None) == "/"


def test_privacy_public_and_health(site):
    r = site.client.get("/privacy")
    assert r.status_code == 200 and "Removal" in r.get_data(as_text=True)
    assert not any(n in r.get_data(as_text=True) for n in PLAYERS.values())
    assert site.client.get("/health").status_code == 200


def test_requester_emailed_on_decision(site):
    site.login(ADMIN)
    site.client.post("/logout", data={"csrf": site.csrf()})
    for who in ["yes@example.com", "no@example.com"]:
        site.login(who)
        site.post("/request", {"role": "parent", "players": "Alder"})
        site.client.post("/logout", data={"csrf": site.csrf()})
    site.login(ADMIN)
    site.mail.clear()
    for r in site.db.list_requests(status="pending"):
        decision = "approve" if r["email"] == "yes@example.com" else "deny"
        site.post(f"/admin/requests/{r['id']}", {"decision": decision, "role": "parent", "players": ["2"]})
    sent = {m["To"]: m for m in site.mail}
    assert set(sent) == {"yes@example.com", "no@example.com"}
    assert "approved" in sent["yes@example.com"]["Subject"] and "/login" in sent["yes@example.com"].get_content()
    assert "not approved" in sent["no@example.com"]["Subject"]
    for m in sent.values():  # no player names or numbers in either email
        assert not any(n in m.get_content() for n in PLAYERS.values())
