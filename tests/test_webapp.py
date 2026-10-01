"""The coaching site: who sees what (per team), CSRF, access requests, the first admin. Synthetic players only, with
numbers no real player wears (over 40)."""

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

TEAMS = {  # 44 is on both teams: players are keyed by team and number
    "jv": {41: "Alder Test", 42: "Birch Test", 43: "Cedar Test", 44: "Dogwood Test"},  # 44: goalkeeper
    "varsity": {51: "Elm Test", 52: "Fir Test", 44: "Gum Test"},
}
PLAYERS = TEAMS["jv"]
ADMIN = "owner@example.com"
GAMES = ["2026-01-02", "2026-01-09"]
JV, VA = "/t/jv", "/t/varsity"


def metrics(rng, players: dict) -> pd.DataFrame:
    rows = []
    for j, name in players.items():
        r = dict(jersey=j, name=name, role="goalkeeper" if j == 44 else ["defender", "midfielder", "forward"][j % 3],
                 goalkeeper=j == 44, minutes=rng.uniform(6, 30), min_h1=5.0, min_h2=5.0, windows=3, m_per_min=80.0,
                 work_rel=rng.uniform(0.8, 1.2), work_k=3, rel_sd=0.15, pct_fast=2.0, pct_sprint=0.5, from_goal=40.0,
                 depth=1.0, abs_y=10.0, roam=20.0, ball_seen_min=5.0, near_ball_pct=20.0, ball_dist_med=15.0,
                 touches=6.0, touches_per_min=0.4, exp_touch=6.0, touch_rel=1.0, on_ball_pct=2.0,
                 grade="Junior" if j == 41 else "")  # fmt: skip
        for ph in ["h1_early", "h1_late", "h2_early", "h2_late", "h1", "h2"]:
            r.update({f"{ph}_min": 3.0, f"{ph}_rel": 1.0, f"{ph}_k": 2})
        rows.append(r)
    return pd.DataFrame(rows).assign(conf="medium")


def write_team(rel: Path, players: dict, games: list, rng) -> None:
    (rel / "season").mkdir(parents=True)
    meds = pd.DataFrame([dict(jersey=j, pct_fast=2.0, from_goal=40.0, depth=0.0, near_ball_pct=20.0, touch_rel=1.0,
                              m_per_min=80.0) for j in players])  # fmt: skip
    first = str(next(iter(players)))
    manifest = []
    for k in games:
        d = rel / "games" / k
        d.mkdir(parents=True)
        metrics(rng, players).to_csv(d / "metrics.csv", index=False)
        meds.to_csv(d / "meds.csv", index=False)
        (d / "tips.json").write_text(json.dumps({first: [["Work rate (strength)", "Covers more ground.", "evidence"]]}))
        smp = pd.DataFrame(dict(jersey=np.repeat(list(players), 50), from_goal=rng.uniform(0, 100, 50 * len(players)),
                                y_team=rng.uniform(-30, 30, 50 * len(players))))  # fmt: skip
        smp.to_csv(d / "samples.csv.gz", index=False)
        manifest.append(dict(id=k, label=k, n_windows=3, length=100.0, width=64.0))
    metrics(rng, players).to_csv(rel / "season" / "metrics.csv", index=False)
    meds.to_csv(rel / "season" / "meds.csv", index=False)
    tagged = {first: [dict(kind="Work rate (strength)", text="Covers more ground.", evidence=[["Pooled", "ev"]],
                           games=games, pooled=True, status="every")]}  # fmt: skip
    (rel / "season" / "tagged.json").write_text(json.dumps(tagged))
    (rel / "manifest.json").write_text(json.dumps(dict(exported="now", games=manifest)))
    (rel / "photos").mkdir()


def write_release(root: Path) -> None:
    rng = np.random.default_rng(0)
    rel = root / "releases" / "r1"
    write_team(rel / "jv", TEAMS["jv"], GAMES, rng)
    write_team(rel / "varsity", TEAMS["varsity"], GAMES[:1], rng)  # one game: season redirects to it
    (rel / "jv" / "photos" / "player_41.jpg").write_bytes(b"\xff\xd8photo41")
    (rel / "jv" / "photos" / "player_42.jpg").write_bytes(b"\xff\xd8photo42")
    (rel / "varsity" / "photos" / "player_44.jpg").write_bytes(b"\xff\xd8varsity44")
    (rel / "teams.json").write_text(json.dumps({"teams": ["jv", "varsity"]}))
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

    def logout(self) -> None:
        self.client.post("/logout", data={"csrf": self.csrf()})

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
    assert site.db.get_user(ADMIN)["admin"] is True
    assert site.client.get("/admin").status_code == 200
    # demoting the only admin is refused
    r = site.post("/admin/users", {"email": ADMIN, "role_jv": "coach", "active": "1"})
    assert r.status_code == 400 and site.db.get_user(ADMIN)["admin"] is True


def test_admin_email_not_promoted_when_an_admin_exists(site):
    site.db.put_user("other@example.com", dict(role="admin", players=[], active=True))  # saved before teams
    site.login(ADMIN)
    assert site.db.get_user(ADMIN) is None
    assert site.client.get("/").headers["Location"].endswith("/request")


def test_signed_out_redirects_to_login(site):
    for path in ["/", f"{JV}/", f"{JV}/games", f"{JV}/players", f"{JV}/players/41", f"{JV}/photo/41", "/admin",
                 f"{JV}/logo"]:  # fmt: skip
        r = site.client.get(path)
        assert r.status_code == 302 and "/login" in r.headers["Location"], path


def test_uninvited_sees_only_request_page(site):
    site.login("stranger@example.com")
    for path in ["/", f"{JV}/", f"{JV}/games", f"{VA}/players", f"{JV}/players/41", f"{JV}/games/{GAMES[0]}", "/admin"]:
        r = site.client.get(path)
        assert r.status_code == 302 and r.headers["Location"].endswith("/request"), path
    for path in [f"{JV}/photo/41", f"{JV}/logo", f"{JV}/logo/game/{GAMES[0]}"]:
        assert site.client.get(path).status_code == 403, path
    page = text(site.client.get("/request"))
    assert not any(n in page for t in TEAMS.values() for n in t.values())
    assert "JV" in page and "Varsity" in page  # the team to ask about


def test_legacy_parent_sees_only_their_jv_players(site):
    """A user saved before teams existed ({role, players}) keeps that access to JV, and nothing else."""
    site.db.put_user("parent@example.com", dict(role="parent", players=[41], active=True))
    site.login("parent@example.com")
    assert site.client.get("/").headers["Location"].endswith(f"{JV}/")
    for path in [f"{JV}/", f"{JV}/games/{GAMES[0]}", f"{JV}/players", f"{JV}/games"]:
        page = text(site.client.get(path))
        assert PLAYERS[41] in page or path.endswith("/games"), path
        for j, n in PLAYERS.items():
            if j != 41:
                assert n not in page, (path, n)
                assert f"/photo/{j}" not in page and f"/players/{j}" not in page, (path, j)
        assert not any(n in page for n in TEAMS["varsity"].values())
        assert 'nav class="teams"' not in page  # one team: no switch
    assert "Showing the 1 player" in text(site.client.get(f"{JV}/"))
    assert site.client.get(f"{JV}/players/41").status_code == 200
    assert site.client.get(f"{JV}/games/{GAMES[1]}/players/41").status_code == 200
    assert site.client.get(f"{JV}/players/42").status_code == 403
    assert site.client.get(f"{JV}/photo/41").status_code == 200
    assert site.client.get(f"{JV}/photo/42").status_code == 403
    for path in [f"{VA}/", f"{VA}/players/44", f"{VA}/photo/44", "/admin"]:
        assert site.client.get(path).status_code == 403, path


def test_varsity_parent_same_number_other_team(site):
    """Access is per team: #44 on Varsity is a different player from #44 on JV."""
    site.db.put_user("vp@example.com", dict(role="member", admin=False, active=True,
                                            teams={"varsity": {"role": "parent", "players": [44]}}))  # fmt: skip
    site.login("vp@example.com")
    assert site.client.get("/").headers["Location"].endswith(f"{VA}/")
    assert site.client.get(f"{VA}/players/44").status_code in (200, 302)  # one game: to that game's page
    assert TEAMS["varsity"][44] in text(site.client.get(f"{VA}/games/{GAMES[0]}/players/44"))
    assert site.client.get(f"{VA}/photo/44").data == b"\xff\xd8varsity44"
    assert site.client.get(f"{VA}/players/51").status_code == 403
    assert site.client.get(f"{JV}/players/44").status_code == 403
    assert site.client.get(f"{JV}/").status_code == 403


def test_coach_of_both_teams_gets_switch(site):
    site.db.put_user("coach@example.com", dict(role="member", admin=False, active=True, teams={
        "jv": {"role": "coach", "players": []}, "varsity": {"role": "coach", "players": []}}))  # fmt: skip
    site.login("coach@example.com")
    page = text(site.client.get(f"{JV}/"))
    assert all(n in page for n in PLAYERS.values())
    assert 'nav class="teams"' in page and f'href="{VA}/"' in page
    assert site.client.get(f"{JV}/players/43").status_code == 200
    assert site.client.get(f"{JV}/photo/43").status_code == 404  # no photo chosen
    assert site.client.get("/admin").status_code == 403


def test_grade_shown_with_player(site):
    site.login(ADMIN)
    assert "Junior · " in text(site.client.get(f"{JV}/players/41"))
    assert "Junior · " in text(site.client.get(f"{JV}/players"))


def test_inactive_user_loses_access(site):
    site.db.put_user("coach@example.com", dict(role="coach", players=[], active=False))
    site.login("coach@example.com")
    assert site.client.get("/").headers["Location"].endswith("/request")


def test_csrf_required(site):
    site.login(ADMIN)
    r = site.client.post("/admin/users", data={"email": "x@example.com", "role_jv": "coach", "active": "1"})
    assert r.status_code == 400 and site.db.get_user("x@example.com") is None
    r = site.client.post("/admin/users", data={"email": "x@example.com", "role_jv": "coach", "csrf": "wrong"})
    assert r.status_code == 400


def test_admin_sets_per_team_access(site):
    site.login(ADMIN)
    r = site.post("/admin/users", {"email": "p@example.com", "role_jv": "parent", "players_jv": ["41", "99"],
                                   "role_varsity": "coach", "active": "1"})  # fmt: skip
    assert r.status_code == 302
    u = site.db.get_user("p@example.com")
    assert u["teams"] == {"jv": {"role": "parent", "players": [41]}, "varsity": {"role": "coach", "players": []}}
    assert u["admin"] is False


def test_access_request_flow(site):
    site.login(ADMIN)
    site.logout()
    site.login("parent@example.com")
    r = site.post("/request", {"team": "varsity", "role": "parent", "players": "Elm", "note": "hi"})
    assert r.status_code == 302
    reqs = site.db.list_requests(status="pending")
    assert len(reqs) == 1 and reqs[0]["email"] == "parent@example.com" and reqs[0]["team"] == "varsity"
    assert len(site.mail) == 1
    msg = site.mail[0]
    assert msg["To"] == ADMIN and "parent@example.com" in msg.get_content()
    assert not any(n in msg.get_content() for t in TEAMS.values() for n in t.values())
    # one pending request at a time
    site.post("/request", {"team": "jv", "role": "parent", "players": "Birch"})
    assert len(site.db.list_requests(email="parent@example.com")) == 1
    # admin approves with a chosen player of the team asked about
    site.logout()
    site.login(ADMIN)
    assert 'Admin<span class="badge">1</span>' in text(site.client.get("/admin"))
    rid = reqs[0]["id"]
    r = site.post(f"/admin/requests/{rid}", {"decision": "approve", "role_varsity": "parent",
                                             "players_varsity": ["51", "41"]})  # fmt: skip
    assert r.status_code == 302
    u = site.db.get_user("parent@example.com")
    assert u["teams"] == {"varsity": {"role": "parent", "players": [51]}}  # 41 is not a Varsity player
    assert site.db.get_request(rid)["status"] == "approved"
    assert any(a["action"] == "request_approve" for a in site.db.list_audit())


def test_approval_keeps_other_team_access(site):
    site.db.put_user("both@example.com", dict(role="parent", players=[41], active=True))  # JV, saved before teams
    site.login("both@example.com")
    site.post("/request", {"team": "varsity", "role": "parent", "players": "Fir"})
    site.logout()
    site.login(ADMIN)
    rid = site.db.list_requests(status="pending")[0]["id"]
    site.post(f"/admin/requests/{rid}", {"decision": "approve", "role_varsity": "parent", "players_varsity": ["52"]})
    assert site.db.get_user("both@example.com")["teams"] == {
        "jv": {"role": "parent", "players": [41]}, "varsity": {"role": "parent", "players": [52]}}  # fmt: skip


def test_denied_request_gives_no_access_and_rate_limit(site):
    site.login(ADMIN)
    site.logout()
    site.login("someone@example.com")
    for i in range(4):
        site.post("/request", {"team": "jv", "role": "coach", "note": str(i)})
        for r in site.db.list_requests(status="pending"):
            site.db.update_request(r["id"], {"status": "denied"})
    assert len(site.db.list_requests(email="someone@example.com")) == 3  # 3 per day
    assert site.db.get_user("someone@example.com") is None
    assert "not approved" in text(site.client.get("/request"))


def test_request_needs_a_published_team(site):
    site.login("someone@example.com")
    site.post("/request", {"team": "nope", "role": "coach"})
    assert site.db.list_requests(email="someone@example.com") == []


def test_no_mail_without_smtp(monkeypatch):
    import notify

    for k in ["NOTIFY_SMTP_HOST", "NOTIFY_SMTP_USER", "NOTIFY_SMTP_PASSWORD"]:
        monkeypatch.delenv(k, raising=False)
    assert notify.send_request_notice([ADMIN], "x@example.com", "https://x/admin/requests") is False


def test_headers(site):
    site.login(ADMIN)
    r = site.client.get(f"{JV}/")
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
    r = site.post(f"/admin/games/jv/{GAMES[0]}", {"opponent": "Rival Test",
                  "logo": (BytesIO(buf.getvalue()), "l.jpg")}, content_type="multipart/form-data")  # fmt: skip
    assert r.status_code == 302
    logo = site.db.get_game(f"jv_{GAMES[0]}")["logo"]
    assert logo.startswith(b"\x89PNG") and max(Image.open(BytesIO(logo)).size) <= 256
    assert "vs Rival Test" in text(site.client.get(f"{JV}/games/{GAMES[0]}"))
    assert "vs Rival Test" not in text(site.client.get(f"{VA}/games/{GAMES[0]}"))  # same date, other team's game
    r = site.post("/admin/team/varsity", {"name": "Varsity Test"}, content_type="multipart/form-data")
    assert r.status_code == 302 and "Varsity Test" in text(site.client.get(f"{VA}/games/{GAMES[0]}"))
    r = site.post("/admin/team/jv", {"name": "School Test", "logo": (BytesIO(b"GIF89a..."), "x.gif")},
                  content_type="multipart/form-data")  # fmt: skip
    assert r.status_code == 400
    with pytest.raises(ValueError):
        process_logo(b"not an image")


def test_legacy_team_settings_still_used(site):
    """The team name and a game's opponent saved before teams existed show on JV."""
    site.db.put_setting("team", {"name": "Old Name Test"})
    site.db.put_game(GAMES[0], {"opponent": "Old Rival Test"})
    site.login(ADMIN)
    page = text(site.client.get(f"{JV}/games/{GAMES[0]}"))
    assert "Old Name Test" in page and "vs Old Rival Test" in page


def test_dev_login_refused_on_cloud_run(tmp_path, monkeypatch):
    monkeypatch.setenv("DEV_LOGIN", "1")
    monkeypatch.setenv("K_SERVICE", "coaching")
    with pytest.raises(SystemExit):
        create_app(MemoryDB(), LocalStore(tmp_path))


def test_safe_next():
    from app import safe_next

    assert safe_next(f"{JV}/players/41") == f"{JV}/players/41"
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
    site.logout()
    for who in ["yes@example.com", "no@example.com"]:
        site.login(who)
        site.post("/request", {"team": "jv", "role": "parent", "players": "Alder"})
        site.logout()
    site.login(ADMIN)
    site.mail.clear()
    for r in site.db.list_requests(status="pending"):
        decision = "approve" if r["email"] == "yes@example.com" else "deny"
        site.post(f"/admin/requests/{r['id']}", {"decision": decision, "role_jv": "parent", "players_jv": ["41"]})
    sent = {m["To"]: m for m in site.mail}
    assert set(sent) == {"yes@example.com", "no@example.com"}
    assert "approved" in sent["yes@example.com"]["Subject"] and "/login" in sent["yes@example.com"].get_content()
    assert "not approved" in sent["no@example.com"]["Subject"]
    for m in sent.values():  # no player names or numbers in either email
        assert not any(n in m.get_content() for n in PLAYERS.values())


def test_icons_public(site):
    for name, mime in [
        ("favicon.ico", "image/x-icon"),
        ("icon-192.png", "image/png"),
        ("apple-touch-icon.png", "image/png"),
    ]:
        r = site.client.get(f"/{name}")
        assert r.status_code == 200 and r.headers["Content-Type"] == mime and len(r.data) > 100, name
    assert site.client.get("/site.webmanifest").json["icons"][0]["src"] == "/icon-192.png"
    assert site.client.get("/icon-999.png").status_code == 404  # only the listed icons are served


def test_team_switch_keeps_the_section(site):
    site.db.put_user("coach@example.com", dict(role="member", admin=False, active=True, teams={
        "jv": {"role": "coach", "players": []}, "varsity": {"role": "coach", "players": []}}))  # fmt: skip
    site.login("coach@example.com")
    assert f'href="{VA}/games"' in text(site.client.get(f"{JV}/games"))
    assert f'href="{VA}/games"' in text(site.client.get(f"{JV}/games/{GAMES[0]}"))  # a game page: the games list
    assert f'href="{VA}/players"' in text(site.client.get(f"{JV}/players"))
    assert f'href="{VA}/players"' in text(site.client.get(f"{JV}/players/41"))
    assert f'href="{VA}/"' in text(site.client.get(f"{JV}/"))
