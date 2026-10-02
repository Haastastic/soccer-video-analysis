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
from PIL import Image

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
    # clips (site_clips.py) live outside the releases; each game's clips.json names them
    clips = {"41": [["Longest on camera", [[100.0, "20 s", "41_100000.mp4"]]]],
             "42": [["On the ball", [[200.0, "touch", "42_200000.mp4"]]]]}  # fmt: skip
    (rel / "jv" / "games" / GAMES[0] / "clips.json").write_text(json.dumps(clips))
    for name in ["41_100000.mp4", "42_200000.mp4", "43_300000.mp4"]:  # 43's is in the bucket but not named
        f = root / "clips" / "jv" / GAMES[0] / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(name.encode() * 100)
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
    assert site.client.get(f"{VA}/players/44").status_code == 200  # one game: that game's player page
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
    r = site.post("/request", {"teams": "varsity", "role": "parent", "players": "Elm", "note": "hi"})
    assert r.status_code == 302
    reqs = site.db.list_requests(status="pending")
    assert len(reqs) == 1 and reqs[0]["email"] == "parent@example.com" and reqs[0]["team"] == "varsity"
    assert len(site.mail) == 1
    msg = site.mail[0]
    assert msg["To"] == ADMIN and "parent@example.com" in msg.get_content()
    assert not any(n in msg.get_content() for t in TEAMS.values() for n in t.values())
    # one pending request at a time
    site.post("/request", {"teams": "jv", "role": "parent", "players": "Birch"})
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
    site.post("/request", {"teams": "varsity", "role": "parent", "players": "Fir"})
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
        site.post("/request", {"teams": "jv", "role": "coach", "note": str(i)})
        for r in site.db.list_requests(status="pending"):
            site.db.update_request(r["id"], {"status": "denied"})
    assert len(site.db.list_requests(email="someone@example.com")) == 3  # 3 per day
    assert site.db.get_user("someone@example.com") is None
    assert "not approved" in text(site.client.get("/request"))


def test_request_needs_a_published_team(site):
    site.login("someone@example.com")
    site.post("/request", {"teams": "nope", "role": "coach"})
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
        site.post("/request", {"teams": "jv", "role": "parent", "players": "Alder"})
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


def test_one_game_team_menu_matches_the_page(site):
    """Varsity has one game: its season and player pages are that game's, with the menu on Season / Players."""
    site.login(ADMIN)
    on = re.compile(r'<a href="[^"]*" class=on>([A-Za-z]+)')
    season = site.client.get(f"{VA}/")
    assert season.status_code == 200 and on.findall(text(season))[-1] == "Season"
    assert TEAMS["varsity"][51] in text(season)
    player = site.client.get(f"{VA}/players/44")
    assert player.status_code == 200 and on.findall(text(player))[-1] == "Players"
    assert on.findall(text(site.client.get(f"{VA}/games/{GAMES[0]}")))[-1] == "Games"


def test_request_for_several_teams_approved_per_team(site):
    site.login("both@example.com")
    site.post("/request", {"teams": ["jv", "varsity"], "role": "parent", "players": "Alder (JV), Elm (Varsity)"})
    r = site.db.list_requests(status="pending")[0]
    assert r["teams"] == ["jv", "varsity"]
    site.logout()
    site.login(ADMIN)
    page = text(site.client.get("/admin/requests"))
    assert "<legend>JV</legend>" in page and "<legend>Varsity</legend>" in page
    site.post(f"/admin/requests/{r['id']}", {"decision": "approve", "role_jv": "parent", "players_jv": ["41"],
                                             "role_varsity": "parent", "players_varsity": ["51"]})  # fmt: skip
    assert site.db.get_user("both@example.com")["teams"] == {
        "jv": {"role": "parent", "players": [41]}, "varsity": {"role": "parent", "players": [51]}}  # fmt: skip


def test_approve_only_some_requested_teams(site):
    site.login("one@example.com")
    site.post("/request", {"teams": ["jv", "varsity"], "role": "coach"})
    r = site.db.list_requests(status="pending")[0]
    site.logout()
    site.login(ADMIN)
    # nothing chosen for either team: refused, the request stays pending
    assert site.post(f"/admin/requests/{r['id']}", {"decision": "approve"}).status_code == 400
    assert site.db.get_request(r["id"])["status"] == "pending"
    site.post(f"/admin/requests/{r['id']}", {"decision": "approve", "role_varsity": "coach"})  # JV left at no access
    assert site.db.get_user("one@example.com")["teams"] == {"varsity": {"role": "coach", "players": []}}


def test_member_can_ask_for_another_team(site):
    site.db.put_user("parent@example.com", dict(role="parent", players=[41], active=True))  # JV parent
    site.login("parent@example.com")
    assert 'href="/request"' in text(site.client.get(f"{JV}/"))  # the header link
    page = text(site.client.get("/request"))
    assert "(you have access)" in page and "You can see this site already" in page
    site.post("/request", {"teams": ["varsity"], "role": "parent", "players": "Fir"})
    assert site.db.list_requests(email="parent@example.com")[0]["teams"] == ["varsity"]
    site.logout()
    site.login(ADMIN)
    assert 'href="/request"' not in text(site.client.get(f"{JV}/"))  # admins see everything already


def test_activity_records_sign_ins_and_pages_not_images(site):
    site.db.put_user("parent@example.com", dict(role="member", admin=False, active=True,
                                                teams={"jv": {"role": "parent", "players": [41]}}))  # fmt: skip
    site.login("parent@example.com")
    site.client.get(f"{JV}/players/41")
    site.client.get(f"{JV}/players/42")  # not theirs: refused, still recorded
    site.client.get(f"{JV}/photo/41")  # images are not recorded
    site.client.get(f"{JV}/logo/game/{GAMES[0]}")
    site.logout()
    site.client.get("/privacy")  # signed out: nothing recorded
    log = list(reversed(site.db.list_activity(0)))
    assert [(e["action"], e.get("path"), e.get("status")) for e in log] == [
        ("sign_in", None, None),
        ("view", f"{JV}/players/41", 200),
        ("view", f"{JV}/players/42", 403),
        ("sign_out", None, None),
    ]
    assert all(e["email"] == "parent@example.com" and e["access"] for e in log)


def test_activity_page_admin_only_and_readable(site):
    site.db.put_user("coach@example.com", dict(role="member", admin=False, active=True,
                                               teams={"jv": {"role": "coach", "players": []}}))  # fmt: skip
    site.login("stranger@example.com")  # no access: the sign-in says so
    site.logout()
    site.login("coach@example.com")
    site.client.get(f"{JV}/games/{GAMES[0]}/players/42")
    assert site.client.get("/admin/activity").status_code == 403
    site.logout()
    site.login(ADMIN)
    site.db.put_user("never@example.com", dict(role="member", admin=False, active=True,
                                               teams={"jv": {"role": "parent", "players": [43]}}))  # fmt: skip
    html = site.client.get("/admin/activity").get_data(as_text=True)
    assert "Signed in (no access yet)" in html and "stranger@example.com" in html
    assert f"JV: Game {GAMES[0]}: #42 Birch Test" in html  # paths read as team, game and player
    assert "never@example.com" in html and "not in this period" in html  # invited, never came
    assert "bootstrap_admin" in html  # admin changes are listed too
    one = site.client.get("/admin/activity?email=coach@example.com&days=7").get_data(as_text=True)
    events = one.split("<h2>Events")[1].split("<h2>Admin changes")[0]
    assert "#42 Birch Test" in events and "stranger@example.com" not in events
    assert "/admin/activity" in site.client.get("/admin").get_data(as_text=True)


def test_activity_log_failure_never_breaks_a_page(site, monkeypatch):
    site.login(ADMIN)

    def broken(entry):
        raise RuntimeError("database down")

    monkeypatch.setattr(site.db, "log_activity", broken)
    assert site.client.get(f"{JV}/players").status_code == 200


def test_season_game_list_names_the_opponent(site):
    site.login(ADMIN)
    site.post(f"/admin/games/jv/{GAMES[1]}", {"opponent": "Northfield JV"})
    html = site.client.get(f"{JV}/").get_data(as_text=True)
    assert f'<a href="{JV}/games/{GAMES[1]}">{GAMES[1]}</a> vs Northfield JV' in html
    assert f'<a href="{JV}/games/{GAMES[0]}">{GAMES[0]}</a></td>' in html  # no opponent set: the date alone
    logo = BytesIO()
    Image.new("RGB", (40, 40)).save(logo, "PNG")
    site.post(f"/admin/games/jv/{GAMES[0]}", {"opponent": "Lake Ridge", "logo": (BytesIO(logo.getvalue()), "l.png")},
              content_type="multipart/form-data")  # fmt: skip
    html = site.client.get(f"{JV}/").get_data(as_text=True)
    assert f'vs <img class="crest sm" src="{JV}/logo/game/{GAMES[0]}" alt="">Lake Ridge' in html
    assert "</a> vs Northfield JV" in html  # no logo: the name alone


def test_clips_only_for_players_the_viewer_may_see(site):
    clip41, clip42, clip43 = (f"{JV}/clip/{GAMES[0]}/{n}" for n in ["41_100000.mp4", "42_200000.mp4", "43_300000.mp4"])
    assert site.client.get(clip41).status_code == 302  # signed out: to the sign-in
    site.login("stranger@example.com")
    assert site.client.get(clip41).status_code == 403
    site.logout()
    site.db.put_user("parent@example.com", dict(role="member", admin=False, active=True,
                                                teams={"jv": {"role": "parent", "players": [41]}}))  # fmt: skip
    site.login("parent@example.com")
    r = site.client.get(clip41)
    assert r.status_code == 200 and r.mimetype == "video/mp4" and r.data.startswith(b"41_100000.mp4")
    r = site.client.get(clip41, headers={"Range": "bytes=0-9"})
    assert r.status_code == 206 and r.data == b"41_100000."
    assert site.client.get(clip42).status_code == 403
    page = text(site.client.get(f"{JV}/games/{GAMES[0]}/players/41"))
    assert clip41 in page and "42_200000" not in page and 'id="watch"' in page
    assert re.search(r'<script nonce="[^"]+">\s*const watch', page)  # the buttons' script runs under the CSP
    assert clip41 in text(site.client.get(f"{JV}/players/41"))  # the season page has the game's clips too
    assert 'id="watch"' not in text(site.client.get(f"{JV}/games/{GAMES[1]}/players/41"))  # no clips that game
    site.logout()
    site.login(ADMIN)
    assert site.client.get(clip42).status_code == 200
    assert site.client.get(clip43).status_code == 404  # in the bucket, but no release names it
    assert site.client.get(f"{JV}/clip/{GAMES[0]}/..%2Fx.mp4").status_code == 404
