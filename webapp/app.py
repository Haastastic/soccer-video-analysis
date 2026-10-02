"""The coaching site: the coaching pages for invited Google accounts, with roles and access requests.

Teams: each published team (JV, Varsity, ...) has its own pages under /t/<team>/. Access is per team: a coach sees
every player of the team, a parent the team's numbers with only the players an admin assigned named. Admins see
every team and manage people. Any Google account can sign in, but an account that is not an
active user sees only the access-request page. ADMIN_EMAIL becomes the first admin on first sign-in, while no admin
exists.

Environment:
  SECRET_KEY                session signing key (Secret Manager)
  GOOGLE_CLIENT_ID/_SECRET  OAuth client (web) for Google sign-in
  ADMIN_EMAIL               the first admin
  SITE_BUCKET               private bucket with the published release (publish_site.py), or SITE_LOCAL_DIR
  DB_BACKEND                firestore (default on Cloud Run) or memory; DB_FILE saves the memory backend to JSON
  NOTIFY_SMTP_*             optional: email admins about access requests (notify.py)
  DEV_LOGIN=1               local development only: sign in by typing an email (refused on Cloud Run)

Run locally: python webapp/app.py (see webapp/README.md).
"""

import os
import re
import secrets
import sys
import time
from datetime import datetime, timedelta
from functools import wraps
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):  # coaching_html.py sits beside this in the container, one level up in the repo
    if str(p) not in sys.path:
        sys.path.append(str(p))

import notify  # noqa: E402
import render  # noqa: E402
import ui  # noqa: E402
from db import FirestoreDB, MemoryDB, norm_email  # noqa: E402
from flask import Flask, abort, g, redirect, request, send_file, session, url_for  # noqa: E402
from store import GcsStore, LocalStore, ReleaseCache  # noqa: E402
from werkzeug.middleware.proxy_fix import ProxyFix  # noqa: E402

from coaching_html import esc  # noqa: E402

MAX_UPLOAD = 2 * 1024 * 1024
LOGO_PX = 256
REQUEST_TEXT_MAX, REQUEST_NOTE_MAX = 200, 500
REQUESTS_PER_DAY = 3
SESSION_HOURS = 12
TEAM_ROLES = ("coach", "parent")
LEGACY_TEAM = "jv"  # users, settings and games saved before there were teams belong to the first team
TEAM_LABELS = {"jv": "JV", "varsity": "Varsity"}  # until an admin sets a team's name
ACTIVITY_SPANS = (1, 7, 30, 90, 180)  # days the admin activity page can show (entries are kept 180 days)
ACTIVITY_LOAD = 3000  # newest entries read for that page
ACTIVITY_SHOWN = 300  # events listed
SITE_TZ = ZoneInfo(os.environ.get("SITE_TZ", "America/Chicago"))  # times on the activity page


def is_admin(u: dict | None) -> bool:
    return bool(u) and (bool(u.get("admin")) or u.get("role") == "admin")


def grants(u: dict | None) -> dict:
    """team -> {"role": coach|parent, "players": [...]}. A user saved before teams existed ({role, players}) has
    that access to the first team."""
    if not u:
        return {}
    if "teams" in u:
        return {t: gr for t, gr in u["teams"].items() if gr.get("role") in TEAM_ROLES}
    if u.get("role") in TEAM_ROLES:
        return {LEGACY_TEAM: {"role": u["role"], "players": list(u.get("players", []))}}
    return {}


def on_cloud_run() -> bool:
    return bool(os.environ.get("K_SERVICE"))


def default_db():
    backend = os.environ.get("DB_BACKEND", "firestore" if on_cloud_run() else "memory")
    if backend == "firestore":
        return FirestoreDB()
    return MemoryDB(os.environ.get("DB_FILE") or None)


def default_store():
    if os.environ.get("SITE_BUCKET"):
        return GcsStore(os.environ["SITE_BUCKET"])
    return LocalStore(Path(os.environ.get("SITE_LOCAL_DIR", HERE / "_local_bucket")))


def safe_next(target: str | None) -> str:
    """Only paths on this site (no scheme, no //host)."""
    if target and target.startswith("/") and not target.startswith("//") and "\\" not in target:
        return target
    return "/"


def process_logo(data: bytes) -> bytes:
    """PNG or JPEG only, re-encoded as a small PNG (drops metadata and anything else hidden in the file)."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 25_000_000
    try:
        img = Image.open(BytesIO(data))
        fmt = img.format
        img.load()
    except Exception as e:  # noqa: BLE001 - any decode failure is a bad upload
        raise ValueError("not an image") from e
    if fmt not in ("PNG", "JPEG"):
        raise ValueError("PNG or JPEG only")
    img = img.convert("RGBA")
    img.thumbnail((LOGO_PX, LOGO_PX))
    out = BytesIO()
    img.save(out, "PNG", optimize=True)
    return out.getvalue()


def create_app(db=None, store=None, send_mail=None, config: dict | None = None) -> Flask:
    app = Flask(__name__)
    dev_login = os.environ.get("DEV_LOGIN") == "1"
    if dev_login and on_cloud_run():
        raise SystemExit("DEV_LOGIN is for local development only")
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY"),
        MAX_CONTENT_LENGTH=MAX_UPLOAD,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=on_cloud_run(),
        SESSION_COOKIE_NAME="__Host-session" if on_cloud_run() else "session",
        PERMANENT_SESSION_LIFETIME=timedelta(hours=SESSION_HOURS),
        ADMIN_EMAIL=norm_email(os.environ.get("ADMIN_EMAIL", "")),
        DEV_LOGIN=dev_login,
    )
    app.config.update(config or {})
    if not app.config["SECRET_KEY"]:
        if on_cloud_run():
            raise SystemExit("SECRET_KEY is not set")
        app.config["SECRET_KEY"] = secrets.token_hex(32)  # local: sessions end when the server restarts
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
    db = db or default_db()
    releases = ReleaseCache(store or default_store())

    oauth = None
    if os.environ.get("GOOGLE_CLIENT_ID"):
        from authlib.integrations.flask_client import OAuth

        oauth = OAuth(app)
        oauth.register(
            "google",
            client_id=os.environ["GOOGLE_CLIENT_ID"],
            client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )

    # ------------------------------------------------------------------ request helpers

    @app.before_request
    def load_user():
        g.nonce = secrets.token_urlsafe(16)
        g.user = None
        email = session.get("email")
        if email:
            u = db.get_user(email)
            if u and u.get("active", True) and (is_admin(u) or grants(u)):
                g.user = u
        if request.method == "POST":
            sent = request.form.get("csrf", "")
            if not sent or not secrets.compare_digest(sent, session.get("csrf", "")):
                abort(400, "The form expired. Go back, reload the page and try again.")

    def record(action: str, **detail) -> None:
        """One line of the activity log for the signed-in person; never lets logging break a page."""
        email = session.get("email")
        if not email:
            return
        try:
            db.log_activity(dict(email=email, action=action, access=g.get("user") is not None, **detail))
        except Exception:  # noqa: BLE001 - the log is a convenience, the page is not
            app.logger.exception("activity log")

    @app.after_request
    def record_view(resp):
        # pages only (not photos, logos, icons or redirects), including pages refused or not found
        image = any(k in request.path for k in ("/photo/", "/logo", "/clip/"))  # also when missing (an HTML 404)
        if request.method == "GET" and resp.mimetype == "text/html" and not image and not 300 <= resp.status_code < 400:
            record("view", path=request.path, status=resp.status_code)
        return resp

    @app.after_request
    def headers(resp):
        nonce = getattr(g, "nonce", "")
        resp.headers["Content-Security-Policy"] = (
            f"default-src 'self'; script-src 'nonce-{nonce}'; style-src 'self' 'nonce-{nonce}'; "
            "style-src-attr 'unsafe-inline'; img-src 'self' data:; form-action 'self'; frame-ancestors 'none'; "
            "base-uri 'none'; object-src 'none'"
        )
        resp.headers["X-Robots-Tag"] = "noindex, nofollow"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "same-origin"
        resp.headers["X-Frame-Options"] = "DENY"
        if on_cloud_run():
            resp.headers["Strict-Transport-Security"] = "max-age=31536000"
        resp.headers.setdefault("Cache-Control", "private, no-store")
        return resp

    def csrf() -> str:
        if "csrf" not in session:
            session["csrf"] = secrets.token_urlsafe(24)
        return session["csrf"]

    def csrf_field() -> str:
        return f'<input type="hidden" name="csrf" value="{esc(csrf())}">'

    def team_settings(team: str) -> dict:
        s = db.get_setting(f"team_{team}")
        if not s and team == LEGACY_TEAM:  # set before there were teams
            s = db.get_setting("team")
        return s

    def team_label(team: str) -> str:
        return team_settings(team).get("name") or TEAM_LABELS.get(team, team.upper())

    def game_meta(team: str, gid: str) -> dict:
        m = db.get_game(f"{team}_{gid}")
        if not m and team == LEGACY_TEAM:  # set before there were teams
            m = db.get_game(gid)
        return m

    def site():
        return releases.get()

    def site_teams() -> list:
        s = site()
        return list(s.teams) if s else []

    def my_teams() -> list:
        """Published teams this viewer may open, in the site's order."""
        if g.user is None:
            return []
        if is_admin(g.user):
            return site_teams()
        mine = grants(g.user)
        return [t for t in site_teams() if t in mine]

    def pending_count() -> int:
        return len(db.list_requests(status="pending")) if g.user and is_admin(g.user) else 0

    def page(title: str, body: str, active: str = "", status: int = 200, team: str | None = None):
        # only members see a team's name and logo; everyone else sees a plain header
        t = team_settings(team) if (g.user and team) else {}
        name = team_label(team) if (g.user and team) else "Coaching"
        teams = [(k, team_label(k)) for k in my_teams()]
        head = ui.header(g.user, session.get("email"), t, name, team, teams, active, csrf(), pending_count(),
                         is_admin(g.user) if g.user else False)  # fmt: skip
        html_doc = ui.document(title, body + ui.NOTE, g.nonce, head, name)
        return html_doc, status, {"Content-Type": "text/html; charset=utf-8"}

    def shell_for(active: str, team: str):
        return lambda title, body: page(title, body, active, team=team)[0]

    def visible(team: str) -> set | None:
        """Jerseys of this team the viewer may see named: None = all (admin, the team's coaches)."""
        if is_admin(g.user):
            return None
        grant = grants(g.user).get(team, {})
        if grant.get("role") == "coach":
            return None
        return {int(j) for j in grant.get("players", [])}

    def can_view(team: str, jersey: int) -> bool:
        v = visible(team)
        return v is None or int(jersey) in v

    def photo_ok_for(team: str, rel):
        return lambda j: int(j) in rel.photos and can_view(team, j)

    def signed_in(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            if not session.get("email"):
                return redirect(url_for("login", next=request.full_path.rstrip("?")))
            return fn(*a, **kw)

        return wrapper

    def member(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            if not session.get("email"):
                return redirect(url_for("login", next=request.full_path.rstrip("?")))
            if g.user is None:
                if any(k in request.path for k in ("/photo/", "/logo", "/clip/")):
                    abort(403)
                return redirect(url_for("access_request"))
            return fn(*a, **kw)

        return wrapper

    def team_member(fn):
        """A page of one team: the team must be published and the viewer allowed in it."""

        @wraps(fn)
        @member
        def wrapper(team, *a, **kw):
            s = site()
            if s is None or team not in s.teams:
                abort(404)
            if team not in my_teams():
                abort(403)
            return fn(team, s.teams[team], *a, **kw)

        return wrapper

    def admin_only(fn):
        @wraps(fn)
        @member
        def wrapper(*a, **kw):
            if not is_admin(g.user):
                abort(403)
            return fn(*a, **kw)

        return wrapper

    def nothing_yet():
        return page("Nothing yet", "<h1>Nothing published yet</h1><p class='sub'>Check back after the next game.</p>")

    # ------------------------------------------------------------------ sign-in

    def start_session(email: str, name: str) -> None:
        session.clear()
        session.permanent = True
        session["email"], session["name"] = norm_email(email), name
        session["csrf"] = secrets.token_urlsafe(24)
        admin = app.config["ADMIN_EMAIL"]
        if admin and norm_email(email) == admin and not db.any_admin():
            db.put_user(admin, dict(role="admin", admin=True, teams={}, active=True, invited_by="bootstrap",
                                    created=time.time()))  # fmt: skip
            db.audit(admin, "bootstrap_admin", {})
        u = db.get_user(email)
        g.user = u if u and u.get("active", True) and (is_admin(u) or grants(u)) else None
        record("sign_in")

    @app.get("/login")
    def login():
        if session.get("email"):
            return redirect(safe_next(request.args.get("next")))
        session["next"] = safe_next(request.args.get("next"))
        dev = ""
        if app.config["DEV_LOGIN"]:
            dev = (
                f'<form method="post" action="/dev-login" class="stack" style="margin:24px auto 0">{csrf_field()}'
                '<label>Development sign-in (local only)<input type="email" name="email" required></label>'
                "<button>Sign in</button></form>"
            )
        google = '<a class="btn primary" href="/login/google">Sign in with Google</a>' if oauth else ""
        body = (
            '<div class="center"><h1>Coaching</h1>'
            "<p class='sub'>Private coaching pages for invited families and coaches.</p>"
            f'<p>{google}</p>{dev}<p class="sub"><a href="/privacy">Privacy</a></p></div>'
        )
        return page("Sign in", body)

    @app.get("/login/google")
    def login_google():
        if oauth is None:
            abort(404)
        return oauth.google.authorize_redirect(url_for("auth_callback", _external=True))

    @app.get("/auth/callback")
    def auth_callback():
        if oauth is None:
            abort(404)
        token = oauth.google.authorize_access_token()
        info = token.get("userinfo") or {}
        if not info.get("email") or not info.get("email_verified"):
            abort(403, "Your Google account's email is not verified.")
        nxt = session.get("next", "/")
        start_session(info["email"], info.get("name", ""))
        return redirect(safe_next(nxt))

    @app.post("/dev-login")
    def dev_login_post():
        if not app.config["DEV_LOGIN"]:
            abort(404)
        nxt = session.get("next", "/")
        start_session(request.form["email"], "")
        return redirect(safe_next(nxt))

    @app.post("/logout")
    def logout():
        record("sign_out")
        session.clear()
        return redirect(url_for("login"))

    ICONS = {  # the site's own icon (webapp/icons), public: browsers fetch these before anyone signs in
        "favicon.ico": "image/x-icon",
        "favicon-32.png": "image/png",
        "apple-touch-icon.png": "image/png",
        "icon-192.png": "image/png",
        "icon-512.png": "image/png",
    }

    @app.get("/<any(" + ", ".join(f'"{n}"' for n in ICONS) + "):name>")
    def icon(name):
        data = (HERE / "icons" / name).read_bytes()
        return data, 200, {"Content-Type": ICONS[name], "Cache-Control": "public, max-age=86400"}

    @app.get("/site.webmanifest")
    def manifest():
        icons = [{"src": f"/icon-{n}.png", "sizes": f"{n}x{n}", "type": "image/png"} for n in (192, 512)]
        body = {"name": "Coaching", "short_name": "Coaching", "icons": icons, "display": "browser"}
        return body, 200, {"Content-Type": "application/manifest+json", "Cache-Control": "public, max-age=86400"}

    @app.get("/privacy")
    def privacy():
        # public: Google's sign-in consent screen links here
        return page("Privacy", ui.privacy(os.environ.get("CONTACT_EMAIL", "")))

    @app.get("/health")  # Cloud Run reserves paths ending in "z" (/healthz never reaches the app)
    def healthz():
        return "ok", 200, {"Content-Type": "text/plain"}

    # ------------------------------------------------------------------ access requests

    @app.route("/request", methods=["GET", "POST"])
    @signed_in
    def access_request():
        email = session["email"]
        mine = db.list_requests(email=email)
        pending = [r for r in mine if r["status"] == "pending"]
        teams = site_teams()
        msg = ""
        if request.method == "POST":
            role = request.form.get("role", "")
            asked = [t for t in teams if t in request.form.getlist("teams")]
            text = request.form.get("players", "").strip()
            note = request.form.get("note", "").strip()
            recent = [r for r in mine if r["created"] > time.time() - 86400]
            if pending:
                msg = "You already have a request waiting."
            elif len(recent) >= REQUESTS_PER_DAY:
                msg = "Too many requests today. Try again tomorrow."
            elif role not in TEAM_ROLES:
                msg = "Choose coach or parent."
            elif not asked:
                msg = "Choose at least one team."
            elif len(text) > REQUEST_TEXT_MAX or len(note) > REQUEST_NOTE_MAX:
                msg = "That is too long."
            elif role == "parent" and not text:
                msg = "Say which player or players you are asking about."
            else:
                rid = db.add_request(
                    dict(email=email, name=session.get("name", ""), role=role, teams=asked, team=asked[0],
                         players_text=text, note=note, created=time.time(), status="pending", decided_by=None)
                )  # fmt: skip
                db.audit(email, "request_access", {"id": rid, "role": role, "teams": asked})
                admins = [u["email"] for u in db.list_users() if is_admin(u) and u.get("active", True)]
                try:
                    notify.send_request_notice(admins, email, url_for("admin_requests", _external=True), send_mail)
                except Exception as e:  # noqa: BLE001 - the request is stored; the email is a courtesy
                    app.logger.warning("access request email failed: %s", e)
                return redirect(url_for("access_request"))
        latest = mine[-1] if mine else None
        has = g.user is not None
        intro = (
            "<p class='sub'>You can see this site already. Ask here for access to more players or another team.</p>"
            if has
            else "<p class='sub'>This site is private. Ask the team's admin for access; they will be notified.</p>"
        )
        status = ""
        if pending:
            status = '<div class="flash">Your request is waiting for an admin.</div>'
        elif latest and latest["status"] == "denied":
            status = '<div class="flash">Your last request was not approved.</div>'
        form = ""
        if not pending:
            # team names only: they say which teams to ask about, nothing about any player
            mine_now = grants(g.user) if g.user else {}

            def now(t: str) -> str:
                if g.user and is_admin(g.user):
                    return " <span class='muted'>(you see everything)</span>"
                role_now = mine_now.get(t, {}).get("role")
                if role_now == "coach":
                    return " <span class='muted'>(you see every player)</span>"
                return " <span class='muted'>(you have access)</span>" if role_now == "parent" else ""

            boxes = "".join(
                f'<label><input type="checkbox" name="teams" value="{esc(t)}"> {esc(team_label(t))}{now(t)}</label>'
                for t in teams
            )
            form = f"""<form method="post" class="stack">{csrf_field()}
<label>Team(s)<div class="checks">{boxes}</div></label>
<label>I am a<select name="role"><option value="parent">Parent or guardian</option>
<option value="coach">Coach</option></select></label>
<label>Which player(s), and on which team? (names; parents only)<input type="text" name="players"
maxlength="{REQUEST_TEXT_MAX}"></label>
<label>Note for the admin (optional)<textarea name="note" rows="3" maxlength="{REQUEST_NOTE_MAX}"></textarea></label>
<button class="primary">Request access</button></form>"""
        flash = f'<div class="flash">{esc(msg)}</div>' if msg else ""
        body = f"<h1>Request access</h1>{intro}<p>Signed in as <b>{esc(email)}</b>.</p>{flash}{status}{form}"
        return page("Request access", body)

    # ------------------------------------------------------------------ pages

    @app.get("/")
    @member
    def home():
        teams = my_teams()
        if not teams:
            return nothing_yet()
        return redirect(url_for("season", team=teams[0]))

    @app.get("/t/<team>/")
    @team_member
    def season(team, rel):
        if len(rel.order) == 1:  # one game: its page is the season, shown here so the menu still says Season
            gid = rel.order[0]
            return render.game_page(f"/t/{team}", rel, gid, shell_for("season", team), visible(team),
                                    photo_ok_for(team, rel), game_meta(team, gid))  # fmt: skip
        meta = {k: game_meta(team, k) for k in rel.order}
        return render.season_page(f"/t/{team}", rel, shell_for("season", team), visible(team),
                                  photo_ok_for(team, rel), meta)  # fmt: skip

    @app.get("/t/<team>/games")
    @team_member
    def games(team, rel):
        meta = {k: game_meta(team, k) for k in rel.order}
        return page("Games", render.games_page(f"/t/{team}", rel, meta, visible(team)), "games", team=team)

    @app.get("/t/<team>/games/<gid>")
    @team_member
    def game(team, rel, gid):
        if gid not in rel.games:
            abort(404)
        return render.game_page(f"/t/{team}", rel, gid, shell_for("games", team), visible(team),
                                photo_ok_for(team, rel), game_meta(team, gid))  # fmt: skip

    @app.get("/t/<team>/games/<gid>/players/<int:jersey>")
    @team_member
    def game_player(team, rel, gid, jersey):
        if gid not in rel.games:
            abort(404)
        if not can_view(team, jersey):
            abort(403)
        out = render.game_player_page(f"/t/{team}", rel, gid, jersey, shell_for("games", team), photo_ok_for(team, rel))
        return out if out is not None else abort(404)

    @app.get("/t/<team>/players")
    @team_member
    def players(team, rel):
        body = render.players_page(f"/t/{team}", rel, visible(team), photo_ok_for(team, rel))
        return page("Players", body, "players", team=team)

    @app.get("/t/<team>/players/<int:jersey>")
    @team_member
    def player(team, rel, jersey):
        if not can_view(team, jersey):
            abort(403)
        if len(rel.order) == 1:  # one game: that game's player page, with the menu on Players
            out = render.game_player_page(f"/t/{team}", rel, rel.order[0], jersey, shell_for("players", team),
                                          photo_ok_for(team, rel))  # fmt: skip
            return out if out is not None else abort(404)
        out = render.season_player_page(f"/t/{team}", rel, jersey, shell_for("players", team), photo_ok_for(team, rel))
        return out if out is not None else abort(404)

    def image(data: bytes | None, mime: str):
        if not data:
            abort(404)
        return data, 200, {"Content-Type": mime, "Cache-Control": "private, max-age=300"}

    @app.get("/t/<team>/photo/<int:jersey>")
    @team_member
    def photo(team, rel, jersey):
        if not can_view(team, jersey):
            abort(403)
        return image(rel.photo(jersey), "image/jpeg")

    @app.get("/t/<team>/clip/<gid>/<name>")
    @team_member
    def clip(team, rel, gid, name):
        m = re.fullmatch(r"(\d{2})_\d+\.mp4", name)
        if not m or gid not in rel.games:
            abort(404)
        if not can_view(team, int(m.group(1))):
            abort(403)
        data = rel.clip(gid, name)
        if not data:
            abort(404)
        # conditional: answers Range requests (206), which iPhone Safari needs to play video
        resp = send_file(BytesIO(data), mimetype="video/mp4", conditional=True, max_age=300)
        resp.headers["Cache-Control"] = "private, max-age=300"
        return resp

    @app.get("/t/<team>/logo")
    @team_member
    def team_logo(team, rel):
        return image(team_settings(team).get("logo"), "image/png")

    @app.get("/t/<team>/logo/game/<gid>")
    @team_member
    def game_logo(team, rel, gid):
        return image(game_meta(team, gid).get("logo"), "image/png")

    # ------------------------------------------------------------------ admin

    def roster(team: str) -> list:
        s = site()
        if s is None or team not in s.teams:
            return []
        m = s.teams[team].season.sort_values("jersey")
        return [(int(r.jersey), render.display_name(r)) for _, r in m.iterrows()]

    def player_checks(team: str, chosen: list) -> str:
        boxes = "".join(
            f'<label><input type="checkbox" name="players_{esc(team)}" value="{j}"{" checked" if j in chosen else ""}>'
            f"{esc(n)} <span class='muted'>#{j}</span></label>"
            for j, n in roster(team)
        )
        return f'<div class="checks">{boxes or "<span class=muted>No players published yet.</span>"}</div>'

    def role_select(team: str, current: str) -> str:
        opts = "".join(
            f'<option value="{r}"{" selected" if r == current else ""}>{label}</option>'
            for r, label in (("", "no access"), ("coach", "coach"), ("parent", "parent"))
        )
        return f'<select name="role_{esc(team)}">{opts}</select>'

    def grant_fields(u: dict) -> str:
        """Per team: a role and, for parents, the players."""
        mine = grants(u)
        parts = []
        for t in site_teams():
            gr = mine.get(t, {})
            parts.append(
                f"<fieldset><legend>{esc(team_label(t))}</legend><label>Role{role_select(t, gr.get('role', ''))}"
                f"</label><label>Players (parents){player_checks(t, gr.get('players', []))}</label></fieldset>"
            )
        return "".join(parts)

    def parse_grants() -> dict:
        out = {}
        for t in site_teams():
            role = request.form.get(f"role_{t}", "")
            if role not in TEAM_ROLES:
                continue
            valid = {j for j, _ in roster(t)}
            ps = sorted({int(x) for x in request.form.getlist(f"players_{t}") if x.isdigit() and int(x) in valid})
            out[t] = dict(role=role, players=ps if role == "parent" else [])
        return out

    def sees(u: dict) -> str:
        if is_admin(u):
            return "everything (admin)"
        parts = []
        for t, gr in grants(u).items():
            if gr.get("role") == "coach":
                parts.append(f"{team_label(t)}: all players")
            else:
                names = dict(roster(t))
                who = ", ".join(names.get(j, f"#{j}") for j in gr.get("players", [])) or "team numbers only"
                parts.append(f"{team_label(t)}: {who}")
        return "; ".join(parts) or "–"

    def requested_teams(r: dict) -> list:
        """The teams a request asks about (requests made before several teams could be asked name one)."""
        return list(r.get("teams") or [r.get("team") or LEGACY_TEAM])

    def other_active_admins(email: str) -> int:
        return sum(1 for u in db.list_users() if is_admin(u) and u.get("active", True) and u["email"] != email)

    @app.get("/admin")
    @admin_only
    def admin():
        users = []
        for u in db.list_users():
            users.append(
                f"<tr><td>{esc(u['email'])}</td><td>{esc(sees(u))}</td>"
                f"<td>{'active' if u.get('active', True) else '<span class=muted>off</span>'}</td>"
                f'<td><details><summary>Edit</summary><form method="post" action="/admin/users" class="stack">'
                f'{csrf_field()}<input type="hidden" name="email" value="{esc(u["email"])}">'
                f'<label><span><input type="checkbox" name="admin" value="1"{" checked" if is_admin(u) else ""}> '
                "Admin (everything, and this page)</span></label>"
                f"{grant_fields(u)}"
                f'<label><span><input type="checkbox" name="active" value="1"'
                f"{' checked' if u.get('active', True) else ''}> Active</span></label>"
                "<button>Save</button></form></details></td></tr>"
            )
        team_forms = []
        for t in site_teams():
            ts = team_settings(t)
            rel = site().teams[t]
            game_rows = []
            for gid in rel.order:
                meta = game_meta(t, gid)
                logo = f'<img class="crest" src="/t/{esc(t)}/logo/game/{esc(gid)}" alt="">' if meta.get("logo") else ""
                game_rows.append(
                    f"<tr><td>{esc(gid)}</td><td>{logo}</td><td>"
                    f'<form method="post" action="/admin/games/{esc(t)}/{esc(gid)}" enctype="multipart/form-data" '
                    f'class="stack">{csrf_field()}<label>Opponent<input type="text" name="opponent" maxlength="80" '
                    f'value="{esc(meta.get("opponent", ""))}"></label>'
                    '<label>Opponent logo (PNG or JPEG)<input type="file" name="logo" accept="image/png,image/jpeg">'
                    "</label><button>Save</button></form></td></tr>"
                )
            crest = f'<img class="crest" src="/t/{esc(t)}/logo" alt="">' if ts.get("logo") else ""
            team_forms.append(
                f"<h2>{esc(team_label(t))}</h2>"
                f'<form method="post" action="/admin/team/{esc(t)}" enctype="multipart/form-data" class="stack">'
                f'{csrf_field()}<label>Team name, as shown on the site<input type="text" name="name" maxlength="80" '
                f'value="{esc(ts.get("name", ""))}"></label>'
                f"<label>Logo (PNG or JPEG, up to 2 MB){crest}"
                '<input type="file" name="logo" accept="image/png,image/jpeg"></label><button>Save</button></form>'
                f'<h3>Games</h3><div class="wrap"><table class="admin"><tr><th>Game</th><th>Logo</th><th>Opponent</th>'
                f"</tr>{''.join(game_rows)}</table></div>"
            )
        n_pending = pending_count()
        body = f"""<h1>Admin</h1>
<p><a class="btn{" primary" if n_pending else ""}" href="/admin/requests">Access requests ({n_pending} waiting)</a>
<a class="btn" href="/admin/activity">Activity</a></p>
<h2>People</h2>
<p class="sub">Per team: coaches see every player; parents see team numbers and only the players ticked for them.
Admins see every team.</p>
<div class="wrap"><table class="admin"><tr><th>Email</th><th>Sees</th><th>Status</th><th></th></tr>
{"".join(users)}</table></div>
<h2>Add a person</h2>
<form method="post" action="/admin/users" class="stack">{csrf_field()}
<label>Google account email<input type="email" name="email" required></label>
{grant_fields({})}
<input type="hidden" name="active" value="1"><button class="primary">Add</button></form>
{"".join(team_forms)}"""
        return page("Admin", body, "admin")

    def save_user(actor: str, email: str, admin_flag: bool, team_grants: dict, active: bool, via: str) -> str | None:
        """Error text, or None when saved. team_grants replaces the person's grants for the teams it names."""
        email = norm_email(email)
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            return "That is not an email address."
        old = db.get_user(email) or {}
        if is_admin(old) and (not admin_flag or not active) and other_active_admins(email) == 0:
            return "Keep at least one active admin."
        user = {k: v for k, v in old.items() if k not in ("players",)}
        user.update(admin=admin_flag, role="admin" if admin_flag else "member", teams=team_grants, active=active)
        user.setdefault("invited_by", actor)
        user.setdefault("created", time.time())
        db.put_user(email, user)
        db.audit(actor, "save_user", {"email": email, "admin": admin_flag, "teams": team_grants, "active": active,
                                      "via": via})  # fmt: skip
        return None

    @app.post("/admin/users")
    @admin_only
    def admin_users():
        err = save_user(
            g.user["email"], request.form.get("email", ""), request.form.get("admin") == "1", parse_grants(),
            request.form.get("active") == "1", "admin",
        )  # fmt: skip
        if err:
            return page("Admin", f'<h1>Admin</h1><div class="flash">{esc(err)}</div><p><a href="/admin">Back</a></p>',
                        "admin", 400)  # fmt: skip
        return redirect(url_for("admin"))

    def bad_logo(e: Exception):
        return page("Admin", f'<div class="flash">Logo: {esc(e)}</div><p><a href="/admin">Back</a></p>', "admin", 400)

    def uploaded_logo() -> bytes | None:
        f = request.files.get("logo")
        if not f or not f.filename:
            return None
        return process_logo(f.read())

    @app.post("/admin/team/<team>")
    @admin_only
    def admin_team(team):
        if team not in site_teams():
            abort(404)
        t = team_settings(team)
        t["name"] = request.form.get("name", "").strip()[:80]
        try:
            logo = uploaded_logo()
        except ValueError as e:
            return bad_logo(e)
        if logo:
            t["logo"] = logo
        db.put_setting(f"team_{team}", t)
        db.audit(g.user["email"], "save_team", {"team": team, "name": t["name"], "logo": bool(logo)})
        return redirect(url_for("admin"))

    @app.post("/admin/games/<team>/<gid>")
    @admin_only
    def admin_game(team, gid):
        s = site()
        if s is None or team not in s.teams or gid not in s.teams[team].games:
            abort(404)
        meta = game_meta(team, gid)
        meta["opponent"] = request.form.get("opponent", "").strip()[:80]
        try:
            logo = uploaded_logo()
        except ValueError as e:
            return bad_logo(e)
        if logo:
            meta["logo"] = logo
        db.put_game(f"{team}_{gid}", meta)
        db.audit(g.user["email"], "save_game", {"team": team, "game": gid, "opponent": meta["opponent"],
                                                "logo": bool(logo)})  # fmt: skip
        return redirect(url_for("admin"))

    def local_time(ts: float, fmt: str = "%b %d, %H:%M") -> str:
        return datetime.fromtimestamp(ts, SITE_TZ).strftime(fmt)

    def describe(path: str) -> str:
        """A page path as an admin reads it: team, page, game and player names."""
        m = re.fullmatch(r"/t/([^/]+)/(.*)", path)
        if not m:
            fixed = {"/": "Home", "/request": "Access request form", "/privacy": "Privacy", "/admin": "Admin",
                     "/admin/requests": "Admin: access requests", "/admin/activity": "Admin: activity"}  # fmt: skip
            return fixed.get(path, path)
        team, rest = m.groups()
        names = dict(roster(team))

        def who(j: str) -> str:
            return f"#{j} {names.get(int(j), '')}".strip()

        def vs(gid: str) -> str:
            opp = game_meta(team, gid).get("opponent")
            return f"{gid} vs {opp}" if opp else gid

        parts = [x for x in rest.split("/") if x]
        if not parts:
            what = "Season"
        elif parts in (["games"], ["players"]):
            what = parts[0].capitalize()
        elif len(parts) == 2 and parts[0] == "players" and parts[1].isdigit():
            what = f"Player {who(parts[1])}"
        elif len(parts) == 2 and parts[0] == "games":
            what = f"Game {vs(parts[1])}"
        elif len(parts) == 4 and parts[0] == "games" and parts[2] == "players" and parts[3].isdigit():
            what = f"Game {vs(parts[1])}: {who(parts[3])}"
        else:
            what = "/".join(parts)
        return f"{team_label(team)}: {what}"

    def event_text(e: dict) -> str:
        if e["action"] == "sign_in":
            return "Signed in" + ("" if e.get("access") else " (no access yet)")
        if e["action"] == "sign_out":
            return "Signed out"
        text = esc(describe(e.get("path", "")))
        status = e.get("status", 200)
        if status == 403:
            return f"{text} <span class='muted'>(refused)</span>"
        if status >= 400:
            return f"{text} <span class='muted'>({status})</span>"
        return text

    @app.get("/admin/activity")
    @admin_only
    def admin_activity():
        days = request.args.get("days", 30, type=int)
        days = days if days in ACTIVITY_SPANS else 30
        only = norm_email(request.args.get("email", ""))
        events = db.list_activity(time.time() - days * 86400, ACTIVITY_LOAD)
        partial = len(events) >= ACTIVITY_LOAD
        people = {u["email"]: u for u in db.list_users()}
        per = {}
        for e in events:
            p = per.setdefault(e["email"], dict(last=0.0, sign_ins=0, views=0, days=set()))
            p["last"] = max(p["last"], e["at"])
            p["sign_ins"] += e["action"] == "sign_in"
            p["views"] += e["action"] == "view"
            p["days"].add(local_time(e["at"], "%Y-%m-%d"))
        emails = sorted(set(people) | set(per), key=lambda m: (-per.get(m, {}).get("last", 0.0), m))
        rows = []
        for m in emails:
            p = per.get(m)
            u = people.get(m)
            access = sees(u) if u and u.get("active", True) else ("access off" if u else "no access")
            link = f'<a href="/admin/activity?days={days}&amp;email={esc(m)}">{esc(m)}</a>'
            if p:
                rows.append(f"<tr><td>{link}</td><td>{esc(access)}</td><td>{local_time(p['last'])}</td>"
                            f"<td>{len(p['days'])}</td><td>{p['sign_ins']}</td><td>{p['views']}</td></tr>")  # fmt: skip
            else:
                rows.append(f"<tr><td>{esc(m)}</td><td>{esc(access)}</td><td class='muted'>not in this period</td>"
                            "<td>0</td><td>0</td><td>0</td></tr>")  # fmt: skip
        shown = [e for e in events if not only or e["email"] == only]
        ev_rows = "".join(
            f"<tr><td>{local_time(e['at'])}</td><td>{esc(e['email'])}</td><td>{event_text(e)}</td></tr>"
            for e in shown[:ACTIVITY_SHOWN]
        )
        audit = db.list_audit(200)
        changes = [a for a in audit if not only or a.get("actor") == only or only in str(a.get("detail"))]
        ch_rows = "".join(
            f"<tr><td>{local_time(a['at'])}</td><td>{esc(a.get('actor', ''))}</td><td>{esc(a['action'])}</td>"
            f"<td>{esc(', '.join(f'{k}: {v}' for k, v in (a.get('detail') or {}).items()))}</td></tr>"
            for a in changes[:100]
        )
        span = "".join(
            f'<option value="{d}"{" selected" if d == days else ""}>last {d} day{"s" if d > 1 else ""}</option>'
            for d in ACTIVITY_SPANS
        )
        pick = '<option value="">everyone</option>' + "".join(
            f"<option{' selected' if m == only else ''}>{esc(m)}</option>" for m in emails
        )
        note = f" Only the newest {ACTIVITY_LOAD} entries are counted." if partial else ""
        more = f", the first {ACTIVITY_SHOWN} shown" if len(shown) > ACTIVITY_SHOWN else ""
        body = f"""<h1>Activity</h1>
<p class="sub">Sign-ins and pages opened by signed-in people (not photos or logos), kept {ACTIVITY_SPANS[-1]} days.
Times are {esc(SITE_TZ.key)}.{note}</p>
<form method="get" action="/admin/activity" class="stack"><label>Period<select name="days">{span}</select></label>
<label>Person<select name="email">{pick}</select></label><button>Show</button></form>
<h2>People</h2>
<div class="wrap"><table class="admin"><tr><th>Email</th><th>Sees</th><th>Last seen</th><th>Days active</th>
<th>Sign-ins</th><th>Pages</th></tr>{"".join(rows)}</table></div>
<h2>{"Events: " + esc(only) if only else "Events"}</h2>
<p class="sub">Newest first ({len(shown)}{more}).</p>
<div class="wrap"><table class="admin"><tr><th>When</th><th>Who</th><th>What</th></tr>
{ev_rows or "<tr><td colspan=3 class=muted>Nothing in this period.</td></tr>"}</table></div>
<h2>Admin changes</h2>
<div class="wrap"><table class="admin"><tr><th>When</th><th>Who</th><th>Action</th><th>Details</th></tr>
{ch_rows or "<tr><td colspan=4 class=muted>None.</td></tr>"}</table></div>
<p><a href="/admin">Back to admin</a></p>"""
        return page("Activity", body, "admin")

    @app.get("/admin/requests")
    @admin_only
    def admin_requests():
        items = []
        for r in db.list_requests(status="pending"):
            current = db.get_user(r["email"]) or {}
            asked = requested_teams(r)
            has = f" (now: {esc(sees(current))})" if current else ""
            fields = ""
            for team in asked:
                grant = grants(current).get(team, {})
                if team not in site_teams():
                    fields += f"<p class='sub'>{esc(team_label(team))} is not published now.</p>"
                    continue
                fields += (
                    f"<fieldset><legend>{esc(team_label(team))}</legend>"
                    f"<label>Role{role_select(team, r['role'])}</label>"
                    f"<label>Players (parents){player_checks(team, grant.get('players', []))}</label></fieldset>"
                )
            names = ", ".join(team_label(t) for t in asked)
            items.append(
                f"<li><b>{esc(r['email'])}</b> {esc(r.get('name') or '')} asks to be a <b>{esc(r['role'])}</b> for "
                f"<b>{esc(names)}</b>{has}"
                f"<span class='ev'>Players: {esc(r.get('players_text') or '–')} · Note: {esc(r.get('note') or '–')}"
                f" · {time.strftime('%Y-%m-%d %H:%M', time.gmtime(r['created']))} UTC</span>"
                f'<form method="post" action="/admin/requests/{esc(r["id"])}" class="stack" style="margin-top:8px">'
                f"{csrf_field()}{fields}"
                '<div><button class="primary" name="decision" value="approve">Approve</button> '
                '<button name="decision" value="deny">Deny</button></div></form></li>'
            )
        body = (
            "<h1>Access requests</h1><p class='sub'>Approving gives the account the role and players chosen here "
            "for each team asked about; a team left at no access is not granted, and access to other teams is "
            "kept. Check who is asking before approving: the players' names are what they typed.</p>"
            + (f'<ul class="obs">{"".join(items)}</ul>' if items else "<p class='sub'>Nothing waiting.</p>")
        )
        return page("Access requests", body, "admin")

    @app.post("/admin/requests/<rid>")
    @admin_only
    def admin_decide(rid):
        r = db.get_request(rid)
        if r is None or r["status"] != "pending":
            abort(404)
        decision = request.form.get("decision")
        if decision == "approve":
            parsed = parse_grants()
            new = {t: parsed[t] for t in requested_teams(r) if t in parsed}
            if not new:
                return page("Access requests", '<div class="flash">Choose coach or parent for at least one team.</div>',
                            "admin", 400)  # fmt: skip
            current = db.get_user(r["email"]) or {}
            err = save_user(g.user["email"], r["email"], is_admin(current), {**grants(current), **new}, True,
                            f"request {rid}")  # fmt: skip
            if err:
                return page("Access requests", f'<div class="flash">{esc(err)}</div>', "admin", 400)
        elif decision != "deny":
            abort(400)
        db.update_request(rid, dict(status="approved" if decision == "approve" else "denied",
                                    decided_by=g.user["email"], decided=time.time()))  # fmt: skip
        db.audit(g.user["email"], f"request_{decision}", {"id": rid, "email": r["email"]})
        try:
            notify.send_decision_notice(r["email"], decision == "approve", url_for("login", _external=True), send_mail)
        except Exception as e:  # noqa: BLE001 - the decision is saved; the email is a courtesy
            app.logger.warning("decision email failed: %s", e)
        return redirect(url_for("admin_requests"))

    @app.errorhandler(403)
    def forbidden(e):
        return page("Not available", "<h1>Not available</h1><p class='sub'>You do not have access to this page.</p>",
                    status=403)  # fmt: skip

    @app.errorhandler(404)
    def not_found(e):
        return page("Not found", "<h1>Not found</h1>", status=404)

    return app


if __name__ == "__main__":  # local development; Cloud Run runs gunicorn "app:create_app()"
    create_app().run(host="127.0.0.1", port=int(os.environ.get("PORT", "8080")), debug=False)
