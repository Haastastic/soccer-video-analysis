"""The coaching site: the coaching pages for invited Google accounts, with roles and access requests.

Roles: admin (everything, manages users), coach (every player), parent (the players an admin assigned; team pages
show team numbers with only their players named). Any Google account can sign in, but an account that is not an
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

Run locally: python site/app.py (see site/README.md).
"""

import os
import re
import secrets
import sys
import time
from datetime import timedelta
from functools import wraps
from io import BytesIO
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):  # coaching_html.py sits beside this in the container, one level up in the repo
    if str(p) not in sys.path:
        sys.path.append(str(p))

import notify  # noqa: E402
import render  # noqa: E402
import ui  # noqa: E402
from db import ROLES, FirestoreDB, MemoryDB, norm_email  # noqa: E402
from flask import Flask, abort, g, redirect, request, session, url_for  # noqa: E402
from store import GcsStore, LocalStore, ReleaseCache  # noqa: E402
from werkzeug.middleware.proxy_fix import ProxyFix  # noqa: E402

from coaching_html import esc  # noqa: E402

MAX_UPLOAD = 2 * 1024 * 1024
LOGO_PX = 256
REQUEST_TEXT_MAX, REQUEST_NOTE_MAX = 200, 500
REQUESTS_PER_DAY = 3
SESSION_HOURS = 12


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
            if u and u.get("active", True) and u.get("role") in ROLES:
                g.user = u
        if request.method == "POST":
            sent = request.form.get("csrf", "")
            if not sent or not secrets.compare_digest(sent, session.get("csrf", "")):
                abort(400, "The form expired. Go back, reload the page and try again.")

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

    def team() -> dict:
        return db.get_setting("team")

    def pending_count() -> int:
        return len(db.list_requests(status="pending")) if g.user and g.user["role"] == "admin" else 0

    def page(title: str, body: str, active: str = "", status: int = 200):
        # only members see the school's name and logo; everyone else sees a plain header
        t = team() if g.user else {}
        head = ui.header(g.user, session.get("email"), t, active, csrf(), pending_count())
        html_doc = ui.document(title, body + ui.NOTE, g.nonce, head, t.get("name") or "Coaching")
        return html_doc, status, {"Content-Type": "text/html; charset=utf-8"}

    def shell_for(active: str):
        return lambda title, body: page(title, body, active)[0]

    def visible() -> set | None:
        """Jerseys this viewer may see named: None = all (admin, coach)."""
        if g.user["role"] in ("admin", "coach"):
            return None
        return {int(j) for j in g.user.get("players", [])}

    def can_view(jersey: int) -> bool:
        v = visible()
        return v is None or int(jersey) in v

    def photo_ok_for(rel):
        return lambda j: int(j) in rel.photos and can_view(j)

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
                if request.path.startswith(("/photo/", "/logo/")):
                    abort(403)
                return redirect(url_for("access_request"))
            return fn(*a, **kw)

        return wrapper

    def admin_only(fn):
        @wraps(fn)
        @member
        def wrapper(*a, **kw):
            if g.user["role"] != "admin":
                abort(403)
            return fn(*a, **kw)

        return wrapper

    def release():
        rel = releases.get()
        if rel is None:
            return None
        return rel

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
            db.put_user(admin, dict(role="admin", players=[], active=True, invited_by="bootstrap", created=time.time()))
            db.audit(admin, "bootstrap_admin", {})

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
        msg = ""
        if request.method == "POST":
            role = request.form.get("role", "")
            text = request.form.get("players", "").strip()
            note = request.form.get("note", "").strip()
            recent = [r for r in mine if r["created"] > time.time() - 86400]
            if pending:
                msg = "You already have a request waiting."
            elif len(recent) >= REQUESTS_PER_DAY:
                msg = "Too many requests today. Try again tomorrow."
            elif role not in ("coach", "parent"):
                msg = "Choose coach or parent."
            elif len(text) > REQUEST_TEXT_MAX or len(note) > REQUEST_NOTE_MAX:
                msg = "That is too long."
            elif role == "parent" and not text:
                msg = "Say which player or players you are asking about."
            else:
                rid = db.add_request(
                    dict(email=email, name=session.get("name", ""), role=role, players_text=text, note=note,
                         created=time.time(), status="pending", decided_by=None)
                )  # fmt: skip
                db.audit(email, "request_access", {"id": rid, "role": role})
                admins = [u["email"] for u in db.list_users() if u["role"] == "admin" and u.get("active", True)]
                try:
                    notify.send_request_notice(admins, email, url_for("admin_requests", _external=True), send_mail)
                except Exception as e:  # noqa: BLE001 - the request is stored; the email is a courtesy
                    app.logger.warning("access request email failed: %s", e)
                return redirect(url_for("access_request"))
        latest = mine[-1] if mine else None
        has = g.user is not None
        intro = (
            "<p class='sub'>You can see this site already. Ask here for access to more players.</p>"
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
            form = f"""<form method="post" class="stack">{csrf_field()}
<label>I am a<select name="role"><option value="parent">Parent or guardian</option>
<option value="coach">Coach</option></select></label>
<label>Which player(s)? (names; parents only)<input type="text" name="players" maxlength="{REQUEST_TEXT_MAX}"></label>
<label>Note for the admin (optional)<textarea name="note" rows="3" maxlength="{REQUEST_NOTE_MAX}"></textarea></label>
<button class="primary">Request access</button></form>"""
        flash = f'<div class="flash">{esc(msg)}</div>' if msg else ""
        body = f"<h1>Request access</h1>{intro}<p>Signed in as <b>{esc(email)}</b>.</p>{flash}{status}{form}"
        return page("Request access", body)

    # ------------------------------------------------------------------ pages

    @app.get("/")
    @member
    def season():
        rel = release()
        if rel is None:
            return nothing_yet()
        if len(rel.order) == 1:
            return redirect(url_for("game", gid=rel.order[0]))
        return render.season_page(rel, shell_for("season"), visible(), photo_ok_for(rel))

    @app.get("/games")
    @member
    def games():
        rel = release()
        if rel is None:
            return nothing_yet()
        meta = {k: db.get_game(k) for k in rel.order}
        return page("Games", render.games_page(rel, meta, visible()), "games")

    @app.get("/games/<gid>")
    @member
    def game(gid):
        rel = release()
        if rel is None or gid not in rel.games:
            abort(404)
        return render.game_page(rel, gid, shell_for("games"), visible(), photo_ok_for(rel), db.get_game(gid))

    @app.get("/games/<gid>/players/<int:jersey>")
    @member
    def game_player(gid, jersey):
        rel = release()
        if rel is None or gid not in rel.games:
            abort(404)
        if not can_view(jersey):
            abort(403)
        out = render.game_player_page(rel, gid, jersey, shell_for("games"), photo_ok_for(rel))
        return out if out is not None else abort(404)

    @app.get("/players")
    @member
    def players():
        rel = release()
        if rel is None:
            return nothing_yet()
        return page("Players", render.players_page(rel, visible(), photo_ok_for(rel)), "players")

    @app.get("/players/<int:jersey>")
    @member
    def player(jersey):
        rel = release()
        if rel is None:
            abort(404)
        if not can_view(jersey):
            abort(403)
        if len(rel.order) == 1:
            return redirect(url_for("game_player", gid=rel.order[0], jersey=jersey))
        out = render.season_player_page(rel, jersey, shell_for("players"), photo_ok_for(rel))
        return out if out is not None else abort(404)

    def image(data: bytes | None, mime: str):
        if not data:
            abort(404)
        return data, 200, {"Content-Type": mime, "Cache-Control": "private, max-age=300"}

    @app.get("/photo/<int:jersey>")
    @member
    def photo(jersey):
        if not can_view(jersey):
            abort(403)
        rel = release()
        return image(rel.photo(jersey) if rel else None, "image/jpeg")

    @app.get("/logo/team")
    @member
    def team_logo():
        return image(team().get("logo"), "image/png")

    @app.get("/logo/game/<gid>")
    @member
    def game_logo(gid):
        return image(db.get_game(gid).get("logo"), "image/png")

    # ------------------------------------------------------------------ admin

    def roster(rel) -> list:
        if rel is None:
            return []
        m = rel.season.sort_values("jersey")
        return [(int(r.jersey), render.display_name(r)) for _, r in m.iterrows()]

    def player_checks(rel, chosen: list, prefix: str = "players") -> str:
        boxes = "".join(
            f'<label><input type="checkbox" name="{prefix}" value="{j}"{" checked" if j in chosen else ""}>'
            f"{esc(n)} <span class='muted'>#{j}</span></label>"
            for j, n in roster(rel)
        )
        return f'<div class="checks">{boxes or "<span class=muted>No players published yet.</span>"}</div>'

    def role_select(current: str) -> str:
        opts = "".join(f'<option value="{r}"{" selected" if r == current else ""}>{r}</option>' for r in ROLES)
        return f'<select name="role">{opts}</select>'

    def other_active_admins(email: str) -> int:
        return sum(1 for u in db.list_users() if u["role"] == "admin" and u.get("active", True) and u["email"] != email)

    def parse_players(rel) -> list:
        valid = {j for j, _ in roster(rel)}
        return sorted({int(x) for x in request.form.getlist("players") if x.isdigit() and int(x) in valid})

    @app.get("/admin")
    @admin_only
    def admin():
        rel = release()
        names = dict(roster(rel))
        users = []
        for u in db.list_users():
            who = ", ".join(f"{names.get(j, '#' + str(j))}" for j in u.get("players", [])) or "–"
            if u["role"] != "parent":
                who = "all players"
            users.append(
                f"<tr><td>{esc(u['email'])}</td><td>{esc(u['role'])}</td><td>{esc(who)}</td>"
                f"<td>{'active' if u.get('active', True) else '<span class=muted>off</span>'}</td>"
                f'<td><details><summary>Edit</summary><form method="post" action="/admin/users" class="stack">'
                f'{csrf_field()}<input type="hidden" name="email" value="{esc(u["email"])}">'
                f"<label>Role{role_select(u['role'])}</label>"
                f"<label>Players (parents){player_checks(rel, u.get('players', []))}</label>"
                f'<label><span><input type="checkbox" name="active" value="1"'
                f"{' checked' if u.get('active', True) else ''}> Active</span></label>"
                "<button>Save</button></form></details></td></tr>"
            )
        t = team()
        game_rows = []
        for gid in rel.order if rel else []:
            meta = db.get_game(gid)
            logo = f'<img class="crest" src="/logo/game/{esc(gid)}" alt="">' if meta.get("logo") else ""
            game_rows.append(
                f"<tr><td>{esc(gid)}</td><td>{logo}</td><td>"
                f'<form method="post" action="/admin/games/{esc(gid)}" enctype="multipart/form-data" class="stack">'
                f'{csrf_field()}<label>Opponent<input type="text" name="opponent" maxlength="80" '
                f'value="{esc(meta.get("opponent", ""))}"></label>'
                '<label>Opponent logo (PNG or JPEG)<input type="file" name="logo" accept="image/png,image/jpeg">'
                "</label><button>Save</button></form></td></tr>"
            )
        n_pending = pending_count()
        body = f"""<h1>Admin</h1>
<p><a class="btn{" primary" if n_pending else ""}" href="/admin/requests">Access requests ({n_pending} waiting)</a></p>
<h2>People</h2>
<p class="sub">Coaches see every player. Parents see team numbers and only the players ticked for them.</p>
<div class="wrap"><table class="admin"><tr><th>Email</th><th>Role</th><th>Sees</th><th>Status</th><th></th></tr>
{"".join(users)}</table></div>
<h2>Add a person</h2>
<form method="post" action="/admin/users" class="stack">{csrf_field()}
<label>Google account email<input type="email" name="email" required></label>
<label>Role{role_select("parent")}</label>
<label>Players (parents){player_checks(rel, [])}</label>
<input type="hidden" name="active" value="1"><button class="primary">Add</button></form>
<h2>Our team</h2>
<form method="post" action="/admin/team" enctype="multipart/form-data" class="stack">{csrf_field()}
<label>School or team name<input type="text" name="name" maxlength="80" value="{esc(t.get("name", ""))}"></label>
<label>Logo (PNG or JPEG, up to 2 MB){'<img class="crest" src="/logo/team" alt="">' if t.get("logo") else ""}
<input type="file" name="logo" accept="image/png,image/jpeg"></label>
<button>Save</button></form>
<h2>Games</h2>
<div class="wrap"><table class="admin"><tr><th>Game</th><th>Logo</th><th>Opponent</th></tr>{"".join(game_rows)}
</table></div>"""
        return page("Admin", body, "admin")

    def save_user(actor: str, email: str, role: str, players: list, active: bool, via: str) -> str | None:
        """Error text, or None when saved."""
        email = norm_email(email)
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            return "That is not an email address."
        if role not in ROLES:
            return "Unknown role."
        old = db.get_user(email) or {}
        if old.get("role") == "admin" and (role != "admin" or not active) and other_active_admins(email) == 0:
            return "Keep at least one active admin."
        user = dict(old, role=role, players=players if role == "parent" else [], active=active)
        user.setdefault("invited_by", actor)
        user.setdefault("created", time.time())
        db.put_user(email, user)
        db.audit(actor, "save_user", {"email": email, "role": role, "players": user["players"], "active": active,
                                      "via": via})  # fmt: skip
        return None

    @app.post("/admin/users")
    @admin_only
    def admin_users():
        rel = release()
        err = save_user(
            g.user["email"], request.form.get("email", ""), request.form.get("role", ""), parse_players(rel),
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

    @app.post("/admin/team")
    @admin_only
    def admin_team():
        t = team()
        t["name"] = request.form.get("name", "").strip()[:80]
        try:
            logo = uploaded_logo()
        except ValueError as e:
            return bad_logo(e)
        if logo:
            t["logo"] = logo
        db.put_setting("team", t)
        db.audit(g.user["email"], "save_team", {"name": t["name"], "logo": bool(logo)})
        return redirect(url_for("admin"))

    @app.post("/admin/games/<gid>")
    @admin_only
    def admin_game(gid):
        rel = release()
        if rel is None or gid not in rel.games:
            abort(404)
        meta = db.get_game(gid)
        meta["opponent"] = request.form.get("opponent", "").strip()[:80]
        try:
            logo = uploaded_logo()
        except ValueError as e:
            return bad_logo(e)
        if logo:
            meta["logo"] = logo
        db.put_game(gid, meta)
        db.audit(g.user["email"], "save_game", {"game": gid, "opponent": meta["opponent"], "logo": bool(logo)})
        return redirect(url_for("admin"))

    @app.get("/admin/requests")
    @admin_only
    def admin_requests():
        rel = release()
        items = []
        for r in db.list_requests(status="pending"):
            current = db.get_user(r["email"]) or {}
            items.append(
                f"<li><b>{esc(r['email'])}</b> {esc(r.get('name') or '')} asks to be a <b>{esc(r['role'])}</b>"
                f"{' (has access as ' + esc(current['role']) + ')' if current else ''}"
                f"<span class='ev'>Players: {esc(r.get('players_text') or '–')} · Note: {esc(r.get('note') or '–')}"
                f" · {time.strftime('%Y-%m-%d %H:%M', time.gmtime(r['created']))} UTC</span>"
                f'<form method="post" action="/admin/requests/{esc(r["id"])}" class="stack" style="margin-top:8px">'
                f"{csrf_field()}<label>Role{role_select(r['role'])}</label>"
                f"<label>Players (parents){player_checks(rel, current.get('players', []))}</label>"
                '<div><button class="primary" name="decision" value="approve">Approve</button> '
                '<button name="decision" value="deny">Deny</button></div></form></li>'
            )
        body = (
            "<h1>Access requests</h1><p class='sub'>Approving gives the account the role and players chosen here. "
            "Check who is asking before approving: the players' names are what they typed.</p>"
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
            err = save_user(g.user["email"], r["email"], request.form.get("role", ""), parse_players(release()),
                            True, f"request {rid}")  # fmt: skip
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
