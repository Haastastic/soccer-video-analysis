"""The page shell: header with the school's logo and name, navigation, theme toggle, and the coaching pages' styles.
Scripts carry the request's CSP nonce; nothing is loaded from outside the site."""

import coaching_html as ch
from coaching_html import esc

SITE_CSS = """
.top { background: var(--surface-2); border-bottom: 1px solid var(--line); }
.top .in { max-width: 1040px; margin: 0 auto; padding: 10px 16px; display: flex; align-items: center; gap: 14px;
  flex-wrap: wrap; }
.brand { display: flex; align-items: center; gap: 10px; color: var(--text-1); text-decoration: none;
  font-weight: 700; font-size: 17px; }
.brand img { width: 36px; height: 36px; object-fit: contain; }
nav.main { display: flex; gap: 4px; flex-wrap: wrap; }
nav.main a { color: var(--text-2); text-decoration: none; padding: 6px 10px; border-radius: 6px; font-size: 14px; }
nav.main a:hover, nav.main a.on { background: var(--surface); color: var(--text-1); }
nav.teams { display: flex; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
nav.teams a { padding: 5px 12px; font-size: 13px; font-weight: 600; color: var(--text-2); text-decoration: none; }
nav.teams a.on { background: var(--accent); color: #fff; }
fieldset { border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px; display: grid; gap: 8px; }
legend { font-weight: 600; font-size: 14px; }
.badge { background: var(--accent); color: #fff; border-radius: 9px; font-size: 11px; padding: 1px 6px;
  margin-left: 4px; font-weight: 600; }
.me { margin-left: auto; display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-3); }
.me form { display: inline; }
button, .btn { font: inherit; font-size: 14px; border: 1px solid var(--line); background: var(--surface);
  color: var(--text-1); border-radius: 6px; padding: 5px 10px; cursor: pointer; text-decoration: none;
  display: inline-block; }
button.primary, .btn.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
main { max-width: 1040px; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 12px; margin-top: 12px; }
.card { display: flex; gap: 14px; align-items: center; background: var(--surface-2); border-radius: 10px;
  padding: 12px 14px; color: var(--text-1); text-decoration: none; border: 1px solid transparent; }
.card:hover { border-color: var(--accent); }
.crest { width: 48px; height: 48px; object-fit: contain; flex: none; }
.crest.sm { width: 24px; height: 24px; vertical-align: middle; margin: -4px 4px 0 2px; }
.crest.blank { display: inline-flex; align-items: center; justify-content: center; background: var(--surface);
  border-radius: 50%; color: var(--text-3); font-size: 13px; }
.match { display: flex; gap: 12px; align-items: center; margin: 6px 0 10px; }
form.stack { display: grid; gap: 10px; max-width: 520px; }
label { font-size: 14px; color: var(--text-2); display: grid; gap: 4px; }
input[type=text], input[type=email], select, textarea { font: inherit; font-size: 14px; padding: 6px 8px;
  border: 1px solid var(--line); border-radius: 6px; background: var(--surface); color: var(--text-1); }
.checks { display: flex; flex-wrap: wrap; gap: 4px 14px; }
.checks label { display: flex; gap: 6px; align-items: center; color: var(--text-1); }
.flash { background: var(--surface-2); border-left: 4px solid var(--accent); padding: 8px 12px; border-radius: 6px;
  margin: 12px 0; }
.center { max-width: 460px; margin: 60px auto; text-align: center; }
.center .crest { width: 80px; height: 80px; }
table.admin td { vertical-align: top; }
.muted { color: var(--text-3); }
td:has(> .avatar.sm) { white-space: nowrap; }
@media (max-width: 640px) { .me { margin-left: 0; width: 100%; }
  .wrap th, .wrap td { white-space: nowrap; } }
"""

THEME_JS = """
(function () {
  const root = document.documentElement;
  try { const t = localStorage.getItem('theme'); if (t) root.dataset.theme = t; } catch (e) {}
  // this runs in <head> (so a saved theme applies before the page paints), before the button exists:
  // listen on the document instead of looking the button up
  document.addEventListener('click', (e) => {
    if (!e.target.closest('#theme')) return;
    const dark = root.dataset.theme ? root.dataset.theme === 'dark'
      : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('theme', root.dataset.theme); } catch (e) {}
  });
})();
"""


def document(title: str, body: str, nonce: str, header: str = "", site_name: str = "Coaching") -> str:
    tooltip = ch.TOOLTIP_JS.replace("<script>", f'<script nonce="{nonce}">')
    if 'id="watch"' in body:  # a player page with clips: its buttons' script, under the CSP nonce
        tooltip += ch.WATCH_JS.replace("<script>", f'<script nonce="{nonce}">')
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex">'
        f"<title>{esc(title)} · {esc(site_name)}</title>"
        '<link rel="icon" href="/favicon.ico" sizes="48x48"><link rel="icon" href="/favicon-32.png" type="image/png">'
        '<link rel="apple-touch-icon" href="/apple-touch-icon.png"><link rel="manifest" href="/site.webmanifest">'
        f'<style nonce="{nonce}">{ch.CSS}{ch.WEB_CSS}{SITE_CSS}</style>'
        f'<script nonce="{nonce}">{THEME_JS}</script></head>'
        f"<body>{header}<main>{body}</main>{tooltip}</body></html>"
    )


def sign_out(email: str, csrf: str, ask: bool = False) -> str:
    """ask: show a link to the access-request page (members who may want another team or more players)."""
    link = '<a href="/request">Request access</a>' if ask else ""
    return (
        f'<div class="me">{link}<span>{esc(email)}</span>'
        '<button id="theme" type="button" title="Light or dark">◐</button>'
        f'<form method="post" action="/logout"><input type="hidden" name="csrf" value="{esc(csrf)}">'
        "<button>Sign out</button></form></div>"
    )


def header(
    user: dict | None,
    email: str | None,
    settings: dict,
    name: str,
    team: str | None,
    teams: list,
    active: str,
    csrf: str,
    pending: int = 0,
    admin: bool = False,
) -> str:
    """user: the member's record, or None (signed out, or signed in without access: a plain header). team: the team
    whose page this is (None on admin and other site pages); teams: (id, label) of every team the viewer may open,
    shown as a switch when there is more than one."""
    if not user:
        me = sign_out(email, csrf) if email else ""
        return f'<header class="top"><div class="in"><span class="brand">Coaching</span>{me}</div></header>'
    base = f"/t/{team}" if team else (f"/t/{teams[0][0]}" if teams else "")
    logo = f'<img src="/t/{esc(team)}/logo" alt="">' if team and settings.get("logo") else ""
    links = [
        ("season", f"{base}/", "Season"),
        ("games", f"{base}/games", "Games"),
        ("players", f"{base}/players", "Players"),
    ]
    if not base:
        links = []
    if admin:
        badge = f'<span class="badge">{pending}</span>' if pending else ""
        links.append(("admin", "/admin", f"Admin{badge}"))
    nav = "".join(f'<a href="{href}"{" class=on" if key == active else ""}>{label}</a>' for key, href, label in links)
    switch = ""
    # the switch keeps the section (Games, Players): a single game or player page goes to that section's list,
    # since the other team has other games and players (owner, 2026-10-01)
    section = {"games": "games", "players": "players"}.get(active, "")
    if len(teams) > 1:
        switch = (
            '<nav class="teams" aria-label="Team">'
            + "".join(
                f'<a href="/t/{esc(t)}/{section}"{" class=on" if t == team else ""}>{esc(label)}</a>'
                for t, label in teams
            )
            + "</nav>"
        )
    home = f"{base}/" if base else "/"
    return (
        f'<header class="top"><div class="in"><a class="brand" href="{home}">{logo}{esc(name)}</a>{switch}'
        f'<nav class="main">{nav}</nav>{sign_out(user["email"], csrf, ask=not admin)}</div></header>'
    )


NOTE = (
    '<p class="note">Automatic analysis of youth soccer video, shared privately with invited families and coaches. '
    "Please do not copy or pass on names, photos, clips or numbers.</p>"
)


def privacy(contact: str) -> str:
    """The public privacy page (linked from Google's sign-in consent screen)."""
    who = f'<a href="mailto:{esc(contact)}">{esc(contact)}</a>' if contact else "the team's site administrator"
    return f"""<h1>Privacy</h1>
<p class="sub">This is a private site run by a volunteer for one youth soccer team's players, families and coaches.
It is not a commercial service.</p>
<h2>What the site shows</h2>
<p>Automatic analysis of game video: how much each player runs, where they play, touches of the ball, observations
for coaching, one photo of each player cropped from the game video, and short clips (about 10 seconds, no sound) of
each player's moments in a game. Nothing is shown to anyone who has not been given access. Coaches see every player;
parents and guardians see team totals and only their own child, including their child's clips only.</p>
<h2>What we keep about you</h2>
<ul>
<li>When you sign in with Google: your email address and name, used only to check what you may see.</li>
<li>If you ask for access: what you typed in the request and when you sent it.</li>
<li>A record of changes the administrators make (who was given access, and when).</li>
<li>A record of when you sign in and which pages of this site you open, kept 180 days and seen only by the site's
administrators, so they can see whether the pages are used.</li>
<li>One session cookie while you are signed in (12 hours), and your light or dark theme choice in your own
browser. No advertising, no third-party analytics, nothing that follows you to other sites.</li>
</ul>
<h2>Where it is kept and who can see it</h2>
<p>On Google Cloud in the United States, in private storage that only this site can read. It is never sold or given
to anyone else; Google hosts it and provides the sign-in.</p>
<h2>Removal</h2>
<p>To have your account, a request, or a child's page, photo or clips removed, contact {who}. Removal is done within
14 days.</p>"""
