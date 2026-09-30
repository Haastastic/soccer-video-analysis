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
@media (max-width: 640px) { .me { margin-left: 0; width: 100%; } }
"""

THEME_JS = """
(function () {
  const root = document.documentElement;
  try { const t = localStorage.getItem('theme'); if (t) root.dataset.theme = t; } catch (e) {}
  const b = document.getElementById('theme');
  if (!b) return;
  b.addEventListener('click', () => {
    const dark = root.dataset.theme ? root.dataset.theme === 'dark'
      : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('theme', root.dataset.theme); } catch (e) {}
  });
})();
"""


def document(title: str, body: str, nonce: str, header: str = "", site_name: str = "Coaching") -> str:
    tooltip = ch.TOOLTIP_JS.replace("<script>", f'<script nonce="{nonce}">')
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex">'
        f"<title>{esc(title)} · {esc(site_name)}</title>"
        f'<style nonce="{nonce}">{ch.CSS}{ch.WEB_CSS}{SITE_CSS}</style>'
        f'<script nonce="{nonce}">{THEME_JS}</script></head>'
        f"<body>{header}<main>{body}</main>{tooltip}</body></html>"
    )


def sign_out(email: str, csrf: str) -> str:
    return (
        f'<div class="me"><span>{esc(email)}</span>'
        '<button id="theme" type="button" title="Light or dark">◐</button>'
        f'<form method="post" action="/logout"><input type="hidden" name="csrf" value="{esc(csrf)}">'
        "<button>Sign out</button></form></div>"
    )


def header(user: dict | None, email: str | None, team: dict, active: str, csrf: str, pending: int = 0) -> str:
    """user: the member's record, or None (signed out, or signed in without access: a plain header)."""
    logo = '<img src="/logo/team" alt="">' if team.get("logo") else ""
    name = esc(team.get("name") or "Coaching")
    if not user:
        me = sign_out(email, csrf) if email else ""
        return f'<header class="top"><div class="in"><span class="brand">Coaching</span>{me}</div></header>'
    links = [("season", "/", "Season"), ("games", "/games", "Games"), ("players", "/players", "Players")]
    if user["role"] == "admin":
        badge = f'<span class="badge">{pending}</span>' if pending else ""
        links.append(("admin", "/admin", f"Admin{badge}"))
    nav = "".join(f'<a href="{href}"{" class=on" if key == active else ""}>{label}</a>' for key, href, label in links)
    return (
        f'<header class="top"><div class="in"><a class="brand" href="/">{logo}{name}</a>'
        f'<nav class="main">{nav}</nav>{sign_out(user["email"], csrf)}</div></header>'
    )


NOTE = (
    '<p class="note">Automatic analysis of youth soccer video, shared privately with invited families and coaches. '
    "Please do not copy or pass on names, photos or numbers.</p>"
)
