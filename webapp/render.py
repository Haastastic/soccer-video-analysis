"""Pages of the site, built per viewer from the release with coaching_html.py's builders (web mode).

Every function takes `base` (the team's URL prefix, e.g. /t/jv) and `visible`: the jerseys the viewer may see named
(None = everyone). Team numbers, medians and
charts always use every player; only rows, links, names and photos are limited.
"""

from store import Release

import coaching_html as ch
from coaching_html import esc


def display_name(r) -> str:
    return r["name"] if isinstance(r["name"], str) else f"#{int(r.jersey)}"


def web_opts(base: str, shell, photo_ok, visible, back_url: str, player_url, intro: str = "") -> ch.Web:
    return ch.Web(
        shell=shell,
        player_url=player_url,
        back_url=back_url,
        photo_url=lambda j: f"{base}/photo/{j}" if photo_ok(j) else None,
        visible=visible,
        intro=intro,
    )


def game_title(rel: Release, gid: str, games_meta: dict) -> str:
    opp = games_meta.get(gid, {}).get("opponent")
    return f"{rel.games[gid]['label']} vs {opp}" if opp else rel.games[gid]["label"]


def opponent_intro(base: str, gid: str, label: str, meta: dict) -> str:
    opp = meta.get("opponent")
    logo = f'<img class="crest" src="{base}/logo/game/{esc(gid)}" alt="">' if meta.get("logo") else ""
    vs = f"vs {esc(opp)}" if opp else "Opponent not set"
    return f'<p class="match">{logo}<span><b>{vs}</b><br><span class="sub">{esc(label)}</span></span></p>'


def season_page(base: str, rel: Release, shell, visible, photo_ok, games_meta: dict | None = None) -> str:
    web = web_opts(base, shell, photo_ok, visible, f"{base}/", lambda j: f"{base}/players/{j}")
    games_meta = games_meta or {}

    def game_cell(label: str) -> str:
        gid = next((k for k, g in rel.games.items() if g["label"] == label), label)
        opp = games_meta.get(gid, {}).get("opponent")
        vs = f" vs {esc(opp)}" if opp else ""
        return f'<a href="{base}/games/{esc(gid)}">{esc(label)}</a>{vs}'

    web.game_cell = game_cell
    per = rel.per
    return ch.multi_team_page(rel.season, per, rel.tagged, rel.order, web)


def game_page(base: str, rel: Release, gid: str, shell, visible, photo_ok, meta: dict) -> str:
    m = rel.per[gid]
    tips = rel.tips[gid]
    counts = {int(j): len(t) for j, t in tips.items()}
    web = web_opts(
        base,
        shell,
        photo_ok,
        visible,
        f"{base}/games/{gid}",
        lambda j: f"{base}/games/{gid}/players/{j}",
        opponent_intro(base, gid, rel.games[gid]["label"], meta),
    )
    return ch.team_page(m, counts, rel.games[gid]["n_windows"], web)


def game_player_page(base: str, rel: Release, gid: str, jersey: int, shell, photo_ok) -> str | None:
    m = rel.per[gid]
    row = m[m.jersey == jersey]
    if not len(row):
        return None
    r = row.iloc[0]
    g = rel.games[gid]
    smp = rel.samples[gid]
    web = web_opts(base, shell, photo_ok, None, f"{base}/games/{gid}", lambda j: f"{base}/games/{gid}/players/{j}")
    med = rel.med(rel.meds[gid], jersey)
    return ch.player_page(
        r, rel.tips[gid].get(jersey, []), med, smp[smp.jersey == jersey], g["length"], g["width"],
        g["n_windows"], r.conf, web,
    )  # fmt: skip


def season_player_page(base: str, rel: Release, jersey: int, shell, photo_ok) -> str | None:
    row = rel.season[rel.season.jersey == jersey]
    if not len(row):
        return None
    r = row.iloc[0]
    rows = {k: q.set_index("jersey").loc[jersey] for k, q in rel.per.items() if jersey in set(q.jersey)}
    meds = {"Pooled": rel.med(rel.season_meds, jersey), **{k: rel.med(rel.meds[k], jersey) for k in rows}}
    samples = {
        k: (rel.samples[k][rel.samples[k].jersey == jersey], rel.games[k]["length"], rel.games[k]["width"])
        for k in rel.order
    }
    web = web_opts(base, shell, photo_ok, None, f"{base}/", lambda j: f"{base}/players/{j}")
    return ch.multi_player_page(r, rows, meds, rel.tagged.get(jersey, []), samples, r.conf, web)


def games_page(base: str, rel: Release, games_meta: dict, visible) -> str:
    cards = []
    for gid in reversed(rel.order):  # newest first
        g, meta = rel.games[gid], games_meta.get(gid, {})
        q = rel.per[gid]
        mine = "" if visible is None else f" · {int(q.jersey.isin(visible).sum())} of yours played"
        logo = (
            f'<img class="crest" src="{base}/logo/game/{esc(gid)}" alt="">'
            if meta.get("logo")
            else '<span class="crest blank" aria-hidden="true">vs</span>'
        )
        opp = f"vs {esc(meta['opponent'])}" if meta.get("opponent") else "Opponent not set"
        cards.append(
            f'<a class="card game" href="{base}/games/{esc(gid)}">{logo}<span><b>{opp}</b><br>'
            f'<span class="sub">{esc(g["label"])} · {len(q)} players · {q.minutes.sum():.0f} identified '
            f"player-minutes{mine}</span></span></a>"
        )
    return f"""<h1>Games</h1>
<p class="sub">Each game on its own: the team table, and every player's page for that game.</p>
<div class="cards">{"".join(cards)}</div>"""


def players_page(base: str, rel: Release, visible, photo_ok) -> str:
    order = {"goalkeeper": 0, "defender": 1, "midfielder": 2, "forward": 3}
    m = rel.season.assign(_o=rel.season.role.map(order)).sort_values(["_o", "minutes"], ascending=[True, False])
    web = web_opts(base, None, photo_ok, visible, f"{base}/", lambda j: f"{base}/players/{j}")
    cards = []
    for _, r in m.iterrows():
        j = int(r.jersey)
        if not web.shows(j):
            continue
        name = display_name(r)
        cards.append(
            f'<a class="card player" href="{base}/players/{j}">{ch.avatar(web, j, name)}<span><b>{esc(name)}</b> '
            f'<span class="sub">#{j}</span><br><span class="sub">{ch.grade_of(r)}{esc(str(r.role).capitalize())} · '
            f"{r.minutes:.0f} min seen</span></span></a>"
        )
    if not cards:
        return "<h1>Players</h1><p class='sub'>No players have been shared with you yet.</p>"
    return f"""<h1>Players</h1>
<p class="sub">Every game together. Open a player for their observations, where they play and how they compare
with teammates game by game.</p>
<div class="cards">{"".join(cards)}</div>"""
