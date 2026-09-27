"""Local HTML pages for coaching_tips.py: one per player plus a team index, fully self-contained.

The pages hold names of minors, so they are written under data/coaching/ (git-ignored), make no network requests
(no fonts, scripts or images from outside; charts are inline SVG) and are never published.

Charts follow the dataviz method: where a player spends time is a sequential single-hue (blue) heatmap on the
pitch, attacking always to the right; work rate through the game is a dot plot of the ratio to teammates with a
+-2 standard error whisker and the team line at 1.0 (a ratio, so no zero-based bars). Light and dark themes use
their own steps; every mark has a hover tooltip and the numbers are also in a table.
"""

import html
import math

import numpy as np
import pandas as pd

PITCH_W = 64.0  # metres across (touchlines about +-32 at this venue)
CELL_X, CELL_Y = 5.0, 64.0 / 14  # heatmap cell size in metres
# heatmap steps (CSS --heat-0..5): blue 150..650 on light; on dark, 600..250 so low values recede
PHASES = [
    ("h1_early", "1st half, early"),
    ("h1_late", "1st half, late"),
    ("h2_early", "2nd half, early"),
    ("h2_late", "2nd half, late"),
]

CSS = """
:root { color-scheme: light;
  --surface: #fcfcfb; --surface-2: #f3f2ef; --text-1: #0b0b0b; --text-2: #52514e; --text-3: #7a7974;
  --line: #d9d8d3; --pitch: #eef5ee; --pitch-line: #9fb39f; --accent: #2a78d6; --ref: #7a7974;
  --heat-0: #b7d3f6; --heat-1: #86b6ef; --heat-2: #5598e7; --heat-3: #2a78d6; --heat-4: #1c5cab; --heat-5: #104281;
}
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) { color-scheme: dark;
  --surface: #1a1a19; --surface-2: #242423; --text-1: #ffffff; --text-2: #c3c2b7; --text-3: #8f8e86;
  --line: #3a3a37; --pitch: #1f2a20; --pitch-line: #4f6650; --accent: #3987e5; --ref: #8f8e86;
  --heat-0: #184f95; --heat-1: #1c5cab; --heat-2: #256abf; --heat-3: #3987e5; --heat-4: #6da7ec; --heat-5: #9ec5f4;
} }
:root[data-theme="dark"] { color-scheme: dark;
  --surface: #1a1a19; --surface-2: #242423; --text-1: #ffffff; --text-2: #c3c2b7; --text-3: #8f8e86;
  --line: #3a3a37; --pitch: #1f2a20; --pitch-line: #4f6650; --accent: #3987e5; --ref: #8f8e86;
  --heat-0: #184f95; --heat-1: #1c5cab; --heat-2: #256abf; --heat-3: #3987e5; --heat-4: #6da7ec; --heat-5: #9ec5f4;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--text-1);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 920px; margin: 0 auto; padding: 24px 16px 48px; }
a { color: var(--accent); }
h1 { font-size: 26px; margin: 4px 0 2px; } h2 { font-size: 18px; margin: 32px 0 8px; }
.sub { color: var(--text-2); margin: 0 0 4px; }
.private { font-size: 13px; color: var(--text-3); border: 1px solid var(--line); border-radius: 6px;
  padding: 6px 10px; margin-bottom: 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; }
.tile { background: var(--surface-2); border-radius: 8px; padding: 12px 14px; }
.tile .k { font-size: 13px; color: var(--text-2); }
.tile .v { font-size: 26px; font-weight: 600; font-variant-numeric: tabular-nums; }
.tile .c { font-size: 13px; color: var(--text-3); }
.obs { list-style: none; padding: 0; margin: 0; display: grid; gap: 10px; }
.obs li { background: var(--surface-2); border-radius: 8px; padding: 10px 14px; border-left: 4px solid var(--line); }
.obs li.strength { border-left-color: var(--accent); }
.obs .kind { font-weight: 600; } .obs .ev { display: block; font-size: 13px; color: var(--text-3); margin-top: 2px; }
.chart { width: 100%; height: auto; display: block; }
.legend { display: flex; align-items: center; gap: 6px; font-size: 13px; color: var(--text-2); margin-top: 6px;
  flex-wrap: wrap; }
.legend .sw { width: 22px; height: 12px; border-radius: 2px; display: inline-block; }
table { border-collapse: collapse; width: 100%; font-size: 14px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); }
th { color: var(--text-2); font-weight: 600; } td.n, th.n { text-align: right; }
.wrap { overflow-x: auto; }
tr.few td { color: var(--text-3); } tr.few small { font-size: 12px; }
details { margin-top: 8px; } summary { cursor: pointer; color: var(--text-2); font-size: 14px; }
.note { font-size: 13px; color: var(--text-3); margin-top: 28px; }
#tip { position: fixed; pointer-events: none; background: var(--text-1); color: var(--surface); font-size: 13px;
  padding: 4px 8px; border-radius: 4px; display: none; z-index: 10; max-width: 260px; }
.hit:hover { stroke: var(--text-1); stroke-width: 0.4; }
"""

TOOLTIP_JS = """
<div id="tip" role="tooltip"></div>
<script>
const tip = document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el => {
  el.addEventListener('mousemove', e => { tip.textContent = el.dataset.tip; tip.style.display = 'block';
    tip.style.left = Math.min(e.clientX + 12, innerWidth - tip.offsetWidth - 8) + 'px';
    tip.style.top = (e.clientY + 14) + 'px'; });
  el.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
});
</script>
"""


def esc(x) -> str:
    return html.escape(str(x))


def page(title: str, body: str) -> str:
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{esc(title)}</title>'
        f"<style>{CSS}</style></head><body><main>{body}</main>{TOOLTIP_JS}</body></html>"
    )


def pitch_lines(length: float) -> str:
    """Pitch markings in metres; x from our goal (0) to theirs (length), y across (-32..32)."""
    w, h = length, PITCH_W / 2
    box_w, box_d, six_w, six_d = 40.32 / 2, 16.5, 18.32 / 2, 5.5
    parts = [
        f'<rect x="0" y="{-h}" width="{w}" height="{2 * h}"/>',
        f'<line x1="{w / 2}" y1="{-h}" x2="{w / 2}" y2="{h}"/>',
        f'<circle cx="{w / 2}" cy="0" r="9.15"/>',
    ]
    for gx, sgn in ((0, 1), (w, -1)):
        parts.append(f'<rect x="{min(gx, gx + sgn * box_d)}" y="{-box_w}" width="{box_d}" height="{2 * box_w}"/>')
        parts.append(f'<rect x="{min(gx, gx + sgn * six_d)}" y="{-six_w}" width="{six_d}" height="{2 * six_w}"/>')
    return f'<g fill="none" stroke="var(--pitch-line)" stroke-width="0.35">{"".join(parts)}</g>'


def heatmap_svg(x: np.ndarray, y: np.ndarray, length: float) -> tuple:
    """(svg, legend html): share of visible time per cell, 6 quantized steps of one blue ramp."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.clip(x[ok], 0, length - 1e-6), np.clip(y[ok], -PITCH_W / 2, PITCH_W / 2 - 1e-6)
    nx, ny = int(math.ceil(length / CELL_X)), int(round(PITCH_W / CELL_Y))
    hist, _, _ = np.histogram2d(x, y, bins=[nx, ny], range=[[0, length], [-PITCH_W / 2, PITCH_W / 2]])
    share = hist / max(hist.sum(), 1)
    top = share.max() if share.max() > 0 else 1
    edges = np.linspace(0, top, 7)[1:]  # 6 equal steps up to the busiest cell
    cells = []
    cw = length / nx
    for i in range(nx):
        for j in range(ny):
            v = share[i, j]
            if v <= 0:
                continue
            step = int(np.searchsorted(edges, v, side="left"))
            step = min(step, 5)
            x0, y0 = i * cw, -PITCH_W / 2 + j * CELL_Y
            cells.append(
                f'<rect class="hit" x="{x0:.2f}" y="{y0:.2f}" width="{cw - 0.3:.2f}" height="{CELL_Y - 0.3:.2f}" '
                f'rx="0.6" fill="var(--heat-{step})" data-tip="{100 * v:.1f}% of visible time here"/>'
            )
    pad = 4
    svg = (
        f'<svg class="chart" viewBox="{-pad} {-PITCH_W / 2 - pad} {length + 2 * pad} {PITCH_W + 2 * pad + 6}" '
        f'role="img" aria-label="Where the player spends visible time on the pitch; attacking to the right">'
        f'<rect x="0" y="{-PITCH_W / 2}" width="{length}" height="{PITCH_W}" fill="var(--pitch)" rx="1"/>'
        f"{''.join(cells)}{pitch_lines(length)}"
        f'<text x="1" y="{PITCH_W / 2 + 5}" font-size="2.2" fill="var(--text-2)">Our goal</text>'
        f'<text x="{length - 1}" y="{PITCH_W / 2 + 5}" font-size="2.2" fill="var(--text-2)" text-anchor="end">'
        "Attacking →</text></svg>"
    )
    legend = (
        '<div class="legend">Less time'
        + "".join(f'<span class="sw" style="background:var(--heat-{k})"></span>' for k in range(6))
        + f"More time (busiest cell {100 * top:.1f}%)</div>"
    )
    return svg, legend


def phase_svg(r: pd.Series) -> str:
    """Relative work rate per quarter of the game, +-2 SE whiskers, team median line at 1.0."""
    pts = []
    for key, label in PHASES:
        k, v, mins = r.get(f"{key}_k", 0), r.get(f"{key}_rel", np.nan), r.get(f"{key}_min", 0)
        k = 0 if pd.isna(k) else int(k)
        pts.append((label, v, k, 0 if pd.isna(mins) else mins))
    vals = [v for _, v, k, _ in pts if k > 0 and np.isfinite(v)]
    se = [2 * r.rel_sd / math.sqrt(k) for _, v, k, _ in pts if k > 0 and np.isfinite(v)]
    lo = min([1.0] + [v - s for v, s in zip(vals, se, strict=True)]) - 0.05
    hi = max([1.0] + [v + s for v, s in zip(vals, se, strict=True)]) + 0.05
    W, H, L, R, T, B = 640, 220, 56, 96, 16, 44  # right margin holds the reference label

    def ypx(v):
        return T + (hi - v) / (hi - lo) * (H - T - B)

    xs = [L + (i + 0.5) * (W - L - R) / 4 for i in range(4)]
    grid = []
    for g in np.arange(math.ceil(lo * 10) / 10, hi, 0.1 if hi - lo < 1 else 0.2):
        grid.append(
            f'<line x1="{L}" x2="{W - R}" y1="{ypx(g):.1f}" y2="{ypx(g):.1f}" stroke="var(--line)" '
            f'stroke-width="1"/><text x="{L - 8}" y="{ypx(g) + 4:.1f}" font-size="12" text-anchor="end" '
            f'fill="var(--text-3)">{g:.1f}</text>'
        )
    ref = (
        f'<line x1="{L}" x2="{W - R}" y1="{ypx(1):.1f}" y2="{ypx(1):.1f}" stroke="var(--ref)" stroke-width="1.5" '
        f'stroke-dasharray="4 4"/><text x="{W - R + 8}" y="{ypx(1) + 4:.1f}" font-size="12" '
        'fill="var(--text-2)">team median</text>'
    )
    marks = []
    for (label, v, k, mins), x in zip(pts, xs, strict=True):
        marks.append(
            f'<text x="{x:.1f}" y="{H - 18}" font-size="12" text-anchor="middle" '
            f'fill="var(--text-2)">{esc(label)}</text>'
        )
        if k == 0 or not np.isfinite(v):
            marks.append(
                f'<text x="{x:.1f}" y="{ypx(1) + 18:.1f}" font-size="12" text-anchor="middle" '
                'fill="var(--text-3)">not enough seen</text>'
            )
            continue
        s = 2 * r.rel_sd / math.sqrt(k)
        tip = f"{label}: {v:.2f} x team ({mins:.1f} min seen, {k} window{'s' if k > 1 else ''}; +-{s:.2f})"
        marks.append(
            f'<g data-tip="{esc(tip)}"><rect x="{x - 22:.1f}" y="{T}" width="44" height="{H - T - B}" '
            f'fill="transparent"/><line x1="{x:.1f}" x2="{x:.1f}" y1="{ypx(v + s):.1f}" y2="{ypx(v - s):.1f}" '
            f'stroke="var(--accent)" stroke-width="2" stroke-linecap="round" opacity="0.55"/>'
            f'<circle cx="{x:.1f}" cy="{ypx(v):.1f}" r="5" fill="var(--accent)" stroke="var(--surface)" '
            f'stroke-width="2"/><text x="{x + 10:.1f}" y="{ypx(v) + 4:.1f}" font-size="12" '
            f'fill="var(--text-1)">{v:.2f}</text></g>'
        )
    return (
        f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Work rate relative to teammates '
        f'through the game">{"".join(grid)}{ref}{"".join(marks)}</svg>'
    )


def signed(v: float) -> str:
    """Whole metres with a sign, without a "-0"."""
    if not np.isfinite(v):
        return "–"
    v = int(round(v))
    return f"{v:+d}" if v else "0"


def tile(label: str, value: str, compare: str) -> str:
    return (
        f'<div class="tile"><div class="k">{esc(label)}</div><div class="v">{value}</div>'
        f'<div class="c">{esc(compare)}</div></div>'
    )


def player_page(
    r: pd.Series, tips: list, med: pd.Series, samples: pd.DataFrame, length: float, n_windows: int, confidence: str
) -> str:
    name = r["name"] if isinstance(r["name"], str) else f"#{r.jersey}"
    role = r.role
    heat, legend = heatmap_svg(samples.from_goal.to_numpy(), samples.y_team.to_numpy(), length)

    med_ok = bool(np.isfinite(med.get("depth", np.nan)))

    def cmp(v, fmt, unit=""):
        return f"{role}s: {fmt.format(v)}{unit}" if np.isfinite(v) else ""

    tiles = "".join(
        [
            tile("Seen", f"{r.minutes:.0f} min", f"{r.windows} of {n_windows} windows · confidence {confidence}"),
            tile("Work rate", f"{r.m_per_min:.0f} m/min", f"{r.work_rel:.2f} × teammates in the same minutes"),
            tile("Time at 4 m/s or faster", f"{r.pct_fast:.1f}%", cmp(med.pct_fast, "{:.1f}", "%")),
            tile("Distance from our goal", f"{r.from_goal:.0f} m", cmp(med.from_goal, "{:.0f}", " m")),
            tile("Depth vs team line", f"{signed(r.depth)} m", f"{role}s: {signed(med.depth)} m" if med_ok else ""),
            tile(
                "Near the ball (10 m)",
                f"{r.near_ball_pct:.0f}%" if np.isfinite(r.near_ball_pct) else "–",
                cmp(med.near_ball_pct, "{:.0f}", "%"),
            ),
        ]
    )
    if r.minutes < 5:
        obs = "<p class='sub'>Not enough time seen for observations (under 5 min).</p>"
    elif not tips:
        obs = (
            "<p class='sub'>Nothing stands out against others in the same role: work rate, position and "
            "involvement are all close to typical.</p>"
        )
    else:
        items = []
        for kind, text, ev in tips:
            cls = "strength" if "strength" in kind else ""
            items.append(
                f'<li class="{cls}"><span class="kind">{esc(kind)}.</span> {esc(text)}'
                f'<span class="ev">{esc(ev)}</span></li>'
            )
        obs = f'<ul class="obs">{"".join(items)}</ul>'

    def cell(key: str, fmt: str) -> str:
        v = r.get(key, np.nan)
        return "–" if pd.isna(v) else fmt.format(v)

    rows = "".join(
        f"<tr><td>{esc(label)}</td><td class='n'>{cell(key + '_min', '{:.1f}')}</td>"
        f"<td class='n'>{cell(key + '_rel', '{:.2f}')}</td><td class='n'>{cell(key + '_k', '{:.0f}')}</td></tr>"
        for key, label in PHASES
    )
    body = f"""
<p><a href="index.html">← Team</a></p>
<div class="private">Private: automatic analysis of a youth game, with names. Keep on this computer; do not post.</div>
<h1>{esc(name)} <span style="color:var(--text-3);font-weight:400">#{r.jersey}</span></h1>
<p class="sub">{esc(role.capitalize())} · first half {r.min_h1:.0f} min, second half {r.min_h2:.0f} min seen</p>
<h2>At a glance</h2>
<div class="tiles">{tiles}</div>
<h2>Observations</h2>
{obs}
<h2>Where they play</h2>
<p class="sub">Share of visible time in each part of the pitch, both halves together, always attacking to the
right. Median {r.abs_y:.0f} m from the centre line; ranges over {r.roam:.0f} m of pitch length.</p>
{heat}{legend}
<h2>Work rate through the game</h2>
<p class="sub">Distance per minute compared with identified teammates at the same time (1.0 = team median). The
whisker shows how far this could move by chance (±2 standard errors).</p>
{phase_svg(r)}
<details><summary>Table</summary><div class="wrap"><table>
<tr><th>Part of the game</th><th class="n">Minutes seen</th><th class="n">× teammates</th>
<th class="n">Windows</th></tr>{rows}</table></div></details>
<p class="note">Unverified automatic analysis of movement only: no passing, shooting or technique. Numbers are per
visible minute; the panning camera and automatic identity see about a third to a half of each player's time.
Check observations on video before acting on them.</p>
"""
    return page(f"{name} coaching", body)


def team_page(m: pd.DataFrame, tip_counts: dict, n_windows: int) -> str:
    rows = []
    order = {"goalkeeper": 0, "defender": 1, "midfielder": 2, "forward": 3}
    m = m.assign(_o=m.role.map(order)).sort_values(["_o", "minutes"], ascending=[True, False])
    for _, r in m.iterrows():
        few = r.minutes < 5
        name = r["name"] if isinstance(r["name"], str) else f"#{r.jersey}"
        near = f"{r.near_ball_pct:.0f}" if np.isfinite(r.near_ball_pct) else "–"
        rows.append(
            f'<tr{" class=few" if few else ""}><td><a href="player_{r.jersey:02d}.html">{esc(name)}</a>'
            f'{" <small>little data</small>" if few else ""}</td><td class="n">{r.jersey}</td>'
            f"<td>{esc(r.role)}</td><td class='n'>{r.minutes:.0f}</td><td class='n'>{r.m_per_min:.0f}</td>"
            f"<td class='n'>{r.work_rel:.2f}</td><td class='n'>{r.pct_fast:.1f}</td>"
            f"<td class='n'>{r.from_goal:.0f}</td><td class='n'>{near}</td>"
            f"<td class='n'>{tip_counts.get(r.jersey, 0)}</td></tr>"
        )
    body = f"""
<div class="private">Private: automatic analysis of a youth game, with names. Keep on this computer; do not post.</div>
<h1>Team</h1>
<p class="sub">{n_windows} five-minute windows, {len(m)} players, {m.minutes.sum():.0f} identified player-minutes.
Players are grouped by role (from how deep they play relative to the team). Click a name for their page.</p>
<div class="wrap"><table>
<tr><th>Player</th><th class="n">#</th><th>Role</th><th class="n">Min seen</th><th class="n">m/min</th>
<th class="n">× team</th><th class="n">Fast %</th><th class="n">From goal m</th><th class="n">Near ball %</th>
<th class="n">Observations</th></tr>
{"".join(rows)}</table></div>
<p class="note">Unverified automatic analysis of movement only. "× team" compares distance per minute with
identified teammates in the same windows. Check observations on video before acting on them.</p>
"""
    return page("Team coaching", body)
