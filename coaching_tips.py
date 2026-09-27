"""Coaching tips for every identified player, from movement and position across a game's windows.

Why: player_stats.py gives per-window numbers; a coach or parent needs a few concrete, comparable observations per
player. Ball events are too sparse to use (CLAUDE.md Phase 7), so tips come from four things the pipeline measures:
  - work rate: metres per minute and time running or faster, relative to teammates measured in the same windows
    (each window has its own noise floor, 17 to 29 m/min, which cancels in the ratio);
  - positioning: distance from our own goal, depth ahead of or behind the team line (median of the visible
    teammates at that moment), width, and how much the player roams;
  - fatigue: late vs early in each half and second half vs first, relative to teammates at the same time;
  - involvement: time within NEAR_BALL_M of the ball and distance to it, on detected ball positions only.
Each player gets a rough role from depth relative to the team line (deepest third defenders, highest third
forwards; the roster's goalkeeper) and is compared with the others in that role. A tip needs MIN_MINUTES of the
player seen and a difference past the thresholds below; every tip lists its evidence and a confidence.

Thresholds are hand-set guesses, and the running numbers are unverified against measured distances (CLAUDE.md
Phase 7). Tips are observations to check on video, not verdicts.

Needs data/game.local.json (halftime, our first-half goal end; see jersey_auto.py) and, per window, identity
(player_identity.csv, identity_segments.csv), tracklet_pitch_xy.csv.gz, tracklet_roles.csv, ball_path.csv and
pitch_anchors_ptz.local.json. Writes data/coaching/ (git-ignored: names of minors): per player a Markdown report
and a self-contained HTML page (coaching_html.py), plus index.html, team_overview.md and player_metrics.csv.

Example:
  python coaching_tips.py --runs data\\clipH,data\\clipI,data\\clipJ,data\\clipA,data\\clipE,data\\clipF,data\\clipG,...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import coaching_html
from jersey_auto import GAME_FILE, PITCH_CAMERA
from pitch_calibrate import Calibration
from player_stats import MAX_SPEED_MPS, ROSTER_FILE, SPEED_BANDS, identity_rows, smooth_steps
from sv_common import Cache, parse_time, require_under_data

OUT = Path(__file__).resolve().parent / "data" / "coaching"
NEAR_BALL_M = 10.0  # "near the ball"
BALL_MATCH_CI = 2  # a ball detection counts for samples within this many cached frames
MIN_MINUTES = 5.0  # seen at least this long for any tip
MIN_PHASE_MINUTES = 2.0  # each side of a fatigue comparison
WORK_DIFF, SPRINT_DIFF, INVOLVE_DIFF, FATIGUE_DIFF = 0.10, 0.5, 0.35, 0.10  # relative differences that make a tip
DEPTH_DIFF_M, WIDTH_DIFF_M, ROAM_DIFF = 5.0, 5.0, 0.35
SOLID_WINDOW_S = 60.0  # a window counts toward the noise estimate and the evidence with this much of the player
MIN_GAP_FAST_PP, MIN_GAP_BALL_PP = 1.5, 5.0  # and at least this many percentage points (small shares are noisy)
# work-rate and fatigue differences must also exceed 2 standard errors from the measured window-to-window spread


def window_samples(run: Path, game: dict, length: float) -> pd.DataFrame:
    """Identified target samples of one window, with speed, game time, position from our goal, team line, ball."""
    fps = float(json.loads((run / "best_config.json").read_text())["fps"])
    cache = Cache(run / "cache")
    start_min = parse_time(json.loads((run / "cache" / "meta.json").read_text())["clip_start"]) / 60
    xy = pd.read_csv(run / "tracklet_pitch_xy.csv.gz")
    d = smooth_steps(identity_rows(run, xy), fps)
    d["t_min"] = start_min + d.ci / cache.fps / 60
    second = d.t_min > game["halftime_min"]
    first_goal = float(game["first_half_our_goal_x"])
    toward = 1.0 if first_goal < length / 2 else -1.0  # +1: our first-half goal at X = 0
    d["half"] = np.where(second, 2, 1)
    sign = np.where(second, -toward, toward)
    d["from_goal"] = np.where(sign > 0, d.x_s, length - d.x_s)
    d["abs_y"] = d.y_s.abs()
    d["y_team"] = np.where(sign > 0, d.y_s, -d.y_s)  # mirrored with the ends, so a side stays the same side
    # team line: median position of every visible target-role player candidate at that moment (not just the named)
    roles = pd.read_csv(run / "tracklet_roles.csv")
    team_ids = roles[(roles.role == "target") & roles.player_candidate.astype(bool)].track_id
    team = xy[xy.track_id.isin(team_ids)].groupby("pf").agg(team_x=("X_m", "median"), team_n=("X_m", "size"))
    d = d.join(team, on="pf")
    d["team_from_goal"] = np.where(sign > 0, d.team_x, length - d.team_x)
    d["depth"] = np.where(d.team_n >= 4, d.from_goal - d.team_from_goal, np.nan)  # need a real line
    # ball: detected positions only (interpolated ones were often ghosts, Phase 4), carried to the pitch
    ball = pd.read_csv(run / "ball_path.csv")
    ball = ball[ball.kind == "detected"]
    anchors = json.loads((run / "pitch_anchors_ptz.local.json").read_text())["anchors"]
    bxy, _ = Calibration(cache, anchors).to_pitch(ball.ci.to_numpy(), ball.x.to_numpy(), ball.y.to_numpy())
    b = pd.DataFrame(dict(ci=ball.ci.to_numpy(), bx=bxy[:, 0], by=bxy[:, 1])).sort_values("ci")
    d = pd.merge_asof(d.sort_values("ci"), b, on="ci", direction="nearest", tolerance=BALL_MATCH_CI)
    d["ball_dist"] = np.hypot(d.X_m - d.bx, d.Y_m - d.by)
    d["run"] = run.name
    d["fps"] = fps
    return d


def phase(t_min: pd.Series, halftime: float, game_end: float) -> pd.Series:
    """early/late half of each half, split at each half's midpoint of covered time."""
    h1_mid, h2_mid = halftime / 2, (halftime + game_end) / 2
    return np.where(
        t_min <= halftime,
        np.where(t_min <= h1_mid, "h1_early", "h1_late"),
        np.where(t_min <= h2_mid, "h2_early", "h2_late"),
    )


def solid_k(q: pd.DataFrame) -> int:
    """Windows with at least SOLID_WINDOW_S of the player: the independent samples behind an average."""
    return int((q.n >= SOLID_WINDOW_S * q.fps).sum())


def metrics(s: pd.DataFrame, game: dict) -> tuple:
    """Per-player metrics and per-(player, run) relative work rates."""
    s = s[s.speed.isna() | (s.speed <= MAX_SPEED_MPS)].copy()
    s["phase"] = phase(s.t_min, game["halftime_min"], float(s.t_min.max()))
    s["fast"] = s.speed >= SPEED_BANDS["run"][0]
    s["sprint"] = s.speed >= SPEED_BANDS["fast"][0]
    # relative work rate: this player's mean speed / the median of identified teammates in the same window
    per = s.groupby(["jersey", "run"]).agg(v=("speed", "mean"), n=("speed", "size"), fps=("fps", "first"))
    team_v = per.groupby("run").v.median()
    per["rel"] = per.v / per.index.get_level_values("run").map(team_v)
    # window-to-window spread of a player's relative work rate (0.135 on this game): sets how big a difference
    # between phases, or from the team, has to be before it is a tip rather than noise
    solid = per[per.n >= SOLID_WINDOW_S * per.fps]  # a few seconds in a window give a wild ratio
    counts = solid.groupby(level="jersey").size()
    spread = solid.groupby(level="jersey").rel.var()[counts >= 4]
    rel_sd = float(np.sqrt(spread.mean())) if len(spread) else 0.15
    # the same ratio per (window, phase) and (window, half), from only the samples in that phase, against teammates
    # in the same window and phase: a window that straddles a boundary must not count on both sides
    s["half_name"] = "h" + s.half.astype(str)
    split = {}
    for col in ("phase", "half_name"):
        q = s.groupby(["jersey", "run", col]).agg(v=("speed", "mean"), n=("speed", "size"), fps=("fps", "first"))
        q["rel"] = q.v / q.groupby(["run", col]).v.transform("median")
        split[col] = q
    rows = []
    for j, g in s.groupby("jersey"):
        fps = g.fps.iloc[0]
        minutes = len(g) / fps / 60
        bv = g.ball_dist.notna()
        p = per.loc[j]
        rel = float(np.average(p.rel, weights=p.n))
        ph = {}
        for col, q in split.items():
            for name, qq in q.loc[j].groupby(level=col):
                ph[name] = (qq.n.sum() / fps / 60, float(np.average(qq.rel, weights=qq.n)), solid_k(qq))
        rows.append(
            dict(
                jersey=int(j),
                minutes=minutes,
                min_h1=(g.half == 1).sum() / fps / 60,
                min_h2=(g.half == 2).sum() / fps / 60,
                windows=g.run.nunique(),
                m_per_min=float(g.speed.mean() * 60),
                work_rel=rel,
                work_k=solid_k(p),
                rel_sd=rel_sd,
                pct_fast=100 * float(g.fast.mean()),
                pct_sprint=100 * float(g.sprint.mean()),
                from_goal=float(g.from_goal.median()),
                depth=float(g.depth.median()),
                abs_y=float(g.abs_y.median()),
                roam=float(g.from_goal.quantile(0.9) - g.from_goal.quantile(0.1)),
                ball_seen_min=bv.sum() / fps / 60,
                near_ball_pct=100 * float((g.ball_dist[bv] <= NEAR_BALL_M).mean()) if bv.any() else np.nan,
                ball_dist_med=float(g.ball_dist[bv].median()) if bv.any() else np.nan,
                **{f"{k}_min": v[0] for k, v in ph.items()},
                **{f"{k}_rel": v[1] for k, v in ph.items()},
                **{f"{k}_k": v[2] for k, v in ph.items()},
            )
        )
    m = pd.DataFrame(rows)
    roster = pd.read_csv(ROSTER_FILE)
    m = m.merge(roster[["jersey", "name", "goalkeeper"]], on="jersey", how="left")
    gk = m.goalkeeper.astype(str).str.lower() == "true"
    # roles by rank of depth vs the team line: the camera shows part of the team, which compresses depth, so a
    # fixed cutoff (+-7 m) made 12 of 19 midfielders. Thirds of the well-seen outfield players set the cut points.
    seen = m[~gk & (m.minutes >= MIN_MINUTES)].depth
    lo, hi = seen.quantile(1 / 3), seen.quantile(2 / 3)
    m["role"] = np.select([gk, m.depth <= lo, m.depth >= hi], ["goalkeeper", "defender", "forward"], "midfielder")
    return m


def confidence(minutes: float) -> str:
    return "high" if minutes >= 20 else "medium" if minutes >= 10 else "low"


def peer_median(r: pd.Series, outfield: pd.DataFrame) -> pd.Series:
    """Median of the other well-seen players in the same role (every outfield player if fewer than 2)."""
    others = outfield[(outfield.jersey != r.jersey) & (outfield.minutes >= MIN_MINUTES)]
    peer = others[others.role == r.role]
    if len(peer) < 2:  # too few in this role to compare with: use every outfield player
        peer = others
    return peer.median(numeric_only=True)


def tips_for(r: pd.Series, outfield: pd.DataFrame) -> list:
    """(kind, text, evidence) tips for one outfield player against the others in the same role."""
    out = []
    if r.minutes < MIN_MINUTES:
        return out
    med = peer_median(r, outfield)
    # work rate (already relative to teammates in the same windows; 1.0 = team median)
    d = r.work_rel - 1.0
    work_thr = max(WORK_DIFF, 2 * r.rel_sd / np.sqrt(max(r.work_k, 1)))
    if d <= -work_thr:
        out.append(
            (
                "Work rate",
                "Covers less ground than teammates in the same minutes. Keep moving off the ball: "
                "re-position after every pass, and track back when possession is lost.",
                f"{r.m_per_min:.0f} m/min, {100 * d:+.0f}% vs identified teammates in the same windows",
            )
        )
    elif d >= work_thr:
        out.append(
            (
                "Work rate (strength)",
                "Covers more ground than teammates in the same minutes.",
                f"{r.m_per_min:.0f} m/min, {100 * d:+.0f}% vs identified teammates in the same windows",
            )
        )
    gap_fast = abs(r.pct_fast - med.pct_fast) >= MIN_GAP_FAST_PP
    if gap_fast and med.pct_fast > 0 and r.pct_fast < (1 - SPRINT_DIFF) * med.pct_fast:
        out.append(
            (
                "Intensity",
                "Rarely runs at speed. Look for moments to burst: into space after passing, and "
                "to close down the ball carrier.",
                f"{r.pct_fast:.1f}% of visible time at 4 m/s or faster vs {med.pct_fast:.1f}% for {r.role}s",
            )
        )
    elif gap_fast and r.pct_fast > (1 + SPRINT_DIFF) * med.pct_fast:
        out.append(
            (
                "Intensity (strength)",
                "Often runs at speed compared with others in the same role.",
                f"{r.pct_fast:.1f}% of visible time at 4 m/s or faster vs {med.pct_fast:.1f}% for {r.role}s",
            )
        )
    # positioning, against the role's usual place
    if np.isfinite(r.depth) and np.isfinite(med.depth) and abs(r.depth - med.depth) >= DEPTH_DIFF_M:
        deeper = r.depth < med.depth
        text = (
            "Plays deeper than others in this role. Step up with the team line when we have the ball, so the "
            "team stays compact."
            if deeper
            else "Plays further forward than others in this role. Check the space behind when we lose the ball."
        )
        out.append(("Positioning", text, f"median {r.depth:+.0f} m vs the team line; {r.role}s {med.depth:+.0f} m"))
    if np.isfinite(med.abs_y) and abs(r.abs_y - med.abs_y) >= WIDTH_DIFF_M:
        wider = r.abs_y > med.abs_y
        text = (
            "Stays wide. Good for stretching the play; also come inside to support when the ball is central."
            if wider
            else "Stays central. Try drifting wide at times to open passing angles and stretch the defence."
        )
        out.append(("Positioning", text, f"median {r.abs_y:.0f} m from the centre line; {r.role}s {med.abs_y:.0f} m"))
    if med.roam > 0 and r.roam < (1 - ROAM_DIFF) * med.roam:
        out.append(
            (
                "Positioning",
                "Stays in a narrow band of the pitch. Move up and down more with the play.",
                f"most time within {r.roam:.0f} m of pitch length vs {med.roam:.0f} m for {r.role}s",
            )
        )

    # fatigue: late vs early in each half, relative to teammates at the same time
    def dropped(e: str, lt: str) -> float | None:
        """late - early relative work rate if both sides are seen enough and the drop is past noise, else None."""
        if min(r.get(f"{e}_min", 0) or 0, r.get(f"{lt}_min", 0) or 0) < MIN_PHASE_MINUTES:
            return None
        diff = r[f"{lt}_rel"] - r[f"{e}_rel"]
        noise = 2 * r.rel_sd * np.sqrt(1 / max(r[f"{e}_k"], 1) + 1 / max(r[f"{lt}_k"], 1))
        return diff if np.isfinite(diff) and diff <= -max(FATIGUE_DIFF, noise) else None

    for h in ("h1", "h2"):
        e, lt = f"{h}_early", f"{h}_late"
        diff = dropped(e, lt)
        if diff is None:
            continue
        half = "first" if h == "h1" else "second"
        out.append(
            (
                "Fatigue",
                f"Work rate drops late in the {half} half compared with teammates. Pace early efforts, and build "
                "match fitness.",
                f"relative work rate {r[f'{e}_rel']:.2f} early -> {r[f'{lt}_rel']:.2f} late "
                f"({100 * diff / r[f'{e}_rel']:+.0f}%)",
            )
        )
    if dropped("h1", "h2") is not None:
        out.append(
            (
                "Fatigue",
                "Less active in the second half than the first, compared with teammates.",
                f"relative work rate {r.h1_rel:.2f} first half -> {r.h2_rel:.2f} second half",
            )
        )
    # involvement, on detected ball positions only
    if r.ball_seen_min >= MIN_PHASE_MINUTES and np.isfinite(med.near_ball_pct) and med.near_ball_pct > 0:
        gap_ball = abs(r.near_ball_pct - med.near_ball_pct) >= MIN_GAP_BALL_PP
        if gap_ball and r.near_ball_pct < (1 - INVOLVE_DIFF) * med.near_ball_pct:
            out.append(
                (
                    "Involvement",
                    "Is near the ball less often than others in this role. Move to offer a "
                    "passing option, and get closer when the ball is on your side.",
                    f"within {NEAR_BALL_M:.0f} m of the ball {r.near_ball_pct:.0f}% of the time vs "
                    f"{med.near_ball_pct:.0f}% for {r.role}s; median {r.ball_dist_med:.0f} m away",
                )
            )
        elif gap_ball and r.near_ball_pct > (1 + INVOLVE_DIFF) * med.near_ball_pct:
            out.append(
                (
                    "Involvement (strength)",
                    "Is often close to the ball compared with others in this role.",
                    f"within {NEAR_BALL_M:.0f} m of the ball {r.near_ball_pct:.0f}% of the time vs "
                    f"{med.near_ball_pct:.0f}% for {r.role}s",
                )
            )
    return out


def report(r: pd.Series, tips: list, n_windows: int) -> str:
    name = r["name"] if isinstance(r["name"], str) else f"#{r.jersey}"
    lines = [
        f"# {name} (#{r.jersey}) - {r.role}",
        "",
        f"Seen {r.minutes:.1f} min in {r.windows} of {n_windows} windows (first half {r.min_h1:.1f}, second half "
        f"{r.min_h2:.1f}). Confidence: {confidence(r.minutes)}. Visibility is limited by the panning camera and by "
        "automatic identity (about 35 to 55% of the team's tracked time is named); numbers are per visible minute.",
        "",
        "## Numbers",
        f"- Work rate: {r.m_per_min:.0f} m/min; {r.work_rel:.2f} x the identified teammates in the same windows",
        f"- Time at 4 m/s or faster: {r.pct_fast:.1f}% (5.5 m/s or faster: {r.pct_sprint:.1f}%)",
        f"- Position: median {r.from_goal:.0f} m from our goal, {r.depth:+.0f} m vs the team line, {r.abs_y:.0f} m "
        f"from the centre line; ranges over {r.roam:.0f} m of pitch length",
        f"- Near the ball (within {NEAR_BALL_M:.0f} m): {r.near_ball_pct:.0f}% of the {r.ball_seen_min:.1f} min "
        "when the ball was detected",
        "",
        "## Observations",
    ]
    if r.minutes < MIN_MINUTES:
        lines.append(f"- Not enough time seen for observations (under {MIN_MINUTES:.0f} min).")
    elif not tips:
        lines.append(
            "- Nothing stands out against others in the same role: work rate, position and involvement are "
            "all close to typical."
        )
    for kind, text, ev in tips:
        lines.append(f"- **{kind}.** {text} _({ev})_")
    lines += [
        "",
        "_Unverified automatic analysis of movement only (no passing, shooting or technique). Check "
        "observations against video before acting on them._",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="comma list of window folders with identity")
    args = ap.parse_args()
    runs = [require_under_data(Path(r)) for r in args.runs.split(",")]
    game = json.loads(GAME_FILE.read_text())
    length = json.loads(PITCH_CAMERA.read_text())["length_m"]
    s = pd.concat([window_samples(r, game, length) for r in runs], ignore_index=True)
    m = metrics(s, game)
    OUT.mkdir(parents=True, exist_ok=True)
    m.round(3).to_csv(OUT / "player_metrics.csv", index=False)
    outfield = m[m.role != "goalkeeper"]
    lines = [
        "# Team overview",
        "",
        f"{len(runs)} windows, {len(m)} players, {m.minutes.sum():.0f} identified "
        "player-minutes. Unverified automatic analysis of movement only.",
        "",
        "| player | role | min | m/min | vs team | fast % | from goal m | depth m | near ball % | observations |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    tip_counts = {}
    for _, r in m.sort_values("minutes", ascending=False).iterrows():
        keeper = r.role == "goalkeeper"  # no peers to compare with: numbers only
        tips = [] if keeper else tips_for(r, outfield)
        tip_counts[r.jersey] = len(tips)
        (OUT / f"player_{r.jersey:02d}.md").write_text(report(r, tips, len(runs)), encoding="utf-8")
        med = pd.Series(dtype=float, index=m.columns).astype(float) if keeper else peer_median(r, outfield)
        page = coaching_html.player_page(
            r, tips, med, s[s.jersey == r.jersey], length, len(runs), confidence(r.minutes)
        )
        (OUT / f"player_{r.jersey:02d}.html").write_text(page, encoding="utf-8")
        name = r["name"] if isinstance(r["name"], str) else f"#{r.jersey}"
        lines.append(
            f"| {name} | {r.role} | {r.minutes:.1f} | {r.m_per_min:.0f} | {r.work_rel:.2f} | "
            f"{r.pct_fast:.1f} | {r.from_goal:.0f} | {r.depth:+.0f} | {r.near_ball_pct:.0f} | {len(tips)} |"
        )
    (OUT / "team_overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "index.html").write_text(coaching_html.team_page(m, tip_counts, len(runs)), encoding="utf-8")
    print(f"wrote {len(m)} player pages (.html and .md), index.html, team_overview.md, player_metrics.csv to {OUT}")
    print(
        m[["jersey", "role", "minutes", "work_rel", "pct_fast", "depth", "abs_y", "near_ball_pct"]]
        .round(2)
        .sort_values("minutes", ascending=False)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
