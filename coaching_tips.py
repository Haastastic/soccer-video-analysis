"""Coaching tips for every identified player, from movement and position across a game's windows.

Why: player_stats.py gives per-window numbers; a coach or parent needs a few concrete, comparable observations per
player. Ball events are too sparse to use (CLAUDE.md Phase 7), so tips come from four things the pipeline measures:
  - work rate: metres per minute and time running or faster, relative to teammates measured in the same windows
    (each window has its own noise floor, 17 to 29 m/min, which cancels in the ratio);
  - positioning: distance from our own goal, depth ahead of or behind the team line (median of the visible
    teammates at that moment), width, and how much the player roams;
  - fatigue: late vs early in each half and second half vs first, relative to teammates at the same time;
  - involvement: time within NEAR_BALL_M of the ball and distance to it, on detected ball positions only;
  - ball events (events.py, retuned on owner labels in CLAUDE.md Phase 12): touches (gaining the ball) per visible
    minute, against the touches identified teammates made per minute in the same windows (each window's ball
    detector quality cancels), and share of visible time on the ball. Every event counts, whatever its confidence:
    the retune found no confidence cut that helped. Passes and turnovers are too unreliable to use.
Each player gets a rough role from depth relative to the team line (deepest third defenders, highest third
forwards; the roster's goalkeeper) and is compared with the others in that role. A tip needs MIN_MINUTES of the
player seen and a difference past the thresholds below; every tip lists its evidence and a confidence.

Thresholds are hand-set guesses, and the running numbers are unverified against measured distances (CLAUDE.md
Phase 7). Tips are observations to check on video, not verdicts.

Several games (--runs from more than one game folder): each game is measured on its own and pooled, a player is
compared with the others in their role in each game (roles can change between games) and over the pooled games,
and each observation says whether it holds in every game the player was seen enough in, only with the games
pooled, in one game only, or points opposite ways in two games. Output goes to data/coaching_games/ (local only).
Games are labeled by the date in game.local.json's "video", else the folder.

Needs data/game.local.json (halftime, our first-half goal end; see jersey_auto.py) and, per window, identity
(player_identity.csv, identity_segments.csv), tracklet_pitch_xy.csv.gz, tracklet_roles.csv, ball_path.csv and
pitch_anchors_ptz.local.json. Writes <game folder>/coaching/ (data/coaching for the first game; git-ignored:
names of minors): per player a Markdown report and a self-contained HTML page (coaching_html.py), plus
index.html, team_overview.md and player_metrics.csv.

Examples:
  python coaching_tips.py --runs data\\clipH,data\\clipI,data\\clipJ,data\\clipA,data\\clipE,data\\clipF,data\\clipG,...
  python coaching_tips.py --runs data\\clipA,...,data\\g0922\\w0000,...     (two games -> data\\coaching_games)
"""

import argparse
import json
import os
import re
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

import coaching_html
from jersey_auto import GAME_FILE, PITCH_CAMERA
from pitch_calibrate import Calibration
from player_stats import MAX_SPEED_MPS, SPEED_BANDS, identity_rows, smooth_steps
from sv_common import (
    DATA_DIR,
    DEFAULT_TEAM,
    Cache,
    game_dir,
    game_file,
    grade,
    parse_time,
    play_mask,
    read_roster,
    require_under_data,
    team_of,
)

NEAR_BALL_M = 10.0  # "near the ball"
BALL_MATCH_CI = 2  # a ball detection counts for samples within this many cached frames
MIN_MINUTES = 5.0  # seen at least this long for any tip
MIN_PHASE_MINUTES = 2.0  # each side of a fatigue comparison
WORK_DIFF, SPRINT_DIFF, INVOLVE_DIFF, FATIGUE_DIFF = 0.10, 0.5, 0.35, 0.10  # relative differences that make a tip
DEPTH_DIFF_M, WIDTH_DIFF_M, ROAM_DIFF = 5.0, 5.0, 0.35
SOLID_WINDOW_S = 60.0  # a window counts toward the noise estimate and the evidence with this much of the player
MIN_GAP_FAST_PP, MIN_GAP_BALL_PP = 1.5, 5.0  # and at least this many percentage points (small shares are noisy)
# work-rate and fatigue differences must also exceed 2 standard errors from the measured window-to-window spread
TOUCH_DIFF, TOUCH_P, MIN_EXPECTED = 0.35, 0.025, 5.0  # touches vs the role: relative gap, one-sided Poisson p, and
# at least this many touches expected (fewer cannot show a difference)
MULTI_OUT = DATA_DIR / "coaching_games"  # the first team's; another team's is coaching_games_<team> (multi_out)


def multi_out(team: str) -> Path:
    return MULTI_OUT if team == DEFAULT_TEAM else DATA_DIR / f"coaching_games_{team}"


def window_samples(run: Path, game: dict, length: float) -> pd.DataFrame:
    """Identified target samples of one window, with speed, game time, position from our goal, team line, ball."""
    fps = float(json.loads((run / "best_config.json").read_text())["fps"])
    cache = Cache(run / "cache")
    start_min = parse_time(json.loads((run / "cache" / "meta.json").read_text())["clip_start"]) / 60
    xy = pd.read_csv(run / "tracklet_pitch_xy.csv.gz")
    ident = identity_rows(run, xy)
    d = smooth_steps(ident[play_mask(run, ident.ci)], fps)  # time outside the game's halves is not play
    d["t_min"] = start_min + d.ci / cache.fps / 60
    d["video_s"] = d.t_min * 60
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


def window_events(run: Path) -> pd.DataFrame:
    """Touches and possessions of identified players (player_stats.py's player_events.csv), every confidence."""
    path = run / "player_events.csv"
    if not path.exists():  # nobody identified in the window (e.g. after the final whistle): no events
        return pd.DataFrame(columns=["run", "jersey", "type", "time_s", "dur_s", "confidence", "video_s"])
    ev = pd.read_csv(path)
    ev = ev[ev.type.isin(["touch", "possession"]) & ev.jersey.notna() & play_mask(run, ev.ci)].copy()
    ev["jersey"] = ev.jersey.astype(int)
    ev["dur_s"] = np.where(ev.type == "possession", ev.end_s - ev.time_s, 0.0)
    ev["run"] = run.name
    ev["video_s"] = parse_time(json.loads((run / "cache" / "meta.json").read_text())["clip_start"]) + ev.time_s
    return ev[["run", "jersey", "type", "time_s", "dur_s", "confidence", "video_s"]]


def game_label(folder: Path, game: dict) -> str:
    """The date in the game's video name, else the folder name."""
    m = re.search(r"\d{4}-\d{2}-\d{2}", str(game.get("video", "")))
    return m.group(0) if m else folder.name


def load_game(runs: list) -> dict:
    """One game's samples (with phase), events and pitch size. Window keys are prefixed by the game label, so two
    games' windows never share a key."""
    folder = game_dir(runs[0])
    game = json.loads(game_file(runs[0], GAME_FILE).read_text())
    cam = json.loads(game_file(runs[0], PITCH_CAMERA).read_text())
    label = game_label(folder, game)
    length, width = cam["length_m"], cam["touchline_near_y"] - cam["touchline_far_y"]
    s = pd.concat([window_samples(r, game, length) for r in runs], ignore_index=True)
    s["phase"] = phase(s.t_min, game["halftime_min"], float(s.t_min.max()))
    ev = pd.concat([window_events(r) for r in runs], ignore_index=True)
    for d in (s, ev):
        d["run"] = label + "/" + d.run
        d["game"] = label
    return dict(
        folder=folder,
        label=label,
        runs=runs,
        length=length,
        width=width,
        s=s,
        ev=ev,
        team=team_of(runs[0]),
        roster=read_roster(team_of(runs[0]), runs[0]),
        video=game.get("video"),
    )


WATCH_N, WATCH_GAP_S, WATCH_LEAD_S = 5, 15.0, 3.0  # moments per list, at least this far apart, played from before


def spaced(t: np.ndarray, order: np.ndarray) -> list:
    """Indices in priority order, skipping any within WATCH_GAP_S of one already taken, up to WATCH_N."""
    taken = []
    for i in order:
        if all(abs(t[i] - t[j]) >= WATCH_GAP_S for j in taken):
            taken.append(i)
            if len(taken) == WATCH_N:
                break
    return sorted(taken, key=lambda i: t[i])


def player_moments(g: dict, jersey) -> list:
    return watch_moments(g["s"][g["s"].jersey == jersey], g["ev"][g["ev"].jersey == jersey])


def watch_moments(s: pd.DataFrame, ev: pd.DataFrame) -> list:
    """One player's moments in one game to check on video: the longest stretches on camera, the fastest running and
    the longest possessions and touches. Each is (video seconds to start at, caption)."""
    groups = []
    if len(s):
        q = s.sort_values(["run", "track_id", "part", "video_s"])
        new = (q.run != q.run.shift()) | (q.track_id != q.track_id.shift()) | (q.part != q.part.shift())
        new |= q.video_s.diff() > 1.0
        st = q.groupby(new.cumsum()).video_s.agg(["min", "max"])
        st = st.assign(dur=st["max"] - st["min"])
        st = st[st.dur >= 10].reset_index(drop=True)
        t, dur = st["min"].to_numpy(), st.dur.to_numpy()
        groups.append(("Longest on camera", [(t[i], f"{dur[i]:.0f} s") for i in spaced(t, np.argsort(-dur))]))
        f = s[(s.speed >= SPEED_BANDS["run"][0]) & (s.speed <= MAX_SPEED_MPS)].reset_index(drop=True)
        t, v = f.video_s.to_numpy(), f.speed.to_numpy()
        fast = [(t[i] - WATCH_LEAD_S, f"{v[i]:.1f} m/s") for i in spaced(t, np.argsort(-v))]
        groups.append(("Fastest running", fast))
    if len(ev):
        e = ev.reset_index(drop=True)
        t = e.video_s.to_numpy()
        prio = np.lexsort((-e.confidence.to_numpy(), -e.dur_s.to_numpy()))  # longest possessions, then touches
        cap = np.where(e.type == "possession", [f"has it {d:.1f} s" for d in e.dur_s], "touch")
        groups.append(("On the ball", [(t[i] - WATCH_LEAD_S, cap[i]) for i in spaced(t, prio)]))
    return groups


def video_href(video, page_dir: Path) -> str | None:
    """The game video relative to a page (local pages only), or None when it is not on this computer."""
    if not video or not (DATA_DIR.parent / video).exists():
        return None
    rel = os.path.relpath(DATA_DIR.parent / video, page_dir)
    return urllib.parse.quote(rel.replace(os.sep, "/"))


def pooled_roster(gs: list) -> pd.DataFrame:
    """One roster for several games: every game's players, the latest game's entry for a jersey listed in several."""
    rs = [g["roster"] for g in sorted(gs, key=lambda g: g["label"])]
    return pd.concat(rs, ignore_index=True).drop_duplicates("jersey", keep="last").reset_index(drop=True)


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


def touch_expectation(per: pd.DataFrame, ev: pd.DataFrame, keepers: set) -> pd.DataFrame:
    """Adds touches, poss_s and exp_touch per (jersey, run): exp_touch is the player's minutes times the rate of
    touches per minute of the other identified outfield players in that window, so each window's ball detection
    (game 2's detector misses about a third of balls, Phase 12) cancels. Keepers do not count toward the rate."""
    per = per.copy()
    per["mins"] = per.n / per.fps / 60
    per["touches"] = ev[ev.type == "touch"].groupby(["jersey", "run"]).size().reindex(per.index).fillna(0)
    poss = ev[ev.type == "possession"].groupby(["jersey", "run"]).dur_s.sum()
    per["poss_s"] = poss.reindex(per.index).fillna(0)
    field = ~per.index.get_level_values("jersey").isin(keepers)
    runs = per.index.get_level_values("run")
    tot_t = per.touches.where(field, 0).groupby(level="run").sum()
    tot_m = per.mins.where(field, 0).groupby(level="run").sum()
    others_t = runs.map(tot_t).to_numpy() - np.where(field, per.touches, 0)
    others_m = runs.map(tot_m).to_numpy() - np.where(field, per.mins, 0)
    per["exp_touch"] = np.where(others_m > 0, per.mins * others_t / np.maximum(others_m, 1e-9), 0.0)
    return per


def metrics(
    s: pd.DataFrame, ev: pd.DataFrame, team: str = DEFAULT_TEAM, roster: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per-player metrics. s needs a phase column (load_game). The grade (Freshman to Senior) is for the school
    year of the latest game in s."""
    s = s[s.speed.isna() | (s.speed <= MAX_SPEED_MPS)].copy()
    s["jersey"] = s.jersey.astype(int)
    s["fast"] = s.speed >= SPEED_BANDS["run"][0]
    s["sprint"] = s.speed >= SPEED_BANDS["fast"][0]
    roster = read_roster(team) if roster is None else roster
    keepers = set(roster.jersey[roster.goalkeeper])
    # relative work rate: this player's mean speed / the median of identified teammates in the same window
    per = s.groupby(["jersey", "run"]).agg(v=("speed", "mean"), n=("speed", "size"), fps=("fps", "first"))
    team_v = per.groupby("run").v.median()
    per["rel"] = per.v / per.index.get_level_values("run").map(team_v)
    per = touch_expectation(per, ev, keepers)
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
                touches=int(p.touches.sum()),
                touches_per_min=float(p.touches.sum() / minutes),
                exp_touch=float(p.exp_touch.sum()),
                touch_rel=float(p.touches.sum() / p.exp_touch.sum()) if p.exp_touch.sum() > 0 else np.nan,
                on_ball_pct=float(100 * p.poss_s.sum() / (minutes * 60)),
                **{f"{k}_min": v[0] for k, v in ph.items()},
                **{f"{k}_rel": v[1] for k, v in ph.items()},
                **{f"{k}_k": v[2] for k, v in ph.items()},
            )
        )
    m = pd.DataFrame(rows)
    m = m.merge(roster[["jersey", "name", "goalkeeper", "class_of"]], on="jersey", how="left")
    when = pd.to_datetime(pd.Series(s.game.unique()), errors="coerce").max() if "game" in s else pd.NaT
    when = pd.Timestamp.today() if pd.isna(when) else when
    m["grade"] = [grade(c, when.year, when.month) for c in m.class_of]
    gk = m.jersey.isin(keepers)
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
    # touches: count against what teammates' rate in the same windows predicts, scaled to the role's usual ratio
    expect = r.exp_touch * med.touch_rel if np.isfinite(med.touch_rel) else np.nan
    if np.isfinite(expect) and expect >= MIN_EXPECTED and np.isfinite(r.touch_rel):
        gap = r.touch_rel / med.touch_rel - 1
        ev = (
            f"{r.touches} touches in {r.minutes:.0f} min seen ({r.touches_per_min:.2f}/min); {r.touch_rel:.2f} x "
            f"teammates in the same windows vs {med.touch_rel:.2f} for {r.role}s"
        )
        if gap <= -TOUCH_DIFF and poisson.cdf(r.touches, expect) < TOUCH_P:
            out.append(
                (
                    "Touches",
                    "Gets on the ball less often than others in this role. Show for the ball: find space away from "
                    "a marker and call for it.",
                    ev,
                )
            )
        elif gap >= TOUCH_DIFF and poisson.sf(r.touches - 1, expect) < TOUCH_P:
            out.append(("Touches (strength)", "Gets on the ball more often than others in this role.", ev))
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
        f"- Touches (gaining the ball): {r.touches} ({r.touches_per_min:.2f} per visible minute), "
        f"{r.touch_rel:.2f} x identified teammates in the same windows; on the ball {r.on_ball_pct:.1f}% of visible "
        "time. events.py finds about 6 in 10 real touches, fewer where the ball detector struggles",
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
        "_Unverified automatic analysis of movement and touches only (no passing, shooting or technique). Check "
        "observations against video before acting on them._",
        "",
    ]
    return "\n".join(lines)


def name_of(r: pd.Series) -> str:
    return r["name"] if isinstance(r["name"], str) else f"#{r.jersey}"


def player_tips(m: pd.DataFrame) -> dict:
    """jersey -> (tips, peer median) for every player; keepers get numbers only (no peers to compare with)."""
    outfield = m[m.role != "goalkeeper"]
    out = {}
    for _, r in m.iterrows():
        if r.role == "goalkeeper":
            out[r.jersey] = ([], pd.Series(dtype=float, index=m.columns).astype(float))
        else:
            out[r.jersey] = (tips_for(r, outfield), peer_median(r, outfield))
    return out


def single_game(g: dict) -> None:
    """The one-game pages in <game folder>/coaching/."""
    out = g["folder"] / "coaching"
    s, runs = g["s"], g["runs"]
    m = metrics(s, g["ev"], g["team"], g["roster"])
    out.mkdir(parents=True, exist_ok=True)
    m.round(3).to_csv(out / "player_metrics.csv", index=False)
    lines = [
        "# Team overview",
        "",
        f"{len(runs)} windows, {len(m)} players, {m.minutes.sum():.0f} identified "
        "player-minutes. Unverified automatic analysis of movement and touches.",
        "",
        "| player | role | min | m/min | vs team | fast % | from goal m | depth m | near ball % | touches/min | "
        "observations |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    tips_med = player_tips(m)
    tip_counts = {}
    for _, r in m.sort_values("minutes", ascending=False).iterrows():
        tips, med = tips_med[r.jersey]
        tip_counts[r.jersey] = len(tips)
        (out / f"player_{r.jersey:02d}.md").write_text(report(r, tips, len(runs)), encoding="utf-8")
        href = video_href(g["video"], out)
        watch = coaching_html.watch_section({g["label"]: (href, player_moments(g, r.jersey))} if href else {})
        page = coaching_html.player_page(
            r,
            tips,
            med,
            s[s.jersey == r.jersey],
            g["length"],
            g["width"],
            len(runs),
            confidence(r.minutes),
            watch=watch,
        )
        (out / f"player_{r.jersey:02d}.html").write_text(page, encoding="utf-8")
        lines.append(
            f"| {name_of(r)} | {r.role} | {r.minutes:.1f} | {r.m_per_min:.0f} | {r.work_rel:.2f} | "
            f"{r.pct_fast:.1f} | {r.from_goal:.0f} | {r.depth:+.0f} | {r.near_ball_pct:.0f} | "
            f"{r.touches_per_min:.2f} | {len(tips)} |"
        )
    (out / "team_overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "index.html").write_text(coaching_html.team_page(m, tip_counts, len(runs)), encoding="utf-8")
    print(f"wrote {len(m)} player pages (.html and .md), index.html, team_overview.md, player_metrics.csv to {out}")
    cols = ["jersey", "role", "minutes", "work_rel", "pct_fast", "depth", "near_ball_pct", "touch_rel"]
    print(m[cols].round(2).sort_values("minutes", ascending=False).to_string(index=False))


def topic(kind: str, text: str) -> str:
    """What a tip is about, without its direction: two tips on one topic with different text point opposite ways."""
    base = kind.replace(" (strength)", "")
    if base == "Positioning":
        if text.startswith("Plays"):
            return "Positioning depth"
        return "Positioning width" if text.startswith(("Stays wide", "Stays central")) else "Positioning roam"
    if base == "Fatigue":
        return f"Fatigue {text}"  # each fatigue tip is its own comparison, with no opposite
    return base


STATUS = [
    ("every", "In every game"),
    ("pooled", "With the games pooled"),
    ("one", "In one game only"),
    ("differs", "Differs between games"),
    ("single", "Only one game seen enough"),
]


def tag_tips(pooled: list, per_game: dict) -> list:
    """Every observation with where it holds. per_game: label -> tips, for the games the player was seen enough in
    (MIN_MINUTES). Two games showing one topic in opposite directions make it "differs"."""
    found = {}  # (kind, text) -> observation
    for kind, text, ev in pooled:
        found[(kind, text)] = dict(kind=kind, text=text, evidence=[("Pooled", ev)], games=[], pooled=True)
    for label, tips in per_game.items():
        for kind, text, ev in tips:
            f = found.setdefault((kind, text), dict(kind=kind, text=text, evidence=[], games=[], pooled=False))
            f["games"].append(label)
            f["evidence"].append((label, ev))
    texts = {}  # topic -> the directions games showed
    for f in found.values():
        if f["games"]:
            texts.setdefault(topic(f["kind"], f["text"]), set()).add(f["text"])
    for f in found.values():
        if len(texts.get(topic(f["kind"], f["text"]), ())) > 1:
            f["status"] = "differs"
        elif len(per_game) < 2:
            f["status"] = "single"
        elif len(f["games"]) == len(per_game):
            f["status"] = "every"
        elif f["pooled"]:
            f["status"] = "pooled"
        else:
            f["status"] = "one"
    order = {k: i for i, (k, _) in enumerate(STATUS)}
    return sorted(found.values(), key=lambda f: (order[f["status"]], f["kind"]))


def multi_game(gs: list, share_dir: Path | None = None) -> None:
    """Pages combining several games in multi_out(team): pooled and per-game numbers, observations tagged by game."""
    gs = sorted(gs, key=lambda g: g["label"])
    out = multi_out(gs[0]["team"])
    s = pd.concat([g["s"] for g in gs], ignore_index=True)
    ev = pd.concat([g["ev"] for g in gs], ignore_index=True)
    m = metrics(s, ev, gs[0]["team"], pooled_roster(gs))
    # each game keeps its own roles: a player moved to another position in one game is compared with that
    # game's players in the same role, not with the role pooled over games
    per = {g["label"]: metrics(g["s"], g["ev"], g["team"], g["roster"]) for g in gs}
    out.mkdir(parents=True, exist_ok=True)
    table = pd.concat([m.assign(scope="pooled")] + [q.assign(scope=k) for k, q in per.items()])
    table.round(3).to_csv(out / "player_metrics.csv", index=False)
    tm = {"Pooled": player_tips(m), **{k: player_tips(q) for k, q in per.items()}}
    summary, tagged_all = [], {}
    for _, r in m.sort_values("minutes", ascending=False).iterrows():
        rows = {k: q.set_index("jersey").loc[r.jersey] for k, q in per.items() if r.jersey in set(q.jersey)}
        enough = {k: tm[k][r.jersey][0] for k, q in rows.items() if q.minutes >= MIN_MINUTES}
        keeper = r.role == "goalkeeper"
        tagged = [] if keeper or r.minutes < MIN_MINUTES else tag_tips(tm["Pooled"][r.jersey][0], enough)
        tagged_all[r.jersey] = tagged
        meds = {k: tm[k][r.jersey][1] for k in ["Pooled", *rows]}
        samples = {g["label"]: (g["s"][g["s"].jersey == r.jersey], g["length"], g["width"]) for g in gs}
        watch = coaching_html.watch_section(
            {
                g["label"]: (href, player_moments(g, r.jersey))
                for g in gs
                if g["label"] in rows and (href := video_href(g["video"], out))
            }
        )
        page = coaching_html.multi_player_page(r, rows, meds, tagged, samples, confidence(r.minutes), watch=watch)
        (out / f"player_{r.jersey:02d}.html").write_text(page, encoding="utf-8")
        seen = ", ".join(f"{k}: {q.minutes:.1f} min" for k, q in rows.items())
        summary += [f"## {name_of(r)} (#{r.jersey}), {r.role}", f"{seen}; pooled {r.minutes:.1f} min"]
        for f in tagged:
            summary.append(f"- [{dict(STATUS)[f['status']]}] **{f['kind']}.** {f['text']}")
            summary += [f"  - {k}: {e}" for k, e in f["evidence"]]
        summary.append("")
    head = [
        "# Players across games",
        "",
        f"{len(gs)} games ({', '.join(g['label'] for g in gs)}), {len(m)} players. Unverified automatic analysis of "
        "movement and touches. Observations are tagged by where they hold.",
        "",
    ]
    (out / "team_overview.md").write_text("\n".join(head + summary), encoding="utf-8")
    page = coaching_html.multi_team_page(m, per, tagged_all, [g["label"] for g in gs])
    (out / "index.html").write_text(page, encoding="utf-8")
    counts = pd.Series([f["status"] for t in tagged_all.values() for f in t], dtype=str).value_counts()
    print(f"wrote {len(m)} player pages, index.html, team_overview.md, player_metrics.csv to {out}")
    print("observations by status:", counts.to_dict())
    if share_dir is not None:
        write_share(share_dir, per, [g["label"] for g in gs], counts)


def write_share(share_dir: Path, per: dict, labels: list, counts: pd.Series) -> None:
    """Aggregates only (CLAUDE.md ground rules: no names, numbers or images): team numbers per game and how many
    observations hold in every game. Safe to share outside this computer."""
    share_dir.mkdir(parents=True, exist_ok=True)
    out = dict(
        note="Unverified automatic analysis of youth soccer video: team aggregates only, no players named.",
        games=coaching_html.season_rows(per, labels),
        observations_by_status={str(k): int(v) for k, v in counts.items()},
    )
    (share_dir / "season_summary.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {share_dir / 'season_summary.json'} (aggregates only)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="comma list of window folders with identity (one or more games)")
    ap.add_argument("--share-dir", type=Path, default=None, help="several games: also write season_summary.json there "
                    "(team aggregates only, no names or numbers)")  # fmt: skip
    args = ap.parse_args()
    by_game = {}
    for r in args.runs.split(","):
        run = require_under_data(Path(r))
        by_game.setdefault(game_dir(run), []).append(run)
    gs = [load_game(rs) for rs in by_game.values()]
    if len({g["label"] for g in gs}) != len(gs):
        raise SystemExit('two games share a label: give each game.local.json a "video" name with its date')
    if len({g["team"] for g in gs}) > 1:
        raise SystemExit("these windows are from more than one team: run each team's games on their own")
    if len(gs) == 1:
        single_game(gs[0])
    else:
        multi_game(gs, args.share_dir)


if __name__ == "__main__":
    main()
