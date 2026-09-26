"""Per-player stats (step 8, first part): running, position and ball events, each with confidence and visibility.

Identity comes from jersey_label.py apply: player_identity.csv (a whole tracklet's jersey) and identity_segments.csv
(named parts of tracklets split where another person starts; the frames around a switch belong to no one). Only
roster (target team and goalkeeper) players are identified, so stats are for them.

Running stats use tracklet_pitch_xy.csv.gz (pitch_calibrate.py) smoothed over SMOOTH_S: raw frame-to-frame positions
jitter far more than players move (raw speed p99 was 22 to 31 m/s on the two validation clips; 1 s smoothing
brings the 90th percentile to 4 to 5 m/s). Positions are only differenced within one continuous run of one
identity (same tracklet and part, no frame gap), never across a gap. Running stats are per visible minute
(CLAUDE.md: off-ball players are often out of frame).

  visible_s, visibility_pct   identified time on camera, and its share of the clip
  distance_m, m_per_min       smoothed distance, and per visible minute
  speed_p95_mps               95th percentile of smoothed speed (not the maximum: noise dominates that)
  pct_walk/jog/run/fast       share of visible time in speed bands SPEED_BANDS (guesses for youth players)
  x_med_m, y_med_m            median pitch position (X from the calibration's reference goal, Y across the field;
                              the direction of attack is not known, so no zones)
  pos_err_m                   estimated position error (from pitch_calibrate.py's cross-anchor check and the time
                              from the nearest anchor): drift, so it biases position far more than distance
  running_confidence          min(1, visible_s / FULL_CONF_S) x share of rows with pos_err_m <= GOOD_ERR_M
  possessions, touches, passes_made, passes_received, turnovers_lost, turnovers_won
                              trusted events (confidence >= 0.5); *_unconfirmed counts the rest (review_queue.csv)
  event_confidence            mean confidence of the player's trusted events

stats_report.json also has the noise floor: the same smoothing on tracklets flagged sideline_suspect (people
standing still off the play), i.e. the distance per minute a motionless person appears to cover. Read m_per_min
against it.

Outputs (git-ignored, they name minors): RUN/player_stats.csv, RUN/player_events.csv, RUN/stats_report.json, and
--db (default data/stats.sqlite) with every run given: tables clips, player_clip_stats, player_events.
--share-dir writes team aggregates only (no names, no jersey numbers), per the ground rules.

Example:
  python player_stats.py --runs data\\clipA,data\\clipB
  python player_stats.py --runs data\\clipA,data\\clipB --share-dir <folder>
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from sv_common import DATA_DIR, Cache, require_under_data

ROSTER_FILE = Path(__file__).resolve().parent / "roster.csv"
SMOOTH_S = 2.0  # 1 s left a 33 to 44 m/min noise floor on still people; 2 s: 24 to 34 (see stats_report.json)
MAX_SPEED_MPS = 10.0  # faster steps are position glitches; they are dropped and counted
SPEED_BANDS = {"walk": (0, 2), "jog": (2, 4), "run": (4, 5.5), "fast": (5.5, MAX_SPEED_MPS)}  # m/s, youth guesses
FULL_CONF_S = 120  # visible seconds for full confidence in running stats
GOOD_ERR_M = 5.0
TRUSTED = 0.5  # events.py's own review threshold


def identity_rows(run: Path, xy: pd.DataFrame) -> pd.DataFrame:
    """xy rows with jersey, name and a run key (one identity, one tracklet part), for identified rows only."""
    pi = pd.read_csv(run / "player_identity.csv")
    if "split_at_switch" not in pi:  # written before tracklets could be split
        pi["split_at_switch"] = False
    whole = pi[pi.jersey.notna() & ~pi.split_at_switch.fillna(False).astype(bool)]
    a = xy.merge(whole[["track_id", "jersey"]], on="track_id")
    a["part"] = 0
    parts = [a]
    seg_path = run / "identity_segments.csv"
    if seg_path.exists():
        segs = pd.read_csv(seg_path)
        for s in segs.itertuples():
            r = xy[(xy.track_id == s.track_id) & (xy.ci >= s.ci_start) & (xy.ci <= s.ci_end)].copy()
            r["jersey"], r["part"] = s.jersey, int(s.seg) + 1
            parts.append(r)
    out = pd.concat(parts, ignore_index=True)
    out["jersey"] = out.jersey.astype(int)
    return out.sort_values(["jersey", "track_id", "part", "pf"]).reset_index(drop=True)


def smooth_steps(d: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Smoothed positions and per-step speed, within continuous runs (same track and part, consecutive frames)."""
    d = d.sort_values(["track_id", "part", "pf"]).copy()
    new_run = (d.track_id != d.track_id.shift()) | (d.part != d.part.shift()) | (d.pf.diff() != 1)
    d["run"] = new_run.cumsum()
    win = max(1, int(round(SMOOTH_S * fps)))
    g = d.groupby("run")
    d["x_s"] = g.X_m.transform(lambda s: s.rolling(win, center=True, min_periods=1).mean())
    d["y_s"] = g.Y_m.transform(lambda s: s.rolling(win, center=True, min_periods=1).mean())
    step = np.hypot(d.groupby("run").x_s.diff(), d.groupby("run").y_s.diff())
    d["speed"] = step * fps  # NaN at the start of each run
    return d


def error_model(run: Path) -> tuple:
    """(floor_m, m_per_s): position error ~ max(anchor fit RMS, drift rate x seconds from the nearest anchor)."""
    rep = json.loads((run / "pitch_calibration_report.json").read_text())
    if rep.get("position_error_floor_m"):  # automatic fixes every 2 s (pitch_ptz.py): measured error, no drift term
        return float(rep["position_error_floor_m"]), 0.0
    rates = [c["median_error_m"] / c["gap_s"] for c in rep["cross_check"] if c["gap_s"] > 0]
    return float(np.median(rep["anchor_fit_rms_m"])), float(np.median(rates)) if rates else 0.0


def running_stats(d: pd.DataFrame, fps: float, clip_s: float, err: tuple) -> dict:
    v = d.speed.dropna()
    glitch = v > MAX_SPEED_MPS
    v = v[~glitch]
    visible_s = len(d) / fps
    pos_err = np.maximum(err[0], err[1] * d.anchor_gap_s.to_numpy())
    good = float((pos_err <= GOOD_ERR_M).mean())
    out = dict(
        visible_s=round(visible_s, 1),
        visibility_pct=round(100 * visible_s / clip_s, 1),
        distance_m=round(float(v.sum() / fps), 1),
        m_per_min=round(float(v.sum() / fps) / (visible_s / 60), 1) if visible_s else np.nan,
        speed_p95_mps=round(float(np.percentile(v, 95)), 2) if len(v) else np.nan,
        glitch_steps=int(glitch.sum()),
        x_med_m=round(float(d.x_s.median()), 1),
        y_med_m=round(float(d.y_s.median()), 1),
        pos_err_m=round(float(np.median(pos_err)), 1),
        running_confidence=round(min(1.0, visible_s / FULL_CONF_S) * good, 2),
    )
    for band, (lo, hi) in SPEED_BANDS.items():
        out[f"pct_{band}"] = round(100 * float(((v >= lo) & (v < hi)).mean()), 1) if len(v) else np.nan
    return out


def event_rows(run: Path, ident: pd.DataFrame, cache_fps: float) -> pd.DataFrame:
    """events.csv with the jersey of the player at each end (at the event's frame), where identified."""
    ev = pd.read_csv(run / "events.csv")
    key = ident.set_index(["track_id", "ci"]).jersey
    by_track = ident.groupby("track_id")

    def who(track_id, ci):
        if pd.isna(track_id) or int(track_id) not in by_track.groups:
            return np.nan
        j = key.get((int(track_id), int(ci)))
        if j is not None:
            return j
        # xy rows are every other cached frame at 15 fps: use rows within 2 frames, but only if they all agree.
        # Frames around a marked switch have no identity at all, so an event there stays unattributed.
        g = by_track.get_group(int(track_id))
        near = g[(g.ci - ci).abs() <= 2]
        return near.jersey.iloc[0] if len(near) and (near.jersey == near.jersey.iloc[0]).all() else np.nan

    ev["jersey"] = [who(t, c) for t, c in zip(ev.track_id, ev.ci, strict=True)]
    end_ci = (ev.end_s * cache_fps).round()
    ev["to_jersey"] = [who(t, c) for t, c in zip(ev.to_track_id, end_ci, strict=True)]
    ev["trusted"] = ev.confidence >= TRUSTED
    return ev


def event_stats(ev: pd.DataFrame, jersey: int) -> dict:
    out = {}
    specs = {
        "possessions": (ev.type == "possession") & (ev.jersey == jersey),
        "touches": (ev.type == "touch") & (ev.jersey == jersey),
        "passes_made": (ev.type == "pass") & (ev.jersey == jersey),
        "passes_received": (ev.type == "pass") & (ev.to_jersey == jersey),
        "turnovers_lost": (ev.type == "turnover") & (ev.jersey == jersey),
        "turnovers_won": (ev.type == "turnover") & (ev.to_jersey == jersey),
    }
    confs = []
    for name, m in specs.items():
        out[name] = int((m & ev.trusted).sum())
        out[f"{name}_unconfirmed"] = int((m & ~ev.trusted).sum())
        confs += ev.confidence[m & ev.trusted].tolist()
    pos = ev[(ev.type == "possession") & (ev.jersey == jersey) & ev.trusted]
    out["possession_s"] = round(float((pos.end_s - pos.time_s).sum()), 1)
    out["event_confidence"] = round(float(np.mean(confs)), 2) if confs else np.nan
    return out


def noise_floor(run: Path, xy: pd.DataFrame, fps: float) -> dict:
    """Distance per minute that people standing still appear to cover (sideline_suspect tracklets)."""
    roles = pd.read_csv(run / "tracklet_roles.csv")
    still = roles[roles.sideline_suspect.astype(bool)].track_id
    d = xy[xy.track_id.isin(still)].assign(part=0)
    if not len(d):
        return {}
    v = smooth_steps(d, fps).speed.dropna()
    v = v[v <= MAX_SPEED_MPS]
    return dict(
        still_tracklets=int(still.nunique()),
        still_minutes=round(len(d) / fps / 60, 1),
        m_per_min=round(float(v.mean() * 60), 1),
        speed_p95_mps=round(float(np.percentile(v, 95)), 2),
    )


def clip_stats(run: Path) -> tuple:
    fps = float(json.loads((run / "best_config.json").read_text())["fps"])
    cache = Cache(run / "cache")
    clip_s = cache.n / cache.fps
    xy = pd.read_csv(run / "tracklet_pitch_xy.csv.gz")
    ident = identity_rows(run, xy)
    err = error_model(run)
    ev = event_rows(run, ident, cache.fps)
    rows = []
    for jersey, d in ident.groupby("jersey"):
        s = running_stats(smooth_steps(d, fps), fps, clip_s, err)
        rows.append(dict(clip=run.name, jersey=int(jersey), **s, **event_stats(ev, jersey)))
    stats = pd.DataFrame(rows)
    roster = pd.read_csv(ROSTER_FILE)
    stats = stats.merge(roster[["jersey", "name"]], on="jersey", how="left")
    report = dict(
        clip=run.name,
        clip_s=round(clip_s, 1),
        players=len(stats),
        identified_player_minutes=round(float(stats.visible_s.sum() / 60), 1),
        position_error_model=dict(floor_m=round(err[0], 2), drift_m_per_s=round(err[1], 3)),
        noise_floor=noise_floor(run, xy, fps),
        events_total=int(len(ev)),
        events_trusted=int(ev.trusted.sum()),
        events_with_identified_player=int(ev.jersey.notna().sum()),
        settings=dict(smooth_s=SMOOTH_S, speed_bands_mps=SPEED_BANDS, trusted_event_confidence=TRUSTED),
        note=(
            "Running stats are per visible minute and assume the named parts are right. m_per_min includes the "
            "noise floor (distance a still person appears to cover). Position error is mostly slow drift: it "
            "biases x/y far more than distance. Event counts inherit events.py's accuracy (touch weak, see "
            "CLAUDE.md Phase 4). UNVERIFIED against measured distances."
        ),
    )
    return stats, ev[ev.jersey.notna() | ev.to_jersey.notna()], report


def share_summary(stats: pd.DataFrame, reports: list) -> dict:
    """Team aggregates per clip only: no names, no jersey numbers (ground rules). Not pooled across clips, since
    running stats compare within a clip (each clip has its own noise floor)."""
    out = []
    for r in reports:
        s = stats[stats["clip"] == r["clip"]]
        out.append(
            dict(
                clip_s=r["clip_s"],
                players=r["players"],
                identified_player_minutes=r["identified_player_minutes"],
                noise_floor_m_per_min=r["noise_floor"].get("m_per_min"),
                m_per_min_median=round(float(s.m_per_min.median()), 1),
                speed_p95_median_mps=round(float(s.speed_p95_mps.median()), 2),
                pct_fast_median=round(float(s.pct_fast.median()), 1),
                trusted_touches_total=int(s.touches.sum()),
                trusted_passes_total=int(s.passes_made.sum()),
            )
        )
    return dict(clips=out, note="Per clip; compare within a clip, not across clips.")


def _has_table(con, name: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="comma list of run folders")
    ap.add_argument("--db", type=require_under_data, default=DATA_DIR / "stats.sqlite")
    ap.add_argument("--share-dir", type=Path, default=None)
    args = ap.parse_args()
    all_stats, all_events, reports = [], [], []
    for r in args.runs.split(","):
        run = require_under_data(Path(r))
        stats, ev, report = clip_stats(run)
        stats.to_csv(run / "player_stats.csv", index=False)
        ev.to_csv(run / "player_events.csv", index=False)
        (run / "stats_report.json").write_text(json.dumps(report, indent=2))
        all_stats.append(stats)
        all_events.append(ev.assign(clip=run.name))
        reports.append(report)
        print(f"{run.name}: {len(stats)} players, {report['identified_player_minutes']} identified player-minutes, "
              f"noise floor {report['noise_floor'].get('m_per_min')} m/min")  # fmt: skip
    stats = pd.concat(all_stats, ignore_index=True)
    clips = pd.DataFrame(reports).drop(columns=["noise_floor", "position_error_model", "settings"])
    tables = {"clips": clips, "player_clip_stats": stats, "player_events": pd.concat(all_events, ignore_index=True)}
    with sqlite3.connect(args.db) as con:
        # replace only the clips in this run, so rerunning one clip keeps the others in the database
        for name, df in tables.items():
            old = pd.read_sql(f"SELECT * FROM {name}", con) if _has_table(con, name) else df.iloc[:0]
            keep = old[~old["clip"].isin(df["clip"].unique())] if "clip" in old else old.iloc[:0]
            pd.concat([keep, df], ignore_index=True).to_sql(name, con, if_exists="replace", index=False)
    print(f"Wrote player_stats.csv, player_events.csv, stats_report.json per run, and {args.db}")
    if args.share_dir:
        args.share_dir.mkdir(parents=True, exist_ok=True)
        (args.share_dir / "team_stats_summary.json").write_text(json.dumps(share_summary(stats, reports), indent=2))
        print(f"Wrote team aggregates (no names or numbers) to {args.share_dir}")


if __name__ == "__main__":
    main()
