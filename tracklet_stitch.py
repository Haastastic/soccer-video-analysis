"""Merge fragmented tracklets into per-player identities (about 13 tracker IDs per real player today).

A tracklet b can continue tracklet a if b starts soon after a ends (--max-gap-s), starts close enough to
where a last was (within --base-slack-h body heights plus --max-speed-h-s times the gap - a bound on how far a
player could physically have moved, not a straight-line velocity extrapolation, since real players change
direction within a second or two), has a similar kit color (--max-color-dist, Lab distance on torso+legs), and
shares the same predicted role. Candidate merges are matched greedily, cheapest first, each tracklet getting at
most one predecessor and one successor - camera-motion-compensated ("stable") coordinates are used throughout
so a gap during a pan does not look like a jump. Chains of merged tracklets become one player_id.

This does NOT fix a tracklet that already drifted onto a different real person mid-lifetime (a known, separate
tracker bug, see Phase 2 findings) - it only joins tracklets that are already individually correct. It also
does not assign names: that is roster-based identity assignment, the next piece of step 7.

Kit color barely discriminates WITHIN a team, since everyone on one side wears the same kit (see
LOCAL_CONTEXT.md) - it only confirms "same team", which role already does. Position/motion continuity is
doing essentially all the real work here, and it is inherently ambiguous with several same-team players moving
through the same area. This is a first-pass proposal to be corrected by roster-based manual review, not a
finished identity system - see the module docstring's parameter comments for how far it currently gets.

Thresholds are hand-set guesses, not tuned: there is no ground truth for this yet. --montage writes sampled
crops per stitched player, local review only, to sanity-check by eye.

Example:
  python tracklet_stitch.py --run data\\clipA --montage
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sv_common import Cache, cache_stride, read_frames, require_under_data

TEAM_ROLES = ("target", "opponent", "goalkeeper")
# The camera follows the ball, so an off-ball player can be out of frame for many seconds, not just a brief
# occlusion - MAX_GAP_S=3 covered only 57% of even the closest same-role gaps on a first real clip. Widened
# empirically until target-team players (roster size 21) stopped grossly outnumbering the roster; still leaves
# 3+ tracklets per player on average, not the ~1 a perfect stitch would give - see the module docstring.
MAX_GAP_S = 15.0  # longest break between tracklets that can still be the same player
BASE_SLACK_H = 1.5  # position slack (body heights) allowed even at zero gap, for box/detection noise
MAX_SPEED_H_S = 2.5  # extra slack per second of gap, body heights per second (a generous running speed)
MAX_COLOR_DIST = 25.0  # Lab distance (torso + legs) allowed between the two tracklets' measured kit color
EDGE_FRAMES = 5  # frames averaged at each end of a tracklet, for a less noisy position there


def tracklet_summary(tr: pd.DataFrame, roles: pd.DataFrame, colors: pd.DataFrame, cache: Cache) -> pd.DataFrame:
    """One row per stitchable tracklet: start/end position and time in stable coordinates, kit color."""
    keep = roles.index[roles.player_candidate & roles.role.isin(TEAM_ROLES)]
    tr = tr[tr.track_id.isin(keep)].sort_values(["track_id", "ci"])
    rows = []
    for tid, d in tr.groupby("track_id"):
        d = d.reset_index(drop=True)
        fx, fy = ((d.x1 + d.x2) / 2).to_numpy(), d.y2.to_numpy()
        sx, sy = cache.to_stable(d.ci.to_numpy(), fx, fy)
        k = min(EDGE_FRAMES, len(d))
        ci = d.ci.to_numpy()
        c = colors.loc[tid]
        rows.append(
            dict(
                track_id=int(tid),
                role=roles.loc[tid, "role"],
                start_ci=int(ci[0]),
                end_ci=int(ci[-1]),
                start_x=float(np.mean(sx[:k])),
                start_y=float(np.mean(sy[:k])),
                end_x=float(np.mean(sx[-k:])),
                end_y=float(np.mean(sy[-k:])),
                h=max(float(np.median(d.y2 - d.y1)), 1.0),
                torso_L=c.torso_L,
                torso_a=c.torso_a,
                torso_b=c.torso_b,
                legs_L=c.legs_L,
                legs_a=c.legs_a,
                legs_b=c.legs_b,
            )
        )
    return pd.DataFrame(rows).set_index("track_id")


def lab_dist(ra, rb, prefix: str) -> float:
    lk, ak, bk = f"{prefix}_L", f"{prefix}_a", f"{prefix}_b"
    return float(np.sqrt((ra[lk] - rb[lk]) ** 2 + (ra[ak] - rb[ak]) ** 2 + (ra[bk] - rb[bk]) ** 2))


def candidate_edges(
    summary: pd.DataFrame,
    fps: float,
    max_gap_s: float,
    base_slack_h: float,
    max_speed_h_s: float,
    max_color_dist: float,
):
    """All (a, b, cost, dist_h, color_dist, gap_s) where tracklet b could plausibly continue tracklet a."""
    edges = []
    max_gap_ci = max_gap_s * fps
    for a, ra in summary.iterrows():
        cands = summary[
            (summary.role == ra.role) & (summary.start_ci > ra.end_ci) & (summary.start_ci - ra.end_ci <= max_gap_ci)
        ]
        for b, rb in cands.iterrows():
            gap_s = (rb.start_ci - ra.end_ci) / fps
            dist_h = float(np.hypot(ra.end_x - rb.start_x, ra.end_y - rb.start_y) / ((ra.h + rb.h) / 2))
            if dist_h > base_slack_h + max_speed_h_s * gap_s:
                continue
            color_dist = lab_dist(ra, rb, "torso") + lab_dist(ra, rb, "legs")
            if color_dist > max_color_dist:
                continue
            cost = dist_h + color_dist / 10 + 0.1 * gap_s
            edges.append((a, b, cost, dist_h, color_dist, round(gap_s, 2)))
    return edges


def greedy_chains(ids: list, edges: list) -> dict:
    """Cheapest-first greedy matching, each tracklet at most one successor and one predecessor, then flatten
    the resulting chains into a track_id -> player_id map."""
    successor, predecessor = {}, {}
    for a, b, _cost, *_ in sorted(edges, key=lambda e: e[2]):
        if a not in successor and b not in predecessor:
            successor[a], predecessor[b] = b, a
    player_of = {}
    n = 0
    for tid in ids:
        if tid in predecessor:
            continue  # not a chain start
        n += 1
        pid = f"P{n:03d}"
        cur = tid
        while True:
            player_of[cur] = pid
            if cur not in successor:
                break
            cur = successor[cur]
    return player_of


def stitch(run: Path, max_gap_s: float, base_slack_h: float, max_speed_h_s: float, max_color_dist: float):
    cache = Cache(run / "cache")
    tr = pd.read_csv(run / "best_tracklets.csv.gz")
    roles = pd.read_csv(run / "tracklet_roles.csv", index_col="track_id")
    colors = pd.read_csv(run / "tracklet_colors.csv", index_col="track_id")
    summary = tracklet_summary(tr, roles, colors, cache)
    edges = candidate_edges(summary, cache.fps, max_gap_s, base_slack_h, max_speed_h_s, max_color_dist)
    player_of = greedy_chains(list(summary.index), edges)
    out = pd.DataFrame({"track_id": list(player_of.keys()), "player_id": list(player_of.values())})
    out["role"] = out.track_id.map(roles.role)
    return out, summary, edges


def cmd_run(args) -> None:
    out, summary, edges = stitch(args.run, args.max_gap_s, args.base_slack_h, args.max_speed_h_s, args.max_color_dist)
    out.to_csv(args.run / "tracklet_stitch.csv", index=False)
    by_role = out.groupby("role").agg(tracklets=("track_id", "count"), players=("player_id", "nunique"))
    report = {
        "tracklets_in": int(len(summary)),
        "players_out": int(out.player_id.nunique()),
        "candidate_edges": len(edges),
        "by_role": {
            r: {
                "tracklets": int(row.tracklets),
                "players": int(row.players),
                "avg_tracklets_per_player": round(row.tracklets / row.players, 1),
            }
            for r, row in by_role.iterrows()
        },
        "target_players_vs_roster": {
            "stitched": int(out[out.role == "target"].player_id.nunique()),
            "roster_size": 21,
            "note": "stitched should be <= roster size; usually well under, since not everyone plays every window",
        },
        "note": (
            "Thresholds are hand-set guesses, not tuned. Does not fix a tracklet that already drifted onto a "
            "different real person mid-lifetime."
        ),
    }
    (args.run / "tracklet_stitch_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if args.montage:
        write_montage(args.run, out)


def write_montage(run: Path, out: pd.DataFrame, per_player: int = 4, max_players: int = 24) -> None:
    """Local review image: sampled crops for a spread of stitched players, to eyeball by eye. Shows people."""
    tr = pd.read_csv(run / "best_tracklets.csv.gz")
    stride = cache_stride(run)
    rng = np.random.default_rng(7)
    counts = out.player_id.value_counts()
    multi = counts[counts > 1].index  # players formed from 2+ tracklets are the ones worth checking
    pids = rng.choice(multi, min(max_players, len(multi)), replace=False) if len(multi) else []
    frames_needed, picks = {}, []
    for pid in pids:
        ids = out[out.player_id == pid].track_id.tolist()
        d = tr[tr.track_id.isin(ids)].sort_values("ci")
        idx = np.linspace(0, len(d) - 1, min(per_player, len(d))).round().astype(int)
        for i in idx:
            row = d.iloc[i]
            frames_needed[int(row.ci) * stride] = row
            picks.append((pid, int(row.ci) * stride))
    if not picks:
        print("No stitched player has more than 1 tracklet - nothing to montage.")
        return
    tiles = {}
    for clip_frame, img in read_frames(run / "clip.mp4", list(frames_needed)):
        row = frames_needed[clip_frame]
        cv2.rectangle(img, (int(row.x1), int(row.y1)), (int(row.x2), int(row.y2)), (0, 255, 0), 2)
        cx, cy = int((row.x1 + row.x2) / 2), int(row.y2)
        x0, y0 = min(max(cx - 80, 0), img.shape[1] - 160), min(max(cy - 140, 0), img.shape[0] - 180)
        tiles[clip_frame] = cv2.resize(img[y0 : y0 + 180, x0 : x0 + 160], (160, 180))
    rows = []
    for pid in pids:
        frame_ids = [f for p, f in picks if p == pid and f in tiles]
        row_tiles = [tiles[f] for f in frame_ids]
        row_tiles += [np.zeros((180, 160, 3), np.uint8)] * (per_player - len(row_tiles))
        strip = np.hstack(row_tiles)
        cv2.putText(strip, str(pid), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(strip)
    cv2.imwrite(str(run / "stitch_montage.png"), np.vstack(rows))
    print(f"Wrote {run / 'stitch_montage.png'} (local only, shows people). Each row is one stitched player.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=require_under_data)
    ap.add_argument("--max-gap-s", type=float, default=MAX_GAP_S)
    ap.add_argument("--base-slack-h", type=float, default=BASE_SLACK_H)
    ap.add_argument("--max-speed-h-s", type=float, default=MAX_SPEED_H_S)
    ap.add_argument("--max-color-dist", type=float, default=MAX_COLOR_DIST)
    ap.add_argument("--montage", action="store_true", help="write stitch_montage.png, local review only")
    ap.set_defaults(fn=cmd_run)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
