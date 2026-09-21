"""Detect possession, touches, passes and turnovers from the ball path and role-labeled tracklets.

Inputs (all in the run folder): ball_path.csv (ball_link.py), best_tracklets.csv.gz (replay_trackers.py) and
tracklet_roles.csv (team_classify.py). Only player candidates with a target, opponent or goalkeeper role are
considered. Distances are in body heights (the player's box height), so no pitch calibration is needed.
Shots are not detected: they need the goal position, which needs pitch calibration.

How it works, on the ball's frame grid (10 fps):
  - Contact: a player is in contact when the ball is within CONTACT_H body heights of their feet.
  - Possession: the nearest player in contact, as a segment of at least MIN_POSSESSION_S. Short gaps are bridged.
  - Touch: a sharp change in ball velocity (in body heights per second) with a player within TOUCH_H.
  - Pass: possession moves between two different tracklets of one team within MAX_PASS_GAP_S. Tracklets are
    fragments (about 13 IDs per player), so a handoff where the two players are less than MIN_PASS_H apart is
    treated as the same player and merged, not counted as a pass.
  - Turnover: possession moves between teams within MAX_PASS_GAP_S.

Every event carries a confidence (0 to 1) and the share of frames around it where the ball was actually detected
rather than interpolated. Events under --min-confidence go to review_queue.csv instead of being trusted.

Example:
  python events.py --run data\\clipA --montage
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sv_common import Cache, cache_stride, read_frames

TEAM_ROLES = {"target": "target", "opponent": "opponent", "goalkeeper": "goalkeeper"}
CONTACT_H = 0.6  # ball this close to the feet (in body heights) counts as contact
TOUCH_H = 1.0  # a velocity change counts as a touch only with a player this close
TOUCH_DV_H = 2.0  # velocity change (body heights per second) needed for a touch
MIN_POSSESSION_S = 0.4
BRIDGE_S = 0.4  # gaps up to this long inside one player's possession are bridged
MAX_PASS_GAP_S = 3.0
MIN_PASS_H = 1.5  # players closer than this at the handoff are treated as one player


def load_inputs(run: Path):
    cache = Cache(run / "cache")
    ball = pd.read_csv(run / "ball_path.csv").sort_values("ci").reset_index(drop=True)
    tr = pd.read_csv(run / "best_tracklets.csv.gz")
    roles = pd.read_csv(run / "tracklet_roles.csv", index_col="track_id")
    keep = roles.index[roles.player_candidate & roles.role.isin(TEAM_ROLES)]
    tr = tr[tr.track_id.isin(keep)]
    return cache, ball, tr, roles


def player_positions(tr: pd.DataFrame, ci: np.ndarray, max_gap_frames: int = 30):
    """Foot x, foot y and box height of every tracklet at each ball frame, NaN where a tracklet is absent."""
    ids = np.sort(tr.track_id.unique())
    foot_x, foot_y, height = (np.full((len(ci), len(ids)), np.nan) for _ in range(3))
    for j, tid in enumerate(ids):
        d = tr[tr.track_id == tid].sort_values("ci")
        t = d.ci.to_numpy()
        inside = (ci >= t[0]) & (ci <= t[-1])
        if not inside.any():
            continue
        # A gap longer than max_gap_frames inside a tracklet means the player was not really tracked there.
        k = np.searchsorted(t, ci[inside])
        near = np.minimum(
            np.abs(t[np.clip(k, 0, len(t) - 1)] - ci[inside]), np.abs(t[np.clip(k - 1, 0, len(t) - 1)] - ci[inside])
        )
        ok = near <= max_gap_frames
        cx, y2, h = (d.x1 + d.x2).to_numpy() / 2, d.y2.to_numpy(), (d.y2 - d.y1).to_numpy()
        for arr, vals in ((foot_x, cx), (foot_y, y2), (height, h)):
            col = np.full(len(ci), np.nan)
            col[inside] = np.where(ok, np.interp(ci[inside], t, vals), np.nan)
            arr[:, j] = col
    return ids, foot_x, foot_y, np.maximum(height, 1.0)


def detect(run: Path, min_confidence: float):
    cache, ball, tr, roles = load_inputs(run)
    fps = cache.fps / max(1, int(np.median(np.diff(ball.ci)))) if len(ball) > 1 else 10.0
    ci = ball.ci.to_numpy()
    bx, by = ball.x.to_numpy(), ball.y.to_numpy()
    detected = (ball.kind == "detected").to_numpy()
    seg = ball.seg.to_numpy()
    ids, fx, fy, h = player_positions(tr, ci)
    role_of = roles.role.reindex(ids).to_numpy()
    role_conf = roles.confidence.reindex(ids).to_numpy()
    n = len(ci)

    # Ball-to-feet distance in body heights, in the image of each frame (camera motion does not matter within a frame).
    dist = np.hypot(bx[:, None] - fx, by[:, None] - fy) / h
    dist = np.where(np.isfinite(dist), dist, np.inf)
    nearest = dist.argmin(axis=1) if dist.shape[1] else np.zeros(n, dtype=int)
    dmin = dist.min(axis=1) if dist.shape[1] else np.full(n, np.inf)
    in_contact = dmin <= CONTACT_H

    # Ball velocity in stable coordinates (px/s), expressed in body heights per second of the nearest player.
    sx, sy = cache.to_stable(ci, bx, by)
    same_seg = np.r_[False, seg[1:] == seg[:-1]]
    vx, vy = np.gradient(sx) * fps, np.gradient(sy) * fps
    scale = np.sqrt(np.abs(np.linalg.det(cache.cum[ci][:, :2, :2])))
    h_near = h[np.arange(n), nearest] if dist.shape[1] else np.full(n, 70.0)
    h_ref = np.where(np.isfinite(h_near), h_near, np.nanmedian(h) if np.isfinite(h).any() else 70.0)

    def visible_share(i: int, half_s: float = 0.3) -> float:
        k = max(1, int(round(half_s * fps)))
        return float(detected[max(0, i - k) : i + k + 1].mean())

    events = []
    # ---- possession segments
    labels = np.where(in_contact, nearest, -1)
    segments, i = [], 0
    while i < n:
        if labels[i] < 0:
            i += 1
            continue
        j = i
        while j + 1 < n:
            nxt = j + 1
            gap = 0
            while nxt < n and labels[nxt] != labels[i] and gap <= BRIDGE_S * fps:
                nxt += 1
                gap += 1
            if nxt < n and labels[nxt] == labels[i] and gap <= BRIDGE_S * fps:
                j = nxt
            else:
                break
        segments.append((i, j, int(labels[i])))
        i = j + 1
    segments = [(a, b, p) for a, b, p in segments if (ci[b] - ci[a]) / cache.fps + 1 / fps >= MIN_POSSESSION_S]

    def conf_for(a: int, b: int, p: int, gaps: float = 0.0) -> float:
        prox = float(np.clip(1 - np.nanmedian(dmin[a : b + 1]) / CONTACT_H, 0, 1))
        return round(
            float(np.clip(0.4 * visible_share((a + b) // 2) + 0.3 * role_conf[p] + 0.3 * prox - gaps, 0, 1)), 3
        )

    for a, b, p in segments:
        events.append(
            dict(
                type="possession",
                time_s=round(ci[a] / cache.fps, 2),
                end_s=round(ci[b] / cache.fps, 2),
                ci=int(ci[a]),
                track_id=int(ids[p]),
                team=TEAM_ROLES[role_of[p]],
                to_track_id=np.nan,
                to_team="",
                confidence=conf_for(a, b, p),
                ball_detected_share=round(float(detected[a : b + 1].mean()), 2),
                distance_h=round(float(np.nanmedian(dmin[a : b + 1])), 2),
            )
        )

    # ---- touches: sharp velocity change with a player close
    dvx, dvy = np.zeros(n), np.zeros(n)
    dvx[1:-1], dvy[1:-1] = vx[2:] - vx[:-2], vy[2:] - vy[:-2]
    ok_seg = same_seg & np.r_[same_seg[1:], False]
    dv_h = np.hypot(dvx, dvy) * scale / h_ref / 2  # body heights per second, over the 2-frame window
    cand = np.flatnonzero(ok_seg & (dv_h >= TOUCH_DV_H) & (dmin <= TOUCH_H))
    last = -(10**9)
    for i in cand:
        window = np.arange(max(0, i - 2), min(n, i + 3))
        if i != window[np.argmax(dv_h[window])] or ci[i] - last < 0.3 * cache.fps:
            continue
        last = ci[i]
        p = int(nearest[i])
        prox = float(np.clip(1 - dmin[i] / TOUCH_H, 0, 1))
        c = float(np.clip(0.4 * visible_share(i) + 0.3 * role_conf[p] + 0.3 * prox, 0, 1))
        events.append(
            dict(
                type="touch",
                time_s=round(ci[i] / cache.fps, 2),
                end_s=np.nan,
                ci=int(ci[i]),
                track_id=int(ids[p]),
                team=TEAM_ROLES[role_of[p]],
                to_track_id=np.nan,
                to_team="",
                confidence=round(c, 3),
                ball_detected_share=round(visible_share(i), 2),
                distance_h=round(float(dmin[i]), 2),
            )
        )

    # ---- passes and turnovers between consecutive possession segments
    merged = []
    for a, b, p in segments:
        if merged and merged[-1][2] == p and (ci[a] - ci[merged[-1][1]]) / cache.fps <= BRIDGE_S:
            merged[-1] = (merged[-1][0], b, p)
        else:
            merged.append((a, b, p))
    for (a1, b1, p1), (a2, b2, p2) in zip(merged[:-1], merged[1:], strict=True):
        gap_s = (ci[a2] - ci[b1]) / cache.fps
        if p1 == p2 or gap_s > MAX_PASS_GAP_S:
            continue
        t1, t2 = TEAM_ROLES[role_of[p1]], TEAM_ROLES[role_of[p2]]
        if "goalkeeper" in (t1, t2):
            continue  # goalkeeper roles are unreliable, do not build team events on them
        # Handoff separation: player 1 at the end of their possession vs player 2 at the start of theirs, in the
        # image of the frame where player 2 starts (both positions taken from the ball frame grid).
        px1, py1 = np.nanmean(fx[b1 : b1 + 1, p1]), np.nanmean(fy[b1 : b1 + 1, p1])
        s1x, s1y = cache.to_stable(ci[[b1]], [px1], [py1])
        s2x, s2y = cache.to_stable(ci[[a2]], fx[a2 : a2 + 1, p2], fy[a2 : a2 + 1, p2])
        sep_h = float(np.hypot(s1x[0] - s2x[0], s1y[0] - s2y[0]) * scale[a2] / h_ref[a2])
        if t1 == t2 and sep_h < MIN_PASS_H:
            continue  # likely the same player under two IDs
        c = float(np.clip(min(conf_for(a1, b1, p1), conf_for(a2, b2, p2)) - 0.1 * gap_s / MAX_PASS_GAP_S, 0, 1))
        events.append(
            dict(
                type="pass" if t1 == t2 else "turnover",
                time_s=round(ci[b1] / cache.fps, 2),
                end_s=round(ci[a2] / cache.fps, 2),
                ci=int(ci[b1]),
                track_id=int(ids[p1]),
                team=t1,
                to_track_id=int(ids[p2]),
                to_team=t2,
                confidence=round(c, 3),
                ball_detected_share=round(float(detected[b1 : a2 + 1].mean()), 2),
                distance_h=round(sep_h, 2),
            )
        )

    ev = pd.DataFrame(events).sort_values(["time_s", "type"]).reset_index(drop=True) if events else pd.DataFrame()
    speed_h = np.where(ok_seg, np.hypot(vx, vy) * scale / h_ref, np.nan)  # not across path segment breaks
    return (
        ev,
        dict(fps=fps, n_frames=n, detected=detected, in_contact=in_contact, speed_h=speed_h, ids=ids, ball=ball),
        cache,
    )


def write_montage(run: Path, ev: pd.DataFrame, info: dict, per_type: int = 4) -> None:
    """Local review image: the ball (circle) and the credited player's box for sampled events. Shows people."""
    tr = pd.read_csv(run / "best_tracklets.csv.gz")
    ball = info["ball"].set_index("ci")
    stride = cache_stride(run)
    rng = np.random.default_rng(5)
    picks = []
    for t in ("possession", "touch", "pass", "turnover"):
        sub = ev[ev.type == t]
        if len(sub):
            picks.append(sub.iloc[np.sort(rng.choice(len(sub), min(per_type, len(sub)), replace=False))])
    if not picks:
        return
    picks = pd.concat(picks)
    frames = {int(r.ci) * stride: r for r in picks.itertuples()}
    tiles = {}
    for clip_frame, img in read_frames(run / "clip.mp4", list(frames)):
        r = frames[clip_frame]
        bx, by = ball.loc[r.ci, ["x", "y"]] if r.ci in ball.index else (np.nan, np.nan)
        d = tr[(tr.track_id == r.track_id) & (tr.ci >= r.ci - 3) & (tr.ci <= r.ci + 3)]
        if len(d):
            row = d.iloc[0]
            cv2.rectangle(img, (int(row.x1), int(row.y1)), (int(row.x2), int(row.y2)), (0, 255, 0), 2)
        if np.isfinite(bx):
            cv2.circle(img, (int(bx), int(by)), 24, (0, 0, 255), 2)
            cx, cy = int(bx), int(by)
        else:
            cx, cy = img.shape[1] // 2, img.shape[0] // 2
        x0, y0 = min(max(cx - 160, 0), img.shape[1] - 320), min(max(cy - 90, 0), img.shape[0] - 180)
        tile = cv2.resize(img[y0 : y0 + 180, x0 : x0 + 320], (320, 180))
        cv2.putText(
            tile,
            f"{r.type} {r.time_s:.0f}s c{r.confidence:.2f}",
            (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        tiles[clip_frame] = tile
    tiles_list = [tiles[k] for k in frames if k in tiles]
    tiles_list += [np.zeros((180, 320, 3), np.uint8)] * (-len(tiles_list) % 4)
    cv2.imwrite(
        str(run / "events_montage.png"),
        np.vstack([np.hstack(tiles_list[i : i + 4]) for i in range(0, len(tiles_list), 4)]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--min-confidence", type=float, default=0.5, help="events below this go to the review queue")
    ap.add_argument("--montage", action="store_true", help="write events_montage.png, local review only")
    args = ap.parse_args()

    ev, info, cache = detect(args.run, args.min_confidence)
    if ev.empty:
        raise SystemExit("No events found.")
    ev.to_csv(args.run / "events.csv", index=False)
    ev[ev.confidence < args.min_confidence].to_csv(args.run / "review_queue.csv", index=False)

    minutes = info["n_frames"] / info["fps"] / 60
    poss = ev[ev.type == "possession"]
    dur = poss.end_s - poss.time_s
    team_time = dur.groupby(poss.team).sum()
    report = {
        "minutes": round(minutes, 2),
        "events_by_type": ev.type.value_counts().to_dict(),
        "events_per_minute": (ev.type.value_counts() / minutes).round(1).to_dict(),
        "possession_time_s_by_team": team_time.round(1).to_dict(),
        "median_possession_s": round(float(dur.median()), 1),
        "ball_frames_detected_pct": round(100 * float(info["detected"].mean()), 1),
        "ball_frames_in_contact_pct": round(100 * float(info["in_contact"].mean()), 1),
        "ball_speed_bodyheights_per_s_p50_p90_p99": np.nanpercentile(info["speed_h"], [50, 90, 99]).round(1).tolist(),
        "median_confidence": round(float(ev.confidence.median()), 2),
        "review_queue": int((ev.confidence < args.min_confidence).sum()),
        "shots": "not detected: needs the goal position, which needs pitch calibration",
        "note": "Proxy report. Events are unverified against labels. Track IDs are fragments, so passes are rough.",
    }
    (args.run / "event_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if args.montage:
        write_montage(args.run, ev, info)


if __name__ == "__main__":
    main()
