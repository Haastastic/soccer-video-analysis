"""Link ball candidates into one plausible path (Option A, no labeling needed).

Picks at most one candidate per frame so that consecutive picks are physically plausible after
cancelling camera motion, favors confident detections, bridges short gaps by interpolation, and
drops isolated low-confidence hits. Solved exactly with dynamic programming.

Output is a hypothesis, not truth. Score it against hand-verified frames before trusting it.

Example:
  python ball_link.py --run data\\clipA --fps 10
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sv_common import BALL, Cache


def link(pf, x, y, conf, fps, vmax, reward_base, w_move, gap_cost, restart_cost, max_gap_s):
    """Return (best_prev, on_path_mask, restart_flags) for candidates sorted by frame."""
    n = len(pf)
    score = np.zeros(n)
    prev = -np.ones(n, dtype=int)
    is_restart = np.zeros(n, dtype=bool)
    max_gap = int(round(max_gap_s * fps))
    frame_start = {}
    for i, f in enumerate(pf):
        frame_start.setdefault(int(f), i)
    prefix_best = -np.inf  # best chain score over candidates in strictly earlier frames
    prefix_arg = -1
    cur_best, cur_arg = -np.inf, -1
    last_frame = None
    for j in range(n):
        f = int(pf[j])
        if last_frame is None or f != last_frame:
            if cur_arg >= 0 and cur_best > prefix_best:
                prefix_best, prefix_arg = cur_best, cur_arg
            cur_best, cur_arg = -np.inf, -1
            last_frame = f
        r = conf[j] + reward_base
        best, bp, rs = 0.0, -1, True  # start a chain with no predecessor
        if prefix_arg >= 0 and prefix_best - restart_cost > best:
            best, bp, rs = prefix_best - restart_cost, prefix_arg, True
        lo = np.searchsorted(pf, f - max_gap, side="left")
        hi = np.searchsorted(pf, f, side="left")
        for i in range(lo, hi):
            k = f - int(pf[i])
            dt = k / fps
            d = np.hypot(x[j] - x[i], y[j] - y[i])
            dn = d / (vmax * dt)
            if dn > 1.5:
                continue
            c = score[i] - (w_move * min(dn, 1.5) ** 2 + gap_cost * (k - 1) / fps)
            if c > best:
                best, bp, rs = c, i, False
        score[j] = r + best
        prev[j], is_restart[j] = bp, rs
        if score[j] > cur_best:
            cur_best, cur_arg = score[j], j
    end = int(np.argmax(score))
    path = []
    j = end
    while j >= 0:
        path.append(j)
        j = prev[j]
    path.reverse()
    return np.array(path), is_restart


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--fps", type=float, default=10, help="processing rate for linking")
    ap.add_argument("--min-conf", type=float, default=0.05)
    ap.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="max ball speed, stable px/s (default 1200 with camera data, 2500 without)",
    )
    ap.add_argument("--reward-base", type=float, default=0.25)
    ap.add_argument("--w-move", type=float, default=0.5)
    ap.add_argument("--gap-cost", type=float, default=0.4, help="cost per second of missing frames inside a link")
    ap.add_argument("--restart-cost", type=float, default=1.0)
    ap.add_argument("--max-gap-s", type=float, default=1.5, help="longest gap a link may span")
    ap.add_argument("--interp-s", type=float, default=1.0, help="interpolate across gaps up to this long")
    ap.add_argument("--min-seg", type=int, default=3, help="drop segments shorter than this unless very confident")
    args = ap.parse_args()

    cache = Cache(args.run / "cache")
    idxs, fps = cache.processed_indices(args.fps)
    pf_of = {int(ci): k for k, ci in enumerate(idxs)}
    b = cache.det[(cache.det.cls == BALL) & (cache.det.conf >= args.min_conf) & cache.det.ci.isin(idxs)].copy()
    b["pf"] = b.ci.map(pf_of)
    b["cx"] = (b.x1 + b.x2) / 2
    b["cy"] = (b.y1 + b.y2) / 2
    b = b.sort_values(["pf", "conf"], ascending=[True, False]).reset_index(drop=True)
    n_proc = len(idxs)
    sx, sy = cache.to_stable(b.ci.to_numpy(), b.cx.to_numpy(), b.cy.to_numpy())
    vmax = args.vmax or (1200.0 if cache.has_camera else 2500.0)

    # Reference numbers: what "take the best score per frame" does.
    top1 = b.drop_duplicates("pf").reset_index(drop=True)
    tsx, tsy = cache.to_stable(top1.ci.to_numpy(), top1.cx.to_numpy(), top1.cy.to_numpy())
    consec = np.diff(top1.pf.to_numpy()) == 1
    jumps = np.hypot(np.diff(tsx), np.diff(tsy))
    top1_teleport = float((jumps[consec] > 150).mean() * 100) if consec.any() else 0.0

    if len(b) == 0:
        report = {
            "processed_frames": n_proc,
            "fps": round(fps, 2),
            "candidates_used": 0,
            "note": "No ball candidates in the cache at this confidence floor.",
        }
        (args.run / "ball_link_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return
    path, is_restart = link(
        b.pf.to_numpy(),
        sx,
        sy,
        b.conf.to_numpy(),
        fps,
        vmax,
        args.reward_base,
        args.w_move,
        args.gap_cost,
        args.restart_cost,
        args.max_gap_s,
    )
    p = b.iloc[path].copy()
    p["sx"], p["sy"] = sx[path], sy[path]
    p["seg"] = np.cumsum(is_restart[path] | (np.arange(len(path)) == 0))
    seg_len = p.groupby("seg").size()
    seg_max_conf = p.groupby("seg").conf.max()
    ok_segs = seg_len.index[(seg_len >= args.min_seg) | (seg_max_conf >= 0.6)]
    p = p[p.seg.isin(ok_segs)].copy()

    # Interpolate gaps inside a segment, up to interp-s.
    rows = []
    for seg, g in p.groupby("seg"):
        g = g.sort_values("pf")
        pfs, sxs, sys_, cs, cis = g.pf.to_numpy(), g.sx.to_numpy(), g.sy.to_numpy(), g.conf.to_numpy(), g.ci.to_numpy()
        for i in range(len(g)):
            rows.append((int(pfs[i]), int(cis[i]), sxs[i], sys_[i], float(cs[i]), "detected", int(seg)))
            if i + 1 < len(g):
                k = int(pfs[i + 1] - pfs[i])
                if k > 1 and k / fps <= args.interp_s:
                    for s in range(1, k):
                        a = s / k
                        pf_mid = int(pfs[i]) + s
                        rows.append(
                            (
                                pf_mid,
                                int(idxs[pf_mid]),
                                sxs[i] + a * (sxs[i + 1] - sxs[i]),
                                sys_[i] + a * (sys_[i + 1] - sys_[i]),
                                np.nan,
                                "interpolated",
                                int(seg),
                            )
                        )
    path_df = (
        pd.DataFrame(rows, columns=["pf", "ci", "sx", "sy", "conf", "kind", "seg"])
        .sort_values("pf")
        .reset_index(drop=True)
    )
    ix, iy = cache.to_image(path_df.ci.to_numpy(), path_df.sx.to_numpy(), path_df.sy.to_numpy())
    path_df["x"], path_df["y"] = np.round(ix, 1), np.round(iy, 1)
    path_df["time_s"] = np.round(path_df.ci.to_numpy() / cache.fps, 3)
    path_df.drop(columns=["sx", "sy"]).to_csv(args.run / "ball_path.csv", index=False)

    have = np.zeros(n_proc, dtype=bool)
    have[path_df.pf.to_numpy()] = True
    det_only = np.zeros(n_proc, dtype=bool)
    det_only[path_df[path_df.kind == "detected"].pf.to_numpy()] = True
    gaps, run = [], 0
    for h in have:
        if h:
            if run:
                gaps.append(run)
            run = 0
        else:
            run += 1
    if run:
        gaps.append(run)
    gaps = np.array(gaps) / fps if gaps else np.array([0.0])
    seg_stats = path_df.groupby("seg").size()

    report = {
        "processed_frames": n_proc,
        "fps": round(fps, 2),
        "candidates_used": int(len(b)),
        "frames_with_any_candidate_pct": round(100 * b.pf.nunique() / n_proc, 1),
        "top1_teleport_pct_before": round(top1_teleport, 1),
        "path_detected_pct": round(100 * float(det_only.mean()), 1),
        "path_with_interpolation_pct": round(100 * float(have.mean()), 1),
        "segments": int(path_df.seg.nunique()),
        "median_segment_s": round(float(seg_stats.median() / fps), 1) if len(seg_stats) else 0.0,
        "longest_gap_s": round(float(gaps.max()), 1),
        "gaps_over_2s": int((gaps > 2).sum()),
        "time_in_gaps_over_2s": round(float(gaps[gaps > 2].sum()), 0),
        "median_conf_on_path": round(float(path_df.conf.median()), 2) if path_df.conf.notna().any() else None,
        "camera_motion_used": cache.has_camera,
        "vmax_px_s": vmax,
        "note": "Hypothesis only. Verify against hand-labeled frames.",
    }
    (args.run / "ball_link_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
