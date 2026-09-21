"""Replay ByteTrack and BoT-SORT on cached detections and score every configuration.

No video and no GPU needed. BoT-SORT uses the camera motion stored in the cache instead of
recomputing it. Scores are proxy metrics (no ground truth): fewer new IDs per minute and fewer
swap suspects are better, and the swap column keeps a tracker from winning by merging players.

Example:
  python replay_trackers.py --run data\\clipA
  python replay_trackers.py --run data\\clipA --grid wide
  python replay_trackers.py --run data\\clipA --grid full --fps-list 10,15,30 --trackers bytetrack,botsort
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sv_common import COLOR_COLS, PERSON, Cache, require_under_data, track_metrics

DEFAULTS = dict(
    tracker_type="bytetrack",
    track_high_thresh=0.25,
    track_low_thresh=0.1,
    new_track_thresh=0.25,
    track_buffer=60,
    match_thresh=0.8,
    fuse_score=True,
    gmc_method="sparseOptFlow",
    proximity_thresh=0.5,
    appearance_thresh=0.8,
    with_reid=False,
    model="auto",
    mot20=False,
)


class Dets:
    """Minimal results-like object the Ultralytics trackers accept."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)
        w = self.xyxy[:, 2] - self.xyxy[:, 0]
        h = self.xyxy[:, 3] - self.xyxy[:, 1]
        self.xywh = np.stack([self.xyxy[:, 0] + w / 2, self.xyxy[:, 1] + h / 2, w, h], axis=1).astype(np.float32)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return Dets(self.xyxy[idx], self.conf[idx], self.cls[idx])


class CachedGMC:
    """Stands in for the tracker's camera motion estimator and returns the cached warp."""

    def __init__(self):
        self.method = "cached"
        self.H = np.eye(2, 3)

    def apply(self, raw_frame=None, detections=None):
        return self.H

    def reset_params(self):
        pass


def build_tracker(cfg: dict):
    from types import SimpleNamespace

    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.trackers.byte_tracker import BYTETracker

    args = SimpleNamespace(**{**DEFAULTS, **cfg})
    cls = BOTSORT if cfg["tracker_type"] == "botsort" else BYTETracker
    try:
        tracker = cls(args)  # newer Ultralytics
    except TypeError:
        tracker = cls(args, frame_rate=30)  # older Ultralytics
    if cfg["tracker_type"] == "botsort":
        tracker.gmc = CachedGMC()
    return tracker


def run_config(cache: Cache, persons: pd.DataFrame, cfg: dict, fps_target: float):
    idxs, fps = cache.processed_indices(fps_target)
    cfg = dict(cfg)
    cfg["track_buffer"] = max(1, int(round(cfg.pop("buffer_s") * fps)))
    tracker = build_tracker(cfg)
    by_ci = {ci: g for ci, g in persons.groupby("ci")}
    dummy = np.zeros((2, 2, 3), dtype=np.uint8)
    out = []
    prev_ci = None
    for pf, ci in enumerate(idxs):
        g = by_ci.get(ci)
        if hasattr(tracker, "gmc") and isinstance(tracker.gmc, CachedGMC):
            tracker.gmc.H = cache.step_warp(prev_ci, ci)[:2, :] if prev_ci is not None else np.eye(2, 3)
        prev_ci = ci
        if g is None or len(g) == 0:
            dets = Dets(np.zeros((0, 4)), [], [])
            rowids = np.zeros(0, dtype=int)
        else:
            dets = Dets(g[["x1", "y1", "x2", "y2"]].to_numpy(), g.conf.to_numpy(), np.zeros(len(g)))
            rowids = g.index.to_numpy()
        try:
            tracks = tracker.update(dets, dummy)
        except TypeError:
            tracks = tracker.update(dets)
        for t in np.asarray(tracks).reshape(-1, np.asarray(tracks).shape[-1] if len(tracks) else 8):
            det_row = rowids[int(t[7])] if len(rowids) and int(t[7]) < len(rowids) else -1
            out.append((pf, ci, int(t[4]), float(t[0]), float(t[1]), float(t[2]), float(t[3]), float(t[5]), det_row))
    tr = pd.DataFrame(out, columns=["pf", "ci", "track_id", "x1", "y1", "x2", "y2", "conf", "det_row"])
    return tr, fps, len(idxs)


def make_grid(kind: str):
    """Configs as dicts. buffer_s is memory in seconds, converted to frames per fps."""
    base = dict(track_high_thresh=0.25, new_track_thresh=0.25, buffer_s=6.0, match_thresh=0.8)
    if kind == "full":
        grid = []
        for hi, buf, mt in itertools.product([0.25, 0.4, 0.5], [3.0, 6.0, 12.0], [0.7, 0.8, 0.9]):
            grid.append(dict(track_high_thresh=hi, new_track_thresh=hi, buffer_s=buf, match_thresh=mt))
        return grid
    if kind == "wide":
        # Wider than "full": on real footage the best configs sat at the edge of the quick and full grids.
        # The baseline goes first so the "Phase 1 equivalent" row is still reported.
        grid = [base]
        for hi, buf, mt in itertools.product([0.5, 0.6, 0.7], [12.0, 20.0, 30.0], [0.9, 0.95]):
            grid.append(dict(track_high_thresh=hi, new_track_thresh=hi, buffer_s=buf, match_thresh=mt))
        return grid
    return [
        base,
        dict(base, track_high_thresh=0.5, new_track_thresh=0.5),
        dict(base, buffer_s=12.0),
        dict(base, match_thresh=0.9),
        dict(base, track_high_thresh=0.5, new_track_thresh=0.5, buffer_s=12.0, match_thresh=0.9),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path, help="run folder that contains cache/")
    ap.add_argument("--grid", choices=["quick", "full", "wide"], default="quick")
    ap.add_argument("--fps-list", default="10,15", help="replay rates, limited to what the cache supports")
    ap.add_argument(
        "--trackers",
        default="botsort",
        help="comma list. ByteTrack is opt-in: it gave 3 to 5 times more IDs and was far slower at 30 fps",
    )
    ap.add_argument("--person-floor", type=float, default=0.1, help="drop person detections below this before tracking")
    ap.add_argument("--expected-players", type=int, default=23)
    args = ap.parse_args()
    if getattr(args, "run", None) is not None:
        args.run = require_under_data(args.run)

    cache = Cache(args.run / "cache")
    persons = cache.det[(cache.det.cls == PERSON) & (cache.det.conf >= args.person_floor)].reset_index(drop=True)
    print(
        f"Cache: {cache.n} frames at {cache.fps:.1f} fps, {len(persons)} person detections, "
        f"camera motion: {cache.has_camera}"
    )
    if not cache.has_camera:
        print(
            "WARNING: no camera.csv in the cache. BoT-SORT behaves like ByteTrack and swap suspects are not comparable."
        )

    fps_list = sorted(
        {
            round(cache.processed_indices(float(f))[1], 3)
            for f in args.fps_list.split(",")
            if float(f) <= cache.fps + 0.5
        }
    )
    grid = make_grid(args.grid)
    baseline_key = (grid[0]["track_high_thresh"], grid[0]["buffer_s"], grid[0]["match_thresh"])
    results, best = [], None
    total = len(fps_list) * len(grid) * len(args.trackers.split(","))
    done = 0
    t0 = time.time()
    for tname in args.trackers.split(","):
        for fps_target in fps_list:
            for params in grid:
                cfg = dict(params, tracker_type=tname)
                tr, fps, n_proc = run_config(cache, persons, cfg, fps_target)
                m = track_metrics(tr, fps, n_proc, cache, args.expected_players) if len(tr) else {"score": 1e9}
                row = dict(
                    tracker=tname,
                    fps=round(fps, 1),
                    high_thresh=params["track_high_thresh"],
                    buffer_s=params["buffer_s"],
                    match_thresh=params["match_thresh"],
                    **m,
                )
                row["is_phase1_equivalent"] = (
                    params["track_high_thresh"],
                    params["buffer_s"],
                    params["match_thresh"],
                ) == baseline_key and abs(fps - 10) < 0.6
                results.append(row)
                done += 1
                print(
                    f"[{done}/{total}] {tname} {fps:.0f}fps hi={params['track_high_thresh']} buf={params['buffer_s']}s "
                    f"mt={params['match_thresh']}: ids={m.get('unique_ids')} new/min={m.get('new_ids_per_min')} "
                    f"swaps/min={m.get('swap_suspects_per_min')} score={m['score']}  ({time.time() - t0:.0f}s)"
                )
                if best is None or m["score"] < best[0]:
                    best = (m["score"], cfg, fps_target, tr, fps)

    res = pd.DataFrame(results).sort_values("score").reset_index(drop=True)
    res.to_csv(args.run / "sweep_results.csv", index=False)

    _, cfg, fps_target, tr, fps = best
    # det_row indexes `persons` (reset_index above), not cache.det, so join against persons.
    tr = tr.join(persons[COLOR_COLS], on="det_row")
    tr.to_csv(args.run / "best_tracklets.csv.gz", index=False)
    (args.run / "best_config.json").write_text(json.dumps(dict(cfg, fps=fps, note="lowest proxy score"), indent=2))

    top = res.head(5)
    base = res[res.is_phase1_equivalent]
    lines = [
        "# Tracker replay report",
        "",
        f"Cache: {cache.n} frames at {cache.fps:.1f} fps. Configs tested: {len(res)}. Grid: {args.grid}.",
        "",
        "Score = new IDs per minute + 3 x swap suspects per minute. Lower is better. Proxy only, no ground truth.",
        f"Camera motion in cache: {cache.has_camera}. "
        f"Median inliers per frame: {cache.meta.get('camera_inlier_median', 'n/a')}.",
        "",
    ]
    cols = [
        "tracker",
        "fps",
        "high_thresh",
        "buffer_s",
        "match_thresh",
        "unique_ids",
        "new_ids_per_min",
        "swap_suspects_per_min",
        "births_that_look_like_restarts",
        "median_track_s",
        "pct_rows_in_tracks_10s_plus",
        "score",
    ]
    lines += ["## Top 5", "", top[cols].to_markdown(index=False), ""]
    if len(base):
        lines += ["## Phase 1 equivalent configs (10 fps, defaults)", "", base[cols].to_markdown(index=False), ""]
    lines += [
        "## Best per tracker and fps",
        "",
        res.sort_values("score").groupby(["tracker", "fps"], as_index=False).first()[cols].to_markdown(index=False),
        "",
    ]
    (args.run / "tracker_report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"Wrote {args.run / 'sweep_results.csv'}, best_tracklets.csv.gz, best_config.json, tracker_report.md")


if __name__ == "__main__":
    main()
