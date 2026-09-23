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
import shutil
import time

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


def veto_botsort_class():
    """BoT-SORT whose appearance term can also forbid a match, not only make one cheaper.

    Stock BoT-SORT takes min(box cost, appearance cost), so appearance can only rescue a match and never blocks a
    swap onto a different-looking person. With reid_veto set, any track/detection pair whose cosine distance
    (track's smoothed embedding vs the detection's) is above it gets cost 1.0, which no match threshold accepts.
    Applies to the first association and to unconfirmed tracks. The low-score second association stays IoU-only
    (it needs IoU 0.5 or more with an actively tracked box, so it is not where long-gap swaps come from).

    refind_only: veto only tracks that are lost (being re-found after a gap), never frame-to-frame continuation.
    Vetoing every frame shattered tracks on clipA (veto 0.4: median track 0.5 s), because per-frame embeddings of
    60 to 90 px players are noisy and about 15 associations per second compound even a small false-block rate.
    Swaps mostly happen at re-finds anyway (see make_grid).
    """
    from ultralytics.trackers.basetrack import TrackState
    from ultralytics.trackers.bot_sort import BOTSORT

    class VetoBOTSORT(BOTSORT):
        reid_veto = None
        refind_only = False

        def get_dists(self, tracks, detections):
            dists = super().get_dists(tracks, detections)
            if self.reid_veto is None or dists.size == 0:
                return dists
            tf = [t.smooth_feat for t in tracks]
            df = [d.curr_feat for d in detections]
            dim = next((len(f) for f in (*tf, *df) if f is not None), 0)
            if not dim:
                return dists
            T = np.array([f if f is not None else np.full(dim, np.nan) for f in tf], dtype=np.float32)
            D = np.array([f if f is not None else np.full(dim, np.nan) for f in df], dtype=np.float32)
            T /= np.linalg.norm(T, axis=1, keepdims=True)
            D /= np.linalg.norm(D, axis=1, keepdims=True)
            cos_dist = 1.0 - T @ D.T  # NaN where either side has no embedding: never vetoed
            block = cos_dist > self.reid_veto
            if self.refind_only:
                block &= np.array([t.state != TrackState.Tracked for t in tracks])[:, None]
            dists = dists.copy()
            dists[block] = 1.0
            return dists

    return VetoBOTSORT


def build_tracker(cfg: dict):
    from types import SimpleNamespace

    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.trackers.byte_tracker import BYTETracker

    cfg = dict(cfg)
    reid = bool(cfg.pop("reid", False))
    veto = cfg.pop("reid_veto", None)
    refind_only = bool(cfg.pop("refind_only", False))
    if reid and cfg["tracker_type"] != "botsort":
        raise SystemExit(
            "ReID and veto configs need BoT-SORT; ByteTrack has no appearance term. Use --trackers botsort."
        )
    if reid:
        cfg.update(with_reid=True, model="auto")  # "auto": features come precomputed from reid_cache.py
    args = SimpleNamespace(**{**DEFAULTS, **cfg})
    if cfg["tracker_type"] != "botsort":
        cls = BYTETracker
    elif veto is not None and not pd.isna(veto):
        cls = veto_botsort_class()
        cls.reid_veto = float(veto)
        cls.refind_only = refind_only
    else:
        cls = BOTSORT
    try:
        tracker = cls(args)  # newer Ultralytics
    except TypeError:
        tracker = cls(args, frame_rate=30)  # older Ultralytics
    if cfg["tracker_type"] == "botsort":
        tracker.gmc = CachedGMC()
    return tracker


def load_reid(path, persons: pd.DataFrame) -> np.ndarray:
    """Embeddings from reid_cache.py aligned to `persons` rows; zero rows (= no feature) where none was cached."""
    z = np.load(path)
    pos = pd.Series(np.arange(len(persons)), index=persons["det_idx"].to_numpy())
    emb = np.zeros((len(persons), z["emb"].shape[1]), dtype=np.float32)
    have = pos.reindex(z["det_idx"])
    ok = have.notna().to_numpy()
    emb[have[ok].astype(int).to_numpy()] = z["emb"][ok].astype(np.float32)
    return emb


def run_config(cache: Cache, persons: pd.DataFrame, cfg: dict, fps_target: float, emb: np.ndarray | None = None):
    idxs, fps = cache.processed_indices(fps_target)
    cfg = dict(cfg)
    cfg["track_buffer"] = max(1, int(round(cfg.pop("buffer_s") * fps)))
    if cfg.get("reid") and emb is None:
        raise SystemExit("A ReID config needs --reid (embeddings from reid_cache.py).")
    tracker = build_tracker(cfg)
    use_feats = bool(cfg.get("reid"))
    if use_feats:
        # Embeddings are cached for one replay rate and person floor. A mismatch would silently leave most
        # detections without a feature (never matched by appearance, never vetoed) and look like "ReID didn't help".
        rows = persons.ci.isin(idxs).to_numpy()
        coverage = float(np.any(emb[rows] != 0, axis=1).mean()) if rows.any() else 0.0
        if coverage < 0.95:
            raise SystemExit(
                f"Only {coverage:.0%} of detections at {fps:.1f} fps have a cached embedding. Build them with "
                "reid_cache.py at this --fps and --person-floor, or pass a matching --fps-list."
            )
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
        if use_feats:
            tracks = tracker.update(dets, dummy, emb[rowids] if len(rowids) else np.zeros((0, emb.shape[1])))
        else:
            try:
                tracks = tracker.update(dets, dummy)
            except TypeError:
                tracks = tracker.update(dets)
        for t in np.asarray(tracks).reshape(-1, np.asarray(tracks).shape[-1] if len(tracks) else 8):
            det_row = rowids[int(t[7])] if len(rowids) and int(t[7]) < len(rowids) else -1
            out.append((pf, ci, int(t[4]), float(t[0]), float(t[1]), float(t[2]), float(t[3]), float(t[5]), det_row))
    tr = pd.DataFrame(out, columns=["pf", "ci", "track_id", "x1", "y1", "x2", "y2", "conf", "det_row"])
    return tr, fps, len(idxs)


# The config the "wide" grid picked (lowest proxy score). Kept as the reference row in the strict and reid grids.
WIDE_BEST = dict(track_high_thresh=0.7, new_track_thresh=0.7, buffer_s=30.0, match_thresh=0.95)


def identity_truth(run, person_floor: float) -> pd.DataFrame | None:
    """Person-detection rows with a known jersey, from the owner's jersey_label.py pass. None if there is none.

    Taken from best_tracklets.csv.gz + player_identity.csv and frozen to identity_rows.csv.gz on first use,
    because a later run can overwrite best_tracklets.csv.gz with new track IDs. det_row is stable across tracker
    configs (it indexes person detections at or above person_floor), so the frozen labels score any config.
    Only tracklets the owner confirmed as one roster player are used, so each src_track is one known person.
    """
    frozen = run / "identity_rows.csv.gz"
    if frozen.exists():
        t = pd.read_csv(frozen)
        if float(t.person_floor.iloc[0]) != person_floor:
            raise SystemExit(f"{frozen} was built with --person-floor {t.person_floor.iloc[0]}, not {person_floor}.")
        return t
    ident, tracks = run / "player_identity.csv", run / "best_tracklets.csv.gz"
    if not (ident.exists() and tracks.exists()):
        return None
    ids = pd.read_csv(ident).dropna(subset=["jersey"])[["track_id", "jersey"]]
    t = pd.read_csv(tracks, usecols=["track_id", "det_row"]).merge(ids, on="track_id")
    t = t[t.det_row >= 0].rename(columns={"track_id": "src_track"})
    t["person_floor"] = person_floor
    t.to_csv(frozen, index=False)
    print(f"Froze {len(t)} identity-labeled detections ({t.jersey.nunique()} jerseys) to {frozen}")
    return t


def identity_metrics(tr: pd.DataFrame, truth: pd.DataFrame, fps: float) -> dict:
    """Score tracks against known identities. Lower is better for all three.

    id_impure_tracks: tracks holding two or more known jerseys (each with at least 0.3 s of rows) = a swap.
      Only swaps between two identified players are visible, so this is a lower bound.
    id_wrong_pct: percent of labeled rows sitting in a track whose majority jersey is someone else.
    id_breaks_per_min: new track IDs inside a known single-person tracklet, per labeled minute (fragmentation;
      the labels came from the "wide" pick, so it scores 0 on this by construction).
    """
    m = tr[["pf", "track_id", "det_row"]].merge(truth[["det_row", "jersey", "src_track"]], on="det_row")
    if not len(m):
        return {}
    cnt = m.groupby(["track_id", "jersey"]).size()
    solid = cnt[cnt >= max(2, round(0.3 * fps))]
    impure = int((solid.groupby(level=0).size() >= 2).sum())
    majority = cnt.groupby(level=0).idxmax().map(lambda k: k[1])
    wrong = (m.jersey != m.track_id.map(majority)).mean()
    m = m.sort_values(["src_track", "pf"])
    same_src = m.src_track.to_numpy()[1:] == m.src_track.to_numpy()[:-1]
    breaks = int((same_src & (m.track_id.to_numpy()[1:] != m.track_id.to_numpy()[:-1])).sum())
    minutes = len(m) / fps / 60
    return dict(
        id_rows=len(m),
        id_impure_tracks=impure,
        id_wrong_pct=round(100 * float(wrong), 2),
        id_breaks_per_min=round(breaks / max(minutes, 1e-6), 2),
    )


def make_grid(kind: str, vetoes=(0.4, 0.5, 0.6)):
    """Configs as dicts. buffer_s is memory in seconds, converted to frames per fps.

    "strict" and "reid" exist because "wide" won on a proxy that rewards leniency: its pick (match 0.95, 30 s
    buffer) produced long tracklets that the owner's jersey labels showed were often two people (a swap inside
    one tracklet). A short tracklet that is one person can be stitched later; a long one that is two people
    cannot. These grids go the other way and are scored on identity labels, not only on ID counts.
    """
    base = dict(track_high_thresh=0.25, new_track_thresh=0.25, buffer_s=6.0, match_thresh=0.8)
    if kind == "strict":
        grid = [dict(WIDE_BEST)]
        # Short buffers matter most: 69% of owner-confirmed "mixed" single tracklets had an internal gap of 1 s or
        # more, versus 22% of confirmed one-person ones, so most swaps look like a lost track being re-found by
        # someone else. High threshold 0.5 was worse than 0.7 on every setting in a first run, so it is dropped.
        for buf, mt in itertools.product([0.5, 1.0, 2.0, 5.0, 10.0], [0.7, 0.8, 0.9, 0.95]):
            grid.append(dict(track_high_thresh=0.7, new_track_thresh=0.7, buffer_s=buf, match_thresh=mt))
        return grid
    if kind == "refind":
        # Appearance veto applied only when a lost track is re-found. Long buffers are where it could pay off:
        # keep a player's ID across a gap, but not hand it to a different-looking person.
        grid = [dict(WIDE_BEST)]
        for buf in [1.0, 5.0, 30.0]:
            b = dict(track_high_thresh=0.7, new_track_thresh=0.7, buffer_s=buf, match_thresh=0.95)
            grid.append(dict(b))
            for v in (0.3, 0.4, 0.5):
                grid.append(dict(b, reid=True, reid_veto=v, refind_only=True, proximity_thresh=0.5))
        return grid
    if kind == "reid":
        # Stock BoT-SORT ReID (appearance can only rescue a match; proximity_thresh is the IoU it needs first),
        # and the veto variant (appearance can also forbid a match). Each against the same box-only baseline.
        grid = [dict(WIDE_BEST)]
        for buf, mt in itertools.product([1.0, 5.0, 30.0], [0.8, 0.95]):
            b = dict(track_high_thresh=0.7, new_track_thresh=0.7, buffer_s=buf, match_thresh=mt)
            grid.append(dict(b))
            grid.append(dict(b, reid=True, proximity_thresh=0.5, appearance_thresh=0.8))
            grid.append(dict(b, reid=True, proximity_thresh=0.2, appearance_thresh=0.8))
            for v in vetoes:
                grid.append(dict(b, reid=True, proximity_thresh=0.5, appearance_thresh=0.8, reid_veto=v))
        return grid
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
    ap.add_argument("--run", required=True, type=require_under_data, help="run folder that contains cache/")
    ap.add_argument("--grid", choices=["quick", "full", "wide", "strict", "reid", "refind"], default="quick")
    ap.add_argument("--fps-list", default="10,15", help="replay rates, limited to what the cache supports")
    ap.add_argument(
        "--trackers",
        default="botsort",
        help="comma list. ByteTrack is opt-in: it gave 3 to 5 times more IDs and was far slower at 30 fps",
    )
    ap.add_argument("--person-floor", type=float, default=0.1, help="drop person detections below this before tracking")
    ap.add_argument("--expected-players", type=int, default=23)
    ap.add_argument("--reid", help="embeddings .npz from reid_cache.py (needed by the reid grid)")
    ap.add_argument("--vetoes", default="0.4,0.5,0.6", help="reid grid: appearance veto cosine distances")
    ap.add_argument(
        "--sweep",
        help="name: write results and every config's tracks to RUN/sweeps/NAME/ and leave best_tracklets.csv.gz, "
        "best_config.json and everything downstream untouched",
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="replace an existing --sweep folder (and any purity labels in it)"
    )
    args = ap.parse_args()

    cache = Cache(args.run / "cache")
    persons = cache.det[(cache.det.cls == PERSON) & (cache.det.conf >= args.person_floor)]
    persons = persons.reset_index(names="det_idx")  # det_idx: row in detections.csv.gz, for the ReID cache
    emb = load_reid(args.reid, persons) if args.reid else None
    truth = identity_truth(args.run, args.person_floor)
    sweep_dir = args.run / "sweeps" / args.sweep if args.sweep else None
    if sweep_dir:
        # Config numbers restart at 0 on every run, and track_purity_label.py's hand labels refer to them, so a
        # rerun with a different grid must not silently renumber configs under existing labels.
        if sweep_dir.exists() and any(sweep_dir.iterdir()):
            if not args.overwrite:
                raise SystemExit(f"{sweep_dir} already exists. Pick another --sweep name or pass --overwrite.")
            shutil.rmtree(sweep_dir)
        sweep_dir.mkdir(parents=True, exist_ok=True)
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
    grid = make_grid(args.grid, [float(v) for v in args.vetoes.split(",")])
    baseline_key = (grid[0]["track_high_thresh"], grid[0]["buffer_s"], grid[0]["match_thresh"])
    results, best = [], None
    total = len(fps_list) * len(grid) * len(args.trackers.split(","))
    done = 0
    t0 = time.time()
    for tname in args.trackers.split(","):
        for fps_target in fps_list:
            for params in grid:
                cfg = dict(params, tracker_type=tname)
                tr, fps, n_proc = run_config(cache, persons, cfg, fps_target, emb)
                m = track_metrics(tr, fps, n_proc, cache, args.expected_players) if len(tr) else {"score": 1e9}
                if truth is not None and len(tr):
                    m.update(identity_metrics(tr, truth, fps))
                row = dict(
                    config=done,
                    tracker=tname,
                    fps=round(fps, 1),
                    high_thresh=params["track_high_thresh"],
                    buffer_s=params["buffer_s"],
                    match_thresh=params["match_thresh"],
                    reid=bool(params.get("reid", False)),
                    proximity=params.get("proximity_thresh") if params.get("reid") else None,
                    veto=params.get("reid_veto"),
                    refind_only=bool(params.get("refind_only", False)),
                    **m,
                )
                if sweep_dir:
                    tr.join(persons[COLOR_COLS], on="det_row").to_csv(sweep_dir / f"tracks_{done}.csv.gz", index=False)
                    (sweep_dir / f"config_{done}.json").write_text(json.dumps(dict(cfg, fps=fps), indent=2))
                row["is_phase1_equivalent"] = (
                    params["track_high_thresh"],
                    params["buffer_s"],
                    params["match_thresh"],
                ) == baseline_key and abs(fps - 10) < 0.6
                results.append(row)
                done += 1
                print(
                    f"[{done}/{total}] config {done - 1}: {tname} {fps:.0f}fps hi={params['track_high_thresh']} "
                    f"buf={params['buffer_s']}s mt={params['match_thresh']}: ids={m.get('unique_ids')} "
                    f"new/min={m.get('new_ids_per_min')} swaps/min={m.get('swap_suspects_per_min')} score={m['score']} "
                    f"reid={row['reid']} prox={row['proximity']} veto={row['veto']} "
                    f"id_impure={m.get('id_impure_tracks')} id_wrong%={m.get('id_wrong_pct')} "
                    f"id_breaks/min={m.get('id_breaks_per_min')}  ({time.time() - t0:.0f}s)"
                )
                if best is None or m["score"] < best[0]:
                    best = (m["score"], cfg, fps_target, tr, fps)

    res = pd.DataFrame(results).sort_values("score").reset_index(drop=True)
    if sweep_dir:
        res.to_csv(sweep_dir / "results.csv", index=False)
        print(res.drop(columns=["tracker"]).to_string(index=False))
        print(f"Wrote {sweep_dir}. best_tracklets.csv.gz and downstream files were not changed.")
        return
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
