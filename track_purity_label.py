"""Blind check of tracker configs: is each sampled tracklet one person, or did the tracker swap onto another?

Why: replay_trackers.py's proxy score (fewest IDs) rewards lenient trackers, and the "wide" pick turned out to
produce long tracklets that were often two people. identity_metrics() in replay_trackers.py only sees swaps
between two already-identified players. This is the direct measurement: the owner looks at a tracklet's crops
in time order and says whether it stays on one person.

`label` samples --n tracklets from each listed config of a replay_trackers.py --sweep, weighted by duration
(long tracklets carry most of the player-time, and are where swaps hide), skipping tracklets under --min-s and
ones that barely move (bench, coaches: a static tracklet is trivially "one person" and would flatter every
config). All configs are shuffled together and the config is never shown, so the labels are blind.

  p                one person the whole time
  x                two or more different people (the box moves to someone else at some point)
  s                skip, cannot tell
  m                more crops: page through further sampled frames of the same tracklet
  b                back one tracklet (undo the last label)
  q                save and quit
  mouse wheel      zoom, r resets

Crops are in time order, left to right then top to bottom, with the time in seconds on each; the green box is
the tracked person. A swap usually shows as a change of kit, build, or position in the group between two
consecutive crops. Progress is saved after every tracklet, so rerunning `label` resumes.

`score` reports, per config: share of tracklets that are one person, the same weighted by duration (the share
of tracked player-time that is clean), and a 95% interval, because samples are small.

Outputs (git-ignored under data/): RUN/sweeps/NAME/purity_items.json, purity_truth.csv, purity_score.json.
The crops show minors: keep them local.

Example:
  python track_purity_label.py label --run data\\clipA --sweep strict --configs 1,9,14 --n 25
  python track_purity_label.py score --run data\\clipA --sweep strict
"""

import argparse
import json
import random

import cv2
import numpy as np
import pandas as pd

from sv_common import Cache, cache_stride, read_frames, require_under_data

WINDOW = "track purity"
CROP_W, CROP_H = 220, 320
COLS, ROWS = 4, 2
PER_PAGE = COLS * ROWS
PAGES = 4
TRUTH_COLS = ["item", "config", "track_id", "duration_s", "verdict"]
ZOOM_STEP, MIN_ZOOM, MAX_ZOOM = 1.25, 0.5, 3.0


def moving_tracks(tr: pd.DataFrame, cache: Cache, min_extent_h: float = 0.75) -> set:
    """Track IDs whose path spreads at least min_extent_h body heights (camera motion removed).

    Same idea as team_classify.py's sideline_suspect: people at the sideline sit still, players do not.
    """
    sx, sy = cache.to_stable(tr.ci.to_numpy(), ((tr.x1 + tr.x2) / 2).to_numpy(), tr.y2.to_numpy())
    d = pd.DataFrame(dict(track_id=tr.track_id.to_numpy(), sx=sx, sy=sy, h=(tr.y2 - tr.y1).to_numpy()))
    g = d.groupby("track_id")
    extent = np.hypot(g.sx.max() - g.sx.min(), g.sy.max() - g.sy.min()) / g.h.median().clip(lower=1)
    return set(extent[extent >= min_extent_h].index)


def sample_items(run, sweep: str, configs: list, n: int, min_s: float, seed: int) -> list:
    rng = random.Random(seed)
    cache = Cache(run / "cache")
    items = []
    for c in configs:
        tr = pd.read_csv(run / "sweeps" / sweep / f"tracks_{c}.csv.gz")
        params = json.loads((run / "sweeps" / sweep / f"config_{c}.json").read_text())
        fps = params["fps"]
        g = tr.groupby("track_id").pf
        dur = (g.max() - g.min() + 1) / fps
        ok = dur[(dur >= min_s) & dur.index.isin(moving_tracks(tr, cache))]
        pool, weights, chosen = list(ok.index), list(ok.to_numpy()), []
        while pool and len(chosen) < n:  # duration-weighted, without replacement
            k = rng.choices(range(len(pool)), weights=weights)[0]
            chosen.append(pool.pop(k))
            weights.pop(k)
        for tid in chosen:
            d = tr[tr.track_id == tid].sort_values("ci").reset_index(drop=True)
            idx = np.linspace(0, len(d) - 1, min(PER_PAGE * PAGES, len(d))).round().astype(int)
            rows = d.iloc[idx][["ci", "x1", "y1", "x2", "y2"]].to_dict("records")
            items.append(
                dict(config=int(c), params=params, track_id=int(tid), duration_s=round(float(ok[tid]), 1), rows=rows)
            )
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["item"] = i
    return items


def crop(img, row) -> np.ndarray:
    cx, cy = int((row["x1"] + row["x2"]) / 2), int(row["y2"])
    x0 = int(np.clip(cx - CROP_W / 2, 0, img.shape[1] - CROP_W))
    y0 = int(np.clip(cy - CROP_H * 0.8, 0, img.shape[0] - CROP_H))
    out = img[y0 : y0 + CROP_H, x0 : x0 + CROP_W].copy()
    p1 = (int(row["x1"]) - x0, int(row["y1"]) - y0)
    p2 = (int(row["x2"]) - x0, int(row["y2"]) - y0)
    cv2.rectangle(out, p1, p2, (0, 255, 0), 2)
    return out


def render(tiles, item, index, total, label, zoom, page, fps_cache) -> np.ndarray:
    rows = item["rows"]
    n_pages = max(1, -(-len(rows) // PER_PAGE))
    page_rows = list(range(page * PER_PAGE, min((page + 1) * PER_PAGE, len(rows))))
    blank = np.zeros((CROP_H, CROP_W, 3), np.uint8)
    cells = []
    for k in range(PER_PAGE):
        if k < len(page_rows):
            j = page_rows[k]
            t = tiles[j].copy()
            txt = f"{(rows[j]['ci'] - rows[0]['ci']) / fps_cache:.1f}s"
            cv2.putText(t, txt, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cells.append(t)
        else:
            cells.append(blank)
    show = np.vstack([np.hstack(cells[r * COLS : (r + 1) * COLS]) for r in range(ROWS)])
    if zoom != 1.0:
        show = cv2.resize(show, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC if zoom > 1 else cv2.INTER_AREA)
    sh, sw = show.shape[:2]
    status = f" [{label}]" if label else ""
    text = (
        f"{index + 1}/{total}  {item['duration_s']:.0f}s tracklet{status}  page {page + 1}/{n_pages}   "
        "p one person | x two+ people | s skip | m more | b back | wheel zoom | q quit"
    )
    cv2.rectangle(show, (0, sh - 24), (sw, sh), (0, 0, 0), -1)
    cv2.putText(show, text, (6, sh - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return show


def check_configs(run, sweep: str, items: list) -> None:
    """Exit if a config number now means different settings than when the sample was drawn (sweep was rerun)."""
    for c in sorted({it["config"] for it in items}):
        path = run / "sweeps" / sweep / f"config_{c}.json"
        now = json.loads(path.read_text()) if path.exists() else None
        then = next(it.get("params") for it in items if it["config"] == c)
        if then is None or now != then:
            raise SystemExit(
                f"Config {c} in {sweep} no longer matches the sampled tracklets (sweep rerun or old sample). "
                "Delete purity_items.json and purity_truth.csv to start over."
            )


def cmd_label(args) -> None:
    sdir = args.run / "sweeps" / args.sweep
    items_path, truth_path = sdir / "purity_items.json", sdir / "purity_truth.csv"
    if items_path.exists():
        items = json.loads(items_path.read_text())
        check_configs(args.run, args.sweep, items)
        print(f"Resuming the saved sample in {items_path} (delete it to draw a new one).")
    else:
        configs = [int(c) for c in args.configs.split(",")]
        items = sample_items(args.run, args.sweep, configs, args.n, args.min_s, args.seed)
        items_path.write_text(json.dumps(items))
    labels = {}
    if truth_path.exists():
        labels = {int(r.item): r.verdict for r in pd.read_csv(truth_path).itertuples()}

    def save():
        rows = [
            dict(item=i, config=items[i]["config"], track_id=items[i]["track_id"],
                 duration_s=items[i]["duration_s"], verdict=v)
            for i, v in sorted(labels.items())
        ]  # fmt: skip
        pd.DataFrame(rows, columns=TRUTH_COLS).to_csv(truth_path, index=False)

    stride = cache_stride(args.run)
    fps_cache = Cache(args.run / "cache").fps
    print(f"{len(items)} tracklets, {len(labels)} already labeled. Loading crops...")
    by_frame = {}
    for i, it in enumerate(items):
        for j, row in enumerate(it["rows"]):
            by_frame.setdefault(row["ci"] * stride, []).append((i, j))
    tiles = {i: [None] * len(it["rows"]) for i, it in enumerate(items)}
    for f, img in read_frames(args.run / "clip.mp4", list(by_frame)):
        for i, j in by_frame[f]:
            tiles[i][j] = crop(img, items[i]["rows"][j])
    for i in tiles:
        tiles[i] = [t if t is not None else np.zeros((CROP_H, CROP_W, 3), np.uint8) for t in tiles[i]]

    idx = next((i for i in range(len(items)) if i not in labels), len(items))
    zoom, page = [1.0], 0

    def on_mouse(event, _x, _y, flags, _p):
        if event == cv2.EVENT_MOUSEWHEEL:
            zoom[0] = float(np.clip(zoom[0] * (ZOOM_STEP if flags > 0 else 1 / ZOOM_STEP), MIN_ZOOM, MAX_ZOOM))

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    verdicts = {ord("p"): "pure", ord("x"): "mixed", ord("s"): "skipped"}
    try:
        while idx < len(items):
            item = items[idx]
            n_pages = max(1, -(-len(item["rows"]) // PER_PAGE))
            cv2.imshow(WINDOW, render(tiles[idx], item, idx, len(items), labels.get(idx), zoom[0], page, fps_cache))
            key = cv2.waitKey(30) & 0xFF
            if key in verdicts:
                labels[idx] = verdicts[key]
                save()
                idx, page = idx + 1, 0
            elif key == ord("m"):
                page = (page + 1) % n_pages
            elif key == ord("b") and idx > 0:
                idx, page = idx - 1, 0
                labels.pop(idx, None)
                save()
            elif key == ord("r"):
                zoom[0] = 1.0
            elif key == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        save()
    print(f"Saved {len(labels)} of {len(items)} labels to {truth_path}. Run `score` when done.")


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def cmd_score(args) -> None:
    sdir = args.run / "sweeps" / args.sweep
    check_configs(args.run, args.sweep, json.loads((sdir / "purity_items.json").read_text()))
    t = pd.read_csv(sdir / "purity_truth.csv")
    res = pd.read_csv(sdir / "results.csv").set_index("config") if (sdir / "results.csv").exists() else None
    out = {}
    for c, d in t[t.verdict != "skipped"].groupby("config"):
        k, n = int((d.verdict == "pure").sum()), len(d)
        row = dict(
            labeled=n,
            skipped=int(((t.config == c) & (t.verdict == "skipped")).sum()),
            pure_pct=round(100 * k / n, 1),
            pure_pct_95ci=wilson(k, n),
            pure_time_pct=round(100 * d.duration_s[d.verdict == "pure"].sum() / d.duration_s.sum(), 1),
            median_sampled_s=float(d.duration_s.median()),
        )
        if res is not None and c in res.index:
            r = res.loc[c]
            row["config"] = dict(
                high=r.high_thresh, buffer_s=r.buffer_s, match=r.match_thresh, reid=bool(r.reid),
                proximity=None if pd.isna(r.proximity) else r.proximity, veto=None if pd.isna(r.veto) else r.veto,
                unique_ids=int(r.unique_ids), median_track_s=r.median_track_s,
            )  # fmt: skip
        out[int(c)] = row
    (sdir / "purity_score.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lab = sub.add_parser("label", help="blind pure/mixed labels on tracklets sampled from several configs")
    lab.add_argument("--run", required=True, type=require_under_data)
    lab.add_argument("--sweep", required=True)
    lab.add_argument("--configs", help="comma list of config numbers from the sweep's results.csv")
    lab.add_argument("--n", type=int, default=25, help="tracklets per config")
    lab.add_argument("--min-s", type=float, default=2.0, help="skip tracklets shorter than this")
    lab.add_argument("--seed", type=int, default=0)
    lab.set_defaults(fn=cmd_label)
    sc = sub.add_parser("score", help="purity per config")
    sc.add_argument("--run", required=True, type=require_under_data)
    sc.add_argument("--sweep", required=True)
    sc.set_defaults(fn=cmd_score)
    args = ap.parse_args()
    if (
        args.cmd == "label"
        and not args.configs
        and not (args.run / "sweeps" / args.sweep / "purity_items.json").exists()
    ):
        ap.error("label needs --configs the first time")
    args.fn(args)


if __name__ == "__main__":
    main()
