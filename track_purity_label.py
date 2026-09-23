"""Blind check of tracker configs: is each sampled tracklet one person, or did the tracker swap onto another?

Why: replay_trackers.py's proxy score (fewest IDs) rewards lenient trackers, and the "wide" pick turned out to
produce long tracklets that were often two people. identity_metrics() in replay_trackers.py only sees swaps
between two already-identified players. This is the direct measurement: the owner looks at a tracklet's crops
in time order and says whether it stays on one person.

`label` samples --n tracklets from each listed config of a replay_trackers.py --sweep, weighted by duration
(long tracklets carry most of the player-time, and are where swaps hide), skipping tracklets under --min-s and
ones that barely move (bench, coaches: a static tracklet is trivially "one person" and would flatter every
config). All configs are shuffled together and the config is never shown, so the labels are blind.

Every crop starts as person A. When the box is on someone else in some crops, group them by person:

  click            cycle that crop's person: A -> B -> ... -> F -> A
  shift+click      a swap here: this crop and every later one move to the next person
  ctrl+click       "?" for that crop (cannot tell who it is), ctrl+click again to undo
  Enter            save the grouping: one person if every crop is A (or ?), else two or more
  p                one person the whole time (resets any grouping)
  x                two or more people, but too hard to group
  s                skip, cannot tell
  m                more crops, if they do not all fit on one page
  b                back one tracklet (undo the last label)
  q                save and quit
  mouse wheel      zoom in or out, centered on the cursor
  right click      pan the zoomed view to that spot
  r                reset zoom (zoom stays set across tracklets otherwise, since the crop grid is the same). The
                   window grows with zoom up to the screen size, then magnifies inside it

Crops are in time order, left to right then top to bottom, with the time in seconds on each; the green box is
the tracked person, the colored frame and letter are the person group. A swap usually shows as a change of kit,
build, or position in the group between two consecutive crops. Progress is saved after every tracklet, so
rerunning `label` resumes. Tracklets marked "two or more" before grouping existed come back once, at the end.

Why group: each crop is one detection (det_row), so the groups are ground truth that does not depend on the
tracker config: two crops in different groups are different people under any config.

`score` reports, per config: share of tracklets that are one person, the same weighted by duration, a 95%
interval (samples are small), and from the grouped ones: the share of tracked time that belongs to the main
person, switches per tracklet, and how many switches sit at a detection gap of 1 s or more (a lost track
re-found by someone else) versus in continuous tracking (two players in contact).

Outputs (git-ignored under data/): RUN/sweeps/NAME/purity_items.json, purity_truth.csv, purity_crops.csv,
purity_score.json.
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

from sv_common import Cache, ZoomView, cache_stride, read_frames, require_under_data, screen_size

WINDOW = "track purity"
CROP_W, CROP_H = 220, 320
ROWS = 2
N_CROPS = 32  # sampled per tracklet
# as many columns as fit the screen (an ultrawide shows all 32 crops at once), 4 to 16
COLS = int(np.clip(screen_size()[0] // CROP_W, 4, N_CROPS // ROWS))
PER_PAGE = COLS * ROWS
TRUTH_COLS = ["item", "config", "track_id", "duration_s", "verdict", "grouped"]
CROP_COLS = ["item", "config", "track_id", "crop", "ci", "det_row", "group"]
GROUPS = "ABCDEF"
GROUP_COLOR = {
    "A": (200, 200, 200), "B": (0, 140, 255), "C": (255, 0, 255), "D": (255, 255, 0), "E": (0, 0, 255),
    "F": (0, 255, 128), "?": (60, 60, 60),
}  # fmt: skip


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
            idx = np.linspace(0, len(d) - 1, min(N_CROPS, len(d))).round().astype(int)
            rows = d.iloc[idx][["ci", "det_row", "x1", "y1", "x2", "y2"]].to_dict("records")
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


def add_det_rows(run, sweep: str, items: list) -> bool:
    """Fill det_row into samples drawn before grouping existed (same ci and track, so the same detection)."""
    missing = [it for it in items if "det_row" not in it["rows"][0]]
    for c in sorted({it["config"] for it in missing}):
        tr = pd.read_csv(run / "sweeps" / sweep / f"tracks_{c}.csv.gz", usecols=["track_id", "ci", "det_row"])
        lookup = tr.set_index(["track_id", "ci"]).det_row
        for it in missing:
            if it["config"] == c:
                for row in it["rows"]:
                    row["det_row"] = int(lookup.get((it["track_id"], row["ci"]), -1))
    return bool(missing)


def switch_to_next(groups: list, j: int) -> None:
    """A swap at crop j: j and every later crop move to the person after j's (unknown crops stay unknown)."""
    base = groups[j] if groups[j] in GROUPS else "A"
    nxt = GROUPS[min(GROUPS.index(base) + 1, len(GROUPS) - 1)]
    for k in range(j, len(groups)):
        if groups[k] != "?":
            groups[k] = nxt


def cycle(groups: list, j: int) -> None:
    g = groups[j]
    groups[j] = GROUPS[(GROUPS.index(g) + 1) % len(GROUPS)] if g in GROUPS else "A"


def toggle_unknown(groups: list, j: int) -> None:
    groups[j] = "A" if groups[j] == "?" else "?"


def render(tiles, item, index, total, label, zv: ZoomView, page, fps_cache, groups) -> np.ndarray:
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
            color = GROUP_COLOR[groups[j]]
            cv2.rectangle(t, (0, 0), (CROP_W - 1, CROP_H - 1), color, 4)
            cv2.putText(t, groups[j], (CROP_W - 34, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
            cells.append(t)
        else:
            cells.append(blank)
    show = zv.apply(np.vstack([np.hstack(cells[r * COLS : (r + 1) * COLS]) for r in range(ROWS)]))
    sh, sw = show.shape[:2]
    status = f" [{label}]" if label else ""
    counts = " ".join(f"{g}:{groups.count(g)}" for g in GROUPS + "?" if groups.count(g))
    pages = f"  page {page + 1}/{n_pages}" if n_pages > 1 else ""
    if GROUPS[-1] in groups:
        counts += f" (max {len(GROUPS)} people: a further swap cannot get its own letter, press x instead)"
    text = (
        f"{index + 1}/{total}  {item['duration_s']:.0f}s tracklet{status}  {counts}{pages}{zv.label()}   "
        "click cycle person | shift+click swap here | ctrl+click ? | Enter save | p one person | "
        "x two+ ungrouped | s skip | b back | wheel zoom | right-click pan | r reset | q quit"
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
    if add_det_rows(args.run, args.sweep, items):
        items_path.write_text(json.dumps(items))
    crops_path = sdir / "purity_crops.csv"
    labels, grouped, groups = {}, {}, {}
    if truth_path.exists():
        t = pd.read_csv(truth_path)
        if "grouped" not in t:
            t["grouped"] = np.nan  # labeled before grouping existed
        for r in t.itertuples():
            labels[int(r.item)] = r.verdict
            if not pd.isna(r.grouped):
                grouped[int(r.item)] = bool(r.grouped)
    if crops_path.exists():
        for i, d in pd.read_csv(crops_path).groupby("item"):
            groups[int(i)] = d.sort_values("crop").group.tolist()

    def save():
        rows = [
            dict(item=i, config=items[i]["config"], track_id=items[i]["track_id"],
                 duration_s=items[i]["duration_s"], verdict=v, grouped=grouped.get(i))
            for i, v in sorted(labels.items())
        ]  # fmt: skip
        pd.DataFrame(rows, columns=TRUTH_COLS).to_csv(truth_path, index=False)
        crows = [
            dict(item=i, config=items[i]["config"], track_id=items[i]["track_id"], crop=j,
                 ci=items[i]["rows"][j]["ci"], det_row=items[i]["rows"][j]["det_row"], group=g)
            for i in sorted(groups) if grouped.get(i) and i in labels for j, g in enumerate(groups[i])
        ]  # fmt: skip
        pd.DataFrame(crows, columns=CROP_COLS).to_csv(crops_path, index=False)

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

    # unlabeled first, then (once) anything marked "two or more" before grouping existed
    todo = [i for i in range(len(items)) if i not in labels]
    todo += [i for i in range(len(items)) if labels.get(i) == "mixed" and i not in grouped]
    zv = ZoomView()
    state = dict(pos=0, page=0)
    history = []  # positions in todo, for b

    def cur_groups():
        i = todo[state["pos"]]
        if i not in groups or len(groups[i]) != len(items[i]["rows"]):
            groups[i] = ["A"] * len(items[i]["rows"])
        return groups[i]

    def on_mouse(event, mx, my, flags, _p):
        if zv.on_mouse(event, mx, my, flags) or event != cv2.EVENT_LBUTTONDOWN or state["pos"] >= len(todo):
            return
        cx, cy = zv.to_content(mx, my)
        col, row = int(cx // CROP_W), int(cy // CROP_H)
        j = state["page"] * PER_PAGE + row * COLS + col
        g = cur_groups()
        if not (0 <= col < COLS and 0 <= row < ROWS and j < len(g)):
            return
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            switch_to_next(g, j)
        elif flags & cv2.EVENT_FLAG_CTRLKEY:
            toggle_unknown(g, j)
        else:
            cycle(g, j)

    def finish(i, verdict, is_grouped):
        labels[i], grouped[i] = verdict, is_grouped
        save()
        history.append(state["pos"])
        state["pos"], state["page"] = state["pos"] + 1, 0

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        while state["pos"] < len(todo):
            i = todo[state["pos"]]
            item, g, page = items[i], cur_groups(), state["page"]
            n_pages = max(1, -(-len(item["rows"]) // PER_PAGE))
            show = render(tiles[i], item, state["pos"], len(todo), labels.get(i), zv, page, fps_cache, g)
            cv2.imshow(WINDOW, show)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10):  # Enter: save the grouping
                finish(i, "pure" if len({x for x in g if x in GROUPS}) <= 1 else "mixed", True)
            elif key == ord("p"):
                groups[i] = ["A"] * len(item["rows"])
                finish(i, "pure", True)
            elif key == ord("x"):
                finish(i, "mixed", False)
            elif key == ord("s"):
                finish(i, "skipped", False)
            elif key == ord("m"):
                state["page"] = (page + 1) % n_pages
            elif key == ord("b") and history:
                state["pos"], state["page"] = history.pop(), 0
                j = todo[state["pos"]]
                labels.pop(j, None)
                grouped.pop(j, None)
                save()
            elif key == ord("r"):
                zv.reset()
            elif key == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        save()
    print(f"Saved {len(labels)} of {len(items)} labels to {truth_path}, groups to {crops_path}. Run `score`.")


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def group_metrics(run, sweep: str, d: pd.DataFrame, crops: pd.DataFrame | None, cache_fps: float) -> dict:
    """From grouped tracklets of one config: how much of the time is the main person, and where swaps happen."""
    is_grouped = d.grouped.fillna(False).astype(bool)
    g = d[(d.verdict == "mixed") & is_grouped]
    ungrouped = (d.verdict == "mixed") & ~is_grouped
    out = dict(
        mixed_grouped=len(g),
        mixed_ungrouped=int(ungrouped.sum()),
        # main_person_time_pct leaves these out (unknown split), and they are likely the worst ones: report
        # their share of labeled time so the percentage is read with that in mind
        mixed_ungrouped_time_pct=round(100 * float(d.duration_s[ungrouped].sum() / d.duration_s.sum()), 1),
    )
    # one-person tracklets are 100% main person whether or not they went through grouping; ungrouped "two or
    # more" ones are left out of the time share, since how much of them is the main person is unknown
    shares = [(r.duration_s, 1.0) for r in d[d.verdict == "pure"].itertuples()]
    switches, at_gap = 0, 0
    if crops is not None and len(g):
        tr = pd.read_csv(run / "sweeps" / sweep / f"tracks_{int(g.config.iloc[0])}.csv.gz", usecols=["track_id", "ci"])
    for r in g.itertuples() if crops is not None else []:
        k = crops[crops.item == r.item].sort_values("crop")
        k = k[k.group != "?"]
        if not len(k):
            continue
        shares.append((r.duration_s, k.group.value_counts().iloc[0] / len(k)))
        track_ci = np.sort(tr.ci[tr.track_id == r.track_id].unique())
        for a, b in zip(k.iloc[:-1].itertuples(), k.iloc[1:].itertuples(), strict=True):
            if a.group == b.group:
                continue
            switches += 1
            between = track_ci[(track_ci >= a.ci) & (track_ci <= b.ci)]
            if len(between) > 1 and np.diff(between).max() >= cache_fps:  # a 1 s+ hole: lost, then re-found
                at_gap += 1
    if shares:
        dur = np.array([s[0] for s in shares])
        out["main_person_time_pct"] = round(100 * float((dur * np.array([s[1] for s in shares])).sum() / dur.sum()), 1)
        out["switches_per_min"] = round(switches / (dur.sum() / 60), 2)
        out["switches"] = switches
        out["switches_at_1s_gap"] = at_gap
    return out


def cmd_score(args) -> None:
    sdir = args.run / "sweeps" / args.sweep
    check_configs(args.run, args.sweep, json.loads((sdir / "purity_items.json").read_text()))
    t = pd.read_csv(sdir / "purity_truth.csv")
    if "grouped" not in t:
        t["grouped"] = False
    crops = pd.read_csv(sdir / "purity_crops.csv") if (sdir / "purity_crops.csv").exists() else None
    res = pd.read_csv(sdir / "results.csv").set_index("config") if (sdir / "results.csv").exists() else None
    cache_fps = Cache(args.run / "cache").fps
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
        row.update(group_metrics(args.run, args.sweep, d, crops, cache_fps))
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
