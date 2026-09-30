"""Profile photos for the coaching site: the best crops of each identified player from the game video.

`candidates` looks through the given windows for each roster number's identified detections (player_identity.csv
and identity_segments.csv, via player_stats.identity_rows) and keeps the clearest: tallest boxes with a confident
detection, away from the frame edge, not overlapping anyone else in that frame. At most PER_TRACKLET per tracklet,
spread over the windows round-robin, CANDIDATES_PER_PLAYER per player. Each is cropped as a portrait around the
player (no box drawn), enlarged with Lanczos and lightly sharpened, and saved under data/site_photos/_candidates/.

`label` shows one player at a time, all their candidates at once (the shared crop grid and zoom):

  left click   choose this crop (yellow frame; click again to clear)
  Enter        save the choice and go to the next player        s  no photo for this player (initials instead)
  b            back one player                                  q  save and quit
  mouse wheel  zoom, keeping the point under the cursor    right click  center there    r  reset zoom
  scroll bars  appear when part of the grid is hidden: drag, or click to jump

The choice is copied to data/site_photos/player_NN.jpg; publish_site.py uploads those to the site's private
bucket. Players are 60 to 115 px tall in the video, so photos are small and soft: pick the clearest face or front.
Everything here shows minors and stays under data/ (git-ignored).

Example:
  python player_photo.py candidates --runs data\\clipA,...,data\\g0922\\w0000,...
  python player_photo.py label [--redo]
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from player_stats import ROSTER_FILE, identity_rows
from sv_common import (
    DATA_DIR,
    FOOTER_H,
    TILE_H,
    TILE_W,
    ZoomView,
    add_footer,
    cache_stride,
    grid_cols,
    read_frames,
    require_under_data,
    tile_grid,
)

OUT = DATA_DIR / "site_photos"
CAND_DIR = OUT / "_candidates"
CANDIDATES = CAND_DIR / "candidates.csv"
CHOICES = OUT / "choices.csv"
WINDOW = "player photo"
HEADER_H = 22
CANDIDATES_PER_PLAYER, PER_TRACKLET = 24, 2
MIN_CONF, MIN_H = 0.5, 50  # detector confidence and box height (px) worth a photo
EDGE_PX, MAX_OVERLAP = 12, 0.02  # away from the frame edge; share of the box another person's box may cover
PHOTO_W, PHOTO_H = 330, 480  # 11:16, the crop tools' tile shape (TILE_W x TILE_H)
PER_RUN_PREFILTER = 60  # tallest detections per player and window kept before the overlap test


def overlap_share(cand: pd.DataFrame, everyone: pd.DataFrame) -> np.ndarray:
    """For each candidate box, the largest share of it covered by another person's box in the same frame."""
    out = np.zeros(len(cand))
    others = everyone[everyone.pf.isin(set(cand.pf))].groupby("pf")
    for i, c in enumerate(cand.itertuples()):
        o = others.get_group(c.pf)
        o = o[o.track_id != c.track_id]
        if not len(o):
            continue
        iw = np.clip(np.minimum(o.x2, c.x2) - np.maximum(o.x1, c.x1), 0, None)
        ih = np.clip(np.minimum(o.y2, c.y2) - np.maximum(o.y1, c.y1), 0, None)
        out[i] = float((iw * ih).max() / max((c.x2 - c.x1) * (c.y2 - c.y1), 1))
    return out


def frame_size(run: Path) -> tuple:
    cap = cv2.VideoCapture(str(run / "clip.mp4"))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (w or 1920), (h or 1080)


def window_candidates(run: Path, jerseys: set | None = None) -> pd.DataFrame:
    xy = pd.read_csv(run / "tracklet_pitch_xy.csv.gz", usecols=["pf", "ci", "track_id", "X_m", "Y_m"])
    ids = identity_rows(run, xy)[["pf", "ci", "track_id", "jersey"]]
    if jerseys is not None:
        ids = ids[ids.jersey.isin(jerseys)]
    tr = pd.read_csv(run / "best_tracklets.csv.gz", usecols=["pf", "ci", "track_id", "x1", "y1", "x2", "y2", "conf"])
    fw, fh = frame_size(run)
    c = ids.merge(tr, on=["pf", "ci", "track_id"])
    c["h"] = c.y2 - c.y1
    ok = (c.conf >= MIN_CONF) & (c.h >= MIN_H) & (c.x1 > EDGE_PX) & (c.y1 > EDGE_PX)
    ok &= (c.x2 < fw - EDGE_PX) & (c.y2 < fh - EDGE_PX)
    c = c[ok].assign(score=lambda f: f.h * f.conf)
    c = c.sort_values("score", ascending=False).groupby("jersey").head(PER_RUN_PREFILTER)
    if not len(c):
        return c
    c = c[overlap_share(c, tr) <= MAX_OVERLAP]
    c = c.groupby(["jersey", "track_id"]).head(PER_TRACKLET)
    return c.assign(run=run.resolve().relative_to(DATA_DIR.resolve()).as_posix())


def portrait(img: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """An 11:16 crop around the player, head room above, enlarged and lightly sharpened."""
    h = y2 - y1
    ch = h * 1.3
    cw = ch * PHOTO_W / PHOTO_H
    cx, top = (x1 + x2) / 2, y1 - 0.15 * h
    a, b = int(round(cx - cw / 2)), int(round(top))
    w, hh = int(round(cw)), int(round(ch))
    pad = max(w, hh)
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    crop = padded[b + pad : b + pad + hh, a + pad : a + pad + w]
    big = cv2.resize(crop, (PHOTO_W, PHOTO_H), interpolation=cv2.INTER_LANCZOS4)
    soft = cv2.GaussianBlur(big, (0, 0), 1.5)
    return cv2.addWeighted(big, 1.4, soft, -0.4, 0)


def cmd_candidates(args) -> None:
    runs = [require_under_data(Path(r)) for r in args.runs.split(",")]
    jerseys = {int(j) for j in args.jerseys.split(",")} if args.jerseys else None
    parts = []
    for run in runs:
        c = window_candidates(run, jerseys)
        print(f"{run.name}: {len(c)} clear detections of {c.jersey.nunique() if len(c) else 0} players")
        parts.append(c)
    c = pd.concat(parts, ignore_index=True)
    if not len(c):
        raise SystemExit("No clear detections of these players in these windows.")
    # round-robin over windows, best first: light and framing differ between windows and games
    c["rank"] = c.sort_values("score", ascending=False).groupby(["jersey", "run"]).cumcount()
    c = c.sort_values(["rank", "score"], ascending=[True, False]).groupby("jersey").head(args.per_player)
    c = c.sort_values(["jersey", "run", "ci"]).reset_index(drop=True)
    if CAND_DIR.exists():
        shutil.rmtree(CAND_DIR)
    CAND_DIR.mkdir(parents=True)
    c["file"] = [f"{int(j):02d}/{k:03d}.jpg" for k, j in enumerate(c.jersey)]
    for run_name, g in c.groupby("run"):
        run = DATA_DIR / run_name
        stride = cache_stride(run)
        want = {}
        for r in g.itertuples():
            want.setdefault(int(r.ci) * stride, []).append(r)
        for f, img in read_frames(run / "clip.mp4", list(want)):
            for r in want[f]:
                path = CAND_DIR / r.file
                path.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(path), portrait(img, r.x1, r.y1, r.x2, r.y2), [cv2.IMWRITE_JPEG_QUALITY, 92])
    c = c[[(CAND_DIR / f).exists() for f in c.file]]
    c.drop(columns=["rank"]).to_csv(CANDIDATES, index=False)
    print(f"{len(c)} candidates for {c.jersey.nunique()} players: {c.groupby('jersey').size().to_dict()}")
    print(f"wrote {CANDIDATES}. Next: python player_photo.py label")


def names() -> dict:
    roster = pd.read_csv(ROSTER_FILE)
    return {int(r.jersey): str(r.name) for r in roster.itertuples()}


def load_choices() -> pd.DataFrame:
    if CHOICES.exists():
        return pd.read_csv(CHOICES)
    return pd.DataFrame(columns=["jersey", "file", "run", "ci", "track_id"])


def save_choice(choices: pd.DataFrame, jersey: int, row) -> pd.DataFrame:
    """row: the chosen candidate, or None for no photo (the site shows initials)."""
    dest = OUT / f"player_{jersey:02d}.jpg"
    if row is None:
        dest.unlink(missing_ok=True)
        new = dict(jersey=jersey, file="", run="", ci=-1, track_id=-1)
    else:
        shutil.copyfile(CAND_DIR / row.file, dest)
        new = dict(jersey=jersey, file=row.file, run=row.run, ci=int(row.ci), track_id=int(row.track_id))
    choices = pd.concat([choices[choices.jersey != jersey], pd.DataFrame([new])], ignore_index=True)
    choices.sort_values("jersey").to_csv(CHOICES, index=False)
    return choices


def render(tiles: list, chosen: int | None, title: str, zv: ZoomView) -> np.ndarray:
    shown = []
    for p, t in enumerate(tiles):
        t = t.copy()
        if p == chosen:
            cv2.rectangle(t, (2, 2), (t.shape[1] - 3, t.shape[0] - 3), (0, 220, 255), 5)
        shown.append(t)
    zoomed = zv.apply(tile_grid(shown, grid_cols(len(shown))))
    top = np.zeros((HEADER_H, zoomed.shape[1], 3), np.uint8)
    cv2.putText(top, title + zv.label(), (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    keys = "click choose | Enter save+next | s no photo | b back | q quit | wheel zoom | right-click pan | r reset"
    return add_footer(np.vstack([top, zoomed]), keys)


def cmd_label(args) -> None:
    if not CANDIDATES.exists():
        raise SystemExit(f"No {CANDIDATES}: run `candidates` first.")
    cands = pd.read_csv(CANDIDATES)
    choices = load_choices()
    who = names()
    items = [(int(j), g.reset_index(drop=True)) for j, g in cands.groupby("jersey")]
    done = set() if args.redo else set(choices.jersey.astype(int))
    idx = next((i for i, (j, _) in enumerate(items) if j not in done), len(items))
    print(f"{len(items)} players, {len(done & {j for j, _ in items})} already chosen.")
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    zv = ZoomView(reserve_h=HEADER_H + FOOTER_H)  # zoom and view carry over between players (owner rule)
    state = {"cols": 1, "chosen": None}

    def on_mouse(event, mx, my, flags, _p):
        if zv.on_mouse(event, mx, my - HEADER_H, flags) or idx >= len(items):
            return
        if event == cv2.EVENT_LBUTTONDOWN and my >= HEADER_H:
            cx, cy = zv.to_content(mx, my - HEADER_H)
            pos = int(cy // TILE_H) * state["cols"] + int(cx // TILE_W)
            if 0 <= cx < state["cols"] * TILE_W and 0 <= pos < len(items[idx][1]):
                state["chosen"] = None if state["chosen"] == pos else pos

    cv2.setMouseCallback(WINDOW, on_mouse)
    tiles_of = {}
    try:
        while idx < len(items):
            jersey, crops = items[idx]
            if jersey not in tiles_of:
                tiles_of[jersey] = [
                    cv2.resize(cv2.imread(str(CAND_DIR / f)), (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
                    for f in crops.file
                ]
                prev = choices[choices.jersey == jersey]
                hit = crops.index[crops.file == prev.file.iloc[0]] if len(prev) else []
                state["chosen"] = int(hit[0]) if len(hit) else None
            tiles = tiles_of[jersey]
            state["cols"] = grid_cols(len(tiles))
            title = f"{idx + 1}/{len(items)}  {who.get(jersey, '')} #{jersey}: click the clearest photo"
            cv2.imshow(WINDOW, render(tiles, state["chosen"], title, zv))
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10) and state["chosen"] is not None:
                choices = save_choice(choices, jersey, crops.iloc[state["chosen"]])
                idx += 1
                tiles_of.pop(jersey, None)
            elif key == ord("s"):
                choices = save_choice(choices, jersey, None)
                idx += 1
            elif key == ord("b") and idx > 0:
                idx -= 1
            elif key == ord("r"):
                zv.reset()
            elif key == ord("q"):
                break
            if key in (13, 10, ord("s"), ord("b")) and idx < len(items):
                tiles_of.pop(items[idx][0], None)  # reload so the saved choice shows
    finally:
        cv2.destroyAllWindows()
    n = len(list(OUT.glob("player_*.jpg")))
    print(f"{n} photos in {OUT}. Next: python publish_site.py")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates", help="find and crop the clearest detections of every identified player")
    c.add_argument("--runs", required=True, help="comma list of windows with identity")
    c.add_argument("--per-player", type=int, default=CANDIDATES_PER_PLAYER, help="candidate crops per player")
    c.add_argument("--jerseys", default="", help="comma list: only these players (e.g. new ones without a photo)")
    c.set_defaults(fn=cmd_candidates)
    lab = sub.add_parser("label", help="choose one photo per player")
    lab.add_argument("--redo", action="store_true", help="show players that already have a choice too")
    lab.set_defaults(fn=cmd_label)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
