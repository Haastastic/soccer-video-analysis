"""Confirm crops of jersey numbers the reader has barely seen, so fine-tuning can learn them.

Why: jersey_auto.py's reader is fine-tuned on the owner's labeled windows. Numbers with few or no labeled crops
there (substitutes, players who were off in those windows) get misread as a similar trained number, consistently
enough to pass the read-agreement check (clipB held out: one such player read as three neighbouring numbers,
19 to 29 agreeing reads). The original, not fine-tuned reader cannot fill the gap on its own: on rare numbers its
confident reads were right 0 to 75% (tiny counts). A few owner-confirmed crops per number can.

`candidates` finds, in the given windows, crops where either reader read a rare number (a roster number, not the
goalkeeper, with under --min-crops labeled crops in the labeled windows) and the crop looks legible. At most
PER_TRACKLET per tracklet and MAX_PER_NUMBER per number, best reads first.

`label` shows one rare number at a time, all its candidate crops at once (windows in order, then time; the same
grid and zoom as the other crop tools):

  left click   toggle: this crop shows that number (yellow frame)    a  all    n  none
  Enter        save this number and go to the next                   b  back one number    q  save and quit
  mouse wheel  zoom, keeping the point under the cursor    right click  center there    r  reset zoom
  scroll bars  appear when part of the grid is hidden: drag, or click to jump

Leave a crop unmarked when unsure. Progress is saved after every number, so rerunning `label` resumes.
Confirmed crops go to data/jersey_rare_truth.csv (git-ignored); `jersey_auto.py finetune` adds them to training.
Crops show people: keep them local.

Example:
  python jersey_rare_label.py candidates --runs data\\clipA,data\\clipB,data\\clipE,data\\clipF,data\\clipG `
      --labeled data\\clipA,data\\clipB,data\\clipE
  python jersey_rare_label.py label
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import jersey_auto as ja
from player_stats import ROSTER_FILE, identity_rows
from sv_common import (
    FOOTER_H,
    TILE_H,
    TILE_W,
    ZoomView,
    add_footer,
    cache_stride,
    crop_tile,
    grid_cols,
    read_frames,
    require_under_data,
    tile_grid,
)

DATA = Path(__file__).resolve().parent / "data"
CANDIDATES = DATA / "jersey_rare_candidates.csv"
TRUTH = DATA / "jersey_rare_truth.csv"
WINDOW = "rare jersey confirm"
HEADER_H = 22
MIN_LEGIBILITY = 0.5
PER_TRACKLET, MAX_PER_NUMBER = 3, 48
TRUTH_COLS = ["run", "ci", "track_id", "number", "shows_number"]


def rare_numbers(labeled: list, min_crops: int) -> list:
    """Roster numbers (goalkeeper aside) with under min_crops owner-labeled crops across the labeled windows."""
    roster = pd.read_csv(ROSTER_FILE)
    field = roster[roster.goalkeeper.astype(str).str.lower() != "true"].jersey.astype(int)
    counts = pd.Series(np.concatenate([ja.labeled_patches(r)[1] for r in labeled])).value_counts()
    return sorted(int(j) for j in field if counts.get(j, 0) < min_crops)


def reads_of(run: Path) -> pd.DataFrame:
    """Every cached read of this window (original and fine-tuned readers), with the box."""
    parts = []
    for p in sorted(run.glob("jersey_reads*.csv.gz")):
        parts.append(pd.read_csv(p, dtype={"text": str}, keep_default_na=False))
    if not parts:
        return pd.DataFrame(columns=["ci", "track_id", "legibility", "text", "read_conf"])
    return pd.concat(parts, ignore_index=True)


def find_candidates(runs: list, rare: list, labeled: list) -> pd.DataFrame:
    out = []
    for run in runs:
        r = reads_of(run)
        if run in labeled:  # the owner already named these crops: either a rare number is not rare, or a misread
            xy = pd.read_csv(run / "tracklet_pitch_xy.csv.gz", usecols=["pf", "ci", "track_id", "X_m", "Y_m"])
            known = identity_rows(run, xy)[["ci", "track_id"]].drop_duplicates()
            r = r.merge(known, on=["ci", "track_id"], how="left", indicator=True)
            r = r[r._merge == "left_only"].drop(columns="_merge")
        r = r[(r.legibility >= MIN_LEGIBILITY) & r.text.isin([str(j) for j in rare])].copy()
        if not len(r):
            continue
        r["number"] = r.text.astype(int)
        r["score"] = r.legibility * r.read_conf
        r = r.sort_values("score", ascending=False).drop_duplicates(["ci", "track_id", "number"])
        r = r.groupby(["number", "track_id"]).head(PER_TRACKLET)
        tr = pd.read_csv(run / "best_tracklets.csv.gz", usecols=["ci", "track_id", "x1", "y1", "x2", "y2"])
        out.append(r.merge(tr, on=["ci", "track_id"]).assign(run=run.name))
    if not out:
        return pd.DataFrame()
    c = pd.concat(out, ignore_index=True).sort_values("score", ascending=False)
    c = c.groupby("number").head(MAX_PER_NUMBER)
    cols = ["number", "run", "ci", "track_id", "x1", "y1", "x2", "y2", "text", "legibility", "read_conf"]
    return c.sort_values(["number", "run", "ci"])[cols].reset_index(drop=True)


def cmd_candidates(args) -> None:
    runs = [require_under_data(Path(r)) for r in args.runs.split(",")]
    labeled = [require_under_data(Path(r)) for r in args.labeled.split(",")]
    rare = rare_numbers(labeled, args.min_crops)
    c = find_candidates(runs, rare, labeled)
    c.to_csv(CANDIDATES, index=False)
    counts = c.groupby("number").size().to_dict() if len(c) else {}
    print(
        f"{len(rare)} rare numbers; candidate crops per number: {counts}; none found for "
        f"{len([j for j in rare if j not in counts])} (probably did not play in these windows)"
    )
    print(f"wrote {CANDIDATES}")


class Session:
    """Per-number crop verdicts. Pure logic, no window, so it can be tested."""

    def __init__(self, items: list, saved: pd.DataFrame | None = None):
        self.items = items  # [(number, crops DataFrame)]
        self.marks = {}  # item index -> set of crop positions marked as showing the number
        self.done_items = set()
        if saved is not None and len(saved):
            key = {(r.run, int(r.ci), int(r.track_id), int(r.number)): bool(r.shows_number) for r in saved.itertuples()}
            for i, (number, crops) in enumerate(items):
                ks = [(c.run, int(c.ci), int(c.track_id), number) for c in crops.itertuples()]
                if all(k in key for k in ks):
                    self.done_items.add(i)
                    self.marks[i] = {p for p, k in enumerate(ks) if key[k]}
        self.idx = next((i for i in range(len(items)) if i not in self.done_items), len(items))

    @property
    def done(self) -> bool:
        return self.idx >= len(self.items)

    def toggle(self, pos: int) -> None:
        m = self.marks.setdefault(self.idx, set())
        m.symmetric_difference_update({pos})

    def set_all(self, on: bool) -> None:
        self.marks[self.idx] = set(range(len(self.items[self.idx][1]))) if on else set()

    def finish(self) -> None:
        self.marks.setdefault(self.idx, set())
        self.done_items.add(self.idx)
        self.idx += 1

    def back(self) -> None:
        if self.idx > 0:
            self.idx -= 1
            self.done_items.discard(self.idx)

    def table(self) -> pd.DataFrame:
        rows = []
        for i in sorted(self.done_items):
            number, crops = self.items[i]
            for p, c in enumerate(crops.itertuples()):
                rows.append((c.run, int(c.ci), int(c.track_id), number, p in self.marks.get(i, set())))
        return pd.DataFrame(rows, columns=TRUTH_COLS)


def save(session: Session, saved: pd.DataFrame | None) -> None:
    """This session's verdicts plus saved rows for crops outside it (a rerun with other candidates keeps them)."""
    table = session.table()
    if saved is not None and len(saved):
        k = ["run", "ci", "track_id", "number"]
        keep = saved.merge(table[k], on=k, how="left", indicator=True)
        table = pd.concat([keep[keep._merge == "left_only"][TRUTH_COLS], table], ignore_index=True)
    table.to_csv(TRUTH, index=False)


def render(tiles: list, marks: set, number: int, index: int, total: int, zv: ZoomView) -> np.ndarray:
    shown = []
    for p, t in enumerate(tiles):
        t = t.copy()
        if p in marks:
            cv2.rectangle(t, (2, 2), (t.shape[1] - 3, t.shape[0] - 3), (0, 220, 255), 5)
        shown.append(t)
    zoomed = zv.apply(tile_grid(shown, grid_cols(len(shown))))
    header = f"{index + 1}/{total}  does the green-boxed player wear #{number}?  marked {len(marks)} of {len(tiles)}"
    top = np.zeros((HEADER_H, zoomed.shape[1], 3), np.uint8)
    cv2.putText(top, header + zv.label(), (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    keys = "click toggle | a all | n none | Enter save+next | b back | q quit | wheel zoom | right-click pan | r reset"
    return add_footer(np.vstack([top, zoomed]), keys)


def load_tiles(cands: pd.DataFrame) -> dict:
    """(run, ci, track_id, number) -> tile, reading each window's frames once."""
    tiles = {}
    for run_name, g in cands.groupby("run"):
        run = DATA / run_name
        stride = cache_stride(run)
        want = {}
        for c in g.itertuples():
            want.setdefault(int(c.ci) * stride, []).append(c)
        for f, img in read_frames(run / "clip.mp4", list(want)):
            for c in want[f]:
                box = dict(x1=c.x1, y1=c.y1, x2=c.x2, y2=c.y2)
                tiles[(run_name, int(c.ci), int(c.track_id), int(c.number))] = crop_tile(img, box)
    return tiles


def cmd_label(args) -> None:
    if not CANDIDATES.exists():
        raise SystemExit(f"No {CANDIDATES}: run `candidates` first.")
    cands = pd.read_csv(CANDIDATES)
    items = [(int(n), g.reset_index(drop=True)) for n, g in cands.groupby("number")]
    saved = pd.read_csv(TRUTH) if TRUTH.exists() else None
    session = Session(items, saved)
    print(f"{len(items)} numbers, {session.idx} already done. Loading crops...")
    tiles = load_tiles(cands)
    blank = np.zeros((TILE_H, TILE_W, 3), np.uint8)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    zv = ZoomView(reserve_h=HEADER_H + FOOTER_H)
    state = {"cols": 1}

    def on_mouse(event, mx, my, flags, _p):
        if zv.on_mouse(event, mx, my - HEADER_H, flags) or session.done:
            return
        if event == cv2.EVENT_LBUTTONDOWN and my >= HEADER_H:
            cx, cy = zv.to_content(mx, my - HEADER_H)
            pos = int(cy // TILE_H) * state["cols"] + int(cx // TILE_W)
            if 0 <= cx < state["cols"] * TILE_W and 0 <= pos < len(session.items[session.idx][1]):
                session.toggle(pos)

    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        while not session.done:
            number, crops = session.items[session.idx]
            item_tiles = [tiles.get((c.run, int(c.ci), int(c.track_id), number), blank) for c in crops.itertuples()]
            state["cols"] = grid_cols(len(item_tiles))
            marks = session.marks.get(session.idx, set())
            cv2.imshow(WINDOW, render(item_tiles, marks, number, session.idx, len(items), zv))
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10):
                session.finish()
                save(session, saved)
                zv.reset()
            elif key == ord("a"):
                session.set_all(True)
            elif key == ord("n"):
                session.set_all(False)
            elif key == ord("b"):
                session.back()
                zv.reset()
            elif key == ord("r"):
                zv.reset()
            elif key == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        save(session, saved)
    t = pd.read_csv(TRUTH) if TRUTH.exists() else pd.DataFrame(columns=TRUTH_COLS)
    confirmed = t[t.shows_number].groupby("number").size().to_dict()
    print(f"Saved to {TRUTH}. Confirmed crops per number: {confirmed}. Next: jersey_auto.py finetune.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates", help="find crops that may show rarely labeled numbers")
    c.add_argument("--runs", required=True, help="comma list of windows to search")
    c.add_argument("--labeled", required=True, help="comma list of owner-labeled windows (the reader's training)")
    c.add_argument("--min-crops", type=int, default=30, help="a number is rare below this many labeled crops")
    c.set_defaults(fn=cmd_candidates)
    lab = sub.add_parser("label", help="confirm which candidate crops show their number")
    lab.set_defaults(fn=cmd_label)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
