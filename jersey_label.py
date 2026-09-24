"""Hand-identify stitched players against the roster, then apply the result to every tracklet.

`label` shows one stitched player at a time (only "target" and "goalkeeper" roles - the roster is one team),
with up to 32 crops spread across all the tracklets tracklet_stitch.py merged into it, all shown at once in a
grid sized to the screen (time order, left to right then top to bottom; no paging). Jersey numbers are not
readable by plain OCR at this resolution (see CLAUDE.md) - but the owner CAN often read one directly, with zoom,
when a crop happens to catch the player's back. That is the identification method that actually works in
practice, more than recognition by build/kit alone, so many sampled frames are shown at once, to give a good
chance of a number-visible one. This is the "manual anchors" the project's roster-based
identity plan always called for:

  0-9 then Enter   type the jersey number, Enter to confirm (checked against roster.csv)
  Backspace        remove the last typed digit
  a                accept the "earlier" jersey shown in the status line (after checking the crops)
  n                confident this is NOT a roster player (e.g. a misclassified opponent)
  x                the crops show TWO DIFFERENT people - tracklet_stitch.py over-merged this one
  o                a substitute in a bib/off the pitch - a roster player, but not playing right now
  s                skip, unsure - you do not have to read a printed number, recognizing the player is enough
  b                back one player (undo the last label)
  q                save and quit

  mouse wheel      zoom in or out; the point under the cursor stays put - point at a jersey back to read it
  right click      center the view on that spot
  scroll bars      appear when part of the grid is hidden (zoomed past the screen): drag, or click to jump
  r                reset zoom to 1x

Zoom stays where you set it across players (the crop layout is the same for every player). The window grows
with zoom up to the screen size; past that, scroll bars appear. Progress is saved after every player, so rerunning
`label` resumes where you stopped. Labels go to OUT/jersey_truth.csv (git-ignored under data/). The crops show
people: keep them local.

`apply` joins the labels with tracklet_stitch.csv and roster.csv into OUT/player_identity.csv (one row per
original track_id) and reports roster coverage plus these worth a second look: the same jersey given to two
different stitched player_ids (a sign tracklet_stitch.py under-merged them - they are probably one real
player), any player_id marked "mixed" (the opposite mistake - it over-merged two different real people), any
tracklet flagged not-on-roster (a role-classification leak worth knowing about), and any marked "bench" (a
substitute/bibbed player who should have failed the on-pitch check in pitch_mask.py or team_classify.py's
sideline_suspect flag but didn't - worth knowing about for the same reason as not-on-roster). Both "mixed" and
"bench" player_ids get no identity in the output, same as not-on-roster, since none of jersey/not-on-roster
would be correct - "mixed" because it is two people, "bench" because the tracked positions are not on-field
play and would corrupt running stats if attributed to that player's game performance.

Example:
  python jersey_label.py label --run data\\clipA
  python jersey_label.py apply --run data\\clipA
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sv_common import (
    PERSON,
    TILE_H,
    TILE_W,
    Cache,
    ZoomView,
    add_footer,
    cache_stride,
    crop_tile,
    grid_cols,
    read_frames,
    require_under_data,
    tile_grid,
)

WINDOW = "jersey label"
ROSTER_FILE = Path(__file__).resolve().parent / "roster.csv"
ROSTER_ROLES = ("target", "goalkeeper")  # roster.csv is one team; opponents are never matched against it
CROP_W, CROP_H = TILE_W, TILE_H
# Jersey numbers aren't OCR-legible, but ARE sometimes readable by eye (with zoom) when a back-facing frame
# happens to be sampled - the owner reported this is the identification method that actually works for them,
# more than recognition by build/kit. More crops means more chances to catch such a frame for a given player.
N_CROPS = 32  # sampled per player, all shown at once
TRUTH_COLS = ["player_id", "jersey", "verdict"]


def earlier_hints(run: Path, tr: pd.DataFrame) -> dict:
    """player_id -> (jersey, detections) from an earlier jersey_label.py pass, via identity_rows.csv.gz.

    identity_rows.csv.gz (frozen by replay_trackers.py) maps person detections (det_row) to the jersey the owner
    confirmed, so it survives regenerating the tracklets. A hint is given only when at least 90% of a player's
    earlier-labeled detections agree on one jersey. It is a suggestion to check by eye, never applied on its own.
    """
    path = run / "identity_rows.csv.gz"
    if not path.exists() or "det_row" not in tr:
        return {}
    ids = pd.read_csv(path, usecols=["det_row", "jersey", "person_floor"])
    # det_row numbers person detections at one confidence floor. If these tracks were built with another floor,
    # the same number is a different detection: every overlapping row must land on the same frame and on a box
    # that overlaps the tracked one (the frame alone is not enough: neighbouring numbers share a frame).
    cache = Cache(run / "cache")
    floor = float(ids.person_floor.iloc[0])
    persons = cache.det[(cache.det.cls == PERSON) & (cache.det.conf >= floor)].reset_index(drop=True)
    m = tr[["det_row", "ci", "x1", "y1", "x2", "y2"]].merge(ids[["det_row"]], on="det_row")
    m = m[m.det_row < len(persons)]
    if len(m):
        d = persons.iloc[m.det_row.to_numpy()]
        iw = (np.minimum(m.x2.to_numpy(), d.x2.to_numpy()) - np.maximum(m.x1.to_numpy(), d.x1.to_numpy())).clip(0)
        ih = (np.minimum(m.y2.to_numpy(), d.y2.to_numpy()) - np.maximum(m.y1.to_numpy(), d.y1.to_numpy())).clip(0)
        area = lambda b: (b.x2.to_numpy() - b.x1.to_numpy()) * (b.y2.to_numpy() - b.y1.to_numpy())  # noqa: E731
        iou = iw * ih / (area(m) + area(d) - iw * ih + 1e-9)
        aligned = (d.ci.to_numpy() == m.ci.to_numpy()) & (iou > 0.5)
    if not len(m):
        return {}  # no earlier-labeled detections in these tracklets
    if aligned.mean() < 0.95:
        print(f"WARNING: {path.name} does not line up with these tracklets (different --person-floor?). No hints.")
        return {}
    n = tr.merge(ids[["det_row", "jersey"]], on="det_row").groupby(["player_id", "jersey"]).size()
    hints = {}
    for pid, g in n.groupby(level=0):
        top = g.droplevel(0)
        if top.max() >= 0.9 * top.sum():
            hints[pid] = (int(top.idxmax()), int(top.max()))
    return hints


def build_items(run: Path) -> list:
    """One item per stitched target/goalkeeper player, with up to N_CROPS sample rows spread over its life, all
    shown at once, for a good chance that one catches the jersey number."""
    stitch = pd.read_csv(run / "tracklet_stitch.csv")
    stitch = stitch[stitch.role.isin(ROSTER_ROLES)]
    tr = pd.read_csv(run / "best_tracklets.csv.gz")
    tr = tr[tr.track_id.isin(stitch.track_id)].merge(stitch, on="track_id").sort_values("ci")
    hints = earlier_hints(run, tr)
    items = []
    for pid, d in tr.groupby("player_id"):
        d = d.reset_index(drop=True)
        n = min(N_CROPS, len(d))
        idx = np.linspace(0, len(d) - 1, n).round().astype(int)
        rows = d.iloc[idx][["ci", "track_id", "x1", "y1", "x2", "y2"]].to_dict("records")
        items.append(
            dict(player_id=pid, role=d.role.iloc[0], n_tracklets=d.track_id.nunique(), rows=rows, hint=hints.get(pid))
        )
    items.sort(key=lambda it: it["player_id"])
    return items


class Session:
    """Labels for a list of items. Pure logic, no window, so it can be tested."""

    def __init__(self, items: list, roster: pd.DataFrame, saved: pd.DataFrame | None = None):
        self.items = items
        self.valid_jerseys = set(roster.jersey.astype(int))
        self.labels = {}
        if saved is not None and len(saved):
            by_pid = {r.player_id: r for r in saved.itertuples()}
            for i, it in enumerate(items):
                r = by_pid.get(it["player_id"])
                if r is not None:
                    jersey = None if pd.isna(r.jersey) else int(r.jersey)
                    self.labels[i] = dict(jersey=jersey, verdict=r.verdict)
        self.idx = next((i for i in range(len(items)) if i not in self.labels), len(items))

    @property
    def done(self) -> bool:
        return self.idx >= len(self.items)

    def _set(self, jersey, verdict) -> None:
        self.labels[self.idx] = dict(jersey=jersey, verdict=verdict)
        self.idx += 1

    def confirm(self, jersey: int) -> bool:
        if jersey not in self.valid_jerseys:
            return False
        self._set(jersey, "confirmed")
        return True

    def not_on_roster(self) -> None:
        self._set(None, "not_on_roster")

    def mixed(self) -> None:
        self._set(None, "mixed")

    def bench(self) -> None:
        self._set(None, "bench")

    def skip(self) -> None:
        self._set(None, "skipped")

    def back(self) -> None:
        if self.idx > 0:
            self.idx -= 1
            self.labels.pop(self.idx, None)

    def table(self) -> pd.DataFrame:
        rows = [{"player_id": self.items[i]["player_id"], **lab} for i, lab in sorted(self.labels.items())]
        return pd.DataFrame(rows, columns=TRUTH_COLS)


def player_tiles(jpgs: dict, item: dict) -> list:
    """The player's crops in time order (decoded once per player, not on every screen refresh)."""
    tiles = []
    for row in item["rows"]:
        buf = jpgs.get(row["ci"])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf is not None else None
        # the green box marks which figure is the one being identified, since nearby players can appear too
        tiles.append(crop_tile(img, row) if img is not None else np.zeros((CROP_H, CROP_W, 3), np.uint8))
    return tiles


def render(tiles: list, item: dict, index: int, total: int, label: dict | None, typed: str, zv: ZoomView) -> np.ndarray:
    show = zv.apply(tile_grid(tiles, grid_cols(len(tiles))))
    status = f" [{label['verdict']}{' #' + str(label['jersey']) if label.get('jersey') else ''}]" if label else ""
    typed_txt = f"  typing: {typed}" if typed else ""
    keys = (
        "digits+Enter=jersey | n not on roster | x mixed (2 people) | o bench/sub (bib) | s skip | "
        "b back | wheel zoom | right-click pan | drag scroll bars | r reset | q quit"
    )
    header = f"{index + 1}/{total}  {item['player_id']} ({item['role']}, {item['n_tracklets']} tracklets)"
    if item.get("hint"):
        header += f"  earlier: #{item['hint'][0]} (a = accept)"
    return add_footer(show, f"{header}{status}{typed_txt}{zv.label()}   {keys}")


def cmd_label(args) -> None:
    run = args.run
    truth_path = run / "jersey_truth.csv"
    roster = pd.read_csv(ROSTER_FILE)
    items = build_items(run)
    if not items:
        raise SystemExit("No stitched target/goalkeeper players found. Run tracklet_stitch.py first.")
    saved = pd.read_csv(truth_path) if truth_path.exists() else None
    session = Session(items, roster, saved)
    stride = cache_stride(run)
    print(f"{len(items)} players to label, {session.idx} already done. Loading frames...")
    needed = {row["ci"] * stride for it in items for row in it["rows"]}
    jpgs = {}
    for clip_frame, img in read_frames(run / "clip.mp4", list(needed)):
        jpgs[clip_frame // stride] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    zv = ZoomView()  # the owner sets it once and keeps it across players
    cv2.setMouseCallback(WINDOW, zv.on_mouse)
    typed = ""
    tiles_for = {}  # session.idx -> crops, built when a player first comes up
    try:
        while not session.done:
            item = session.items[session.idx]
            if session.idx not in tiles_for:
                tiles_for = {session.idx: player_tiles(jpgs, item)}  # keep only the current player's crops
            label = session.labels.get(session.idx)
            cv2.imshow(WINDOW, render(tiles_for[session.idx], item, session.idx, len(items), label, typed, zv))
            key = cv2.waitKey(30) & 0xFF
            changed = False
            if ord("0") <= key <= ord("9"):
                typed += chr(key)
            elif key in (13, 10):  # Enter
                if typed and session.confirm(int(typed)):
                    changed = True
                typed = ""
            elif key == 8:  # Backspace
                typed = typed[:-1]
            elif key == ord("r"):
                zv.reset()
            elif key == ord("a") and item.get("hint"):
                if session.confirm(item["hint"][0]):
                    typed, changed = "", True
            elif key == ord("n"):
                session.not_on_roster()
                typed, changed = "", True
            elif key == ord("x"):
                session.mixed()
                typed, changed = "", True
            elif key == ord("o"):
                session.bench()
                typed, changed = "", True
            elif key == ord("s"):
                session.skip()
                typed, changed = "", True
            elif key == ord("b"):
                session.back()
                typed = ""
            elif key == ord("q"):
                break
            if changed:
                session.table().to_csv(truth_path, index=False)
    finally:
        cv2.destroyAllWindows()
        session.table().to_csv(truth_path, index=False)
    print(f"Saved {len(session.labels)} of {len(items)} labels to {truth_path}. Run `apply` when you are done.")


def apply_identity(run: Path, roster: pd.DataFrame) -> tuple:
    truth = pd.read_csv(run / "jersey_truth.csv")
    stitch = pd.read_csv(run / "tracklet_stitch.csv")
    confirmed = truth[truth.verdict == "confirmed"][["player_id", "jersey"]]
    out = stitch.merge(confirmed, on="player_id", how="left")
    out = out.merge(roster.rename(columns={"jersey": "jersey"}), on="jersey", how="left")
    dupes = confirmed.groupby("jersey").player_id.nunique()
    conflicts = dupes[dupes > 1]
    report = {
        "target_goalkeeper_player_ids": int(stitch[stitch.role.isin(ROSTER_ROLES)].player_id.nunique()),
        "identified": int(confirmed.player_id.nunique()),
        "not_on_roster": int((truth.verdict == "not_on_roster").sum()),
        "mixed_bad_merges": truth[truth.verdict == "mixed"].player_id.tolist(),
        "bench_or_sub": truth[truth.verdict == "bench"].player_id.tolist(),
        "skipped": int((truth.verdict == "skipped").sum()),
        "roster_players_seen": sorted(int(j) for j in confirmed.jersey.unique()),
        "roster_players_not_seen": sorted(int(j) for j in roster.jersey if j not in set(confirmed.jersey)),
        "jersey_conflicts": {int(j): confirmed[confirmed.jersey == j].player_id.tolist() for j in conflicts.index},
        "note": (
            "jersey_conflicts: tracklet_stitch.py probably under-merged those - likely one real player. "
            "mixed_bad_merges: the opposite - tracklet_stitch.py joined two different real people. "
            "bench_or_sub: a substitute/bibbed player that should have failed the on-pitch check upstream but "
            "didn't - feedback for pitch_mask.py/team_classify.py, not tracklet_stitch.py. All three get no "
            "identity in the output."
        ),
    }
    return out, report


def cmd_apply(args) -> None:
    roster = pd.read_csv(ROSTER_FILE)
    out, report = apply_identity(args.run, roster)
    out.to_csv(args.run / "player_identity.csv", index=False)
    (args.run / "player_identity_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lab = sub.add_parser("label", help="identify stitched target/goalkeeper players against the roster")
    lab.add_argument("--run", required=True, type=require_under_data)
    lab.set_defaults(fn=cmd_label)
    ap_ = sub.add_parser("apply", help="join labels with tracklet_stitch.csv and roster.csv")
    ap_.add_argument("--run", required=True, type=require_under_data)
    ap_.set_defaults(fn=cmd_apply)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
