"""Hand-identify stitched players against the roster, then apply the result to every tracklet.

`label` shows one stitched player at a time (only "target" and "goalkeeper" roles - the roster is one team),
with up to 4 crops spread across all the tracklets tracklet_stitch.py merged into it. Jersey numbers are not
readable by plain OCR at this resolution (see CLAUDE.md), so this is recognition by eye (kit, build, position),
the "manual anchors" the project's roster-based identity plan always called for:

  0-9 then Enter   type the jersey number, Enter to confirm (checked against roster.csv)
  Backspace        remove the last typed digit
  n                confident this is NOT a roster player (e.g. a misclassified opponent)
  s                skip, unsure
  b                back one player (undo the last label)
  q                save and quit

Progress is saved after every player, so rerunning `label` resumes where you stopped. Labels go to
OUT/jersey_truth.csv (git-ignored under data/). The crops show people: keep them local.

`apply` joins the labels with tracklet_stitch.csv and roster.csv into OUT/player_identity.csv (one row per
original track_id) and reports roster coverage plus two things worth a second look: the same jersey given to
two different stitched player_ids (a sign tracklet_stitch.py under-merged them - they are probably one real
player) and any tracklet flagged not-on-roster (a role-classification leak worth knowing about).

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

from sv_common import cache_stride, read_frames, require_under_data

WINDOW = "jersey label"
ROSTER_FILE = Path(__file__).resolve().parent / "roster.csv"
ROSTER_ROLES = ("target", "goalkeeper")  # roster.csv is one team; opponents are never matched against it
CROP_W, CROP_H = 220, 320
PER_PLAYER = 4
TRUTH_COLS = ["player_id", "jersey", "verdict"]


def build_items(run: Path) -> list:
    """One item per stitched target/goalkeeper player, with up to PER_PLAYER sample rows spread over its life."""
    stitch = pd.read_csv(run / "tracklet_stitch.csv")
    stitch = stitch[stitch.role.isin(ROSTER_ROLES)]
    tr = pd.read_csv(run / "best_tracklets.csv.gz")
    tr = tr[tr.track_id.isin(stitch.track_id)].merge(stitch, on="track_id").sort_values("ci")
    items = []
    for pid, d in tr.groupby("player_id"):
        d = d.reset_index(drop=True)
        idx = np.linspace(0, len(d) - 1, min(PER_PLAYER, len(d))).round().astype(int)
        rows = d.iloc[idx][["ci", "track_id", "x1", "y1", "x2", "y2"]].to_dict("records")
        items.append(dict(player_id=pid, role=d.role.iloc[0], n_tracklets=d.track_id.nunique(), rows=rows))
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

    def skip(self) -> None:
        self._set(None, "skipped")

    def back(self) -> None:
        if self.idx > 0:
            self.idx -= 1
            self.labels.pop(self.idx, None)

    def table(self) -> pd.DataFrame:
        rows = [{"player_id": self.items[i]["player_id"], **lab} for i, lab in sorted(self.labels.items())]
        return pd.DataFrame(rows, columns=TRUTH_COLS)


def render(imgs: dict, item: dict, index: int, total: int, label: dict | None, typed: str) -> np.ndarray:
    tiles = []
    for row in item["rows"]:
        img = imgs.get(row["ci"])
        if img is None:
            tiles.append(np.zeros((CROP_H, CROP_W, 3), np.uint8))
            continue
        cx, cy = int((row["x1"] + row["x2"]) / 2), int(row["y2"])
        x0 = int(np.clip(cx - CROP_W / 2, 0, img.shape[1] - CROP_W))
        y0 = int(np.clip(cy - CROP_H, 0, img.shape[0] - CROP_H))
        tiles.append(img[y0 : y0 + CROP_H, x0 : x0 + CROP_W].copy())
    tiles += [np.zeros((CROP_H, CROP_W, 3), np.uint8)] * (PER_PLAYER - len(tiles))
    show = np.hstack(tiles)
    sh, sw = show.shape[:2]
    status = f" [{label['verdict']}{' #' + str(label['jersey']) if label.get('jersey') else ''}]" if label else ""
    typed_txt = f"  typing: {typed}" if typed else ""
    keys = "digits+Enter=jersey | n not on roster | s skip | b back | q quit"
    header = f"{index + 1}/{total}  {item['player_id']} ({item['role']}, {item['n_tracklets']} tracklets)"
    text = f"{header}{status}{typed_txt}   {keys}"
    cv2.rectangle(show, (0, sh - 24), (sw, sh), (0, 0, 0), -1)
    cv2.putText(show, text, (6, sh - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return show


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
    typed = ""
    try:
        while not session.done:
            item = session.items[session.idx]
            imgs = {
                row["ci"]: cv2.imdecode(jpgs[row["ci"]], cv2.IMREAD_COLOR) for row in item["rows"] if row["ci"] in jpgs
            }
            cv2.imshow(WINDOW, render(imgs, item, session.idx, len(items), session.labels.get(session.idx), typed))
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
            elif key == ord("n"):
                session.not_on_roster()
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
        "skipped": int((truth.verdict == "skipped").sum()),
        "roster_players_seen": sorted(int(j) for j in confirmed.jersey.unique()),
        "roster_players_not_seen": sorted(int(j) for j in roster.jersey if j not in set(confirmed.jersey)),
        "jersey_conflicts": {int(j): confirmed[confirmed.jersey == j].player_id.tolist() for j in conflicts.index},
        "note": "jersey_conflicts: tracklet_stitch.py probably under-merged those player_ids - likely one real player.",
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
