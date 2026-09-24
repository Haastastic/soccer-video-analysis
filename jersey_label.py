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
  b                back one player (undo the last label, restoring what was there before)
  q                save and quit

Tracking changes are marked: when a player is several tracklets joined by tracklet_stitch.py, each crop shows
its tracklet (T1, T2, ... with its own frame color) and a yellow bar marks where one tracklet ends and the next
begins. Every tracklet gets at least a few crops. When the joins put different people together, name the
tracklets one by one instead of losing the whole player (owner request; most "mixed" players were bad joins):

  click a crop     select its tracklet (click again, or Esc, to deselect)
  with a tracklet selected: digits+Enter / a / n / o / s label THAT tracklet (a uses its own "earlier" jersey),
                   then the next unnamed tracklet is selected; Backspace with nothing typed clears its label
  x or Enter       (nothing typed, no selection) finish the player: named tracklets keep their labels, the
                   rest get no identity. Saved as verdict "split". Without any named tracklet, x is "mixed".

  mouse wheel      zoom in or out; the point under the cursor stays put - point at a jersey back to read it
  right click      center the view on that spot
  scroll bars      appear when part of the grid is hidden (zoomed past the screen): drag, or click to jump
  r                reset zoom to 1x

Zoom stays where you set it across players (the crop layout is the same for every player). The window grows
with zoom up to the screen size; past that, scroll bars appear. Progress is saved after every player, so rerunning
`label` resumes where you stopped. `label --redo-mixed` re-opens the players marked "mixed" to name their
tracklets. Labels go to OUT/jersey_truth.csv (players) and OUT/jersey_tracklets.csv (named tracklets of "split"
players), git-ignored under data/. The crops show people: keep them local.

`apply` joins the labels with tracklet_stitch.csv and roster.csv into OUT/player_identity.csv (one row per
original track_id; a named tracklet's label overrides its player's) and reports roster coverage plus these worth
a second look: the same jersey given to two different stitched player_ids (a sign tracklet_stitch.py
under-merged them - they are probably one real player), any player_id marked "mixed" (the opposite mistake - it
over-merged two different real people), any tracklet flagged not-on-roster (a role-classification leak worth
knowing about), and any marked "bench" (a substitute/bibbed player who should have failed the on-pitch check in
pitch_mask.py or team_classify.py's sideline_suspect flag but didn't - worth knowing about for the same reason as
not-on-roster). "mixed" and "bench" get no identity in the output, same as not-on-roster, since none of
jersey/not-on-roster would be correct - "mixed" because it is two people, "bench" because the tracked positions
are not on-field play and would corrupt running stats if attributed to that player's game performance.

Example:
  python jersey_label.py label --run data\\clipA
  python jersey_label.py label --run data\\clipA --redo-mixed
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
MIN_PER_TRACKLET = 3  # so a short tracklet in a stitched player still gets enough crops to be named
TRUTH_COLS = ["player_id", "jersey", "verdict"]
TRACKLET_COLS = ["track_id", "player_id", "jersey", "verdict"]
TRACKLET_COLORS = [(255, 200, 0), (0, 200, 255), (255, 0, 200), (0, 255, 160), (160, 120, 255), (0, 128, 255)]
DIVIDER = (0, 255, 255)  # yellow bar where one tracklet ends and the next begins
SHORT = {"confirmed": "#", "not_on_roster": "not roster", "bench": "bench", "skipped": "skip"}


def earlier_hints(run: Path, tr: pd.DataFrame) -> tuple:
    """(player_id -> (jersey, n), track_id -> (jersey, n)) from an earlier jersey_label.py pass.

    identity_rows.csv.gz (frozen by replay_trackers.py) maps person detections (det_row) to the jersey the owner
    confirmed, so it survives regenerating the tracklets. A hint is given only when at least 90% of the
    earlier-labeled detections of that player (or tracklet) agree on one jersey. It is a suggestion to check by
    eye, never applied on its own.
    """
    path = run / "identity_rows.csv.gz"
    if not path.exists() or "det_row" not in tr:
        return {}, {}
    ids = pd.read_csv(path, usecols=["det_row", "jersey", "person_floor"])
    # det_row numbers person detections at one confidence floor. If these tracks were built with another floor,
    # the same number is a different detection: every overlapping row must land on the same frame and on a box
    # that overlaps the tracked one (the frame alone is not enough: neighbouring numbers share a frame).
    cache = Cache(run / "cache")
    floor = float(ids.person_floor.iloc[0])
    persons = cache.det[(cache.det.cls == PERSON) & (cache.det.conf >= floor)].reset_index(drop=True)
    m = tr[["det_row", "ci", "x1", "y1", "x2", "y2"]].merge(ids[["det_row"]], on="det_row")
    m = m[m.det_row < len(persons)]
    if not len(m):
        return {}, {}  # no earlier-labeled detections in these tracklets
    d = persons.iloc[m.det_row.to_numpy()]
    iw = (np.minimum(m.x2.to_numpy(), d.x2.to_numpy()) - np.maximum(m.x1.to_numpy(), d.x1.to_numpy())).clip(0)
    ih = (np.minimum(m.y2.to_numpy(), d.y2.to_numpy()) - np.maximum(m.y1.to_numpy(), d.y1.to_numpy())).clip(0)
    area = lambda b: (b.x2.to_numpy() - b.x1.to_numpy()) * (b.y2.to_numpy() - b.y1.to_numpy())  # noqa: E731
    iou = iw * ih / (area(m) + area(d) - iw * ih + 1e-9)
    if ((d.ci.to_numpy() == m.ci.to_numpy()) & (iou > 0.5)).mean() < 0.95:
        print(f"WARNING: {path.name} does not line up with these tracklets (different --person-floor?). No hints.")
        return {}, {}
    j = tr.merge(ids[["det_row", "jersey"]], on="det_row")

    def clear_winner(key: str) -> dict:
        hints = {}
        for k, g in j.groupby([key, "jersey"]).size().groupby(level=0):
            top = g.droplevel(0)
            if top.max() >= 0.9 * top.sum():
                hints[k] = (int(top.idxmax()), int(top.max()))
        return hints

    return clear_winner("player_id"), clear_winner("track_id")


def sample_rows(d: pd.DataFrame) -> pd.DataFrame:
    """Up to N_CROPS rows spread over a player's life, with at least MIN_PER_TRACKLET from every tracklet.

    With many tracklets the minimum shrinks so the total stays within N_CROPS; every tracklet keeps at least one
    crop, so only a player with more than N_CROPS tracklets gets more crops than that.
    """
    sizes = d.groupby("track_id", sort=False).size()
    floor = max(1, min(MIN_PER_TRACKLET, N_CROPS // len(sizes)))
    alloc = {t: min(n, max(floor, round(N_CROPS * n / len(d)))) for t, n in sizes.items()}
    while sum(alloc.values()) > N_CROPS:  # trim the largest, never below the floor
        t = max(alloc, key=alloc.get)
        if alloc[t] <= floor:
            break
        alloc[t] -= 1
    parts = []
    for t, g in d.groupby("track_id", sort=False):
        parts.append(g.iloc[np.linspace(0, len(g) - 1, alloc[t]).round().astype(int)])
    return pd.concat(parts).sort_values("ci")


def build_items(run: Path) -> list:
    """One item per stitched target/goalkeeper player, with up to N_CROPS sample rows spread over its life, all
    shown at once, for a good chance that one catches the jersey number."""
    stitch = pd.read_csv(run / "tracklet_stitch.csv")
    stitch = stitch[stitch.role.isin(ROSTER_ROLES)]
    tr = pd.read_csv(run / "best_tracklets.csv.gz")
    tr = tr[tr.track_id.isin(stitch.track_id)].merge(stitch, on="track_id").sort_values("ci")
    hints, thints = earlier_hints(run, tr)
    items = []
    for pid, d in tr.groupby("player_id"):
        d = d.reset_index(drop=True)
        rows = sample_rows(d)[["ci", "track_id", "x1", "y1", "x2", "y2"]].to_dict("records")
        tracklets = list(dict.fromkeys(d.track_id))  # in order of first appearance
        items.append(
            dict(
                player_id=pid,
                role=d.role.iloc[0],
                n_tracklets=len(tracklets),
                tracklets=tracklets,
                rows=rows,
                hint=hints.get(pid),
                thints={t: thints[t] for t in tracklets if t in thints},
            )
        )
    items.sort(key=lambda it: it["player_id"])
    return items


class Session:
    """Labels for a list of items. Pure logic, no window, so it can be tested.

    order: which items to visit, in order (all of them by default; --redo-mixed visits only "mixed" ones).
    labels: item index -> player-level label. tlabels: track_id -> label of that one tracklet, used when the
    player is "split". back() restores exactly what an item had before it was labeled in this session.
    """

    def __init__(
        self,
        items: list,
        roster: pd.DataFrame,
        saved: pd.DataFrame | None = None,
        saved_tracklets: pd.DataFrame | None = None,
        redo_mixed: bool = False,
    ):
        self.items = items
        self.valid_jerseys = set(roster.jersey.astype(int))
        self.labels, self.tlabels = {}, {}
        if saved is not None and len(saved):
            by_pid = {r.player_id: r for r in saved.itertuples()}
            for i, it in enumerate(items):
                r = by_pid.get(it["player_id"])
                if r is not None:
                    jersey = None if pd.isna(r.jersey) else int(r.jersey)
                    self.labels[i] = dict(jersey=jersey, verdict=r.verdict)
        if saved_tracklets is not None:
            for r in saved_tracklets.itertuples():
                self.tlabels[int(r.track_id)] = dict(
                    jersey=None if pd.isna(r.jersey) else int(r.jersey), verdict=r.verdict
                )
        if redo_mixed:
            self.order = [i for i in range(len(items)) if self.labels.get(i, {}).get("verdict") == "mixed"]
            self.pos = 0
        else:
            self.order = list(range(len(items)))
            self.pos = next((k for k, i in enumerate(self.order) if i not in self.labels), len(self.order))
        self.history = []  # (pos, previous player label, previous labels of its tracklets)
        self.selected = None  # track_id whose label the next key sets, or None for the whole player

    @property
    def done(self) -> bool:
        return self.pos >= len(self.order)

    @property
    def idx(self) -> int:
        return self.order[self.pos]

    @property
    def item(self) -> dict:
        return self.items[self.idx]

    def _set(self, jersey, verdict) -> None:
        tids = self.item["tracklets"]
        self.history.append((self.pos, self.labels.get(self.idx), {t: self.tlabels.get(t) for t in tids}))
        if verdict != "split":  # a whole-player label replaces any tracklet names
            for t in tids:
                self.tlabels.pop(t, None)
        self.labels[self.idx] = dict(jersey=jersey, verdict=verdict)
        self.pos += 1
        self.selected = None

    def confirm(self, jersey: int) -> bool:
        if jersey not in self.valid_jerseys:
            return False
        self._set(jersey, "confirmed")
        return True

    def not_on_roster(self) -> None:
        self._set(None, "not_on_roster")

    def mixed(self) -> None:
        """Two or more people. Keeps any tracklets named so far (then it is "split"), else plain "mixed"."""
        self._set(None, "split" if self.named() else "mixed")

    def bench(self) -> None:
        self._set(None, "bench")

    def skip(self) -> None:
        self._set(None, "skipped")

    def named(self) -> list:
        return [t for t in self.item["tracklets"] if t in self.tlabels]

    def select(self, track_id) -> None:
        self.selected = None if track_id == self.selected else track_id

    def label_tracklet(self, jersey, verdict) -> bool:
        """Label the selected tracklet, then select the next unnamed one (or none)."""
        if self.selected is None or (verdict == "confirmed" and jersey not in self.valid_jerseys):
            return False
        self.tlabels[self.selected] = dict(jersey=jersey, verdict=verdict)
        tids = self.item["tracklets"]
        k = tids.index(self.selected)
        self.selected = next((t for t in tids[k + 1 :] + tids[:k] if t not in self.tlabels), None)
        return True

    def clear_tracklet(self) -> None:
        if self.selected is not None:
            self.tlabels.pop(self.selected, None)

    def back(self) -> None:
        if not self.history:
            return
        pos, prev, prev_t = self.history.pop()
        self.pos, self.selected = pos, None
        if prev is None:
            self.labels.pop(self.idx, None)
        else:
            self.labels[self.idx] = prev
        for t, lab in prev_t.items():
            if lab is None:
                self.tlabels.pop(t, None)
            else:
                self.tlabels[t] = lab

    def table(self) -> pd.DataFrame:
        rows = [{"player_id": self.items[i]["player_id"], **lab} for i, lab in sorted(self.labels.items())]
        return pd.DataFrame(rows, columns=TRUTH_COLS)

    def tracklet_table(self) -> pd.DataFrame:
        """Named tracklets of players saved as "split" (names on an unfinished player are not saved)."""
        rows = []
        for i, lab in sorted(self.labels.items()):
            if lab["verdict"] == "split":
                it = self.items[i]
                rows += [
                    dict(track_id=t, player_id=it["player_id"], **self.tlabels[t])
                    for t in it["tracklets"]
                    if t in self.tlabels
                ]
        return pd.DataFrame(rows, columns=TRACKLET_COLS)


def player_tiles(jpgs: dict, item: dict) -> list:
    """The player's crops in time order (decoded once per player, not on every screen refresh)."""
    tiles = []
    for row in item["rows"]:
        buf = jpgs.get(row["ci"])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf is not None else None
        # the green box marks which figure is the one being identified, since nearby players can appear too
        tiles.append(crop_tile(img, row) if img is not None else np.zeros((CROP_H, CROP_W, 3), np.uint8))
    return tiles


def tracklet_text(k: int, lab: dict | None) -> str:
    if not lab:
        return f"T{k + 1}"
    return f"T{k + 1} {SHORT[lab['verdict']]}{lab['jersey'] if lab['verdict'] == 'confirmed' else ''}"


def mark_tracklets(tiles: list, item: dict, session: "Session") -> list:
    """Tracklet tag, frame color and a divider where the tracklet changes (only for stitched players)."""
    if item["n_tracklets"] < 2:
        return tiles
    out, prev = [], None
    for tile, row in zip(tiles, item["rows"], strict=True):
        t = tile.copy()
        k = item["tracklets"].index(row["track_id"])
        color = TRACKLET_COLORS[k % len(TRACKLET_COLORS)]
        sel = row["track_id"] == session.selected
        cv2.rectangle(t, (0, 0), (CROP_W - 1, CROP_H - 1), (255, 255, 255) if sel else color, 8 if sel else 3)
        if prev is not None and row["track_id"] != prev:
            cv2.rectangle(t, (0, 0), (7, CROP_H - 1), DIVIDER, -1)
        text = tracklet_text(k, session.tlabels.get(row["track_id"]))
        cv2.putText(t, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(t, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        out.append(t)
        prev = row["track_id"]
    return out


def render(tiles: list, session: "Session", total: int, typed: str, zv: ZoomView) -> np.ndarray:
    item, label = session.item, session.labels.get(session.idx)
    show = zv.apply(tile_grid(mark_tracklets(tiles, item, session), grid_cols(len(tiles))))
    status = f" [{label['verdict']}{' #' + str(label['jersey']) if label.get('jersey') else ''}]" if label else ""
    typed_txt = f"  typing: {typed}" if typed else ""
    header = f"{session.pos + 1}/{total}  {item['player_id']} ({item['role']}, {item['n_tracklets']} tracklets)"
    if session.selected is not None:
        k = item["tracklets"].index(session.selected)
        hint = item["thints"].get(session.selected)
        header += f"  T{k + 1} selected" + (f", earlier: #{hint[0]} (a = accept)" if hint else "")
        keys = "T: digits+Enter / a / n / o / s label this tracklet | Backspace clear | Esc or click deselect"
    else:
        if item.get("hint"):
            header += f"  earlier: #{item['hint'][0]} (a = accept)"
        named = len(session.named())
        if named:
            header += f"  {named}/{item['n_tracklets']} tracklets named: x or Enter finishes as split"
        keys = (
            "digits+Enter=jersey | n not on roster | x mixed (2 people) | o bench/sub (bib) | s skip | "
            + ("click a crop to name its tracklet | " if item["n_tracklets"] > 1 else "")
            + "b back | wheel zoom | right-click pan | drag scroll bars | r reset | q quit"
        )
    return add_footer(show, f"{header}{status}{typed_txt}{zv.label()}   {keys}")


def cmd_label(args) -> None:
    run = args.run
    truth_path, tracklet_path = run / "jersey_truth.csv", run / "jersey_tracklets.csv"
    roster = pd.read_csv(ROSTER_FILE)
    items = build_items(run)
    if not items:
        raise SystemExit("No stitched target/goalkeeper players found. Run tracklet_stitch.py first.")
    saved = pd.read_csv(truth_path) if truth_path.exists() else None
    saved_t = pd.read_csv(tracklet_path) if tracklet_path.exists() else None
    session = Session(items, roster, saved, saved_t, redo_mixed=args.redo_mixed)
    if args.redo_mixed and session.done:
        raise SystemExit("No players marked mixed to redo.")

    def save():
        session.table().to_csv(truth_path, index=False)
        session.tracklet_table().to_csv(tracklet_path, index=False)

    stride = cache_stride(run)
    todo = [items[i] for i in session.order[session.pos :]]
    print(f"{len(session.order)} players in this pass, {session.pos} already done. Loading frames...")
    needed = {row["ci"] * stride for it in todo for row in it["rows"]}
    jpgs = {}
    for clip_frame, img in read_frames(run / "clip.mp4", list(needed)):
        jpgs[clip_frame // stride] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    zv = ZoomView()  # the owner sets it once and keeps it across players

    def on_mouse(event, mx, my, flags, _p):
        if zv.on_mouse(event, mx, my, flags) or event != cv2.EVENT_LBUTTONDOWN or session.done:
            return
        item = session.item
        if item["n_tracklets"] < 2:
            return
        cx, cy = zv.to_content(mx, my)
        cols = grid_cols(len(item["rows"]))
        col, row = int(cx // CROP_W), int(cy // CROP_H)
        j = row * cols + col
        if 0 <= col < cols and row >= 0 and j < len(item["rows"]):
            session.select(item["rows"][j]["track_id"])

    cv2.setMouseCallback(WINDOW, on_mouse)
    typed = ""
    tiles_for = {}  # item index -> crops, built when a player first comes up
    try:
        while not session.done:
            item = session.item
            if session.idx not in tiles_for:
                tiles_for = {session.idx: player_tiles(jpgs, item)}  # keep only the current player's crops
            cv2.imshow(WINDOW, render(tiles_for[session.idx], session, len(session.order), typed, zv))
            key = cv2.waitKey(30) & 0xFF
            sel = session.selected is not None
            before = session.pos
            if ord("0") <= key <= ord("9"):
                typed += chr(key)
            elif key in (13, 10):  # Enter
                if typed:
                    _ = session.label_tracklet(int(typed), "confirmed") if sel else session.confirm(int(typed))
                    typed = ""  # an invalid number (not on the roster) is simply dropped
                elif not sel and session.named():
                    session.mixed()  # finish as "split"
            elif key == 8:  # Backspace
                if typed:
                    typed = typed[:-1]
                elif sel:
                    session.clear_tracklet()
            elif key == 27:  # Esc
                session.selected, typed = None, ""
            elif key == ord("r"):
                zv.reset()
            elif key == ord("a"):
                hint = item["thints"].get(session.selected) if sel else item.get("hint")
                if hint:
                    _ = session.label_tracklet(hint[0], "confirmed") if sel else session.confirm(hint[0])
                typed = ""
            elif key in (ord("n"), ord("o"), ord("s")):
                verdict = {ord("n"): "not_on_roster", ord("o"): "bench", ord("s"): "skipped"}[key]
                if sel:
                    session.label_tracklet(None, verdict)
                else:
                    {"not_on_roster": session.not_on_roster, "bench": session.bench, "skipped": session.skip}[verdict]()
                typed = ""
            elif key == ord("x") and not sel:
                session.mixed()
                typed = ""
            elif key == ord("b"):
                session.back()
                typed = ""
            elif key == ord("q"):
                break
            if session.pos != before or key == ord("b"):
                save()
    finally:
        cv2.destroyAllWindows()
        save()
    print(
        f"Saved {len(session.labels)} player labels to {truth_path}, named tracklets to {tracklet_path}. Run `apply`."
    )


def apply_identity(run: Path, roster: pd.DataFrame) -> tuple:
    truth = pd.read_csv(run / "jersey_truth.csv")
    stitch = pd.read_csv(run / "tracklet_stitch.csv")
    tpath = run / "jersey_tracklets.csv"
    named = pd.read_csv(tpath) if tpath.exists() else pd.DataFrame(columns=TRACKLET_COLS)
    confirmed = truth[truth.verdict == "confirmed"][["player_id", "jersey"]]
    out = stitch.merge(confirmed, on="player_id", how="left")
    # a named tracklet (of a "split" player) overrides its player's label: its jersey, or no identity
    tl = named.set_index("track_id")
    has = out.track_id.isin(tl.index)
    tl_jersey = tl.jersey.where(tl.verdict == "confirmed")
    out.loc[has, "jersey"] = out.loc[has, "track_id"].map(tl_jersey)
    out = out.merge(roster, on="jersey", how="left")
    named_ok = named[named.verdict == "confirmed"]
    ids = pd.concat([confirmed.assign(level="player"), named_ok[["player_id", "jersey"]].assign(level="tracklet")])
    # the same jersey on two stitched players (whole-player or a named tracklet): likely an under-merge
    dupes = ids.groupby("jersey").player_id.nunique()
    conflicts = dupes[dupes > 1]
    seen = set(ids.jersey.astype(int))
    report = {
        "target_goalkeeper_player_ids": int(stitch[stitch.role.isin(ROSTER_ROLES)].player_id.nunique()),
        "identified": int(confirmed.player_id.nunique()),
        "split_players": int((truth.verdict == "split").sum()),
        "tracklets_named_in_split_players": int(len(named_ok)),
        "not_on_roster": int((truth.verdict == "not_on_roster").sum() + (named.verdict == "not_on_roster").sum()),
        "mixed_bad_merges": truth[truth.verdict == "mixed"].player_id.tolist(),
        "bench_or_sub": truth[truth.verdict == "bench"].player_id.tolist(),
        "skipped": int((truth.verdict == "skipped").sum()),
        "roster_players_seen": sorted(seen),
        "roster_players_not_seen": sorted(int(j) for j in roster.jersey if j not in seen),
        "jersey_conflicts": {int(j): sorted(set(ids[ids.jersey == j].player_id)) for j in conflicts.index},
        "note": (
            "jersey_conflicts: tracklet_stitch.py probably under-merged those - likely one real player. "
            "mixed_bad_merges: the opposite - tracklet_stitch.py joined two different real people. "
            "split_players: mixed players whose tracklets were named one by one; unnamed ones get no identity. "
            "bench_or_sub: a substitute/bibbed player that should have failed the on-pitch check upstream but "
            "didn't - feedback for pitch_mask.py/team_classify.py, not tracklet_stitch.py. All get no identity "
            "in the output except named tracklets."
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
    lab.add_argument("--redo-mixed", action="store_true", help="re-open players marked mixed to name tracklets")
    lab.set_defaults(fn=cmd_label)
    ap_ = sub.add_parser("apply", help="join labels with tracklet_stitch.csv and roster.csv")
    ap_.add_argument("--run", required=True, type=require_under_data)
    ap_.set_defaults(fn=cmd_apply)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
