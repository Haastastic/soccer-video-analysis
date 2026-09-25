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
  a                accept the jersey shown in the status line (after checking the crops): "earlier" (the owner's own
                   label of the same detections in a previous pass) or "suggest" (jersey_suggest.py's appearance
                   model, with its confidence and the next two guesses). Unnamed parts show their guess as "~N".
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
  shift+click      a different person from this crop on: splits the crop's tracklet there into parts (T1a, T1b,
                   a magenta bar), each named like a tracklet; shift+click the same crop again to undo. Works on
                   single-tracklet players too (a swap inside one tracklet). Frames between the last crop before
                   the split and the split crop belong to no part, since the exact switch frame is not known.
  x or Enter       (nothing typed, no selection) finish the player: named tracklets and parts keep their labels,
                   the rest get no identity. Saved as verdict "split". With nothing named, x is "mixed".

  mouse wheel      zoom in or out; the point under the cursor stays put - point at a jersey back to read it
  right click      center the view on that spot
  scroll bars      appear when part of the grid is hidden (zoomed past the screen): drag, or click to jump
  r                reset zoom to 1x

Zoom stays where you set it across players (the crop layout is the same for every player). The window grows
with zoom up to the screen size; past that, scroll bars appear. Progress is saved after every player, so rerunning
`label` resumes where you stopped. `label --redo-mixed` re-opens the players marked "mixed" to name their
tracklets. Labels go to OUT/jersey_truth.csv (players) and OUT/jersey_tracklets.csv (named tracklets and parts of
"split" players, with frame ranges), git-ignored under data/. The crops show people: keep them local.

`apply` joins the labels with tracklet_stitch.csv and roster.csv into OUT/player_identity.csv (one row per
original track_id; a named tracklet's label overrides its player's; a tracklet split at a switch has no single
identity there, split_at_switch=True) and OUT/identity_segments.csv (the named parts of split tracklets, as frame
ranges), and reports roster coverage plus these worth
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
import bisect
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
TRACKLET_COLS = ["track_id", "seg", "ci_start", "ci_end", "player_id", "jersey", "verdict"]
TRACKLET_COLORS = [(255, 200, 0), (0, 200, 255), (255, 0, 200), (0, 255, 160), (160, 120, 255), (0, 128, 255)]
DIVIDER = (0, 255, 255)  # yellow bar where one tracklet ends and the next begins
SPLIT_BAR = (255, 0, 255)  # magenta bar where the owner marked a switch to another person inside a tracklet
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
    sugg = load_suggestions(run)
    items = []
    for pid, d in tr.groupby("player_id"):
        d = d.reset_index(drop=True)
        rows = sample_rows(d)[["ci", "track_id", "x1", "y1", "x2", "y2"]].to_dict("records")
        tracklets = list(dict.fromkeys(d.track_id))  # in order of first appearance
        span = d.groupby("track_id").ci.agg(["min", "max"])
        items.append(
            dict(
                player_id=pid,
                role=d.role.iloc[0],
                n_tracklets=len(tracklets),
                tracklets=tracklets,
                span={int(t): (int(r["min"]), int(r["max"])) for t, r in span.iterrows()},
                rows=rows,
                hint=hints.get(pid),
                thints={t: thints[t] for t in tracklets if t in thints},
                psugg=sugg["player"].get(pid),
                tsugg={t: sugg["tracklet"][t] for t in tracklets if t in sugg["tracklet"]},
            )
        )
    for it in items:  # per-crop probabilities, for parts split during labeling
        it["P"] = [sugg["crop"].get((r["track_id"], r["ci"])) for r in it["rows"]]
    items.sort(key=lambda it: it["player_id"])
    return items


def load_suggestions(run: Path) -> dict:
    """jersey_suggest.py output: per tracklet and per player (jersey, confidence, alternates), per crop probabilities.

    Empty if jersey_suggest.py has not been run for this window (the tool works the same, without suggestions)."""
    out = dict(player={}, tracklet={}, crop={}, classes=None)
    csv, npz = run / "jersey_suggestions.csv", run / "jersey_suggest.npz"
    if not (csv.exists() and npz.exists()):
        return out
    for r in pd.read_csv(csv).itertuples():
        alts = [int(a) for a in str(r.alts).split()] if isinstance(r.alts, str) else []
        entry = (int(r.jersey), float(r.conf), alts)
        if r.level == "player":
            out["player"][r.player_id] = entry
        else:
            out["tracklet"][int(r.track_id)] = entry
    z = np.load(npz)
    SUGG_CLASSES[:] = [int(c) for c in z["classes"]]
    out["crop"] = {(int(t), int(c)): p for t, c, p in zip(z["track_id"], z["ci"], z["P"], strict=True)}
    return out


SUGG_CLASSES: list = []  # jersey classes of the per-crop probabilities (set by load_suggestions)


def suggestion(item: dict, session: "Session", key: tuple | None) -> tuple | None:
    """What `a` would accept, as (jersey, confidence or None, alternates, source).

    An exact earlier label of the same detections ("earlier") wins over the appearance model ("suggest").
    key None means the whole player. A part of a split tracklet gets a suggestion from its own crops.
    """
    if key is None:
        if item.get("hint"):
            return item["hint"][0], None, [], "earlier"
        return (*item["psugg"], "suggest") if item.get("psugg") else None
    tid = key[0]
    if tid not in session.splits:
        if tid in item["thints"]:
            return item["thints"][tid][0], None, [], "earlier"
        return (*item["tsugg"][tid], "suggest") if tid in item["tsugg"] else None
    ps = [
        p
        for p, r in zip(item["P"], item["rows"], strict=True)
        if p is not None and session.seg_of(r["track_id"], r["ci"]) == key
    ]
    if not ps or not SUGG_CLASSES:
        return None
    lp = np.log(np.stack(ps) + 1e-9).mean(0)
    pr = np.exp(lp - lp.max())
    pr /= pr.sum()
    order = np.argsort(-pr)
    return SUGG_CLASSES[order[0]], float(pr[order[0]]), [SUGG_CLASSES[k] for k in order[1:3]], "suggest"


def suggestion_text(sg: tuple | None) -> str:
    if sg is None:
        return ""
    jersey, conf, alts, source = sg
    if source == "earlier":
        return f"earlier: #{jersey} (a = accept)"
    then = f"; then {' '.join('#' + str(a) for a in alts)}" if alts else ""
    return f"suggest #{jersey} ({conf:.2f}{then}) (a = accept)"


class Session:
    """Labels for a list of items. Pure logic, no window, so it can be tested.

    order: which items to visit, in order (all of them by default; --redo-mixed visits only "mixed" ones).
    labels: item index -> player-level label.
    Segments: a tracklet is one segment, or several after the owner marks switch points (splits: track_id ->
    sorted ci where a different person starts). A segment is keyed (track_id, k), k = 0, 1, ...
    tlabels: segment -> label, used when the player is "split". back() restores exactly what an item had before.
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
        self.labels, self.tlabels, self.splits = {}, {}, {}
        if saved is not None and len(saved):
            by_pid = {r.player_id: r for r in saved.itertuples()}
            for i, it in enumerate(items):
                r = by_pid.get(it["player_id"])
                if r is not None:
                    jersey = None if pd.isna(r.jersey) else int(r.jersey)
                    self.labels[i] = dict(jersey=jersey, verdict=r.verdict)
        if saved_tracklets is not None and len(saved_tracklets):
            t = saved_tracklets.copy()
            if "seg" not in t:  # saved before tracklets could be split
                t["seg"], t["ci_start"] = 0, np.nan
            for r in t.itertuples():
                key = (int(r.track_id), int(r.seg))
                if r.seg > 0:
                    self.splits.setdefault(key[0], []).append(int(r.ci_start))
                if r.verdict != "unnamed":
                    jersey = None if pd.isna(r.jersey) else int(r.jersey)
                    self.tlabels[key] = dict(jersey=jersey, verdict=r.verdict)
            self.splits = {k: sorted(v) for k, v in self.splits.items()}
        if redo_mixed:
            self.order = [i for i in range(len(items)) if self.labels.get(i, {}).get("verdict") == "mixed"]
            self.pos = 0
        else:
            self.order = list(range(len(items)))
            self.pos = next((k for k, i in enumerate(self.order) if i not in self.labels), len(self.order))
        self.history = []  # (pos, previous player label, previous segment labels, previous splits)
        self.selected = None  # segment whose label the next key sets, or None for the whole player

    @property
    def done(self) -> bool:
        return self.pos >= len(self.order)

    @property
    def idx(self) -> int:
        return self.order[self.pos]

    @property
    def item(self) -> dict:
        return self.items[self.idx]

    # ---- segments

    def seg_of(self, track_id: int, ci: int) -> tuple:
        return (track_id, bisect.bisect_right(self.splits.get(track_id, []), ci))

    def segments(self, item: dict | None = None) -> list:
        item = item or self.item
        return [(t, k) for t in item["tracklets"] for k in range(len(self.splits.get(t, [])) + 1)]

    def multi(self, item: dict | None = None) -> bool:
        """More than one segment: tracklet tags, selection and per-segment naming apply."""
        return len(self.segments(item)) > 1

    def toggle_split(self, row: dict) -> bool:
        """Shift+click on a crop: a different person from this crop on (again on the same crop: undo the split).

        Not on a tracklet's first crop. The tracklet's segment names are cleared, since its segments change.
        """
        tid, ci = int(row["track_id"]), int(row["ci"])
        first = min(r["ci"] for r in self.item["rows"] if r["track_id"] == tid)
        if ci == first:
            return False
        pts = self.splits.setdefault(tid, [])
        if ci in pts:
            pts.remove(ci)
        else:
            bisect.insort(pts, ci)
        if not pts:
            del self.splits[tid]
        for key in [k for k in self.tlabels if k[0] == tid]:
            del self.tlabels[key]
        self.selected = None
        return True

    def seg_range(self, item: dict, key: tuple) -> tuple:
        """Frames (ci) a segment covers. The frames between the last crop before a switch and the crop where the
        other person shows are in neither segment: the exact switch frame is not known."""
        tid, k = key
        pts = self.splits.get(tid, [])
        first, last = item["span"][tid]
        start = first if k == 0 else pts[k - 1]
        if k == len(pts):
            return start, last
        before = [r["ci"] for r in item["rows"] if r["track_id"] == tid and r["ci"] < pts[k]]
        return start, max(before)

    def tag(self, item: dict, key: tuple) -> str:
        tid, k = key
        n = item["tracklets"].index(tid) + 1
        return f"T{n}" + (chr(ord("a") + k) if tid in self.splits else "")

    # ---- labels

    def _snapshot(self) -> tuple:
        tids = set(self.item["tracklets"])
        return (
            self.pos,
            self.labels.get(self.idx),
            {k: v for k, v in self.tlabels.items() if k[0] in tids},
            {t: list(v) for t, v in self.splits.items() if t in tids},
        )

    def _set(self, jersey, verdict) -> None:
        self.history.append(self._snapshot())
        if verdict != "split":  # a whole-player label replaces any segment names and switch points
            tids = set(self.item["tracklets"])
            self.tlabels = {k: v for k, v in self.tlabels.items() if k[0] not in tids}
            self.splits = {t: v for t, v in self.splits.items() if t not in tids}
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
        """Two or more people. Keeps any segments named so far (then it is "split"), else plain "mixed"."""
        self._set(None, "split" if self.named() else "mixed")

    def bench(self) -> None:
        self._set(None, "bench")

    def skip(self) -> None:
        self._set(None, "skipped")

    def named(self) -> list:
        return [k for k in self.segments() if k in self.tlabels]

    def select(self, key) -> None:
        self.selected = None if key == self.selected else key

    def label_tracklet(self, jersey, verdict) -> bool:
        """Label the selected segment, then select the next unnamed one (or none)."""
        if self.selected is None or (verdict == "confirmed" and jersey not in self.valid_jerseys):
            return False
        self.tlabels[self.selected] = dict(jersey=jersey, verdict=verdict)
        segs = self.segments()
        k = segs.index(self.selected)
        self.selected = next((s for s in segs[k + 1 :] + segs[:k] if s not in self.tlabels), None)
        return True

    def clear_tracklet(self) -> None:
        if self.selected is not None:
            self.tlabels.pop(self.selected, None)

    def back(self) -> None:
        if not self.history:
            return
        pos, prev, prev_t, prev_s = self.history.pop()
        self.pos, self.selected = pos, None
        tids = set(self.item["tracklets"])
        if prev is None:
            self.labels.pop(self.idx, None)
        else:
            self.labels[self.idx] = prev
        self.tlabels = {k: v for k, v in self.tlabels.items() if k[0] not in tids} | prev_t
        self.splits = {t: v for t, v in self.splits.items() if t not in tids} | prev_s

    def table(self) -> pd.DataFrame:
        rows = [{"player_id": self.items[i]["player_id"], **lab} for i, lab in sorted(self.labels.items())]
        return pd.DataFrame(rows, columns=TRUTH_COLS)

    def tracklet_table(self) -> pd.DataFrame:
        """Segments of players saved as "split": named ones, plus unnamed segments of split tracklets (so the switch
        points survive a reload). Names on an unfinished player are not saved."""
        rows = []
        for i, lab in sorted(self.labels.items()):
            if lab["verdict"] != "split":
                continue
            it = self.items[i]
            for key in self.segments(it):
                if key not in self.tlabels and key[0] not in self.splits:
                    continue
                start, end = self.seg_range(it, key)
                lab_k = self.tlabels.get(key, dict(jersey=None, verdict="unnamed"))
                rows.append(
                    dict(track_id=key[0], seg=key[1], ci_start=start, ci_end=end, player_id=it["player_id"], **lab_k)
                )
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


def segment_text(tag: str, lab: dict | None) -> str:
    if not lab:
        return tag
    return f"{tag} {SHORT[lab['verdict']]}{lab['jersey'] if lab['verdict'] == 'confirmed' else ''}"


def mark_tracklets(tiles: list, item: dict, session: "Session") -> list:
    """Segment tag, frame color, and bars where the tracklet changes (yellow) or a switch was marked (magenta)."""
    if not session.multi(item):
        return tiles
    out, prev = [], None
    for tile, row in zip(tiles, item["rows"], strict=True):
        t = tile.copy()
        key = session.seg_of(row["track_id"], row["ci"])
        color = TRACKLET_COLORS[item["tracklets"].index(key[0]) % len(TRACKLET_COLORS)]
        sel = key == session.selected
        cv2.rectangle(t, (0, 0), (CROP_W - 1, CROP_H - 1), (255, 255, 255) if sel else color, 8 if sel else 3)
        if prev is not None and key != prev:
            cv2.rectangle(t, (0, 0), (7, CROP_H - 1), DIVIDER if key[0] != prev[0] else SPLIT_BAR, -1)
        text = segment_text(session.tag(item, key), session.tlabels.get(key))
        if key not in session.tlabels:
            sg = suggestion(item, session, key)
            if sg is not None:
                text += f" ~{sg[0]}" + (f" {sg[1]:.2f}" if sg[1] is not None else "")
        cv2.putText(t, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(t, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        out.append(t)
        prev = key
    return out


def render(tiles: list, session: "Session", total: int, typed: str, zv: ZoomView) -> np.ndarray:
    item, label = session.item, session.labels.get(session.idx)
    show = zv.apply(tile_grid(mark_tracklets(tiles, item, session), grid_cols(len(tiles))))
    status = f" [{label['verdict']}{' #' + str(label['jersey']) if label.get('jersey') else ''}]" if label else ""
    typed_txt = f"  typing: {typed}" if typed else ""
    header = f"{session.pos + 1}/{total}  {item['player_id']} ({item['role']}, {item['n_tracklets']} tracklets)"
    if session.selected is not None:
        sg = suggestion_text(suggestion(item, session, session.selected))
        header += f"  {session.tag(item, session.selected)} selected" + (f", {sg}" if sg else "")
        keys = "digits+Enter / a / n / o / s label this part | Backspace clear | Esc or click deselect"
    else:
        sg = suggestion_text(suggestion(item, session, None))
        if sg:
            header += f"  {sg}"
        named = len(session.named())
        if named:
            header += f"  {named}/{len(session.segments())} parts named: x or Enter finishes as split"
        keys = (
            "digits+Enter=jersey | n not on roster | x mixed (2 people) | o bench/sub (bib) | s skip | "
            + ("click a crop to name its part | " if session.multi() else "")
            + "shift+click where another person starts | b back | wheel zoom | right-click pan | r reset | q quit"
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
        cx, cy = zv.to_content(mx, my)
        cols = grid_cols(len(item["rows"]))
        col, row = int(cx // CROP_W), int(cy // CROP_H)
        j = row * cols + col
        if not (0 <= col < cols and row >= 0 and j < len(item["rows"])):
            return
        r = item["rows"][j]
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            session.toggle_split(r)
        elif session.multi():
            session.select(session.seg_of(r["track_id"], r["ci"]))

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
                sg = suggestion(item, session, session.selected if sel else None)
                if sg:
                    _ = session.label_tracklet(sg[0], "confirmed") if sel else session.confirm(sg[0])
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
    print(f"Saved {len(session.labels)} player labels to {truth_path}, named parts to {tracklet_path}. Run `apply`.")


def apply_identity(run: Path, roster: pd.DataFrame) -> tuple:
    """player_identity.csv (one row per tracklet) and identity_segments.csv (frame ranges of split tracklets)."""
    truth = pd.read_csv(run / "jersey_truth.csv")
    stitch = pd.read_csv(run / "tracklet_stitch.csv")
    tpath = run / "jersey_tracklets.csv"
    named = pd.read_csv(tpath) if tpath.exists() else pd.DataFrame(columns=TRACKLET_COLS)
    if "seg" not in named:  # saved before tracklets could be split
        named["seg"], named["ci_start"], named["ci_end"] = 0, np.nan, np.nan
    split_tids = set(named[named.seg > 0].track_id)
    confirmed = truth[truth.verdict == "confirmed"][["player_id", "jersey"]]
    out = stitch.merge(confirmed, on="player_id", how="left")
    # a named tracklet (of a "split" player) overrides its player's label: its jersey, or no identity. A tracklet
    # split at a switch point has no single identity: its parts are in identity_segments.csv instead.
    whole = named[~named.track_id.isin(split_tids)].set_index("track_id")
    has = out.track_id.isin(whole.index) | out.track_id.isin(split_tids)
    out.loc[has, "jersey"] = out.loc[has, "track_id"].map(whole.jersey.where(whole.verdict == "confirmed"))
    out["split_at_switch"] = out.track_id.isin(split_tids)
    out = out.merge(roster, on="jersey", how="left")
    segs = named[named.track_id.isin(split_tids) & (named.verdict == "confirmed")]
    segs = segs[["track_id", "seg", "ci_start", "ci_end", "player_id", "jersey"]].merge(roster, on="jersey", how="left")
    named_ok = named[named.verdict == "confirmed"]
    ids = pd.concat([confirmed, named_ok[["player_id", "jersey"]]])
    # the same jersey on two stitched players (whole-player or a named part): likely an under-merge
    dupes = ids.groupby("jersey").player_id.nunique()
    conflicts = dupes[dupes > 1]
    seen = set(ids.jersey.astype(int))
    report = {
        "target_goalkeeper_player_ids": int(stitch[stitch.role.isin(ROSTER_ROLES)].player_id.nunique()),
        "identified": int(confirmed.player_id.nunique()),
        "split_players": int((truth.verdict == "split").sum()),
        "parts_named_in_split_players": int(len(named_ok)),
        "tracklets_split_at_a_switch": len(split_tids),
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
            "split_players: mixed players whose tracklets (or parts of a tracklet, split where another person "
            "starts) were named one by one; unnamed ones get no identity. Split tracklets' parts are in "
            "identity_segments.csv (frame ranges in ci); frames around a switch belong to no part. "
            "bench_or_sub: a substitute/bibbed player that should have failed the on-pitch check upstream but "
            "didn't - feedback for pitch_mask.py/team_classify.py, not tracklet_stitch.py. All get no identity "
            "in the output except named tracklets and parts."
        ),
    }
    return out, segs, report


def cmd_apply(args) -> None:
    roster = pd.read_csv(ROSTER_FILE)
    out, segs, report = apply_identity(args.run, roster)
    out.to_csv(args.run / "player_identity.csv", index=False)
    segs.to_csv(args.run / "identity_segments.csv", index=False)
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
