"""Hand-label who has the ball on a window of a clip, then score events.py against it.

`label` shows one frame at a time (every --step-s seconds), with every player-candidate tracklet's box drawn
and color-coded by team, plus the pipeline's detected ball position if there is one. One key or click per frame:

  click box    that player has the ball
  n            no one has the ball right now (loose: rolling, in the air, contested)
  x            the ball is not visible / you cannot tell who has it
  s            skip, unsure
  b            back one frame (undo the last label)
  q            save and quit

  mouse wheel  zoom in or out, centered on the cursor - for a crowd of overlapping players
  right click  pan the zoomed view to that spot
  r            reset zoom to the full frame

Zoom resets to the full frame automatically whenever you move to a new frame. Progress is saved after every
frame, so rerunning `label` resumes where you stopped. Labels go to OUT/events_truth.csv (git-ignored under
data/). The frames show people: keep them local.

`score` derives ground-truth touches, possessions, passes and turnovers from the labeled per-frame possessor
(a change of possessor is a touch; same-team is a pass, different-team is a turnover; segments are built only
from consecutive LABELED frames, never bridged over an unlabeled or not-visible gap, since there is no ground
truth for what happened there) and compares them with a fresh run of events.csv within --tol-s seconds.

Example:
  python event_label.py label --run data\\clipA --start 150 --duration 60
  python event_label.py score --run data\\clipA
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from events import TEAM_ROLES, detect
from sv_common import Cache, cache_stride, read_frames, require_under_data

WINDOW = "event label"
TOL_FRAMES = 5  # a tracklet or ball point within this many cached frames of the target ci counts as present
ROLE_COLOR = {"target": (255, 180, 0), "opponent": (0, 0, 255), "goalkeeper": (0, 255, 255)}
TRUTH_COLS = ["time_s", "ci", "truth_track_id", "verdict"]


def build_items(run: Path, start: float, duration: float, step_s: float) -> list:
    """The frames to label, each with nearby player boxes and the ball prediction."""
    cache = Cache(run / "cache")
    ball = pd.read_csv(run / "ball_path.csv").drop_duplicates("ci").set_index("ci").sort_index()
    tr = pd.read_csv(run / "best_tracklets.csv.gz")
    roles = pd.read_csv(run / "tracklet_roles.csv", index_col="track_id")
    keep = roles.index[roles.player_candidate & roles.role.isin(TEAM_ROLES)]
    tr = tr[tr.track_id.isin(keep)].sort_values("ci")
    items = []
    for t in np.arange(start, start + duration, step_s):
        ci = int(round(t * cache.fps))
        if ci >= cache.n:
            break
        boxes = []
        for tid, d in tr.groupby("track_id"):
            j = (d.ci - ci).abs().idxmin()
            row = d.loc[j]
            if abs(int(row.ci) - ci) <= TOL_FRAMES:
                boxes.append(
                    dict(
                        track_id=int(tid),
                        x1=float(row.x1),
                        y1=float(row.y1),
                        x2=float(row.x2),
                        y2=float(row.y2),
                        role=roles.loc[tid, "role"],
                    )
                )
        bi = ball.index[(ball.index - ci).map(abs) <= TOL_FRAMES]
        if len(bi):
            r = ball.loc[bi[np.argmin(np.abs(bi - ci))]]
            ball_xy, ball_kind = (float(r.x), float(r.y)), r.kind
        else:
            ball_xy, ball_kind = None, "none"
        items.append(dict(time_s=round(ci / cache.fps, 3), ci=ci, boxes=boxes, ball_xy=ball_xy, ball_kind=ball_kind))
    return items


class Session:
    """Labels for a list of items. Pure logic, no window, so it can be tested."""

    def __init__(self, items: list, saved: pd.DataFrame | None = None):
        self.items = items
        self.labels = {}
        if saved is not None and len(saved):
            by_ci = {int(r.ci): r for r in saved.itertuples()}
            for i, it in enumerate(items):
                r = by_ci.get(it["ci"])
                if r is not None:
                    self.labels[i] = dict(truth_track_id=r.truth_track_id, verdict=r.verdict)
        self.idx = next((i for i in range(len(items)) if i not in self.labels), len(items))

    @property
    def done(self) -> bool:
        return self.idx >= len(self.items)

    def _set(self, track_id, verdict) -> None:
        self.labels[self.idx] = dict(truth_track_id=track_id, verdict=verdict)
        self.idx += 1

    def possess(self, track_id: int) -> None:
        self._set(track_id, "possessed")

    def loose(self) -> None:
        self._set(np.nan, "loose")

    def not_visible(self) -> None:
        self._set(np.nan, "not_visible")

    def skip(self) -> None:
        self._set(np.nan, "skipped")

    def back(self) -> None:
        if self.idx > 0:
            self.idx -= 1
            self.labels.pop(self.idx, None)

    def table(self) -> pd.DataFrame:
        rows = [
            {**{k: v for k, v in self.items[i].items() if k in ("time_s", "ci")}, **lab}
            for i, lab in sorted(self.labels.items())
        ]
        return pd.DataFrame(rows, columns=TRUTH_COLS)


class Viewer:
    """Draws a frame with player boxes and the ball, and maps clicks to the box they landed in."""

    ZOOM_STEP = 1.25
    MAX_ZOOM = 8.0

    def __init__(self, scale: float):
        self.base_scale = scale
        self.canvas_w, self.canvas_h = 0, 0  # fixed display size, set on the first frame
        self.zoom = 1.0  # 1.0 = whole frame visible; higher crops in, centered on (cx, cy)
        self.cx, self.cy = None, None  # full-res center of the current view
        self.view = (0, 0, 0, 0)  # x0, y0, view_w, view_h in full-res pixels, from the last render
        self.boxes_shown = []  # [(track_id, x1, y1, x2, y2)] in displayed pixels, for click hit-testing

    def reset_zoom(self) -> None:
        self.zoom, self.cx, self.cy = 1.0, None, None

    def zoom_by(self, factor: float, at_full: tuple) -> None:
        self.cx, self.cy = at_full
        self.zoom = float(np.clip(self.zoom * factor, 1.0, self.MAX_ZOOM))

    def pan_to(self, at_full: tuple) -> None:
        self.cx, self.cy = at_full

    def render(self, img: np.ndarray, item: dict, index: int, total: int, label: dict | None) -> np.ndarray:
        ih, iw = img.shape[:2]
        if not self.canvas_w:
            self.canvas_w, self.canvas_h = int(iw * self.base_scale), int(ih * self.base_scale)
        cx = self.cx if self.cx is not None else iw / 2
        cy = self.cy if self.cy is not None else ih / 2
        vw, vh = iw / self.zoom, ih / self.zoom
        x0 = float(np.clip(cx - vw / 2, 0, max(0, iw - vw)))
        y0 = float(np.clip(cy - vh / 2, 0, max(0, ih - vh)))
        self.view = (x0, y0, vw, vh)
        crop = img[int(y0) : int(round(y0 + vh)), int(x0) : int(round(x0 + vw))]
        show = cv2.resize(crop, (self.canvas_w, self.canvas_h), interpolation=cv2.INTER_AREA)
        sx, sy = self.canvas_w / vw, self.canvas_h / vh

        def to_show(px, py):
            return int((px - x0) * sx), int((py - y0) * sy)

        self.boxes_shown = []
        truth_id = label["truth_track_id"] if label and np.isfinite(label.get("truth_track_id", np.nan)) else None
        for b in item["boxes"]:
            x1, y1 = to_show(b["x1"], b["y1"])
            x2, y2 = to_show(b["x2"], b["y2"])
            color = ROLE_COLOR.get(b["role"], (200, 200, 200))
            thick = 3 if b["track_id"] == truth_id else 1
            cv2.rectangle(show, (x1, y1), (x2, y2), color, thick)
            cv2.putText(show, str(b["track_id"]), (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            self.boxes_shown.append((b["track_id"], x1, y1, x2, y2))
        if item["ball_xy"]:
            bx, by = to_show(*item["ball_xy"])
            color = (0, 255, 0) if item["ball_kind"] == "detected" else (0, 165, 255)
            cv2.circle(show, (bx, by), max(4, int(10 * sx / self.base_scale)), color, 2)
        sh, sw = show.shape[:2]
        status = f" [{label['verdict']}]" if label else ""
        zoom_txt = f" zoom {self.zoom:.1f}x" if self.zoom > 1.0 else ""
        keys = "click=has ball | wheel zoom | right-click pan | r reset | n loose | x n/a | s skip | b back | q quit"
        text = f"{index + 1}/{total}  t={item['time_s']:.1f}s  ball={item['ball_kind']}{status}{zoom_txt}   {keys}"
        cv2.rectangle(show, (0, sh - 24), (sw, sh), (0, 0, 0), -1)
        cv2.putText(show, text, (6, sh - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return show

    def to_full(self, mx: int, my: int) -> tuple:
        x0, y0, vw, vh = self.view
        return x0 + mx * vw / self.canvas_w, y0 + my * vh / self.canvas_h

    def track_id_at(self, mx: int, my: int) -> int | None:
        hits = [(tid, x1, y1, x2, y2) for tid, x1, y1, x2, y2 in self.boxes_shown if x1 <= mx <= x2 and y1 <= my <= y2]
        if not hits:
            return None
        # smallest box wins, so overlapping players resolve to whichever box the click is most specifically inside
        return min(hits, key=lambda h: (h[3] - h[1]) * (h[4] - h[2]))[0]


def cmd_label(args) -> None:
    run = args.run
    truth_path = run / "events_truth.csv"
    items = build_items(run, args.start, args.duration, args.step_s)
    saved = pd.read_csv(truth_path) if truth_path.exists() else None
    session = Session(items, saved)
    stride = cache_stride(run)
    print(f"{len(items)} frames to label, {session.idx} already done. Loading frames...")
    jpgs = {}
    for clip_frame, img in read_frames(run / "clip.mp4", [it["ci"] * stride for it in items]):
        jpgs[clip_frame // stride] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]
    viewer = Viewer(args.scale)
    pending = {}

    def on_mouse(event, mx, my, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending["click"] = (mx, my)
        elif event == cv2.EVENT_RBUTTONDOWN:
            pending["pan"] = (mx, my)
        elif event == cv2.EVENT_MOUSEWHEEL:
            pending["zoom"] = (mx, my, 1 if flags > 0 else -1)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        while not session.done:
            item = session.items[session.idx]
            if item["ci"] not in jpgs:
                session.skip()
                continue
            img = cv2.imdecode(jpgs[item["ci"]], cv2.IMREAD_COLOR)
            cv2.imshow(WINDOW, viewer.render(img, item, session.idx, len(items), session.labels.get(session.idx)))
            key = cv2.waitKey(30) & 0xFF
            changed = False
            if "zoom" in pending:
                mx, my, direction = pending.pop("zoom")
                factor = viewer.ZOOM_STEP if direction > 0 else 1 / viewer.ZOOM_STEP
                viewer.zoom_by(factor, viewer.to_full(mx, my))
            elif "pan" in pending:
                viewer.pan_to(viewer.to_full(*pending.pop("pan")))
            elif "click" in pending:
                mx, my = pending.pop("click")
                tid = viewer.track_id_at(mx, my)
                if tid is not None:
                    session.possess(tid)
                    changed = True
            elif key == ord("r"):
                viewer.reset_zoom()
            elif key == ord("n"):
                session.loose()
                changed = True
            elif key == ord("x"):
                session.not_visible()
                changed = True
            elif key == ord("s"):
                session.skip()
                changed = True
            elif key == ord("b"):
                session.back()
                changed = True
            elif key == ord("q"):
                break
            if changed:
                viewer.reset_zoom()
                session.table().to_csv(truth_path, index=False)
    finally:
        cv2.destroyAllWindows()
        session.table().to_csv(truth_path, index=False)
    print(f"Saved {len(session.labels)} of {len(items)} labels to {truth_path}. Run `score` when you are done.")


def truth_events(truth: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    """Ground-truth touches, possessions, passes and turnovers from the labeled per-frame possessor.

    Segments are built only from consecutive rows in the (sorted) truth table whose ci is exactly one label-step
    apart, so an unlabeled, skipped or not-visible frame breaks a segment instead of being bridged over.
    """
    t = truth.sort_values("ci").reset_index(drop=True)
    step = int(t.ci.diff().dropna().median()) if len(t) > 1 else 1
    known = t.verdict.isin(["possessed", "loose"])
    consecutive = np.r_[False, t.ci.diff().to_numpy()[1:] == step]
    group = (~(known & consecutive)).cumsum()
    events = []
    for _, seg in t[known].groupby(group[known]):
        seg = seg.reset_index(drop=True)
        runs, i = [], 0
        while i < len(seg):
            j = i
            while j + 1 < len(seg) and (
                (pd.isna(seg.truth_track_id[j + 1]) and pd.isna(seg.truth_track_id[i]))
                or seg.truth_track_id[j + 1] == seg.truth_track_id[i]
            ):
                j += 1
            runs.append((i, j))
            i = j + 1
        for a, b in runs:
            tid = seg.truth_track_id[a]
            if pd.isna(tid):
                continue  # a "loose" run is not a possession
            events.append(
                dict(
                    type="possession",
                    time_s=float(seg.time_s[a]),
                    end_s=float(seg.time_s[b]),
                    track_id=int(tid),
                    team=TEAM_ROLES[roles.loc[int(tid), "role"]],
                    to_track_id=np.nan,
                    to_team="",
                )
            )
        for (a1, _), (a2, _) in zip(runs[:-1], runs[1:], strict=True):
            id1, id2 = seg.truth_track_id[a1], seg.truth_track_id[a2]
            if pd.notna(id2):
                # a touch is a player gaining the ball, whether from another player or a loose ball; the reverse
                # (a player releasing the ball into a "loose" run) is not itself recorded as a separate touch.
                events.append(
                    dict(
                        type="touch",
                        time_s=float(seg.time_s[a2]),
                        end_s=np.nan,
                        track_id=int(id2),
                        team=TEAM_ROLES[roles.loc[int(id2), "role"]],
                        to_track_id=np.nan,
                        to_team="",
                    )
                )
            if pd.isna(id1) or pd.isna(id2) or id1 == id2:
                continue
            t1, t2 = TEAM_ROLES[roles.loc[int(id1), "role"]], TEAM_ROLES[roles.loc[int(id2), "role"]]
            events.append(
                dict(
                    type="pass" if t1 == t2 else "turnover",
                    time_s=float(seg.time_s[a2]),
                    end_s=np.nan,
                    track_id=int(id1),
                    team=t1,
                    to_track_id=int(id2),
                    to_team=t2,
                )
            )
    return pd.DataFrame(events, columns=["type", "time_s", "end_s", "track_id", "team", "to_track_id", "to_team"])


def match_events(truth: pd.DataFrame, pred: pd.DataFrame, tol_s: float) -> dict:
    """Precision and recall per event type, matching by type, time within tol_s, and track_id."""
    report = {}
    for etype in ("touch", "possession", "pass", "turnover"):
        tr = truth[truth.type == etype]
        pr = pred[pred.type == etype] if len(pred) else pred
        matched_truth = np.zeros(len(tr), dtype=bool)
        matched_pred = np.zeros(len(pr), dtype=bool)
        for i, (_, trow) in enumerate(tr.iterrows()):
            for j, (_, prow) in enumerate(pr.iterrows()):
                if matched_pred[j]:
                    continue
                if abs(trow.time_s - prow.time_s) <= tol_s and trow.track_id == prow.track_id:
                    matched_truth[i], matched_pred[j] = True, True
                    break

        def pct(num, den):
            return round(100 * float(num) / den, 1) if den else None

        report[etype] = {
            "truth_events": int(len(tr)),
            "predicted_events": int(len(pr)),
            "recall_pct": pct(matched_truth.sum(), len(tr)),
            "precision_pct": pct(matched_pred.sum(), len(pr)),
        }
    return report


def cmd_score(args) -> None:
    truth = pd.read_csv(args.run / "events_truth.csv")
    roles = pd.read_csv(args.run / "tracklet_roles.csv", index_col="track_id")
    gt = truth_events(truth, roles)
    pred, _, _ = detect(args.run, args.min_confidence)
    if len(pred):
        pred = pred[pred.time_s.between(truth.time_s.min() - args.tol_s, truth.time_s.max() + args.tol_s)]
    report = {
        "labeled_frames": int(len(truth)),
        "by_verdict": truth.verdict.value_counts().to_dict(),
        "ground_truth_events": gt.type.value_counts().to_dict() if len(gt) else {},
        "match": match_events(gt, pred, args.tol_s),
        "tol_s": args.tol_s,
        "note": "Hand-labeled sample, small. Segments never bridge an unlabeled or not-visible gap.",
    }
    (args.run / "events_score.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lab = sub.add_parser("label", help="label who has the ball, one frame at a time")
    lab.add_argument("--run", required=True, type=require_under_data)
    lab.add_argument("--start", type=float, default=150, help="window start, seconds into the clip")
    lab.add_argument("--duration", type=float, default=60)
    lab.add_argument("--step-s", type=float, default=0.5, help="seconds between labeled frames")
    lab.add_argument("--scale", type=float, default=0.7, help="display scale, lower it if the window is too big")
    lab.set_defaults(fn=cmd_label)
    sc = sub.add_parser("score", help="derive ground-truth events from the labels and compare with events.csv")
    sc.add_argument("--run", required=True, type=require_under_data)
    sc.add_argument("--tol-s", type=float, default=1.0, help="seconds within which a prediction counts as a match")
    sc.add_argument(
        "--min-confidence", type=float, default=0.0, help="passed to events.detect; 0 includes review-queue events"
    )
    sc.set_defaults(fn=cmd_score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
