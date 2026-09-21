"""Hand-check the ball path: label the true ball position on a window of a clip, then score the path.

`label` shows one frame at a time (every --step-s seconds) with the pipeline's predicted ball circled and a
zoomed inset, because the ball is only about 10 px wide. One key or click per frame:

  y            the circle is on the ball (accept the prediction)
  left click   the ball is here (click on the main image, or on the zoomed inset for precision)
  right click  move the zoomed inset to that spot, to look around
  x            the ball is not visible in this frame
  s            skip, unsure
  b            back one frame (undo the last label)
  q            save and quit

Progress is saved after every frame, so rerunning `label` resumes where you stopped. Labels go to
OUT/ball_truth.csv (git-ignored under data/). The frames show people: keep them local.

`score` compares OUT/ball_path.csv with the labels and writes OUT/ball_score.json.

Example:
  python ball_label.py label --run data\\clipA --start 150 --duration 60
  python ball_label.py score --run data\\clipA
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sv_common import Cache, cache_stride, read_frames

WINDOW = "ball label"
INSET_SRC, INSET_ZOOM = 128, 3  # source pixels shown in the inset, and its magnification
TRUTH_COLS = ["time_s", "ci", "pred_x", "pred_y", "pred_kind", "truth_x", "truth_y", "visible", "verdict"]


def build_items(run: Path, start: float, duration: float, step_s: float) -> list:
    """The frames to label, each with the pipeline's prediction (or none)."""
    cache = Cache(run / "cache")
    ball = pd.read_csv(run / "ball_path.csv").drop_duplicates("ci").set_index("ci")
    items = []
    for t in np.arange(start, start + duration, step_s):
        ci = int(round(t * cache.fps))
        if ci >= cache.n:
            break
        if ci in ball.index:
            r = ball.loc[ci]
            items.append(
                dict(time_s=round(ci / cache.fps, 3), ci=ci, pred_x=float(r.x), pred_y=float(r.y), pred_kind=r.kind)
            )
        else:
            items.append(dict(time_s=round(ci / cache.fps, 3), ci=ci, pred_x=np.nan, pred_y=np.nan, pred_kind="none"))
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
                    self.labels[i] = dict(truth_x=r.truth_x, truth_y=r.truth_y, visible=r.visible, verdict=r.verdict)
        self.idx = next((i for i in range(len(items)) if i not in self.labels), len(items))

    @property
    def done(self) -> bool:
        return self.idx >= len(self.items)

    def _set(self, tx, ty, visible, verdict) -> None:
        self.labels[self.idx] = dict(truth_x=tx, truth_y=ty, visible=visible, verdict=verdict)
        self.idx += 1

    def accept(self) -> bool:
        it = self.items[self.idx]
        if not np.isfinite(it["pred_x"]):
            return False  # nothing to accept
        self._set(it["pred_x"], it["pred_y"], 1, "accepted")
        return True

    def click(self, x: float, y: float) -> None:
        self._set(float(x), float(y), 1, "clicked")

    def not_visible(self) -> None:
        self._set(np.nan, np.nan, 0, "not_visible")

    def skip(self) -> None:
        self._set(np.nan, np.nan, np.nan, "skipped")

    def back(self) -> None:
        if self.idx > 0:
            self.idx -= 1
            self.labels.pop(self.idx, None)

    def table(self) -> pd.DataFrame:
        rows = [{**self.items[i], **lab} for i, lab in sorted(self.labels.items())]
        return pd.DataFrame(rows, columns=TRUTH_COLS)


class Viewer:
    """Draws a frame with the prediction and an inset, and maps mouse clicks back to full-resolution pixels."""

    def __init__(self, scale: float):
        self.scale = scale
        self.inset_center = None  # full-res (x, y) the inset looks at, None means follow the prediction
        self.inset_rect = (0, 0, 0, 0)  # x, y, w, h on the displayed image
        self.inset_origin = (0, 0)  # full-res top-left of the inset source
        self.shown_size = (0, 0)

    def render(self, img: np.ndarray, item: dict, index: int, total: int, label: dict | None) -> np.ndarray:
        h, w = img.shape[:2]
        pred = (item["pred_x"], item["pred_y"]) if np.isfinite(item["pred_x"]) else None
        center = self.inset_center or pred or (w / 2, h / 2)
        x0 = int(np.clip(center[0] - INSET_SRC / 2, 0, w - INSET_SRC))
        y0 = int(np.clip(center[1] - INSET_SRC / 2, 0, h - INSET_SRC))
        self.inset_origin = (x0, y0)
        inset = cv2.resize(
            img[y0 : y0 + INSET_SRC, x0 : x0 + INSET_SRC],
            None,
            fx=INSET_ZOOM,
            fy=INSET_ZOOM,
            interpolation=cv2.INTER_CUBIC,
        )

        def mark(canvas, pt, color, k, radius):
            cv2.circle(canvas, (int(pt[0] * k), int(pt[1] * k)), radius, color, 1, cv2.LINE_AA)

        color = (0, 255, 255) if item["pred_kind"] == "interpolated" else (0, 0, 255)
        show = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        if pred:
            mark(show, pred, color, self.scale, 14)
            mark(inset, (pred[0] - x0, pred[1] - y0), color, INSET_ZOOM, 22)
        if label and np.isfinite(label["truth_x"]):
            truth = (label["truth_x"], label["truth_y"])
            mark(show, truth, (0, 255, 0), self.scale, 8)
            mark(inset, (truth[0] - x0, truth[1] - y0), (0, 255, 0), INSET_ZOOM, 12)
        ih, iw = inset.shape[:2]
        sh, sw = show.shape[:2]
        show[0:ih, sw - iw : sw] = inset
        cv2.rectangle(show, (sw - iw, 0), (sw - 1, ih - 1), (255, 255, 255), 1)
        self.inset_rect = (sw - iw, 0, iw, ih)
        self.shown_size = (sw, sh)
        pred_text = f"pred {item['pred_kind']}" if pred else "no prediction"
        status = f" [{label['verdict']}]" if label else ""
        keys = "y accept | click ball | x not visible | s skip | b back | q quit"
        text = f"{index + 1}/{total}  t={item['time_s']:.1f}s  {pred_text}{status}   {keys}"
        cv2.rectangle(show, (0, sh - 24), (sw, sh), (0, 0, 0), -1)
        cv2.putText(show, text, (6, sh - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return show

    def to_full(self, mx: int, my: int) -> tuple:
        """Map a click on the displayed image to full-resolution pixels, using the inset when the click is in it."""
        ix, iy, iw, ih = self.inset_rect
        if ix <= mx < ix + iw and iy <= my < iy + ih:
            return self.inset_origin[0] + (mx - ix) / INSET_ZOOM, self.inset_origin[1] + (my - iy) / INSET_ZOOM
        return mx / self.scale, my / self.scale

    def in_inset(self, mx: int, my: int) -> bool:
        ix, iy, iw, ih = self.inset_rect
        return ix <= mx < ix + iw and iy <= my < iy + ih


def cmd_label(args) -> None:
    run = args.run
    truth_path = run / "ball_truth.csv"
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

    def on_mouse(event, mx, my, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending["click"] = (mx, my)
        elif event == cv2.EVENT_RBUTTONDOWN:
            pending["pan"] = (mx, my)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        while not session.done:
            item = session.items[session.idx]
            if item["ci"] not in jpgs:
                session.skip()  # frame could not be read
                continue
            img = cv2.imdecode(jpgs[item["ci"]], cv2.IMREAD_COLOR)
            cv2.imshow(WINDOW, viewer.render(img, item, session.idx, len(items), session.labels.get(session.idx)))
            key = cv2.waitKey(30) & 0xFF
            changed = False
            if "click" in pending:
                session.click(*viewer.to_full(*pending.pop("click")))
                viewer.inset_center, changed = None, True
            elif "pan" in pending:
                viewer.inset_center = viewer.to_full(*pending.pop("pan"))
            elif key == ord("y"):
                changed = session.accept()
                viewer.inset_center = None
            elif key == ord("x"):
                session.not_visible()
                viewer.inset_center, changed = None, True
            elif key == ord("s"):
                session.skip()
                viewer.inset_center, changed = None, True
            elif key == ord("b"):
                session.back()
                viewer.inset_center, changed = None, True
            elif key == ord("q"):
                break
            if changed:
                session.table().to_csv(truth_path, index=False)
    finally:
        cv2.destroyAllWindows()
        session.table().to_csv(truth_path, index=False)
    print(f"Saved {len(session.labels)} of {len(items)} labels to {truth_path}. Run `score` when you are done.")


def score_path(ball: pd.DataFrame, truth: pd.DataFrame, tol_px: float) -> dict:
    """Compare the ball path with hand labels. Skipped and unlabeled frames are ignored."""
    t = truth[truth.verdict != "skipped"].copy()
    ball = ball.drop_duplicates("ci").set_index("ci")
    t["has_pred"] = t.ci.isin(ball.index)
    px = t.ci.map(ball.x) if len(ball) else pd.Series(np.nan, index=t.index)
    py = t.ci.map(ball.y) if len(ball) else pd.Series(np.nan, index=t.index)
    kind = t.ci.map(ball.kind) if len(ball) else pd.Series("none", index=t.index)
    t["pred_kind"] = kind.fillna("none")
    t["err_px"] = np.hypot(px - t.truth_x, py - t.truth_y)
    vis, hid = t[t.visible == 1], t[t.visible == 0]
    correct = vis.has_pred & (vis.err_px <= tol_px)

    def pct(num, den):
        return round(100 * float(num) / den, 1) if den else None

    by_kind = {}
    for k in ("detected", "interpolated"):
        sub = vis[vis.pred_kind == k]
        by_kind[k] = {"frames": int(len(sub)), "correct_pct": pct((sub.err_px <= tol_px).sum(), len(sub))}
    return {
        "labeled_frames": int(len(t)),
        "ball_visible_frames": int(len(vis)),
        "ball_not_visible_frames": int(len(hid)),
        "tolerance_px": tol_px,
        "path_present_when_visible_pct": pct(vis.has_pred.sum(), len(vis)),
        "path_correct_when_visible_pct": pct(correct.sum(), len(vis)),
        "path_wrong_when_visible_pct": pct((vis.has_pred & ~correct).sum(), len(vis)),
        "path_missing_when_visible_pct": pct((~vis.has_pred).sum(), len(vis)),
        "precision_of_predictions_pct": pct(correct.sum(), int(vis.has_pred.sum() + hid.has_pred.sum())),
        "path_present_when_not_visible_pct": pct(hid.has_pred.sum(), len(hid)),
        "by_prediction_kind_when_visible": by_kind,
        "median_error_px_when_correct": round(float(vis.err_px[correct].median()), 1) if correct.any() else None,
        "note": "Hand-labeled sample, small. Precision counts predictions on labeled frames only.",
    }


def cmd_score(args) -> None:
    truth = pd.read_csv(args.run / "ball_truth.csv")
    ball = pd.read_csv(args.run / "ball_path.csv")
    report = score_path(ball, truth, args.tol)
    (args.run / "ball_score.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lab = sub.add_parser("label", help="label the true ball position, one frame at a time")
    lab.add_argument("--run", required=True, type=Path)
    lab.add_argument("--start", type=float, default=150, help="window start, seconds into the clip")
    lab.add_argument("--duration", type=float, default=60)
    lab.add_argument("--step-s", type=float, default=0.5, help="seconds between labeled frames")
    lab.add_argument("--scale", type=float, default=0.7, help="display scale, lower it if the window is too big")
    lab.set_defaults(fn=cmd_label)
    sc = sub.add_parser("score", help="compare ball_path.csv with the labels")
    sc.add_argument("--run", required=True, type=Path)
    sc.add_argument("--tol", type=float, default=20, help="pixels within which a prediction counts as on the ball")
    sc.set_defaults(fn=cmd_score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
