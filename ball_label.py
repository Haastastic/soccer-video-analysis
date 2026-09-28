"""Hand-check the ball path: label the true ball position on a window of a clip, then score the path.

`label` shows one frame at a time (every --step-s seconds) with the pipeline's predicted ball circled. The ball
is only about 10 px wide: zoom in to click it precisely. One key or click per frame:

  y            the circle is on the ball (accept the prediction)
  left click   the ball is here
  mouse wheel  zoom, keeping the point under the cursor; the window grows with the zoom until it fills the
               screen, then scroll bars appear (drag or click)
  right click  center the zoomed view on that spot
  r            reset zoom to the whole frame
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

from sv_common import FOOTER_H, Cache, ZoomView, add_footer, cache_stride, read_frames, require_under_data

WINDOW = "ball label"
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
    """Draws a full-resolution frame with the prediction, shown through sv_common.ZoomView: the window grows with
    zoom until the screen constrains it (owner rule), then scroll bars. Clicks map back to frame pixels."""

    def __init__(self, scale: float):
        self.zv = ZoomView(reserve_h=FOOTER_H, base=scale)

    def render(self, img: np.ndarray, item: dict, index: int, total: int, label: dict | None) -> np.ndarray:
        show = img.copy()
        line = max(1, round(1.5 / self.zv.mag))  # about 1.5 window pixels at any zoom
        pred = (item["pred_x"], item["pred_y"]) if np.isfinite(item["pred_x"]) else None
        color = (0, 255, 255) if item["pred_kind"] == "interpolated" else (0, 0, 255)
        if pred:
            cv2.circle(show, (int(pred[0]), int(pred[1])), 20, color, line, cv2.LINE_AA)
        if label and np.isfinite(label["truth_x"]):
            cv2.circle(show, (int(label["truth_x"]), int(label["truth_y"])), 11, (0, 255, 0), line, cv2.LINE_AA)
        pred_text = f"pred {item['pred_kind']}" if pred else "no prediction"
        status = f" [{label['verdict']}]" if label else ""
        keys = "y accept | click ball | wheel zoom | right-click pan | r reset | x not visible | s skip | b back | q"
        where = f"{index + 1}/{total}  t={item['time_s']:.1f}s  {pred_text}{status}{self.zv.label()}"
        return add_footer(self.zv.apply(show), f"{where}   {keys}")

    def to_full(self, mx: int, my: int) -> tuple:
        return self.zv.to_content(mx, my)


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

    def on_mouse(event, mx, my, flags, _param):
        if viewer.zv.on_mouse(event, mx, my, flags):  # wheel zoom, right-click pan, scroll bars
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            pending["click"] = (mx, my)

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
                changed = True
            elif key == ord("y"):
                changed = session.accept()
            elif key == ord("x"):
                session.not_visible()
                changed = True
            elif key == ord("s"):
                session.skip()
                changed = True
            elif key == ord("b"):
                session.back()
                changed = True
            elif key == ord("r"):
                viewer.zv.reset()
            elif key == ord("q"):
                break
            if changed:
                viewer.zv.reset()  # a new frame starts at the whole picture
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
    lab.add_argument("--run", required=True, type=require_under_data)
    lab.add_argument("--start", type=float, default=150, help="window start, seconds into the clip")
    lab.add_argument("--duration", type=float, default=60)
    lab.add_argument("--step-s", type=float, default=0.5, help="seconds between labeled frames")
    lab.add_argument("--scale", type=float, default=0.7, help="display scale, lower it if the window is too big")
    lab.set_defaults(fn=cmd_label)
    sc = sub.add_parser("score", help="compare ball_path.csv with the labels")
    sc.add_argument("--run", required=True, type=require_under_data)
    sc.add_argument("--tol", type=float, default=20, help="pixels within which a prediction counts as on the ball")
    sc.set_defaults(fn=cmd_score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
