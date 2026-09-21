"""Map image positions to pitch coordinates (meters) from a few hand-picked anchor frames.

Automatic line detection is fragile on a zoomed follow-cam, so calibration is anchor based. For a handful of
frames the owner reads the pixel position of known pitch landmarks (line intersections, penalty box corners,
goal posts) and gives their pitch coordinates in meters, which also removes any assumption about the pitch
size. Each anchor frame gives a homography. The per-frame camera motion in the cache carries it to every other
frame, and the nearest anchor is used, so error grows with the time from an anchor. The report says by how much.

Anchors live in a git-ignored file (pitch_anchors.local.json):
  {"anchors": [{"time_s": 12.5, "points": [
      {"name": "far-left penalty corner", "img": [812, 301], "pitch": [0, 20.2]}, ...]}]}
Use at least 4 points per anchor, spread out and not on one line. Pitch axes: X along the length, Y across.

Commands:
  python pitch_calibrate.py frame --run data\\clipA --time 12.5   # still with a pixel grid, to read landmarks (local)
  python pitch_calibrate.py apply --run data\\clipA --anchors pitch_anchors.local.json
  python pitch_calibrate.py self-test
"""

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sv_common import CAM_COLS, Cache, cache_stride, require_under_data

ANCHORS_FILE = Path(__file__).resolve().parent / "pitch_anchors.local.json"


def solve_anchor(points: list) -> tuple:
    """Homography image -> pitch from anchor points, plus the reprojection RMS error in meters."""
    if len(points) < 4:
        raise ValueError("An anchor needs at least 4 points.")
    img = np.array([p["img"] for p in points], dtype=np.float64)
    pitch = np.array([p["pitch"] for p in points], dtype=np.float64)
    h, _ = cv2.findHomography(img, pitch, method=0)
    if h is None:
        raise ValueError("Degenerate anchor: the points are collinear or repeated.")
    return h, float(np.sqrt(((apply_h(h, img) - pitch) ** 2).sum(1).mean()))


def apply_h(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    out = np.hstack([pts, np.ones((len(pts), 1))]) @ h.T
    return out[:, :2] / out[:, 2:3]


class Calibration:
    """Per-frame image -> pitch homographies carried from anchor frames by the cached camera motion."""

    def __init__(self, cache: Cache, anchors: list):
        self.cache = cache
        self.anchor_ci, self.anchor_h, self.anchor_rms = [], [], []
        for a in anchors:
            ci = int(round(a["time_s"] * cache.fps))
            h, rms = solve_anchor(a["points"])
            self.anchor_ci.append(ci)
            self.anchor_h.append(h)
            self.anchor_rms.append(rms)
        self.anchor_ci = np.array(self.anchor_ci)

    def nearest(self, ci: np.ndarray) -> np.ndarray:
        return np.abs(np.asarray(ci)[:, None] - self.anchor_ci[None]).argmin(axis=1)

    def homography(self, ci: int, anchor: int) -> np.ndarray:
        """Image at cached frame ci -> pitch, via anchor frame a: Ha @ cum[a] @ cum_inv[ci]."""
        a = self.anchor_ci[anchor]
        return self.anchor_h[anchor] @ self.cache.cum[a] @ self.cache.cum_inv[ci]

    def to_pitch(self, ci, x, y):
        """Pitch coordinates of image points, with the seconds to the anchor that was used."""
        ci = np.asarray(ci, dtype=int)
        idx = self.nearest(ci)
        out = np.zeros((len(ci), 2))
        for k in range(len(ci)):
            out[k] = apply_h(self.homography(int(ci[k]), int(idx[k])), [[x[k], y[k]]])[0]
        return out, np.abs(ci - self.anchor_ci[idx]) / self.cache.fps

    def cross_check(self, anchors: list) -> list:
        """Carry each anchor's points through every other anchor and report the error in meters."""
        rows = []
        for i, a in enumerate(anchors):
            img = np.array([p["img"] for p in a["points"]], dtype=np.float64)
            truth = np.array([p["pitch"] for p in a["points"]], dtype=np.float64)
            for j in range(len(anchors)):
                if i == j:
                    continue
                h = self.homography(int(self.anchor_ci[i]), j)
                err = np.hypot(*(apply_h(h, img) - truth).T)
                rows.append(
                    {
                        "points_of_anchor": i,
                        "carried_from_anchor": j,
                        "gap_s": round(abs(int(self.anchor_ci[i]) - int(self.anchor_ci[j])) / self.cache.fps, 1),
                        "median_error_m": round(float(np.median(err)), 2),
                        "max_error_m": round(float(err.max()), 2),
                    }
                )
        return rows


def cmd_frame(args) -> None:
    cache = Cache(args.run / "cache")
    ci = int(round(args.time * cache.fps))
    cap = cv2.VideoCapture(str(args.run / "clip.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, ci * cache_stride(args.run))
    ok, img = cap.read()
    if not ok:
        raise SystemExit("Could not read that frame.")
    for x in range(0, img.shape[1], 100):
        cv2.line(img, (x, 0), (x, img.shape[0]), (0, 255, 255), 1)
        cv2.putText(img, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for y in range(0, img.shape[0], 100):
        cv2.line(img, (0, y), (img.shape[1], y), (0, 255, 255), 1)
        cv2.putText(img, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    dest = args.run / f"anchor_frame_{args.time:g}s.png"
    cv2.imwrite(str(dest), img)
    print(f"Wrote {dest} (local only, shows people). Read landmark pixel positions off the grid.")


def cmd_apply(args) -> None:
    anchors = json.loads(args.anchors.read_text())["anchors"]
    cache = Cache(args.run / "cache")
    cal = Calibration(cache, anchors)
    tr = pd.read_csv(args.run / "best_tracklets.csv.gz")
    foot_x, foot_y = ((tr.x1 + tr.x2) / 2).to_numpy(), tr.y2.to_numpy()
    xy, gap = cal.to_pitch(tr.ci.to_numpy(), foot_x, foot_y)
    out = tr[["pf", "ci", "track_id"]].copy()
    out["X_m"], out["Y_m"], out["anchor_gap_s"] = xy[:, 0].round(2), xy[:, 1].round(2), gap.round(1)
    out.to_csv(args.run / "tracklet_pitch_xy.csv.gz", index=False)
    report = {
        "anchors": len(anchors),
        "anchor_fit_rms_m": [round(r, 3) for r in cal.anchor_rms],
        "cross_check": cal.cross_check(anchors),
        "rows": int(len(out)),
        "rows_within_10s_of_anchor_pct": round(100 * float((gap <= 10).mean()), 1),
        "note": "Camera motion is modeled as a similarity, so error grows with the time from an anchor.",
    }
    (args.run / "pitch_calibration_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def self_test() -> None:
    """Exactness check: with camera motion that is a similarity, propagated homographies must be exact to 1 mm."""
    rng = np.random.default_rng(0)
    n = 60
    # Pitch to frame-0 image: a genuine perspective homography.
    h0 = np.array([[9.0, 2.0, 300.0], [0.5, 4.0, 200.0], [0.0006, 0.004, 1.0]])
    rels = [np.eye(3)]
    for _ in range(1, n):
        ang, sc = rng.normal(0, 0.01), 1 + rng.normal(0, 0.005)
        m = np.array(
            [
                [sc * np.cos(ang), -sc * np.sin(ang), rng.normal(0, 25)],
                [sc * np.sin(ang), sc * np.cos(ang), rng.normal(0, 8)],
                [0, 0, 1],
            ]
        )
        rels.append(m)
    cum = [np.eye(3)]
    for k in range(1, n):
        cum.append(rels[k] @ cum[-1])
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        (cache_dir / "meta.json").write_text(json.dumps({"n_cached_frames": n, "cache_fps": 30.0}))
        pd.DataFrame(columns=["ci", "conf"]).to_csv(cache_dir / "detections.csv.gz", index=False)
        cam = pd.DataFrame(
            [[k, *rels[k][:2].reshape(-1).tolist(), 400] for k in range(n)], columns=["ci", *CAM_COLS, "inliers"]
        )
        cam.to_csv(cache_dir / "camera.csv", index=False)
        cache = Cache(cache_dir)
        pitch_pts = np.array([[0, 0], [16.5, 0], [16.5, 40], [0, 40], [8, 20], [30, 10], [30, 30]], dtype=float)

        def anchor_at(k):
            img = apply_h(cum[k] @ h0, pitch_pts)  # pitch -> frame-0 image -> frame-k image
            return {
                "time_s": k / 30.0,
                "points": [{"img": i.tolist(), "pitch": p.tolist()} for i, p in zip(img, pitch_pts, strict=True)],
            }

        anchors = [anchor_at(5), anchor_at(55)]
        cal = Calibration(cache, anchors)
        worst = 0.0
        for k in (0, 12, 30, 40, 59):
            probe = np.array([[5.0, 5.0], [20.0, 25.0], [28.0, 12.0]])
            img = apply_h(cum[k] @ h0, probe)
            got, _ = cal.to_pitch([k] * 3, img[:, 0], img[:, 1])
            worst = max(worst, float(np.abs(got - probe).max()))
        worst_cross = max(r["max_error_m"] for r in cal.cross_check(anchors))
    print(f"anchor fit RMS (m): {[round(r, 6) for r in cal.anchor_rms]}")
    print(f"worst error at probe frames (m): {worst:.2e}, worst cross-anchor error (m): {worst_cross:.2e}")
    if worst > 1e-3 or worst_cross > 1e-3:  # 1 mm, above float noise
        raise SystemExit("SELF-TEST FAILED")
    print("self-test passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frame", help="write a still with a pixel grid to read landmark positions from")
    f.add_argument("--run", required=True, type=Path)
    f.add_argument("--time", required=True, type=float, help="seconds into the clip")
    f.set_defaults(fn=cmd_frame)
    a = sub.add_parser("apply", help="convert tracklet foot positions to pitch meters")
    a.add_argument("--run", required=True, type=Path)
    a.add_argument("--anchors", type=Path, default=ANCHORS_FILE)
    a.set_defaults(fn=cmd_apply)
    t = sub.add_parser("self-test", help="check the homography propagation on synthetic camera motion")
    t.set_defaults(fn=lambda _: self_test())
    args = ap.parse_args()
    if getattr(args, "run", None) is not None:
        args.run = require_under_data(args.run)
    args.fn(args)


if __name__ == "__main__":
    main()
