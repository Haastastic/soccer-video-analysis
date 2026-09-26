"""Automatic pitch anchors: snap the penalty-area lines onto the painted lines, chained from the owner's anchors.

Why: pitch_calibrate.py carries each owner anchor to other frames through the cached camera motion, which drifts
badly while the camera zooms (clipD: anchors 22 s apart disagreed by 9 to 16 m; clipE's zoomy first half needed 6
anchors). Placing anchors was about an hour per 5-minute window.

How: in each frame, a white top-hat finds thin bright lines (settings from a sweep scored against the owner's
anchors: 85% of the true penalty-area lines found). The goal-end geometry (goal line, six-yard and 18-yard boxes,
penalty arc; Law of the Game sizes, so no pitch size is assumed) is projected with the current estimate and a
small homography correction is fitted, coarse to fine, so the projected lines sit on painted ones - only within a
band around where they should be, so touchlines, goal frames and white kits stay out. Starting from each owner
anchor, this is chained every STEP_S seconds in both directions: each frame starts from the previous accepted fit
carried through 2 s of camera motion, which stays well inside what a snap can recover. A snap is accepted only if
at least MIN_ON_LINE of the projected lines land on painted lines AND it moved the lines at most MAX_MOVE_PX from
the chain's prediction (a wrong-line lock can fit well: one test case aligned 94% while 83 m off). Rejected or
box-less frames carry the last fit; the chain stops at the next owner anchor.

Accepted snaps are written as anchors in pitch_calibrate.py's format (the 10 standard landmarks, projected), marked
"auto": true, together with the owner's anchors. pitch_calibrate.py cross-checks only the owner's anchors.

Measured, holding out each owner anchor and calibrating from the rest, with the pipeline's blended conversion (owner
anchors alone -> with automatic ones, median (worst)): clipA 1.3 (16.3) -> 0.9 (5.0) m, clipB 1.3 (2.6) -> 0.9 (2.4)
m, clipD 1.9 (12.4) -> 1.3 (12.4) m, clipE 2.7 (9.4) -> 1.1 (1.5) m. The main gain is removing the large errors
during zooms. It does not bridge long stretches without the penalty area in view (camera at midfield: clipD's 204 s
anchor stays 12 m off), so the owner needs about ONE anchor per stretch where the box is visible, not one every 30
to 40 s; the report lists stretches without snaps far from any anchor.

Outputs (git-ignored): RUN/pitch_anchors_auto.local.json, RUN/pitch_autoanchor_report.json.

Example:
  python pitch_autoanchor.py run --run data\\clipE --anchors data\\clipE\\pitch_anchors.local.json
  python pitch_calibrate.py apply --run data\\clipE --anchors data\\clipE\\pitch_anchors_auto.local.json
  python pitch_autoanchor.py evaluate --run data\\clipE --anchors data\\clipE\\pitch_anchors.local.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from pitch_calibrate import Calibration, apply_h, solve_anchor
from sv_common import Cache, cache_stride, read_frames, require_under_data

STEP_S = 2.0
MIN_ON_LINE = 0.6  # 57% of prototype snaps passed; 96% of those were within 2 m
MAX_MOVE_PX = 40
BAND_PX = 150
# Chosen on clipE against both the owner's held-out anchors (at the penalty area) and the noise floor (still people
# near the touchlines, far from it): perspective terms fitted to penalty-area lines alone are poorly pinned and
# threw sideline positions around (noise floor 31.6 m/min with, 25.2 without); anchors 8 s apart instead of every
# snap blend fewer small errors (23.9). Held-out error: owner anchors alone 4.1 m median (20.9 worst), with these
# settings 1.09 m (1.52 worst).
PERSPECTIVE = False  # affine correction on top of the camera motion only
MIN_SPACING_S = 8.0  # keep automatic anchors at least this far apart
LANDMARKS = {  # the anchor UI's standard points, pitch meters (X from the goal line, Y toward the near touchline)
    "goalpost, near touchline": (0, 3.66),
    "goalpost, far touchline": (0, -3.66),
    "six-yard corner, near touchline": (5.5, 9.16),
    "six-yard corner, far touchline": (5.5, -9.16),
    "penalty spot": (11, 0),
    "eighteen-yard corner, near touchline": (16.5, 20.16),
    "eighteen-yard corner, far touchline": (16.5, -20.16),
    "arc tangent, near touchline": (16.5, 7.31),
    "arc tangent, far touchline": (16.5, -7.31),
    "arc apex": (20.15, 0),
}


def model_segments(step: float = 0.25) -> list:
    """Pitch-plane points along each goal-end line, one array per continuous line."""
    segs = [
        ((0, -20.16), (0, 20.16)),  # goal line across the penalty area
        ((0, -9.16), (5.5, -9.16)), ((5.5, -9.16), (5.5, 9.16)), ((5.5, 9.16), (0, 9.16)),  # six-yard box
        ((0, -20.16), (16.5, -20.16)), ((16.5, -20.16), (16.5, 20.16)), ((16.5, 20.16), (0, 20.16)),  # 18-yard box
    ]  # fmt: skip
    pts = []
    for (x0, y0), (x1, y1) in segs:
        n = max(2, int(np.hypot(x1 - x0, y1 - y0) / step))
        pts.append(np.c_[np.linspace(x0, x1, n), np.linspace(y0, y1, n)])
    ang = np.linspace(-np.arccos(5.5 / 9.15), np.arccos(5.5 / 9.15), 60)  # penalty arc outside the box
    pts.append(np.c_[11 + 9.15 * np.cos(ang), 9.15 * np.sin(ang)])
    return pts


def model_points(step: float = 0.25) -> np.ndarray:
    """All goal-end line points in one array (for fitting; see model_segments for drawing)."""
    return np.concatenate(model_segments(step))


def line_mask(img: np.ndarray) -> np.ndarray:
    """Thin bright, low-saturation structures: white top-hat (7 px) on brightness."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    tophat = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return ((tophat > 18) & (hsv[..., 1] < 130)).astype(np.uint8)


def snap(h_img2pitch: np.ndarray, mask: np.ndarray, P: np.ndarray) -> tuple:
    """Refine an image->pitch homography so the projected lines sit on painted ones. Returns (H, info)."""
    h0 = np.linalg.inv(h_img2pitch)  # pitch -> image
    base = apply_h(h0, P)
    H, W = mask.shape
    inside = (base[:, 0] > 5) & (base[:, 0] < W - 5) & (base[:, 1] > 5) & (base[:, 1] < H - 5)
    if inside.sum() < 50:
        return h_img2pitch, dict(ok=False, on_line=0.0, n=int(inside.sum()))
    # draw each line on its own: one polyline over all points would join unrelated lines with fake strokes
    band = np.zeros(mask.shape, np.uint8)
    segs = [apply_h(h0, seg).astype(np.int32).reshape(-1, 1, 2) for seg in model_segments()]
    cv2.polylines(band, segs, False, 1, BAND_PX)
    dist = cv2.distanceTransform(((mask == 0) | (band == 0)).astype(np.uint8), cv2.DIST_L2, 5)
    pts = base[inside]
    c = pts.mean(0)
    t = np.array([[1, 0, c[0]], [0, 1, c[1]], [0, 0, 1.0]])
    t_inv = np.linalg.inv(t)

    def warp(p):
        p = np.r_[p, np.zeros(8 - len(p))]
        return t @ np.array([[1 + p[0], p[1], p[2]], [p[3], 1 + p[4], p[5]], [p[6], p[7], 1.0]]) @ t_inv

    def resid(p, trunc):
        q = apply_h(warp(p), pts)
        return np.minimum(map_coordinates(dist, [q[:, 1], q[:, 0]], order=1, mode="nearest"), trunc)

    scale = np.array([1e-2, 1e-2, 10, 1e-2, 1e-2, 10, 1e-5, 1e-5])
    p = np.zeros(6)
    for trunc in (60, 25, 8):  # coarse to fine: a wide basin first, perspective terms only in the fine pass
        p = least_squares(
            lambda x, tr=trunc: resid(x, tr), p, x_scale=scale[: len(p)], loss="soft_l1", f_scale=trunc / 3
        ).x
        if trunc == 25 and PERSPECTIVE:
            p = np.r_[p, 0, 0]
    g = warp(p)
    on_line = float((resid(p, 8) < 2).mean())
    return np.linalg.inv(g @ h0), dict(ok=True, on_line=on_line, n=int(inside.sum()))


def chain(cache: Cache, frames: dict, seeds: list, P: np.ndarray) -> dict:
    """ci -> (image->pitch H, 'seed' | 'snap' | 'carry', on_line, seed index), chained outward from each seed and
    stopping at the next seed. Where two chains reach a frame, the one nearer its seed wins."""
    out = {}
    seed_cis = sorted(s[0] for s in seeds)
    for k, (s_ci, s_h) in enumerate(seeds):
        out[s_ci] = (s_h, "seed", 1.0, k, 0)
        for direction in (1, -1):
            stop = [c for c in seed_cis if (c - s_ci) * direction > 0]
            limit = (min(stop) if direction == 1 else max(stop)) if stop else None
            cis = sorted(
                c for c in frames if (c - s_ci) * direction > 0 and (limit is None or (limit - c) * direction > 0)
            )
            ci_prev, h_prev = s_ci, s_h
            for ci in cis if direction == 1 else cis[::-1]:
                h_pred = h_prev @ cache.cum[ci_prev] @ cache.cum_inv[ci]  # image(ci) -> image(prev) -> pitch
                h_snap, info = snap(h_pred, line_mask(frames[ci]), P)
                ok = info["ok"] and info["on_line"] >= MIN_ON_LINE
                if ok:  # a wrong-line lock can fit well: also require a small move from the prediction
                    move = np.hypot(*(apply_h(np.linalg.inv(h_pred), P) - apply_h(np.linalg.inv(h_snap), P)).T)
                    ok = float(np.median(move)) <= MAX_MOVE_PX
                h = h_snap if ok else h_pred
                dist_seed = abs(ci - s_ci)
                if ci not in out or out[ci][4] > dist_seed:
                    out[ci] = (h, "snap" if ok else "carry", info["on_line"], k, dist_seed)
                ci_prev, h_prev = ci, h
    return out


def auto_anchor(h: np.ndarray, ci: int, fps: float, on_line: float, shape: tuple) -> dict | None:
    """An anchor in pitch_calibrate.py's format: the standard landmarks projected into the frame."""
    inv = np.linalg.inv(h)
    pts = []
    for name, xy in LANDMARKS.items():
        u, v = apply_h(inv, [xy])[0]
        if 0 <= u < shape[1] and 0 <= v < shape[0]:
            pts.append(dict(name=name, img=[round(float(u), 1), round(float(v), 1)], pitch=list(xy)))
    if len(pts) < 4:
        return None
    return dict(time_s=round(ci / fps, 2), points=pts, auto=True, on_line=round(on_line, 3))


def run_chain(run: Path, anchors: list) -> tuple:
    cache = Cache(run / "cache")
    step = int(round(STEP_S * cache.fps))
    seeds = [(int(round(a["time_s"] * cache.fps)), solve_anchor(a["points"])[0]) for a in anchors]
    grid = sorted(set(range(0, cache.n, step)) | {s[0] for s in seeds})
    stride = cache_stride(run)  # cached frame ci is video frame ci * stride
    frames = {f // stride: img for f, img in read_frames(run / "clip.mp4", [c * stride for c in grid])}
    P = model_points()
    res = chain(cache, frames, seeds, P)
    shape = next(iter(frames.values())).shape
    auto = []
    for ci, (h, how, q, _k, _d) in sorted(res.items()):
        if how != "snap" or (auto and ci / cache.fps - auto[-1]["time_s"] < MIN_SPACING_S):
            continue
        a = auto_anchor(h, ci, cache.fps, q, shape)
        if a is not None:
            auto.append(a)
    return cache, res, auto


def gaps_report(cache: Cache, res: dict, anchors: list, auto: list) -> list:
    """Stretches of STEP_S frames with no snap, and how far they are from any anchor (owner or automatic)."""
    times = np.array(sorted([a["time_s"] for a in anchors] + [a["time_s"] for a in auto]))
    carry = sorted(ci / cache.fps for ci, v in res.items() if v[1] == "carry")
    out, start, prev = [], None, None
    for t in carry + [None]:
        if t is not None and prev is not None and t - prev <= STEP_S * 1.5:
            prev = t
            continue
        if start is not None:
            mid = (start + prev) / 2
            far = float(np.max([np.abs(times - x).min() for x in (start, mid, prev)]))
            out.append(dict(from_s=round(start, 1), to_s=round(prev, 1), max_s_from_anchor=round(far, 1)))
        start = prev = t
    return [g for g in out if g["max_s_from_anchor"] > 10]


def cmd_run(args) -> None:
    anchors = [a for a in json.loads(args.anchors.read_text())["anchors"] if not a.get("auto")]
    cache, res, auto = run_chain(args.run, anchors)
    out = args.run / "pitch_anchors_auto.local.json"
    out.write_text(json.dumps(dict(anchors=anchors + auto), indent=1))
    kinds = [v[1] for v in res.values()]
    report = dict(
        owner_anchors=len(anchors),
        auto_anchors=len(auto),
        frames_checked=len(res),
        snapped_pct=round(100 * kinds.count("snap") / len(kinds), 1),
        stretches_without_snaps_far_from_anchors=gaps_report(cache, res, anchors, auto),
        note=(
            "Add an owner anchor inside a listed stretch if one shows the penalty area; otherwise its positions "
            "rely on camera motion. Then: pitch_calibrate.py apply --anchors " + str(out)
        ),
    )
    (args.run / "pitch_autoanchor_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def cmd_evaluate(args) -> None:
    """Hold out each owner anchor in turn: calibrate from the others (plus snaps) and measure its points' error,
    next to the error of the owner's other anchors alone (today's manual calibration)."""
    anchors = [a for a in json.loads(args.anchors.read_text())["anchors"] if not a.get("auto")]
    rows = []
    for k, held in enumerate(anchors):
        rest = anchors[:k] + anchors[k + 1 :]
        cache, _res, auto = run_chain(args.run, rest)
        ci = int(round(held["time_s"] * cache.fps))
        img = np.array([p["img"] for p in held["points"]], float)
        truth = np.array([p["pitch"] for p in held["points"]], float)
        errs = {}
        for name, set_ in (("manual", rest), ("auto", rest + auto)):
            # the same blended conversion the pipeline uses (pitch_calibrate.Calibration.to_pitch)
            xy, _ = Calibration(cache, set_).to_pitch(np.full(len(img), ci), img[:, 0], img[:, 1])
            errs[name] = round(float(np.median(np.hypot(*(xy - truth).T))), 2)
        rows.append(dict(held_out_s=held["time_s"], manual_err_m=errs["manual"], with_auto_err_m=errs["auto"]))
        print(rows[-1], flush=True)
    m = np.median([r["manual_err_m"] for r in rows])
    a = np.median([r["with_auto_err_m"] for r in rows])
    print(json.dumps(dict(held_out=rows, median_manual_m=float(m), median_with_auto_m=float(a)), indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, hlp in (
        ("run", cmd_run, "write automatic anchors next to the owner's"),
        ("evaluate", cmd_evaluate, "hold out each owner anchor and compare errors"),
    ):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--run", required=True, type=require_under_data)
        p.add_argument("--anchors", required=True, type=Path, help="the owner's anchors JSON")
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
