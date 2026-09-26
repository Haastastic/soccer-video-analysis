"""Automatic pitch calibration with no owner anchors per window: a fixed-position pan/tilt/zoom camera model.

Why: placing anchors was about an hour per 5-minute window, and pitch_autoanchor.py still needed one owner anchor per
stretch where a penalty area shows, because snapping only refines a nearby guess (a guess from far away locked onto
the wrong lines, once 83 m off with 94% of its lines on paint).

Finding that makes it possible: the camera stands still for the whole game. Every owner anchor (clipA, B, D, E, both
halves) implies the same camera centre within a few metres, and one camera fitted to all 20 anchors reprojects them
about as well as each anchor's own free homography (4.5 to 13.8 px vs 3.9 to 10.3 px). So a frame has only pan, tilt,
zoom (and a roll near zero) to find, not 8 homography terms, and three numbers can be searched exhaustively: no
starting guess, no wrong-goal mixups (the pan says which end is in view), and midfield frames are pinned by the
touchlines and the halfway line, which the penalty-area-only snapping could not use.

Per game (fit): camera centre, pitch length and the two touchline positions, from the owner anchors already placed
in any windows of that game (each anchor file's goal side is detected from its homography's handedness), refined so
the painted lines of those frames sit on the model. Written to data/pitch_camera.local.json.

Per window (run): every STEP_S seconds, the pose is refined from the previous fix carried by the cached camera
motion; if that scores too low (or there is no recent fix), a global search over pan/tilt/zoom runs on a quarter-
resolution distance map and the best distinct candidates are refined. Score = line length in pixels that lands on
painted lines minus length that lands on grass (a wrong lock can sit partly on paint, but then drags other lines
over grass). A fix is kept only if it scores at least ACCEPT. Fixes are written as anchors (a grid of projected pitch
points) in pitch_calibrate.py's format, so the rest of the pipeline is unchanged:
  python pitch_ptz.py run --run data\\clipF
  python pitch_calibrate.py apply --run data\\clipF --anchors data\\clipF\\pitch_anchors_ptz.local.json

Pitch axes follow pitch_calibrate.py: X metres from the reference goal (the one the owner's anchors mostly used),
Y across, positive toward the near touchline.

Check against the owner's anchors of a window (not used by run; fit the camera without that window's anchors first
for an honest number) and the noise floor:
  python pitch_ptz.py evaluate --run data\\clipE --anchors data\\clipE\\pitch_anchors.local.json
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from pitch_autoanchor import line_mask
from pitch_calibrate import Calibration
from pitch_mask import grass_mask
from sv_common import Cache, cache_stride, read_frames, require_under_data

CAMERA_FILE = Path(__file__).resolve().parent / "data" / "pitch_camera.local.json"
STEP_S = 2.0
ACCEPT = 2000.0  # anchor frames: correct fits scored 2600 to 6300, wrong locks 35 to 1600
MAX_CARRY_S = 30.0  # start from the previous fix carried by camera motion if it is this recent
ON_PX, OFF_PX = 2.5, 5.0  # a projected line point is on paint within ON_PX, on grass beyond OFF_PX
COARSE_ON_PX, COARSE_OFF_PX = 12.0, 24.0  # the coarse grid steps about 25 px, so its tolerance must match
TRUNC_PX = 12.0
MIN_STROKE_PX = 40  # connected paint pixels; smaller specks are turf texture
GRID_PAN = np.arange(-60, 60.01, 0.5)
GRID_TILT = np.arange(2.5, 12.01, 0.5)
GRID_F = 1800 * 1.06 ** np.arange(0, 23)  # 1800 to 6500 px
BOUNDS = ([-90, 0, np.log(1200), -3], [90, 25, np.log(9000), 3])  # pan, tilt (deg), log focal (px), roll (deg)


# ---------------------------------------------------------------- pitch model


def _seg(a, b, step):
    n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / step) + 1)
    return np.c_[np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)]


def _seg_n(a, b, n):
    return np.c_[np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)]


def pitch_lines(cam: dict, step: float, counts: list | None = None) -> list:
    """Pitch-plane points along each painted line: goal lines, both penalty and goal areas with the arcs, the two
    touchlines and the halfway line. Law sizes for the goal ends; length and touchlines from the camera file.
    counts fixes the number of points per line (the joint fit changes the length but needs a fixed layout)."""
    L, y0, y1 = cam["length_m"], cam["touchline_far_y"], cam["touchline_near_y"]
    k = iter(counts) if counts else None

    def _seg(a, b, step):  # noqa: F811 - same sampling, optionally with fixed counts
        n = next(k) if k else max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / step) + 1)
        return _seg_n(a, b, n)

    lines = [_seg((0, y0), (L, y0), step), _seg((0, y1), (L, y1), step), _seg((L / 2, y0), (L / 2, y1), step)]
    ang = np.linspace(-np.arccos(5.5 / 9.15), np.arccos(5.5 / 9.15), max(8, int(12 / step)))
    for x0, s in ((0.0, 1.0), (L, -1.0)):
        lines.append(_seg((x0, y0), (x0, y1), step))
        for d, w in ((5.5, 9.16), (16.5, 20.16)):
            lines += [_seg((x0, -w), (x0 + s * d, -w), step), _seg((x0 + s * d, -w), (x0 + s * d, w), step)]
            lines.append(_seg((x0 + s * d, w), (x0, w), step))
        if k:
            next(k)
        lines.append(np.c_[x0 + s * (11 + 9.15 * np.cos(ang)), 9.15 * np.sin(ang)])
    r = cam.get("centre_ring_m")
    if r:  # the painted ring round the centre spot pins the zoom in midfield, where only two straight lines show
        ang = np.linspace(0, 2 * np.pi, next(k) if k else int(2 * np.pi * r / step))
        lines.append(np.c_[L / 2 + r * np.cos(ang), r * np.sin(ang)])
    return lines


class Model:
    """Line points relative to the camera centre, with which consecutive points belong to the same line."""

    def __init__(self, cam: dict, step: float, counts: list | None = None):
        lines = pitch_lines(cam, step, counts)
        self.counts = [len(v) for v in lines]
        self.P = np.concatenate(lines)
        self.rel = np.c_[self.P, np.zeros(len(self.P))] - np.asarray(cam["centre_m"])
        seg = np.concatenate([np.full(len(v), i) for i, v in enumerate(lines)])
        self.same_next = np.r_[seg[1:] == seg[:-1], False]


def rotation(pan: float, tilt: float, roll: float = 0.0) -> np.ndarray:
    """World -> camera rotation. pan turns about the vertical (0 = looking straight across the pitch from the near
    side), tilt looks down, roll turns about the optical axis. Rows: image right, image down, forward."""
    p, t = np.radians(pan), np.radians(tilt)
    fwd = np.array([np.sin(p) * np.cos(t), -np.cos(p) * np.cos(t), -np.sin(t)])
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(fwd, right), fwd])
    if roll:
        c, s = np.cos(np.radians(roll)), np.sin(np.radians(roll))
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R
    return R


def project(rel: np.ndarray, q, centre_px) -> np.ndarray:
    """Image positions of points (relative to the camera centre) for pose q = (pan, tilt, log focal, roll).
    Points behind the camera are NaN."""
    Xc = rel @ rotation(q[0], q[1], q[3] if len(q) > 3 else 0.0).T
    z = np.where(Xc[:, 2] > 0.1, Xc[:, 2], np.nan)
    return np.exp(q[2]) * Xc[:, :2] / z[:, None] + centre_px


def homography(cam: dict, q, centre_px) -> np.ndarray:
    """Pitch plane (X, Y) -> image homography for pose q."""
    R = rotation(q[0], q[1], q[3] if len(q) > 3 else 0.0)
    f = np.exp(q[2])
    K = np.array([[f, 0, centre_px[0]], [0, f, centre_px[1]], [0, 0, 1.0]])
    t = -R @ np.asarray(cam["centre_m"])
    return K @ np.c_[R[:, 0], R[:, 1], t]


def pose_readout(R: np.ndarray) -> tuple:
    fwd = R[2]
    return float(np.degrees(np.arctan2(fwd[0], -fwd[1]))), float(np.degrees(np.arcsin(-fwd[2])))


# ---------------------------------------------------------------- scoring and search


def pitch_region(img: np.ndarray) -> np.ndarray:
    """The turf: the largest connected grass area (lines and players inside filled), grown a little so touchlines
    on its edge count. Trees and hedges pass a grass-colour test too, and the white-line mask turns foliage into
    dense speckle that a coarse search happily puts pitch lines on; they are separate areas beyond the track."""
    g = grass_mask(img).astype(np.uint8)  # half resolution
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(g, connectivity=4)
    if n < 2:
        return np.ones(img.shape[:2], bool)
    turf = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    turf = cv2.morphologyEx(turf, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    turf = cv2.dilate(turf, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    return cv2.resize(turf, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def distance_map(img: np.ndarray) -> np.ndarray:
    """Distance (px) to the nearest painted-line pixel on the turf."""
    paint = ((line_mask(img) > 0) & pitch_region(img)).astype(np.uint8)
    # turf texture leaves 1 to 5 px specks everywhere; with the coarse search's wide tolerance they make any line on
    # the turf look painted, so only connected strokes of MIN_STROKE_PX pixels or more count
    n, lab, stats, _ = cv2.connectedComponentsWithStats(paint, connectivity=8)
    keep = np.r_[False, stats[1:, cv2.CC_STAT_AREA] >= MIN_STROKE_PX]
    paint = keep[lab]
    return cv2.distanceTransform((~paint).astype(np.uint8), cv2.DIST_L2, 5)


def score(model: Model, d: np.ndarray, uv: np.ndarray, scale=1.0, on_px=ON_PX, off_px=OFF_PX) -> float:
    """Line length (full-resolution px) on paint minus PEN x length on grass, for points inside the frame."""
    step = np.hypot(*np.diff(uv, axis=0).T)
    step = np.minimum(np.nan_to_num(np.r_[np.where(model.same_next[:-1], step, 0), 0]), 40 / scale)
    h, w = d.shape
    ok = np.isfinite(uv[:, 0]) & (uv[:, 0] > 1) & (uv[:, 0] < w - 2) & (uv[:, 1] > 1) & (uv[:, 1] < h - 2)
    if ok.sum() < 20:
        return -1e9
    dd = d[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] * scale
    wt = step[ok] * scale
    return float((wt * (dd < on_px)).sum() - (wt * (dd > off_px)).sum())


def refine(model: Model, d: np.ndarray, q0, centre_px) -> np.ndarray:
    h, w = d.shape

    def resid(q):
        uv = project(model.rel, q, centre_px)
        ok = np.isfinite(uv[:, 0]) & (uv[:, 0] > 1) & (uv[:, 0] < w - 2) & (uv[:, 1] > 1) & (uv[:, 1] < h - 2)
        r = np.full(len(uv), TRUNC_PX)
        r[ok] = np.minimum(map_coordinates(d, [uv[ok, 1], uv[ok, 0]], order=1), TRUNC_PX)
        return r

    q0 = np.clip(np.r_[q0, np.zeros(4 - len(q0))], BOUNDS[0], BOUNDS[1])
    return least_squares(resid, q0, bounds=BOUNDS, loss="soft_l1", f_scale=3, x_scale=[0.1, 0.1, 0.01, 0.1]).x


def global_search(coarse: Model, fine: Model, d: np.ndarray, centre_px, keep: int = 10) -> list:
    """[(score, q)] best first: grid over pan/tilt/zoom on a quarter-resolution map, then the best distinct
    candidates refined at full resolution."""
    ds = cv2.resize(d, (d.shape[1] // 4, d.shape[0] // 4), interpolation=cv2.INTER_AREA) / 4
    c4 = np.asarray(centre_px) / 4
    cand = []
    for f in GRID_F:
        for tilt in GRID_TILT:
            for pan in GRID_PAN:
                uv = project(coarse.rel, (pan, tilt, np.log(f / 4)), c4)
                cand.append((score(coarse, ds, uv, 4.0, COARSE_ON_PX, COARSE_OFF_PX), pan, tilt, f))
    cand.sort(key=lambda c: -c[0])
    top = []
    for c in cand:
        if all(abs(c[1] - t[1]) > 2 or abs(np.log(c[3] / t[3])) > 0.15 for t in top):
            top.append(c)
        if len(top) == keep:
            break
    out = []
    for _s, pan, tilt, f in top:
        q = refine(fine, d, (pan, tilt, np.log(f), 0.0), centre_px)
        out.append((score(fine, d, project(fine.rel, q, centre_px)), q))
    return sorted(out, key=lambda r: -r[0])


def carry(model: Model, q_prev, M: np.ndarray, centre_px) -> np.ndarray:
    """Pose in a new frame predicted from the previous pose and the camera motion M (image prev -> image now)."""
    uv = project(model.rel, q_prev, centre_px)
    ok = np.isfinite(uv[:, 0])
    ok &= (
        (uv[:, 0] > -500)
        & (uv[:, 0] < 2 * centre_px[0] + 500)
        & (uv[:, 1] > -300)
        & (uv[:, 1] < 2 * centre_px[1] + 300)
    )
    idx = np.flatnonzero(ok)[:: max(1, ok.sum() // 300)]
    tgt = cv2.perspectiveTransform(uv[idx][None].astype(np.float64), M)[0]
    rel = model.rel[idx]
    return least_squares(lambda q: (project(rel, q, centre_px) - tgt).ravel(), q_prev, bounds=BOUNDS).x


# ---------------------------------------------------------------- per window


def grid_anchor(cam: dict, q, centre_px, shape, time_s: float, s: float) -> dict | None:
    """An anchor in pitch_calibrate.py's format: a grid of pitch points projected with the fitted pose."""
    L, y0, y1 = cam["length_m"], cam["touchline_far_y"], cam["touchline_near_y"]
    xs, ys = np.r_[np.arange(0, L, 5.0), L], np.r_[np.arange(y0, y1, 5.0), y1]
    P = np.array([(x, y) for x in xs for y in ys])
    rel = np.c_[P, np.zeros(len(P))] - np.asarray(cam["centre_m"])
    uv = project(rel, q, centre_px)
    h, w = shape[:2]
    ok = np.isfinite(uv[:, 0]) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    if ok.sum() < 6:
        return None
    pts = [
        dict(
            name=f"grid {x:g},{y:g}",
            img=[round(float(u), 1), round(float(v), 1)],
            pitch=[round(float(x), 2), round(float(y), 2)],
        )
        for (x, y), (u, v) in zip(P[ok], uv[ok], strict=True)
    ]
    pose = dict(pan=round(float(q[0]), 3), tilt=round(float(q[1]), 3), focal_px=round(float(np.exp(q[2])), 1),
                roll=round(float(q[3]), 3))  # fmt: skip
    return dict(time_s=round(time_s, 2), points=pts, auto=True, ptz=True, score=round(s), pose=pose)


def solve_window(run: Path, cam: dict, log=print) -> tuple:
    cache = Cache(run / "cache")
    step = int(round(STEP_S * cache.fps))
    grid = list(range(0, cache.n, step))
    stride = cache_stride(run)
    coarse, fine = Model(cam, 1.0), Model(cam, 0.5)
    rows, prev = [], None
    t0 = time.time()
    for f, img in read_frames(run / "clip.mp4", [c * stride for c in grid]):
        ci = f // stride
        centre_px = (img.shape[1] / 2, img.shape[0] / 2)
        d = distance_map(img)
        how, s, q = "none", -1e9, None
        if prev is not None and (ci - prev[0]) / cache.fps <= MAX_CARRY_S:
            M = cache.cum[ci] @ cache.cum_inv[prev[0]]
            q = refine(fine, d, carry(fine, prev[1], M, centre_px), centre_px)
            s, how = score(fine, d, project(fine.rel, q, centre_px)), "tracked"
        if s < ACCEPT:
            res = global_search(coarse, fine, d, centre_px)
            if res and res[0][0] > s:
                (s, q), how = res[0], "searched"
        ok = s >= ACCEPT
        if ok:
            prev = (ci, q)
        rows.append(dict(ci=ci, time_s=round(ci / cache.fps, 2), how=how if ok else "rejected", score=round(s),
                         pan=q[0] if q is not None else np.nan, tilt=q[1] if q is not None else np.nan,
                         focal_px=np.exp(q[2]) if q is not None else np.nan, roll=q[3] if q is not None else np.nan,
                         shape=img.shape))  # fmt: skip
        if len(rows) % 15 == 0:
            n_ok = sum(r["how"] != "rejected" for r in rows)
            log(f"  {rows[-1]['time_s']:6.1f}s  {n_ok}/{len(rows)} frames fixed  ({time.time() - t0:.0f}s)")
    return cache, rows


def cmd_run(args) -> None:
    cam = json.loads(args.camera.read_text())
    cache, rows = solve_window(args.run, cam)
    anchors, last = [], -1e9
    for r in rows:
        if r["how"] == "rejected" or r["time_s"] - last < args.min_spacing - 1e-6:
            continue
        q = (r["pan"], r["tilt"], np.log(r["focal_px"]), r["roll"])
        a = grid_anchor(cam, q, (r["shape"][1] / 2, r["shape"][0] / 2), r["shape"], r["time_s"], r["score"])
        if a is not None:
            anchors.append(a)
            last = r["time_s"]
    out = args.run / "pitch_anchors_ptz.local.json"
    out.write_text(json.dumps(dict(camera=str(args.camera), anchors=anchors), indent=1))
    poses = pd.DataFrame(rows).drop(columns="shape")
    poses.to_csv(args.run / "pitch_ptz_poses.csv", index=False)
    fixed = poses[poses.how != "rejected"]
    gaps = np.diff(np.r_[0.0, fixed.time_s.to_numpy(), cache.n / cache.fps])
    report = dict(
        frames=len(poses),
        fixed_pct=round(100 * len(fixed) / len(poses), 1),
        tracked=int((poses.how == "tracked").sum()),
        searched=int((poses.how == "searched").sum()),
        rejected=int((poses.how == "rejected").sum()),
        longest_gap_between_fixes_s=round(float(gaps.max()), 1),
        anchors_written=len(anchors),
        note="Then: pitch_calibrate.py apply --run RUN --anchors " + str(out),
    )
    (args.run / "pitch_ptz_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


# ---------------------------------------------------------------- per game


def _handedness(points: list) -> float:
    """Sign of the camera height implied by an anchor's homography (flips when X is measured from the other goal)."""
    pitch = np.array([p["pitch"] for p in points], float)
    img = np.array([p["img"] for p in points], float)
    H, _ = cv2.findHomography(pitch, img, 0)
    Hc = np.array([[1, 0, -960.0], [0, 1, -540.0], [0, 0, 1]]) @ H
    h1, h2 = Hc[:, 0], Hc[:, 1]
    f2 = [e for e in (-(h1[0] * h2[0] + h1[1] * h2[1]) / (h1[2] * h2[2]),) if e > 0]
    f = np.sqrt(f2[0]) if f2 else 3000.0
    M = np.diag([1 / f, 1 / f, 1]) @ Hc
    M /= np.linalg.norm(M[:, 0])
    if M[2, 2] < 0:
        M = -M
    r3 = np.cross(M[:, 0], M[:, 1])
    R = np.column_stack([M[:, 0], M[:, 1], r3])
    return float(np.sign((-R.T @ M[:, 2])[2]))


def load_anchor_sets(pairs: str) -> list:
    """'data\\clipA:data\\clipA\\pitch_anchors.local.json,...' -> [(run, path, owner anchors)]."""
    out = []
    for item in pairs.split(","):
        # Separator is the first colon after a possible drive letter (C:\...).
        sep = item.find(":", 2)
        run, path = (item[:sep], item[sep + 1:]) if sep >= 0 else (item, None)
        run = Path(run)
        path = Path(path) if path else run / "pitch_anchors.local.json"
        anchors = [a for a in json.loads(path.read_text())["anchors"] if not a.get("auto") and len(a["points"]) >= 4]
        out.append((run, path, anchors))
    return out


def cmd_fit(args) -> None:
    sets = load_anchor_sets(args.anchors)
    # goal side per file: the handedness most anchors share is the reference goal
    sides = {str(p): float(np.sign(np.sum([_handedness(a["points"]) for a in A]))) for _r, p, A in sets}
    ref = float(np.sign(np.sum([_handedness(a["points"]) for _r, _p, A in sets for a in A])))
    mirrored = {p: s != ref for p, s in sides.items()}
    items = [(run, a, mirrored[str(p)]) for run, p, A in sets for a in A]
    print(
        f"{len(items)} owner anchors; measured from the other goal: {[p for p, m in mirrored.items() if m] or 'none'}"
    )

    def world(a, m, L):
        P = np.array([p["pitch"] for p in a["points"]], float)
        if m:
            P[:, 0] = L - P[:, 0]
        return P

    # stage 1: one camera through every owner point (principal point at the frame centre, no distortion)
    C0, L0 = np.array(args.centre_guess, float), args.length_guess
    x0 = [*C0, L0]
    for _run, a, m in items:
        best = None
        for f in (2000, 2500, 3000, 3500, 4500):
            K = np.array([[f, 0, 960], [0, f, 540], [0, 0, 1.0]])
            P3 = np.c_[world(a, m, L0), np.zeros(len(a["points"]))]
            _ok, rv, tv = cv2.solvePnP(P3, np.array([p["img"] for p in a["points"]], float), K, None)
            R, _ = cv2.Rodrigues(rv)
            e = np.linalg.norm((-R.T @ tv).ravel() - C0)
            if best is None or e < best[0]:
                pan, tilt = pose_readout(R)
                best = (e, [pan, tilt, np.log(f), 0.0])
        x0 += best[1]

    def unpack(x):
        return x[:3], x[3], x[4:].reshape(-1, 4)

    def point_resid(x):
        C, L, Q = unpack(x)
        r = []
        for (_run, a, m), q in zip(items, Q, strict=True):
            rel = np.c_[world(a, m, L), np.zeros(len(a["points"]))] - C
            r.append((project(rel, q, (960, 540)) - np.array([p["img"] for p in a["points"]], float)).ravel())
        return np.nan_to_num(np.concatenate(r), nan=1e3)

    has_mirror = any(m for *_x, m in items)
    lo = [-np.inf] * 3 + ([50.0] if has_mirror else [L0 - 1e-6]) + [BOUNDS[0][i % 4] for i in range(4 * len(items))]
    hi = [np.inf] * 3 + ([130.0] if has_mirror else [L0 + 1e-6]) + [BOUNDS[1][i % 4] for i in range(4 * len(items))]
    x = least_squares(point_resid, np.clip(x0, lo, hi), bounds=(lo, hi), loss="soft_l1", f_scale=10).x
    C, L, Q = unpack(x)
    print(f"stage 1 (owner points): centre {np.round(C, 2)} m, length {L:.1f} m, "
          f"median residual {np.median(np.abs(point_resid(x))):.1f} px")  # fmt: skip

    # stage 2: painted lines of the anchor frames, with the touchlines scanned first, then everything jointly
    maps = []
    for run, group in pd.DataFrame([(str(r), i) for i, (r, _a, _m) in enumerate(items)], columns=["run", "i"]).groupby(
        "run"
    ):
        cache = Cache(Path(run) / "cache")
        stride = cache_stride(Path(run))
        want = {int(round(items[i][1]["time_s"] * cache.fps)) * stride: i for i in group.i}
        for fno, img in read_frames(Path(run) / "clip.mp4", list(want)):
            maps.append((want[fno], distance_map(img)))
    maps.sort()
    cam = dict(centre_m=C.tolist(), length_m=float(L), touchline_far_y=-34.0, touchline_near_y=34.0)
    for key, rng in (("touchline_far_y", np.arange(-45, -20, 0.25)), ("touchline_near_y", np.arange(20, 45, 0.25))):
        best = []
        for y in rng:
            seg = _seg((0, y), (L, y), 0.5)
            rel = np.c_[seg, np.zeros(len(seg))] - C
            tot = 0.0
            for i, d in maps:
                uv = project(rel, Q[i], (960, 540))
                h, w = d.shape
                ok = np.isfinite(uv[:, 0]) & (uv[:, 0] > 1) & (uv[:, 0] < w - 2) & (uv[:, 1] > 1) & (uv[:, 1] < h - 2)
                if ok.sum() >= 30:  # a touchline seen by this frame: the share of it on paint
                    tot += float((d[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] < 2).mean())
            best.append((tot, y))
        cam[key] = float(max(best)[1])
    print(f"touchlines from the line scan: far {cam['touchline_far_y']}, near {cam['touchline_near_y']}")

    fine = Model(cam, 1.0)
    n_pts = len(fine.P)

    def joint_resid(z):
        Cz, Lz, yf, yn = z[:3], z[3], z[4], z[5]
        Qz = z[6:].reshape(-1, 4)
        cz = dict(centre_m=Cz, length_m=Lz, touchline_far_y=yf, touchline_near_y=yn)
        m = Model(cz, 1.0, fine.counts)
        r = []
        for i, d in maps:
            uv = project(m.rel, Qz[i], (960, 540))
            h, w = d.shape
            ok = np.isfinite(uv[:, 0]) & (uv[:, 0] > 1) & (uv[:, 0] < w - 2) & (uv[:, 1] > 1) & (uv[:, 1] < h - 2)
            rr = np.zeros(n_pts)  # points off the frame say nothing about this frame
            rr[ok] = np.minimum(map_coordinates(d, [uv[ok, 1], uv[ok, 0]], order=1), TRUNC_PX)
            r.append(rr)
        pr = point_resid(np.r_[Cz, Lz, Qz.ravel()])
        return np.r_[np.concatenate(r) * args.line_weight, pr]

    z0 = np.r_[C, L, cam["touchline_far_y"], cam["touchline_near_y"], Q.ravel()]
    lo2 = [-np.inf] * 3 + [lo[3], -60, 15] + lo[4:]
    hi2 = [np.inf] * 3 + [hi[3], -15, 60] + hi[4:]
    z = least_squares(joint_resid, z0, bounds=(lo2, hi2), loss="soft_l1", f_scale=3, max_nfev=args.max_nfev).x
    C, L = z[:3], z[3]
    cam = dict(
        centre_m=[round(float(v), 3) for v in C],
        length_m=round(float(L), 3),
        touchline_far_y=round(float(z[4]), 3),
        touchline_near_y=round(float(z[5]), 3),
        reference="X from the goal most owner anchors used; Y positive toward the near touchline",
        mirrored_anchor_files=[p for p, m in mirrored.items() if m],
        fitted_from=[str(p) for _r, p, _A in sets],
    )
    Qf = z[6:].reshape(-1, 4)
    res = point_resid(np.r_[C, L, Qf.ravel()]).reshape(-1, 2)
    cam["owner_point_residual_px"] = dict(median=round(float(np.median(np.hypot(*res.T))), 2),
                                          p90=round(float(np.percentile(np.hypot(*res.T), 90)), 2))  # fmt: skip
    args.out.write_text(json.dumps(cam, indent=2))
    print(json.dumps(cam, indent=2))


def cmd_ring(args) -> None:
    """Radius of the painted ring round the centre spot, scanned on midfield frames of a window. At the first venue
    the white ring was a logo's edge at about 7.5 m, not the 9.15 m law circle, so it is measured, not assumed."""
    cam = json.loads(args.camera.read_text())
    cache = Cache(args.run / "cache")
    stride = cache_stride(args.run)
    times = [float(t) for t in args.times.split(",")]
    maps = [
        distance_map(img)
        for _f, img in read_frames(args.run / "clip.mp4", [int(round(t * cache.fps)) * stride for t in times])
    ]
    rows = []
    for r in np.arange(args.min_r, args.max_r + 1e-6, 0.25):
        m = Model(dict(cam, centre_ring_m=float(r)), 0.5)
        tot = 0.0
        for d in maps:
            c = (d.shape[1] / 2, d.shape[0] / 2)
            best = max(global_search(Model(dict(cam, centre_ring_m=float(r)), 1.0), m, d, c, keep=10)[:1])[0]
            tot += best
        rows.append((tot, float(r)))
        print(f"  ring {r:5.2f} m: total score {tot:.0f}", flush=True)
    cam["centre_ring_m"] = max(rows)[1]
    args.camera.write_text(json.dumps(cam, indent=2))
    print(f"centre ring radius {cam['centre_ring_m']} m written to {args.camera}")


# ---------------------------------------------------------------- evaluation


def cmd_evaluate(args) -> None:
    """Owner-point error at the window's owner anchors, and the noise floor, from the automatic fixes alone."""
    from player_stats import noise_floor

    cam = json.loads(args.camera.read_text())
    anchors = json.loads((args.run / "pitch_anchors_ptz.local.json").read_text())["anchors"]
    cache = Cache(args.run / "cache")
    cal = Calibration(cache, anchors)
    owner = [a for a in json.loads(args.anchors.read_text())["anchors"] if not a.get("auto")]
    # Compare resolved paths: fit and evaluate may name the same file differently.
    this = args.anchors.resolve()
    fitted = {Path(p).resolve() for p in cam.get("fitted_from", [])}
    mirrored = this in {Path(p).resolve() for p in cam.get("mirrored_anchor_files", [])} or args.mirrored
    if this not in fitted and not args.mirrored:
        print(f"warning: {args.anchors} was not used by fit, so its goal side is unknown; "
              "pass --mirrored if it measures X from the other goal")
    rows = []
    for a in owner:
        img = np.array([p["img"] for p in a["points"]], float)
        truth = np.array([p["pitch"] for p in a["points"]], float)
        if mirrored:
            truth[:, 0] = cam["length_m"] - truth[:, 0]
        ci = int(round(a["time_s"] * cache.fps))
        xy, gap = cal.to_pitch(np.full(len(img), ci), img[:, 0], img[:, 1])
        err = np.hypot(*(xy - truth).T)
        rows.append(dict(owner_anchor_s=a["time_s"], median_err_m=round(float(np.median(err)), 2),
                         max_err_m=round(float(err.max()), 2), s_from_fix=round(float(gap[0]), 1)))  # fmt: skip
        print(rows[-1], flush=True)
    fps = float(json.loads((args.run / "best_config.json").read_text())["fps"])
    tr = pd.read_csv(args.run / "best_tracklets.csv.gz")
    xy, _gap = cal.to_pitch(tr.ci.to_numpy(), ((tr.x1 + tr.x2) / 2).to_numpy(), tr.y2.to_numpy())
    new = tr[["pf", "ci", "track_id"]].assign(X_m=xy[:, 0], Y_m=xy[:, 1])
    old = pd.read_csv(args.run / "tracklet_pitch_xy.csv.gz")
    out = dict(
        run=str(args.run),
        owner_anchors=rows,
        median_err_m=float(np.median([r["median_err_m"] for r in rows])) if rows else None,
        worst_err_m=float(np.max([r["median_err_m"] for r in rows])) if rows else None,
        noise_floor_ptz=noise_floor(args.run, new, fps),
        noise_floor_current=noise_floor(args.run, old, fps),
    )
    print(json.dumps({k: v for k, v in out.items() if k != "owner_anchors"}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit the game's fixed camera from owner anchors already placed in any windows")
    f.add_argument("--anchors", required=True, help="comma list of RUN or RUN:ANCHORS_JSON")
    f.add_argument("--out", type=Path, default=CAMERA_FILE)
    f.add_argument("--centre-guess", type=float, nargs=3, default=(50.0, 50.0, 6.0))
    f.add_argument("--length-guess", type=float, default=100.0)
    f.add_argument("--line-weight", type=float, default=0.3)
    f.add_argument("--max-nfev", type=int, default=60)
    f.set_defaults(fn=cmd_fit)
    r = sub.add_parser("run", help="find the pose every STEP_S seconds and write automatic anchors")
    r.add_argument("--run", required=True, type=require_under_data)
    r.add_argument("--camera", type=Path, default=CAMERA_FILE)
    r.add_argument("--min-spacing", type=float, default=STEP_S, help="seconds between written anchors")
    r.set_defaults(fn=cmd_run)
    g = sub.add_parser("ring", help="measure the centre ring radius on midfield frames of a window")
    g.add_argument("--run", required=True, type=require_under_data)
    g.add_argument("--times", required=True, help="comma list of seconds that show the centre of the pitch")
    g.add_argument("--camera", type=Path, default=CAMERA_FILE)
    g.add_argument("--min-r", type=float, default=5.0)
    g.add_argument("--max-r", type=float, default=11.0)
    g.set_defaults(fn=cmd_ring)
    e = sub.add_parser("evaluate", help="error at the window's owner anchors and the noise floor")
    e.add_argument("--run", required=True, type=require_under_data)
    e.add_argument("--anchors", required=True, type=Path, help="the owner's anchors for this window")
    e.add_argument("--camera", type=Path, default=CAMERA_FILE)
    e.add_argument("--mirrored", action="store_true", help="the owner's anchors measure X from the other goal")
    e.set_defaults(fn=cmd_evaluate)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
