"""Classify tracklets into roles (target, opponent, official, goalkeeper, other) from cached kit colors.

Works on best_tracklets.csv.gz from replay_trackers.py. No video is needed to classify, only to make the
review montage in `calibrate`. Roles come from numeric color prototypes stored in a git-ignored local file
(kit_prototypes.local.json). Prototypes are only initial centers: `classify` refines them on each clip, so
lighting differences between clips are absorbed, and reports how far each center moved.

Color alone cannot tell players from people at the sideline (bench, coaches) who wear the same kit, so
each tracklet also gets motion features, a sideline_suspect flag and pitch_mask.py's on-pitch test, which
combine into player_candidate.

Workflow:
  python team_classify.py calibrate --run data\\clipA               # clusters + review montage (local only)
  python team_classify.py assign --run data\\clipA --map 0:other,2:target,5:opponent,4:official
  python team_classify.py classify --run data\\clipB --refine         # any clip, uses the local prototypes
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pitch_mask import on_pitch_table
from sv_common import Cache, cache_stride, cached_per_tracklet, read_frames, sample_rows

ROLES = ["target", "opponent", "official", "goalkeeper", "other"]
COLOR_COLS = ["torso_r", "torso_g", "torso_b", "legs_r", "legs_g", "legs_b"]
PROTO_FILE = Path(__file__).resolve().parent / "kit_prototypes.local.json"
MIN_ROWS = 30  # tracklets shorter than this give unreliable colors
SAMPLES = 8  # frames re-measured per tracklet
LAB_COLS = ["torso_L", "torso_a", "torso_b", "legs_L", "legs_a", "legs_b"]
DESCRIPTOR = "pixels-v1"  # prototypes are only valid for the descriptor they were made with


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """Nx3 RGB in 0..255 to Nx3 CIE Lab."""
    return cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.float32) / 255, cv2.COLOR_RGB2Lab).reshape(-1, 3)


def kmeans(x, k, w, seed=0, iters=60):
    rng = np.random.default_rng(seed)
    c = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d = np.min([((x - ci) ** 2).sum(1) for ci in c], axis=0) * w
        c.append(x[rng.choice(len(x), p=d / d.sum())])
    c = np.array(c)
    for _ in range(iters):
        lab = np.argmin(((x[:, None] - c[None]) ** 2).sum(2), axis=1)
        for j in range(k):
            if (lab == j).any():
                c[j] = np.average(x[lab == j], axis=0, weights=w[lab == j])
    return lab, c


def tracklet_features(run: Path) -> pd.DataFrame:
    """One row per tracklet: median kit colors in Lab, duration, and how much it moved."""
    tr = pd.read_csv(run / "best_tracklets.csv.gz")
    cfg = json.loads((run / "best_config.json").read_text())
    fps = float(cfg["fps"])
    cache = Cache(run / "cache")
    tr["cx"], tr["h"] = (tr.x1 + tr.x2) / 2, (tr.y2 - tr.y1).clip(lower=1)
    tr["sx"], tr["sy"] = cache.to_stable(tr.ci.to_numpy(), tr.cx.to_numpy(), ((tr.y1 + tr.y2) / 2).to_numpy())
    g = tr.groupby("track_id")
    med = g[COLOR_COLS].median()
    f = pd.DataFrame({"n_rows": g.size(), "first_pf": g.pf.min(), "last_pf": g.pf.max(), "med_h": g.h.median()})
    f["dur_s"] = (f.last_pf - f.first_pf + 1) / fps
    # Spread of the stable-coordinate path in body heights: near 0 means sitting or standing still.
    rad = g.apply(
        lambda d: np.percentile(np.hypot(d.sx - d.sx.median(), d.sy - d.sy.median()), 90), include_groups=False
    )
    f["extent_h"] = rad / f.med_h
    if (run / "clip.mp4").exists():
        f = f.join(pixel_colors(run, tr))
    else:  # fall back to the cached patch medians (grass-contaminated, so not valid for pixels-v1 prototypes)
        ok = med.notna().all(axis=1)
        lab = np.full((len(med), 6), np.nan)
        lab[ok.to_numpy()] = np.hstack(
            [to_lab(med.loc[ok, COLOR_COLS[:3]].to_numpy()), to_lab(med.loc[ok, COLOR_COLS[3:]].to_numpy())]
        )
        for i, name in enumerate(LAB_COLS):
            f[name] = lab[:, i]
    f["sideline_suspect"] = (f.dur_s >= 8) & (f.extent_h < 0.75)
    return f


def patch_lab(img, x1, y1, x2, y2):
    """Median Lab of the non-grass pixels in a patch, or NaNs if too few remain.

    The cached colors are medians over the whole patch, which on players 35 to 60 px tall is mostly grass
    and skin. Masking bright saturated green (dark green kit stays) removes most of that contamination.
    """
    x1, y1 = max(int(x1), 0), max(int(y1), 0)
    x2, y2 = min(int(x2), img.shape[1]), min(int(y2), img.shape[0])
    if x2 - x1 < 3 or y2 - y1 < 3:
        return np.full(3, np.nan)
    crop = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    grass = (hsv[:, 0] >= 30) & (hsv[:, 0] <= 85) & (hsv[:, 1] > 70) & (hsv[:, 2] > 80)
    keep = crop.reshape(-1, 3)[~grass]
    if len(keep) < 12:
        return np.full(3, np.nan)
    lab = cv2.cvtColor(keep.reshape(-1, 1, 3).astype(np.float32) / 255, cv2.COLOR_BGR2Lab).reshape(-1, 3)
    return np.median(lab, axis=0)


def pixel_colors(run: Path, tr: pd.DataFrame) -> pd.DataFrame:
    """Re-measure kit colors from SAMPLES frames per tracklet. Cached in tracklet_colors.csv."""
    return cached_per_tracklet(run, "tracklet_colors", tr, f"{DESCRIPTOR}:{SAMPLES}", lambda: _measure_colors(run, tr))


def _measure_colors(run: Path, tr: pd.DataFrame) -> pd.DataFrame:
    stride = cache_stride(run)
    samples = sample_rows(tr, SAMPLES)
    by_frame = {}
    for r in samples.itertuples():
        by_frame.setdefault(int(r.ci), []).append(r)
    vals = {}
    for ci_frame, img in read_frames(run / "clip.mp4", [ci * stride for ci in by_frame]):
        for r in by_frame[ci_frame // stride]:
            w, h = r.x2 - r.x1, r.y2 - r.y1
            tx1, tx2 = r.x1 + 0.25 * w, r.x2 - 0.25 * w
            vals.setdefault(r.track_id, []).append(
                np.concatenate(
                    [
                        patch_lab(img, tx1, r.y1 + 0.20 * h, tx2, r.y1 + 0.50 * h),
                        patch_lab(img, tx1, r.y1 + 0.55 * h, tx2, r.y1 + 0.85 * h),
                    ]
                )
            )
    out = {
        tid: np.nanmedian(np.array(v), axis=0) if np.isfinite(v).any() else np.full(6, np.nan)
        for tid, v in vals.items()
    }
    return pd.DataFrame.from_dict(out, orient="index", columns=LAB_COLS)


def cmd_calibrate(args) -> None:
    run = args.run
    f = tracklet_features(run)
    use = f[(f.n_rows >= MIN_ROWS) & f[LAB_COLS].notna().all(axis=1)]
    x, w = use[LAB_COLS].to_numpy(), use.n_rows.to_numpy().astype(float)
    best = None
    for s in range(8):
        lab, c = kmeans(x, args.k, w, seed=s)
        sse = (((x - c[lab]) ** 2).sum(1) * w).sum()
        if best is None or sse < best[0]:
            best = (sse, lab, c)
    _, lab, c = best
    out = {"k": args.k, "centers": c.round(2).tolist(), "track_ids": use.index.tolist(), "labels": lab.tolist()}
    (run / "kit_clusters.json").write_text(json.dumps(out))
    print(f"{len(use)} tracklets of {len(f)} used ({100 * use.n_rows.sum() / f.n_rows.sum():.0f}% of rows)")
    for j in range(args.k):
        m = lab == j
        print(
            f"  cluster {j}: {m.sum():3d} tracklets, {100 * w[m].sum() / w.sum():4.1f}% rows, "
            f"sideline_suspect {100 * use.sideline_suspect.to_numpy()[m].mean():3.0f}%"
        )
    groups = {f"c{j}": use.index.to_numpy()[lab == j] for j in range(args.k)}
    if write_montage(run, groups, run / "kit_clusters.png"):
        print(f"Wrote {run / 'kit_clusters.json'} and the review montage {run / 'kit_clusters.png'} (local only)")


def write_montage(run: Path, groups: dict, dest: Path, per_row: int = 8) -> bool:
    """One row of sample person crops per group. Local review only: it shows people, never commit or share."""
    clip = run / "clip.mp4"
    if not clip.exists():
        print("No clip.mp4 in the run folder, skipping the montage.")
        return False
    tr = pd.read_csv(run / "best_tracklets.csv.gz")
    stride = cache_stride(run)
    cap = cv2.VideoCapture(str(clip))
    rng = np.random.default_rng(3)
    rows = []
    for name, members in groups.items():
        tiles = []
        for tid in rng.choice(members, min(per_row, len(members)), replace=False) if len(members) else []:
            d = tr[tr.track_id == tid]
            r = d.iloc[len(d) // 2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.ci) * stride)
            ok, img = cap.read()
            if not ok:
                continue
            x1, y1, x2, y2 = (int(v) for v in (r.x1, r.y1, r.x2, r.y2))
            crop = img[max(y1 - 6, 0) : y2 + 6, max(x1 - 6, 0) : x2 + 6]
            tiles.append(cv2.resize(crop, (64, 128), interpolation=cv2.INTER_CUBIC))
        tiles += [np.zeros((128, 64, 3), np.uint8)] * (per_row - len(tiles))
        row = np.hstack(tiles)
        cv2.putText(row, name, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        rows.append(row)
    cv2.imwrite(str(dest), np.vstack(rows))
    return True


def cmd_assign(args) -> None:
    clusters = json.loads((args.run / "kit_clusters.json").read_text())
    protos = json.loads(PROTO_FILE.read_text()) if PROTO_FILE.exists() else {"roles": {r: [] for r in ROLES}}
    for pair in args.map.split(","):
        j, role = pair.split(":")
        if role not in ROLES:
            raise SystemExit(f"Unknown role {role}. Roles: {ROLES}")
        protos["roles"][role].append(clusters["centers"][int(j)])
    protos["unknown_dist"] = args.unknown_dist
    protos["descriptor"] = DESCRIPTOR
    PROTO_FILE.write_text(json.dumps(protos, indent=1))
    print(f"Wrote {PROTO_FILE.name} (git-ignored): " + ", ".join(f"{r}={len(c)}" for r, c in protos["roles"].items()))


def assign_roles(x, centers, roles, unknown_dist):
    """Nearest role by prototype distance, plus a confidence from the margin to the runner-up role."""
    d = np.sqrt(((x[:, None] - centers[None]) ** 2).sum(2))
    names = sorted(set(roles))
    per_role = np.stack([d[:, [i for i, r in enumerate(roles) if r == n]].min(axis=1) for n in names], axis=1)
    order = np.argsort(per_role, axis=1)
    d1 = per_role[np.arange(len(x)), order[:, 0]]
    d2 = per_role[np.arange(len(x)), order[:, 1]] if len(names) > 1 else np.full(len(x), np.inf)
    role = np.array(names)[order[:, 0]]
    conf = np.clip(1 - d1 / np.maximum(d2, 1e-6), 0, 1)
    far = d1 > unknown_dist
    role = np.where(far, "unknown", role)
    conf = np.where(far, 0.0, conf)
    return role, conf, d1, d2


def cmd_classify(args) -> None:
    run = args.run
    if not PROTO_FILE.exists():
        raise SystemExit("No kit_prototypes.local.json. Run calibrate and assign on a reference clip first.")
    protos = json.loads(PROTO_FILE.read_text())
    if protos.get("descriptor") != DESCRIPTOR:
        raise SystemExit(
            f"Prototypes were made with a different color descriptor. Rerun calibrate and assign ({DESCRIPTOR})."
        )
    if not (run / "clip.mp4").exists():
        raise SystemExit("classify needs clip.mp4 in the run folder to measure pixel colors.")
    unknown_dist = float(protos.get("unknown_dist", 40))
    centers = np.array([c for r in ROLES for c in protos["roles"][r]])
    roles = [r for r in ROLES for _ in protos["roles"][r]]
    f = tracklet_features(run)
    use = f[(f.n_rows >= MIN_ROWS) & f[LAB_COLS].notna().all(axis=1)]
    x, w = use[LAB_COLS].to_numpy(), use.n_rows.to_numpy().astype(float)
    shifts = {}
    if args.refine:
        start = centers.copy()
        for _ in range(10):
            role, _, d1, _ = assign_roles(x, centers, roles, unknown_dist)
            d = np.sqrt(((x[:, None] - centers[None]) ** 2).sum(2))
            nearest = np.argmin(d, axis=1)
            for i in range(len(centers)):
                m = (nearest == i) & (d1 <= unknown_dist)
                if m.sum() >= 3:
                    centers[i] = np.average(x[m], axis=0, weights=w[m])
        for i, r in enumerate(roles):
            shifts.setdefault(r, []).append(round(float(np.linalg.norm(centers[i] - start[i])), 1))
    role, conf, d1, d2 = assign_roles(x, centers, roles, unknown_dist)
    res = use[["n_rows", "dur_s", "extent_h", "sideline_suspect"]].copy()
    res["role"], res["confidence"], res["dist_best"], res["dist_second"] = (
        role,
        conf.round(3),
        d1.round(1),
        np.round(d2, 1),
    )
    short = f.index.difference(use.index)
    res = pd.concat(
        [
            res,
            pd.DataFrame({"role": "unknown", "confidence": 0.0}, index=short).join(
                f[["n_rows", "dur_s", "extent_h", "sideline_suspect"]]
            ),
        ]
    )
    res.index.name = "track_id"
    res = res.join(on_pitch_table(run)[["foot_grass", "on_pitch"]])
    res["on_pitch"] = res.on_pitch.fillna(False).astype(bool)
    # A player candidate stands on grass and does not sit still for 8 s or more. Sideline people fail one or both.
    res["player_candidate"] = res.on_pitch & ~res.sideline_suspect.astype(bool)
    res.sort_index().to_csv(run / "tracklet_roles.csv")

    tr = pd.read_csv(run / "best_tracklets.csv.gz", usecols=["pf", "track_id"])
    tr = tr.join(res[["role", "player_candidate"]], on="track_id")
    moving = tr[tr.player_candidate]
    per_frame = moving.assign(n=1).pivot_table(index="pf", columns="role", values="n", aggfunc="sum", fill_value=0)
    report = {
        "tracklets": int(len(res)),
        "rows_share_by_role_pct": (100 * tr.role.value_counts(normalize=True)).round(1).to_dict(),
        "tracklets_by_role": res.role.value_counts().to_dict(),
        "player_candidates_by_role": res[res.player_candidate].role.value_counts().to_dict(),
        "median_per_frame_player_candidates": per_frame.median().round(1).to_dict(),
        "off_pitch_pct": round(100 * float((~res.on_pitch).mean()), 1),
        "median_confidence": float(res.confidence[res.role != "unknown"].median()),
        "low_confidence_pct": round(100 * float((res.confidence[res.role != "unknown"] < 0.3).mean()), 1),
        "sideline_suspect_pct_by_role": (100 * res.groupby("role").sideline_suspect.mean()).round(0).to_dict(),
        "refined": bool(args.refine),
        "prototype_shift_lab": shifts,
        "note": "Proxy report. Roles are unverified against labels. Sideline people share kit colors with players.",
    }
    (run / "team_report.json").write_text(json.dumps(report, indent=2))
    if args.montage:
        ok = res.n_rows >= MIN_ROWS
        groups = {
            f"{r}{'' if c else ' EXCLUDED'}": res.index[(res.role == r) & ok & (res.player_candidate == c)].to_numpy()
            for r in [*ROLES, "unknown"]
            for c in (True, False)
        }
        write_montage(run, {k: v for k, v in groups.items() if len(v)}, run / "roles_montage.png")
    print(json.dumps(report, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="cluster tracklet colors and write a review montage")
    c.add_argument("--run", required=True, type=Path)
    c.add_argument("--k", type=int, default=6)
    c.set_defaults(fn=cmd_calibrate)
    a = sub.add_parser("assign", help="turn cluster numbers into role prototypes in the local file")
    a.add_argument("--run", required=True, type=Path)
    a.add_argument("--map", required=True, help="e.g. 2:target,5:opponent,4:official")
    a.add_argument("--unknown-dist", type=float, default=40, help="Lab distance beyond which a tracklet is unknown")
    a.set_defaults(fn=cmd_assign)
    k = sub.add_parser("classify", help="assign a role and confidence to every tracklet")
    k.add_argument("--run", required=True, type=Path)
    k.add_argument("--refine", action="store_true", help="adapt prototypes to this clip before assigning")
    k.add_argument("--montage", action="store_true", help="also write roles_montage.png, local review only")
    k.set_defaults(fn=cmd_classify)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
