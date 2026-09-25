"""Suggest jersey numbers for a new window from appearance learned on the owner's labeled windows (step 7 assist).

Why: jersey labeling is the largest manual cost per window (about 85 min for 83 players on clipE). An offline test
(CLAUDE.md Phase 7c) trained a classifier on frozen DINOv2 features of owner-labeled crops: trained on clipA, it
got 72% of clipE's tracklet parts right (top guess), 83% in the top 3, and a third of parts had a confident
suggestion that was right 94 to 95% of the time. It did NOT transfer from clipB (58 min away, other half: 18%), so
labeled windows are weighted by how close in time they are to the target (exp(-minutes apart / TAU_MIN)), and
windows should be labeled in order through the game.

`suggest` computes, for exactly the crops jersey_label.py will show (its own sampling), a probability per roster
jersey, then:
  - per tracklet: the mean log-probability over its crops, with the same-frame rule applied (tracklets on screen at
    the same time are different people): tracklets are assigned most confident first, each skipping jerseys already
    given to a tracklet that overlaps it in time. This lifted the top guess by about 10 points in the test.
  - per stitched player: the same over all its crops (meaningful when the player is one person).
jersey_label.py shows these as "suggest #N" with the confidence; `a` accepts after the owner checks the crops. A
suggestion is never applied on its own. Parts split during labeling get a suggestion from their own crops.

`evaluate` scores suggestions for a window that is already labeled (leave it out of --from), at the tracklet level.

Outputs (git-ignored, derived from footage of minors): RUN/jersey_suggest.npz (per-crop probabilities),
RUN/jersey_suggestions.csv, and per labeled window a feature cache RUN/jersey_features_<model>.npz.

Example:
  python jersey_suggest.py suggest --run data\\clipF --from data\\clipA,data\\clipB,data\\clipE
  python jersey_suggest.py evaluate --run data\\clipE --from data\\clipA,data\\clipB
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import jersey_label as jl
from player_stats import identity_rows
from sv_common import cache_stride, parse_time, read_frames, require_under_data, tracklet_fingerprint

MODEL = "dinov2_vits14"
SIZE = (224, 112)  # H, W, multiples of 14
PER_PART = 16  # training crops per labeled tracklet part
TAU_MIN = 10.0  # time weighting of labeled windows, in minutes
C = 8.0  # logistic regression; chosen on clipA<->clipB only, before looking at clipE
_model = None


def clip_mid_min(run: Path) -> float:
    meta = json.loads((run / "cache" / "meta.json").read_text())
    return (parse_time(meta["clip_start"]) + float(meta["clip_duration"]) / 2) / 60


def box_crops(run: Path, rows: pd.DataFrame) -> list:
    """Person box crops (slightly padded) for rows with ci, x1..y2, in row order."""
    stride = cache_stride(run)
    by_frame = {}
    for k, ci in enumerate(rows.ci.to_numpy()):
        by_frame.setdefault(int(ci) * stride, []).append(k)
    boxes = rows[["x1", "y1", "x2", "y2"]].to_numpy(float)
    out = [None] * len(rows)
    for f, img in read_frames(run / "clip.mp4", list(by_frame)):
        for k in by_frame[f]:
            x1, y1, x2, y2 = boxes[k]
            w, h = x2 - x1, y2 - y1
            a, b = int(max(0, x1 - 0.1 * w)), int(max(0, y1 - 0.05 * h))
            c, d = int(min(img.shape[1], x2 + 0.1 * w)), int(min(img.shape[0], y2 + 0.05 * h))
            out[k] = img[b:d, a:c].copy() if d > b and c > a else np.zeros((SIZE[0], SIZE[1], 3), np.uint8)
    return [o if o is not None else np.zeros((SIZE[0], SIZE[1], 3), np.uint8) for o in out]


def features(imgs: list) -> np.ndarray:
    """L2-normalized DINOv2 ViT-S/14 embeddings (GPU if available)."""
    import torch

    global _model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if _model is None:
        _model = torch.hub.load("facebookresearch/dinov2", MODEL, verbose=False).to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), 64):
            batch = [
                cv2.resize(im[..., ::-1], (SIZE[1], SIZE[0]), interpolation=cv2.INTER_CUBIC) for im in imgs[i : i + 64]
            ]
            x = torch.from_numpy(np.stack(batch)).to(dev).permute(0, 3, 1, 2).float() / 255
            f = torch.nn.functional.normalize(_model((x - mean) / std).float(), dim=1)
            out.append(f.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 384), np.float32)


def labeled_samples(run: Path) -> pd.DataFrame:
    """Up to PER_PART crops per identified tracklet part of a labeled window, with jersey and box."""
    xy = pd.read_csv(run / "tracklet_pitch_xy.csv.gz", usecols=["pf", "ci", "track_id", "X_m", "Y_m", "anchor_gap_s"])
    ident = identity_rows(run, xy)[["ci", "track_id", "part", "jersey"]]
    boxes = pd.read_csv(run / "best_tracklets.csv.gz", usecols=["ci", "track_id", "x1", "y1", "x2", "y2"])
    d = ident.merge(boxes, on=["ci", "track_id"]).sort_values("ci")
    parts = [
        g.iloc[np.linspace(0, len(g) - 1, min(PER_PART, len(g))).round().astype(int)]
        for _, g in d.groupby(["track_id", "part"])
    ]
    return pd.concat(parts, ignore_index=True)


def training_set(run: Path) -> tuple:
    """(features, jerseys) of a labeled window. Features are cached, tied to exactly which crops were sampled; the
    jerseys are always read fresh from the current labels, so a corrected label (e.g. two swapped) takes effect."""
    rows = labeled_samples(run)
    y = rows.jersey.to_numpy(int)
    stamp = tracklet_fingerprint(rows, salt=f"{MODEL}:{PER_PART}")
    path = run / f"jersey_features_{MODEL}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z["stamp"]) == stamp:
            return z["F"], y
    print(f"{run.name}: embedding {len(rows)} labeled crops...", flush=True)
    F = features(box_crops(run, rows))
    np.savez_compressed(path, F=F, stamp=stamp)
    return F, y


def train(target: Path, sources: list):
    from sklearn.linear_model import LogisticRegression

    Fs, ys, ws = [], [], []
    t0 = clip_mid_min(target)
    for src in sources:
        F, y = training_set(src)
        w = np.exp(-abs(clip_mid_min(src) - t0) / TAU_MIN)
        Fs.append(F)
        ys.append(y)
        ws.append(np.full(len(y), w))
        print(f"  {src.name}: {len(y)} crops, {len(set(y))} players, weight {w:.2f}", flush=True)
    w = np.concatenate(ws)
    w = w / w.max()  # the nearest window counts fully, so C keeps the meaning it was chosen with
    clf = LogisticRegression(max_iter=4000, C=C)
    clf.fit(np.concatenate(Fs), np.concatenate(ys), sample_weight=w)
    return clf


def aggregate(P: np.ndarray) -> np.ndarray:
    """Probabilities for one unit from its crops: normalized mean log-probability."""
    lp = np.log(P + 1e-9).mean(0)
    p = np.exp(lp - lp.max())
    return p / p.sum()


def joint_assign(units: list, classes: np.ndarray) -> None:
    """Most confident first; skip jerseys already given to a unit overlapping in time (same-frame rule)."""
    for u in units:
        u["conf0"] = float(u["p"].max())
    taken = []
    for u in sorted(units, key=lambda u: -u["conf0"]):
        banned = {v["jersey"] for v in taken if not (v["hi"] < u["lo"] or v["lo"] > u["hi"])}
        order = [k for k in np.argsort(-u["p"]) if classes[k] not in banned]
        u["jersey"], u["conf"] = int(classes[order[0]]), float(u["p"][order[0]])
        u["alts"] = [int(classes[k]) for k in order[1:3]]
        taken.append(u)


def cmd_suggest(args) -> None:
    sources = [require_under_data(Path(r)) for r in args.sources.split(",")]
    run = args.run
    print(f"Training on {len(sources)} labeled windows for {run.name}:")
    clf = train(run, sources)
    items = jl.build_items(run)
    rows = pd.DataFrame([dict(player_id=it["player_id"], **r) for it in items for r in it["rows"]])
    print(f"{run.name}: embedding {len(rows)} crops shown by jersey_label.py...", flush=True)
    P = clf.predict_proba(features(box_crops(run, rows)))
    np.savez_compressed(
        run / "jersey_suggest.npz",
        track_id=rows.track_id.to_numpy(int),
        ci=rows.ci.to_numpy(int),
        P=P.astype(np.float32),
        classes=clf.classes_.astype(int),
    )
    units = []
    for it in items:
        for tid in it["tracklets"]:
            k = (rows.player_id == it["player_id"]).to_numpy() & (rows.track_id == tid).to_numpy()
            lo, hi = it["span"][tid]
            units.append(
                dict(level="tracklet", player_id=it["player_id"], track_id=tid, lo=lo, hi=hi, p=aggregate(P[k]))
            )
    joint_assign(units, clf.classes_)
    players = []
    for it in items:
        k = (rows.player_id == it["player_id"]).to_numpy()
        p = aggregate(P[k])
        order = np.argsort(-p)
        players.append(
            dict(level="player", player_id=it["player_id"], track_id=-1, jersey=int(clf.classes_[order[0]]),
                 conf=float(p[order[0]]), alts=[int(clf.classes_[j]) for j in order[1:3]])
        )  # fmt: skip
    out = pd.DataFrame(units + players)[["level", "player_id", "track_id", "jersey", "conf", "alts"]]
    out["alts"] = out.alts.map(lambda a: " ".join(str(x) for x in a))
    out.to_csv(run / "jersey_suggestions.csv", index=False)
    t = out[out.level == "tracklet"]
    print(
        f"Wrote {run / 'jersey_suggestions.csv'}: {len(t)} tracklets, "
        f"{int((t.conf >= 0.5).sum())} with confidence >= 0.5"
    )


def cmd_evaluate(args) -> None:
    """Score suggestions on an already-labeled window (must not be among --from), per whole tracklet."""
    run = args.run
    sources = [require_under_data(Path(r)) for r in args.sources.split(",")]
    if run.resolve() in [s.resolve() for s in sources]:
        raise SystemExit("Leave the evaluated window out of --from.")
    args_s = argparse.Namespace(run=run, sources=args.sources)
    cmd_suggest(args_s)
    sug = pd.read_csv(run / "jersey_suggestions.csv")
    pi = pd.read_csv(run / "player_identity.csv")
    truth = pi[pi.jersey.notna() & ~pi.split_at_switch.fillna(False).astype(bool)][["track_id", "jersey"]]
    t = sug[sug.level == "tracklet"].merge(truth, on="track_id", suffixes=("", "_truth"))
    t["ok"] = t.jersey == t.jersey_truth
    t["ok3"] = [
        j in [a, *map(int, str(b).split())] for j, a, b in zip(t.jersey_truth, t.jersey, t.alts.fillna(""), strict=True)
    ]
    rep = dict(tracklets=len(t), top1_pct=round(100 * t.ok.mean(), 1), top3_pct=round(100 * t.ok3.mean(), 1))
    for thr in (0.3, 0.5, 0.7):
        m = t.conf >= thr
        rep[f"conf_{thr}"] = dict(
            cover_pct=round(100 * m.mean(), 1), right_pct=round(100 * t.ok[m].mean(), 1) if m.any() else None
        )
    print(json.dumps(rep, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, hlp in (
        ("suggest", cmd_suggest, "write suggestions for a window"),
        ("evaluate", cmd_evaluate, "score on a labeled window"),
    ):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--run", required=True, type=require_under_data)
        p.add_argument("--from", dest="sources", required=True, help="comma list of labeled run folders")
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
