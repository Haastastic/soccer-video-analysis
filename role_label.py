"""Hand-check team roles: label the true role for a sample of tracklets, then score team_classify.py.

`label` shows each sampled tracklet as up to 4 crops spread across its lifetime, with the predicted role,
confidence and flags overlaid. One key per tracklet:

  y   accept the predicted role (does nothing if the prediction is "unknown", since that is not a real role)
  t   target        o   opponent        f   official        g   goalkeeper        x   other (not a player)
  s   skip, unsure                b   back one tracklet, to undo                q   save and quit

  mouse wheel  zoom in or out on the crops, centered on the cursor    right click  pan    r  reset zoom

Progress is saved after every tracklet, so rerunning `label` resumes. Labels go to OUT/role_truth.csv
(git-ignored under data/). Crops show people: keep them local.

Sampling aims for --n tracklets (default 20), spread evenly across the predicted roles first (so rare
categories like goalkeeper and official are not missed), then filled from the rest at random.

`score` re-reads OUT/tracklet_roles.csv (so it reflects the current team_classify.py settings, not whatever
was current when you labeled) and compares it with the labels: confusion matrix, accuracy, and per-role
precision and recall.

Example:
  python role_label.py label --run data\\clipA --n 20
  python role_label.py score --run data\\clipA
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sv_common import ZoomView, cache_stride, read_frames, require_under_data, sample_rows
from team_classify import ROLES

WINDOW = "role label"
MIN_SAMPLE_ROWS = 10  # tracklets shorter than this rarely give a clear crop
CROP_H = 220
HEADER_H, FOOTER_H = 22, 22
TRUTH_COLS = ["track_id", "pred_role", "pred_confidence", "player_candidate", "n_rows", "truth_role", "verdict"]
KEY_ROLE = {ord("t"): "target", ord("o"): "opponent", ord("f"): "official", ord("g"): "goalkeeper", ord("x"): "other"}


def pick_items(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratified sample across predicted roles, then filled at random, so rare categories are not missed."""
    rng = np.random.default_rng(seed)
    buckets = sorted(pool.role.unique())
    quota = max(1, n // max(1, len(buckets)))
    picks = []
    for b in buckets:
        idx = pool.index[pool.role == b].to_numpy().copy()
        rng.shuffle(idx)
        picks.extend(idx[:quota].tolist())
    remaining = [i for i in pool.index if i not in picks]
    rng.shuffle(remaining)
    picks.extend(remaining[: max(0, n - len(picks))])
    picks = picks[:n]
    rng.shuffle(picks)
    return pool.loc[picks]


def build_items(run: Path, n: int, seed: int) -> list:
    roles = pd.read_csv(run / "tracklet_roles.csv", index_col="track_id")
    pool = roles[roles.n_rows >= MIN_SAMPLE_ROWS]
    if not len(pool):
        raise SystemExit("No tracklets with enough rows to label.")
    picked = pick_items(pool, min(n, len(pool)), seed)
    tr = pd.read_csv(run / "best_tracklets.csv.gz")
    tr = tr[tr.track_id.isin(picked.index)]
    samples = sample_rows(tr, n=4, min_h=0)
    crops_by_id = {tid: [] for tid in picked.index}
    for r in samples.itertuples():
        crops_by_id[r.track_id].append((int(r.ci), float(r.x1), float(r.y1), float(r.x2), float(r.y2)))
    items = []
    for tid, row in picked.iterrows():
        items.append(
            dict(
                track_id=int(tid),
                pred_role=row.role,
                pred_confidence=float(row.confidence),
                player_candidate=bool(row.player_candidate),
                sideline_suspect=bool(row.sideline_suspect),
                n_rows=int(row.n_rows),
                crops=sorted(crops_by_id[tid]),
            )
        )
    return items


class Session:
    """Labels for a list of items. Pure logic, no window, so it can be tested."""

    def __init__(self, items: list, saved: pd.DataFrame | None = None):
        self.items = items
        self.labels = {}
        if saved is not None and len(saved):
            by_id = {int(r.track_id): r for r in saved.itertuples()}
            for i, it in enumerate(items):
                r = by_id.get(it["track_id"])
                if r is not None:
                    self.labels[i] = dict(truth_role=r.truth_role, verdict=r.verdict)
        self.idx = next((i for i in range(len(items)) if i not in self.labels), len(items))

    @property
    def done(self) -> bool:
        return self.idx >= len(self.items)

    def _set(self, role, verdict) -> None:
        self.labels[self.idx] = dict(truth_role=role, verdict=verdict)
        self.idx += 1

    def accept(self) -> bool:
        role = self.items[self.idx]["pred_role"]
        if role not in ROLES:
            return False  # "unknown" is a punt, not a real role, so there is nothing to accept
        self._set(role, "accepted")
        return True

    def label(self, role: str) -> None:
        self._set(role, "labeled")

    def skip(self) -> None:
        self._set(None, "skipped")

    def back(self) -> None:
        if self.idx > 0:
            self.idx -= 1
            self.labels.pop(self.idx, None)

    def table(self) -> pd.DataFrame:
        keys = ("track_id", "pred_role", "pred_confidence", "player_candidate", "n_rows")
        rows = [{**{k: self.items[i][k] for k in keys}, **lab} for i, lab in sorted(self.labels.items())]
        return pd.DataFrame(rows, columns=TRUTH_COLS)


def render(img_lookup: dict, item: dict, index: int, total: int, label: dict | None, zv: ZoomView) -> np.ndarray:
    tiles = []
    for ci, x1, y1, x2, y2 in item["crops"]:
        img = img_lookup.get(ci)
        if img is None:
            continue
        pad = 8
        x1p, y1p = max(int(x1 - pad), 0), max(int(y1 - pad), 0)
        x2p, y2p = min(int(x2 + pad), img.shape[1]), min(int(y2 + pad), img.shape[0])
        crop = img[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            continue
        scale = CROP_H / crop.shape[0]
        tiles.append(cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), CROP_H), interpolation=cv2.INTER_CUBIC))
    if not tiles:
        tiles = [np.zeros((CROP_H, 120, 3), np.uint8)]
    gap = 6
    width = sum(t.shape[1] for t in tiles) + gap * (len(tiles) - 1)
    strip = np.zeros((CROP_H, width, 3), np.uint8)
    x = 0
    for t in tiles:
        strip[:, x : x + t.shape[1]] = t
        x += t.shape[1] + gap
    canvas = np.zeros((HEADER_H + CROP_H + FOOTER_H, strip.shape[1], 3), np.uint8)
    canvas[HEADER_H : HEADER_H + CROP_H] = zv.apply(strip)
    status = f"  [{label['verdict']}: {label['truth_role']}]" if label else ""
    flags = f"candidate={item['player_candidate']} sideline={item['sideline_suspect']} rows={item['n_rows']}"
    header = (
        f"{index + 1}/{total}  pred={item['pred_role']} ({item['pred_confidence']:.2f})  {flags}{status}{zv.label()}"
    )
    cv2.putText(canvas, header, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    keys = (
        "y accept | t target | o opponent | f official | g goalkeeper | x other | s skip | b back | q quit | wheel zoom"
    )
    cv2.putText(canvas, keys, (6, canvas.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def save_labels(truth_path: Path, items: list, session: "Session", carry_over: pd.DataFrame | None) -> pd.DataFrame:
    """Write this session's labels, plus any previously saved rows for track_ids outside this sample.

    A rerun with a different --n, --seed, or a re-tuned team_classify.py picks a different sample, so writing
    only session.table() would silently drop earlier hand labels for track_ids no longer sampled: expensive to
    redo since it is manual labeling of footage of minors.
    """
    table = session.table()
    if carry_over is not None and len(carry_over):
        item_ids = {it["track_id"] for it in items}
        carry = carry_over[~carry_over.track_id.isin(item_ids)]
        table = pd.concat([carry, table], ignore_index=True)[TRUTH_COLS]
    table.to_csv(truth_path, index=False)
    return table


def cmd_label(args) -> None:
    run = args.run
    truth_path = run / "role_truth.csv"
    items = build_items(run, args.n, args.seed)
    saved = pd.read_csv(truth_path) if truth_path.exists() else None
    session = Session(items, saved)
    stride = cache_stride(run)
    print(f"{len(items)} tracklets to label, {session.idx} already done. Loading frames...")
    frame_numbers = [ci * stride for it in items for ci, *_ in it["crops"]]
    jpgs = {}
    for clip_frame, img in read_frames(run / "clip.mp4", frame_numbers):
        jpgs[clip_frame // stride] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    zv = ZoomView()  # zooms the crop strip only; the header row sits above it, so shift mouse y to strip pixels
    cv2.setMouseCallback(WINDOW, lambda event, mx, my, flags, _p: zv.on_mouse(event, mx, my - HEADER_H, flags))
    try:
        while not session.done:
            item = session.items[session.idx]
            lookup = {ci: cv2.imdecode(jpgs[ci], cv2.IMREAD_COLOR) for ci, *_ in item["crops"] if ci in jpgs}
            cv2.imshow(WINDOW, render(lookup, item, session.idx, len(items), session.labels.get(session.idx), zv))
            key = cv2.waitKey(30) & 0xFF
            changed = False
            if key == ord("y"):
                changed = session.accept()
            elif key in KEY_ROLE:
                session.label(KEY_ROLE[key])
                changed = True
            elif key == ord("s"):
                session.skip()
                changed = True
            elif key == ord("b"):
                session.back()
                changed = True
            elif key == ord("r"):
                zv.reset()
            elif key == ord("q"):
                break
            if changed:
                zv.reset()  # strips differ in width per tracklet, so a zoom position does not carry over
                save_labels(truth_path, items, session, saved)
    finally:
        cv2.destroyAllWindows()
        save_labels(truth_path, items, session, saved)
    print(f"Saved {len(session.labels)} of {len(items)} labels to {truth_path}. Run `score` when you are done.")


def score_roles(run: Path, truth: pd.DataFrame) -> dict:
    """Compare hand labels with the current tracklet_roles.csv (not whatever was current when labeled)."""
    roles = pd.read_csv(run / "tracklet_roles.csv", index_col="track_id")
    t = truth[truth.verdict != "skipped"].drop(columns=["player_candidate"], errors="ignore").copy()
    t = t.join(roles[["role", "confidence", "player_candidate"]], on="track_id")
    if not len(t):
        return {"labeled": 0, "note": "No usable labels (all skipped, or none saved yet)."}
    correct = t.truth_role == t.role
    conf_mat = pd.crosstab(t.truth_role, t.role)
    confusion = {str(r): {str(c): int(v) for c, v in row.items()} for r, row in conf_mat.iterrows()}
    per_role = []
    for r in sorted(set(t.truth_role) | set(t.role)):
        tp = int(((t.truth_role == r) & (t.role == r)).sum())
        fp = int(((t.truth_role != r) & (t.role == r)).sum())
        fn = int(((t.truth_role == r) & (t.role != r)).sum())
        support = int((t.truth_role == r).sum())
        per_role.append(
            {
                "role": r,
                "support": support,
                "precision_pct": round(100 * tp / (tp + fp), 1) if (tp + fp) else None,
                "recall_pct": round(100 * tp / (tp + fn), 1) if (tp + fn) else None,
            }
        )
    return {
        "labeled": int(len(t)),
        "accuracy_pct": round(100 * float(correct.mean()), 1),
        "accuracy_player_candidates_pct": round(100 * float(correct[t.player_candidate].mean()), 1)
        if t.player_candidate.any()
        else None,
        "confusion_matrix_rows_truth_cols_predicted": confusion,
        "per_role": per_role,
        "mean_confidence_when_correct": round(float(t.confidence[correct].mean()), 3) if correct.any() else None,
        "mean_confidence_when_wrong": round(float(t.confidence[~correct].mean()), 3) if (~correct).any() else None,
        "note": "Hand-labeled sample, small. Scored against the current tracklet_roles.csv: rerunning "
        "team_classify.py with different settings changes this without relabeling.",
    }


def cmd_score(args) -> None:
    truth = pd.read_csv(args.run / "role_truth.csv")
    report = score_roles(args.run, truth)
    (args.run / "role_score.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lab = sub.add_parser("label", help="label the true role, one tracklet at a time")
    lab.add_argument("--run", required=True, type=require_under_data)
    lab.add_argument("--n", type=int, default=20, help="how many tracklets to sample")
    lab.add_argument("--seed", type=int, default=7)
    lab.set_defaults(fn=cmd_label)
    sc = sub.add_parser("score", help="compare tracklet_roles.csv with the labels")
    sc.add_argument("--run", required=True, type=require_under_data)
    sc.set_defaults(fn=cmd_score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
