"""Fine-tune a ball detector on the owner's ball labels (ball_label.py) for this venue.

Why: on game 2 the COCO detector misses the ball on about half the visible frames of some windows and follows a
wrong object on others (CLAUDE.md Phase 12 and 14). The owner's clicks (ball_truth.csv) are the training data.

A separate one-class model: person detections in the cache are untouched; `detect` writes ball candidates to
RUN/cache/ball_ft.csv.gz in the cache's format, and ball_link.py --ball-cache uses them instead of the cache's
COCO "sports ball" rows.

  dataset   640 px tiles cut from full frames at native scale (the ball stays about 10 px, as at inference on the
            full 1920 px frame): TILES_POS tiles around each visible labeled ball at random offsets, one random
            tile away from it, and a tile on every confident COCO candidate further than NEG_DIST_PX from the ball
            (wrong objects: spare balls, markers, heads). A ball box takes the size of a COCO candidate on it,
            else BALL_BOX_PX. Frames of --val runs go to the validation split.
  train     yolo11m (COCO weights) to one class at 640, light augmentation (no rotation; small scale changes).
  detect    the model on every frame ball_link.py uses (10 fps) at 1920, conf floor 0.05.
  apply     detect with models/ball/venue.pt, then ball_link.py --ball-cache: the pipeline stage (run_windows.py).
  evaluate  on a run's labeled frames: the top candidate within TOL_PX of the owner's click, per confidence
            cut, for the fine-tuned model and the cache's COCO candidates.

Models and datasets hold footage of minors: they stay under models/ and data/ (git-ignored).

Example:
  python ball_finetune.py dataset --runs data\\clipA,data\\clipB,data\\g0922\\w0500 --val data\\g0922\\w2000
      --out data\\_ball_ft\\foldA
  python ball_finetune.py train --data data\\_ball_ft\\foldA --name foldA
  python ball_finetune.py detect --run data\\g0922\\w4840 --model models\\ball\\foldA.pt
  python ball_finetune.py evaluate --run data\\g0922\\w4840
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sv_common import BALL, Cache, cache_stride, read_frames, require_under_data

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models" / "ball"
TILE = 640
TILES_POS = 2  # tiles per visible ball
NEG_DIST_PX = 30  # a COCO candidate this far from the owner's ball is a wrong object
NEG_MIN_CONF = 0.15  # wrong objects at least this confident become hard-negative tiles
MAX_NEG = 3  # hard-negative tiles per frame at most
BALL_BOX_PX = 12  # box side when no COCO candidate sits on the ball (median COCO ball box: 10 px)
TOL_PX = 20  # as ball_label.py score
LINK_FPS = 10  # ball_link.py's processing rate
FT_FILE = "ball_ft.csv.gz"
FT_META = "ball_ft_meta.json"  # written last by `apply`: marks a window's ball path as made from the venue model
VENUE_MODEL = MODELS / "venue.pt"  # the production model; run_windows.py uses it when the file exists


def frame_boxes(run: Path) -> pd.DataFrame:
    """Labeled frames of a run with the ball box (visible) and the COCO candidates that are wrong objects."""
    t = pd.read_csv(run / "ball_truth.csv")
    t = t[t.verdict != "skipped"].copy()
    det = pd.read_csv(run / "cache" / "detections.csv.gz", usecols=["ci", "cls", "conf", "x1", "y1", "x2", "y2"])
    det = det[(det.cls == BALL) & det.ci.isin(t.ci)]
    rows = []
    for r in t.itertuples():
        d = det[det.ci == r.ci]
        cx, cy = (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2
        box, wrong = None, []
        if r.visible == 1:
            dist = np.hypot(cx - r.truth_x, cy - r.truth_y)
            on = d[dist <= TOL_PX / 2]
            w = h = BALL_BOX_PX
            if len(on):
                best = on.loc[on.conf.idxmax()]
                w, h = max(best.x2 - best.x1, 6), max(best.y2 - best.y1, 6)
            box = (r.truth_x - w / 2, r.truth_y - h / 2, r.truth_x + w / 2, r.truth_y + h / 2)
            far = d[(dist > NEG_DIST_PX) & (d.conf >= NEG_MIN_CONF)]
        else:
            far = d[d.conf >= NEG_MIN_CONF]
        wrong = list(zip((far.x1 + far.x2) / 2, (far.y1 + far.y2) / 2, strict=True))
        rows.append(dict(ci=int(r.ci), box=box, wrong=wrong))
    return pd.DataFrame(rows)


def tile_origin(rng, cx: float, cy: float, w: int, h: int) -> tuple:
    """Top-left of a TILE x TILE tile containing (cx, cy) at a random offset, inside the frame."""
    x0 = int(np.clip(cx - rng.uniform(0.15, 0.85) * TILE, 0, w - TILE))
    y0 = int(np.clip(cy - rng.uniform(0.15, 0.85) * TILE, 0, h - TILE))
    return x0, y0


def write_tile(img, x0: int, y0: int, box, out_img: Path, out_lbl: Path) -> None:
    cv2.imwrite(str(out_img), img[y0 : y0 + TILE, x0 : x0 + TILE], [cv2.IMWRITE_JPEG_QUALITY, 95])
    lines = []
    if box is not None:
        x1, y1, x2, y2 = box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0
        if 0 <= (x1 + x2) / 2 < TILE and 0 <= (y1 + y2) / 2 < TILE:  # centre inside: clip the box to the tile
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, TILE), min(y2, TILE)
            cx, cy, bw, bh = (x1 + x2) / 2 / TILE, (y1 + y2) / 2 / TILE, (x2 - x1) / TILE, (y2 - y1) / TILE
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""))


def cmd_dataset(args) -> None:
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    rng = np.random.default_rng(0)
    counts = {"train": [0, 0], "val": [0, 0]}  # tiles with a ball, tiles without
    runs = [(r, "train") for r in args.runs] + [(r, "val") for r in args.val]
    for run, split in runs:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        fb = frame_boxes(run).set_index("ci")
        stride = cache_stride(run)
        tag = f"{run.parent.name}_{run.name}"
        for clip_frame, img in read_frames(run / "clip.mp4", [ci * stride for ci in fb.index]):
            ci = clip_frame // stride
            r = fb.loc[ci]
            h, w = img.shape[:2]
            tiles = []
            if r.box is not None:
                bx, by = (r.box[0] + r.box[2]) / 2, (r.box[1] + r.box[3]) / 2
                tiles += [tile_origin(rng, bx, by, w, h) for _ in range(TILES_POS)]
            for _ in range(20):  # one random tile away from the ball
                x0, y0 = int(rng.integers(0, w - TILE + 1)), int(rng.integers(0, h - TILE + 1))
                if r.box is None or not (x0 - 20 <= r.box[0] and r.box[2] <= x0 + TILE + 20 and y0 - 20 <= r.box[1]
                                         and r.box[3] <= y0 + TILE + 20):  # fmt: skip
                    tiles.append((x0, y0))
                    break
            tiles += [
                tile_origin(rng, x, y, w, h) for x, y in r.wrong[:MAX_NEG]
            ]  # the ball is labeled if it falls inside
            for k, (x0, y0) in enumerate(tiles):
                name = f"{tag}_{ci:05d}_{k}"
                write_tile(
                    img, x0, y0, r.box, out / "images" / split / f"{name}.jpg", out / "labels" / split / f"{name}.txt"
                )
                has = (out / "labels" / split / f"{name}.txt").read_text().strip() != ""
                counts[split][0 if has else 1] += 1
        print(f"{run}: {len(fb)} labeled frames -> {split}", flush=True)
    yaml = f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n"
    (out / "data.yaml").write_text(yaml)
    print({k: {"with_ball": v[0], "without": v[1]} for k, v in counts.items()})


def cmd_train(args) -> None:
    from ultralytics import YOLO

    MODELS.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.base)
    model.train(
        data=str(args.data / "data.yaml"),
        imgsz=TILE,
        epochs=args.epochs,
        batch=args.batch,
        project=str((args.data / "runs").resolve()),  # a relative project lands under runs/detect/ in the repo
        name=args.name,
        exist_ok=True,
        degrees=0.0,
        scale=0.2,
        mosaic=0.5,
        mixup=0.0,
        fliplr=0.5,
        patience=15,
        workers=4,
        plots=False,
        verbose=False,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    shutil.copy(best, MODELS / f"{args.name}.pt")
    print(f"model: {MODELS / f'{args.name}.pt'}")


def cmd_detect(args) -> None:
    from ultralytics import YOLO

    run = args.run
    cache = Cache(run / "cache")
    stride = cache_stride(run)
    idxs, _ = cache.processed_indices(LINK_FPS)
    wanted = set(int(i) for i in idxs) | set(pd.read_csv(run / "ball_truth.csv").ci) if (
        run / "ball_truth.csv").exists() else set(int(i) for i in idxs)  # fmt: skip
    model = YOLO(str(args.model))
    rows = []
    for clip_frame, img in read_frames(run / "clip.mp4", [ci * stride for ci in sorted(wanted)]):
        ci = clip_frame // stride
        res = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False, half=True)[0]
        for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), strict=True):
            rows.append((ci, clip_frame, round(ci / cache.fps, 3), BALL, float(c), x1, y1, x2, y2))
    cols = ["ci", "frame", "time_s", "cls", "conf", "x1", "y1", "x2", "y2"]
    df = pd.DataFrame(rows, columns=cols).round({"conf": 4, "x1": 1, "y1": 1, "x2": 1, "y2": 1})
    df.to_csv(run / "cache" / FT_FILE, index=False)
    print(f"{run}: {len(df)} ball candidates on {len(wanted)} frames -> cache/{FT_FILE}")


def cmd_apply(args) -> None:
    """detect, then ball_link.py on the new candidates (its defaults, min-conf 0.25: best or within a point of it
    on both held-out folds), replacing RUN/ball_path.csv. Events must be rerun after this."""
    cmd_detect(args)
    subprocess.run([sys.executable, str(HERE / "ball_link.py"), "--run", str(args.run), "--ball-cache", FT_FILE],
                   check=True)  # fmt: skip
    meta = {"model": str(args.model), "imgsz": args.imgsz, "conf": args.conf}
    (args.run / "cache" / FT_META).write_text(json.dumps(meta))


def top_hits(det: pd.DataFrame, truth: pd.DataFrame, cuts: list) -> list:
    """Per confidence cut: of visible balls, share where the top candidate is on the ball (hit), elsewhere
    (wrong) or absent (miss); and on frames with no visible ball, share with any candidate (ghost)."""
    det = det.assign(cx=(det.x1 + det.x2) / 2, cy=(det.y1 + det.y2) / 2)
    out = []
    for cut in cuts:
        top = det[det.conf >= cut].sort_values("conf", ascending=False).drop_duplicates("ci").set_index("ci")
        t = truth.assign(has=truth.ci.isin(top.index))
        err = np.hypot(t.ci.map(top.cx) - t.truth_x, t.ci.map(top.cy) - t.truth_y)
        vis = t.visible == 1
        n = max(int(vis.sum()), 1)
        out.append(
            dict(
                cut=cut,
                hit=round(100 * float((vis & (err <= TOL_PX)).sum()) / n, 1),
                wrong=round(100 * float((vis & t.has & (err > TOL_PX)).sum()) / n, 1),
                miss=round(100 * float((vis & ~t.has).sum()) / n, 1),
                ghost=round(100 * float((~vis & t.has).sum()) / max(int((~vis).sum()), 1), 1),
            )
        )
    return out


def cmd_evaluate(args) -> None:
    run = args.run
    truth = pd.read_csv(run / "ball_truth.csv")
    truth = truth[truth.verdict != "skipped"]
    coco = pd.read_csv(run / "cache" / "detections.csv.gz", usecols=["ci", "cls", "conf", "x1", "y1", "x2", "y2"])
    coco = coco[coco.cls == BALL]
    ft = pd.read_csv(run / "cache" / FT_FILE)
    cuts = [0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.65]
    rep = {"frames": len(truth), "visible": int((truth.visible == 1).sum())}
    for name, det in (("coco", coco), ("finetuned", ft)):
        rep[name] = top_hits(det[det.ci.isin(truth.ci)], truth, cuts)
    (run / "ball_ft_eval.json").write_text(json.dumps(rep, indent=2))
    print(f"{run}: {rep['frames']} labeled frames, ball visible in {rep['visible']}")
    for name in ("coco", "finetuned"):
        print(f"  {name:9s} " + "  ".join(f"@{r['cut']}: hit {r['hit']} wrong {r['wrong']} ghost {r['ghost']}"
                                          for r in rep[name] if r["cut"] in (0.1, 0.25, 0.5)))  # fmt: skip


def runs_arg(text: str) -> list:
    return [require_under_data(Path(r)) for r in text.split(",") if r]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("--runs", required=True, type=runs_arg, help="labeled runs for training")
    d.add_argument("--val", default=[], type=runs_arg, help="labeled runs for the validation split")
    d.add_argument("--out", required=True, type=require_under_data)
    d.set_defaults(fn=cmd_dataset)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True, type=require_under_data)
    t.add_argument("--name", required=True)
    t.add_argument("--base", default="yolo11m.pt")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch", type=int, default=8)
    t.set_defaults(fn=cmd_train)
    de = sub.add_parser("detect")
    de.add_argument("--run", required=True, type=require_under_data)
    de.add_argument("--model", required=True, type=Path)
    de.add_argument("--imgsz", type=int, default=1920)
    de.add_argument("--conf", type=float, default=0.05)
    de.set_defaults(fn=cmd_detect)
    a = sub.add_parser("apply", help="detect with the venue model and relink the ball path (pipeline stage)")
    a.add_argument("--run", required=True, type=require_under_data)
    a.add_argument("--model", type=Path, default=VENUE_MODEL)
    a.add_argument("--imgsz", type=int, default=1920)
    a.add_argument("--conf", type=float, default=0.05)
    a.set_defaults(fn=cmd_apply)
    e = sub.add_parser("evaluate")
    e.add_argument("--run", required=True, type=require_under_data)
    e.set_defaults(fn=cmd_evaluate)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
