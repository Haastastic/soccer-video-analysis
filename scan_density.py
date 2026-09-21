"""Scan a full game for crowded stretches so validation clips can be picked without watching the video.

Samples one frame every --every seconds, counts people, records the median person box height, and counts
overlapping person pairs (a crowd proxy: set pieces, scrambles, tackles). Windows must show zoomed-in play
(median box height at least --min-height in at least --min-play-frac of the samples), which rules out
pre-game and post-game stretches shot from a static wide view, where huddles look crowded. Then proposes
the best window of --window seconds that does not overlap the --exclude ranges. Writes OUT/density_scan.csv
and prints the proposal. Use --reuse to re-pick from a saved scan without reading the video again.

Example:
  python scan_density.py --video D:\\games\\game.mp4 --out data --exclude 00:14:00-00:19:00
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sv_common import PERSON, parse_time, require_under_data


def overlapping_pairs(xyxy: np.ndarray, iou_min: float = 0.1) -> int:
    n = len(xyxy)
    if n < 2:
        return 0
    x1 = np.maximum(xyxy[:, None, 0], xyxy[None, :, 0])
    y1 = np.maximum(xyxy[:, None, 1], xyxy[None, :, 1])
    x2 = np.minimum(xyxy[:, None, 2], xyxy[None, :, 2])
    y2 = np.minimum(xyxy[:, None, 3], xyxy[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    iou = inter / (area[:, None] + area[None, :] - inter + 1e-9)
    return int((np.triu(iou, 1) > iou_min).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--every", type=float, default=20, help="seconds between sampled frames")
    ap.add_argument("--window", type=float, default=300, help="proposed clip length in seconds")
    ap.add_argument("--exclude", action="append", default=[], help="HH:MM:SS-HH:MM:SS range to avoid, repeatable")
    ap.add_argument("--min-people", type=float, default=15, help="a window needs this mean person count (live play)")
    ap.add_argument(
        "--min-height", type=float, default=55, help="a sample counts as zoomed-in play above this box height"
    )
    ap.add_argument("--min-play-frac", type=float, default=0.9, help="share of samples in a window that must be play")
    ap.add_argument(
        "--reuse", action="store_true", help="reuse OUT/density_scan.csv instead of scanning the video again"
    )
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()
    args.out = require_under_data(args.out)

    args.out.mkdir(parents=True, exist_ok=True)
    scan_csv = args.out / "density_scan.csv"
    if args.reuse and scan_csv.exists():
        df = pd.read_csv(scan_csv)
    else:
        from ultralytics import YOLO

        model = YOLO(args.model)
        cap = cv2.VideoCapture(str(args.video))
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        rows = []
        for t in np.arange(0, duration, args.every):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, img = cap.read()
            if not ok:
                continue
            r = model.predict(img, classes=[PERSON], conf=0.3, imgsz=args.imgsz, verbose=False)[0]
            xyxy = r.boxes.xyxy.cpu().numpy()
            med_h = float(np.median(xyxy[:, 3] - xyxy[:, 1])) if len(xyxy) else 0.0
            rows.append((float(t), len(xyxy), overlapping_pairs(xyxy), med_h))
        cap.release()
        df = pd.DataFrame(rows, columns=["time_s", "people", "overlap_pairs", "median_h"])
        df.to_csv(scan_csv, index=False)

    excluded = []
    for text in args.exclude:
        a, b = text.split("-")
        excluded.append((parse_time(a), parse_time(b)))
    k = int(round(args.window / args.every))
    best = None
    for i in range(0, len(df) - k + 1):
        w = df.iloc[i : i + k]
        start, end = float(w.time_s.iloc[0]), float(w.time_s.iloc[-1]) + args.every
        if (
            any(start < hi and end > lo for lo, hi in excluded)
            or w.people.mean() < args.min_people
            or (w.median_h >= args.min_height).mean() < args.min_play_frac
        ):
            continue
        score = float(w.overlap_pairs.mean())
        if best is None or score > best[0]:
            best = (score, start, end, float(w.people.mean()))
    if best is None:
        print("No window met the constraints. Lower --min-people or relax --exclude.")
        return
    score, start, end, people = best
    hms = f"{int(start // 3600):02d}:{int(start % 3600 // 60):02d}:{int(start % 60):02d}"
    print(
        f"Most crowded window: start {hms} for {end - start:.0f} s, "
        f"mean people {people:.1f}, mean overlap pairs {score:.1f}"
    )
    print(f"Whole game: mean people {df.people.mean():.1f}, mean overlap pairs {df.overlap_pairs.mean():.1f}")


if __name__ == "__main__":
    main()
