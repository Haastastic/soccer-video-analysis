"""Detector-only ball baseline: no tracker, so the numbers are not filtered by ByteTrack/BoT-SORT.

Reads clip.mp4 from --out (made by phase1_track.py), runs the same model, image size, confidence
floor and frame stride, and writes under --out:
  ball_detections.csv   every ball box the detector produced
  ball_baseline.json    ball_frames_pct and longest_gap at each confidence threshold

Example (PowerShell, from the project folder):
  python phase1_ball_baseline.py --out data\\run_botsort
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from phase1_track import BALL, longest_gap_seconds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="run folder that already contains clip.mp4")
    ap.add_argument("--model", default="yolo11m.pt", help="Ultralytics weights")
    ap.add_argument("--imgsz", type=int, default=1920, help="inference size, keep equal to the tracking runs")
    ap.add_argument("--conf", type=float, default=0.1, help="detector floor, keep equal to the tracking runs")
    ap.add_argument("--target-fps", type=float, default=10, help="approximate processing rate")
    args = ap.parse_args()

    clip = args.out / "clip.mp4"
    if not clip.exists():
        raise SystemExit(f"{clip} not found. Run phase1_track.py with --out {args.out} first.")

    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    stride = max(1, round(fps / args.target_fps))
    eff_fps = fps / stride

    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model)
    results = model.predict(
        source=str(clip),
        stream=True,
        classes=[BALL],
        conf=args.conf,
        imgsz=args.imgsz,
        vid_stride=stride,
        device=device,
        half=(device != "cpu"),
        verbose=False,
    )

    rows = []
    n = -1
    for n, r in enumerate(results):
        boxes = r.boxes
        if boxes is not None and len(boxes):
            for (x1, y1, x2, y2), c in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), strict=True):
                rows.append(
                    (
                        n,
                        round(n * stride / fps, 3),
                        round(float(c), 3),
                        round(float(x1), 1),
                        round(float(y1), 1),
                        round(float(x2), 1),
                        round(float(y2), 1),
                    )
                )
        if n % 100 == 0:
            print(f"  processed {n + 1} frames")
    n_frames = n + 1

    df = pd.DataFrame(rows, columns=["proc_idx", "time_s", "conf", "x1", "y1", "x2", "y2"])
    df.to_csv(args.out / "ball_detections.csv", index=False)

    out = {
        "processed_frames": n_frames,
        "effective_fps": round(eff_fps, 2),
        "model": args.model,
        "imgsz": args.imgsz,
        "tracker": None,
    }
    for thr in sorted({args.conf, 0.3}):
        present = set(df[df["conf"] >= thr]["proc_idx"].tolist())
        has_ball = np.array([i in present for i in range(n_frames)])
        out[f"ball_frames_pct_conf_{thr}"] = round(100 * float(has_ball.mean()), 1)
        out[f"ball_longest_gap_s_conf_{thr}"] = round(longest_gap_seconds(has_ball, eff_fps), 1)
    (args.out / "ball_baseline.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
