"""Phase 1: person and ball detection plus tracking on a clip of game video.

Outputs, all under --out:
  clip.mp4        the cut clip (reused on later runs unless --recut)
  tracks.csv      one row per detection: frame, time, class, track id, confidence, box
  annotated.mp4   clip with boxes and track IDs, for manual ID-switch review
  summary.json    proxy quality metrics
  crops/          optional person crops for the jersey OCR feasibility test

Example (PowerShell, from the project folder):
  python phase1_track.py --video D:\\games\\game.mp4 --start 00:14:00 --duration 300 `
      --tracker trackers\\bytetrack_long.yaml --out data\\run_bytetrack
  python phase1_track.py --video D:\\games\\game.mp4 --start 00:14:00 --duration 300 `
      --out data\\run_botsort    (BoT-SORT is the default tracker)
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

PERSON, BALL = 0, 32  # COCO class ids: person, sports ball
HERE = Path(__file__).resolve().parent


def parse_time(text: str) -> float:
    """Accepts SS, MM:SS, or HH:MM:SS."""
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def cut_clip(video: Path, start: float, duration: float, dest: Path) -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH. Install it with: winget install Gyan.FFmpeg")
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        str(video),
        "-t",
        str(duration),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        str(dest),
    ]
    print("Cutting clip:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def longest_gap_seconds(has_ball: np.ndarray, eff_fps: float) -> float:
    longest = run = 0
    for present in has_ball:
        run = 0 if present else run + 1
        longest = max(longest, run)
    return longest / eff_fps


def summarize(df: pd.DataFrame, n_frames: int, eff_fps: float, args) -> dict:
    out = {
        "processed_frames": n_frames,
        "effective_fps": round(eff_fps, 2),
        "clip_minutes": round(n_frames / eff_fps / 60, 2),
    }

    people = df[(df["cls"] == PERSON) & (df["track_id"] >= 0) & (df["conf"] >= args.person_conf)]
    per_frame = people.groupby("proc_idx").size().reindex(range(n_frames), fill_value=0)
    out["people_per_frame_mean"] = round(float(per_frame.mean()), 2)
    out["people_per_frame_median"] = int(per_frame.median())

    lengths = people.groupby("track_id").size() / eff_fps  # seconds each ID was visible
    out["unique_person_ids"] = int(lengths.size)
    out["track_len_s_median"] = round(float(lengths.median()), 1) if lengths.size else 0.0
    out["track_len_s_p90"] = round(float(lengths.quantile(0.9)), 1) if lengths.size else 0.0
    out["person_tracks_under_3s"] = int((lengths < 3).sum())

    # Fragmentation proxy: after the first 10 seconds every visible player already has an ID,
    # so a steady stream of new IDs means tracks are breaking and restarting.
    first_seen = people.groupby("track_id")["proc_idx"].min()
    later_minutes = max(n_frames / eff_fps / 60 - 10 / 60, 1e-6)
    late_births = int((first_seen > 10 * eff_fps).sum())
    out["new_person_ids_after_10s"] = late_births
    out["new_person_ids_per_minute"] = round(late_births / later_minutes, 1)

    ball = df[df["cls"] == BALL]
    for thr in sorted({args.conf, 0.3}):
        present = set(ball[ball["conf"] >= thr]["proc_idx"].tolist())
        has_ball = np.array([i in present for i in range(n_frames)])
        out[f"ball_frames_pct_conf_{thr}"] = round(100 * float(has_ball.mean()), 1)
        out[f"ball_longest_gap_s_conf_{thr}"] = round(longest_gap_seconds(has_ball, eff_fps), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path, help="full game video file")
    ap.add_argument("--out", required=True, type=Path, help="output folder for this run")
    ap.add_argument("--start", default="00:00:00", help="clip start, HH:MM:SS")
    ap.add_argument("--duration", type=float, default=300, help="clip length in seconds")
    ap.add_argument("--model", default="yolo11m.pt", help="Ultralytics weights, downloaded on first use")
    ap.add_argument("--tracker", default=str(HERE / "trackers" / "botsort_long.yaml"))
    ap.add_argument("--imgsz", type=int, default=1920, help="inference size, keep high for the ball")
    ap.add_argument("--conf", type=float, default=0.1, help="detector floor, low so the ball survives")
    ap.add_argument("--person-conf", type=float, default=0.3, help="min confidence counted in summary")
    ap.add_argument("--target-fps", type=float, default=10, help="approximate processing rate")
    ap.add_argument("--save-crops", type=int, default=0, help="save person crops every N processed frames, 0 = off")
    ap.add_argument("--min-crop-height", type=int, default=60, help="skip crops shorter than this many pixels")
    ap.add_argument("--recut", action="store_true", help="force re-cutting the clip")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    clip = args.out / "clip.mp4"
    if args.recut or not clip.exists():
        cut_clip(args.video, parse_time(args.start), args.duration, clip)

    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    stride = max(1, round(fps / args.target_fps))
    eff_fps = fps / stride
    print(
        f"Clip: {width}x{height}, {fps:.2f} fps, {total} frames. Processing every {stride} frames ({eff_fps:.2f} fps)."
    )

    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, running on CPU. Check the PyTorch install.")
    else:
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO(args.model)
    writer = cv2.VideoWriter(str(args.out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), eff_fps, (width, height))
    crops_dir = args.out / "crops"
    if args.save_crops:
        crops_dir.mkdir(exist_ok=True)

    results = model.track(
        source=str(clip),
        stream=True,
        persist=True,
        tracker=args.tracker,
        classes=[PERSON, BALL],
        conf=args.conf,
        imgsz=args.imgsz,
        vid_stride=stride,
        device=device,
        half=(device != "cpu"),
        verbose=False,
    )

    rows = []
    n = 0
    for n, r in enumerate(results):
        frame_idx = n * stride
        boxes = r.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.full(len(cls), -1)
            for (x1, y1, x2, y2), c, k, tid in zip(xyxy, conf, cls, ids, strict=True):
                rows.append(
                    (
                        n,
                        frame_idx,
                        round(frame_idx / fps, 3),
                        int(k),
                        int(tid),
                        round(float(c), 3),
                        round(float(x1), 1),
                        round(float(y1), 1),
                        round(float(x2), 1),
                        round(float(y2), 1),
                    )
                )
                if (
                    args.save_crops
                    and n % args.save_crops == 0
                    and k == PERSON
                    and tid >= 0
                    and c >= 0.5
                    and (y2 - y1) >= args.min_crop_height
                ):
                    crop = r.orig_img[max(int(y1), 0) : int(y2), max(int(x1), 0) : int(x2)]
                    if crop.size:
                        cv2.imwrite(str(crops_dir / f"t{tid}_f{frame_idx}.jpg"), crop)
        writer.write(r.plot(line_width=2))
        if n % 100 == 0:
            print(f"  processed {n + 1} frames")
    writer.release()

    n_frames = n + 1
    df = pd.DataFrame(rows, columns=["proc_idx", "frame", "time_s", "cls", "track_id", "conf", "x1", "y1", "x2", "y2"])
    df.to_csv(args.out / "tracks.csv", index=False)

    summary = summarize(df, n_frames, eff_fps, args)
    summary["model"] = args.model
    summary["tracker"] = Path(args.tracker).name
    summary["imgsz"] = args.imgsz
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Done. Review {args.out / 'annotated.mp4'} and log ID switches by hand.")


if __name__ == "__main__":
    main()
