"""Detect once, cache everything downstream stages need.

One detector pass over the clip at a low confidence floor (0.05) so ball candidates survive,
for both people and ball. For every cached frame it also records:
  - kit colors for each person (torso and legs), so team classification later needs no video
  - camera motion from the previous cached frame, so trackers and ball linking can cancel the pan

Outputs under --out/cache:
  meta.json, detections.csv.gz, camera.csv

Example:
  python detect_cache.py --video D:\\games\\game.mp4 --start 00:14:00 --duration 300 --out data\\clipA
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sv_common import BALL, CAM_COLS, COLOR_COLS, PERSON, cut_clip, parse_time


def region_median_rgb(img, x1, y1, x2, y2):
    x1, y1 = max(int(x1), 0), max(int(y1), 0)
    x2, y2 = min(int(x2), img.shape[1]), min(int(y2), img.shape[0])
    if x2 - x1 < 3 or y2 - y1 < 3:
        return (np.nan, np.nan, np.nan)
    reg = img[y1:y2, x1:x2].reshape(-1, 3)
    b, g, r = np.median(reg, axis=0)
    return (float(r), float(g), float(b))


def kit_colors(img, x1, y1, x2, y2):
    """Median RGB of a torso patch and a shorts/socks patch, avoiding box edges and grass."""
    w, h = x2 - x1, y2 - y1
    if h < 24 or w < 8:
        return (np.nan,) * 6
    tx1, tx2 = x1 + 0.25 * w, x2 - 0.25 * w
    torso = region_median_rgb(img, tx1, y1 + 0.20 * h, tx2, y1 + 0.50 * h)
    legs = region_median_rgb(img, tx1, y1 + 0.55 * h, tx2, y1 + 0.85 * h)
    return torso + legs


def estimate_motion(prev_gray, gray, prev_boxes, scale=0.5):
    """2x3 similarity transform mapping the previous frame to this one, plus inlier count."""
    ident = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    sp = cv2.resize(prev_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sc = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = np.full(sp.shape, 255, np.uint8)
    for x1, y1, x2, y2 in prev_boxes:  # ignore moving players when estimating camera motion
        mask[max(int(y1 * scale) - 3, 0) : int(y2 * scale) + 3, max(int(x1 * scale) - 3, 0) : int(x2 * scale) + 3] = 0
    pts = cv2.goodFeaturesToTrack(sp, maxCorners=400, qualityLevel=0.01, minDistance=8, mask=mask)
    if pts is None or len(pts) < 12:
        return ident, 0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(sp, sc, pts, None)
    good = st.ravel() == 1
    if good.sum() < 12:
        return ident, 0
    m, inl = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if m is None:
        return ident, 0
    m = m.astype(np.float64)
    m[:, 2] /= scale  # translation back to full resolution pixels
    return m, int(inl.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="run folder, the cache goes in OUT/cache")
    ap.add_argument("--start", default=None, help="clip start HH:MM:SS, omit to use the whole video")
    ap.add_argument("--duration", type=float, default=None, help="clip seconds")
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.05, help="detector floor, keep low for the ball")
    ap.add_argument(
        "--cache-fps", type=float, default=30, help="approximate cache rate, 30 keeps every frame of 30 fps video"
    )
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N cached frames, for quick tests")
    ap.add_argument("--recut", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out / "cache"
    cache_dir.mkdir(exist_ok=True)

    source = args.video
    if args.start is not None and args.duration is not None:
        source = args.out / "clip.mp4"
        if args.recut or not source.exists():
            cut_clip(args.video, parse_time(args.start), args.duration, source)

    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    stride = max(1, round(fps / args.cache_fps))
    cache_fps = fps / stride
    print(
        f"Source: {width}x{height}, {fps:.2f} fps, {total} frames. "
        f"Caching every {stride} frame(s), {cache_fps:.2f} fps."
    )

    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    print(
        "GPU:" if device == 0 else "WARNING: no CUDA, running on CPU.",
        torch.cuda.get_device_name(0) if device == 0 else "",
    )
    model = YOLO(args.model)
    results = model.predict(
        source=str(source),
        stream=True,
        classes=[PERSON, BALL],
        conf=args.conf,
        imgsz=args.imgsz,
        vid_stride=stride,
        device=device,
        half=(device != "cpu"),
        verbose=False,
    )

    rows, cam_rows = [], []
    prev_gray, prev_boxes = None, []
    t0 = time.time()
    n = -1
    for n, r in enumerate(results):
        img = r.orig_img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            m, inl = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64), 0
        else:
            m, inl = estimate_motion(prev_gray, gray, prev_boxes)
        cam_rows.append([n, *m.reshape(-1).tolist(), inl])
        prev_gray = gray

        prev_boxes = []
        frame_idx = n * stride
        b = r.boxes
        if b is not None and len(b):
            xyxy = b.xyxy.cpu().numpy()
            conf = b.conf.cpu().numpy()
            cls = b.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls, strict=True):
                col = (np.nan,) * 6
                if k == PERSON and c >= 0.3:
                    col = kit_colors(img, x1, y1, x2, y2)
                    prev_boxes.append((x1, y1, x2, y2))
                rows.append(
                    (
                        n,
                        frame_idx,
                        round(frame_idx / fps, 3),
                        int(k),
                        round(float(c), 3),
                        round(float(x1), 1),
                        round(float(y1), 1),
                        round(float(x2), 1),
                        round(float(y2), 1),
                        *[None if np.isnan(v) else round(v, 1) for v in col],
                    )
                )
        if n % 300 == 0:
            el = time.time() - t0
            print(f"  cached {n + 1} frames, {(n + 1) / max(el, 1e-6):.1f} frames/s")
        if args.max_frames and n + 1 >= args.max_frames:
            break

    n_cached = n + 1
    det = pd.DataFrame(rows, columns=["ci", "frame", "time_s", "cls", "conf", "x1", "y1", "x2", "y2", *COLOR_COLS])
    det.to_csv(cache_dir / "detections.csv.gz", index=False)
    cam = pd.DataFrame(cam_rows, columns=["ci", *CAM_COLS, "inliers"])
    cam.to_csv(cache_dir / "camera.csv", index=False)

    meta = {
        "source_fps": fps,
        "cache_fps": cache_fps,
        "stride": stride,
        "n_cached_frames": n_cached,
        "width": width,
        "height": height,
        "model": args.model,
        "imgsz": args.imgsz,
        "conf_floor": args.conf,
        "clip_start": args.start,
        "clip_duration": args.duration,
        "camera_inlier_median": float(cam.inliers.median()),
        "camera_frames_with_few_inliers": int((cam.inliers < 12).sum()),
        "seconds_elapsed": round(time.time() - t0, 1),
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"Detections: {len(det)} rows ({(det.cls == PERSON).sum()} person, {(det.cls == BALL).sum()} ball).")


if __name__ == "__main__":
    main()
