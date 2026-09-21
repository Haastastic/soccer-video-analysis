"""Decide which tracklets are on the pitch, from the ground under their feet.

Color cannot tell players from bench people, coaches or spectators who wear the same kit. Players run on
the grass, while the bench, the track and the stands are not grass. For SAMPLES frames per tracklet this
measures the fraction of grass pixels in a small window at the feet, and takes the median.

Grass is bright, saturated green. It is measured per frame, so lighting changes and shadows on the pitch
are handled by generous thresholds rather than a fixed pitch polygon. Trees and hillsides above the pitch
are also green, but they are never at a player's feet.

Outputs OUT/tracklet_pitch.csv: track_id, foot_grass (0 to 1), on_pitch (foot_grass >= --min-grass), n_samples.

Example:
  python pitch_mask.py --run data\\clipA
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sv_common import cache_stride, cached_per_tracklet, read_frames, require_under_data, sample_rows

SAMPLES = 8
SCALE = 0.5  # the mask is computed at half resolution
MIN_GRASS = 0.5


def grass_mask(img: np.ndarray) -> np.ndarray:
    """Boolean mask of grass at SCALE resolution: bright saturated green, lightly cleaned."""
    small = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h >= 30) & (h <= 85) & (s > 60) & (v > 60)).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)


def foot_grass_fraction(mask: np.ndarray, x1, y1, x2, y2) -> float:
    """Grass fraction in a window at the feet, from just above the box bottom to just below it."""
    h, w = y2 - y1, x2 - x1
    cx = (x1 + x2) / 2
    xa, xb = int((cx - 0.35 * w) * SCALE), int((cx + 0.35 * w) * SCALE)
    ya, yb = int((y2 - 0.04 * h) * SCALE), int((y2 + 0.14 * h) * SCALE)
    xa, xb = max(xa, 0), min(xb + 1, mask.shape[1])
    ya, yb = max(ya, 0), min(yb + 1, mask.shape[0])
    if xb <= xa or yb <= ya:
        return np.nan
    return float(mask[ya:yb, xa:xb].mean())


def _measure(run: Path, tr: pd.DataFrame) -> pd.DataFrame:
    stride = cache_stride(run)
    by_frame = {}
    for r in sample_rows(tr, SAMPLES).itertuples():
        by_frame.setdefault(int(r.ci), []).append(r)
    vals = {}
    for clip_frame, img in read_frames(run / "clip.mp4", [ci * stride for ci in by_frame]):
        mask = grass_mask(img)
        for r in by_frame[clip_frame // stride]:
            vals.setdefault(r.track_id, []).append(foot_grass_fraction(mask, r.x1, r.y1, r.x2, r.y2))
    rows = {tid: (float(np.nanmedian(v)), int(np.isfinite(v).sum())) for tid, v in vals.items() if np.isfinite(v).any()}
    return pd.DataFrame.from_dict(rows, orient="index", columns=["foot_grass", "n_samples"])


def on_pitch_table(run: Path, min_grass: float = MIN_GRASS) -> pd.DataFrame:
    """Per-tracklet foot_grass and on_pitch, cached and tied to the current tracklets."""
    tr = pd.read_csv(run / "best_tracklets.csv.gz")
    df = cached_per_tracklet(run, "tracklet_pitch", tr, f"pitch-v1:{SAMPLES}", lambda: _measure(run, tr))
    df = df.copy()
    df["on_pitch"] = df.foot_grass >= min_grass
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument(
        "--min-grass", type=float, default=MIN_GRASS, help="foot grass fraction needed to count as on pitch"
    )
    args = ap.parse_args()
    if getattr(args, "run", None) is not None:
        args.run = require_under_data(args.run)
    df = on_pitch_table(args.run, args.min_grass)
    print(f"{len(df)} tracklets measured, {int(df.on_pitch.sum())} on the pitch, {int((~df.on_pitch).sum())} off it.")
    print("foot_grass quantiles:", df.foot_grass.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(2).to_dict())


if __name__ == "__main__":
    main()
