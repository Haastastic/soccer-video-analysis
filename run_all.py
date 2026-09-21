"""Run the whole Phase 1b stage with one command and produce one report.

Steps: detect_cache (skipped if the cache exists) -> replay_trackers -> ball_link -> report.md

The report holds aggregate numbers only (no images, no crops, no names). --share-dir copies it,
plus the sweep table and ball report, to a folder Claude can read, so results do not need to be
pasted back into chat.

Example (PowerShell):
  python run_all.py --video D:\\games\\game.mp4 --start 00:14:00 --duration 300 --out data\\clipA `
      --share-dir <share-folder>
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def step(title: str, cmd: list) -> None:
    print(f"\n=== {title} ===")
    print(" ".join(str(c) for c in cmd))
    subprocess.run([sys.executable, *[str(c) for c in cmd]], check=True, cwd=HERE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--start", default=None)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--grid", choices=["quick", "full"], default="quick")
    ap.add_argument("--fps-list", default="10,15,30")
    ap.add_argument("--redetect", action="store_true", help="rebuild the cache even if it exists")
    ap.add_argument("--share-dir", type=Path, default=None)
    args = ap.parse_args()

    out = args.out.resolve()
    if args.redetect or not (out / "cache" / "meta.json").exists():
        cmd = ["detect_cache.py", "--video", args.video, "--out", out, "--model", args.model]
        if args.start is not None and args.duration is not None:
            cmd += ["--start", args.start, "--duration", args.duration]
            if args.redetect:
                cmd.append("--recut")  # a changed clip must not reuse an old clip.mp4
        step("1/3 Detection cache", cmd)
    else:
        print("Cache exists, skipping detection. Use --redetect to rebuild.")
        old = json.loads((out / "cache" / "meta.json").read_text())
        if (args.start, args.duration) != (old.get("clip_start"), old.get("clip_duration")) or (
            old.get("model") != args.model
        ):
            print(
                "WARNING: the cache was built with a different start, duration or model "
                f"({old.get('clip_start')}, {old.get('clip_duration')}, {old.get('model')}). "
                "Add --redetect if you meant to change them."
            )

    step("2/3 Tracker replay", ["replay_trackers.py", "--run", out, "--grid", args.grid, "--fps-list", args.fps_list])
    step("3/3 Ball linking", ["ball_link.py", "--run", out, "--fps", "10"])

    meta = json.loads((out / "cache" / "meta.json").read_text())
    ball = json.loads((out / "ball_link_report.json").read_text())
    lines = [
        f"# Phase 1b report: {out.name}",
        "",
        "## Cache",
        f"- Source {meta['width']}x{meta['height']} at {meta['source_fps']:.1f} fps, "
        f"cached at {meta['cache_fps']:.1f} fps, {meta['n_cached_frames']} frames",
        f"- Model {meta['model']}, image size {meta['imgsz']}, confidence floor {meta['conf_floor']}",
        f"- Detection time {meta['seconds_elapsed']} s",
        f"- Camera motion: median inliers {meta['camera_inlier_median']}, "
        f"frames with few inliers {meta['camera_frames_with_few_inliers']}",
        "",
        "## Ball linking",
        "```json",
        json.dumps(ball, indent=2),
        "```",
        "",
        (out / "tracker_report.md").read_text(),
    ]
    report = out / "report.md"
    report.write_text("\n".join(lines))
    print(f"\nWrote {report}")

    if args.share_dir:
        args.share_dir.mkdir(parents=True, exist_ok=True)
        stem = out.name
        shutil.copy(report, args.share_dir / f"{stem}_report.md")
        shutil.copy(out / "sweep_results.csv", args.share_dir / f"{stem}_sweep_results.csv")
        shutil.copy(out / "ball_link_report.json", args.share_dir / f"{stem}_ball_link_report.json")
        shutil.copy(out / "cache" / "meta.json", args.share_dir / f"{stem}_meta.json")
        print(f"Copied report files to {args.share_dir}")


if __name__ == "__main__":
    main()
