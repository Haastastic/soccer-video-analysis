"""Process game windows end to end with no owner steps, resumably, then per-player stats across all of them.

Per window (NAME=HH:MM:SS, a 5-minute window starting there, in data/NAME):
  run_all.py (detection cache, chosen tracker, ball linking) -> pitch_mask.py -> team_classify.py classify
  -> tracklet_stitch.py -> events.py -> pitch_ptz.py run + pitch_calibrate.py apply (automatic pitch)
  -> jersey_auto.py identify --write (automatic identity, learned from the owner-labeled windows in --labeled).
A stage is skipped when its output already exists, so an interrupted run resumes where it stopped (delete an
output to redo that stage). A window the owner labeled is never given automatic identity.
At the end, player_stats.py over --stats-runs plus every window here.

Check a window is live play before spending GPU time on it (density scan, a few stills): CLAUDE.md Phase 1b.
Compute per window: about 11 min detection, 5 min pitch, 10 min reads and features, under 2 min the rest.
Everything written stays under data/ (git-ignored): it is derived from footage of minors.

Example:
  python run_windows.py --video "videos\\game.mp4" --windows clipH=00:00:00,clipI=00:05:00 `
      --labeled data\\clipA,data\\clipB,data\\clipE --stats-runs data\\clipA,data\\clipB,data\\clipE
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from sv_common import require_under_data

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def run(cmd: list, log) -> None:
    line = " ".join(str(c) for c in cmd)
    print(f"  {line}", flush=True)
    log.write(f"\n=== {line}\n")
    log.flush()
    t = time.time()
    subprocess.run([sys.executable, "-u", *[str(c) for c in cmd]], check=True, cwd=HERE, stdout=log, stderr=log)
    print(f"    done in {time.time() - t:.0f} s", flush=True)


def stages(video: Path, run_dir: Path, start: str, labeled: list) -> list:
    """(output that marks the stage done, command) in order."""
    r = run_dir
    auto_identity = r / "identity_segments.csv"
    return [
        (r / "ball_path.csv", ["run_all.py", "--video", video, "--start", start, "--duration", "300", "--out", r]),
        (r / "tracklet_pitch.csv", ["pitch_mask.py", "--run", r]),
        (r / "tracklet_roles.csv", ["team_classify.py", "classify", "--run", r]),
        (r / "tracklet_stitch.csv", ["tracklet_stitch.py", "--run", r]),
        (r / "events.csv", ["events.py", "--run", r]),
        (r / "pitch_anchors_ptz.local.json", ["pitch_ptz.py", "run", "--run", r]),
        (
            r / "tracklet_pitch_xy.csv.gz",
            ["pitch_calibrate.py", "apply", "--run", r, "--anchors", r / "pitch_anchors_ptz.local.json"],
        ),
        (
            auto_identity,
            ["jersey_auto.py", "identify", "--run", r, "--from", ",".join(str(p) for p in labeled), "--write"],
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--windows", required=True, help="comma list of NAME=HH:MM:SS")
    ap.add_argument("--labeled", required=True, help="comma list of owner-labeled windows (identity sources)")
    ap.add_argument("--stats-runs", default="", help="comma list of other windows to include in player_stats")
    args = ap.parse_args()

    labeled = [require_under_data(Path(p)).resolve() for p in args.labeled.split(",")]
    windows = []
    for item in args.windows.split(","):
        name, start = item.split("=", 1)
        run_dir = require_under_data(DATA / name).resolve()
        if run_dir in labeled:
            raise SystemExit(f"{name} is owner-labeled: it must not get automatic identity")
        windows.append((name, start, run_dir))

    for name, start, run_dir in windows:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{name} ({start})", flush=True)
        with open(DATA / f"{name}_window.log", "a", encoding="utf-8") as log:
            for done, cmd in stages(args.video, run_dir, start, labeled):
                if done.exists():
                    print(f"  skip {cmd[0]} ({done.name} exists)")
                    continue
                run(cmd, log)

    stats = [p for p in args.stats_runs.split(",") if p] + [str(w[2]) for w in windows]
    with open(DATA / "run_windows_stats.log", "a", encoding="utf-8") as log:
        run(["player_stats.py", "--runs", ",".join(stats)], log)
    print("\nall windows done")


if __name__ == "__main__":
    main()
