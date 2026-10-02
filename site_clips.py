"""Short video clips of each identified player's moments, for the coaching site (owner decision 2026-10-02).

The moments are the local pages' "Watch on video" lists (coaching_tips.watch_moments): longest stretches on camera,
fastest running, longest possessions and touches. Each clip is CLIP_S seconds from the moment's start, cropped to a
CROP_W x CROP_H window at native resolution that follows the player (their identified boxes, smoothed), with a small
marker above the player's head, and NO AUDIO (voices and names). The full game video never leaves this computer.

Clips are cut once and kept in data/site_clips/<team>/<game>/<NN>_<start ms>.mp4 (git-ignored, they show minors);
site_export.py calls game_clips() for every game and writes the index (clips.json) next to the game's page data.
publish_site.py uploads the clips the index names, only those missing in the bucket, and the site serves each one only
to people allowed to see that player.

Example (normally run through site_export.py):
  python site_clips.py --runs data\\g1001\\w0100,...      (cut the clips of these games' windows, print a summary)
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import coaching_tips as ct
from sv_common import DATA_DIR, game_dir, require_under_data

CLIPS = DATA_DIR / "site_clips"
CLIP_S = 10.0  # seconds per clip (a moment already starts coaching_tips.WATCH_LEAD_S early)
CROP_W, CROP_H = 960, 540  # the window that follows the player, at native resolution (players are 60-90 px tall)
SMOOTH_S = 1.0  # crop centre moving average, so the window pans smoothly
MARK_GAP_S = 0.2  # the marker is drawn only where an identified box is this close in time
CRF = 26


def clip_name(jersey: int, start_s: float) -> str:
    return f"{int(jersey):02d}_{int(round(start_s * 10)) * 100}.mp4"


def player_boxes(g: dict, jersey: int, t0: float, t1: float, boxes: dict) -> pd.DataFrame:
    """The player's identified boxes between video times t0 and t1: video_s, cx, cy (box centre), top, x."""
    s = g["s"]
    q = s[(s.jersey == jersey) & (s.video_s >= t0 - MARK_GAP_S) & (s.video_s <= t1 + MARK_GAP_S)]
    if not len(q):
        return pd.DataFrame(columns=["video_s", "cx", "cy", "top"])
    runs = {r.name: r for r in g["runs"]}
    out = []
    for wname, part in q.groupby(q.run.str.split("/").str[-1]):
        if wname not in boxes:
            tr = pd.read_csv(runs[wname] / "best_tracklets.csv.gz", usecols=["ci", "track_id", "x1", "y1", "x2", "y2"])
            boxes[wname] = tr.set_index(["track_id", "ci"])
        b = boxes[wname].reindex(pd.MultiIndex.from_arrays([part.track_id, part.ci])).to_numpy()
        out.append(
            pd.DataFrame(
                dict(
                    video_s=part.video_s.to_numpy(),
                    cx=(b[:, 0] + b[:, 2]) / 2,
                    cy=(b[:, 1] + b[:, 3]) / 2,
                    top=b[:, 1],
                )
            )
        )
    return pd.concat(out).dropna().sort_values("video_s").reset_index(drop=True)


def cut(video: Path, start_s: float, pb: pd.DataFrame, dest: Path) -> bool:
    """Write one clip: CLIP_S seconds from start_s, cropped around the player, marker above their head, no audio."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cw, ch = min(CROP_W, W), min(CROP_H, H)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    n = int(round(CLIP_S * fps))
    t = start_s + np.arange(n) / fps
    # crop centre: the player's box centre, held at the ends, then smoothed so the window pans gently
    cx = np.interp(t, pb.video_s, pb.cx)
    cy = np.interp(t, pb.video_s, pb.cy)
    k = max(1, int(round(SMOOTH_S * fps)))
    ker = np.ones(k) / k
    cx = np.convolve(np.pad(cx, (k // 2, k - 1 - k // 2), mode="edge"), ker, mode="valid")
    cy = np.convolve(np.pad(cy, (k // 2, k - 1 - k // 2), mode="edge"), ker, mode="valid")
    x0 = np.clip(np.round(cx - cw / 2), 0, W - cw).astype(int)
    y0 = np.clip(np.round(cy - ch / 2), 0, H - ch).astype(int)
    near = np.abs(t[:, None] - pb.video_s.to_numpy()[None, :]).min(axis=1) <= MARK_GAP_S
    mx, mtop = np.interp(t, pb.video_s, pb.cx), np.interp(t, pb.video_s, pb.top)
    tmp = dest.with_suffix(".part.mp4")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{cw}x{ch}",
           "-r", f"{fps:.3f}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(CRF),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    written = 0
    try:
        for i in range(n):
            ok, img = cap.read()
            if not ok:
                break
            crop = np.ascontiguousarray(img[y0[i] : y0[i] + ch, x0[i] : x0[i] + cw])
            if near[i]:
                x, y = int(mx[i] - x0[i]), int(mtop[i] - y0[i]) - 6
                tri = np.array([[x - 9, y - 16], [x + 9, y - 16], [x, y]], np.int32)
                cv2.polylines(crop, [tri], True, (0, 0, 0), 4, cv2.LINE_AA)  # a dark rim shows on white lines too
                cv2.fillConvexPoly(crop, tri, (255, 255, 255), cv2.LINE_AA)
            proc.stdin.write(crop.tobytes())
            written += 1
    finally:
        proc.stdin.close()
        proc.wait()
        cap.release()
    if proc.returncode != 0 or written < n // 2:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dest)
    return True


def game_clips(g: dict, cut_missing: bool = True) -> dict:
    """{jersey: [[group title, [[video seconds, caption, clip name], ...]], ...]} for one game, cutting clips not
    yet on disk; a moment whose clip does not exist (not cut, or failed) is left out."""
    video = DATA_DIR.parent / g["video"] if g.get("video") else None
    if video is None or not video.exists():
        print(f"{g['label']}: no video on this computer, no clips")
        return {}
    folder = CLIPS / g["team"] / g["label"]
    boxes, index, made = {}, {}, 0
    for jersey in sorted(g["s"].jersey.unique()):
        groups = []
        for title, items in ct.player_moments(g, jersey):
            rows = []
            for t, cap in items:
                start = round(max(0.0, t), 1)
                name = clip_name(jersey, start)
                dest = folder / name
                if not dest.exists() and cut_missing:
                    pb = player_boxes(g, jersey, start, start + CLIP_S, boxes)
                    if len(pb) and cut(video, start, pb, dest):
                        made += 1
                if dest.exists():
                    rows.append([float(t), cap, name])
            if rows:
                groups.append([title, rows])
        if groups:
            index[str(int(jersey))] = groups
    total = sum(len(r) for gs in index.values() for _, r in gs)
    print(f"{g['label']}: {total} clips for {len(index)} players ({made} cut now) in {folder}")
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="comma list of window folders with identity (one or more games)")
    args = ap.parse_args()
    by_game = {}
    for r in args.runs.split(","):
        run = require_under_data(Path(r))
        by_game.setdefault(game_dir(run), []).append(run)
    for rs in by_game.values():
        game_clips(ct.load_game(rs))


if __name__ == "__main__":
    main()
