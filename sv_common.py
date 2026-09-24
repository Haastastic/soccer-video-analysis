"""Shared helpers for the soccer analysis pipeline.

Cache layout (written by detect_cache.py, read by everything else):
  meta.json          clip facts: fps, size, model, settings, number of cached frames
  detections.csv.gz  every person and ball detection, one row each
  camera.csv         per cached frame: 2x3 camera motion from the previous cached frame to this one

Frames are addressed by "ci", the position in the cache (0, 1, 2, ...). Replays at a lower
frame rate simply take every m-th cached frame and compose the camera motion in between.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PERSON, BALL = 0, 32
DATA_DIR = Path(__file__).resolve().parent / "data"


def require_under_data(path: Path) -> Path:
    """Exit unless path is this repo's data/ folder or inside it, and return it unchanged.

    Everything the pipeline writes is derived from footage of minors, and only data/ is git-ignored. The check
    compares resolved paths, case-insensitively on Windows, so a folder that merely has "data" in its name, or a
    path outside the repo, is refused.
    """
    resolved = Path(os.path.normcase(str(Path(path).resolve())))
    root = Path(os.path.normcase(str(DATA_DIR.resolve())))
    if resolved != root and root not in resolved.parents:
        raise SystemExit(
            f"{path} is not under {DATA_DIR}. Outputs are derived from footage of minors "
            "and must stay in the git-ignored data/ folder of this repo."
        )
    return Path(path)


CAM_COLS = ["m00", "m01", "m02", "m10", "m11", "m12"]
COLOR_COLS = ["torso_r", "torso_g", "torso_b", "legs_r", "legs_g", "legs_b"]


def parse_time(text: str) -> float:
    """Accepts SS, MM:SS, or HH:MM:SS."""
    seconds = 0.0
    for part in str(text).split(":"):
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


def read_frames(video: Path, frame_numbers):
    """Yield (frame_number, image) for the requested clip frames in ascending order.

    Decodes the clip once from the start and skips with grab(), which is far faster than seeking to each frame
    (a few thousand random seeks on 1080p H.264 took about 8 minutes, one sequential pass about a minute).
    A frame that cannot be decoded is skipped. If the stream ends early, the remaining frames are lost, so the
    shortfall is printed instead of being silent: callers otherwise see tracklets with missing samples.
    """
    wanted = sorted({int(f) for f in frame_numbers})
    cap = cv2.VideoCapture(str(video))
    pos, yielded, ended_at = 0, 0, None
    try:
        for f in wanted:
            while pos < f:
                if not cap.grab():
                    ended_at = pos
                    return
                pos += 1
            ok, img = cap.read()
            pos += 1
            if not ok:
                continue
            yielded += 1
            yield f, img
    finally:
        cap.release()
        if yielded < len(wanted):
            where = f" (stream ended near frame {ended_at})" if ended_at is not None else ""
            print(f"WARNING: read_frames got {yielded} of {len(wanted)} requested frames from {video}{where}.")


def cache_stride(run: Path) -> int:
    """Video frames per cached frame. ci counts cached frames, so the clip frame is ci * stride."""
    return int(json.loads((run / "cache" / "meta.json").read_text()).get("stride", 1))


def sample_rows(tr: pd.DataFrame, n: int = 8, min_h: float = 24) -> pd.DataFrame:
    """Up to n evenly spaced rows per tracklet, skipping boxes too small to measure."""
    parts = []
    for _, d in tr[(tr.y2 - tr.y1) >= min_h].groupby("track_id"):
        parts.append(d.iloc[np.linspace(0, len(d) - 1, min(n, len(d))).round().astype(int)])
    return pd.concat(parts) if parts else tr.iloc[:0]


def tracklet_fingerprint(tr: pd.DataFrame, salt: str = "") -> str:
    """Identifies this exact set of tracklets, so cached per-tracklet results are never joined onto regenerated IDs."""
    cols = tr[["track_id", "ci", "x1", "y1", "x2", "y2"]].to_numpy()
    return hashlib.sha1(cols.tobytes() + salt.encode()).hexdigest()


def cached_per_tracklet(run: Path, name: str, tr: pd.DataFrame, salt: str, compute):
    """Return compute() as a DataFrame indexed by track_id, cached as run/name.csv and tied to the tracklets."""
    dest, stamp = run / f"{name}.csv", run / f"{name}.sha1"
    fingerprint = tracklet_fingerprint(tr, salt)
    if dest.exists() and stamp.exists() and stamp.read_text().strip() == fingerprint:
        return pd.read_csv(dest, index_col="track_id")
    if dest.exists():
        print(f"Tracklets changed since {name} was measured, re-measuring.")
    df = compute()
    df.index.name = "track_id"
    df.to_csv(dest)
    stamp.write_text(fingerprint)
    return df


# ---------------------------------------------------------------- cache IO


class Cache:
    """Loaded detection cache plus camera transforms."""

    def __init__(self, cache_dir):
        self.dir = Path(cache_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.det = pd.read_csv(self.dir / "detections.csv.gz")
        self.n = int(self.meta["n_cached_frames"])
        self.fps = float(self.meta["cache_fps"])
        cam_path = self.dir / "camera.csv"
        self.has_camera = cam_path.exists()
        if self.has_camera:
            cam = pd.read_csv(cam_path).sort_values("ci")
            rel = np.zeros((self.n, 3, 3))
            rel[:] = np.eye(3)
            idx = cam["ci"].to_numpy()
            m = cam[CAM_COLS].to_numpy().reshape(-1, 2, 3)
            rel[idx, :2, :] = m
        else:
            rel = np.tile(np.eye(3), (self.n, 1, 1))
        self.rel = rel  # rel[k] maps frame k-1 coordinates to frame k coordinates
        cum = np.empty_like(rel)
        cum[0] = np.eye(3)
        for k in range(1, self.n):
            cum[k] = rel[k] @ cum[k - 1]
        self.cum = cum  # cum[k] maps frame 0 coordinates to frame k coordinates
        self.cum_inv = np.linalg.inv(cum)  # frame k coordinates back to the frame 0 ("stable") system

    def processed_indices(self, target_fps: float):
        """Cache indices to use for a replay at roughly target_fps, and the actual fps."""
        m = max(1, int(round(self.fps / target_fps)))
        return np.arange(0, self.n, m), self.fps / m

    def step_warp(self, ci_prev: int, ci_now: int) -> np.ndarray:
        """Camera motion from cached frame ci_prev to ci_now as a 3x3 matrix."""
        h = np.eye(3)
        for k in range(ci_prev + 1, ci_now + 1):
            h = self.rel[k] @ h
        return h

    def to_stable(self, ci, x, y):
        """Map image points at cached frames ci into the frame 0 coordinate system."""
        ci = np.asarray(ci, dtype=int)
        pts = np.stack([np.asarray(x, float), np.asarray(y, float), np.ones(len(ci))], axis=1)
        out = np.einsum("nij,nj->ni", self.cum_inv[ci], pts)
        return out[:, 0], out[:, 1]

    def to_image(self, ci, sx, sy):
        """Inverse of to_stable."""
        ci = np.asarray(ci, dtype=int)
        pts = np.stack([np.asarray(sx, float), np.asarray(sy, float), np.ones(len(ci))], axis=1)
        out = np.einsum("nij,nj->ni", self.cum[ci], pts)
        return out[:, 0], out[:, 1]


# ---------------------------------------------------------------- track metrics


def track_metrics(tr: pd.DataFrame, fps: float, n_proc: int, cache: Cache, expected_players: int = 23) -> dict:
    """Proxy quality metrics for a set of person tracks. No ground truth needed.

    tr needs: pf (processed frame number), ci, track_id, x1, y1, x2, y2.
    Lower is better for: unique_ids, new_ids_per_min, swap_suspects_per_min, tracks_under_2s.
    Higher is better for: pct_rows_in_tracks_10s_plus, median_track_s.
    """
    out = {}
    tr = tr.sort_values(["track_id", "pf"]).reset_index(drop=True)
    tr["cx"] = (tr.x1 + tr.x2) / 2
    tr["cy"] = (tr.y1 + tr.y2) / 2
    tr["h"] = (tr.y2 - tr.y1).clip(lower=1)
    sx, sy = cache.to_stable(tr.ci.to_numpy(), tr.cx.to_numpy(), tr.cy.to_numpy())
    tr["sx"], tr["sy"] = sx, sy

    minutes = n_proc / fps / 60
    g = tr.groupby("track_id")
    length_s = g.size() / fps
    first = g.pf.min()
    last = g.pf.max()
    out["unique_ids"] = int(length_s.size)
    out["ids_per_expected_player"] = round(length_s.size / expected_players, 1)
    out["median_track_s"] = round(float(length_s.median()), 1)
    out["tracks_under_2s"] = int((length_s < 2).sum())
    out["pct_rows_in_tracks_10s_plus"] = round(100 * float(g.size()[length_s >= 10].sum() / len(tr)), 1)

    late_min = max(minutes - 10 / 60, 1e-6)
    births = first[first > 10 * fps]
    out["new_ids_after_10s"] = int(len(births))
    out["new_ids_per_min"] = round(len(births) / late_min, 1)

    # Births that start soon and near where another track ended: likely the same player restarting.
    deaths = last[last < n_proc - 10 * fps]
    if len(births) and len(deaths):
        b = g[["pf", "sx", "sy", "h"]].first().loc[births.index]
        lastrows = tr.loc[tr.groupby("track_id").pf.idxmax()].set_index("track_id").loc[deaths.index]
        bt = b.pf.to_numpy()[:, None]
        dt = lastrows.pf.to_numpy()[None, :]
        near_time = (bt > dt) & ((bt - dt) <= 1.5 * fps)
        dist = np.hypot(
            b.sx.to_numpy()[:, None] - lastrows.sx.to_numpy()[None, :],
            b.sy.to_numpy()[:, None] - lastrows.sy.to_numpy()[None, :],
        )
        near = dist <= 2.5 * b.h.to_numpy()[:, None]
        out["births_that_look_like_restarts"] = int((near_time & near).any(axis=1).sum())
    else:
        out["births_that_look_like_restarts"] = 0

    # Swap suspects: the same ID jumps further between short-gap observations than a player can move.
    same = tr.track_id.to_numpy()[1:] == tr.track_id.to_numpy()[:-1]
    dpf = np.diff(tr.pf.to_numpy())
    disp = np.hypot(np.diff(tr.sx.to_numpy()), np.diff(tr.sy.to_numpy()))
    h = tr.h.to_numpy()[1:]
    dts = dpf / fps
    thr = np.maximum(0.5 * h, 10 * h * dts)
    susp = same & (dts <= 0.5) & (disp > thr)
    out["swap_suspects"] = int(susp.sum())
    out["swap_suspects_per_min"] = round(float(susp.sum()) / max(minutes, 1e-6), 1)
    out["score"] = round(out["new_ids_per_min"] + 3 * out["swap_suspects_per_min"], 1)
    return out


# ---------------------------------------------------------------- review UI zoom


def screen_size() -> tuple:
    """Usable display size in pixels for review windows (primary screen, minus room for title bar and taskbar)."""
    try:
        import ctypes

        user32 = ctypes.windll.user32
        w, h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except (AttributeError, OSError):
        w, h = 1920, 1080
    return max(640, w - 40), max(480, h - 140)


TILE_W, TILE_H = 220, 320  # one crop in the crop review tools
FOOTER_H = 24
SCROLL_BAR = 14  # scroll bar thickness in window pixels


def crop_tile(img: np.ndarray, box: dict, w: int = TILE_W, h: int = TILE_H) -> np.ndarray:
    """A fixed-size crop around a person box (feet near the bottom), with the box drawn in green."""
    cx, fy = int((box["x1"] + box["x2"]) / 2), int(box["y2"])
    x0 = int(np.clip(cx - w / 2, 0, img.shape[1] - w))
    y0 = int(np.clip(fy - h * 0.8, 0, img.shape[0] - h))
    out = img[y0 : y0 + h, x0 : x0 + w].copy()
    cv2.rectangle(
        out, (int(box["x1"]) - x0, int(box["y1"]) - y0), (int(box["x2"]) - x0, int(box["y2"]) - y0), (0, 255, 0), 2
    )
    return out


def grid_cols(n: int, tile_w: int = TILE_W) -> int:
    """Columns for n tiles: fit the screen width (leaving room for a vertical scroll bar), balanced across rows.

    32 crops on a 5120 px screen give 16 x 2, not 23 + 9.
    """
    fit = max(1, (screen_size()[0] - SCROLL_BAR) // tile_w)
    rows = -(-max(1, n) // fit)
    return -(-max(1, n) // rows)


def tile_grid(tiles: list, cols: int) -> np.ndarray:
    """Tiles in time order, left to right then top to bottom. Every crop of an item is shown at once (no paging)."""
    th, tw = tiles[0].shape[:2]
    rows = -(-len(tiles) // cols)
    grid = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        grid[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = t
    return grid


def add_footer(img: np.ndarray, text: str) -> np.ndarray:
    """Status/key line BELOW the image, so it never covers content or the bottom scroll bar."""
    bar = np.zeros((FOOTER_H, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (6, FOOTER_H - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([img, bar])


class ZoomView:
    """Zoom and scroll for the OpenCV crop review tools. Every clip review tool should have one (owner rule).

    The window grows with zoom until it reaches the screen size (owner preference). When content is hidden past
    that, or the content is already bigger than the screen, scroll bars appear on the bottom and right edges:
    drag a thumb, or click the bar to jump there. Mouse wheel zooms keeping the point under the cursor fixed,
    right click centers the view on a spot, reset() goes back to 1x at the top left.
    Tools with their own left clicks call on_mouse() first (it returns True when it used the event, e.g. a click
    on a scroll bar) and map other clicks with to_content(). Put text in add_footer(), not over the content.
    reserve_h: pixels the caller adds above/below the zoomed content (header, footer), kept on screen too.
    """

    STEP = 1.25
    MAX = 8.0
    BAR = SCROLL_BAR

    def __init__(self, reserve_h: int = FOOTER_H):
        self.max_w, self.max_h = screen_size()
        self.max_h -= reserve_h
        self.zoom = 1.0
        self.x0 = self.y0 = 0.0  # top left of the view, in content pixels
        self.content_size = self.out_size = self.area = (1, 1)
        self.bars = (False, False)
        self.drag = None  # "h" or "v" while a scroll bar thumb is dragged

    def reset(self) -> None:
        self.zoom, self.x0, self.y0 = 1.0, 0.0, 0.0

    def _view(self) -> tuple:
        """Visible content size, in content pixels."""
        return self.area[0] / self.zoom, self.area[1] / self.zoom

    def to_content(self, mx: float, my: float) -> tuple:
        return self.x0 + mx / self.zoom, self.y0 + my / self.zoom

    def _scroll_to(self, bar: str, m: float) -> None:
        vw, vh = self._view()
        if bar == "h":
            self.x0 = m / self.area[0] * self.content_size[0] - vw / 2
        else:
            self.y0 = m / self.area[1] * self.content_size[1] - vh / 2

    def on_mouse(self, event, mx, my, flags, _param=None) -> bool:
        """Handle wheel, right click and scroll bars. Returns True if the event was used."""
        if event == cv2.EVENT_MOUSEWHEEL:
            px, py = self.to_content(mx, my)
            factor = self.STEP if flags > 0 else 1 / self.STEP
            self.zoom = float(np.clip(self.zoom * factor, 1.0, self.MAX))
            self.x0, self.y0 = px - mx / self.zoom, py - my / self.zoom  # keep the point under the cursor
            return True
        if event == cv2.EVENT_RBUTTONDOWN:
            px, py = self.to_content(mx, my)
            vw, vh = self._view()
            self.x0, self.y0 = px - vw / 2, py - vh / 2
            return True
        hbar, vbar = self.bars
        if event == cv2.EVENT_LBUTTONDOWN:
            if hbar and my >= self.area[1] and mx < self.area[0]:
                self.drag = "h"
            elif vbar and mx >= self.area[0] and my < self.area[1]:
                self.drag = "v"
            else:
                return mx >= self.area[0] or my >= self.area[1]  # the corner or a bar edge: nothing to click
            self._scroll_to(self.drag, mx if self.drag == "h" else my)
            return True
        if event == cv2.EVENT_MOUSEMOVE and self.drag:
            self._scroll_to(self.drag, mx if self.drag == "h" else my)
            return True
        if event == cv2.EVENT_LBUTTONUP and self.drag:
            self.drag = None
            return True
        return False

    def apply(self, content: np.ndarray) -> np.ndarray:
        h, w = content.shape[:2]
        self.content_size = (w, h)
        zw, zh = w * self.zoom, h * self.zoom
        # scroll bars get their own strips, so they never cover content; the window widens (or heightens) by a
        # bar's thickness when there is room, and a bar only appears when content is really hidden
        hbar = vbar = False
        for _ in range(2):
            aw = min(zw, self.max_w - (self.BAR if vbar else 0))
            ah = min(zh, self.max_h - (self.BAR if hbar else 0))
            hbar, vbar = zw > aw + 0.5, zh > ah + 0.5
        aw = int(round(min(zw, self.max_w - (self.BAR if vbar else 0))))
        ah = int(round(min(zh, self.max_h - (self.BAR if hbar else 0))))
        out_w, out_h = aw + (self.BAR if vbar else 0), ah + (self.BAR if hbar else 0)
        self.bars, self.area, self.out_size = (hbar, vbar), (aw, ah), (out_w, out_h)
        vw, vh = self._view()
        self.x0 = float(np.clip(self.x0, 0, max(0.0, w - vw)))
        self.y0 = float(np.clip(self.y0, 0, max(0.0, h - vh)))
        crop = content[int(self.y0) : int(np.ceil(self.y0 + vh)), int(self.x0) : int(np.ceil(self.x0 + vw))]
        if crop.shape[1] != aw or crop.shape[0] != ah:
            crop = cv2.resize(crop, (aw, ah), interpolation=cv2.INTER_CUBIC)
        out = np.full((out_h, out_w, 3), 40, np.uint8)
        out[:ah, :aw] = crop
        thumb = (190, 190, 190)
        if hbar:
            t0, t1 = int(self.x0 / w * aw), int((self.x0 + vw) / w * aw)
            cv2.rectangle(out, (t0, ah + 2), (max(t0 + 8, t1), out_h - 3), thumb, -1)
        if vbar:
            t0, t1 = int(self.y0 / h * ah), int((self.y0 + vh) / h * ah)
            cv2.rectangle(out, (aw + 2, t0), (out_w - 3, max(t0 + 8, t1)), thumb, -1)
        return out

    def label(self) -> str:
        return f"  zoom {self.zoom:.1f}x" if self.zoom > 1.0 else ""
