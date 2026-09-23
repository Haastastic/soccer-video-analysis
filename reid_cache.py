"""Cache a person re-identification (appearance) embedding for every person detection, once.

replay_trackers.py --reid passes these to BoT-SORT as precomputed features, so appearance matching can be swept
offline like everything else, with no video or model in the loop. Only the frames a replay at --fps uses are
embedded (plus nothing else), because embedding every cached frame would double the time for no benefit.

The default model is Ultralytics' yolo26s-reid.onnx, a generic person ReID model downloaded on first use. It runs
on CPU through onnxruntime (about 0.1 to 0.2 s per frame, about 15 min per 5 minute clip at 15 fps). The GPU
build of onnxruntime needs a different CUDA version than this environment's PyTorch, so it is not used.

Caution: generic ReID models are trained on pedestrians, not on same-kit players at 60 to 90 px, so the
embeddings mostly separate the two teams and officials. Whether they help within a team is what the sweep checks.

Output: RUN/cache/reid_<model stem>_<fps>fps.npz with det_idx (row index into detections.csv.gz) and emb
(float16, L2-normalized).

Example:
  python reid_cache.py --run data\\clipA
"""

import argparse
import time
from pathlib import Path

import numpy as np

from sv_common import PERSON, Cache, read_frames, require_under_data


def reid_path(run: Path, model: str, fps: float) -> Path:
    return run / "cache" / f"reid_{Path(model).stem}_{fps:g}fps.npz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=require_under_data, help="run folder with cache/ and clip.mp4")
    ap.add_argument("--model", default="yolo26s-reid.onnx")
    ap.add_argument("--fps", type=float, default=15.0, help="replay rate the embeddings are for")
    ap.add_argument("--person-floor", type=float, default=0.1, help="same floor replay_trackers.py uses")
    args = ap.parse_args()

    from ultralytics.trackers.utils.reid import ReID

    cache = Cache(args.run / "cache")
    idxs, fps = cache.processed_indices(args.fps)
    dest = reid_path(args.run, args.model, round(fps, 3))
    persons = cache.det[(cache.det.cls == PERSON) & (cache.det.conf >= args.person_floor) & cache.det.ci.isin(idxs)]
    by_ci = {ci: g for ci, g in persons.groupby("ci")}
    enc = ReID(args.model)
    stride = int(cache.meta.get("stride", 1))
    det_idx, embs = [], []
    t0 = time.time()
    for k, (f, img) in enumerate(read_frames(args.run / "clip.mp4", [ci * stride for ci in by_ci])):
        g = by_ci[f // stride]
        xyxy = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
        xywh = np.c_[(xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]]
        for i, e in zip(g.index, enc(img, xywh), strict=True):
            if e is not None:
                det_idx.append(i)
                embs.append(e / max(np.linalg.norm(e), 1e-9))
        if k % 500 == 0:
            print(f"{k}/{len(by_ci)} frames, {len(det_idx)} embeddings ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(dest, det_idx=np.asarray(det_idx), emb=np.asarray(embs, dtype=np.float16))
    print(f"Wrote {dest}: {len(det_idx)} of {len(persons)} detections embedded in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
