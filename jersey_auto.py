"""Automatic jersey identity for a window, no owner labeling (step 7 without the manual pass).

Why: jersey labeling took about 85 min per 5-minute window, and the owner asked for it to be automated for the whole
team. jersey_suggest.py already predicts jerseys from appearance, but only as hints, per whole tracklet, and a
tracklet often switches people partway (clipE: 40 of the owner's tracklets were split at a switch).

How: identity is predicted every SAMPLE_CI cached frames along each target/goalkeeper tracklet, not per tracklet:
  - appearance: DINOv2 features of the person crop, logistic regression trained on the labeled windows, weighted by
    closeness in time (jersey_suggest.py's model);
  - then smoothed along the tracklet (Viterbi: the identity stays the same unless the evidence changes clearly,
    SWITCH_COST), which is also where tracklets get split automatically.
Samples below MIN_CONF stay unidentified: they are left out of per-player stats rather than guessed.

`features` caches the per-sample crops' features; `evaluate` scores a labeled window left out of --from.
Outputs (git-ignored, derived from footage of minors): RUN/jersey_auto_feats.npz.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import jersey_suggest as js
from player_stats import identity_rows
from sv_common import require_under_data, tracklet_fingerprint

SAMPLE_CI = 16  # about 0.5 s at 30 fps; replayed tracklets have rows on even cached frames
ROLES = ("target", "goalkeeper")


def samples(run: Path) -> pd.DataFrame:
    """Rows of candidate target/goalkeeper tracklets every SAMPLE_CI cached frames, with box and pitch position."""
    roles = pd.read_csv(run / "tracklet_roles.csv")
    keep = roles[roles.role.isin(ROLES) & roles.player_candidate.astype(bool)].track_id
    tr = pd.read_csv(run / "best_tracklets.csv.gz", usecols=["pf", "ci", "track_id", "x1", "y1", "x2", "y2"])
    tr = tr[tr.track_id.isin(keep) & (tr.ci % SAMPLE_CI == 0)]
    xy = pd.read_csv(run / "tracklet_pitch_xy.csv.gz", usecols=["ci", "track_id", "X_m", "Y_m"])
    return tr.merge(xy, on=["ci", "track_id"], how="left").sort_values(["track_id", "ci"]).reset_index(drop=True)


def sample_features(run: Path) -> tuple:
    """(samples, features), cached in RUN/jersey_auto_feats.npz and tied to exactly these samples."""
    rows = samples(run)
    stamp = tracklet_fingerprint(rows[["ci", "track_id", "x1", "y1", "x2", "y2"]], salt=f"{js.MODEL}:{SAMPLE_CI}")
    path = run / "jersey_auto_feats.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z["stamp"]) == stamp:
            return rows, z["F"]
    print(f"{run.name}: embedding {len(rows)} samples...", flush=True)
    F = js.features(js.box_crops(run, rows))
    np.savez_compressed(path, F=F, stamp=stamp)
    return rows, F


def truth(run: Path, rows: pd.DataFrame) -> np.ndarray:
    """The owner's jersey per sample (NaN where unlabeled, split away or not identified)."""
    xy = pd.read_csv(run / "tracklet_pitch_xy.csv.gz", usecols=["pf", "ci", "track_id", "X_m", "Y_m", "anchor_gap_s"])
    ident = identity_rows(run, xy)[["ci", "track_id", "jersey"]].drop_duplicates(["ci", "track_id"])
    return rows[["ci", "track_id"]].merge(ident, on=["ci", "track_id"], how="left").jersey.to_numpy(float)


READ_CI = 4  # read numbers at about 7.5 per second along each tracklet: legible back views are brief
READER_CKPT = Path(__file__).resolve().parent / "models" / "jersey" / "parseq_soccernet.ckpt"
LEGIBILITY_PTH = Path(__file__).resolve().parent / "models" / "jersey" / "legibility_soccernet.pth"
# where the number sits in a person box (fractions of height, width). Chosen on clipE's labeled crops: the whole
# torso squashed into the reader's 4:1 input read 62% right; this back patch 89% (legible, confident, on roster)
BACK = (0.12, 0.42, 0.20, 0.80)
_reader = None


def reader(finetuned: Path | None = None):
    """(PARSeq fine-tuned on SoccerNet jerseys, ResNet-34 legibility classifier), from the jersey-number-pipeline
    project (Koshkina & Elder 2024, non-commercial licence). Weights live in models/jersey/ (git-ignored)."""
    global _reader
    if _reader is None or _reader[3] != finetuned:
        import torch
        import torchvision

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        parseq = torch.hub.load("baudm/parseq", "parseq", pretrained=False, trust_repo=True, verbose=False)
        parseq.model.load_state_dict(torch.load(READER_CKPT, map_location="cpu", weights_only=False)["state_dict"])
        if finetuned:  # fine-tuned on this game's owner-labeled crops (jersey_auto.py finetune)
            parseq.model.load_state_dict(torch.load(finetuned, map_location="cpu"))
        leg = torchvision.models.resnet34()
        leg.fc = torch.nn.Linear(512, 1)
        sd = torch.load(LEGIBILITY_PTH, map_location="cpu")
        leg.load_state_dict({k.removeprefix("model_ft."): v for k, v in sd.items()})
        _reader = (parseq.to(dev).eval(), leg.to(dev).eval(), dev, finetuned)
    return _reader


def read_numbers(crops: list, finetuned: Path | None = None) -> pd.DataFrame:
    """Per crop: legibility (0..1), the text read off the back patch, and the reader's confidence."""
    import cv2
    import torch

    parseq, leg, dev, _ = reader(finetuned)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    y0, y1, x0, x1 = BACK
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), 256):
            batch = crops[i : i + 256]
            full = np.stack([cv2.resize(c[..., ::-1], (256, 256)) for c in batch])
            x = torch.from_numpy(full).to(dev).permute(0, 3, 1, 2).float() / 255
            lg = torch.sigmoid(leg((x - mean) / std)).squeeze(1).cpu().numpy()
            backs = []
            for c in batch:
                h, w = c.shape[:2]
                b = c[int(y0 * h) : max(int(y1 * h), int(y0 * h) + 2), int(x0 * w) : max(int(x1 * w), int(x0 * w) + 2)]
                backs.append(cv2.resize(b[..., ::-1], (128, 32), interpolation=cv2.INTER_CUBIC))
            x = torch.from_numpy(np.stack(backs)).to(dev).permute(0, 3, 1, 2).float() / 255
            text, conf = parseq.tokenizer.decode(parseq((x - 0.5) / 0.5).softmax(-1))
            out += [
                dict(legibility=float(a), text=t, read_conf=float(c.prod()))
                for a, t, c in zip(lg, text, conf, strict=True)
            ]
    return pd.DataFrame(out)


def number_reads(run: Path, finetuned: Path | None = None) -> pd.DataFrame:
    """Reads for candidate target/goalkeeper tracklet rows every READ_CI cached frames, cached in
    RUN/jersey_reads.csv.gz and tied to exactly these rows."""
    roles = pd.read_csv(run / "tracklet_roles.csv")
    keep = roles[roles.role.isin(ROLES) & roles.player_candidate.astype(bool)].track_id
    tr = pd.read_csv(run / "best_tracklets.csv.gz", usecols=["ci", "track_id", "x1", "y1", "x2", "y2"])
    tr = tr[tr.track_id.isin(keep) & (tr.ci % READ_CI == 0)].sort_values(["track_id", "ci"]).reset_index(drop=True)
    tag = f"_{finetuned.stem}" if finetuned else ""
    stamp = tracklet_fingerprint(tr, salt=f"reads:{READ_CI}:{BACK}:{tag}")
    path, stamp_path = run / f"jersey_reads{tag}.csv.gz", run / f"jersey_reads{tag}.sha1"
    if path.exists() and stamp_path.exists() and stamp_path.read_text().strip() == stamp:
        return pd.read_csv(path, dtype={"text": str}, keep_default_na=False)
    print(f"{run.name}: reading numbers on {len(tr)} crops...", flush=True)
    reads = pd.concat([tr[["ci", "track_id"]], read_numbers(js.box_crops(run, tr), finetuned)], axis=1)
    reads.to_csv(path, index=False)
    stamp_path.write_text(stamp)
    return reads


def cmd_features(args) -> None:
    for r in args.runs.split(","):
        run = require_under_data(Path(r))
        rows, F = sample_features(run)
        reads = number_reads(run)
        print(f"{r}: {len(rows)} samples, {F.shape}; {len(reads)} number reads")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features", help="cache per-sample appearance features")
    f.add_argument("--runs", required=True, help="comma list of run folders")
    f.set_defaults(fn=cmd_features)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
