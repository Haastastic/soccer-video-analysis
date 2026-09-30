"""Automatic jersey identity for a window, no owner labeling (step 7 without the manual pass).

Why: jersey labeling took about 85 min per 5-minute window, and the owner asked for it to be automated for the whole
team. jersey_suggest.py already predicts jerseys from appearance, but only as hints, per whole tracklet, and a
tracklet often switches people partway (clipE: 40 of the owner's tracklets were split at a switch).

How: identity is decided every SAMPLE_CI cached frames along each target/goalkeeper tracklet, not per tracklet.
  - Evidence per sample: jersey numbers read off the back (a SoccerNet reader fine-tuned on the owner's labeled
    crops of this game, `finetune`, read on every tracklet row), appearance (DINOv2 + logistic regression) and
    position relative to the team's visible centre, both learned from the labeled windows (--from).
  - Smoothed along the tracklet (Viterbi, SWITCH_COST): the identity changes only when the evidence clearly does,
    which is also where tracklets get split. Appearance is then retrained on this window's read-confirmed samples
    only (the other half of the game looks different) and the tracklets decoded again.
  - TRUST comes from reads alone, never from summed appearance (that was overconfident: "99% sure" was right 41 to
    90% of the time): a sample is identified only if its decoded stretch has MIN_AGREE reads of that number making
    up AGREE_SHARE of the stretch's reads, and one of them is within MAX_READ_GAP_S. Everything else stays
    unidentified and is left out of per-player stats rather than guessed.
Leave-one-window-out (held-out readers, data/_jersey_exp/auto/exp9.py): 39 to 43% of the owner's labeled time
identified, right 98.4% (clipE), 99.8% (clipA), 91% (clipB: one player appears only there, so the held-out reader
never saw their number and misread it as a similar one). Players with no labeled crops carry that risk in production.

`features` caches per-sample features, `finetune` trains the reader, `identify` writes player_identity.csv and
identity_segments.csv (the same files jersey_label.py apply writes; every identified stretch is a segment).
Outputs (git-ignored, derived from footage of minors): RUN/jersey_auto_feats.npz, RUN/jersey_reads*.csv.gz.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import jersey_suggest as js
from player_stats import ROSTER_FILE, identity_rows
from sv_common import game_file, require_under_data, tracklet_fingerprint

SAMPLE_CI = 16  # about 0.5 s at 30 fps; replayed tracklets have rows on even cached frames
ROLES = ("target", "goalkeeper")
# identify: settings chosen leave-one-window-out on clipA, B, E (exp8.py, exp9.py)
PROD_READ_CI = 2  # every tracklet row: twice the reads of READ_CI, +2 to 12 points of identified time
READER_FT = Path(__file__).resolve().parent / "models" / "jersey" / "parseq_ft_game.pt"
PITCH_CAMERA = "pitch_camera.local.json"  # per game, beside its windows (sv_common.game_file); pitch_ptz.py fit
# Game facts (git-ignored): {"halftime_min": M, "first_half_our_goal_x": 0 or the pitch length}. Teams swap ends at
# halftime. When present, the goalkeeper rule uses the known end instead of guessing it from appearance.
GAME_FILE = "game.local.json"  # per game, beside its windows (sv_common.game_file)
MIN_LEGIBILITY, MIN_READ_CONF, P_READ = 0.5, 0.8, 0.9  # a read counts if both scores pass; P_READ: chance it is right
W_APP, W_SELF_APP, W_POS = 0.3, 0.6, 0.5  # evidence weights against reads (log-probability scale)
SWITCH_COST = 20.0  # switching only at box overlaps was tried and was no better (exp8.py)
MIN_AGREE, AGREE_SHARE, MAX_READ_GAP_S = 3, 0.8, 10.0  # 30 s: +5 points of time, clipE 98.4 -> 97.0% right


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


def number_reads(run: Path, finetuned: Path | None = None, read_ci: int = READ_CI) -> pd.DataFrame:
    """Reads for candidate target/goalkeeper tracklet rows every read_ci cached frames, cached in
    RUN/jersey_reads.csv.gz and tied to exactly these rows."""
    roles = pd.read_csv(run / "tracklet_roles.csv")
    keep = roles[roles.role.isin(ROLES) & roles.player_candidate.astype(bool)].track_id
    tr = pd.read_csv(run / "best_tracklets.csv.gz", usecols=["ci", "track_id", "x1", "y1", "x2", "y2"])
    tr = tr[tr.track_id.isin(keep) & (tr.ci % read_ci == 0)].sort_values(["track_id", "ci"]).reset_index(drop=True)
    tag = (f"_{finetuned.stem}" if finetuned else "") + (f"_ci{read_ci}" if read_ci != READ_CI else "")
    weights = f"{finetuned.stat().st_size}:{finetuned.stat().st_mtime_ns}" if finetuned else ""  # retrained -> reread
    stamp = tracklet_fingerprint(tr, salt=f"reads:{read_ci}:{BACK}:{tag}:{weights}")
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


def roster_numbers() -> np.ndarray:
    return np.array(sorted(pd.read_csv(ROSTER_FILE).jersey.astype(int)))


def read_evidence(rows: pd.DataFrame, reads: pd.DataFrame, roster: np.ndarray) -> tuple:
    """(usable reads with the sample each belongs to, per-sample read log-likelihoods over the roster)."""
    num = pd.to_numeric(reads.text.where(reads.text.str.fullmatch(r"\d{1,2}")), errors="coerce")
    ok = reads[(reads.legibility >= MIN_LEGIBILITY) & (reads.read_conf >= MIN_READ_CONF) & num.isin(roster)]
    ok = ok.assign(num=num[ok.index].astype(int))
    ok = ok[ok.track_id.isin(rows.track_id)].reset_index(drop=True)
    by = {t: (g.index.to_numpy(), g.ci.to_numpy()) for t, g in rows.groupby("track_id")}
    ok["i"] = [by[t][0][np.abs(by[t][1] - c).argmin()] for t, c in zip(ok.track_id, ok.ci, strict=True)]
    k = len(roster)
    lp = np.zeros((len(rows), k))
    miss, hit = np.log((1 - P_READ) / (k - 1)), np.log(P_READ)
    col = {j: n for n, j in enumerate(roster)}
    for i, j in zip(ok.i, ok.num, strict=True):
        lp[i] += miss
        lp[i, col[j]] += hit - miss
    return ok, lp - lp.max(1, keepdims=True)


def viterbi(logp: np.ndarray, cost: float) -> np.ndarray:
    """Best label path: per-sample log-likelihoods, a fixed cost per change of label."""
    n, k = logp.shape
    s, back = logp[0].copy(), np.zeros((n, k), int)
    for i in range(1, n):
        best = s.argmax()
        sw = s[best] - cost
        back[i] = np.where(s >= sw, np.arange(k), best)
        s = np.maximum(s, sw) + logp[i]
    path = np.zeros(n, int)
    path[-1] = s.argmax()
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def decode(rows: pd.DataFrame, lp: np.ndarray) -> tuple:
    """(roster index per sample, stretch id: a run of one label within one tracklet)."""
    pred, stretch, base = np.zeros(len(rows), int), np.zeros(len(rows), int), 0
    for _t, idx in rows.groupby("track_id").indices.items():
        path = viterbi(lp[idx], SWITCH_COST)
        pred[idx] = path
        s = np.r_[0, np.cumsum(path[1:] != path[:-1])]
        stretch[idx] = base + s
        base += s[-1] + 1
    return pred, stretch


def trusted(rows: pd.DataFrame, reads: pd.DataFrame, jersey: np.ndarray, stretch: np.ndarray, ci_per_s: float):
    """(trusted per sample, agreeing reads in its stretch): the read rule in the module docstring."""
    agree = jersey[reads.i.to_numpy()] == reads.num.to_numpy()
    rs = reads.assign(stretch=stretch[reads.i.to_numpy()], agree=agree)
    n_all = rs.groupby("stretch").size().reindex(range(stretch.max() + 1), fill_value=0).to_numpy()[stretch]
    n_ag = rs[rs.agree].groupby("stretch").size().reindex(range(stretch.max() + 1), fill_value=0).to_numpy()[stretch]
    gap = np.full(len(rows), np.inf)
    ci = rows.ci.to_numpy()
    for s, g in rs[rs.agree].groupby("stretch"):
        cs = np.sort(g.ci.to_numpy())
        idx = np.flatnonzero(stretch == s)
        p = np.searchsorted(cs, ci[idx])
        lo, hi = cs[(p - 1).clip(0, len(cs) - 1)], cs[p.clip(0, len(cs) - 1)]
        gap[idx] = np.minimum(np.abs(ci[idx] - lo), np.abs(ci[idx] - hi)) / ci_per_s
    ok = (n_ag >= MIN_AGREE) & (n_ag >= AGREE_SHARE * n_all) & (gap <= MAX_READ_GAP_S)
    return ok, n_ag


def appearance_logp(clf, F: np.ndarray, roster: np.ndarray) -> np.ndarray:
    col = {j: n for n, j in enumerate(roster)}
    la = np.full((len(F), len(roster)), np.log(1e-3))
    la[:, [col[c] for c in clf.classes_]] = np.log(clf.predict_proba(F) + 1e-3)
    return la - np.logaddexp.reduce(la, axis=1, keepdims=True)


def relative_xy(rows: pd.DataFrame) -> np.ndarray:
    """Pitch position relative to the median of the team's visible players at that moment."""
    cen = rows.groupby("ci")[["X_m", "Y_m"]].transform("median")
    return np.c_[rows.X_m - cen.X_m, rows.Y_m - cen.Y_m]


def position_logp(xy_train: np.ndarray, y_train: np.ndarray, xy: np.ndarray, roster: np.ndarray) -> np.ndarray:
    """Per-player Gaussian over relative position (players with under 5 labeled samples get none)."""
    out = np.full((len(xy), len(roster)), -30.0)
    xy = np.nan_to_num(xy)
    for k, j in enumerate(roster):
        Z = xy_train[y_train == j]
        if len(Z) < 5:
            continue
        S = np.cov(Z.T) + 4.0 * np.eye(2)
        d = xy - Z.mean(0)
        out[:, k] = -0.5 * np.einsum("ij,jk,ik->i", d, np.linalg.inv(S), d) - 0.5 * np.log(np.linalg.det(S))
    return out - np.logaddexp.reduce(out, axis=1, keepdims=True)


def identify(run: Path, sources: list, reader_path: Path) -> pd.DataFrame:
    """Samples with jersey (NaN where not trusted) and the number of agreeing reads behind it."""
    from sklearn.linear_model import LogisticRegression

    roster = roster_numbers()
    rows, F = sample_features(run)
    reads, lp_read = read_evidence(rows, number_reads(run, reader_path, PROD_READ_CI), roster)
    meta = json.loads((run / "cache" / "meta.json").read_text())
    ci_per_s = float(meta["cache_fps"])
    t0 = js.clip_mid_min(run)
    Xs, ys, ws, P, Y = [], [], [], [], []
    for src in sources:
        rows2, F2 = sample_features(src)
        y2 = truth(src, rows2)
        m = np.isfinite(y2)
        Xs.append(F2[m])
        ys.append(y2[m].astype(int))
        ws.append(np.full(m.sum(), np.exp(-abs(js.clip_mid_min(src) - t0) / js.TAU_MIN)))
        xy2 = relative_xy(rows2)
        m &= np.isfinite(xy2).all(1)
        P.append(xy2[m])
        Y.append(y2[m])
    w = np.concatenate(ws)
    clf = LogisticRegression(max_iter=4000, C=js.C).fit(
        np.concatenate(Xs), np.concatenate(ys), sample_weight=w / w.max()
    )
    lp_pos = position_logp(np.concatenate(P), np.concatenate(Y), relative_xy(rows), roster)
    pred, stretch = decode(rows, lp_read + W_APP * appearance_logp(clf, F, roster) + W_POS * lp_pos)
    ok, _ = trusted(rows, reads, roster[pred], stretch, ci_per_s)
    if len(np.unique(pred[ok])) >= 2:  # retrain appearance on this window's read-confirmed samples only
        own = LogisticRegression(max_iter=4000, C=js.C).fit(F[ok], roster[pred[ok]])
        pred, stretch = decode(rows, lp_read + W_SELF_APP * appearance_logp(own, F, roster) + W_POS * lp_pos)
        ok, _ = trusted(rows, reads, roster[pred], stretch, ci_per_s)
    _, n_ag = trusted(rows, reads, roster[pred], stretch, ci_per_s)
    jersey = np.where(ok, roster[pred], np.nan)
    gk_jersey, gk = goalkeeper_samples(run, rows, F, sources)
    gk &= ~ok  # a trusted read wins
    jersey[gk] = gk_jersey
    return rows.assign(jersey=jersey, agree_reads=n_ag, stretch=stretch, by_goalkeeper_rule=gk)


# The goalkeeper's back is rarely toward the camera (no reads on clipF), so the roster's one goalkeeper is found by
# role, place and look instead: team_classify's goalkeeper role, inside our penalty box, looking like our goalkeeper
# (DINOv2 + logistic regression on the labeled windows' goalkeeper vs everyone else labeled or goalkeeper-role).
# "Our end" is the end whose goalkeeper-role people look more like our goalkeeper. Leave-one-window-out
# (exp11.py): end right on all three windows; goalkeeper's labeled time found 100 / 64 / 99% (A / B / E), right
# 100 / 100 / 93%.
GK_BOX_M, GK_HALF_WIDTH_M, GK_MIN_P, GK_MIN_END_P, GK_TRACK_SHARE = 18.0, 25.0, 0.3, 0.2, 0.3
# With the end known (GAME_FILE), appearance only has to rule out the obvious: in the second half it scores our
# goalkeeper low (other light), so 0.3 kept nobody in 4 of 7 windows. clipB held out: p >= 0 found 78% at 100%.
GK_MIN_P_KNOWN_END = 0.1


def known_goal_end(run: Path, length: float) -> float | None:
    """X of the goal our goalkeeper defends in this window, from GAME_FILE (None without one)."""
    path = game_file(run, GAME_FILE)
    if not path.exists():
        return None
    game = json.loads(path.read_text())
    if game.get("first_half_our_goal_x") is None:  # a new game before new_game.py's goalkeeper vote
        return None
    first = float(game["first_half_our_goal_x"])
    second_half = js.clip_mid_min(run) > float(game["halftime_min"])
    return (length - first if first < length / 2 else 0.0) if second_half else first


# Without a goalkeeper kit class (game 3: the keeper's kit fell in with the opponents'), the goalkeeper is found by
# place and behaviour over EVERY tracklet (identity's samples hold only target/goalkeeper ones): not classed as ours
# or an official, on the pitch, within GKP_BOX_M of our goal line and inside the box's width for GKP_SHARE of its
# samples, X spread (p90 - p10) under GKP_SPREAD_M, GKP_MIN_SAMPLES or more samples; one per moment (the closest to
# our goal line). Game 1's owner windows, counting every pick on an opponent/other tracklet as wrong: 94% right,
# 62% of the goalkeeper's time found (exp13b.py; looser settings found 82% at 82 to 87% right).
GKP_BOX_M, GKP_HALF_WIDTH_M, GKP_SHARE, GKP_SPREAD_M, GKP_MIN_SAMPLES = 11.0, 20.16, 0.6, 6.0, 40


def has_keeper_kit(run: Path) -> bool:
    """Whether this window's game has a goalkeeper kit class (team_classify.py assign mapped one)."""
    from team_classify import proto_file

    path = proto_file(run)
    return path.exists() and len(json.loads(path.read_text())["roles"].get("goalkeeper", [])) > 0


def goalkeeper_by_position(run: Path) -> pd.DataFrame:
    """(track_id, ci) samples of any tracklet taken to be our goalkeeper by place and behaviour (see GKP_*); empty
    without a known goal end."""
    rows = pd.read_csv(run / "tracklet_pitch_xy.csv.gz", usecols=["ci", "track_id", "X_m", "Y_m"])
    rows = rows[rows.ci % SAMPLE_CI == 0].reset_index(drop=True)
    length = json.loads(game_file(run, PITCH_CAMERA).read_text())["length_m"]
    end = known_goal_end(run, length)
    keep = np.zeros(len(rows), bool)
    if end is None:
        return rows.iloc[:0][["track_id", "ci"]]
    roles = pd.read_csv(run / "tracklet_roles.csv").set_index("track_id")
    role = rows.track_id.map(roles.role).to_numpy()
    on_pitch = rows.track_id.map(roles.player_candidate).fillna(False).astype(bool).to_numpy()
    x, y = rows.X_m.to_numpy(), rows.Y_m.to_numpy()
    inbox = (np.abs(x - end) <= GKP_BOX_M) & (np.abs(y) <= GKP_HALF_WIDTH_M) & on_pitch
    inbox &= ~np.isin(role, ["target", "official"])
    t = pd.DataFrame(dict(track=rows.track_id.to_numpy(), inbox=inbox, x=x))
    agg = t.groupby("track").agg(
        share=("inbox", "mean"), n=("inbox", "size"), spread=("x", lambda s: s.quantile(0.9) - s.quantile(0.1))
    )
    good = agg.index[(agg.share >= GKP_SHARE) & (agg.spread <= GKP_SPREAD_M) & (agg.n >= GKP_MIN_SAMPLES)]
    idx = np.flatnonzero(inbox & np.isin(rows.track_id.to_numpy(), good))
    if len(idx):
        dist = pd.Series(np.abs(x[idx] - end), index=idx)
        keep[dist.groupby(rows.ci.to_numpy()[idx]).idxmin().to_numpy()] = True
    return rows.loc[keep, ["track_id", "ci"]]


def goalkeeper_samples(run: Path, rows: pd.DataFrame, F: np.ndarray, sources: list) -> tuple:
    """(goalkeeper's jersey, per-sample flag) for samples taken to be our goalkeeper."""
    from sklearn.linear_model import LogisticRegression

    roster = pd.read_csv(ROSTER_FILE)
    keepers = roster[roster.goalkeeper.astype(str).str.lower() == "true"].jersey.astype(int)
    none = np.zeros(len(rows), bool)
    if len(keepers) != 1:
        return None, none  # the rule assumes one goalkeeper on the roster
    gk = int(keepers.iloc[0])
    if not has_keeper_kit(run):
        return gk, none  # no goalkeeper class: cmd_identify adds goalkeeper_by_position over every tracklet
    t0 = js.clip_mid_min(run)
    X, Y, W = [], [], []
    for src in sources:
        rows2, F2 = sample_features(src)
        y2 = truth(src, rows2)
        role2 = rows2.track_id.map(pd.read_csv(src / "tracklet_roles.csv").set_index("track_id").role)
        m = np.isfinite(y2) | (role2 == "goalkeeper").to_numpy()
        X.append(F2[m])
        Y.append((y2[m] == gk).astype(int))
        W.append(np.full(m.sum(), np.exp(-abs(js.clip_mid_min(src) - t0) / js.TAU_MIN)))
    Y_all = np.concatenate(Y)
    if Y_all.sum() < 5:
        return gk, none  # the labeled windows barely show the goalkeeper
    clf = LogisticRegression(max_iter=4000, C=js.C, class_weight="balanced")
    p = clf.fit(np.concatenate(X), Y_all, sample_weight=np.concatenate(W)).predict_proba(F)[:, 1]
    length = json.loads(game_file(run, PITCH_CAMERA).read_text())["length_m"]
    role = rows.track_id.map(pd.read_csv(run / "tracklet_roles.csv").set_index("track_id").role).to_numpy()
    x, y = rows.X_m.to_numpy(), rows.Y_m.to_numpy()
    at = {0.0: (role == "goalkeeper") & (x < length / 2), length: (role == "goalkeeper") & (x >= length / 2)}
    mean_p = {end: p[m].mean() if m.any() else 0.0 for end, m in at.items()}
    end, min_p = known_goal_end(run, length), GK_MIN_P_KNOWN_END
    if end is None:  # no game file: the end whose goalkeeper-role people look more like ours
        end, min_p = max(mean_p, key=mean_p.get), GK_MIN_P
        if mean_p[end] < GK_MIN_END_P:
            return gk, none  # neither end's goalkeeper looks like ours
    cand = (role == "goalkeeper") & (np.abs(x - end) <= GK_BOX_M) & (np.abs(y) <= GK_HALF_WIDTH_M) & (p >= min_p)
    keep = np.zeros(len(rows), bool)
    idx = np.flatnonzero(cand)
    if len(idx):  # one goalkeeper per moment: the most goalkeeper-like candidate
        keep[pd.Series(p[idx], index=idx).groupby(rows.ci.to_numpy()[idx]).idxmax().to_numpy()] = True
    share = pd.Series(keep).groupby(rows.track_id.to_numpy()).transform("mean").to_numpy()
    return gk, keep & (share >= GK_TRACK_SHARE)


def segments(s: pd.DataFrame) -> pd.DataFrame:
    """identity_segments.csv rows: runs of consecutive trusted samples of one jersey within a tracklet, each
    widened by half a sample step on both sides."""
    s = s[s.jersey.notna()].sort_values(["track_id", "ci"])
    new = (s.track_id.ne(s.track_id.shift()) | s.jersey.ne(s.jersey.shift()) | s.ci.diff().ne(SAMPLE_CI)).cumsum()
    g = s.groupby(new)
    out = pd.DataFrame(
        dict(
            track_id=g.track_id.first(),
            ci_start=(g.ci.min() - SAMPLE_CI // 2).clip(lower=0),
            ci_end=g.ci.max() + SAMPLE_CI // 2 - 1,
            jersey=g.jersey.first().astype(int),
            agree_reads=g.agree_reads.max(),
        )
    ).reset_index(drop=True)
    out.insert(1, "seg", out.groupby("track_id").cumcount())
    return out


def cmd_identify(args) -> None:
    run = require_under_data(args.run)
    sources = [require_under_data(Path(r)) for r in args.sources.split(",")]
    if run.resolve() in {s.resolve() for s in sources}:  # compare resolved: same folder, any spelling
        raise SystemExit("--from must not include --run: its own owner labels would leak into the evidence")
    s = identify(run, sources, args.reader)
    seg_path = run / "identity_segments.csv"
    auto = seg_path.exists() and (pd.read_csv(seg_path).player_id.astype(str) == "auto").any()
    owner_made = (run / "player_identity.csv").exists() and not auto  # never score our own output
    y = truth(run, s) if owner_made and not args.write else None
    if y is not None and np.isfinite(y).any():
        m, t = np.isfinite(y), s.jersey.notna().to_numpy()
        print(
            f"{run.name} vs owner labels: identified {100 * (m & t).sum() / m.sum():.1f}% of labeled samples, "
            f"right {100 * (s.jersey.to_numpy()[m & t] == y[m & t]).mean():.1f}%"
        )
    ident = s.jersey.notna()
    print(
        f"{run.name}: {100 * ident.mean():.1f}% of target/goalkeeper samples identified, {s.jersey.nunique()} players"
    )
    if not args.write:
        print("(dry run: pass --write to write player_identity.csv and identity_segments.csv)")
        return
    if (run / "jersey_truth.csv").exists() and not args.force:
        raise SystemExit(f"{run} has owner jersey labels (jersey_truth.csv); pass --force to replace their outputs")
    roster = pd.read_csv(ROSTER_FILE)
    s_seg = s
    if not has_keeper_kit(run):  # the goalkeeper by place, from every tracklet; reads and other names win
        gk = roster[roster.goalkeeper.astype(str).str.lower() == "true"].jersey.astype(int)
        k = goalkeeper_by_position(run)
        named = s[s.jersey.notna()]
        k = k[~k.track_id.isin(named.track_id) & ~k.ci.isin(named[named.jersey == int(gk.iloc[0])].ci)]
        if len(gk) == 1 and len(k):
            s_seg = pd.concat([s, k.assign(jersey=float(gk.iloc[0]), agree_reads=0)], ignore_index=True)
            print(f"{run.name}: goalkeeper by place and behaviour (no goalkeeper kit class): {len(k)} samples")
    segs = segments(s_seg).merge(roster[["jersey", "name", "goalkeeper"]], on="jersey", how="left")
    segs.insert(4, "player_id", "auto")
    roles = pd.read_csv(run / "tracklet_roles.csv")[["track_id", "role"]]
    pi = roles.assign(player_id="", jersey=np.nan, name="", goalkeeper="")
    pi["split_at_switch"] = pi.track_id.isin(segs.track_id)  # identity lives in the segments, never whole
    pi = pi[["track_id", "player_id", "role", "jersey", "split_at_switch", "name", "goalkeeper"]]
    pi.to_csv(run / "player_identity.csv", index=False)
    segs.to_csv(run / "identity_segments.csv", index=False)
    print(f"wrote player_identity.csv and identity_segments.csv ({len(segs)} identified stretches)")


# Fine-tuning: the owner's labeled crops of this game. A wider patch than BACK is kept per crop so training can
# jitter the back crop inside it. Held out on each labeled window, reads went 74-87% -> 90-98% right.
WIDE = (0.05, 0.50, 0.10, 0.90)
WIDE_PX = (192, 96)
FT_EPOCHS, FT_LR, FT_BATCH = 8, 2e-5, 48
RARE_TRUTH = Path(__file__).resolve().parent / "data" / "jersey_rare_truth.csv"  # jersey_rare_label.py
RARE_REPEAT = 4  # each confirmed rare crop is seen this many times per epoch (jittered differently each time)


def labeled_patches(run: Path) -> tuple:
    """(wide patches, owner jersey) for read crops the legibility model half-accepts, cached in
    RUN/jersey_ft_patches.npz."""
    path = run / "jersey_ft_patches.npz"
    # tied to the owner's labels and the tracklets: relabeling (e.g. --redo-mixed) rebuilds the patches
    labels = [run / n for n in ("player_identity.csv", "identity_segments.csv", "best_tracklets.csv.gz")]
    stamp = hashlib.sha1(b"".join(p.read_bytes() for p in labels if p.exists())).hexdigest()
    if path.exists():
        z = np.load(path)
        if "stamp" in z and str(z["stamp"]) == stamp:
            return z["P"], z["y"]
    reads = number_reads(run)
    xy = pd.read_csv(run / "tracklet_pitch_xy.csv.gz", usecols=["pf", "ci", "track_id", "X_m", "Y_m", "anchor_gap_s"])
    ident = identity_rows(run, xy)[["ci", "track_id", "jersey"]].drop_duplicates(["ci", "track_id"])
    d = reads.merge(ident, on=["ci", "track_id"])
    d = d[d.legibility >= 0.3].reset_index(drop=True)
    P, y = wide_patches(run, d[["ci", "track_id"]]), d.jersey.to_numpy(int)
    np.savez_compressed(path, P=P, y=y, stamp=stamp)
    return P, y


def wide_patches(run: Path, keys: pd.DataFrame) -> np.ndarray:
    """WIDE patches, in row order, for rows with a box (ci, x1..y2) or for (ci, track_id) rows of this window's
    current tracklets (every row must exist there)."""
    import cv2

    if {"x1", "y1", "x2", "y2"} <= set(keys.columns):
        rows = keys[["ci", "x1", "y1", "x2", "y2"]]
    else:
        tr = pd.read_csv(run / "best_tracklets.csv.gz", usecols=["ci", "track_id", "x1", "y1", "x2", "y2"])
        rows = keys[["ci", "track_id"]].merge(tr, on=["ci", "track_id"], how="left")
        if rows.x1.isna().any():  # a missing box would silently crop the whole frame
            raise SystemExit(f"{run}: {int(rows.x1.isna().sum())} rows are not in best_tracklets.csv.gz")
    y0, y1, x0, x1 = WIDE
    P = []
    for c in js.box_crops(run, rows):
        h, w = c.shape[:2]
        p = c[int(y0 * h) : max(int(y1 * h), int(y0 * h) + 2), int(x0 * w) : max(int(x1 * w), int(x0 * w) + 2)]
        P.append(cv2.resize(p, WIDE_PX, interpolation=cv2.INTER_CUBIC))
    return np.stack(P)


def confirmed_rare_patches(truth_csv: Path) -> tuple:
    """(wide patches, jersey) for crops the owner confirmed in jersey_rare_label.py. The truth file keeps each
    crop's box, so the crop is the one the owner saw even if the tracker is rerun and track ids change."""
    if not truth_csv.exists():
        return np.zeros((0, WIDE_PX[1], WIDE_PX[0], 3), np.uint8), np.zeros(0, int)
    t = pd.read_csv(truth_csv)
    t = t[t.shows_number.astype(bool)]
    Ps, ys = [], []
    for run_name, g in t.groupby("run"):
        Ps.append(wide_patches(require_under_data(truth_csv.parent / run_name), g))
        ys.append(g.number.to_numpy(int))
    if not Ps:
        return np.zeros((0, WIDE_PX[1], WIDE_PX[0], 3), np.uint8), np.zeros(0, int)
    return np.concatenate(Ps), np.concatenate(ys)


def back_batch(P: np.ndarray, rng, jitter: bool):
    """BACK patches cut from wide patches, as the reader's input tensor; with jitter: box and lighting noise."""
    import cv2
    import torch

    (wy0, wy1, wx0, wx1), (pw, ph) = WIDE, WIDE_PX
    out = []
    for im in P:
        y0, y1, x0, x1 = BACK
        if jitter:
            y0, y1 = y0 + rng.uniform(-0.04, 0.04), y1 + rng.uniform(-0.04, 0.04)
            x0, x1 = x0 + rng.uniform(-0.06, 0.06), x1 + rng.uniform(-0.06, 0.06)
        ry0, ry1 = (int(np.clip((v - wy0) / (wy1 - wy0), 0, 1) * ph) for v in (y0, y1))
        rx0, rx1 = (int(np.clip((v - wx0) / (wx1 - wx0), 0, 1) * pw) for v in (x0, x1))
        b = im[ry0 : max(ry1, ry0 + 2), rx0 : max(rx1, rx0 + 2)][..., ::-1]
        b = cv2.resize(b, (128, 32), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        if jitter:
            b = np.clip(b * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25) + rng.normal(0, 4, b.shape), 0, 255)
        out.append(b)
    return (torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float() / 255 - 0.5) / 0.5


def cmd_finetune(args) -> None:
    import torch

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    data = [labeled_patches(require_under_data(Path(r))) for r in args.runs.split(",")]
    Pr, Yr = confirmed_rare_patches(args.rare)
    if len(Yr):  # repeated so a few confirmed crops weigh like a labeled player's
        rep = int(np.clip(RARE_REPEAT, 1, None))
        data.append((np.repeat(Pr, rep, axis=0), np.repeat(Yr, rep)))
        print(
            f"adding {len(Yr)} owner-confirmed crops of rare numbers (x{rep}): {pd.Series(Yr).value_counts().to_dict()}"
        )
    P, Y = np.concatenate([d[0] for d in data]), np.concatenate([d[1] for d in data])
    parseq, _leg, dev, _ = reader()
    parseq.log = lambda *a, **k: None
    opt = torch.optim.AdamW(parseq.parameters(), lr=FT_LR, weight_decay=0.01)
    print(f"fine-tuning on {len(Y)} labeled crops from {args.runs}")
    for ep in range(FT_EPOCHS):
        parseq.train()
        perm, tot = rng.permutation(len(Y)), 0.0
        for i in range(0, len(perm), FT_BATCH):
            b = perm[i : i + FT_BATCH]
            loss = parseq.training_step((back_batch(P[b], rng, True).to(dev), [str(v) for v in Y[b]]), 0)
            loss = loss["loss"] if isinstance(loss, dict) else loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        print(f"  epoch {ep} loss {tot / len(Y):.3f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(parseq.model.state_dict(), args.out)
    print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features", help="cache per-sample appearance features")
    f.add_argument("--runs", required=True, help="comma list of run folders")
    f.set_defaults(fn=cmd_features)
    f = sub.add_parser("finetune", help="fine-tune the jersey reader on owner-labeled windows")
    f.add_argument("--runs", required=True, help="comma list of labeled run folders")
    f.add_argument("--out", type=Path, default=READER_FT)
    f.add_argument("--rare", type=Path, default=RARE_TRUTH, help="owner-confirmed rare crops (jersey_rare_label.py)")
    f.set_defaults(fn=cmd_finetune)
    f = sub.add_parser("identify", help="identify players in a window from reads, appearance and position")
    f.add_argument("--run", type=Path, required=True)
    f.add_argument("--from", dest="sources", required=True, help="comma list of owner-labeled run folders")
    f.add_argument("--reader", type=Path, default=READER_FT, help="fine-tuned reader weights (finetune --out)")
    f.add_argument("--write", action="store_true", help="write player_identity.csv and identity_segments.csv")
    f.add_argument("--force", action="store_true", help="replace outputs of a window the owner labeled")
    f.set_defaults(fn=cmd_identify)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
