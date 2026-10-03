"""Process a new game end to end: plan the live-play windows, set the game up, run every window, write tips.

A game lives in data/<game>/: its windows (data/<game>/w0500 = 5-minute window from 00:05:00), and the files
that belong to the game rather than the pipeline (sv_common.game_file): game.local.json (the halves, halftime,
which goal we defend), pitch_camera.local.json and kit_prototypes.local.json.

  1. plan   Density scan of the video, then a proposal: first half, halftime, second half and any water breaks
            (in hot games there is one midway through each half: a stoppage, not a change of ends). Halftime is
            the longest break near the middle of the video. The end of the game cannot be found from the video
            alone (other teams often take the pitch right after), so it is proposed from the first half's length
            and shown on a contact sheet (data/<game>/live_play_check.jpg) for the owner to check. Writes
            game.local.json with "confirmed": false.
  2. confirm  Accept the proposal, or correct it: --first-half 00:00:00-00:40:30 --second-half 00:43:30-01:21:00.
  3. setup  A pilot window through the pipeline. The pitch camera is refitted from the painted lines when the
            previous game's camera fixes too few frames (the tripod moves a little between games; pitch_ptz.py
            refit); if the refit does not recover, the owner places a few anchors (printed instructions). Kit
            colours change between games (home and away, a new opponent), so the pilot's kit clusters are written
            with a review montage and the command stops for the mapping (team_classify.py assign; one command).
  4. run    Every window of both halves (run_windows.py), then which goal we defend (a vote of where our
            goalkeeper was found in the first-half windows, cross-checked against which half our players stand in
            at the second-half kickoff in a 2-minute probe clip around the restart; if they disagree the run stops
            for the owner), identity again with that
            known, player stats and the coaching pages (data/<game>/coaching/, local only).

  5. numbers  The per-game jersey round (about 5 min of the owner's time): candidate crops of numbers whose reads
            conflict or are rarely trusted in this game, confirmed with jersey_rare_label.py label; then
            `numbers --apply` retrains the reader and redoes identity, stats and pages (CLAUDE.md Phase 17).
            Optional: worth it when a number is badly under-trusted (game 3: +1.6% identified time).
            A new team's first game needs it: `numbers --seed` shows every roster number (about 15-20 min), since
            the reader has not seen that team's numbers. Set "team" in game.local.json before `setup` (see
            sv_common's teams note; the roster goes in data/teams/<team>/roster.csv). A game's own roster (the
            players listed for that game) goes in data/<game>/roster.csv and wins over the team's.
  6. publish  Put every game on the coaching site (webapp/README.md): export the pages' data for all games
            (site_export.py), offer the photo picker for players who have no photo decision yet (player_photo.py;
            players already chosen or skipped are left alone), then upload (publish_site.py lists the files and
            asks first). The bucket comes from --bucket or SITE_BUCKET.

Each step is resumable: rerun it and finished work is skipped. Everything written stays under data/ (git-ignored),
including the contact sheet: it shows minors.

Example (PowerShell):
  python new_game.py plan --video "videos\\<game>.mp4" --game g0922
  python new_game.py confirm --game g0922
  python new_game.py setup --game g0922
  python new_game.py run --game g0922
  python new_game.py publish --bucket <bucket>
"""

import argparse
import json
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from run_windows import stages
from sv_common import DATA_DIR, DEFAULT_TEAM, game_dir, parse_time, read_roster, require_under_data, team_of

HERE = Path(__file__).resolve().parent
WINDOW_S = 300
ACTIVE_PEOPLE = 12  # a density sample shows play with at least this many people
SMOOTH_SAMPLES = 5  # rolling median over this many samples (20 s apart: 100 s)
MIN_BREAK_S = 60  # a stretch without play this long is a break
MIN_TAIL_S = 120  # a half's last partial window is kept if it covers this much play
PILOT_FIXED_PCT, PILOT_MAX_GAP_S = 70.0, 20.0  # a pilot below these is refitted (see pitch_ok)
PREVIOUS_CAMERA = DATA_DIR / "pitch_camera.local.json"  # the first game's
LABELED = "data/clipA,data/clipB,data/clipE"  # owner-labeled windows: the jersey reader and appearance learn here


def hms(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def game_folder(game: str) -> Path:
    return require_under_data(DATA_DIR / game)


def load_game(game: str) -> dict:
    path = game_folder(game) / "game.local.json"
    if not path.exists():
        raise SystemExit(f"No {path}: run `plan` first.")
    return json.loads(path.read_text())


def save_game(game: str, g: dict) -> None:
    (game_folder(game) / "game.local.json").write_text(json.dumps(g, indent=2))


def run(cmd: list, log=None) -> None:
    print("  " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([sys.executable, "-u", *[str(c) for c in cmd]], check=True, cwd=HERE, stdout=log, stderr=log)


# ------------------------------------------------------------------------------------------------ plan


def propose(scan: pd.DataFrame) -> dict | None:
    """Halves, halftime and water breaks from the density scan."""
    t = scan.time_s.to_numpy()
    step = float(np.median(np.diff(t)))
    active = (scan.people.rolling(SMOOTH_SAMPLES, center=True, min_periods=1).median() >= ACTIVE_PEOPLE).to_numpy()
    breaks, start = [], None  # runs of samples without play
    for i, a in enumerate(active):
        if not a and start is None:
            start = i
        if (a or i == len(active) - 1) and start is not None:
            end = i if a else i + 1
            if (end - start) * step >= MIN_BREAK_S:
                breaks.append((t[start], t[end - 1] + step))
            start = None
    first_play = t[np.argmax(active)] if active.any() else 0.0
    breaks = [b for b in breaks if b[0] > first_play]  # before kick-off is not a break
    video_end = t[-1] + step
    middle = [b for b in breaks if 0.3 * video_end < (b[0] + b[1]) / 2 < 0.7 * video_end]
    if not middle:  # e.g. the camera was paused over halftime (game 3): the owner picks the halves from an overview
        return None
    half_break = max(middle, key=lambda b: b[1] - b[0])
    h1 = (first_play, half_break[0])
    h2_start = half_break[1]
    # the end: about the first half's length after the restart (plus a little stoppage time), never past the
    # video; the owner checks it on the contact sheet
    h2 = (h2_start, min(video_end, h2_start + (h1[1] - h1[0]) + 60))
    water = [b for b in breaks if b != half_break and (h1[0] < b[0] < h1[1] or h2[0] < b[0] < h2[1])]
    return dict(first_half=[hms(h1[0]), hms(h1[1])], second_half=[hms(h2[0]), hms(h2[1])],
                halftime_min=round((half_break[0] + half_break[1]) / 2 / 60, 1),
                proposed_breaks=[[hms(a), hms(b)] for a, b in water])  # fmt: skip


def contact_sheet(video: Path, g: dict, dest: Path) -> None:
    """Small thumbnails at every boundary, labelled, for the owner to check (local only: shows minors)."""
    h1 = [parse_time(x) for x in g["first_half"]]
    h2 = [parse_time(x) for x in g["second_half"]]
    # Warm-ups at halftime and other teams after the game look like play to the density scan (game 1: the proposed
    # restart came 3 min before kick-off), so the sheet steps through both boundaries a minute at a time.
    marks = [(h1[0] + 30, "1st half start"), (h1[1] - 30, "1st half end -30s")]
    marks += [(t, "halftime") for t in np.arange(h1[1] + 60, h2[0], 60)]
    marks += [(h2[0] + k * 60, f"2nd half start {k:+d}min" if k else "2nd half start") for k in range(0, 4)]
    marks += [(h2[1] + k * 60, f"2nd half end {k:+d}min" if k else "2nd half end") for k in range(-3, 4)]
    # the scan's guesses only, shown for the owner; real water breaks (not play) come from the play check in `run`
    marks += [((parse_time(a) + parse_time(b)) / 2, "break?") for a, b in g.get("proposed_breaks", [])]
    cap = cv2.VideoCapture(str(video))
    tiles = []
    for t, label in marks:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
        ok, f = cap.read()
        f = cv2.resize(f, (320, 180)) if ok else np.zeros((180, 320, 3), np.uint8)
        cv2.rectangle(f, (0, 0), (320, 24), (0, 0, 0), -1)
        cv2.putText(f, f"{hms(t)} {label}", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(f)
    while len(tiles) % 3:
        tiles.append(np.zeros((180, 320, 3), np.uint8))
    sheet = np.vstack([np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3)])
    cv2.imwrite(str(dest), sheet)


OVERVIEW_STEP_S = 120


def overview_sheet(video: Path, dest: Path, step_s: float = OVERVIEW_STEP_S) -> None:
    """A labelled thumbnail every step_s over the whole video (local only: shows minors), for picking the halves
    by eye when the density scan shows no halftime break."""
    cap = cv2.VideoCapture(str(video))
    end = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
    tiles = []
    for t in np.arange(0, end, step_s):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        f = cv2.resize(f, (320, 180)) if ok else np.zeros((180, 320, 3), np.uint8)
        cv2.rectangle(f, (0, 0), (320, 24), (0, 0, 0), -1)
        cv2.putText(f, hms(t), (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(f)
    while len(tiles) % 5:
        tiles.append(np.zeros((180, 320, 3), np.uint8))
    cv2.imwrite(str(dest), np.vstack([np.hstack(tiles[i : i + 5]) for i in range(0, len(tiles), 5)]))


def cmd_plan(args) -> None:
    folder = game_folder(args.game)
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / "density_scan.csv").exists():
        run(["scan_density.py", "--video", args.video, "--out", folder])
    proposal = propose(pd.read_csv(folder / "density_scan.csv"))
    if proposal is None:
        save_game(args.game, dict(video=str(args.video), proposed_breaks=[], confirmed=False))
        overview_sheet(Path(args.video), folder / "overview_check.jpg")
        print(f"No break near the middle of the video. Check {folder / 'overview_check.jpg'} (a frame every "
              f"{OVERVIEW_STEP_S // 60} min, local only) and pass the halves to `confirm --first-half A-B "
              "--second-half C-D`.")  # fmt: skip
        return
    g = dict(video=str(args.video), **proposal, confirmed=False)
    save_game(args.game, g)
    contact_sheet(Path(args.video), g, folder / "live_play_check.jpg")
    print(json.dumps(g, indent=2))
    print(f"\nCheck {folder / 'live_play_check.jpg'} (local only), then `confirm` (with corrections if needed).")


def cmd_confirm(args) -> None:
    g = load_game(args.game)
    before = windows(g)  # the windows as they are, before any correction of the halves
    if args.first_half:
        g["first_half"] = args.first_half.split("-")
    if args.second_half:
        g["second_half"] = args.second_half.split("-")
    if (args.first_half or args.second_half) and any((game_folder(args.game) / n).exists() for n, _ in before):
        g.setdefault("windows", before)  # already processed: keep those windows, stats keep to the new halves
    if args.first_half or args.second_half:
        g["halftime_min"] = round((parse_time(g["first_half"][1]) + parse_time(g["second_half"][0])) / 2 / 60, 1)
    g["confirmed"] = True
    save_game(args.game, g)
    print(json.dumps(g, indent=2))


# ----------------------------------------------------------------------------------------------- setup


def windows(g: dict) -> list:
    """(name, start) 5-minute windows over both halves; a half's last window ends at the half's end. A game whose
    halves were corrected after its windows were processed keeps those windows ("windows" in the game file): stats
    count only the time inside the halves (sv_common.play_mask)."""
    if g.get("windows"):
        return [tuple(w) for w in g["windows"]]
    out = []
    for key in ("first_half", "second_half"):
        a, b = (parse_time(x) for x in g[key])
        starts = list(np.arange(a, b - WINDOW_S + 1, WINDOW_S))
        covered = starts[-1] + WINDOW_S if starts else a
        if b - covered >= MIN_TAIL_S:
            starts.append(b - WINDOW_S)
        out += [(f"w{int(s) // 60:02d}{int(s) % 60:02d}", hms(s)) for s in starts]
    return out


# GPU-heavy stages share the laptop GPU (6 GB) in units: two detections at once barely slow each other (one unit
# each), but an identity run (jersey reader + appearance model) next to two detections filled GPU memory and ran at a
# tenth of its speed, with both detections at a quarter of theirs (Varsity game 1: 3 fps instead of 13). So identity
# takes both units: it runs alone on the GPU, and detections wait for it.
GPU_UNITS = threading.Semaphore(2)
GPU_TAKE = threading.Lock()  # a stage needing both units takes them together (no two half-held claims)
UNITS = {"run_all.py": 1, "ball_finetune.py": 1, "jersey_auto.py": 2}


class gpu_units:
    def __init__(self, n: int):
        self.n = n

    def __enter__(self):
        with GPU_TAKE:
            for _ in range(self.n):
                GPU_UNITS.acquire()

    def __exit__(self, *exc):
        for _ in range(self.n):
            GPU_UNITS.release()


def team_ready(team: str) -> bool:
    """Whether the jersey reader has learned this team: owner-labeled windows of the team, or owner-confirmed crops
    of its numbers (a seed round). Before that, identity is skipped in `run` (it would be redone after the seed)."""
    if any(team_of(Path(p)) == team for p in LABELED.split(",")):
        return True
    truth = DATA_DIR / "jersey_rare_truth.csv"
    if not truth.exists():
        return False
    t = pd.read_csv(truth)
    runs = t[t.shows_number.astype(bool)].run.unique()
    return any((DATA_DIR / r).exists() and team_of(DATA_DIR / r) == team for r in runs)


def run_stages(video: Path, game: str, name: str, start: str, upto: str | None = None, identity: bool = True) -> Path:
    run_dir = game_folder(game) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    labeled = [Path(p).resolve() for p in LABELED.split(",")]
    with open(game_folder(game) / f"{name}_window.log", "a", encoding="utf-8") as log:
        for done, cmd in stages(video, run_dir.resolve(), start, labeled):
            if upto and cmd[0] == upto:
                break
            if cmd[0] == "jersey_auto.py" and not identity:
                continue
            if not done.exists():
                with gpu_units(UNITS[cmd[0]]) if cmd[0] in UNITS else nullcontext():
                    run(cmd, log)
    return run_dir


def pitch_ok(run_dir: Path) -> tuple:
    """(good enough, summary). Gaps between fixes are bridged by the cached camera motion, so what matters is
    that fixes are frequent and no gap is long: game 2's pilot fixed 77% (fainter lines in its light), longest gap
    14 s, and its noise floor (25.5 m/min) sat inside game 1's (16.8 to 28.7)."""
    rep = json.loads((run_dir / "pitch_ptz_report.json").read_text())
    pct, gap = float(rep["fixed_pct"]), float(rep["longest_gap_between_fixes_s"])
    return pct >= PILOT_FIXED_PCT and gap <= PILOT_MAX_GAP_S, f"{pct:.0f}% of frames fixed, longest gap {gap:.0f} s"


# Mapping kits from jersey reads does not work: the reader is fine-tuned on our numbers and leans toward them on any
# shirt, and opponents wear the same low numbers (9/3 pilots: opponent clusters read as our numbers 47 to 82% of the
# time). Colour against a previous game of the same team does, when the team wears the same kit (home games).
KIT_MATCH = 18.0  # Lab distance (torso + legs) under which a cluster is taken to be the reference game's kit
KIT_TEAM_ROWS = 0.10  # an unmatched cluster with at least this share of the rows, mostly on the pitch, is the opponent


def kits_like(pilot: Path, ref_game: str) -> dict:
    """cluster -> role for a pilot's colour clusters, by the nearest prototype of a previous game of the same team:
    near our kit -> target, near its other roles -> that role, else a big on-pitch cluster -> opponent, else other."""
    cl = json.loads((pilot / "kit_clusters.json").read_text())
    ref = json.loads((game_folder(ref_game) / "kit_prototypes.local.json").read_text())["roles"]
    protos = [(np.array(c), role) for role, cs in ref.items() for c in cs if role != "opponent"]
    rows = pd.read_csv(pilot / "best_tracklets.csv.gz", usecols=["track_id"]).track_id.value_counts()
    lab = pd.Series(cl["labels"], index=cl["track_ids"])
    total = rows.reindex(lab.index).fillna(0).sum()
    out = {}
    for k, c in enumerate(cl["centers"]):
        d, role = min((float(np.linalg.norm(np.array(c) - pc)), r) for pc, r in protos)
        members = lab.index[lab == k]
        share = rows.reindex(members).fillna(0).sum() / max(total, 1)
        if d <= KIT_MATCH and (role != "target" or share >= KIT_TEAM_ROWS):  # our team is never a sliver of the rows
            out[k] = role
        elif share >= KIT_TEAM_ROWS:
            out[k] = "opponent"
        else:
            out[k] = "other"
        print(f"  cluster {k}: nearest {role} at {d:.1f}, {100 * share:.0f}% of rows -> {out[k]}")
    return out


def cmd_setup(args) -> None:
    g = load_game(args.game)
    if not g.get("confirmed"):
        raise SystemExit("The halves are not confirmed yet: check the contact sheet, then `confirm`.")
    folder, video = game_folder(args.game), Path(g["video"])
    name, start = windows(g)[1] if len(windows(g)) > 1 else windows(g)[0]  # the second window: play is settled
    camera = folder / "pitch_camera.local.json"
    if not camera.exists():
        shutil.copy(PREVIOUS_CAMERA, camera)
    # kits first: classifying needs this game's kit colours, so stop before it until they exist. Colours drift
    # between halves as the light changes (both games so far: the first half's prototypes read many second-half
    # players as goalkeeper or unknown), so a pilot in each half is mapped; `assign` adds to the same file.
    kits = folder / "kit_prototypes.local.json"
    second = [w for w in windows(g) if parse_time(w[1]) >= parse_time(g["second_half"][0])]
    pilots = [(name, start)] + ([second[len(second) // 2]] if second else [])
    for pname, pstart in pilots:
        p = run_stages(video, args.game, pname, pstart, upto="team_classify.py")
        clusters = p / "kit_clusters.json"
        mapped = clusters.exists() and kits.exists() and kits.stat().st_mtime > clusters.stat().st_mtime
        if not mapped:
            if not clusters.exists():
                run(["team_classify.py", "calibrate", "--run", p])
            if args.kits_like:  # the owner away: map by colour against a previous game of the same team
                print(f"kit clusters of {pname} against {args.kits_like}:")
                m = kits_like(p, args.kits_like)
                if "target" not in m.values():
                    raise SystemExit("No cluster is near our kit in that game: map the kits by hand (see the montage).")
                run(["team_classify.py", "assign", "--run", p, "--map", ",".join(f"{c}:{r}" for c, r in m.items())])
                g.setdefault("kits_note", []).append(f"{pname}: mapped by colour like {args.kits_like}: {m}")
                save_game(args.game, g)
                continue
            raise SystemExit(
                f"Kit colours ({pname}): look at {p / 'kit_clusters.png'} (local only) and map the clusters:\n"
                f"  python team_classify.py assign --run {p} --map 0:target,1:opponent,...\n"
                f"(it adds to {kits}), then run `setup` again."
            )
        roles = p / "tracklet_roles.csv"
        if roles.exists() and roles.stat().st_mtime < kits.stat().st_mtime:
            # classified with other kit colours before: redo roles and what depends on them
            for stale in ("tracklet_roles.csv", "events.csv", "identity_segments.csv", "player_identity.csv"):
                (p / stale).unlink(missing_ok=True)
    pilot = game_folder(args.game) / name
    # then the pitch: the pilot through the camera fit, not identity yet
    run_stages(video, args.game, name, start, upto="pitch_calibrate.py")
    ok, summary = pitch_ok(pilot)
    print(f"pilot {name}: {summary} with the current camera")
    if not ok and not json.loads(camera.read_text()).get("refit"):
        run(["pitch_ptz.py", "refit", "--runs", pilot, "--camera", camera, "--out", camera])
        for stale in ("pitch_anchors_ptz.local.json", "pitch_ptz_report.json", "tracklet_pitch_xy.csv.gz"):
            (pilot / stale).unlink(missing_ok=True)
        run_stages(video, args.game, name, start, upto="pitch_calibrate.py")
        ok, summary = pitch_ok(pilot)
        print(f"after the refit: {summary}")
    if not ok:
        raise SystemExit(
            "The camera still fixes too few frames. Place a few owner anchors in two windows "
            f"(pitch_anchor_ui.py --run {pilot}), then pitch_ptz.py fit --anchors <those windows> --out {camera}."
        )
    run_stages(video, args.game, name, start)
    print(f"setup done: pilot pitch {summary}, kits in {kits.name}. Next: `run`.")


# ------------------------------------------------------------------------------------------------- run


def goal_vote(game: str, names: list, g: dict) -> float | None:
    """X of the goal our goalkeeper defends in the first half: where the goalkeeper stretches sit, by majority."""
    roster = read_roster(g.get("team", DEFAULT_TEAM), game_folder(game) / names[0])
    gk = roster[roster.goalkeeper].jersey
    length = json.loads((game_folder(game) / "pitch_camera.local.json").read_text())["length_m"]
    half_end = parse_time(g["first_half"][1]) / 60
    votes = []
    for name in names:
        run_dir = game_folder(game) / name
        if parse_time(name_start(g, name)) / 60 > half_end or not (run_dir / "identity_segments.csv").exists():
            continue
        segs = pd.read_csv(run_dir / "identity_segments.csv")
        segs = segs[segs.jersey.isin(gk)]
        if not len(segs):
            continue
        xy = pd.read_csv(run_dir / "tracklet_pitch_xy.csv.gz", usecols=["ci", "track_id", "X_m"])
        rows = [xy[(xy.track_id == s.track_id) & xy.ci.between(s.ci_start, s.ci_end)] for s in segs.itertuples()]
        x = pd.concat(rows).X_m.median()
        if not np.isfinite(x):
            continue  # the stretches fell where the pitch has no positions: no vote from this window
        votes.append(0.0 if x < length / 2 else length)
    if not votes:
        return None
    return max(set(votes), key=votes.count)


KICKOFF_PROBE_BEFORE_S, KICKOFF_PROBE_AFTER_S = 90, 30  # the probe clip around the confirmed second-half start
FORMATION_BIN_S, FORMATION_SIDE, FORMATION_MIN_S, FORMATION_GAP_S = 2.0, 0.7, 6.0, 2.0


def formation(run_dir: Path, length: float) -> tuple:
    """(which end we defend in this clip, seconds of formation) from kickoff formations: FORMATION_BIN_S bins where
    at least FORMATION_SIDE of our players' detections stand on one side of halfway and at least FORMATION_SIDE of
    the opponents' on the other (4+ of each per frame), held FORMATION_MIN_S or longer (gaps up to FORMATION_GAP_S
    bridged). Bins, not single frames: one misclassified player among four visible (game 2's restart) drops a frame
    to 75%. The longest such run
    decides: 0.0 if we stand at low X (we defend X = 0), else the far goal. (None, 0) without one. Open play rarely
    splits the teams this cleanly for seconds; a probe of 2 minutes around the restart keeps false hits out."""
    fps = json.loads((run_dir / "cache" / "meta.json").read_text())["cache_fps"]
    xy = pd.read_csv(run_dir / "tracklet_pitch_xy.csv.gz", usecols=["ci", "track_id", "X_m"])
    roles = pd.read_csv(run_dir / "tracklet_roles.csv")
    d = xy.merge(roles[["track_id", "role", "player_candidate"]], on="track_id")
    d = d[d.player_candidate.astype(bool) & d.role.isin(["target", "opponent"])].assign(
        low=lambda f: f.X_m < length / 2
    )
    n = d.pivot_table(index="ci", columns="role", values="low", aggfunc="size")
    if not {"target", "opponent"} <= set(n.columns):
        return None, 0.0
    ok = n.index[(n.target >= 4) & (n.opponent >= 4)]
    d = d[d.ci.isin(ok)].assign(bin=lambda f: (f.ci / fps // FORMATION_BIN_S).astype(int))
    low = d.pivot_table(index="bin", columns="role", values="low", aggfunc="mean")
    if not {"target", "opponent"} <= set(low.columns):
        return None, 0.0
    ours_low = (low.target >= FORMATION_SIDE) & (low.opponent <= 1 - FORMATION_SIDE)
    ours_high = (low.target <= 1 - FORMATION_SIDE) & (low.opponent >= FORMATION_SIDE)
    best = (None, 0.0)
    for side, mask in ((0.0, ours_low), (float(length), ours_high)):
        t = np.sort(mask.index[mask.to_numpy()].to_numpy()) * FORMATION_BIN_S
        if not len(t):
            continue
        breaks = np.flatnonzero(np.diff(t) > FORMATION_GAP_S)
        starts, ends = np.r_[t[0], t[breaks + 1]], np.r_[t[breaks], t[-1]]
        longest = float((ends - starts).max()) + FORMATION_BIN_S
        if longest >= FORMATION_MIN_S and longest > best[1]:
            best = (side, longest)
    return best


def kickoff_probe(video: Path, game: str, g: dict) -> Path:
    """A 2-minute clip around the confirmed second-half start, processed as far as pitch positions and roles, in
    data/<game>/_kickoff (not a window: never counted in stats). The confirmed start is when play is seen to
    resume, and the kickoff can be just before it (game 3: under way at the first second-half window's start)."""
    start = max(0.0, parse_time(g["second_half"][0]) - KICKOFF_PROBE_BEFORE_S)
    r = game_folder(game) / "_kickoff"
    dur = str(KICKOFF_PROBE_BEFORE_S + KICKOFF_PROBE_AFTER_S)
    for done, cmd in [
        (r / "ball_path.csv", ["run_all.py", "--video", video, "--start", hms(start), "--duration", dur, "--out", r]),
        (r / "tracklet_pitch.csv", ["pitch_mask.py", "--run", r]),
        (r / "tracklet_roles.csv", ["team_classify.py", "classify", "--run", r]),
        (r / "pitch_anchors_ptz.local.json", ["pitch_ptz.py", "run", "--run", r]),
        (r / "tracklet_pitch_xy.csv.gz",
         ["pitch_calibrate.py", "apply", "--run", r, "--anchors", r / "pitch_anchors_ptz.local.json"]),
    ]:  # fmt: skip
        if not done.exists():
            with gpu_units(UNITS[cmd[0]]) if cmd[0] in UNITS else nullcontext():
                run(cmd)
    return r


def kickoff_end(game: str, g: dict, video: Path | None = None) -> float | None:
    """X of the goal we defend in the first half, from the second-half kickoff formation (each team in its own
    half). Looks in the kickoff probe (made here when video is given), else the first second-half window. None
    without a clear formation. Game 2's goalkeeper vote followed the opponent's keeper and got the end wrong; the
    kickoff, checked on a still, caught it."""
    length = json.loads((game_folder(game) / "pitch_camera.local.json").read_text())["length_m"]
    first = next((n for n, s in windows(g) if s == g["second_half"][0]), None)
    try:
        dirs = [kickoff_probe(video, game, g)] if video is not None else []
    except subprocess.CalledProcessError as e:  # e.g. no painted lines found in the probe (night game, huddle)
        print(f"kickoff probe failed ({e.cmd[2] if len(e.cmd) > 2 else e}): no kickoff evidence")
        return None
    dirs += [game_folder(game) / first] if first else []
    for run_dir in dirs:
        if (run_dir / "tracklet_pitch_xy.csv.gz").exists():
            side, secs = formation(run_dir, length)
            if side is not None:
                print(f"kickoff formation in {run_dir.name}: {secs:.0f} s, we defend X = {side:.0f} in the 2nd half")
                return float(length) - side  # the other end in the first half
    return None


END_MIN_PLAYERS = 6  # players seen 5+ min in both games, needed to compare depths


def end_check(game: str) -> list:
    """(other game, correlation) of player depth (behind or ahead of the team line) between this game's coaching
    metrics and every other processed game of the same team. Depth is measured from the goal we defend, so with the
    goal end the wrong way round it flips sign for everyone: 2026-10-03, a 6 s kickoff formation gave the wrong end
    and the correlations were -0.71 to -0.95 (right ends: +0.77 to +0.97)."""
    team = load_game(game).get("team", DEFAULT_TEAM)
    mine = pd.read_csv(game_folder(game) / "coaching" / "player_metrics.csv").set_index("jersey")
    out = []
    for gf in [DATA_DIR / "game.local.json", *sorted(DATA_DIR.glob("*/game.local.json"))]:
        folder = gf.parent
        metrics = folder / "coaching" / "player_metrics.csv"
        if folder.name == game or folder.name.startswith("_") or not metrics.exists():
            continue
        if json.loads(gf.read_text()).get("team", DEFAULT_TEAM) != team:
            continue
        other = pd.read_csv(metrics).set_index("jersey")
        both = [j for j in mine.index.intersection(other.index) if mine.minutes[j] >= 5 and other.minutes[j] >= 5]
        if len(both) >= END_MIN_PLAYERS:
            name = folder.name if folder != DATA_DIR else "first game"
            out.append((name, float(mine.depth[both].corr(other.depth[both]))))
    return out


def name_start(g: dict, name: str) -> str:
    return dict(windows(g))[name]


MIN_OURS = 3  # a minute with fewer of our players visible per frame (median) is not our match


def ours_per_minute(run_dirs: list) -> pd.Series:
    """Median number of our players (target-role player candidates) per frame, per minute of the video."""
    out = {}
    for run_dir in run_dirs:
        start = json.loads((run_dir / "cache" / "meta.json").read_text())["clip_start"]
        tr = pd.read_csv(run_dir / "best_tracklets.csv.gz", usecols=["ci", "track_id"])
        roles = pd.read_csv(run_dir / "tracklet_roles.csv").set_index("track_id")
        ours = roles.index[(roles.role == "target") & roles.player_candidate.astype(bool)]
        fps = json.loads((run_dir / "cache" / "meta.json").read_text())["cache_fps"]
        frames = tr.assign(ours=tr.track_id.isin(ours)).groupby("ci").ours.sum()
        minute = ((parse_time(start) + frames.index / fps) // 60).astype(int)
        for m, v in frames.groupby(minute).median().items():
            out[m] = v
    return pd.Series(out).sort_index()


def find_breaks(per: pd.Series, g: dict) -> list:
    """Water breaks: runs of minutes with fewer than MIN_OURS of our players per frame that have a covered minute of
    play right before and right after, inside the same half (when the game has halves). A thin run at the start or
    end of a half is not a break: it means the half's time is off (check_play says so)."""
    halves = [tuple(parse_time(x) for x in g[k]) for k in ("first_half", "second_half") if g.get(k)]
    thin = per < MIN_OURS
    out, m, prev = [], None, None
    for minute in per.index:
        if m is not None and minute != prev + 1:  # a gap between windows: the end of this run is not seen
            m = None
        prev = minute
        if thin[minute] and m is None:
            m = minute
        if m is not None and not thin[minute]:
            a, b = m * 60, minute * 60  # the break's minutes: from m up to (not including) this one
            before = m - 1 in per.index and not thin[m - 1]
            inside = not halves or any(lo <= (m - 1) * 60 and (minute + 1) * 60 <= hi for lo, hi in halves)
            if before and inside:
                out.append([hms(a), hms(b)])
            m = None
    return out


def check_play(game: str, g: dict, names: list) -> None:
    """Warn about minutes inside the confirmed halves where our team is not on the pitch, and record those inside a
    half as water breaks (not play: sv_common.play_mask). Game 2 (2026-09-28): the contact sheet's thumbnails made
    halftime warm-ups and the next game's players look like play, and the halves were confirmed 6 min too long at
    each end; counting our players per minute showed the real ends at once."""
    per = ours_per_minute([game_folder(game) / n for n, _ in names])
    breaks = find_breaks(per, g)
    if breaks != g.get("water_breaks"):
        g["water_breaks"] = breaks
        save_game(game, g)
    if breaks:
        print("play check: water breaks (not counted as play): " + ", ".join(f"{a}-{b}" for a, b in breaks))
    in_break = [m for m in per.index if any(parse_time(a) <= m * 60 < parse_time(b) for a, b in breaks)]
    thin = per[(per < MIN_OURS) & ~per.index.isin(in_break)]
    if not len(thin):
        print("play check: our team is on the pitch in every other minute of the confirmed halves")
        return
    print(f"play check: WARNING, fewer than {MIN_OURS} of our players per frame in minute(s) "
          + ", ".join(f"{m // 60}:{m % 60:02d}" for m in thin.index))  # fmt: skip
    print("  if that is the start or end of a half, correct it with `confirm --first-half/--second-half` and rerun")


def cmd_breaks(args) -> None:
    """Water breaks for games already processed (`run` finds them itself now): every game, or --game. Dry run
    unless --write; then rerun player_stats.py and coaching_tips.py for the games whose breaks changed."""
    by_game = {}
    for w in all_windows():
        by_game.setdefault(game_dir(w), []).append(w)
    for folder, runs in by_game.items():
        if args.game and folder.name != args.game:
            continue
        path = folder / "game.local.json"
        g = json.loads(path.read_text()) if path.exists() else {}
        breaks = find_breaks(ours_per_minute(runs), g)
        same = breaks == (g.get("water_breaks") or [])
        print(f"{folder.name}: {', '.join(f'{a}-{b}' for a, b in breaks) or 'none'}{'' if same else ' (changed)'}")
        if args.write and not same:
            g["water_breaks"] = breaks
            path.write_text(json.dumps(g, indent=2))


def cmd_run(args) -> None:
    g = load_game(args.game)
    if not (game_folder(args.game) / "kit_prototypes.local.json").exists():
        raise SystemExit("Run `setup` first (camera and kits).")
    video, names = Path(g["video"]), windows(g)
    kits = game_folder(args.game) / "kit_prototypes.local.json"
    for name, _ in names:  # roles classified before the kit colours last changed are redone, with what follows
        roles = game_folder(args.game) / name / "tracklet_roles.csv"
        if roles.exists() and roles.stat().st_mtime < kits.stat().st_mtime:
            for stale in ("tracklet_roles.csv", "events.csv", "identity_segments.csv", "player_identity.csv"):
                (roles.parent / stale).unlink(missing_ok=True)

    # a team the reader has not learned yet: identity would be thrown away after the seed round, so it waits
    ready = team_ready(g.get("team", DEFAULT_TEAM))
    failed = []

    def one(item):
        name, start = item
        print(f"{name} ({start}) started", flush=True)
        try:
            run_stages(video, args.game, name, start, identity=ready)
        except subprocess.CalledProcessError as e:  # one window failing must not stop the others
            failed.append(name)
            print(f"{name} FAILED ({e.cmd[2] if len(e.cmd) > 2 else e}); see {name}_window.log", flush=True)
            return
        print(f"{name} done", flush=True)

    # windows in parallel: two detections at once barely slow each other on the laptop GPU (11.6 vs 11.3 min), and
    # the pitch fit is CPU work; each worker runs one window's stages in order
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, names))
    if failed:
        raise SystemExit(f"windows failed: {failed}. Fix the cause (see their logs) and rerun `run`: finished windows "
                         "are skipped.")  # fmt: skip
    if g.get("first_half_our_goal_x") is None:
        x = goal_vote(args.game, [n for n, _ in names], g) if ready else None
        kick = kickoff_end(args.game, g, video)
        if x is not None and kick is not None and x != kick:
            raise SystemExit(
                f"Which goal we defend is unclear: the goalkeeper vote says X = {x:.0f} in the first half, the "
                f"second-half kickoff says X = {kick:.0f}. Check the contact sheet at the second-half start (which "
                'side our players stand on), set "first_half_our_goal_x" in game.local.json, and rerun `run`.'
            )
        print(f"goal end: goalkeeper vote {x}, second-half kickoff {kick}")
        x = kick if x is None else x
        if x is None:
            print("Neither our goalkeeper nor the second-half kickoff shows the end: it stays per window.")
        else:
            g["first_half_our_goal_x"] = x
            save_game(args.game, g)
            print(f"we defend X = {x:.0f} in the first half; identity again with that known")
            for name, _ in names if ready else []:
                run(["jersey_auto.py", "identify", "--run", game_folder(args.game) / name, "--from", LABELED,
                     "--write", "--force"])  # fmt: skip
    check_play(args.game, g, names)
    if not ready:
        print("\nA new team: the jersey reader has not seen its numbers yet, so identity waits for the seed round.")
        print(f"Next (about 15-20 min of the owner's time): `new_game.py numbers --game {args.game} --seed`, then "
              f"`--apply` (identity, stats and coaching pages).")  # fmt: skip
        if g.get("first_half_our_goal_x") is None:
            print('Which goal we defend is not known: set "first_half_our_goal_x" in game.local.json before --apply.')
        return
    runs = ",".join(str(game_folder(args.game) / n) for n, _ in names)
    run(["player_stats.py", "--runs", runs])
    if g.get("first_half_our_goal_x") is None:
        raise SystemExit("Coaching pages need which goal we defend: add first_half_our_goal_x to game.local.json.")
    run(["coaching_tips.py", "--runs", runs])
    checks = end_check(args.game)
    if checks:
        print("goal end check, player depth vs other games: " + ", ".join(f"{k} {c:+.2f}" for k, c in checks))
    if checks and np.median([c for _, c in checks]) < 0:
        raise SystemExit(
            "The goal end looks the wrong way round (player depths run opposite to the team's other games). Set "
            '"first_half_our_goal_x" in game.local.json to the other end, rerun jersey_auto.py identify --force on '
            "every window (the goalkeeper is found by end), then `run` again. Do not publish before that."
        )
    print(f"\ndone: coaching pages in {game_folder(args.game) / 'coaching'} (local only)")
    print(f"Optional (about 5 min of the owner's time): `new_game.py numbers --game {args.game}`")


def cmd_numbers(args) -> None:
    """The per-game jersey round. Without --apply: candidate crops of the numbers whose reads conflict or are
    rarely trusted in this game (jersey_rare_label.py candidates --auto), for the owner to confirm with
    jersey_rare_label.py label. With --apply: retrain the reader with every confirmed crop, then identity, stats
    and coaching pages for this game again. Windows of other games keep their reads until they are rerun."""
    g = load_game(args.game)
    runs = [game_folder(args.game) / n for n, _ in windows(g)]
    joined = ",".join(str(r) for r in runs)
    if not args.apply and args.seed:
        # a new team's first game: the reader has never seen this team's numbers, so every roster number gets a
        # screen, from the original reader's reads (the fine-tuned one leans towards the first team's numbers)
        run(["jersey_auto.py", "reads", "--runs", joined])
        numbers = ",".join(str(j) for j in read_roster(g.get("team", DEFAULT_TEAM), runs[0]).jersey)
        run(["jersey_rare_label.py", "candidates", "--numbers", numbers, "--max-per-number", "36", "--labeled",
             LABELED, "--runs", joined])  # fmt: skip
    elif not args.apply:
        run(["jersey_rare_label.py", "candidates", "--auto", "--max-per-number", "36", "--labeled", LABELED,
             "--runs", joined])  # fmt: skip
    if not args.apply:
        print("\nOwner: python jersey_rare_label.py label  (one screen per number: click the crops that clearly show")
        print(f"it on our players, then Enter). Then: python new_game.py numbers --game {args.game} --apply")
        return
    run(["jersey_auto.py", "finetune", "--runs", LABELED])
    for r in runs:
        run(["jersey_auto.py", "identify", "--run", r, "--from", LABELED, "--write"])
    run(["player_stats.py", "--runs", joined])
    run(["coaching_tips.py", "--runs", joined])
    print(f"done: identity, stats and coaching pages for {args.game} with the retrained reader")


def all_windows() -> list:
    """Every processed window of every game: data/<window> (the first game) and data/<game>/<window>. Folders
    starting with "_" (experiments, windows outside play, kickoff probes) are not games or windows."""
    found = []
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        if (d / "player_identity.csv").exists() and (d / "player_events.csv").exists():
            found.append(d)
        elif (d / "game.local.json").exists():
            found += [
                w for w in sorted(d.iterdir())
                if w.is_dir() and not w.name.startswith("_") and (w / "player_identity.csv").exists()
                and (w / "player_events.csv").exists()
            ]  # fmt: skip
    return found


def missing_photos(team: str) -> list:
    """A team's players on the site (its export) with no photo decision: neither a chosen photo nor "no photo"."""
    players = pd.read_csv(DATA_DIR / "site_export" / team / "season" / "metrics.csv").jersey.astype(int)
    folder = DATA_DIR / "site_photos" if team == DEFAULT_TEAM else DATA_DIR / "site_photos" / team
    choices = folder / "choices.csv"
    decided = set(pd.read_csv(choices).jersey.astype(int)) if choices.exists() else set()
    return sorted(set(players) - decided)


def cmd_publish(args) -> None:
    import os

    bucket = args.bucket or os.environ.get("SITE_BUCKET")
    if not bucket:
        raise SystemExit("Give --bucket (or set SITE_BUCKET): the site's private bucket, see webapp/README.md.")
    windows_all = all_windows()
    runs = ",".join(str(w) for w in windows_all)
    print(f"{len(windows_all)} windows")
    run(["site_export.py", "--runs", runs])  # one export per team
    for team in sorted({team_of(w) for w in windows_all}):
        team_runs = ",".join(str(w) for w in windows_all if team_of(w) == team)
        new = missing_photos(team)
        if new and not args.no_photos:
            print(f"{team}: {len(new)} player(s) without a photo decision: {new}. Finding candidate crops...")
            run(["player_photo.py", "--team", team, "candidates", "--runs", team_runs, "--jerseys",
                 ",".join(map(str, new))])  # fmt: skip
            print("Owner: pick one photo per player in the window (s = no photo, initials instead; q = stop).")
            run(["player_photo.py", "--team", team, "label"])
        elif new:
            print(f"{team}: {len(new)} player(s) without a photo decision (shown with initials): {new}")
    # publish_site.py asks before uploading, so it runs attached to this terminal
    subprocess.run([sys.executable, "publish_site.py", "--bucket", bucket], check=True, cwd=HERE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="density scan and a proposal of the halves, with a contact sheet")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--game", required=True, help="a short name, e.g. g0922 (the folder data/<game>)")
    p.set_defaults(fn=cmd_plan)
    c = sub.add_parser("confirm", help="accept the proposed halves, or correct them")
    c.add_argument("--game", required=True)
    c.add_argument("--first-half", help="HH:MM:SS-HH:MM:SS")
    c.add_argument("--second-half", help="HH:MM:SS-HH:MM:SS")
    c.set_defaults(fn=cmd_confirm)
    s = sub.add_parser("setup", help="pilot window: pitch camera and kit colours for this game")
    s.add_argument("--game", required=True)
    s.add_argument(
        "--kits-like",
        default="",
        help="a previous game of the same team in the same kit: map the colours "
        "against its kits instead of stopping for the owner (check the montage afterwards)",
    )
    s.set_defaults(fn=cmd_setup)
    b = sub.add_parser("breaks", help="water breaks of games already processed (minutes our team leaves the pitch)")
    b.add_argument("--game", default="", help="one game folder (default: every game)")
    b.add_argument("--write", action="store_true", help="save them in each game.local.json")
    b.set_defaults(fn=cmd_breaks)
    r = sub.add_parser("run", help="every window, which goal we defend, stats and coaching pages")
    r.add_argument("--game", required=True)
    r.add_argument("--workers", type=int, default=3, help="windows processed at once (GPU memory: about 2 GB each)")
    r.set_defaults(fn=cmd_run)
    nb = sub.add_parser("numbers", help="the per-game jersey round: confirm crops of conflicting numbers")
    nb.add_argument("--game", required=True)
    nb.add_argument("--apply", action="store_true", help="after the owner labeled: retrain the reader, redo the game")
    nb.add_argument("--seed", action="store_true", help="a new team's first game: a screen for every roster number")
    nb.set_defaults(fn=cmd_numbers)
    pb = sub.add_parser("publish", help="every game to the coaching site: export, photos for new players, upload")
    pb.add_argument("--bucket", help="the site's private bucket (default: SITE_BUCKET)")
    pb.add_argument("--no-photos", action="store_true", help="skip the photo picker for new players")
    pb.set_defaults(fn=cmd_publish)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
