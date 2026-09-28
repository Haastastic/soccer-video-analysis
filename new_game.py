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
            goalkeeper was found in the first-half windows), identity again with that known, player stats and the
            coaching pages (data/<game>/coaching/, local only).

Each step is resumable: rerun it and finished work is skipped. Everything written stays under data/ (git-ignored),
including the contact sheet: it shows minors.

Example (PowerShell):
  python new_game.py plan --video "videos\\<game>.mp4" --game g0922
  python new_game.py confirm --game g0922
  python new_game.py setup --game g0922
  python new_game.py run --game g0922
"""

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from run_windows import stages
from sv_common import DATA_DIR, parse_time, require_under_data

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


def propose(scan: pd.DataFrame) -> dict:
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
    if not middle:
        raise SystemExit("No break near the middle of the video: pass the halves to `confirm` by hand.")
    half_break = max(middle, key=lambda b: b[1] - b[0])
    h1 = (first_play, half_break[0])
    h2_start = half_break[1]
    # the end: about the first half's length after the restart (plus a little stoppage time), never past the
    # video; the owner checks it on the contact sheet
    h2 = (h2_start, min(video_end, h2_start + (h1[1] - h1[0]) + 60))
    water = [b for b in breaks if b != half_break and (h1[0] < b[0] < h1[1] or h2[0] < b[0] < h2[1])]
    return dict(first_half=[hms(h1[0]), hms(h1[1])], second_half=[hms(h2[0]), hms(h2[1])],
                halftime_min=round((half_break[0] + half_break[1]) / 2 / 60, 1),
                water_breaks=[[hms(a), hms(b)] for a, b in water])  # fmt: skip


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
    marks += [((parse_time(a) + parse_time(b)) / 2, "water break") for a, b in g["water_breaks"]]
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


def cmd_plan(args) -> None:
    folder = game_folder(args.game)
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / "density_scan.csv").exists():
        run(["scan_density.py", "--video", args.video, "--out", folder])
    g = dict(video=str(args.video), **propose(pd.read_csv(folder / "density_scan.csv")), confirmed=False)
    save_game(args.game, g)
    contact_sheet(Path(args.video), g, folder / "live_play_check.jpg")
    print(json.dumps(g, indent=2))
    print(f"\nCheck {folder / 'live_play_check.jpg'} (local only), then `confirm` (with corrections if needed).")


def cmd_confirm(args) -> None:
    g = load_game(args.game)
    if args.first_half:
        g["first_half"] = args.first_half.split("-")
    if args.second_half:
        g["second_half"] = args.second_half.split("-")
    if args.first_half or args.second_half:
        g["halftime_min"] = round((parse_time(g["first_half"][1]) + parse_time(g["second_half"][0])) / 2 / 60, 1)
    g["confirmed"] = True
    save_game(args.game, g)
    print(json.dumps(g, indent=2))


# ----------------------------------------------------------------------------------------------- setup


def windows(g: dict) -> list:
    """(name, start) 5-minute windows over both halves; a half's last window ends at the half's end."""
    out = []
    for key in ("first_half", "second_half"):
        a, b = (parse_time(x) for x in g[key])
        starts = list(np.arange(a, b - WINDOW_S + 1, WINDOW_S))
        covered = starts[-1] + WINDOW_S if starts else a
        if b - covered >= MIN_TAIL_S:
            starts.append(b - WINDOW_S)
        out += [(f"w{int(s) // 60:02d}{int(s) % 60:02d}", hms(s)) for s in starts]
    return out


def run_stages(video: Path, game: str, name: str, start: str, upto: str | None = None) -> Path:
    run_dir = game_folder(game) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    labeled = [Path(p).resolve() for p in LABELED.split(",")]
    with open(game_folder(game) / f"{name}_window.log", "a", encoding="utf-8") as log:
        for done, cmd in stages(video, run_dir.resolve(), start, labeled):
            if upto and cmd[0] == upto:
                break
            if not done.exists():
                run(cmd, log)
    return run_dir


def pitch_ok(run_dir: Path) -> tuple:
    """(good enough, summary). Gaps between fixes are bridged by the cached camera motion, so what matters is
    that fixes are frequent and no gap is long: game 2's pilot fixed 77% (fainter lines in its light), longest gap
    14 s, and its noise floor (25.5 m/min) sat inside game 1's (16.8 to 28.7)."""
    rep = json.loads((run_dir / "pitch_ptz_report.json").read_text())
    pct, gap = float(rep["fixed_pct"]), float(rep["longest_gap_between_fixes_s"])
    return pct >= PILOT_FIXED_PCT and gap <= PILOT_MAX_GAP_S, f"{pct:.0f}% of frames fixed, longest gap {gap:.0f} s"


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
    roster = pd.read_csv(HERE / "roster.csv")
    gk = roster[roster.goalkeeper.astype(str).str.lower() == "true"].jersey.astype(int)
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


def name_start(g: dict, name: str) -> str:
    return dict(windows(g))[name]


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

    def one(item):
        name, start = item
        print(f"{name} ({start}) started", flush=True)
        run_stages(video, args.game, name, start)
        print(f"{name} done", flush=True)

    # windows in parallel: two detections at once barely slow each other on the laptop GPU (11.6 vs 11.3 min), and
    # the pitch fit is CPU work; each worker runs one window's stages in order
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, names))
    if g.get("first_half_our_goal_x") is None:
        x = goal_vote(args.game, [n for n, _ in names], g)
        if x is None:
            print("Our goalkeeper was not found in the first half: the goalkeeper end stays per window.")
        else:
            g["first_half_our_goal_x"] = x
            save_game(args.game, g)
            print(f"we defend X = {x:.0f} in the first half (goalkeeper vote); identity again with that known")
            for name, _ in names:
                run(["jersey_auto.py", "identify", "--run", game_folder(args.game) / name, "--from", LABELED,
                     "--write", "--force"])  # fmt: skip
    runs = ",".join(str(game_folder(args.game) / n) for n, _ in names)
    run(["player_stats.py", "--runs", runs])
    if g.get("first_half_our_goal_x") is None:
        raise SystemExit("Coaching pages need which goal we defend: add first_half_our_goal_x to game.local.json.")
    run(["coaching_tips.py", "--runs", runs])
    print(f"\ndone: coaching pages in {game_folder(args.game) / 'coaching'} (local only)")


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
    s.set_defaults(fn=cmd_setup)
    r = sub.add_parser("run", help="every window, which goal we defend, stats and coaching pages")
    r.add_argument("--game", required=True)
    r.add_argument("--workers", type=int, default=3, help="windows processed at once (GPU memory: about 2 GB each)")
    r.set_defaults(fn=cmd_run)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
