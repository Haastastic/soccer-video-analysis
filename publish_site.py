"""Publish the coaching site's data to its private bucket: the one way per-player data leaves this computer.

Uploads data/site_export/<team>/ (site_export.py) and each team's chosen photos (player_photo.py:
data/site_photos/player_NN.jpg for the first team, data/site_photos/<team>/ for others) as a new release under
releases/<UTC stamp>/<team>/ with teams.json, then points current.json at it, so the site never shows half an upload.
Only those files are sent, by an allow-list of names; nothing else under data/ (video, crops, caches, candidate
photos) can go. Lists everything first and asks before uploading. Keeps the newest KEEP releases.

After the upload, games with no opponent set on the site get one from their video's file name ("... vs <opponent>
<date>.mp4"), and its logo if the video's folder holds "<school> Logo.png/.jpg/.jfif" whose name starts the
opponent's (St. Elm JV -> "St Elm Logo.png"). Saved in the site's database (Firestore), like the admin page
does; never in git. Anything an admin set already is kept.

Needs: gcloud auth application-default login (the owner), and the bucket from webapp/README.md.

Example:
  python publish_site.py --bucket <bucket> --dry-run
  python publish_site.py --bucket <bucket>
  python publish_site.py --local-dir webapp/_local_bucket     (local development: a folder stands in for the bucket)
"""

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

from sv_common import DATA_DIR, DEFAULT_TEAM

EXPORT = DATA_DIR / "site_export"
PHOTOS = DATA_DIR / "site_photos"
HERE = Path(__file__).resolve().parent
KEEP = 3
TEAM_RE = r"[a-z0-9_-]+"
FILE_RE = re.compile(  # the allow-list: paths inside a release (teams.json, then each team's folder)
    r"^(teams\.json|" + TEAM_RE + r"/(manifest\.json|season/(metrics\.csv|meds\.csv|tagged\.json)"
    r"|games/[A-Za-z0-9_-]+/(metrics\.csv|meds\.csv|tips\.json|samples\.csv\.gz)"
    r"|photos/player_\d{2}\.jpg))$"
)
MIME = {".json": "application/json", ".csv": "text/csv", ".gz": "application/gzip", ".jpg": "image/jpeg"}


def release_files() -> list:
    """(path inside the release, local file) for everything to publish; refuses anything off the allow-list."""
    teams = sorted(d.name for d in EXPORT.iterdir() if (d / "manifest.json").exists()) if EXPORT.exists() else []
    if not teams:
        raise SystemExit(f"No {EXPORT}/<team>/manifest.json: run site_export.py first.")
    out = []
    for team in teams:
        for f in sorted((EXPORT / team).rglob("*")):
            if f.is_file():
                out.append((f"{team}/{f.relative_to(EXPORT / team).as_posix()}", f))
        photos = PHOTOS if team == DEFAULT_TEAM else PHOTOS / team  # player_photo.py's folders
        for f in sorted(photos.glob("player_*.jpg")):
            out.append((f"{team}/photos/{f.name}", f))
    bad = [p for p, _ in out if not FILE_RE.match(p)]
    if bad:
        raise SystemExit(f"Refusing unexpected files: {bad}")
    return out


LOGO_EXT = (".png", ".jpg", ".jpeg", ".jfif")
LEVELS = {"jv", "varsity", "c", "b", "a", "freshman"}  # team-level words after a school name ("Lake Ridge-JV")


def words(name: str) -> list:
    return re.findall(r"[a-z0-9]+", name.lower().replace(".", ""))


def opponent_from_video(video: str) -> str | None:
    """ "<us> vs <opponent> <YYYY-MM-DD>.mp4" -> "<opponent>", written as the admins do: a space before the level
    ("Lake Ridge-JV" -> "Lake Ridge JV"), no "Varsity"."""
    m = re.search(r"\bvs\.?\s+(.+?)\s+\d{4}-\d{2}-\d{2}\s*$", Path(video).stem, re.I)
    if not m:
        return None
    name = re.sub(r"\s*-\s*(?=[A-Za-z]+$)", " ", m.group(1).strip())
    return re.sub(r"\s+varsity$", "", name, flags=re.I) or None


def find_logo(folder: Path, opponent: str) -> Path | None:
    """The "<school> Logo.<ext>" in folder whose school words start the opponent's (level words dropped); the
    longest such school wins."""
    opp = words(opponent)
    while opp and opp[-1] in LEVELS:
        opp.pop()
    best = None
    for f in folder.glob("*"):
        if f.suffix.lower() not in LOGO_EXT or not f.stem.lower().endswith(" logo"):
            continue
        school = words(f.stem[: -len(" logo")])
        if school and opp[: len(school)] == school and (best is None or len(school) > best[0]):
            best = (len(school), f)
    return best[1] if best else None


def game_videos() -> dict:
    """(team, site game id) -> the game's video path, from every game.local.json (ids as site_export.py makes them)."""
    from coaching_tips import game_label

    out = {}
    for gf in [DATA_DIR / "game.local.json", *sorted(DATA_DIR.glob("*/game.local.json"))]:
        if not gf.exists() or gf.parent.name.startswith("_"):
            continue
        game = json.loads(gf.read_text())
        if game.get("video"):
            out[(game.get("team", DEFAULT_TEAM), game_label(gf.parent, game))] = HERE / game["video"]
    return out


def set_opponents(db, games: dict) -> None:
    """Fill each published game's missing opponent name and logo from its video (see the module notes)."""
    sys.path.insert(0, str(HERE / "webapp"))
    from app import process_logo  # the admin page's re-encoding, so a logo from here is stored the same way

    videos = game_videos()
    for team, gids in games.items():
        for gid in gids:
            video = videos.get((team, gid))
            opponent = opponent_from_video(str(video)) if video else None
            if not opponent:
                print(f"  {team} {gid}: no opponent in the video name")
                continue
            key = f"{team}_{gid}"
            meta = db.get_game(key) or (db.get_game(gid) if team == DEFAULT_TEAM else {})  # legacy key, before teams
            changed = []
            if not meta.get("opponent"):
                meta["opponent"] = opponent[:80]
                changed.append("opponent")
            logo = find_logo(video.parent, opponent) if not meta.get("logo") else None
            if logo:
                try:
                    meta["logo"] = process_logo(logo.read_bytes())
                    changed.append(f"logo {logo.name}")
                except ValueError as e:
                    print(f"  {team} {gid}: {logo.name} not used ({e})")
            if changed:
                db.put_game(key, meta)
                db.audit("publish_site", "save_game", {"team": team, "game": gid, "opponent": meta["opponent"],
                                                        "logo": "logo" in meta})  # fmt: skip
            print(f"  {team} {gid}: vs {meta['opponent']}" + (f" (set {', '.join(changed)})" if changed else ""))


class GcsTarget:
    def __init__(self, bucket: str):
        from google.cloud import storage

        self.bucket = storage.Client().bucket(bucket)

    def put(self, path: str, data: bytes, mime: str) -> None:
        blob = self.bucket.blob(path)
        blob.cache_control = "private, no-store"
        blob.upload_from_string(data, content_type=mime)

    def releases(self) -> list:
        names = {b.name.split("/")[1] for b in self.bucket.list_blobs(prefix="releases/")}
        return sorted(names)

    def delete_release(self, stamp: str) -> None:
        for b in self.bucket.list_blobs(prefix=f"releases/{stamp}/"):
            b.delete()


class LocalTarget:
    def __init__(self, root: Path):
        self.root = root

    def put(self, path: str, data: bytes, mime: str) -> None:
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def releases(self) -> list:
        d = self.root / "releases"
        return sorted(p.name for p in d.iterdir()) if d.exists() else []

    def delete_release(self, stamp: str) -> None:
        shutil.rmtree(self.root / "releases" / stamp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--bucket", help="the site's private Cloud Storage bucket")
    where.add_argument("--local-dir", type=Path, help="a local folder standing in for the bucket (development)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be uploaded, upload nothing")
    ap.add_argument("--yes", action="store_true", help="do not ask before uploading")
    ap.add_argument("--project", help='the site Google Cloud project (default: the bucket name without "-data")')
    ap.add_argument("--db-file", type=Path, help="with --local-dir: the local site's DB_FILE, to set opponents there")
    args = ap.parse_args()
    files = release_files()
    total = sum(f.stat().st_size for _, f in files)
    photos = sum("/photos/" in p for p, _ in files)
    teams = sorted({p.split("/")[0] for p, _ in files})
    games = [g for t in teams for g in json.loads((EXPORT / t / "manifest.json").read_text())["games"]]
    for p, f in files:
        print(f"  {p:48s} {f.stat().st_size / 1024:8.0f} KB")
    target_name = args.bucket or args.local_dir
    print(f"{len(files)} files, {total / 1e6:.1f} MB: teams {teams}, {len(games)} game(s), {photos} photos -> "
          f"{target_name}")  # fmt: skip
    print("These pages name minors and show their photos; they go to the private bucket behind sign-in only.")
    if args.dry_run:
        return
    if not args.yes and input("Upload? [y/N] ").strip().lower() != "y":
        print("Nothing uploaded.")
        return
    target = GcsTarget(args.bucket) if args.bucket else LocalTarget(args.local_dir)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for p, f in files:
        target.put(f"releases/{stamp}/{p}", f.read_bytes(), MIME.get(f.suffix, "application/octet-stream"))
    target.put(f"releases/{stamp}/teams.json", json.dumps({"teams": teams}).encode(), "application/json")
    target.put("current.json", json.dumps({"release": f"releases/{stamp}"}).encode(), "application/json")
    for old in target.releases()[:-KEEP]:
        target.delete_release(old)
    print(f"published releases/{stamp}; the site picks it up within a minute")
    sys.path.insert(0, str(HERE / "webapp"))
    from db import FirestoreDB, MemoryDB

    if args.bucket:
        project = args.project or (args.bucket[: -len("-data")] if args.bucket.endswith("-data") else None)
        if not project:
            print("Opponents not set: give --project.")
            return
        db = FirestoreDB(project)
    elif args.db_file:
        db = MemoryDB(args.db_file)
    else:
        return
    print("Opponents (only missing ones are set):")
    set_opponents(db, {t: [g["id"] for g in json.loads((EXPORT / t / "manifest.json").read_text())["games"]]
                       for t in teams})  # fmt: skip


if __name__ == "__main__":
    main()
