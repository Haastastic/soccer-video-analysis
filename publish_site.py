"""Publish the coaching site's data to its private bucket: the one way per-player data leaves this computer.

Uploads data/site_export/<team>/ (site_export.py) and each team's chosen photos (player_photo.py:
data/site_photos/player_NN.jpg for the first team, data/site_photos/<team>/ for others) as a new release under
releases/<UTC stamp>/<team>/ with teams.json, then points current.json at it, so the site never shows half an upload.
Only those files are sent, by an allow-list of names; nothing else under data/ (video, crops, caches, candidate
photos) can go. Lists everything first and asks before uploading. Keeps the newest KEEP releases.

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
import time
from pathlib import Path

from sv_common import DATA_DIR, DEFAULT_TEAM

EXPORT = DATA_DIR / "site_export"
PHOTOS = DATA_DIR / "site_photos"
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


if __name__ == "__main__":
    main()
