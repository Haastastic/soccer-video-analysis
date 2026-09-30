"""Publish the coaching site's data to its private bucket: the one way per-player data leaves this computer.

Uploads data/site_export/ (site_export.py) and the chosen photos data/site_photos/player_NN.jpg (player_photo.py)
as a new release under releases/<UTC stamp>/, then points current.json at it, so the site never shows half an
upload. Only those files are sent, by an allow-list of names; nothing else under data/ (video, crops, caches,
candidate photos) can go. Lists everything first and asks before uploading. Keeps the newest KEEP releases.

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

from sv_common import DATA_DIR

EXPORT = DATA_DIR / "site_export"
PHOTOS = DATA_DIR / "site_photos"
KEEP = 3
FILE_RE = re.compile(  # the allow-list: paths inside a release
    r"^(manifest\.json|season/(metrics\.csv|meds\.csv|tagged\.json)"
    r"|games/[A-Za-z0-9_-]+/(metrics\.csv|meds\.csv|tips\.json|samples\.csv\.gz)"
    r"|photos/player_\d{2}\.jpg)$"
)
MIME = {".json": "application/json", ".csv": "text/csv", ".gz": "application/gzip", ".jpg": "image/jpeg"}


def release_files() -> list:
    """(path inside the release, local file) for everything to publish; refuses anything off the allow-list."""
    if not (EXPORT / "manifest.json").exists():
        raise SystemExit(f"No {EXPORT / 'manifest.json'}: run site_export.py first.")
    out = []
    for f in sorted(EXPORT.rglob("*")):
        if f.is_file():
            out.append((f.relative_to(EXPORT).as_posix(), f))
    for f in sorted(PHOTOS.glob("player_*.jpg")):
        out.append((f"photos/{f.name}", f))
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
    photos = sum(p.startswith("photos/") for p, _ in files)
    games = json.loads((EXPORT / "manifest.json").read_text())["games"]
    for p, f in files:
        print(f"  {p:48s} {f.stat().st_size / 1024:8.0f} KB")
    target_name = args.bucket or args.local_dir
    print(f"{len(files)} files, {total / 1e6:.1f} MB: {len(games)} game(s), {photos} photos -> {target_name}")
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
    target.put("current.json", json.dumps({"release": f"releases/{stamp}"}).encode(), "application/json")
    for old in target.releases()[:-KEEP]:
        target.delete_release(old)
    print(f"published releases/{stamp}; the site picks it up within a minute")


if __name__ == "__main__":
    main()
