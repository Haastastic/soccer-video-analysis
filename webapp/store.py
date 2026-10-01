"""The published release: every team's page data (site_export.py) plus the chosen photos, read from the private
bucket (or a local folder in development and tests).

Layout (publish_site.py writes it): current.json -> {"release": "releases/<stamp>"}; under that prefix teams.json
({"teams": [...]}) and one folder per team with the files of data/site_export/<team>/ and photos/player_NN.jpg. A
release is loaded once into memory and reloaded when current.json points somewhere new (checked at most every
RELOAD_S seconds).
"""

import io
import json
import threading
import time
from pathlib import Path

import pandas as pd

RELOAD_S = 60


class LocalStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def read(self, path: str) -> bytes | None:
        p = (self.root / path).resolve()
        if self.root.resolve() not in p.parents or not p.is_file():
            return None
        return p.read_bytes()


class GcsStore:
    def __init__(self, bucket: str):
        from google.cloud import storage

        self.bucket = storage.Client().bucket(bucket)

    def read(self, path: str) -> bytes | None:
        from google.api_core.exceptions import NotFound

        try:
            return self.bucket.blob(path).download_as_bytes()
        except NotFound:
            return None


def _csv(data: bytes | None, **kw) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), **kw) if data else pd.DataFrame()


class Release:
    """Everything the pages need, parsed once."""

    def __init__(self, store, prefix: str):
        self.store, self.prefix = store, prefix.rstrip("/")
        self.manifest = json.loads(self._read("manifest.json"))
        self.games = {g["id"]: g for g in self.manifest["games"]}
        self.order = sorted(self.games)  # labels are dates: date order
        self.season = _csv(self._read("season/metrics.csv"))
        self.season_meds = _csv(self._read("season/meds.csv"))
        self.tagged = {int(k): v for k, v in json.loads(self._read("season/tagged.json")).items()}
        for t in self.tagged.values():
            for f in t:
                f["evidence"] = [tuple(e) for e in f["evidence"]]
        self.per, self.meds, self.tips, self.samples = {}, {}, {}, {}
        for k in self.order:
            self.per[k] = _csv(self._read(f"games/{k}/metrics.csv"))
            self.meds[k] = _csv(self._read(f"games/{k}/meds.csv"))
            tips = json.loads(self._read(f"games/{k}/tips.json"))
            self.tips[k] = {int(j): [tuple(t) for t in v] for j, v in tips.items()}
            self.samples[k] = _csv(self._read(f"games/{k}/samples.csv.gz"), compression="gzip")
        self.photos = set()
        for j in self.season.jersey if len(self.season) else []:
            if self._read(self.photo_path(int(j))) is not None:
                self.photos.add(int(j))

    def _read(self, path: str) -> bytes | None:
        return self.store.read(f"{self.prefix}/{path}")

    @staticmethod
    def photo_path(jersey: int) -> str:
        return f"photos/player_{int(jersey):02d}.jpg"

    def photo(self, jersey: int) -> bytes | None:
        return self._read(self.photo_path(jersey)) if int(jersey) in self.photos else None

    def players(self) -> pd.DataFrame:
        return self.season

    def med(self, table: pd.DataFrame, jersey: int) -> pd.Series:
        row = table[table.jersey == jersey].drop(columns="jersey")
        return row.iloc[0].astype(float) if len(row) else pd.Series(dtype=float)


class Site:
    """Every team's release: teams in the order teams.json lists them."""

    def __init__(self, store, prefix: str):
        teams = json.loads(store.read(f"{prefix}/teams.json"))["teams"]
        self.teams = {t: Release(store, f"{prefix}/{t}") for t in teams}


class ReleaseCache:
    """The current release, reloaded when the bucket's current.json changes."""

    def __init__(self, store):
        self.store = store
        self.lock = threading.Lock()
        self.release, self.prefix, self.checked = None, None, 0.0

    def get(self) -> Site | None:
        now = time.time()
        if self.release is not None and now - self.checked < RELOAD_S:
            return self.release
        with self.lock:
            self.checked = now
            raw = self.store.read("current.json")
            if raw is None:
                return self.release
            prefix = json.loads(raw)["release"]
            if prefix != self.prefix:
                self.release, self.prefix = Site(self.store, prefix), prefix
        return self.release
