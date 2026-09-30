"""Stage the coaching site for Cloud Run: webapp/_build/ holds only the site's code, coaching_html.py and the
Dockerfile, so `gcloud run deploy --source webapp/_build` can never upload data/ or anything else from the repo.

  python webapp/deploy.py stage
"""

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "_build"
FILES = ["app.py", "db.py", "notify.py", "render.py", "store.py", "ui.py", "requirements.txt", "Dockerfile"]


def stage() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir()
    for f in FILES:
        shutil.copy(HERE / f, BUILD / f)
    shutil.copy(HERE.parent / "coaching_html.py", BUILD / "coaching_html.py")
    print(f"staged {len(FILES) + 1} files in {BUILD}")
    print(f"next: gcloud run deploy coaching --source {BUILD} ... (see webapp/README.md)")


if __name__ == "__main__":
    if sys.argv[1:] != ["stage"]:
        raise SystemExit(__doc__)
    stage()
