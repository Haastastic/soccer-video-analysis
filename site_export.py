"""Export the coaching pages' data for the invited-users site (site/app.py), which renders them per viewer.

The site builds the same pages as coaching_tips.py (coaching_html.py's builders), but per request, so a parent sees
team numbers with only their own players named. So this writes the builders' inputs, not HTML:

One folder per team (data/site_export/<team>/, team from each game's game.local.json), each holding:
  data/site_export/<team>/manifest.json   games (id = label, pitch size, windows), when exported
  data/site_export/<team>/season/metrics.csv     pooled metrics over every game given (one row per player)
  data/site_export/<team>/season/meds.csv        pooled role medians per player
  data/site_export/<team>/season/tagged.json     observations tagged by game (coaching_tips.tag_tips)
  data/site_export/<team>/games/<id>/metrics.csv that game's metrics; meds.csv, tips.json the same for that game
  data/site_export/<team>/games/<id>/samples.csv.gz  identified positions (jersey, from_goal, y_team) for the heatmaps

It holds names of minors (from roster.csv) and stays under data/ (git-ignored) until publish_site.py uploads it to
the private bucket. No video, crops or detections are exported; photos are chosen separately (player_photo.py).

Example:
  python site_export.py --runs data\\clipA,...,data\\g0922\\w0000,...
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd

import coaching_tips as ct
from sv_common import DATA_DIR, game_dir, require_under_data

OUT = DATA_DIR / "site_export"


def meds_table(tm: dict) -> pd.DataFrame:
    """jersey -> role median (a Series per player) as one table."""
    rows = []
    for j, (_, med) in tm.items():
        rows.append(pd.Series(med, dtype=float).rename(None).to_frame().T.assign(jersey=int(j)))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["jersey"])


def with_confidence(m: pd.DataFrame) -> pd.DataFrame:
    return m.assign(conf=[ct.confidence(x) for x in m.minutes])


def export(gs: list, out: Path | None = None) -> None:
    """One team's games into out (default OUT/<team>): other teams' exports are left alone."""
    gs = sorted(gs, key=lambda g: g["label"])
    out = out or OUT / gs[0]["team"]
    for old in ("manifest.json", "season", "games"):  # the single-team layout written before teams existed
        p = OUT / old
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    if out.exists():
        shutil.rmtree(out)
    (out / "season").mkdir(parents=True)
    per = {g["label"]: ct.metrics(g["s"], g["ev"], g["team"]) for g in gs}
    tm = {k: ct.player_tips(q) for k, q in per.items()}
    games = []
    for g in gs:
        k = g["label"]
        d = out / "games" / k
        d.mkdir(parents=True)
        with_confidence(per[k]).to_csv(d / "metrics.csv", index=False)
        meds_table(tm[k]).to_csv(d / "meds.csv", index=False)
        tips = {str(j): [list(t) for t in v[0]] for j, v in tm[k].items()}
        (d / "tips.json").write_text(json.dumps(tips, indent=1))
        smp = g["s"][["jersey", "from_goal", "y_team"]]  # unrounded: the heatmap bins must match the local pages
        smp.to_csv(d / "samples.csv.gz", index=False)
        games.append(dict(id=k, label=k, n_windows=len(g["runs"]), length=g["length"], width=g["width"]))
    s = pd.concat([g["s"] for g in gs], ignore_index=True)
    ev = pd.concat([g["ev"] for g in gs], ignore_index=True)
    m = ct.metrics(s, ev, gs[0]["team"])
    pooled = ct.player_tips(m)
    tagged = {}
    for _, r in m.iterrows():
        rows = {k: q.set_index("jersey").loc[r.jersey] for k, q in per.items() if r.jersey in set(q.jersey)}
        enough = {k: tm[k][r.jersey][0] for k, q in rows.items() if q.minutes >= ct.MIN_MINUTES}
        keeper = r.role == "goalkeeper"
        t = [] if keeper or r.minutes < ct.MIN_MINUTES else ct.tag_tips(pooled[r.jersey][0], enough)
        tagged[str(int(r.jersey))] = [dict(f, evidence=[list(e) for e in f["evidence"]]) for f in t]
    with_confidence(m).to_csv(out / "season" / "metrics.csv", index=False)
    meds_table(pooled).to_csv(out / "season" / "meds.csv", index=False)
    (out / "season" / "tagged.json").write_text(json.dumps(tagged, indent=1))
    manifest = dict(exported=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), games=games)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"exported {len(gs)} game(s), {len(m)} players to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="comma list of window folders with identity (one or more games)")
    args = ap.parse_args()
    by_game = {}
    for r in args.runs.split(","):
        run = require_under_data(Path(r))
        by_game.setdefault(game_dir(run), []).append(run)
    gs = [ct.load_game(rs) for rs in by_game.values()]
    for team in sorted({g["team"] for g in gs}):
        tg = [g for g in gs if g["team"] == team]
        if len({g["label"] for g in tg}) != len(tg):
            raise SystemExit('two games share a label: give each game.local.json a "video" name with its date')
        export(tg)


if __name__ == "__main__":
    main()
