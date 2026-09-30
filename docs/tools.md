# Tool guide

How to run each stage and each hand-labeling tool. Every script also has `--help`. Paths are examples. Everything
written goes under the git-ignored `data/` folder, and every stage refuses to write anywhere else.

- [A new game](#a-new-game)
- [One window by hand](#one-window-by-hand)
- [Labeling windows and zoom](#labeling-windows-and-zoom)
- [Checking stages by hand](#checking-stages-by-hand)
- [Identity](#identity)
- [Pitch](#pitch)
- [Ball model](#ball-model)
- [Coaching pages and the site](#coaching-pages-and-the-site)

## A new game

```powershell
python new_game.py plan --video "videos\<game>.mp4" --game g0922
python new_game.py confirm --game g0922 [--first-half 00:00:30-00:40:00 --second-half 00:47:00-01:26:00]
python new_game.py setup --game g0922      # stops once per half for the kit mapping below
python team_classify.py assign --run data\g0922\w0500 --map 4:target,0:opponent,3:opponent,1:other
python new_game.py run --game g0922        # every window, about 3 hours
python new_game.py numbers --game g0922    # optional jersey round (about 5 minutes of clicking), then --apply
python new_game.py publish --bucket <bucket>
```

1. **`plan`** scans the video's crowd density and proposes the halves, with a contact sheet
   (`data/<game>/live_play_check.jpg`). If it finds no halftime break, it writes `overview_check.jpg` (a frame
   every 2 minutes) instead. Warm-ups and other teams after the game look like play, so check the sheet.
2. **`confirm`** saves the halves.
3. **`setup`** runs a pilot window in each half, clusters kit colours, and stops for the mapping. Look at
   `kit_clusters.png` in the pilot window: each row is one cluster. Map ours as `target`, the other team as
   `opponent`, and referees, bench and spectators as `official` or `other`. Map our goalkeeper as `goalkeeper` if
   they have a cluster of their own. A mixed cluster goes to its biggest group.
4. **`run`** processes every window. It then works out which goal we defend (from where our goalkeeper stands,
   cross-checked with the second-half kickoff) and stops if the two disagree. Last, it writes stats and pages.
5. **`numbers`** is an optional round: candidate crops of jersey numbers whose reads conflict, to confirm by
   clicking. `numbers --seed` is for a new team's first game: it shows every roster number, since the reader hasn't
   seen that team yet.
6. **`publish`** exports every team, offers the photo picker for players without a photo, lists the files and asks
   before uploading.

**A new team:**
1. Set `"team": "<id>"` in the game's `game.local.json` before `setup`.
2. Put its roster in `data/teams/<id>/roster.csv`, with the columns `jersey,name,goalkeeper,class_of,position`.
3. Run `numbers --seed` after `run`.

## One window by hand

```powershell
python run_all.py --video videos\<game>.mp4 --start 00:14:00 --duration 300 --out data\clipA
python scan_density.py --video videos\<game>.mp4 --out data --exclude 00:14:00-00:19:00   # pick a busy window
```

`run_all.py` runs detection (once, cached), tracker replay and ball linking. Then run the stages in order, each
with `--run data\clipA`:

- `pitch_mask.py`
- `team_classify.py classify`
- `tracklet_stitch.py`
- `events.py`
- `pitch_ptz.py run`, then `pitch_calibrate.py apply --anchors data\clipA\pitch_anchors_ptz.local.json`
- `jersey_auto.py identify --from <labeled windows> --write`
- `player_stats.py`

`run_windows.py` does all of this for a list of windows, resumably.

What a window holds after `run_all.py`:

- `report.md`: everything in one place; read this first.
- `sweep_results.csv`: every tracker configuration and its proxy metrics.
- `best_tracklets.csv.gz`: tracks from the chosen configuration, with kit colours attached.
- `ball_path.csv`: one ball position per frame where a plausible path exists, marked detected or interpolated.
- `cache\`: `detections.csv.gz`, `camera.csv` and `meta.json`, the input to every later stage.

In `report.md`:

- **Camera inliers:** the median should be well above 50. If it's low, camera motion is unreliable, and tracking
  is too.
- **Tracker score:** new IDs per minute plus 3 × swap suspects per minute; lower is better. It's a proxy: the
  chosen configuration came from hand labels of whether a track stays on one person (`track_purity_label.py`).

## Labeling windows and zoom

Every labeling window shares one layout, from `sv_common.ZoomView`:

- **Mouse wheel:** zooms, keeping the point under the cursor.
- **Right click:** centres the view there.
- **`r`:** resets the zoom.
- **Window size:** it grows with zoom until it fills the screen, then scroll bars appear.
- **Between items:** the zoom and view carry over.
- **Crop tools:** they show all of an item's crops at once, in time order.

Progress is saved after every item, so you can close a tool and rerun it to resume.

## Checking stages by hand

### Team roles

```powershell
python role_label.py label --run data\clipA --n 20
python role_label.py score --run data\clipA           # writes role_score.json
```

Shows each tracklet's crops with the predicted role and its confidence.

| Key | Action |
|---|---|
| `y` | Accept the prediction |
| `t` `o` `f` `g` `x` | Target, opponent, official, goalkeeper, other |
| `s` | Skip |
| `b` | Back |
| `q` | Quit |

### Ball path

```powershell
python ball_label.py label --run data\clipA --start 150 --duration 60
python ball_label.py score --run data\clipA           # writes ball_score.json
```

Shows one frame every 0.5 s, with the predicted ball circled.

| Key or click | Action |
|---|---|
| `y` | The circle is on the ball |
| Left click | Mark where the ball is |
| `x` | Ball not visible |
| `s` | Skip |
| `b` | Back |
| `q` | Quit |

`score` reports how often the path is on the ball, off it, or missing, split into detected and interpolated
positions.

### Events

```powershell
python events.py --run data\clipA --montage
python event_label.py label --run data\clipA --start 150 --duration 60
python event_label.py score --run data\clipA          # writes events_score.json
```

Labeling shows one frame every 0.5 s with every tracked person boxed: click who has the ball. Possession, touches,
passes and turnovers are scored against those labels.

### Track purity (tracker choice)

```powershell
python track_purity_label.py label --run data\clipA --sweep <name> --configs 7,24 --n 20
python track_purity_label.py score --run data\clipA --sweep <name>
```

The configurations are blind: you don't see which one a track came from. Group each track's crops by person.

| Input | Action |
|---|---|
| Click | Cycles the crop's person, A to H |
| Shift+click | Marks a swap from that crop on |
| Ctrl+click | Marks the crop as unknown |
| Enter | Saves |

## Identity

```powershell
python jersey_auto.py identify --run data\g0922\w0500 --from data\clipA,data\clipB,data\clipE --write
python jersey_auto.py finetune --runs data\clipA,data\clipB,data\clipE
python jersey_rare_label.py candidates --runs <windows> --labeled <labeled windows> [--auto | --numbers 88,99]
python jersey_rare_label.py label        # one screen per number: click the crops that show it, Enter
python jersey_label.py label --run data\clipA   # full hand labeling of a window (about 1.5 hours)
python jersey_label.py apply --run data\clipA
```

- Without `--write`, `identify` is a dry run. On a hand-labeled window, it scores itself against the labels.
- `finetune` retrains the shared jersey reader from the labeled windows plus every confirmed crop in
  `data/jersey_rare_truth.csv`.

In `jersey_label.py`, a stitched player shows their tracklets, with a yellow bar at each join.

| Input | Action |
|---|---|
| Type a number, then Enter | Name the player |
| Click a crop | Name just that tracklet |
| Shift+click a crop | Split the tracklet where another person starts |
| `x` | Finish the player as split |

## Pitch

```powershell
python pitch_ptz.py run --run data\clipA         # pan, tilt and zoom per frame from the painted lines
python pitch_calibrate.py apply --run data\clipA --anchors data\clipA\pitch_anchors_ptz.local.json
python pitch_ptz.py evaluate                     # held-out check against hand-placed anchors
python pitch_anchor_ui.py --run data\clipA       # local browser UI for hand anchors, if a camera fit is needed
```

- The camera model (`pitch_camera.local.json`) comes from `pitch_ptz.py fit` on a few hand-anchored windows.
- For a new game, `new_game.py setup` refits it from the painted lines if the pilot window needs that.
- X is metres from one fixed reference goal. Y is metres across the pitch, positive toward the camera side.

## Ball model

```powershell
python ball_label.py label --run data\g0922\w0500 --start 20 --duration 150
python ball_finetune.py dataset --runs <labeled windows> --val <one window> --out data\_ball_ft\venue
python ball_finetune.py train --data data\_ball_ft\venue --name venue
python ball_finetune.py apply --run data\g0922\w0500    # the pipeline stage (models/ball/venue.pt)
```

## Coaching pages and the site

```powershell
python player_stats.py --runs data\clipA,data\clipB
python coaching_tips.py --runs <windows of one team>      # one game: data/<game>/coaching/; several: data/coaching_games*/
python site_export.py --runs <windows of every team>
python player_photo.py --team jv candidates --runs <windows>
python player_photo.py --team jv label                    # click one photo per player; s = no photo
python publish_site.py --bucket <bucket> --dry-run
```

To deploy and redeploy the site, see [webapp/README.md](../webapp/README.md).
