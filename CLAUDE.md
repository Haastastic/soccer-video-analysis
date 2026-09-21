# Soccer video analysis: project context

Personal project: per-player stats and coaching tips from follow-cam video of youth soccer games.
Windows, RTX 3060 Laptop GPU, VS Code, Python.

## Ground rules
- Videos, crops, rosters, and per-player outputs show minors. Keep them local. Never commit data/, roster.csv, *.mp4, *.pt, LOCAL_CONTEXT.md.
- Never put names, jersey numbers, school or team names, or kit colors in committed files, commit messages, or issues.
- Reports may be shared only as aggregate numbers (no images, no names). run_all.py --share-dir does this.
- Every stage writes its output to disk so any stage can be rerun and scored on its own.
- Every stat carries a confidence value and a visibility percentage. Low-confidence events go to a review queue.
- Be direct. Outline format with real detail. No filler.

## Footage facts
- 1080p, roughly 30 fps, elevated sideline auto-pan camera that follows the ball. Not a fixed wide shot.
- Players are about 60 to 90 px tall. Ball is about 10 px wide. Jersey numbers are not readable by plain OCR.
- Team kit colors and the target player's jersey number live in LOCAL_CONTEXT.md (git-ignored). Read it first if it exists.
- Because the camera follows the ball, off-ball players are often out of frame. Report running stats as percent of visible time.

## Decisions so far
- Chose the full-team automated pipeline, with a human review layer and roster constraints for identity.
- Detection: Ultralytics YOLO (yolo11m) at 1920 px, confidence floor 0.05, cached once.
- Tracking: never rely on raw tracker IDs for identity. Replay trackers offline, then stitch tracklets (planned).
- Ball: link candidates over time (ball_link.py). Fine-tune on hand-labeled frames later.
- Roster is a closed set of 21 players (one goalkeeper). Identity assignment uses roster constraints plus manual anchors.

## Phase 1 findings (5 min clip, BoT-SORT at 10 fps)
- Detection is fine: about 22 people per frame.
- Tracking is not: 790 IDs for about 23 people, 160 new IDs per minute. 60% of new IDs start where another track just ended.
- Camera pans reach about 60 px per 0.1 s at the 90th percentile, which drops box overlap to zero. Replaying trackers without camera motion gave over twice the IDs, so pan handling is the main lever.
- Ball: tracked run reported 14.9% but that was a tracker artifact. Detector-only baseline: 67.7% of frames at conf 0.1, but 22% of consecutive top-1 picks teleport. Needs linking and labels.

## Phase 1b validation (two full 5 min clips, 2026-09-21)
- Clip A: open play, players 60 to 90 px tall. Clip B: live play from a wider framing (about 27 people per frame, players about 35 to 50 px), ending in a celebration huddle. Chosen with scan_density.py.
- Detection cache: about 13.7 frames/s on the 3060, about 11 min per clip. Camera motion is solid (median 400 inliers, the feature cap, 1 weak frame in 9000).
- The Phase 1 config replays to 800 IDs (A) and 958 IDs (B), matching the original 790, so the replay is a faithful stand-in for the live tracker.
- Best config on both clips: BoT-SORT, 15 fps, high and new-track threshold 0.7, buffer 30 s, match threshold 0.95. IDs fall to 298 (A) and 330 (B), new IDs per minute to about 60, median track about 12 s. Every winning parameter sits at the edge of the "wide" grid, so the true optimum may be further out.
- Caution: the proxy score rewards leniency. A match threshold of 0.95 accepts boxes with IoU as low as 0.05, and the swap-suspect count only sees large jumps, not swaps between adjacent players. Verify on labeled data before trusting it. Still about 13 IDs per player, so tracklet stitching is required.
- ByteTrack is 3 to 5 times worse (1719+ IDs) and takes about 7 min per config at 30 fps. It is now opt-in (--trackers bytetrack,botsort).
- Ball linking without a clutter filter locked onto static white objects (a sideline marker, an item on a cart) and stationary spare balls: 4 of 8 sampled path points were wrong. ball_link.py now drops weak candidates (conf below 0.6) that sit at the same stable spot for one continuous stretch of 3 s or more (gaps up to 1 s allowed, separate revisits are not added together). Every one of 12 sampled removals from the first version was clutter or a spare ball. With the final version, 10 of 12 sampled detected points were the game ball; the misses were the weak white marker (conf 0.12) and an edge-of-frame blur (conf 0.05). Interpolated points are weaker (1 of 4 correct in the earlier sample). Sample sizes are tiny.
- Ball path coverage after filtering: A 52% detected (81% with interpolation, longest gap 5.1 s), B 63% detected (97% with interpolation, longest gap 1.3 s). UNVERIFIED against hand-labeled frames. Needs about 60 s of hand-checked ball positions.
- scan_density.py must require zoomed-in play: the first pick was a pre-game stretch (static wide shot, cones, huddles) that looked crowded by overlap alone.

## Phase 2 findings: team classification (team_classify.py)
- Roles: target, opponent, official, goalkeeper, other, plus unknown for short or unmeasurable tracklets. Role prototypes are numeric Lab centers in kit_prototypes.local.json (git-ignored, made by calibrate + assign on a reference clip). Kit colors never go in committed files.
- The cached patch medians were too contaminated by grass and skin on small players. classify re-measures colors from 8 frames per tracklet using non-grass pixels (about 8 min per clip because of video seeking, cached in tracklet_colors.csv, and re-measured automatically if the tracklets change). Clip frames are ci times the cache stride. This made the clusters clean where the cached colors gave mixed clusters.
- Color cannot separate players from people at the sideline (bench, coaches, vests) who wear the same kit. Each tracklet gets extent_h (path spread in body heights) and a sideline_suspect flag (8 s or more and extent under 0.75). Phase 3's pitch mask should decide on-pitch.
- Clip A (the calibration clip): median on-pitch counts per frame 7 target, 7 opponent, 1 goalkeeper, 1 official. This is circular because the prototypes came from this clip.
- Clip B (held-out, wider framing), sampled 8 tracklets per role by eye: opponent 7 of 8, official 3 of 3, other 8 of 8 non-players, target 5 of 8 (errors were sideline people and an official), goalkeeper 1 of 3 (two striped officials leaked in). Median confidence only 0.21 there, so most labels are low confidence. UNVERIFIED against labels, and samples are tiny.
- `classify --refine` (adapts prototypes to the clip) leaked target players into other and opponents into official in a first test, so it is opt-in and not recommended.
- Goalkeepers are one tracklet each per team and share color with officials in the hi-vis range. Treat the goalkeeper role as weak until identity (step 7) or labels exist.

## Pipeline status
1. Ingest and detection cache: detect_cache.py (done, validated on two full clips)
2. Offline tracker replay and sweep: replay_trackers.py (done, validated; use --grid wide)
3. Ball linking: ball_link.py (done with clutter filter, ball path unverified)
4. Team classification: team_classify.py (done, unverified, goalkeeper weak, needs pitch mask to drop sideline people)
5. Pitch calibration from visible lines: NOT STARTED (next)
6. Event detection (possession, touch, pass, shot): NOT STARTED
7. Identity assignment with roster, tracklet stitching, review UI: NOT STARTED
8. Stats database and coaching tips: NOT STARTED

## Commands
- One command for the current stage:
  python run_all.py --video <game.mp4> --start HH:MM:SS --duration 300 --out data\clipA --grid wide --share-dir <folder Claude can read>
- Individual stages: detect_cache.py, replay_trackers.py, ball_link.py (each has --help)
- Team roles: team_classify.py calibrate / assign / classify (needs clip.mp4 in the run folder)
- Pick validation clips: scan_density.py (samples the whole game, use --reuse to re-pick from a saved scan)
- phase1_track.py is the original tracker-in-the-loop script. Superseded, kept for reference.

## Next actions
1. (Needs the owner) Hand-check about 60 s of ball positions to score ball_path.csv. Until then treat the ball path as unverified.
2. (Needs the owner) Label roles for about 40 tracklets to score team_classify.py. Until then treat roles as unverified.
3. Start pitch calibration using the per-frame camera motion as the between-anchor transform.
