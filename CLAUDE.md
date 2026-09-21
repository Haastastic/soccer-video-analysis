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
- Ball linking without a clutter filter locked onto static white objects (a sideline marker, an item on a cart) and stationary spare balls: 4 of 8 sampled path points were wrong. ball_link.py now drops weak candidates (conf below 0.6) that sit at the same stable spot for 3 s or more. Every one of 12 sampled removals was clutter or a spare ball. After the filter, 6 of 8 sampled detected points were the game ball, but only 1 of 4 interpolated points was correct. Sample sizes are tiny.
- Ball path coverage after filtering: A 43% detected (64% with interpolation, longest gap about 20 s), B 59% detected (95% with interpolation). UNVERIFIED against hand-labeled frames. Needs about 60 s of hand-checked ball positions.
- scan_density.py must require zoomed-in play: the first pick was a pre-game stretch (static wide shot, cones, huddles) that looked crowded by overlap alone.

## Pipeline status
1. Ingest and detection cache: detect_cache.py (done, validated on two full clips)
2. Offline tracker replay and sweep: replay_trackers.py (done, validated; use --grid wide)
3. Ball linking: ball_link.py (done with clutter filter, ball path unverified)
4. Team classification from cached kit colors: NOT STARTED (next)
5. Pitch calibration from visible lines: NOT STARTED
6. Event detection (possession, touch, pass, shot): NOT STARTED
7. Identity assignment with roster, tracklet stitching, review UI: NOT STARTED
8. Stats database and coaching tips: NOT STARTED

## Commands
- One command for the current stage:
  python run_all.py --video <game.mp4> --start HH:MM:SS --duration 300 --out data\clipA --grid wide --share-dir <folder Claude can read>
- Individual stages: detect_cache.py, replay_trackers.py, ball_link.py (each has --help)
- Pick validation clips: scan_density.py (samples the whole game, use --reuse to re-pick from a saved scan)
- phase1_track.py is the original tracker-in-the-loop script. Superseded, kept for reference.

## Next actions
1. (Needs the owner) Hand-check about 60 s of ball positions to score ball_path.csv. Until then treat the ball path as unverified.
2. Build team classification (target team vs opponent vs referee vs goalkeepers, colors in LOCAL_CONTEXT.md) from the cached torso and leg colors.
3. Start pitch calibration using the per-frame camera motion as the between-anchor transform.
