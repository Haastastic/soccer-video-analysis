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

## Pipeline status
1. Ingest and detection cache: detect_cache.py (done, untested on real footage)
2. Offline tracker replay and sweep: replay_trackers.py (done, untested with real camera data)
3. Ball linking: ball_link.py (done)
4. Team classification from cached kit colors: NOT STARTED (next)
5. Pitch calibration from visible lines: NOT STARTED
6. Event detection (possession, touch, pass, shot): NOT STARTED
7. Identity assignment with roster, tracklet stitching, review UI: NOT STARTED
8. Stats database and coaching tips: NOT STARTED

## Commands
- One command for the current stage:
  python run_all.py --video <game.mp4> --start HH:MM:SS --duration 300 --out data\clipA --share-dir <folder Claude can read>
- Individual stages: detect_cache.py, replay_trackers.py, ball_link.py (each has --help)
- phase1_track.py is the original tracker-in-the-loop script. Superseded, kept for reference.

## Next actions
1. Run run_all.py on an open-play clip, then a crowded clip. Check the report, especially camera inliers and the best tracker config.
2. Watch the ball_path.csv result against 60 seconds of hand-verified ball positions to score linking.
3. Build team classification (target team vs opponent vs referee vs goalkeepers, colors in LOCAL_CONTEXT.md) from the cached torso and leg colors.
4. Start pitch calibration using the per-frame camera motion as the between-anchor transform.
