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
- Ball path coverage with the current defaults (min-conf 0.25, interpolation up to 0.5 s): A 34% detected (47% with interpolation, longest gap 12.4 s), B 40% detected (58%, longest gap 5.1 s). It was 52% and 63% detected before the confidence floor was raised: cleaner but sparser.
- HAND-CHECKED on two windows (owner labeled with ball_label.py, 119 frames each: clip A 150 to 210 s, clip B 100 to 160 s). Small samples, and accepting a shown prediction may anchor the labeler. Clip B is harder: the ball was not visible in 39% of its frames (16% in A), and the old defaults (min-conf 0.05, interpolation 0.5 s) claimed a ball on 89% of those, so precision was only 47% there (84% on A). Diagnosis: ghost and wrong detections have low confidence (median 0.08 to 0.13) while correct ones sit at 0.46 to 0.60; on B only 3 of 23 detections at conf 0.15 or below were right. Interpolation over low-confidence links produced mostly ghosts on B (4 of 36 correct).
- Tuning: swept the candidate confidence floor and interpolation length on both windows, tuned on the first half of each and checked on the second. Top settings (min-conf 0.25 to 0.35 with 0.5 to 1.0 s interpolation) form a flat plateau, mean F1 77 to 78, within noise of each other. Chose min-conf 0.25 with 0.5 s. Result: clip A 76% correct when visible, 16% ghosts, 92% precision; clip B 70%, 15%, 76% (before: A 85%, 53%, 84%; B 71%, 89%, 47%). Detected predictions are 98% (A) and 95% (B) correct. The trade is fewer positions: min-conf is the precision/recall knob.
- scan_density.py must require zoomed-in play: the first pick was a pre-game stretch (static wide shot, cones, huddles) that looked crowded by overlap alone.

## Phase 2 findings: team classification (team_classify.py)
- Roles: target, opponent, official, goalkeeper, other, plus unknown for short or unmeasurable tracklets. Role prototypes are numeric Lab centers in kit_prototypes.local.json (git-ignored, made by calibrate + assign on a reference clip). Kit colors never go in committed files.
- The cached patch medians were too contaminated by grass and skin on small players. classify re-measures colors from 8 frames per tracklet using non-grass pixels (about 16 s per clip with a sequential decoder; it was 8 min with per-frame seeking; cached in tracklet_colors.csv and re-measured automatically if the tracklets change). Clip frames are ci times the cache stride. This made the clusters clean where the cached colors gave mixed clusters.
- Color cannot separate players from people at the sideline (bench, coaches, vests) who wear the same kit. Each tracklet gets extent_h (path spread in body heights) and a sideline_suspect flag (8 s or more and extent under 0.75). Phase 3's pitch mask should decide on-pitch.
- Clip A (the calibration clip): median on-pitch counts per frame 7 target, 7 opponent, 1 goalkeeper, 1 official. This is circular because the prototypes came from this clip.
- Clip B (held-out, wider framing), sampled 8 tracklets per role by eye: opponent 7 of 8, official 3 of 3, other 8 of 8 non-players, target 5 of 8 (errors were sideline people and an official), goalkeeper 1 of 3 (two striped officials leaked in). Median confidence only 0.21 there, so most labels are low confidence. UNVERIFIED against labels, and samples are tiny.
- `classify --refine` (adapts prototypes to the clip) leaked target players into other and opponents into official in a first test, so it is opt-in and not recommended.
- Goalkeepers are one tracklet each per team and share color with officials in the hi-vis range. Treat the goalkeeper role as weak until identity (step 7) or labels exist.

## Phase 3 findings: pitch mask and calibration (pitch_mask.py, pitch_calibrate.py)
- pitch_mask.py measures the grass fraction in a window at each tracklet's feet (8 sampled frames, median). Combined with sideline_suspect it gives player_candidate in tracklet_roles.csv. team_classify.py now uses it.
- Clip A: only 3% of tracklets are off the pitch, because its bench sits on grass, so the motion flag (sitting still 8 s or more) does most of the work there. Clip B: 22% are off the pitch (track, stands).
- Sampled on clip B: candidates were 8 of 8 target and 7 of 8 opponent real players. Excluded tracklets were mostly non-players (bench, spectators, coaches), but roughly a third of excluded target and opponent tracklets looked like real players (near the touchline or briefly static), so recall drops. min-grass and the static rule are the knobs.
- Goalkeeper is unreliable on clip B: striped officials and the tan-kit goalkeeper land in the wrong classes. Not fixed.
- pitch_calibrate.py is anchor based: the owner reads pixel positions of known landmarks in a few frames (`pitch_calibrate.py frame` writes a gridded still) and gives pitch coordinates in meters, so no pitch size is assumed. Each anchor's homography is carried to other frames by the cached camera motion (modeled as a similarity, so error grows with the time from an anchor; `apply` reports the cross-anchor error in meters). Exact to 1 mm on synthetic similarity motion (`self-test`). NOT run on real footage, because there are no anchors yet.
- Shared frame reader: sv_common.read_frames decodes sequentially, about 30 times faster than seeking.

## Phase 4 findings: events (events.py)
- Contact and touches use detected ball positions only (`--allow-interpolated` restores the old behavior), because interpolated ones were often ghosts. With min-conf 0.25 the ball path is sparser, so events fell to clip A 25 touches, 30 possessions, 3 passes, 8 turnovers and clip B 33, 43, 3, 6 (from 48/45/6/9 and 120/81/15/25). Fewer but cleaner; the event thresholds still need hand-labeled events. The counts below are older.
- Detects possession segments, touches, passes and turnovers from ball_path.csv, best_tracklets.csv.gz and tracklet_roles.csv. Distances are in body heights, so it needs no pitch calibration. Shots are NOT detected (they need the goal position, so they wait for calibration).
- Every event has a confidence and ball_detected_share. Events under 0.5 go to review_queue.csv. Thresholds (contact 0.6 body heights, touch velocity change 2 body heights per second, possession 0.4 s, pass gap 3 s, same-player merge under 1.5 body heights) are hand-set guesses, not tuned.
- Clip A: 60 touches, 57 possessions, 9 passes, 17 turnovers in 4 min; possession time 42 s target, 25 s opponent; median possession 0.7 s; ball in contact for 33% of frames. Clip B: 138 touches, 103 possessions, 19 passes, 32 turnovers, median confidence 0.48, over half in the review queue.
- Sampled 16 events on clip A by eye: about 11 looked plausible (ball at the credited player's feet). Touches were weakest (2 of 4). The failures come from ball path errors (a resting spare ball near the bench, interpolated ball positions on empty grass) and sideline people counted as players. Confidence is only weakly informative: some wrong events scored 0.7. UNVERIFIED against labels.
- Tracklets fragment (about 13 IDs per player), so the pass count is rough. Handoffs under 1.5 body heights apart are merged as one player.
- Ball velocity is computed within each ball-path segment (time-aware), never across a break. Taking it across breaks gave a p99 speed of 74 body heights per s and 10 false touches on clip A (71 before, 60 after; p99 is now 14). Touch windows must also be three consecutive grid frames, so a hole inside a segment cannot stretch them. The 16-event sample above was taken before these fixes.

## Pipeline status
1. Ingest and detection cache: detect_cache.py (done, validated on two full clips)
2. Offline tracker replay and sweep: replay_trackers.py (done, validated; use --grid wide)
3. Ball linking: ball_link.py (done; hand-checked on one 60 s window, 97% correct when detected, interpolation weak)
4. Team classification: team_classify.py (done, unverified, goalkeeper weak)
5. Pitch: pitch_mask.py on-pitch test (done, sampled), pitch_calibrate.py anchor homography (tool done, self-tested, BLOCKED on owner anchors for metric coordinates)
6. Event detection: events.py (possession, touch, pass, turnover done and unverified; shots not started, need calibration)
7. Identity assignment with roster, tracklet stitching, review UI: NOT STARTED
8. Stats database and coaching tips: NOT STARTED

## Commands
- One command for the current stage:
  python run_all.py --video <game.mp4> --start HH:MM:SS --duration 300 --out data\clipA --grid wide --share-dir <folder Claude can read>
- Individual stages: detect_cache.py, replay_trackers.py, ball_link.py (each has --help)
- Events: events.py --run <run> [--montage] (needs ball_path.csv, tracklet_roles.csv)
- Team roles: team_classify.py calibrate / assign / classify (needs clip.mp4 in the run folder)
- Pick validation clips: scan_density.py (samples the whole game, use --reuse to re-pick from a saved scan)
- phase1_track.py is the original tracker-in-the-loop script. Superseded, kept for reference.

## Next actions
1. (Optional, owner) A third ball window from a different game would test whether min-conf 0.25 generalizes beyond these two clips.
2. (Needs the owner) Label roles for about 40 tracklets to score team_classify.py. Until then treat roles as unverified.
3. (Needs the owner) Give 4 or more landmark points, with pitch coordinates in meters, in 3 or more frames per clip so pitch_calibrate.py apply can run. Use `pitch_calibrate.py frame --run <run> --time <s>` to get a gridded still. Until then, downstream work uses image or stable coordinates in body heights.
4. (Needs the owner) Hand-label about 60 s of events (touches, passes, possession changes) to tune the events.py thresholds and score it.
5. STOP POINT REACHED (owner's limit was step 6). Step 7 (identity, roster, review UI) needs roster.csv, which does not exist, plus jersey anchors. Do not start it without the owner.
