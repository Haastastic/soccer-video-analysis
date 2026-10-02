# Soccer video analysis: project context

Personal project: per-player stats and coaching tips from follow-cam video of youth soccer games.
Windows, RTX 3060 Laptop GPU, VS Code, Python.

## Ground rules
- Videos, crops, rosters, and per-player outputs show minors. Keep them local. Never commit data/, roster.csv, *.mp4, *.pt, LOCAL_CONTEXT.md.
- Never put names, jersey numbers, school or team names, or kit colors in committed files, commit messages, or issues.
- Reports may be shared only as aggregate numbers (no images, no names). run_all.py --share-dir does this.
- ONE EXCEPTION (owner decision, 2026-09-30): the coaching site (webapp/, Phase 21). Per-player pages with names and
  a chosen photo leave this computer only through publish_site.py (allow-listed files) into a private bucket, served
  by Cloud Run to invited Google accounts (admin / coach / parent-of-listed-players). Consent is the owner's, handled
  offline. School and opponent names and logos are entered in the site's admin page (Firestore), never in git.
  WIDENED (owner decision, 2026-10-02): the site also carries short clips (10 s, no audio, cropped to follow one of
  our players, site_clips.py) of each player's moments, served only to viewers allowed to see that player. Full
  game videos never leave this computer.
- Every stage writes its output to disk so any stage can be rerun and scored on its own.
- Every stat carries a confidence value and a visibility percentage. Low-confidence events go to a review queue.
- Be direct. Outline format with real detail. No filler.
- Every clip review/labeling tool needs zoom (owner rules, 2026-09-23/24). Crop tools (track_purity_label.py, jersey_label.py, role_label.py) share one layout and zoom from sv_common: all of an item's crops at once in time order, left to right then top to bottom, in a grid sized to the screen (crop_tile, tile_grid, grid_cols) - no paging. ZoomView: mouse wheel zooms keeping the point under the cursor, right click centers, r resets; the window grows with zoom up to the screen size, then scroll bars appear (drag or click); status text goes below the image (add_footer). event_label.py and ball_label.py are single-frame viewers on the same ZoomView (full-resolution frame, base scale fits the screen at 1x). EVERY labeling window grows with zoom until the screen constrains it, then scroll bars, and the zoom and view carry over when it moves to the next item (owner rules, 2026-09-28): any new labeling tool must use sv_common.ZoomView. pitch_anchor_ui.py is a browser UI.

## Footage facts
- 1080p, roughly 30 fps, elevated sideline auto-pan camera. Not a fixed wide shot. CORRECTED 2026-09-23 (owner): it does NOT follow the ball - earlier phases assumed it did and reasoned from that; those specific causal claims are wrong and are being corrected as they're found (see Phase 5 findings for one). What the pan actually follows or is driven by is not established; do not assume ball-following, and do not invent a replacement cause without asking.
- Players are about 60 to 90 px tall. Ball is about 10 px wide. Jersey numbers are not readable by plain OCR - but ARE sometimes readable by the owner directly (human eyes, with zoom), when a back-facing frame with the number visible happens to be sampled. Tools that ask the owner to identify a player (jersey_label.py) should give a good chance of finding such a frame, not just recognition by build/kit.
- Team kit colors and the target player's jersey number live in LOCAL_CONTEXT.md (git-ignored). Read it first if it exists.
- Off-ball players are often out of frame (observed; the panning camera does not hold a fixed wide shot). Report running stats as percent of visible time.

## Decisions so far
- Chose the full-team automated pipeline, with a human review layer and roster constraints for identity.
- Detection: Ultralytics YOLO (yolo11m) at 1920 px, confidence floor 0.05, cached once.
- Tracking: never rely on raw tracker IDs for identity. Replay trackers offline, then stitch tracklets. Config in use (2026-09-24): box-only BoT-SORT, 15 fps, high/new 0.7, buffer 1 s, match 0.95 (replay_trackers.py CHOSEN, --grid chosen, the default), chosen by blind owner purity labels (Phase 6).
- Ball: link candidates over time (ball_link.py). Venue ball model (ball_finetune.py, Phase 14) for game 2 and new games.
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
- GENERALIZATION CHECK on a third hand-labeled window (a different part of the same game, 43:00 to 48:00, verified as live single-ball play before running detection): precision held up (75%, versus 92% and 76% on A and B) and detected-frame correctness stayed in range (84%, versus 98% and 95%), so min-conf 0.25 is not overfit to the first two windows for what it was tuned to fix. Overall "correct when visible" was lower (46%, versus 76% and 70%) because of a separate problem: 44% of visible-ball frames had no detector candidate at all, at any confidence down to 0.05, so no threshold choice could have caught them. Of those complete misses, 61% were frames where the ball sat on the boundary running track rather than the grass pitch: a background the detector handles much worse. This is a detector-recall limitation, not a ball_link.py tuning problem, and would need retraining or hard examples of an in-play ball on non-grass backgrounds to fix.
- Picking that third window surfaced a scan_density.py gap: its first automatic pick (about 100 minutes into the 120-minute recording) was a multi-ball shooting drill with several real balls scattered on the field at once, not match play. The detector was right to flag every ball; the scene just breaks the pipeline's one-ball assumption. scan_density.py's height and crowding heuristics do not distinguish a drill from a match, so a candidate window should be watched for a few seconds before spending GPU time on it. Live 11-a-side play in this recording appears to run through roughly the first 80 minutes; time past that was warm-down or another team's drills in the frames checked.
- scan_density.py must require zoomed-in play: the first pick was a pre-game stretch (static wide shot, cones, huddles) that looked crowded by overlap alone.

## Phase 2 findings: team classification (team_classify.py)
- Roles: target, opponent, official, goalkeeper, other, plus unknown for short or unmeasurable tracklets. Role prototypes are numeric Lab centers in kit_prototypes.local.json (git-ignored, made by calibrate + assign on a reference clip). Kit colors never go in committed files.
- The cached patch medians were too contaminated by grass and skin on small players. classify re-measures colors from 8 frames per tracklet using non-grass pixels (about 16 s per clip with a sequential decoder; it was 8 min with per-frame seeking; cached in tracklet_colors.csv and re-measured automatically if the tracklets change). Clip frames are ci times the cache stride. This made the clusters clean where the cached colors gave mixed clusters.
- Color cannot separate players from people at the sideline (bench, coaches, vests) who wear the same kit. Each tracklet gets extent_h (path spread in body heights) and a sideline_suspect flag (8 s or more and extent under 0.75). Phase 3's pitch mask should decide on-pitch.
- Clip A (the calibration clip): median on-pitch counts per frame 7 target, 7 opponent, 1 goalkeeper, 1 official. This is circular because the prototypes came from this clip.
- Clip B (held-out, wider framing), sampled 8 tracklets per role by eye: opponent 7 of 8, official 3 of 3, other 8 of 8 non-players, target 5 of 8 (errors were sideline people and an official), goalkeeper 1 of 3 (two striped officials leaked in). Median confidence only 0.21 there, so most labels are low confidence. UNVERIFIED against labels, and samples are tiny.
- `classify --refine` (adapts prototypes to the clip) leaked target players into other and opponents into official in a first test, so it is opt-in and not recommended.
- Goalkeepers are one tracklet each per team and share color with officials in the hi-vis range. Treat the goalkeeper role as weak until identity (step 7) or labels exist.
- HAND-CHECKED with role_label.py (owner, stratified sample across predicted roles, 20 tracklets per clip): 17 of 20 usable on clip A (3 skipped), 11 of 20 usable on clip B (9 skipped). Accuracy 64.7% (A) and 72.7% (B, tiny sample: single-digit support per role). Target and official are the most reliable, 100% precision on both clips. "Other" (non-players) is the weak point: only 40 to 50% recall, so about half of true non-players get misclassified as a real role. Confidence tracks correctness on A (mean 0.69 correct vs 0.16 wrong) but is compressed on B (0.25 vs 0.07).
- The high skip rate on clip B (9 of 20, versus 3 of 20 on A) is not random: in at least 6 of those 9, the tracklet's box visibly follows two different people mid-lifetime (a player then a different player, or a player then match-official staff). One more is a real goalmouth scrum with several players overlapping. This is downstream of tracker ID fragmentation (about 13 fragments per real player, noted above): a tracklet whose box straddles two people gets a kit color that is the median of both, which plausibly explains why wrong predictions have much lower confidence than correct ones. Role accuracy is capped by tracklet quality and will not improve much until tracklets are stitched (step 7), which is past the current stop point. Retuning team_classify.py's thresholds on this sample was not attempted: per-role support is single digits, too small to trust a parameter change.

## Phase 3 findings: pitch mask and calibration (pitch_mask.py, pitch_calibrate.py)
- pitch_mask.py measures the grass fraction in a window at each tracklet's feet (8 sampled frames, median). Combined with sideline_suspect it gives player_candidate in tracklet_roles.csv. team_classify.py now uses it.
- Clip A: only 3% of tracklets are off the pitch, because its bench sits on grass, so the motion flag (sitting still 8 s or more) does most of the work there. Clip B: 22% are off the pitch (track, stands).
- Sampled on clip B: candidates were 8 of 8 target and 7 of 8 opponent real players. Excluded tracklets were mostly non-players (bench, spectators, coaches), but roughly a third of excluded target and opponent tracklets looked like real players (near the touchline or briefly static), so recall drops. min-grass and the static rule are the knobs.
- Goalkeeper is unreliable on clip B: striped officials and one team's goalkeeper land in the wrong classes. Not fixed.
- pitch_calibrate.py is anchor based: the owner reads pixel positions of known landmarks in a few frames and gives pitch coordinates in meters, so no pitch size is assumed. X = distance from ONE fixed reference goal (the same goal in every anchor - not whichever goal happens to be visible), Y = across the field, positive toward the touchline nearer the camera. Each anchor's homography is carried to other frames by the cached camera motion (modeled as a similarity, so error grows with the time from an anchor; `apply` reports the cross-anchor error in meters). Exact to 1 mm on synthetic similarity motion (`self-test`).
- `pitch_anchor_ui.py --run <run>` is a local-only browser UI (127.0.0.1, no external calls, since frames show minors) for placing anchors: click landmarks on a frame with a zoomable crosshair, quick-pick chips for the 9 standard Law-of-the-Game points (labeled "near/far touchline", not +Y/-Y - see below) fill in the pitch meters, live reprojection RMS, saves the same JSON `pitch_calibrate.py` reads. On save it also flags any anchor whose homography orientation disagrees with the others (a fixed sideline camera should keep the same handedness across anchors), which is the check that caught the bug below.
- CLIPB CALIBRATION DONE (2026-09-22): a first attempt at a 4th anchor (30s) was wildly inconsistent with the other three (67 to 152 m error), both before and after relabeling its points with near/far touchline wording, which ruled out a sign-convention mixup. Root cause, confirmed by the owner: that frame showed the OPPOSITE goal from the other three (the camera had panned to the far end of the pitch at that timestamp; see the Footage facts correction on why - not because it follows the ball). This calibration's X=0 is one fixed reference goal, not whichever goal is visible, and converting a landmark measured from the other goal needs the exact pitch length - which anchor-based calibration deliberately avoids assuming. Not fixable by relabeling; fixed by swapping in a different anchor frame.
- Replaced it with an anchor at 130s (same reference goal, confirmed by the orientation-mismatch warning showing none). Final 4 anchors (130s, 180.1s, 197s, 270s) all cross-check cleanly with each other: 0.5 to 5.05 m error across every pair, including the largest gap tested (140 s) - real validation that the camera-motion propagation holds up across most of the clip's duration. `tracklet_pitch_xy.csv.gz` and `pitch_calibration_report.json` in data/clipB reflect this set. clipB's first ~130s is extrapolated past the nearest anchor rather than directly cross-checked, but the observed drift rate (roughly 3-5 m per 140 s) suggests it's in a similar range.
- CLIPA CALIBRATION DONE WITH A KNOWN, ACCEPTED LIMITATION (2026-09-22): 6 anchors (4s, 66s, 96s, 125s, 156s, 220s), no orientation warnings, `tracklet_pitch_xy.csv.gz` in data/clipA reflects this set. Verified each anchor's own points are precisely and correctly placed with a leave-one-out test (fit the homography from all-but-one point, check how well it predicts the held-out point, no camera motion involved) - sub-2 m for every full anchor. Despite that, cross-anchor error is much higher than clipB's: even the tightest pairs (29 to 31 s gaps) show 3 to 8 m error, growing to 20 to 60 m for the largest gaps (worst involving 220s). Measured why: clipA's cached camera motion is about 3.5x more volatile than clipB's (median 2 s-step scale change 0.030 vs 0.0085, sampled from the cumulative transform) - this game's camera zooms and pans more actively in clipA's time range, and the "camera motion modeled as a similarity" propagation approximation accumulates real error faster as a result. More/denser anchors reduce the worst-case tail (coverage within 10s of an anchor went 27% to 38.8% after 2 more anchors) but do not beat this floor - confirmed by testing the closest anchor pairs directly. ACCEPTED: use clipA's pitch positions knowing they carry several-meters-plus of uncertainty away from an anchor, worse than clipB's. Revisit only if a downstream feature turns out to need tighter precision than this.
- A DEAD END WORTH NOT REPEATING: one of clipA's early anchors (87.3s) was fundamentally invalid, not just thin - its points were placed on the CENTER CIRCLE, not the penalty arc. The center circle and penalty arc are both exactly 9.15 m radius by law, just centered on a different reference point (center spot vs. penalty spot, tens of meters apart), so they look identical in shape and are an easy mix-up on a frame that turns out to be showing midfield, not the goal end. This fit its own frame fine (locally self-consistent, since the two circles are the same size) but was numerically incompatible with every goal-area anchor, and unlike a thin/partial anchor, it could not be fixed by adding more points - the frame doesn't show the goal at all, so no set of correctly-labeled points there can be converted to this calibration's goal-referenced coordinates. Fixed by dropping it (66s and 96s already bracketed that time closely, so no replacement was needed). When a frame's cross-check is bad, check what it's actually showing before adding more points to it.
- Shared frame reader: sv_common.read_frames decodes sequentially, about 30 times faster than seeking.

## Phase 4 findings: events (events.py)
- Contact and touches use detected ball positions only (`--allow-interpolated` restores the old behavior), because interpolated ones were often ghosts. With min-conf 0.25 the ball path is sparser, so events fell to clip A 25 touches, 30 possessions, 3 passes, 8 turnovers and clip B 33, 43, 3, 6 (from 48/45/6/9 and 120/81/15/25). Fewer but cleaner; the event thresholds still need hand-labeled events. The counts below are older.
- Detects possession segments, touches, passes and turnovers from ball_path.csv, best_tracklets.csv.gz and tracklet_roles.csv. Distances are in body heights, so it needs no pitch calibration. Shots are NOT detected (they need the goal position, so they wait for calibration).
- Every event has a confidence and ball_detected_share. Events under 0.5 go to review_queue.csv. Thresholds (contact 0.6 body heights, touch velocity change 2 body heights per second, possession 0.4 s, pass gap 3 s, same-player merge under 1.5 body heights) are hand-set guesses, not tuned.
- Clip A: 60 touches, 57 possessions, 9 passes, 17 turnovers in 4 min; possession time 42 s target, 25 s opponent; median possession 0.7 s; ball in contact for 33% of frames. Clip B: 138 touches, 103 possessions, 19 passes, 32 turnovers, median confidence 0.48, over half in the review queue.
- Sampled 16 events on clip A by eye: about 11 looked plausible (ball at the credited player's feet). Touches were weakest (2 of 4). The failures come from ball path errors (a resting spare ball near the bench, interpolated ball positions on empty grass) and sideline people counted as players. Confidence is only weakly informative: some wrong events scored 0.7. UNVERIFIED against labels.
- Tracklets fragment (about 13 IDs per player), so the pass count is rough. Handoffs under 1.5 body heights apart are merged as one player.
- Ball velocity is computed within each ball-path segment (time-aware), never across a break. Taking it across breaks gave a p99 speed of 74 body heights per s and 10 false touches on clip A (71 before, 60 after; p99 is now 14). Touch windows must also be three consecutive grid frames, so a hole inside a segment cannot stretch them. The 16-event sample above was taken before these fixes.
- HAND-CHECKED, first real score (2026-09-22, event_label.py, clipA 150 to 210 s, 120 frames at 0.5 s steps: 37 possessed, 63 loose, 18 not-visible, 2 skipped). At events.py's real 0.5 confidence threshold (what actually lands in events.csv as trusted, not review_queue.csv): possession 56.2% recall / 69.2% precision (16 true, 13 predicted) - the strongest event type. Touch is the weakest, confirming the earlier eyeball sample: 21.4% recall / 33.3% precision (14 true, 9 predicted) - most real touches are missed, and most predicted touches are wrong. Turnover: 1 true event in this window, recovered (100% recall) but with 2 false positives (33.3% precision); pass: 0 true events, 1 false positive. Small sample (especially 1 turnover, 0 passes) - do not retune thresholds on this alone, but the touch weakness is consistent with two independent samples now.
- TRIED AND REVERTED: hypothesized that the same-player merge guard (`MIN_PASS_H`, currently only applied when the two predicted teams match) should also apply to turnovers, since a misclassified-role fragment would show as a team mismatch and evade it. Implemented, regenerated events.csv, rescored: turnover precision improved (14.3% to 33.3% at confidence 0) but recall dropped to 0%, because the single real hand-confirmed turnover in this sample happened at 0.28 body heights separation - a genuine close-range steal, exactly the low-separation signature the fix assumed meant "same player." Reverted. Lesson: low separation is not on its own evidence of a tracking artifact for a turnover the way it is for a pass, because real turnovers often happen at close range by nature (contests, tackles, interceptions) while real passes do not start and end in the same spot.
- Found and fixed a real bug while scoring: `events.detect()` accepts `min_confidence` but never used it - only `events.py`'s own CLI applied it, after the call. `event_label.py score --min-confidence` was a no-op before this was caught (0.5 and 0.0 gave identical scores). Fixed in event_label.py.

## Phase 5 findings: step 7, tracklet stitching (tracklet_stitch.py)
- roster.csv exists (2026-09-23): 21 players, one goalkeeper, from owner-provided screenshots. Git-ignored, matches LOCAL_CONTEXT.md's target-player jersey number.
- First real attempt extrapolated a straight-line velocity from each tracklet's last few frames to predict where the next one should start. This badly overpredicted position error for real gaps (players change direction within a second or two), rejecting almost every genuine match: only 70 candidate edges out of 198 clipA tracklets, target-team "players" barely reduced from 84 to 64 against the 21-player roster. Replaced with a simple bound (a fixed slack plus a max-speed-per-second allowance from the last known position, not a velocity prediction).
- With the position model fixed, the real bottleneck was the max-gap setting: an off-ball player can leave frame for many seconds, not just a brief occlusion (originally attributed to the camera following the ball - CORRECTED, see Footage facts: it does not; the real cause is unconfirmed), so a 3 s cap covered only 57% of even the closest real same-role tracklet gaps. Widened empirically (15 s max gap, 2.5 body-heights/s speed bound) until target-team players stopped grossly outnumbering the roster - then found, from 35 real jersey_label.py verdicts, that this widening had overshot badly (51% of stitched target/goalkeeper players were actually 2+ different people). Retuned against that real data instead of the roster-count proxy: position distance on the actual merge decisions separates correct from incorrect merges somewhat (median 1.87 vs 2.98 body heights) while kit color shows none at all. New defaults (10 s gap, 1.0 body-height base slack, 0.5 body-heights/s speed) raise estimated precision among kept merges from about 32% to about 48% - real improvement, not a fix; no clean separating threshold exists on these signals. jersey_label.py's "mixed" (x) verdict remains the real safety net.
- Kit color barely discriminates WITHIN a team, since everyone on one side wears the same kit - it only confirms "same team," which the role match already does. Motion continuity is doing essentially all the real work, and that is inherently ambiguous with several same-team players moving through the same area at once.
- Result on both clips (checked for generalization, not tuned to one): tracklets-per-player goes from 1.0 (no merging) to 3.1 (clipA target), 2.5 (clipA opponent), 2.3 (clipB target), 1.9 (clipB opponent). Real progress, well short of the approximately 1 a fully correct stitch would give.
- Spot-checked `stitch_montage.png` by eye on clipA (3 stitched players): 2 clearly correct, one confirmed by a visible jersey number legible across crops. 1 likely false merge, at a contested-ball moment where a target and an opponent player overlap - a known hard case for this pipeline generally (Phase 2 already noted tracklets that straddle two people during overlaps), not a new failure mode.
- Does NOT fix a tracklet that already drifted onto a different real person mid-lifetime (separate, pre-existing tracker bug). This is a first-pass proposal for manual review, not a finished identity system.

## Phase 5b findings: step 7, jersey-based identity (jersey_label.py)
- Built jersey_label.py: shows one stitched target/goalkeeper player at a time (up to 32 sampled crops, paged 4 at a time, zoomable, green box marking which figure is the one being identified), the owner types a jersey number to confirm against roster.csv. The owner reported build/kit recognition alone (the tool's original premise) is unreliable for them - what actually works is reading the printed number off the back when a crop catches that angle, so sampling was increased specifically for this (see [[owner-identifies-players-by-jersey-number]]).
- Two more verdicts added from real usage, beyond confirmed/not_on_roster/skipped: "mixed" (x) when the crops show two different physical people (tracklet_stitch.py over-merged), and "bench" (o) for a substitute in a bib - a real roster player, but not on-field play, so crediting their tracked position to that player's stats would be wrong. Both get no identity in the output, same practical effect as not_on_roster, but reported separately since each is a different kind of upstream feedback (mixed -> tracklet_stitch.py; bench -> pitch_mask.py/team_classify.py's on-pitch check).
- A THIRD failure mode was found but not yet given a verdict: a stitched player_id whose tracklets span a substitution transition itself (part of it is the player walking off and putting on a bib, part is them still playing) - neither "mixed" (one real person, not two) nor "bench" (some of it IS real play) fits. Guidance for now: skip it. Only one instance reported so far; worth a dedicated fix (splitting a tracklet at the transition frame, not just accepting/rejecting the whole group) if it turns out to be common rather than a one-off.
- FIRST FULL IDENTITY PASS on clipA (2026-09-23, 69 stitched target/goalkeeper players labeled): 23 confirmed (33%), 29 mixed (42%), 5 not_on_roster, 12 skipped, 0 bench (the one substitution-transition case was skipped, not marked bench). Mixed rate improved from the initial 51% (pre-retune) to 42% (post-retune) - matches the ~48% kept-merge precision estimate from the threshold retuning.
- IMPORTANT: the final player_identity.csv is more complete than the 23-confirmed number suggests, because jersey number is the true identity key, not the stitched player_id - a player fragmented into several separate stitched groups (tracklet_stitch.py under-merging, still real: one jersey alone was split across 5 different player_ids) still gets ALL of those track_ids correctly attributed to the same jersey/name once the owner confirms each group. Verified directly: all 5 of that jersey's groups map to the same name in player_identity.csv. Net result: 26 of 95 target/goalkeeper tracklets (27%) now carry a real identity, spanning 11 of the 21 roster players seen in this 5-minute window. roster_players_not_seen in the report lists who did not appear (bench for the whole clip, or missed).
- Roster-based identity assignment is DONE, first pass, for clipA. The review UI (third piece of step 7) has not been started. clipA still has 5 not_on_roster and 29 mixed player_ids whose tracklets carry no identity at all - real gaps, not resolved by this pass.
- SECOND FULL IDENTITY PASS on clipB (2026-09-23, 54 stitched target/goalkeeper players labeled from scratch, same retuned tracklet_stitch.py defaults): 14 confirmed (26%), 24 mixed (44%), 9 not_on_roster (17%, higher than clipA's 7%), 7 skipped, 0 bench. Net 16 of 74 target/goalkeeper tracklets (22%) identified, spanning 9 of 21 roster players seen. Close to clipA's 27%/11-players result - the retuned thresholds and the tool generalize across clips, not overfit to clipA. clipB's higher not_on_roster rate is unexplained (not investigated); could be more role-classification leakage in this clip or just noise from small samples.
- Both clips DONE for this identity-assignment pass. The 5 owner-provided identity data points now available across both clips (mixed rate, confirmed rate, not_on_roster rate, roster coverage, jersey_conflicts) are real signal if step 7 work continues - e.g. deciding whether to invest in the substitution-transition fix, or in a better re-identification signal than position+role.

## Phase 6 findings: tracker revisited for swaps, not fragmentation (2026-09-23)
- Question from the owner: would a better tracker improve accuracy enough to reconsider the choice? Measured against the jersey_label.py verdicts instead of the proxy score.
- The "wide" pick (buffer 30 s, match 0.95) swaps people inside single tracklets. Of the "mixed" stitched groups, 17 of 29 (clipA) and 15 of 24 (clipB) are ONE tracklet, so tracklet_stitch.py merged nothing there. 32 to 37% of labeled single-tracklet groups were mixed, and mixed groups hold about 68% of labeled player-time on both clips. Mixed single tracklets are long (median about 37 s vs 8 to 18 s for confirmed ones).
- Mechanism: re-finding a lost track. 69% of mixed single tracklets have an internal gap of 1 s or more versus 22% of confirmed ones (81% vs 34% at 0.5 s, 53% vs 16% at 2 s; 32 of each). With a 30 s buffer, a lost track waits at its last spot and whoever arrives takes the ID. This is what Phase 1b's warning (the proxy rewards leniency) looked like in practice.
- Fragmentation a better online tracker could fix is small: 50 to 57% of tracklet ends are at the frame edge (player left the shot, needs re-identification, not tracking), and only about 10% are a mid-frame end with a restart within 1.5 body heights in 1 s.
- Option 1, strict grid (replay_trackers.py --grid strict, box-only, 15 fps, high 0.7): buffer is the lever, match threshold should stay at 0.95 (lower adds fragments, and 0.6 to 0.7 even adds label-visible swaps). Buffer 1 s cuts player-time in tracks with a 1 s+ gap from 54 to 60% to 9%, at 402 (A) / 442 (B) IDs instead of 298 / 330, and identified players get split only 0.9 (A) / 0.7 (B) times per minute. Buffer 0.5 s: 444 / 498 IDs. The gap measure is partly circular (a short buffer cannot contain long gaps), so it is not proof that swaps are gone.
- Option 2, appearance (reid_cache.py, Ultralytics yolo26s-reid.onnx, generic pedestrian ReID, CPU; 20 min per clip alone, much longer when sharing the CPU): the embeddings do carry some within-team signal (cosine distance to the same player 1 to 3 s later: median 0.27; to another roster player: median 0.49), but no way of using them in the tracker helped:
  - Stock BoT-SORT ReID only lowers a match cost (min of box and appearance), never blocks one. Equal or slightly worse than box-only.
  - Veto every frame (replay_trackers.py VetoBOTSORT): 0.4 shatters tracks (median 0.5 to 1.4 s) because about 15 associations per second compound even a 16% false-block rate. 0.5 added label-visible swaps on clipA. 0.6 is about the same as box-only.
  - Veto only at re-find (--grid refind): still more label-visible swaps on clipA (2 to 3 impure tracks vs 0 to 1). Seen directly: a true re-find is blocked, then a nearby teammate whose embedding happens to be closer takes the ID, 1.1 s later and 2 to 3 body heights away. Generic ReID separates teams, not teammates.
- Label-visible swaps (identity_metrics in replay_trackers.py) only see swaps between two identified players, and the labels come from the old config's clean tracklets, so they favor it on fragmentation by construction. Small numbers (0 to 3 impure tracks per config).
- HAND-CHECKED, blind (track_purity_label.py, owner, 2026-09-23/24). The owner grouped each sampled tracklet's 32 crops by person (up to 4 groups, A to D, on clipA; up to 6, A to F, on clipB; the tool now allows 8), so each result is a share of time plus the swap positions, not only pure/mixed. Groups are saved per crop with its det_row (purity_crops.csv), so they are config-independent ground truth.
  - clipA, 20 tracklets per config: current (30 s buffer) 35% one-person tracklets, 74% of time on the main person, 28 swaps of which 10 at a 1 s+ detection gap (after the redo below; one of its tracklets held 7 different people); box-only 1 s buffer 50%, 90%, 13 swaps (1 at a gap); appearance veto 0.6 with 5 s buffer 55%, 86%, 15 swaps (0 at a gap).
  - clipB, 15 per config (current config dropped as clearly worst): box-only 1 s / match 0.95: 67% one-person, 92% main-person time, 0.48 swaps per tracked minute; veto 0.6 / 5 s: 67%, 88%, 0.69 (5 of its 8 swaps at gaps: a 5 s buffer still lets re-finds swap); stock ReID proximity 0.2 / 1 s / match 0.8: 53%, 84%, 0.79; veto 0.4 / 5 s / match 0.8: 43%, 84%, 1.03.
  - Pooled over both clips (35 tracklets each): box-only 1 s 8.9% of tracked time on the wrong person, veto 0.6 12.9%. Current config about 27% on clipA.
- A TRAP HIT AND CAUGHT: scoring every sweep config against the clipA crop groups (det_row pairs: different-person pairs sharing a track, consecutive same-person pairs split) ranked stock ReID p0.2 and veto 0.4 with match 0.8 far ahead. Their blind clipB round put both last. Cause: the labeled pairs only come from tracklets of the sampled configs, so sampled configs pay for all their swaps while any other config pays only for swaps inside those same tracklets. Use that cross-scoring to shortlist, never to pick; pick only on a fresh blind sample.
- DECISION: box-only BoT-SORT, 15 fps, high/new 0.7, buffer 1 s, match 0.95 (strict sweep config 8 = reid sweep config 7). Best or tied-best on both clips, no appearance model needed. Appearance (generic ReID) is dropped: it separates teams, not teammates.
- What is left: the remaining swaps are almost all in continuous tracking (1 of 13 and 1 of 5 at gaps), i.e. players in contact or crossing, so no buffer setting fixes them. About 9% of tracked time is still the wrong person. A later fix would split tracklets at contact/crossing moments rather than tune association.
- APPLIED 2026-09-24 on both clips: best_tracklets.csv.gz regenerated with --grid chosen (402 tracklets on clipA, 442 on clipB, identical to the labeled sweep config), then pitch_mask.py, team_classify.py classify, pitch_calibrate.py apply (clipA: data/clipA/pitch_anchors.local.json; clipB: the repo-root default), tracklet_stitch.py and events.py rerun. Everything derived from the old tracks is in data/<clip>/archive_wide_tracker/, including owner labels keyed by old track IDs (role_truth.csv, events_truth.csv, jersey_truth.csv): the Phase 2, 4 and 5b hand-checked numbers describe the OLD tracks and cannot be rescored as-is. identity_rows.csv.gz and the purity groups are keyed by det_row and still valid.
- Jersey labeling redone on both clips (see Phase 6b), with earlier confirmations offered as hints ("earlier: #N", key a) where at least 90% of a player's earlier-labeled detections agreed: 25 players on clipA, 16 on clipB.
- Tracklets 13, 27 and 30 of clipA's purity sample hit the 4-group cap of the first grouping version. REDONE with the 8-group tool (2026-09-24, `--redo 13,27,30`): two current-config tracklets gained swaps (24 to 28 swaps, 9 to 10 at gaps); the box-only and veto configs' numbers did not change, so the pooled comparison stands.

## Phase 6b findings: identity on the new tracks, naming tracklets and parts (2026-09-24)
- Owner relabeled both clips with jersey_label.py on the new tracks. First pass on clipA (whole players only): 28 of 71 confirmed, 28 mixed; identified target/goalkeeper tracked time 27% (old tracks) to 34%.
- Mixed players held 60% of clipA's target/goalkeeper time: 17 of 28 were bad stitcher joins between tracklets that were each one person, 11 were a swap inside one tracklet. Two tool changes, both owner requests: (1) mark tracklet joins (T1, T2, yellow bar) and name each tracklet on its own; (2) shift+click to split a tracklet where another person starts (parts T1a, T1b, magenta bar; the frames between the last crop before the switch and the switch crop get no identity).
- Result, both clips: identified target/goalkeeper tracked time clipA 27% (old) to 80%, clipB 13% (old) to 78%; roster players found clipA 11 to 15 of 21, clipB 9 to 16 of 21. clipB: 37 of 66 players split into named parts, 31 tracklets split at a switch. clipA: all 28 mixed players redone (--redo-mixed): 26 split, 2 were one person after all; 26 tracklets split at a switch.
- Swaps inside tracklets are still common (26 and 31 tracklets split at a switch), consistent with Phase 6's remaining ~9% wrong-person time from contact swaps. Named parts are exactly where they were found by eye, so they are the ground truth to measure any future contact-swap fix against.
- jersey_conflicts rose (clipA 15, clipB 13): the same jersey on several stitched players. That is one real player spread over several stitched groups (tracklet_stitch.py under-merging), not a labeling error; identity is keyed by jersey, so per-player stats still combine them.
- Outputs: player_identity.csv (per tracklet; split_at_switch=True means no single identity) and identity_segments.csv (named parts of split tracklets as ci ranges). Step 8 must read both. UNVERIFIED beyond the owner's own labeling: the 78 to 80% assumes the part boundaries are right.

## Phase 7 findings: step 8, per-player stats (player_stats.py, 2026-09-24)
- player_stats.py builds per-player, per-clip stats from identity (player_identity.csv + identity_segments.csv), pitch positions and events.csv, with confidence and visibility on every stat. Outputs are local: RUN/player_stats.csv, player_events.csv, stats_report.json, and data/stats.sqlite across clips; --share-dir writes team aggregates only.
- Raw frame-to-frame pitch positions are too noisy for speed (raw speed p99 22 to 31 m/s). Positions are smoothed with a 2 s moving average within continuous runs of one identity. Noise floor, measured on sideline_suspect tracklets (people standing still off the play): 34 m/min clipA, 24 m/min clipB at 2 s (44 and 33 at 1 s). Heavier smoothing lowers players and the floor together, and clipA's floor stays high at any setting (calibration drift). Every running stat is reported with the clip's floor, not corrected for it.
- Sanity check that passed: the goalkeeper shows 38 m/min on clipA, about the floor, as a mostly stationary keeper should. Outfield medians: 112 m/min clipA, 88 m/min clipB, plausible for youth play. clipA runs higher than clipB across the board, consistent with its higher floor, so compare players within a clip, not across clips.
- Ball events per player are too sparse to use yet: most identified players have 0 to 2 trusted events in 5 minutes. events.py's detections are thin (sparse ball path at min-conf 0.25, touch recall 21% in Phase 4) and only events on identified players count. Tips based on ball involvement need better event detection first.
- UNVERIFIED: no measured distances or speeds to check against. Speed bands (walk under 2, jog 2 to 4, run 4 to 5.5, fast 5.5+ m/s) are guesses for youth players.

## Phase 7b: a third window end to end, and what a window costs the owner (2026-09-25)
- clipD (43:00 to 48:00, existing cache) was dropped after anchoring: the goal end only shows in the second half, and the camera zooms hard exactly around the anchors (cumulative scale 0.73 to 0.96 to 0.71 within 46 s), so anchors 22 s apart disagreed by 9 to 16 m although each anchor was precise alone (leave-one-out under 2 m). First-half positions would have been extrapolated about 150 s. Lesson: pick windows where a goal end shows near the start and the end, and check zoom volatility (median 2 s scale change from the cache) before anchoring.
- clipE (20:00 to 25:00, owner's pick) run end to end: 314 tracklets, 83 stitched target/goalkeeper players. The camera zooms a lot in the first 3 minutes (median 2 s scale change 0.030 to 0.034, clipA-like) and little after. 6 anchors (7, 78, 132, 159, 222, 291 s), each precise alone (leave-one-out 1.5 to 2.1 m); calm-section pairs agree within 1 to 2 m, zoomy-section pairs 6 to 14 m. Every frame is within 36 s of an anchor (median 14 s).
- Identity on clipE: 82% of target/goalkeeper tracked time identified, 11 roster players (the whole on-field side). 35 players confirmed whole, 39 split into named parts, 40 tracklets split at a switch.
- OWNER TIME PER WINDOW (clock time, breaks included): anchors about 65 min for 6, jersey labeling about 85 min for 83 players, so about 2.5 h per 5-minute window. Live play is about 80 minutes, so a full game at this rate is about 16 windows and 40 hours. The manual steps, not compute, limit scaling; jersey labeling is the larger one.

## Phase 7c: jersey suggestions from the owner's labels (jersey_suggest.py, 2026-09-25)
- Goal: cut jersey labeling time (about 85 min per window). Frozen image features plus logistic regression trained on the owner's labeled crops, predicting the jersey of each tracklet part in another window. Settings chosen on clipA<->clipB only, then clipE scored once.
- Offline test on clipE (tracklet parts; chance about 5% with 19 players): generic person-ReID features 45% top guess, ConvNeXt 55%, DINOv2 ViT-S/14 56%. Assigning parts that overlap in time to different players (same-frame rule) adds about 10 points: DINOv2 + rule 70% top guess, 79% top 3, with a third of parts confident (>= 0.5) at 95% right.
- It does NOT transfer across the game: trained on clipA (6 min before clipE, same half) 72% on clipE; trained on clipB (58 min away, other half) 18%; clipA->clipB and clipB->clipA only 20 to 30%. Probably lighting, ends and framing. So labeled windows are weighted by closeness in time (exp(-minutes apart / 10)), and windows should be labeled in order through the game, each helped by its labeled neighbors.
- In the tool (jersey_suggest.py suggest, then jersey_label.py): on clipE's whole tracklets from clipA+clipB, 63% top guess, 75% top 3; half the tracklets have confidence >= 0.3 at 94% right, a quarter >= 0.5 at 100%. Each part (including parts split during labeling) shows "~N"; `a` accepts. An exact earlier label of the same detections still wins. Not applied without the owner.
- UNVERIFIED: whether it saves time. Measure on the next window labeled in order (25:00 to 30:00, next to clipE) against clipE's 85 min.

## Phase 7d: automatic pitch anchors by snapping to painted lines (pitch_autoanchor.py, 2026-09-26)
- Goal: cut anchor placing (about 65 min per window) and fix drift while the camera zooms. The penalty-area geometry (goal line, six-yard and 18-yard boxes, penalty arc; law sizes, no pitch size assumed) is projected with the current estimate and a small correction is fitted so the projected lines sit on painted lines found by a white top-hat. Mask settings came from a sweep scored against owner anchors: the saturation limit mattered most (50% of true lines found at saturation < 90, 85% at < 130).
- Snapping from a distant anchor fails often (wrong-line locks; one case aligned 94% of its lines while 83 m off). Chaining works: from each owner anchor, snap every 2 s, each frame starting from the previous accepted fit carried through 2 s of camera motion. Accept only if 60% or more of the projected lines land on paint AND the fit moved the lines at most 40 px from the prediction. It does not bridge long stretches with the penalty area out of view (8 to 27 m there in clipD and clipE), so the owner still needs about one anchor per stretch where the box is visible. The run report lists uncovered stretches.
- Leave-one-out at the owner's anchors, with the pipeline's blended conversion and the final settings (owner anchors alone -> with automatic ones, median (worst)): clipA 1.3 (16.3) -> 0.9 (5.0) m, clipB 1.3 (2.6) -> 0.9 (2.4), clipD 1.9 (12.4) -> 1.3 (12.4; its 204 s anchor is past a midfield stretch), clipE 2.7 (9.4) -> 1.1 (1.5). The main gain is removing the large errors during zooms. Earlier figures (4.9 -> 0.8 m on clipA etc.) used the nearest anchor instead of blending on both sides and overstated the gain; blending alone accounts for much of the median improvement.
- A trade-off only a second measure showed: with perspective terms and an anchor every snap, the noise floor (still people near the touchlines, far from the box) rose from 24 to 32 m/min on clipE, because penalty-area lines alone pin perspective poorly. Chosen: affine correction only, automatic anchors at least 8 s apart (clipE noise floor 23.9 m/min, same as owner anchors alone). Also changed pitch_calibrate.py to blend linearly between the two surrounding anchors instead of switching to the nearest (switching made positions jump), and to cross-check only owner anchors.
- Applied to clipA, clipB, clipE: noise floor 33.3 -> 30.5, 23.8 -> 21.8, 24.2 -> 23.9 m/min (owner anchors blended -> with automatic ones). 19, 15 and 28 automatic anchors; 43%, 43% and 71% of 2 s frames snapped. tracklet_pitch_xy.csv.gz and player stats now use them.
- UNVERIFIED: how few owner anchors a new window needs in practice. Next window: place about one per box-visible stretch, run pitch_autoanchor.py, add anchors only where the report shows uncovered stretches.

## Phase 8 (in progress, 2026-09-26): no owner anchors, no jersey labeling
- Owner decision: pitch anchoring and jersey identity must be automated (whole team). clipF (25:00 to 30:00) is
  processed through events; the owner stopped jersey labeling it after 2 players (about 13 min).
- AUTOMATIC PITCH (pitch_ptz.py), DONE, VERIFIED AND APPLIED. The camera stands still all game: every owner
  anchor (clipA, B, D, E, both halves) implies the same camera centre within a few metres, and one camera fitted to
  all 20 anchors reprojects them about as well as each anchor's free homography. So each frame has only pan, tilt
  and zoom (roll ~0), searched exhaustively against the painted lines every 2 s (tracked from the previous fix via
  camera motion, global search when that fails). Needed to work: only paint on the turf counts (largest grass
  area; foliage passes a grass test) and specks under 40 px are dropped (turf texture); the painted centre ring is
  measured (7.5 m at this venue, a logo edge, not the 9.15 m law circle) and pins zoom in midfield.
- Held out (camera fitted without that clip's anchors, no anchors in the run), median (worst) error at the owner's
  anchors vs the owner-anchor + auto-snap calibration: clipA 0.36 (2.4) vs 0.9 (5.0) m, clipD 0.21 (0.37) vs 1.3
  (12.4), clipE 0.31 (0.44) vs 1.1 (1.5). Noise floor 29.1 vs 30.5, 26.0 vs 40.2, 21.4 vs 23.9 m/min. clipB cannot
  be held out (the only window that saw the other goal, which pins the pitch length).
- FOUND: clipB's owner anchors measure X from the other goal than clipA/D/E. Automatic fixes use one reference goal
  for all windows, so clipB positions flip when applied (within-clip speeds unaffected).
- Final camera data/pitch_camera.local.json; `pitch_ptz.py run` done on clipA, B, D, E, F (fixed 89 to 98% of 2 s
  frames, longest gap 4 to 12 s).
- APPLIED 2026-09-26 on clipA, B, D, E, F (pitch_calibrate.py apply --anchors RUN/pitch_anchors_ptz.local.json;
  previous outputs in data/<clip>/archive_owner_anchor_pitch/). Player positions on the pitch: 85 to 93%. Against
  the previous calibration, median (p90) difference per detection: clipE 1.5 (10.5) m, clipA 5.2 (14.6), clipD
  10.9 (30.1; its owner calibration extrapolated ~150 s), clipB 5.3 (12.6) after mirroring X only (slope -0.86):
  the reference-goal flip, confirmed. Y is defined by the camera side, so it does not flip.
- player_stats rerun (A, B, E): outfield median m/min 99.7 -> 84.3 (A), 77.1 -> 70.5 (B), 84.9 -> 76.9 (E); noise
  floor 30.5 -> 28.7, 21.8 -> 24.0, 23.9 -> 17.8. Goalkeeper (median X 4 to 5 m) 34 to 35 m/min, at the floor.
  The anchor-less error model is flat (pos_err_m 0.5 for everyone); the held-out tests above say typical error is
  0.2 to 0.4 m with worst cases 2 to 2.4 m, so it understates rare bad frames. UNVERIFIED: no measured distances.
- AUTOMATIC IDENTITY (jersey_auto.py, experiments in data/_jersey_exp/auto/), IN PROGRESS. Per-sample (0.5 s) evidence
  along each tracklet, smoothed with Viterbi. Leave-one-window-out on clipA, B, E:
  - Appearance from the owner's other windows: 67 to 82% (first half), 15% on clipB (other half: substitutes,
    lighting). Relative position (to the team's visible centre) adds coverage; absolute position hurts.
    Same-moment exclusivity and in-window self-training did not help (self-training collapses).
  - Jersey numbers: SoccerNet-fine-tuned PARSeq reader + legibility classifier (jersey-number-pipeline, Koshkina &
    Elder 2024, non-commercial licence; weights in models/, git-ignored). Crop choice matters: the back patch (12 to
    42% of box height, middle 60% of width) read 89% right where the whole torso read 62%. Fine-tuned on the
    owner's labeled crops of two windows, tested on the third: reads right 77 -> 98% (clipA), 87 -> 97% (clipE),
    74 -> 90 to 94% (clipB), with about 60% more reads.
  - Main remaining error: a read spread along a tracklet past a person swap (right 94 to 96% within 1 s of a read,
    55% beyond 30 s; 39 to 63% on tracklets the owner split). 47 of 48 owner switch points have a box overlap
    (>= 0.1 of the smaller box) within 1.5 s, but overlaps happen about 10 times per tracked minute.
  - TRIED, NO GAIN (exp8.py, 2026-09-26): allow identity switches only at box-overlap episodes (Viterbi switch cost
    low there, high or forbidden elsewhere). Cut pieces are pure (98.6 to 100% of labeled samples in one-person
    pieces vs 82 to 90% uncut tracklets) but only because they are tiny: 28 to 82 pieces per tracked minute, a cut
    every 1 to 2 s. Top-1 accuracy leave-one-window-out was slightly WORSE than switching anywhere at cost 20
    (mean 63.9 vs 65.8%; clipE 82.9 vs 84.4, clipA 74.6 vs 75.2, clipB 34.1 vs 37.9). Where a switch may happen was
    never the limit; deciding whether one happened is, and that needs evidence on both sides.
  - The real limits: (1) evidence. Most samples have no read nearby, so appearance decides them, and appearance
    fails across halves (clipB 34 to 38%). (2) Confidence is unusable: segment posteriors >= 0.99 cover 80 to 95% of
    samples but are right only 41 to 90%, because summing correlated per-sample appearance log-probs is
    overconfident. A production output must be trustable, so confidence has to come from reads (count of agreeing
    reads per segment), not appearance sums.
  - DONE (exp9.py, jersey_auto.py identify, 2026-09-26): trust from reads. A sample is identified only if its
    decoded stretch has >= 3 reads of that number, >= 80% of the stretch's reads, and one within 10 s. Leave-one-
    window-out, held-out readers: identified 39.4 / 40.6 / 43.4% of labeled samples (E / A / B), right 98.4 / 99.8 /
    91.0%. Reads on every tracklet row (instead of every 4th cached frame) doubled usable reads and added 2 to 12
    points of coverage. Distance limit: 30 s instead of 10 adds about 5 points but clipE drops to 97.0%.
  - Appearance retrained on the window's own read-confirmed samples (pseudo-labels 90 to 100% right) helps clipB,
    the other half (top-1 37.9 -> 54.1%), not A or E. Kept: it only moves stretch boundaries, trust is still reads.
    Trusting read-less stretches by mean appearance probability: no usable coverage at any threshold. Dropped.
  - clipB's 91%: nearly all errors are one player who appears only in clipB, so the held-out reader never
    saw that number and read it consistently as one of three similar trained numbers (19 to 29 agreeing reads). Requiring the original,
    not fine-tuned, reader to agree: 91 -> 94.7% but -8 points coverage on every window; a check keyed on the
    predicted number cannot catch it (the wrong number was a trained one). PRODUCTION RISK: players with few or no
    labeled crops (3 of 21 roster players have none, 8 under 30) can be misread as a similar trained number.
  - Coverage ceiling: 61 to 72% of labeled time is on tracklets with any read; the rest (shorter tracklets, median
    18 to 28 s) has none. Linking identity across tracklets would be the next lever (stitching precision was ~48%).
  - Production reader: jersey_auto.py finetune on clipA+B+E (3974 crops) -> models/jersey/parseq_ft_game.pt.
  - clipF WRITTEN (jersey_auto.py identify --from A,B,E): 50.1% of target/goalkeeper samples identified, 10 players,
    107 stretches, 20.5 identified player-minutes (clipE with owner labels: 34.0). Against the owner's 2 labeled
    clipF players (69 s overlap): 91% agree, disagreements at switch boundaries. Two clipE players are absent and
    two new numbers appear, each with 49 to 60 confident reads on 5 to 6 tracklets: probably substitutions,
    UNVERIFIED. The goalkeeper got no reads at all (26 training crops; back rarely toward the camera) so has no clipF stats.
- FIRST WINDOW WITH NO OWNER STEPS (clipG, 30:00 to 35:00, 2026-09-26). Checked first: density scan steady (about 20
  people, players 50 to 115 px) and two downscaled stills showed live 11-a-side play. Then run_all.py, pitch_mask,
  team_classify classify, tracklet_stitch, events, pitch_ptz run + pitch_calibrate apply, jersey_auto identify
  --from A,B,E --write, player_stats (script: data/clipG_stages.sh, git-ignored). Owner time: none.
  - Compute: detection 13 to 14 frames/s (about 11 min; the run's wall clock was 72 min, time not spent detecting,
    probably the laptop sleeping), pitch_ptz 4.7 min, identify 9.4 min (reads and features), the rest under 1 min.
  - Pitch: all 150 two-second frames fixed. Noise floor 27.3 m/min (clipF 21.5, clipE 17.8).
  - Identity: 30.3% of target/goalkeeper samples identified (clipF 50.1%), 11 players, 54 stretches, 12.4 identified
    player-minutes (clipF 20.5, clipE owner-labeled 34.0). Same tracklet count and box size as clipF; the drop is
    legibility: 5.6% of crops pass the legibility check vs 8.6% on clipF, so 1594 usable reads vs 2314 on the same
    number of crops. Why fewer crops are legible here is not established. Every identified number has reader
    training crops (the fewest: 10). UNVERIFIED: no owner labels on clipG, so accuracy is only the held-out estimate.
- LINKING IDENTITY ACROSS TRACKLETS (exp10.py, 2026-09-26), TRIED, NOT ADOPTED. Unidentified stretches get the
  identity of a player whose identified stretches before/after fit by motion on the pitch (2D Gaussian, sigma 1.5 m +
  2 m/s x gap, up to 15 s), who is not identified elsewhere at the same time, plus appearance trained on the
  window's read-confirmed samples; accepted only if it beats the runner-up and "someone else" (anywhere on the
  pitch, flat appearance) by a margin. Two scoring bugs found on the way (candidates with evidence on one side
  scored higher than with both; a lone candidate always accepted) made the first run look like 30 to 45%.
  - Leave-one-window-out, held-out readers: margin 3, one round: +9.2 / +10.7 / +7.1 points of labeled time (E / A
    / B) at 100 / 87.7 / 74.1% right (mean 87%). More rounds or lower margins add coverage at 70 to 84%; margin 5
    or more leaves under 5 points. Read-based identity is 91 to 99.8%, so linking would dilute it.
  - Why the ceiling is low: of labeled time left unidentified, 13 to 44% belongs to players never identified
    anywhere in the window (no reads at all), 24 to 34% is over 15 s from that player's nearest identified stretch
    (motion says nothing there), and only 25 to 45% has a usable candidate. Duplicate tracks and overlap conflicts
    are rare (under 7%).
  - What would move coverage: more players identified at all (the goalkeeper and the players with no reads), not
    better linking. Appearance within a team stays weak (Phase 6: generic ReID separates teams, not teammates).
- GOALKEEPER BY ROLE, PLACE AND LOOK (exp11.py -> jersey_auto.py goalkeeper_samples, 2026-09-27). The goalkeeper's
  back rarely faces the camera (no reads on clipF). The goalkeeper's labeled tracklets get team_classify's
  goalkeeper role (all on A and E, 78% on B) and stay within about 7 m of the goal line. Rule: goalkeeper-role
  samples within 18 m of our goal line and 25 m of the centre line across, goalkeeper appearance >= 0.3 (DINOv2 +
  logistic regression, our goalkeeper vs everyone else labeled or goalkeeper-role in the labeled windows), one per
  moment, tracklets mostly kept. "Our end" = the end whose goalkeeper-role people look more like our goalkeeper:
  right on all three windows, both halves. A trusted read always wins.
  - Held out (goalkeeper's own labeled time): found 100 / 64 / 99% (A / B / E), right 100 / 100 / 93%. Whole
    identify, held out: identified 40.6 -> 47.2% (A), 43.4 -> 44.7% (B), 39.4 -> 48.4% (E); right 99.8 -> 99.9,
    91.0 -> 91.3, 98.4 -> 97.5%.
  - Applied: clipF 50.1 -> 54.8%, clipG 30.3 -> 33.5% of target/goalkeeper samples; the goalkeeper now has stats
    there (113 s and 80 s visible, 47 and 39 m/min, near the noise floor as for a keeper).
- RARE NUMBERS: OWNER CONFIRMS A FEW CROPS (jersey_rare_label.py, 2026-09-27). The original (not fine-tuned) reader
  cannot label rare numbers by itself: its confident reads of numbers rare in the other windows were right 0 to
  75% (3 to 110 reads). So the owner confirms. `candidates` found 145 crops for 5 of the 7 rare numbers (roster,
  not goalkeeper, under 30 labeled crops) across clipA, B, E, F, G, skipping crops the owner already named; the
  other 2 numbers were never read (probably did not play). `label`: one screen per number, click the crops that
  show it. `jersey_auto.py finetune` adds the confirmed crops (each repeated 4 times, jittered).
  - DONE 2026-09-27: owner labeled all 5 numbers in about 10 min (window load included). 31 of 145 candidates
    confirmed (17, 8, 3, 3; one number 0 of 43: its candidates were all misreads). Reader retrained (4098 crops),
    clipF and clipG reread and identified again, stats rerun. Small gain: clipF 54.8 -> 55.4%, clipG 33.5 -> 33.9%
    of target/goalkeeper samples (+14 s and +8 s of identified time); the rare players simply play little in these
    two windows. Not measurable held out (no labels on clipF/G). Worth repeating only when a new window shows
    numbers the reader has not seen (candidates finds them).

## Phase 9: the whole game, no owner steps (2026-09-27)
- Live play from stills at boundaries (5 downscaled, deleted): first half 0:00 to about 36:00, halftime to about
  43:30, second half to about 81:00; 82:30 is players leaving, 86:00 another game on the pitch. clipB (78:00 to
  83:00) therefore holds about 2 min after the final whistle.
- run_windows.py processed clipD's identity plus 9 new windows (clipH 00:00, clipI 05:00, clipJ 09:00, clipK 48:00,
  clipL 53:00, clipM 58:00, clipN 63:00, clipO 68:00, clipP 73:00) with no errors, about 22 to 25 min each
  (detection about 12, pitch_ptz about 6, identify about 3). With A, B, E, F, G: 15 windows covering 0 to 35 and 43
  to 83 min. Gaps 19-20 and 35-36 min; clipJ overlaps clipI by 1 min (started at 09:00 to end where clipA begins).
- Automatic identity per window: 34 to 55% of target/goalkeeper samples (first half 34 to 55, second half 34 to
  55; no trend by half). 289 identified player-minutes, 19 roster numbers; one number with 6 s in one window is
  probably a misread (the owner rejected all its rare-number candidates).
- Goalkeeper in the second half: the appearance cutoff (0.3) kept nobody in 4 of 7 second-half windows (our
  goalkeeper scores low in the other light). Now data/game.local.json (git-ignored) gives halftime and our first-half
  goal end; with the end known the cutoff is 0.1 (held out unchanged: A 47.2%, B 44.9%, E 48.7% identified at 99.9,
  91.3, 97.2% right). Second-half goalkeeper time 1.9 -> 3.2 min, first half 10.8. The rest looks genuine: most
  second-half goalkeeper-role samples at our end are touchline people (|Y| about 33 m), and the owner's own clipB
  labels show 14 s of goalkeeper in 5 min vs 56 to 87 s on clipA and clipE.
- Whole-game per-player summary (local only): data/game_player_summary.csv, rates weighted by visible time,
  position as distance from our own goal (flips at halftime). UNVERIFIED like all running stats; noise floors
  differ by window (16.8 to 28.7 m/min), so compare players, not absolute values.

## Phase 10: coaching tips (coaching_tips.py, 2026-09-27)
- Owner's choice: tips for every player, on work rate, positioning, fatigue and involvement. Movement only: ball
  events are too sparse (Phase 7). Output local (names of minors): data/coaching/ (a report per player,
  team_overview.md, player_metrics.csv).
- Measures, from every identified sample of 15 windows: work rate relative to identified teammates in the same
  windows (cancels each window's noise floor); time at 4 m/s or faster; distance from our own goal, depth vs the
  team line (median of visible target-role players, 4 or more), width, roam (10th to 90th percentile of depth on
  the pitch); fatigue as early vs late in each half and first vs second half; involvement as time within 10 m of
  the ball on detected ball positions (carried to the pitch with the window's automatic calibration).
- Roles by thirds of depth vs the team line among well-seen outfield players (a fixed +-7 m made 12 of 19
  midfielders: the camera shows part of the team, which compresses depth). Tips compare a player with the others
  in the same role.
- NOISE: a player's relative work rate varies window to window with sd 0.145 (windows with at least 1 min of the
  player; including a few-second windows gave 0.30). Fatigue and work-rate tips must exceed 2 standard errors
  given the windows behind each side, 19 to 27% for 2 to 4 windows per phase. With that, NO fatigue tip passes for
  any player: the 10 to 13% late-half drops seen first were inside the noise. Intensity also needs 1.5 points,
  involvement 5 points.
- Result: 14 tips for 7 of the 15 players seen 5 min or more (3 work rate strengths, 2 + 2 intensity, 2
  positioning, 1 + 2 involvement); 8 get "nothing stands out"; 4 players under 5 min get none. Thresholds are
  hand-set; UNVERIFIED like the running numbers, and phrased as observations to check on video.
- What would make tips richer: more identified time per player (coverage), ball events (Phase 4), and more games
  (a player's pattern across games is far more trustworthy than one game's 5 to 40 minutes).
- HTML pages (coaching_html.py, 2026-09-27): coaching_tips.py also writes data/coaching/index.html (team table,
  roles ordered goalkeeper to forwards, players under 5 min muted) and player_NN.html per player: stat tiles vs the
  role median, observations with evidence, a pitch heatmap of visible time (both halves, attacking to the right,
  one blue ramp in 6 steps, hover values) and relative work rate per quarter of the game with +-2 SE whiskers and
  the team line. Self-contained (inline SVG, a few lines of inline script for tooltips, no network requests), light
  and dark themes. Local only: they carry names of minors. Checked by rendering in headless Edge, both themes.

## Phase 11: a second game, and one command for any new game (new_game.py, 2026-09-28)
- Games now have folders: data/<game>/<window> with the game's own game.local.json (halves, halftime, which goal we
  defend), pitch_camera.local.json and kit_prototypes.local.json beside them (sv_common.game_file). The first
  game's windows stay in data/ and its files where they were, so nothing about it changed.
- new_game.py: `plan` (density scan, proposed halves, contact sheet data/<game>/live_play_check.jpg), `confirm`
  (with corrections), `setup` (pilot window: pitch check, refit if needed; kit clusters and a stop for the mapping),
  `run` (all windows, goalkeeper vote for which goal we defend, identity again, stats, coaching pages).
- Live play cannot be found from people counts alone: halftime warm-ups and other teams after the game look like
  play. On game 1 the proposal put the restart 3 min before kick-off and the end 3 min early. So the contact sheet
  steps through both boundaries a minute at a time and the owner confirms. Water breaks (hot games, midway through
  each half) are stoppages, not a change of ends: halftime is the longest break near the middle.
- Game 2 (data/g0922, 90 min video): first half 0:00 to 34:00, halftime 34:00 to 43:40, second half 43:40 to
  1:20:00; 14 windows. Same venue as game 1. FIRST CONFIRMED WRONG (40:40 and 86:00) from the contact sheet:
  halftime warm-ups and the next game's players look like play in small thumbnails. Counting OUR players per frame,
  minute by minute (target-role candidates), showed the ends at once: 6 to 9 in play, falling to 1 in minute 34
  and minute 80. The 4 windows outside play are kept aside in data/g0922/_outside_play/. new_game.py run now prints
  this check (a warning for any minute in the halves with fewer than 3 of ours per frame).
- Pitch camera: the pilot fixed 77% of frames (game 1: 89 to 100%). Not the tripod: an anchor-free refit
  (pitch_ptz.py refit, the line stage of `fit` on the new game's own fixed frames) moved the centre 5 cm. The
  painted lines are fainter in this game's light (accepted frames score a median 3198 vs 4270; rejected ones 1788,
  just under the 2000 cut). Gaps are bridged by camera motion: longest 14 s, and the noise floor (25.5 m/min) sits
  inside game 1's range (16.8 to 28.7). So `setup` gates on at least 70% fixed and no gap over 20 s, not 90%.
- Kits: the first game's prototypes misread the new opponent (3 opponents per frame, 69% low-confidence roles).
  Calibrated on the pilot (6 clusters, mapped from the montage; two mixed: officials with some of our bench, and
  the opponent with some bib-wearers): 8 target and 8 opponent per frame, confidence 0.71, 14% low-confidence.
- Identity on the pilot: 27% of target samples (game 1 windows 34 to 55%), learned from game 1's labeled windows.
  Not yet explained.
- GAME 2 DONE (2026-09-28): 14 windows, 18 players, 206 identified player-minutes (game 1: 289). Identity 35 to 55% of
  target samples per window, about 20% in the last two windows before the end (fading light or late substitutions;
  not investigated). Goalkeeper vote: we defend X = 0 in the first half. Events on identified players: 114 touches
  (+39 unconfirmed), 118 possessions. Coaching pages data/g0922/coaching/ (local): 16 observations, including the
  first fatigue observation to pass the noise bar. Wall time with 3 workers and GPU slots: about 3 h for 9 windows.

## Phase 12: ball events retuned on owner labels (events.py, 2026-09-28)
- Ceiling test first (clipA's old window with the owner's ball positions fed in as the path): possession recall
  56 -> 75% but 10 predicted turnovers for 1 real one, touch unchanged. So the losses are in the event logic, not
  only the ball detector. Touches cannot be tested that way (0.5 s ball labels are too coarse for a touch).
- Owner labeled 4 more 60 s possession windows with event_label.py, about 6 to 9 min each: game 1 first half
  (clipH 40-100 s), game 1 second half (clipP 100-160 s), game 2 first half (g0922/w0500 20-80 s), game 2 second
  half (g0922/w5340 160-220 s). With the old window: 111 possessions, 107 touches, 11 passes, 11 turnovers.
- Labeler fixes found while labeling (owner): the window grows with zoom and keeps the zoom between frames (now a
  rule for every labeling window, ZoomView gained a base scale); every tracked person is boxed (a goalkeeper
  standing in goal failed the player-candidate test and could not be clicked); `u` = has the ball, no box.
- Where possessions were lost (labeled possessed frames): no ball detection within 0.1 s in 22% (game 1) and 34%
  (game 2); when detected, the ball was within the 0.6 body-height contact distance 85% (game 1) but 55% (game 2:
  a quarter of detections 3 to 5 body heights away, a wrong object).
- Grid over contact distance, gap bridging, minimum possession, who can hold the ball, touch definition and
  confidence cut, chosen leave-one-window-out (5 folds; the same setting won 4 of 5). Held-out, pooled:
  possession recall/precision 42/76% -> 62/63%, touch 11/35% -> 62/60%, turnover 36/25% -> 64/24% (after the
  chain rule below), pass 1 of 11 found either way. New defaults: MIN_POSSESSION_S 0.2 (was 0.4), TOUCH_MODE
  "gain" (a touch is a player gaining the ball, as the owner labels it; the velocity rule found 11%), INCLUDE
  "keepers", CHAIN_MIN_S 0.4 (passes/turnovers chain only possessions of 0.4 s or more: turnover F1 0.28 -> 0.35,
  chosen in every fold). Contact 0.6 and bridging 0.4 s unchanged. No confidence cut helped.
- Keepers (owner's suggestion): a track the colours called goalkeeper/other/unknown whose median pitch position is
  inside a penalty area (not behind the goal) becomes a player of the team defending that goal (game file, or where
  our goalkeeper was identified). Goalkeepers held the ball in about 4% of labeled possession frames, so the gain
  is small (0.623 vs 0.616 mean F1) but their possessions and distribution now count for the right team.
- Per window, new defaults: game 1 73 to 76% possession recall at 57 to 73% precision; game 2 38 to 47% at 53 to
  75%. Game 2 is limited by the ball detector (misses and wrong objects), the next lever there. Passes stay
  UNRELIABLE (the ball in flight between teammates is lost).
- Kits drift between halves (game 2 second half: 83 player tracks read as goalkeeper, 65 unknown with the
  first-half prototypes). Mapping a second-half pilot too (assign adds prototypes): 8 target, 10 opponent, 1 keeper,
  1 official per frame. new_game.py setup now maps a pilot in each half.
- Speed: new_game.py run processes windows in parallel (--workers, default 3). Two detections at once barely slow
  each other on the laptop GPU (11.6 vs 11.3 min per window), the pitch fit is CPU work (16 threads), and GPU
  memory was 1.7 of 6 GB. But three identity runs at once filled GPU memory (5.7 of 6 GB) and crawled (40+ min vs
  about 3): the stages share GPU slots, at most 2 detections and 1 identity run at a time. Windows whose roles
  predate the kit file are redone automatically.

## Phase 13: coaching across games, touches in tips, game 2's ends fixed (2026-09-28)
- coaching_tips.py takes windows from several games: each game is measured alone and pooled; a player is compared
  with their role in each game (roles can change) and pooled. Each observation is tagged: in every game the player
  was seen 5+ min in, only with the games pooled, in one game only, or differs (opposite directions in two games).
  Output data/coaching_games/ (local): index.html, player_NN.html (per-game comparison table, a heatmap and
  work-rate chart per game), team_overview.md, player_metrics.csv (scope column). One game works as before.
  Games are labeled by the date in game.local.json "video" (added to data/game.local.json).
- Touches in tips and pages (both modes): touches per visible minute, against the touches identified outfield
  teammates made per minute in the same windows (leave-self-out expected count, so each window's ball detection
  cancels), then against the role's median ratio with a one-sided Poisson test (p < 0.025, 35% gap, 5+ expected).
  All confidences count: the Phase 12 grid chose no confidence cut (min_conf 0 same precision, more recall).
  Passes and turnovers are not used (unreliable). Share of visible time on the ball is shown, not tipped.
- Stale events: game 2's w0500 and w2500 events.csv predated the Phase 12 defaults (events doubled on rerun).
  Every window's events and player_stats were rerun.
- FOUND AND FIXED: game 2's first_half_our_goal_x was wrong (0; truly the far end). The goalkeeper vote had
  followed the OPPONENT's goalkeeper (identified with the appearance cutoff 0.1). Seen as player depths mirrored
  between the games (forwards of one game were defenders of the other, midfielders at 0). Confirmed by the second-
  half kickoff (w4340, first 20 s: our players median X 33, opponents 58; one downscaled still, deleted) and by the
  identified "goalkeeper" standing at the opponents' end then. Fixed in data/g0922/game.local.json; identity,
  events, stats and pages redone for game 2. Players seen 5+ min in both games with the same role: 2 of 13 before,
  11 of 13 after. Game 2 goalkeeper now 7 min, 16 m behind the team line. Identity coverage per window unchanged
  (19 to 55%), so item (3) below still stands.
- new_game.py run now cross-checks the goalkeeper vote with kickoff_end (which half our players stand in during the
  first 30 s of the second half: median X gap 10 m+, 70% of frames agreeing) and stops for the owner if they
  disagree. Tried and dropped as end signals: our median X minus the opponents' over whole windows (near 0, sign
  flips within a half on game 1), and scanning every frame for a clean halfway split (false hit in game 1).
- Result: 20 players, 495 identified player-minutes; 34 observations for well-seen players: 3 in every game,
  9 pooled only, 15 in one game only, 2 differ, 5 for players seen enough in one game. UNVERIFIED like all tips.

## Phase 14: a ball detector for this venue (ball_finetune.py, 2026-09-29)
- Owner labeled 400 game-2 frames with ball_label.py (4 windows x 150 s, 1.5 s apart: w0500, w2000, w4840, w6840;
  about 30 min; 46% clicked, 42% accepted, 8% not visible). COCO path on them, correct when the ball is visible:
  40 / 46 / 39 / 65%; w4840 followed a wrong object 48% of the time, w0500 had no ball on 51%.
- ball_finetune.py: a separate one-class model (yolo11m from COCO), trained on 640 px tiles cut at native scale
  (2 around each labeled ball, 1 random, up to 3 on confident COCO candidates far from the ball = wrong objects),
  run on full 1920 px frames at 10 fps. Person detections untouched; ball_link.py --ball-cache uses the new
  candidates. Training data: game 1's ball labels (clipA, B, D) + game 2's.
- HELD OUT, two folds (model never saw the test window), linked path at min-conf 0.25:
  fold A on w4840: correct 39.4 -> 68.1%, wrong object 47.9 -> 23.4%, missing 12.8 -> 8.5%.
  fold B on w0500: correct 39.5 -> 69.8%, missing 51.2 -> 22.1%, precision 75.6 -> 88.2%.
  min-conf 0.15 to 0.35 is flat in both folds; 0.25 (the existing default) kept. Tile validation mAP50 about 0.6.
- Production models/ball/venue.pt (all labels, 40 epochs; both folds peaked near epoch 38). Applied to game 2's
  14 windows: frames with a detected ball 27-60% -> 48-69% (w4840 unchanged). run_windows.py/new_game.py run it
  after run_all.py when the model exists (`ball_finetune.py apply`: detect, relink; marker cache/ball_ft_meta.json).
- EVENTS DID NOT IMPROVE. Held out on 4 event-labeled windows (clipH, clipP with the venue model, w0500 with fold B,
  w5340): possession F1 0.63 -> 0.61. Game 1 flat (its COCO ball was already adequate for events); game 2 recall up,
  precision down (w5340 possession 38/53% -> 43/39%). A leave-one-window-out retune on the new paths (contact
  0.4-0.8, bridge 0.2-1.0, min possession 0.2-0.6) gained 2 of 95 possessions and doubled false turnovers: not
  adopted. Events are now limited by the event rules, not the ball.
- Kept: venue model for game 2 and new games (better ball positions feed involvement near the ball). Game 1 stays
  on the COCO path (clipH/clipP reverted; no game-1 ball labels outside training to show a gain). Note: rerunning
  run_windows.py on a game-1 window would now apply the venue model.
- Traps: Ultralytics puts a relative `project` under runs/detect/ (fixed: absolute path, trainer.save_dir). The
  fold A run logged 9.2 h for 53 epochs (about 1 h of work: the laptop slept).

## Phase 15: game 2's late identity drop, and height as a cue (2026-09-29)
- Game 2's last two windows (w6840, w7340) identified 19 to 20% of target samples vs 28 to 55% elsewhere. Ruled
  out: fading light (they are the brightest windows), off-pitch "ours" in the denominator (about 1 point), more
  swaps inside tracklets (17 vs 18 conflicting tracklets per window). Cause: a late substitute whose number the
  reader confused with a similar teammate's number (never labeled in game 2's light and kit), so their tracklets'
  reads split and failed the 80% agreement rule; fewer legible crops (6% in w6840 vs 8 to 16%) added to it.
- Fix: owner confirmed crops of the two numbers across game 2 (jersey_rare_label.py candidates --numbers, 72 per
  number spread round-robin over windows; about 5 min) and a second screen on the two late windows, including crops
  the reader had read as the other number (--also N:T; about 2 min). Reader retrained, both games re-identified:
  w6840 18.8 -> 24.4%, w7340 20.2 -> 22.4%; other windows within about +-2 points (several up); identified
  player-minutes over both games 495 -> 511. The late windows stay low: accepted as a legibility limit there.
  The owner rejected every candidate of the other number in the late windows (it was off the pitch).
- jersey_rare_label.py: crops of windows in game folders are stored as g0922/wNNNN (the path under data/).
- HEIGHT (experiment, data/_height_exp/, not in the pipeline). Height per detection from the fixed camera centre
  and the ground homography: h = C_z (1 - dF / dG), dF the feet's and dG the head ray's ground distance from the
  camera. On the owner-labeled windows (59k clean detections, 18 players): median 1.70 m; the same player agrees
  within about 2 cm across windows (sd), players spread 7 cm (sd; 1.55 to 1.86 m); 58% of player pairs separable
  at 2 sd with a window of data. A tracklet part pins height to about 8 cm (5 to 10 s) or 5 cm (20 s+).
  Across games it carries (unlike appearance): 17 players, correlation 0.91, game 2 about 3 cm lower overall
  (camera refit), 3 cm sd after that offset. The confused pair above has the same height: no help there.
- Height as a filter on trusted stretches (leave-one-window-out, held-out readers): wrong stretches differ from the
  named player by 10.8 cm (median) vs 3.0 cm for right ones, but identity is already 97.8% right (8 wrong
  stretches); the best rule drops 7% of right time for +1 point. Not adopted.
- Height in linking unidentified stretches (exp12.py = exp10 + a height term: unit height vs the candidate's height
  from their read-identified units in the window, "someone else" vs the team's spread). Leave-one-window-out, same
  windows as exp10. Mean added coverage / pooled precision: without height, margin 3: +8.9% at 88.9%, margin 5:
  +4.4% at 90.6%; with height (weight 1), margin 5: +5.4% at 94.4%; weight 2: +7.2% at 91.9%. Per window, height
  helps clipE (margin 5: +6.5 -> +8.4% at 97 to 98%) and clipA (+4.6 -> +5.6% at 89 -> 97%), but clipB (the other
  half) stays at 63 to 73% right with or without it: its wrong links come from motion, appearance and its own
  misread base identity, not from height. NOT ADOPTED: no setting clears about 95% on every window, and trusted
  identity is 91 to 99.8%. Linking stays shelved; coverage needs more players read at all (Phase 8).

## Phase 16: a third game (data/g0916, 2026-09-30)
- 90 min video, same venue. The density scan found no halftime break: the opposing varsity team practised on the
  pitch at halftime, so people counts never dropped. new_game.py plan now writes overview_check.jpg (a frame every
  2 min) when no break is found, instead of stopping. Owner gave the halves: first 0:00:30-36:00, second from
  46:40. The play check (our players per frame per minute) put the end at 1:23:00 (5 to 8 before, 0 to 3 after);
  the two windows after it are in data/g0916/_outside_play/. 14 windows.
- Kits: owner mapped both pilots (first half: a mixed bench/keeper cluster as other; second half: warm-up
  tops as other). No goalkeeper kit class, so the goalkeeper got no identity in this game.
- Goal end: the goalkeeper vote had no goalkeeper; the kickoff check found no split because play was already
  under way at 46:40 (the kickoff is just before the first second-half window). Owner answered which side of the
  camera view we defended (right = X 0). Checked: player depth correlates with games 1 and 2 at 0.77 and 0.90
  (a flipped end gives negative).
- FIXED: the kickoff check now uses a probe, a 2-minute clip from 90 s before to 30 s after the confirmed second-half
  start (data/<game>/_kickoff, processed to roles and pitch positions only, never counted in stats). It looks for a
  kickoff formation: 2 s bins with 70%+ of our players' detections on one side of halfway and 70%+ of the
  opponents' on the other, held 6 s or more. Single frames at 80% missed game 2's restart (one misclassified player
  of four visible). Checked on all three games: game 1 (clipD, 26 s) and game 2 (w4340, 16 s) right; game 3's probe
  found 38 s just before 46:40 and matched the owner's answer. Open-play windows show such formations too (5 of 6
  consistent with that half's ends, probably kickoffs after goals; one 10 s first-half case the wrong way round),
  so the probe stays narrow and a disagreement with the goalkeeper vote still stops the run for the owner.
- Pitch: the existing camera fixed 100% of the pilot's frames (no refit). Identity 21 to 46% of target samples per
  window (games 1 and 2: 34 to 55%), 17 players, 159 identified player-minutes. Coaching pages
  data/g0916/coaching/; the multi-game view data/coaching_games/ now covers three games (20 players; 3 observations
  hold in every game a player was seen enough in, 8 only pooled, 18 in one game, 4 differ).
- Owner time: halves (a few minutes on the overview), two kit mappings, one question on the goal end.

## Phase 17: a per-game jersey round (new_game.py numbers, 2026-09-30)
- jersey_rare_label.py candidates --auto picks the numbers from a game's own reads: pairs of numbers splitting 5+
  of our tracklets' reads below the 80% agreement share, and numbers read 500+ times with under 10 s identified
  per 100 reads (other numbers: 20 to 75). At most 4 numbers, 36 crops each, spread over the windows. new_game.py
  numbers builds them; the owner labels (jersey_rare_label.py label); numbers --apply retrains the reader and
  redoes that game's identity, stats and pages.
- Bug found and fixed on the way: with two numbers confusing each other both ways, a read's text mapped to only
  one screen, so each number's own reads were shown as its partner's.
- Partner crops (reads of the other number shown as possible misreads) were never the number: 3 of 27 (game 2's
  late windows), 0 of 48 (game 3). The automatic round shows own reads only; --also stays for manual use.
- Game 3 round (about 5 min): 72 crops confirmed. The worst number (1231 reads, 47 s identified) rose to 106 s and
  its conflicting tracklets fell 13 -> 5; the other pair did not change; the game's identified time +1.6%
  (159 -> 161 player-minutes). Small: optional, worth it when a number is badly under-trusted, not every game.

## Phase 18: season view (coaching_html.py, 2026-09-30)
- Multi-game player pages: "Through the season", one point per game in date order for work rate and touches
  relative to teammates, +-2 SE whiskers (work rate: window-to-window spread; touches: Poisson on the count),
  team median at 1.0. Team page: a season table per game (players, identified player-minutes, team m/min weighted
  by minutes, touches per minute, on-the-ball share).
- coaching_tips.py --share-dir <folder> (several games): season_summary.json with those team numbers and the
  observation counts only (no names, numbers or images), per the sharing rule.
- Wording is count-neutral ("All games"; status "every" rather than "both") now that there are three games.
- Three games: 20 players; observations 3 in every game, 6 pooled only, 19 in one game, 6 differ, 4 with one game
  seen enough. Team m/min 82 / 87 / 88 by game (not comparable across games: noise floors differ).

## Phase 19: goalkeeper without a goalkeeper kit class (jersey_auto.py, 2026-09-30)
- Game 3's kit mapping had no goalkeeper class; its keeper was classed opponent (people "classed opponent" spent
  1710 s standing in our box). When a game's kit prototypes have no goalkeeper, jersey_auto.py identify now adds
  the goalkeeper by place and behaviour over every tracklet: not ours or an official, on the pitch, within 11 m of
  our goal line and inside the box's width for 60%+ of its samples, X spread under 6 m, 40+ samples; one per
  moment (closest to our goal line); trusted reads and other names win. Games 1 and 2 keep the kit-based rule.
- A mistake caught on the way: a first test (exp13) looked 96% right, but it ran on identity's samples, which only
  hold target/goalkeeper tracklets, so opponents were never in it. Retested over every tracklet (exp13b), counting
  every pick on an opponent/other tracklet as wrong: 94% right, 62% of the goalkeeper's owner-labeled time found
  (looser settings: 82% found at 82 to 87% right).
- Game 3: keeper 12.1 min, 2.8 m from our goal, 16.5 m behind the team line, 27 m/min, never at 4 m/s (games 1
  and 2: 3.5 to 4.3 m, 15 to 16 m behind, 45 m/min). Identified player-minutes 161 -> 174, 18 players.

## Phase 20: a learned possessor model (experiment, data/_events_exp/exp_learned*.py, 2026-09-30)
- Per ball-path frame and nearby player (within 3 body heights): distance, rank, gap to the next nearest, ball
  speed, ball detected and frames since, the player's speed, closeness over the surrounding half second, team;
  gradient-boosted classifier; the same segment logic and matcher as events.py. Leave-one-window-out on the 5 owner
  possession windows (111 possessions): possession F1 0.648 (R 64% P 66%) vs the current rules 0.633 (R 68% P
  59%; their defaults were chosen on these windows); touch 0.633 vs 0.627. A tie. NOT ADOPTED.
- Learning curve (held-out possession F1 by training windows): 1: 0.634, 2: 0.658, 3: 0.664, 4: 0.648 (sd about
  0.09). Flat after two windows: more owner event labels would not help.
- Why, on the owner's labeled possessed frames: 64% have the possessor nearest with the ball within 1 body height
  (easy for any method); 15% have no ball position at all; 12% have someone else nearer (crowding); 6% a detected
  ball over 1 body height away (a wrong object, mostly w5340); 3% an untracked possessor. Events are limited by
  the inputs (ball recall, crowding), not by the rules. Event work stops here unless ball recall improves.

## Phase 21: the coaching site for invited users (webapp/, 2026-09-30)
- Owner decisions: Google Cloud Run, Google sign-in, roles admin / coach (all players) / parent (assigned players;
  team pages keep team totals and name only their players), a photo per player from the video, school logo on team
  pages, opponent name and logo per game, signed-in strangers can request access (admins notified in the app and by
  email if SMTP is set).
- site_export.py writes coaching_tips' page inputs to data/site_export/; the site renders coaching_html.py's
  builders per request (web= option: shell, links, photos, visible jerseys). Local pages are byte-identical with
  web=None (checked on game 1 and all three games), and pages rebuilt from the export match the local ones exactly
  (0 of 42 differ; heatmap samples must stay unrounded, rounding to 1 mm moved cells).
- player_photo.py: candidates are the tallest confident identified detections, off the frame edge, not overlapping
  anyone (at most 2 per tracklet, round-robin over windows, 24 per player), 11:16 portrait, Lanczos x4 and mild
  sharpening; the owner picks one per player (crop grid + ZoomView). About 40 s for 3 windows.
- publish_site.py uploads a release (allow-list regex) and swaps current.json; webapp/deploy.py stages only the
  site's code for `gcloud run deploy`. Setup steps: webapp/README.md.
- tests/test_webapp.py (synthetic players): who sees what, CSRF, request flow and limits, mail content, headers,
  logo re-encoding, first admin. CI runs them (webapp-tests.yml).
- DEPLOYED 2026-09-30: GCP project coaching-site-dba494 (us-central1, billing on), Cloud Run service `coaching`
  (URL in webapp/README.md's redeploy command; the image builds on Cloud Build), private bucket
  coaching-site-dba494-data (uniform access, public access prevented), Firestore, service account coaching-site
  (bucket read, Firestore, its secrets only). Secrets: site-secret-key, site-google-secret, site-smtp-password
  (Gmail app password). Google sign-in app published (External, In production; basic scopes need no verification;
  run.app was accepted as the authorized domain once home page and /privacy links were set). New build projects
  need roles/run.builder on the compute default service account or `gcloud run deploy --source` fails.
- Emails: admins on each access request; the requester on approve/deny (PR #96). Only addresses, outcome, link.
- Cloud Run reserves paths ending in "z": /healthz never reaches the app, so the health route is /health.
- Icons (webapp/icons) were cut from the owner's icon sheet (a single mockup image, not separate files): the
  simplified play icon for 16-48 px, the full icon for 180-512 px.
- `new_game.py publish --bucket <bucket>`: every game's windows -> site_export.py -> photo picker for players with no
  photo decision yet -> publish_site.py (asks before upload). One command per new game after `run`.
- publish_site.py then fills each published game's MISSING opponent name (from the video name "... vs <opponent>
  <date>.mp4", styled as the admins wrote theirs: "X JV", "X C", no "Varsity") and logo ("<school> Logo.<png|jpg|
  jfif>" in the video's folder whose words start the opponent's) in Firestore; admin-set values are kept.
- Phone width (390 px, Playwright + Edge on the local site): no page-level sideways scroll on any page; tables keep
  cells on one line and scroll inside their box.
- Opponents (2026-10-02, PR #111/#112): after an upload, publish_site.py fills each published game's MISSING
  opponent name (from the video name "... vs <opponent> <date>.mp4", in the admins' style: "X JV", "X C", no
  "Varsity") and logo ("<school> Logo.<png|jpg|jfif>" in the video's folder whose words start the opponent's) in
  Firestore; admin-set values are kept.
- Activity (PR #113): sign-ins, sign-outs and every page a signed-in person opens (not photos or logos; refused pages
  included) go to Firestore `activity`, expiring after 180 days (TTL on expire_at, ACTIVE). Admin > Activity: per
  person (invited people who never came included) last seen, days active, sign-ins, pages; events with readable page
  names; the admin-change audit. Times in SITE_TZ (America/Chicago). The privacy page says so.
- Season page (PR #113/#114): the game list shows each game's opponent logo and name and links to the game.
- Theme button (PR #115): it never worked on the site (its script ran in <head> before the button existed); it now
  listens on the document. Checked in Edge with both OS themes.

## Phase 22: ball recall, where the ball is lost (experiment, data/_ball_recall/, 2026-09-30)
- Held out (game 2's w4840 / w0500 with the fold A / B venue models): of visible balls the linked path misses,
  97 / 85% are at a player's feet (within one body height of a box's bottom), and the detector has no candidate on
  60 to 77% of them even at confidence 0.05: the ball is hidden by feet and legs, not below a threshold.
- 2x-magnified overlapping tiles made the detector worse (candidate on the ball 70 -> 58%, 73 -> 66%): the model is
  trained at native scale.
- Carrying the ball with the player (fill a gap with a track's feet when the ball was lost and found at that same
  player's feet; 48 settings of radius, gap and one-sided carry) on 5 ball-labeled and 4 event-labeled windows:
  correct when visible 65.9 -> 66.3% at best, precision 80.5 -> 74-76%, possession F1 0.619 -> 0.627 (noise).
  Most missed at-feet balls are not in short gaps bracketed by one player. NOT ADOPTED. Ball recall is now limited
  by occlusion; the remaining idea (possession from player motion without the ball) is a research project.

## Phase 23: teams (JV and Varsity) and grades (2026-09-30)
- A game belongs to a team: "team" in its game.local.json (none = jv, the first team; ids are neutral, school and
  team names never go in committed files). Rosters: roster.csv (jv, repo root, as before) and
  data/teams/<team>/roster.csv; sv_common.team_of / read_roster / roster_file. Every roster reader goes through
  them (player_stats, events, coaching_tips, jersey_auto, jersey_label, jersey_rare_label, new_game, player_photo).
- Rosters carry class_of (graduation year); sv_common.grade gives Freshman..Senior for the school year of the
  latest game (a school year starts in August), shown with the player on local pages and the site.
- Identity for a team without owner-labeled windows: `--from` windows of another team are ignored (same_team);
  the first decode uses reads only, then appearance is learned from this window's read-confirmed samples (the
  existing second pass), and the goalkeeper is found by place and behaviour (keeper_by_place). JV unchanged
  (clipE held out from A+B: 49.2% identified, 97.5% right; documented 48.7 / 97.2 before later reader retraining).
- A new team's first game: `new_game.py numbers --seed` (reads with the original reader, jersey_auto.py reads;
  candidate crops for every roster number) -> owner labels -> `numbers --apply` retrains the shared reader.
- Coaching pages per team: data/coaching_games (jv) and data/coaching_games_<team>. The site export and release
  are per team (site_export/<team>/, release <stamp>/<team>/ + teams.json); the site serves /t/<team>/... with a
  team switch, per-team names, logos, opponents and access (users: admin flag + teams {team: {role, players}}; users,
  the team setting and game opponents saved before teams count as jv).
- Varsity game 1 (data/v0903, 2026-09-03 video, 2.5 h recording at night): halves first from the overview, then the
  owner's exact times 0:05:53-0:51:18 and 1:03:10-1:30:20 (a 27-minute second half). Kit mapping by the owner per
  half (second half: one cluster mixed our players with the opposing goalkeeper, mapped target). Goal end from the
  owner (the kickoff probe found no painted lines: halftime huddle under the lights); in this camera's view X grows
  to the left, so the left goal is X = length.
- Seed round: 562 candidate crops for 17 numbers, owner confirmed 347 in about 25 min (none for the goalkeeper,
  whose back rarely faces the camera, nor for one number that hardly played). Held out on the confirmed second-half
  crops (133, 14 numbers), a reader trained without them: 131 confident reads at 98.5% right vs 118 at 98.3% for
  the JV-trained reader (candidates come from reads, which flatters every reader). After `numbers --apply`:
  37-64% of target samples identified per window (JV games 34-55%), 204 identified player-minutes (202 in play).
- Goalkeeper: found by place in only 2 windows. Tracks living in our box at night are classed unknown (16 of 19):
  the keeper's kit matches no mapped cluster and the track breaks into short pieces, under the rule's 40 samples.
  FIXED in part (owner screenshot of the keeper's kit): goalkeeper prototypes from the medians of the tracks in that
  kit colour living in our box (29 of 37 such tracks, all classed unknown or other before), one per half, and a track
  the colours call goalkeeper needs only 8 samples (about 4 s) for the place rule (GKP_MIN_SAMPLES_ROLE; the rule is
  only used for a team without labeled windows or a game without a keeper kit class, so games 1-3 are unaffected).
  Keeper identified 1.3 -> 1.9 min, 2 -> 5 windows; the keeper is on camera in our box only about 3.7 min all game.
- Kit mapping without the owner: jersey reads cannot do it (the fine-tuned reader leans toward our numbers on any
  shirt, and opponents wear the same low numbers: on the 9/3 pilots opponent clusters read as our numbers 47 to 82%
  of the time). `new_game.py setup --kits-like <game>` maps clusters by colour against a previous game of the same
  team in the same kit: near our kit -> target, near its other roles -> that role, else a cluster with 10%+ of the
  rows -> opponent, else other. Reproduces the owner's 9/3 mapping on both 9/3 pilots.
  On the new games it was not enough: 9/3 was at night under lights, 9/5 and 9/16 start in daylight, so our 9/3
  kit moved past the cut (distance 21 vs 18) and the setup stopped safely (CORRECTED, Phase 24: 9/16 was a
  different kit, not the light); once (9/16 first half) it called a 5%
  sliver cluster ours, now refused (our team must hold 10%+ of the rows). Reads with the original reader did not
  decide either: on 9/16 one team's kit read 72% Varsity numbers (only 32 reads) and the other's 46%; on 9/5
  the home kit's most read numbers were JV ones (16, 1, 7, 27), so that recording may not show the Varsity game at all.
  Left for the owner: which kit was ours. The GPU-heavy stages (detection, ball model, pitch mask) ran for every
  window meanwhile, so setup/run finish fast once the kits are mapped.
- Varsity 9/16 (data/v0916): owner mapped our kit (a different kit from 9/3). Kick-off formation in the
  first second-half window gave the goal end (X = 0 in the first half). Halves from the overview, then trimmed with
  the play check to 0:23-1:09 and 1:19-2:11. 20 windows with identity, about 286 identified player-minutes.
  Published. The last window (2:12-2:17) is after the final whistle: no reads at all, which crashed identity
  (empty reads table; fixed) and then coaching (no player_events.csv; fixed).
- Varsity 9/5: ABANDONED (owner, 2026-10-02): the recording holds only the last 12 minutes of the game, too little
  to use and with no way to confirm the goal end. Its processed data is set aside in data/_abandoned/v0905 (folders
  starting with "_" are never picked up), and the owner removed the video.
- Off-roster numbers: on our tracks, #16 (2961 reads on 9/5, 2031 on 9/16), #7 (997, 426), #10 and #45 (9/5) read
  as often as the top roster numbers, so players not on those games' rosters played (JV players playing up?).
  Identity only names roster numbers, so they stay unnamed until the owner adds them to the game rosters.
- Recordings of 9/5 and 9/16 (owner's videos): 9/5 holds about 12 min of play at its start, then an empty pitch for
  2 h; 9/16 a full game, halves estimated 0:23-1:09 and 1:17-2:18 (halftime 1:09-1:17 clear in the crowd counts).
- Night pitch fit: 50-99% of 2 s frames fixed per window (daylight 89-100%). A camera refit from this game's fixed
  frames moved the centre 3 cm and changed nothing (62 -> 62, 50 -> 49, 59 -> 60%): the lines are fainter, not
  the camera elsewhere. Rejected frames score a median of about 1300 against the 2000 cut, and wrong locks scored up
  to 1600 in the anchor fit, so lowering the cut is unsafe without night anchors. Gaps are bridged by camera motion;
  the noise floor (21-30 m/min) is within game 1's daylight range (17-29). Left as it is.
- Rosters per game (owner, 2026-10-01): data/<game>/roster.csv (game 1: data/roster.csv) wins over the team's;
  sv_common.roster_file / read_roster take the window. Several games pooled (coaching, site) use every game's
  players, the latest game's entry for a jersey listed twice. The owner's six game rosters: the three JV ones equal
  the JV roster, the 9/5 and 9/16 Varsity ones the Varsity roster; the 9/3 Varsity screenshot shows the JV list
  (likely the wrong file: the 9/3 seed round confirmed Varsity-only numbers), so 9/3 keeps the Varsity roster.
  Players are still keyed by jersey within a team: a number worn by different players in different games would
  merge them in the season view (not seen so far).
- Halves corrected after processing: the game file keeps its processed windows ("windows"), and stats and coaching
  count only time inside the halves (sv_common.play_mask; games without halves, the first game, are untouched).
- Lessons from the run: (1) stopping a run while it cuts clips leaves truncated clip.mp4 files that the next run
  reuses ("moov atom not found"): delete those folders. (2) An identity run beside two detections filled GPU memory
  and everything ran at a quarter speed or less (detection 3 fps instead of 13); new_game.py now gives identity the
  whole GPU (two units, detection one). (3) One failed window stopped the queue; failures are now reported at the end
  and the other windows finish. (4) A team the reader has not learned skips identity in `run` (team_ready) and goes
  straight to the seed round. (5) Background tasks stop after 2 h: a whole-game run is started as its own process.

## Phase 24: an outside review checked against the findings (2026-10-02)
- The owner asked for an evaluation of an outside AI review of the pipeline. Already done in the code: per-sample
  fatigue phases (coaching_tips.phase), pitch width from the camera file, Lab torso/legs on non-grass pixels,
  comparison within roles. Contradicted by earlier measurements: touches from ball velocity changes (Phase 12: 11%
  recall), a physics/carry ball filter (Phase 22), a multi-frame ball detector (Phase 22: misses are balls hidden at
  feet), formation priors for linking (Phases 8, 15). Not possible with a panning camera that shows part of the
  pitch: pitch control, line compactness, rest defense (they also need passes and turnovers, both unreliable).
- Tried: matching kits across games with lightness down-weighted (data from every pilot; truth = the owner's
  mappings, recovered as the pilot cluster centres stored in each game's prototypes). JV 9/16 <-> 9/22 (same kit):
  full Lab 4 of 4 right, our kit 1.8 to 6.7 times nearer than the next big cluster; weight 0.25: 1.1 to 2.4;
  colour only (a*b*): 3 of 4. Lightness separates our kit from others, so dropping it only shrinks the margin. Not
  adopted. Varsity 9/3 <-> 9/16 fails with any weight because the kits differ (refusing is right). No same-kit pair
  under different light exists yet (9/18 Varsity will be the first). g1001 (JV 10/1, mapped by colour, no owner
  check) has thin margins (1.2 to 1.3) and 4 to 6 opponents per frame with 2 to 5 "other": an opponent cluster may
  sit in other/official; worth an owner look at its kit_clusters.png before publishing.
- Added: "Watch on video" on local player pages (single game and across games, never the site): the longest
  stretches on camera, the fastest running (4 m/s or more) and the longest possessions and touches, 5 each, 15 s
  apart; a click plays the game video (relative link from the page, so only on this computer) from 3 s before.
  Checked in Edge: the video seeks to the moment. The player is not marked in the video.
- Tried: feet from pose keypoints (yolo11m-pose on each tracklet crop, midpoint of the ankles, lower ankle's height)
  instead of the box's bottom centre, clipE, same tracks and calibration. Both ankles confident on 96% of rows;
  ankles sit 9% of box height above the box bottom (sd 2.6%). Noise floor on still people 17.8 -> 20.5 m/min
  (worse), player median 76.9 -> 78.6 m/min. Keypoint jitter on 60 to 90 px players exceeds the box bottom's. Not
  adopted (scratch script only; the pose weights in the repo root are git-ignored).
- Site clips (owner's choice of four options, 2026-10-02: short clips of our players, not whole videos or links):
  site_clips.py cuts each "Watch on video" moment into a 10 s clip, 960x540 at native resolution following the
  player (identified boxes, 1 s smoothing), a white triangle over the player where an identified box is within
  0.2 s, no audio; about 3.5 s and 0.5 to 0.9 MB each. Cut once into data/site_clips/<team>/<game>/; site_export.py
  writes clips.json per game; publish_site.py uploads them outside the releases (clips/<team>/<game>/, only new
  ones, deletes unnamed ones); the site serves /t/<team>/clip/<game>/<NN>_<ms>.mp4 only to viewers allowed to see
  #NN and only names its release lists (Range requests answered for iPhone Safari). Privacy page updated.

## Pipeline status
1. Ingest and detection cache: detect_cache.py (done, validated on two full clips)
2. Offline tracker replay and sweep: replay_trackers.py (done; config retuned by blind owner purity labels to buffer 1 s, match 0.95 and APPLIED to both clips - see Phase 6 findings)
3. Ball linking: ball_link.py (done; hand-checked on one 60 s window, 97% correct when detected, interpolation weak)
4. Team classification: team_classify.py (done; hand-checked, 65 to 73% accuracy, capped by tracklet fragmentation, goalkeeper weak)
5. Pitch: pitch_mask.py on-pitch test (done, sampled), pitch_calibrate.py anchor homography (done for both clips - clipB: 4 anchors, 0.5 to 5 m cross-check; clipA: 6 anchors, accepted with a known higher error floor (3 to 60+ m) from noisier camera motion in that clip - see Phase 3 findings)
6. Event detection: events.py (possession, touch, pass, turnover done and unverified; shots not started, need calibration)
7. Identity assignment with roster, tracklet stitching, review UI: roster.csv, tracklet_stitch.py and jersey_label.py DONE on the new tracks for both clips, with per-tracklet naming and splitting at switches (Phase 6b): clipA 80% and clipB 78% of target/goalkeeper tracked time identified, 15 and 16 of 21 roster players. The review UI (third piece) is NOT STARTED, not urgent. AUTOMATIC identity (jersey_auto.py identify, Phase 8): about 40% of labeled time held out at 91 to 99.8% right; written for clipF.
8. Stats database and coaching tips: per-player stats DONE, first pass (player_stats.py, Phase 7): running stats usable within a clip, event stats too sparse. Coaching tips NOT STARTED.

## Commands
- One command for the current stage:
  python run_all.py --video <game.mp4> --start HH:MM:SS --duration 300 --out data\clipA --share-dir <folder Claude can read>
- Individual stages: detect_cache.py, replay_trackers.py, ball_link.py (each has --help)
- Events: events.py --run <run> [--montage] (needs ball_path.csv, tracklet_roles.csv)
- Hand-label events: event_label.py label --run <run> --start <s> --duration <s>, then event_label.py score --run <run> (writes events_truth.csv, events_score.json)
- Team roles: team_classify.py calibrate / assign / classify (needs clip.mp4 in the run folder)
- Venue ball model: ball_finetune.py dataset/train/detect/evaluate (see its --help); pipeline stage: ball_finetune.py apply --run <run> (models/ball/venue.pt, then events.py)
- Pick validation clips: scan_density.py (samples the whole game, use --reuse to re-pick from a saved scan)
- Automatic pitch anchors: pitch_autoanchor.py run --run <run> --anchors <owner anchors json> (writes RUN/pitch_anchors_auto.local.json and a report of uncovered stretches), then pitch_calibrate.py apply --anchors RUN/pitch_anchors_auto.local.json. Check against held-out owner anchors: pitch_autoanchor.py evaluate
- Automatic pitch with no owner anchors (current default): pitch_ptz.py run --run <run> (needs data/pitch_camera.local.json from pitch_ptz.py fit; writes RUN/pitch_anchors_ptz.local.json), then pitch_calibrate.py apply --run <run> --anchors RUN/pitch_anchors_ptz.local.json. Held-out check: pitch_ptz.py evaluate
- Place pitch anchors: pitch_anchor_ui.py --run <run> (local browser UI; writes pitch_anchors.local.json)
- Tracker sweeps that leave the pipeline untouched: replay_trackers.py --run <run> --grid strict|reid|refind --fps-list 15 --sweep <name> [--reid <npz>] (writes data/<run>/sweeps/<name>/; identity-scored against identity_rows.csv.gz, frozen from the jersey labels on first use)
- Appearance embeddings: reid_cache.py --run <run> (CPU, writes cache/reid_*.npz)
- Blind tracklet purity check: track_purity_label.py label --run <run> --sweep <name> [--configs 7,24 --n 20] [--redo ITEMS], then track_purity_label.py score --run <run> --sweep <name>. In the window: click cycles a crop's person (A-H), shift+click marks a swap from that crop on, ctrl+click marks unknown, Enter saves
- Stitch tracklets: tracklet_stitch.py --run <run> [--montage] (needs tracklet_roles.csv, tracklet_colors.csv; writes tracklet_stitch.csv)
- Per-player stats: player_stats.py --runs data\clipA,data\clipB [--share-dir <folder>] (needs player_identity.csv, identity_segments.csv, tracklet_pitch_xy.csv.gz, events.csv; writes player_stats.csv, player_events.csv, stats_report.json per run and data/stats.sqlite)
- Jersey suggestions for a new window: jersey_suggest.py suggest --run <run> --from <labeled runs> (DINOv2 on GPU; writes jersey_suggestions.csv, jersey_suggest.npz; jersey_label.py shows them). Score on a labeled window: jersey_suggest.py evaluate --run <run> --from <other runs>
- Coaching tips for every identified player (local output data/coaching/, or data/<game>/coaching/): coaching_tips.py --runs <windows with identity> (needs the game.local.json). Windows of several games -> data/coaching_games/ (per-game and pooled, observations tagged by game)
- A new game, end to end: new_game.py plan --video <video> --game <name>; confirm [--first-half/--second-half]; setup (stops once for the kit mapping: team_classify.py assign); run
- Process game windows end to end, no owner steps (resumable; skips stages whose output exists): run_windows.py --video <game.mp4> --windows clipH=00:00:00,... --labeled <owner-labeled windows> --stats-runs <other windows>
- Identify players automatically (no owner labeling): jersey_auto.py identify --run <run> --from <labeled runs> [--write] (dry run by default; on a labeled window it scores against the owner's labels; writes player_identity.csv and identity_segments.csv, every identified stretch a segment). Reader: jersey_auto.py finetune --runs <labeled runs> (models/jersey/parseq_ft_game.pt, git-ignored)
- Confirm rarely labeled jersey numbers (owner, a few minutes): jersey_rare_label.py candidates --runs <windows> --labeled <owner-labeled windows>, then jersey_rare_label.py label (one screen per number: click the crops that show it, Enter). Writes data/jersey_rare_truth.csv; jersey_auto.py finetune picks it up.
- Identify players by hand: jersey_label.py label --run <run> [--redo-mixed], then jersey_label.py apply --run <run> (needs roster.csv, tracklet_stitch.csv; writes jersey_truth.csv, jersey_tracklets.csv, player_identity.csv). Stitched players show their tracklets (T1, T2, a yellow bar at each join); click a crop to name just that tracklet, shift+click a crop to split its tracklet where another person starts (parts T1a/T1b, magenta bar; frames around the switch get no identity), then x or Enter finishes the player as "split". apply also writes identity_segments.csv (named parts of split tracklets as ci ranges). --redo-mixed re-opens players marked mixed to name their tracklets.
- Coaching site (webapp/README.md): after a new game's `run`, `new_game.py publish --bucket coaching-site-dba494-data`
  (export, photos for new players, upload; asks first). By hand: site_export.py, player_photo.py candidates/label,
  publish_site.py. Redeploy code: python webapp/deploy.py stage, then gcloud run deploy coaching --source
  webapp/_build --region us-central1 --project coaching-site-dba494 --quiet (keeps env and secrets). Tests: pytest -q tests
- phase1_track.py is the original tracker-in-the-loop script. Superseded, kept for reference.

## Next actions
1. DONE: a third hand-labeled window (same game, a different 5 minutes) confirmed min-conf 0.25 generalizes for precision (see Phase 1b findings). It also found a real detector blind spot on the boundary track, which is a separate, deeper limitation (retraining or new examples, not a pipeline setting) and is not being worked on now.
2. DONE, with caveats: role_label.py hand-check completed on both clips (see Phase 2 findings). 65 to 73% accuracy, capped by tracklet ID fragmentation rather than the color model. Not retuned: per-role sample sizes are too small to trust a parameter change. Revisit after tracklet stitching (step 7) or a larger labeled set.
3. DONE for both clips: clipB has 4 anchors (130s, 180.1s, 197s, 270s), all cross-checking within 0.5 to 5.05 m. clipA has 6 anchors (4s, 66s, 96s, 125s, 156s, 220s); its calibration is accepted with a known, measured limitation (3 to 60+ m error depending on distance from an anchor, versus clipB's much tighter fit) rather than fixed further - see Phase 3 findings for why. `tracklet_pitch_xy.csv.gz` in both data/clipA and data/clipB is ready to use.
4. DONE, first pass: 60 s hand-labeled on clipA (150 to 210 s) with event_label.py and scored - see Phase 4 findings. Possession scores reasonably (56 to 69%), touch is weak (21 to 33%), turnover/pass samples are too small (1 and 0 true events) to say much. A threshold-fix hypothesis (extend the same-player merge guard to turnovers) was tried and reverted - it broke the one real turnover in this sample. Not retuned: this single window is too small to trust a parameter change, same caution as Phase 2's role thresholds. A larger labeled sample (more windows, ideally on clipB too) would be needed before tuning is worthwhile.
5. STOP POINT LIFTED by the owner (2026-09-23): moving into step 7.
6. DONE for both clips: step 7's identity-assignment pass (roster.csv, tracklet_stitch.py retuned against real labels, jersey_label.py) - see Phase 5/5b findings. clipA 26/95 tracklets identified (11/21 roster players), clipB 16/74 (9/21) - consistent, not clipA-specific. Open, not urgent: the review UI (third piece of step 7), and a fix for the substitution-transition tracklet failure mode (one report so far, not common enough yet to justify the work). Owner's call on what step 7 or step 8 work comes next.
7. DONE 2026-09-24: tracker swap retune (Phase 6) applied, and identity relabeled on both clips with per-tracklet naming and splitting (Phase 6b): about 80% of target/goalkeeper tracked time identified. Owner's call on what comes next (step 8, stats, is now well supported on identity).
12. RESUME HERE (saved 2026-10-02 09:45): two teams (JV, Varsity) on the site. Done this session: 9/16 Varsity
   redone with #16 (a JV player who played up) and uploaded with his photo; site changes PRs #111-#115 (Phase 21
   notes). Running when saved, as detached processes (they survive a restart): (a) JV 10/1 (data/g1001): setup done
   (kits matched by colour to 9/16, pilot pitch 99%), `run` started 09:34; (b) then Varsity 9/18 vs a new opponent
   (data/v0918, waits for (a)): setup --kits-like v0916, then run. Its halves are Claude's estimates from 30 s stills
   (0:02:30-0:50:00, 1:00:00-1:49:30; the plan's proposal took the start of halftime for a water break). Next, for
   each: trim halves with the play check (`confirm` keeps windows, rerun `run`), check identity per window and the
   goal end, `new_game.py publish` (opponent and logo fill in automatically). 9/5 Varsity abandoned. Details of the
   session state: the memory resume point.
11. DONE, kept for history (saved 2026-09-30): THREE GAMES processed (data/ = game 1 2026-09-18, data/g0922 = game 2,
   data/g0916 = game 3), Phases 13 to 20 done, main clean at PR #92. Owner's list after game 3 (per-game jersey
   round, season view, goalkeeper without a keeper kit, event detection) is finished: the jersey round is optional
   (small gain), the learned event model tied the rules and more event labels would not help (Phase 20).
   Current state of the tools:
   - A new game: new_game.py plan -> confirm (owner: halves; overview_check.jpg if no halftime break shows) ->
     setup (owner: kit mapping per half) -> run (goal end from the goalkeeper vote and the kickoff probe; stops if
     they disagree) -> optional numbers (about 5 min). Then coaching_tips.py over all games' windows for
     data/coaching_games/ (add --share-dir for the aggregate-only season_summary.json).
   - Coaching pages: per game in <game>/coaching/, all games in data/coaching_games/ (local only, names).
   Next, owner's call: more games (the main lever: most observations still hold in one game only); ball recall is
   the only event lever left (15% of labeled possessed frames have no ball position). Open, not pursued: running
   stats were never checked against a measured distance (a timed, measured run on camera would settle it).
10. DONE, kept for history (saved 2026-09-28): two games processed (Phases 9 to 13). Owner's order: (1) DONE (Phase 13):
   two-game coaching view with touches, command: coaching_tips.py --runs <game 1 windows>,<game 2 windows>; it
   also found and fixed game 2's flipped ends. (2) DONE (Phase 14): venue ball model, ball path +30 points held
   out, events unchanged; next lever for events is the event rules. (3) DONE (Phase 15): late identity drop
   explained (a substitute's number misread as a teammate's), partly fixed (19-20% -> 22-24%). Next: height in
   linking (tested, Phase 15: not adopted), more games. Was: (2) a ball
   detector fine-tuned for this venue (game 2 misses about a third of balls in possessed frames and a quarter of
   its detections are wrong objects; owner ball labels exist for clipA, clipB, clipD; may need ~30 min of owner
   ball labelling on game 2), (3) why identity falls to about 20% in game 2's last two windows, (4) more games
   with new_game.py as videos arrive.
9. DONE, kept for history (saved 2026-09-26, later): see Phase 8. (a) DONE 2026-09-26: automatic pitch applied to clipA, B, D,
   E, F, player_stats rerun. (b) DONE 2026-09-26: automatic identity (jersey_auto.py identify) with
   read-count trust, production reader, clipF identity written and stats rerun (Phase 8). Open, owner's call:
   raise coverage (about 40% held out, 50% on clipF) by linking identity across tracklets; the goalkeeper (no
   reads); confirm the two probable clipF substitutions. DONE: a window end to end with no owner steps (clipG):
   it works; identity coverage is the weak point (30% of target time) and falls as windows get harder to read.
8. SUPERSEDED by 9 (owner chose full automation over timing manual steps). Next: window 25:00 to 30:00 as data\clipF, the first window using both jersey suggestions (Phase 7c) and automatic pitch anchors (Phase 7d); the point is to measure owner time against clipE's (anchors ~65 min, jersey ~85 min). Steps:
   a. python run_all.py --video <the game video in videos\> --start 00:25:00 --duration 300 --out data\clipF (detection ~11 min GPU, chosen tracker, ball linking), then pitch_mask.py, team_classify.py classify, tracklet_stitch.py, events.py on data\clipF. Check zoom volatility first (median 2 s scale change from the cache, Phase 7b).
   b. Owner: pitch_anchor_ui.py --run data\clipF, about ONE anchor per stretch where a penalty area is visible (record start/end times). Then pitch_autoanchor.py run --run data\clipF --anchors data\clipF\pitch_anchors.local.json; if its report lists uncovered stretches that show a box, owner adds an anchor there and it is rerun. Then pitch_calibrate.py apply --anchors data\clipF\pitch_anchors_auto.local.json.
   c. jersey_suggest.py suggest --run data\clipF --from data\clipA,data\clipB,data\clipE (clipE, 5 min earlier, carries the weight). Owner: jersey_label.py label --run data\clipF with suggestions (record times), then apply.
   d. player_stats.py --runs data\clipA,data\clipB,data\clipE,data\clipF, and record owner time per step and suggestion hit rate in CLAUDE.md.
   After that: coaching tips (step 8 part 2) are not started; ball-event improvement (more event labels) was deferred. clipD (43:00 to 48:00) is calibrated but has no jersey labels and was set aside (goal end only in its second half).
