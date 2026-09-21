# Soccer analysis: Phase 1b (detection cache, tracker replay, ball linking)

## Setup (Windows PowerShell), once
1. python -m venv .venv
2. .venv\Scripts\Activate.ps1
3. Install PyTorch with CUDA from https://pytorch.org
4. pip install -r requirements.txt
5. winget install Gyan.FFmpeg   (then open a new terminal)
6. python -c "import torch; print(torch.cuda.is_available())"   must print True

## Run everything with one command
python run_all.py --video D:\games\game.mp4 --start 00:14:00 --duration 300 --out data\clipA --share-dir <share-folder>

- Detection runs at about 12 fps on the 3060 (yolo11m, 1920 px), so expect roughly 12 minutes for a 5 minute clip. It runs once.
- Tracker replay and ball linking take a few minutes with the quick grid. They reuse the cache.
- Rerun with a wider search, no redetection: python run_all.py ... --grid full
- Rebuild the cache only if you change the model or clip: add --redetect

## What you get in data\clipA
- report.md            everything in one place, the file to read first
- sweep_results.csv    every tracker configuration and its proxy metrics
- best_tracklets.csv.gz  tracks from the best config, with kit colors attached
- ball_path.csv        one ball position per frame where a plausible path exists, marked detected or interpolated
- cache\               detections.csv.gz, camera.csv, meta.json (input to every later stage)

## Reading the report
- Camera inliers: the median should be well above 50. If it is low, camera motion is unreliable and BoT-SORT results are not trustworthy.
- Tracker score = new IDs per minute + 3 x swap suspects per minute. Lower is better. It is a proxy, not ground truth.
- Ball path is a hypothesis. Verify against a minute of hand-checked frames.

## Phase 1 record (tracker in the loop, superseded by the pipeline above)
`phase1_track.py` and `phase1_ball_baseline.py` are kept for reference. Clip: one JV game video, start 00:14:00,
300 s, yolo11m, imgsz 1920. BoT-SORT (`trackers\botsort_long.yaml`) was chosen by visual review plus summary.json.
Manual ID-switch counts were skipped, so the "4 players per tracker" exit criterion was not met as written.

| Metric | ByteTrack | BoT-SORT |
|---|---|---|
| Unique person IDs | 2041 | 790 |
| Median track length | 1.2 s | 4.0 s |
| New IDs per minute after 10 s | 417.5 | 159.7 |
| Ball frames, conf >= 0.1 | 9.2% | 14.9% |

Detector-only ball baseline (`phase1_ball_baseline.py`, no tracker), the reference for Phase 4:
- conf >= 0.1: ball box in 67.7% of frames, longest gap 5.1 s
- conf >= 0.3: ball box in 34.4% of frames, longest gap 13.0 s

These count a frame as a hit if any ball box is present, so false positives inflate them. Nothing has checked the
boxes against the real ball. Even BoT-SORT still produced about 790 IDs for about 23 people, which is why the
pipeline above replays trackers offline with the camera motion from the cache.

## Privacy
data\ and roster.csv are git-ignored. The share-dir copy contains aggregate numbers only.
Do not put names, jersey numbers, school or team names in committed files. See CLAUDE.md.
