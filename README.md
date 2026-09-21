# Soccer analysis, Phase 1: detection and tracking

## Setup (Windows PowerShell)
1. python -m venv .venv
2. .venv\Scripts\Activate.ps1
3. Install PyTorch with CUDA using the selector at https://pytorch.org
4. pip install -r requirements.txt
5. winget install Gyan.FFmpeg   (then open a new terminal)
6. python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   Must print True and your GPU name before you continue.

## Run
Pick 5 minutes of live play after kickoff, not warmups or halftime.

python phase1_track.py --video D:\games\game.mp4 --start 00:14:00 --duration 300 --tracker trackers\bytetrack_long.yaml --out data\run_bytetrack
python phase1_track.py --video D:\games\game.mp4 --start 00:14:00 --duration 300 --out data\run_botsort

The second run reuses the same clip only if you point it at the same --out folder,
so either copy data\run_bytetrack\clip.mp4 across or let it re-cut (about a minute).

## Review
1. Open annotated.mp4 from each run.
2. Follow 4 players (pick them by jersey number and color) for the full clip.
3. Log every ID switch (time, old ID, new ID) in id_switches.csv in the run folder.
4. Compare summary.json across the two runs.

## Exit criteria for Phase 1
- ID switches counted for 4 tracked players on each tracker.
- Tracker chosen based on those counts.
- Ball detection percentage recorded, as the baseline for Phase 4.

## Phase 1 result
Chosen tracker: BoT-SORT (`trackers\botsort_long.yaml`).

Basis: visual review of both annotated.mp4 files (BoT-SORT tracked players more consistently) plus
summary.json. Manual ID-switch counts were skipped by decision, so the "4 players per tracker" exit
criterion was not met as written.

Clip: one JV game video, start 00:14:00, 300 s, yolo11m, imgsz 1920.

| Metric | ByteTrack | BoT-SORT |
|---|---|---|
| Unique person IDs | 2041 | 790 |
| Median track length | 1.2 s | 4.0 s |
| New IDs per minute after 10 s | 417.5 | 159.7 |
| Ball frames, conf >= 0.1 | 9.2% | 14.9% |

Ball detection baseline for Phase 4, detector only (`phase1_ball_baseline.py`, no tracker):
- conf >= 0.1: ball box in 67.7% of frames, longest gap 5.1 s
- conf >= 0.3: ball box in 34.4% of frames, longest gap 13.0 s

These count a frame as a hit if any ball box is present, so false positives inflate them. Nothing has
checked the boxes against the real ball. The tracked runs showed only 9.2% (ByteTrack) and 14.9%
(BoT-SORT) at conf >= 0.1, because the trackers drop most ball boxes.

## Privacy
data/ and roster.csv are git-ignored. Crops and annotated video show minors, keep them local.
