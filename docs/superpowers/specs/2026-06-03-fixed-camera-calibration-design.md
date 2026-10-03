# Fixed-Camera Calibration for ChessWorld AI Streams — Design

**Date:** 2026-06-03
**Goal:** Make `chessworld.mp4`, `chess2.mp4`, `chess3.mp4` (screen-recorded from
ChessWorld AI's YouTube streams, all the same "CP IP Cam" green-board setup) produce
game3-level-accurate PGN plus a 3-panel demo video each.

## Problem (evidence-backed)

The three streams share identical hardware (IP cam + green vinyl roll-up board + plastic
Staunton pieces) but differ in geometric distortion. Grid-overlay visualizations
(`exploration/visualize_grid.py`) showed:

- **chessworld** — mild top-edge bow; recall already good (41 pieces detected).
- **chess2** — steep keystone; the auto board-detector grabs the **wrong quadrilateral**,
  mapping the 8×8 grid onto empty table. Pieces assigned to wrong cells; recall low (10–24).
- **chess3** — IP-cam **barrel curvature**; the uniform grid can't match the curved
  squares, so rows drift across cells at top/bottom.

Piece *recognition* works when the warp is good (chessworld). **The bottleneck is geometry
— the flat homography can't flatten keystone/curvature.** The diagnosis (occupancy score
capped ~50 on noisy footage vs 59–64 on clean game3) is in `docs/CCTV_NOISE_DIAGNOSIS.md`.

## Approach: lock the geometry once per static camera

These are fixed cameras, so calibrate each once and reuse it for the whole video —
standard practice for deployed fixed-camera CV. Two corrections:

1. **Radial undistortion** (`k1`,`k2`) to flatten lens curvature (chess3).
2. **Locked board corners** for a correct, stable warp (fixes chess2's wrong-quad; removes
   per-frame drift everywhere).

## Components (each independently testable)

### 1. Calibration config — `calibration/<stem>.json`
`{ k1, k2, corners (4×[x,y] post-undistort), rotation, border_margin }`. Written once.

### 2. Calibration builder — `scripts/calibrate_camera.py`
- Sample early frames.
- **Grid-search** `k1 × rotation × margin`, mirroring the existing rotation/margin search:
  - undistort(k1) → robust green-board 4-corner detection (HSV mask → largest component →
    convex hull → 4 corners, ordered) → homography warp → rotate.
  - Score (position-independent, piece-robust):
    - **Border straightness** — the printed board border should fit 4 straight lines after
      undistortion; minimize line-fit residual (primary signal for `k1`; pieces don't touch
      the border).
    - **Grid periodicity** — warped board should have strong green/white edges at 80px
      spacing; score the periodic edge energy.
    - Tie-break: YOLO piece count + how centered detections sit in cells.
- Lock best combo → write JSON.
- If best score < confidence bar, save an annotated frame and request a one-time manual
  4-corner click (`--manual` fallback). Expected possible for chess2.

### 3. Pipeline integration — `src/pipeline.py`, `main.py`
- `main.py --calibration <json>` (auto-loads `calibration/<stem>.json` if present).
- When a calibration is loaded, per processed frame: `undistort(k1,k2) → warp(locked
  corners) → rotate`. Board auto-detection is skipped entirely.
- **No calibration loaded → today's behavior unchanged.** game3 path, defaults, and all 25
  unit tests are untouched.

### 4. Run + demos
For each of the three: `python main.py --input videos/<v>.mp4 --output output/<v>.pgn
--calibration calibration/<v>.json --save-demo output/demo_<v>.mp4`.

### 5. Validation
- In-loop proxy: reliability median (the metric from Issue 20) must cross **55**; visual
  review of each demo.
- Exact accuracy: compare against ground-truth games / end-position screenshots (user
  provides). No ground-truth PGN exists otherwise.
- If chess2 recall stays low after calibration → fallback: targeted retrain on chess2
  frames (extraction tooling already exists: `scripts/extract_cctv_frames.py`).

## Out of scope / YAGNI
- No multi-camera auto-discovery, no live calibration UI beyond the one-time corner click.
- No model retrain unless calibration is measured to be insufficient.

## Risks
- **chess2 steep angle** foreshortens/overlaps pieces → recall may need a retrain.
- **No ground truth** → "game3-level accuracy" certified only via user-provided games.
- Undistortion is estimated from the board itself (no checkerboard calibration capture),
  so `k1` is approximate; grid-search + border-straightness keeps it well-conditioned.

## Build order
1. Calibration builder + scoring → run on all three, inspect warped output visually.
2. Pipeline `--calibration` integration (keep game3 path + tests green).
3. Calibrate + run chessworld (easiest) → confirm reliability ≥55 + clean demo.
4. chess3 (undistortion-critical) → measure.
5. chess2 → measure; retrain only if needed.
6. Generate all three demos; final verification (game3=36, 25 tests green).
