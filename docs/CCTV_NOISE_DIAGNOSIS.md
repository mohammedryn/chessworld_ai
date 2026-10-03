# CCTV False-Move Noise — Root-Cause Diagnosis

**Question:** When the pipeline was run live on ChessWorld AI's Karnataka Championship
CCTV footage (`chess3.mp4`), it emitted a flood of false moves — "pieces flickering
between squares get counted as moves." Why, and what actually fixes it?

**Answer (one line):** The CCTV warp lands pieces on the *wrong squares*, the detected
board never cleanly matches a legal position, and the state machine launders that jitter
into legal-but-fake moves. No state-machine threshold can fix it — the fix is a model
retrained on this camera's own frames. The data below proves both halves of that claim.

---

## How the evidence was gathered

`exploration/diagnose_moves.py` runs the **real** pipeline (board detect → warp → YOLO →
state machine) but injects an instrumented state machine that records, for every emitted
move: source/dest squares, whether they sit on the board edge, whether it is an
adjacent-square jitter, the occupancy **match score** the move achieved (out of 64), and
how many squares changed since the last committed board. No production code is modified.

Run per video:
```
python exploration/diagnose_moves.py --video videos/<name>.mp4
```

---

## Finding 1 — chess3 is pure oscillation, not chess

chess3 emits **75 moves** for a single game. The per-move trace shows the same piece
flickering between adjacent squares:

- Black king: `f7 → g8 → f7 → g8 …` repeats **8+ times**
- Queen `g5 ↔ h5`, bishop `g4 ↔ h5 ↔ f5` oscillate the same way
- ~77% of moves touch the board's outer ring
- **10–26 squares change between consecutive "moves"** — a real move changes ~2

A move that changes 15 squares is not a move; it is the detector reshuffling the whole
board, and the state machine finding *some* legal move that fits.

## Finding 2 — the score wall: real moves and noise live in separate bands

The board-match **score** (how many of 64 squares agree with the nearest legal position)
cleanly separates clean footage from noisy footage:

| Video  | Camera angle        | Moves emitted | Match score (min–max) | Reality            |
|--------|---------------------|---------------|-----------------------|--------------------|
| game3  | ~50° (clean)        | 36            | **59 – 64**           | coherent real game |
| chess3 | ~55° CCTV (curved)  | 75            | 35 – 49               | noise              |
| game1  | ~30° oblique        | 71            | 32 – 49               | noise              |
| game4  | ~35° oblique        | 83            | 32 – 50               | noise              |
| game5  | ~70° overhead       | 12            | 33 – 43               | noise              |

There is a **hard gap between 50 and 59 that nothing crosses.** game3's real moves make
the board match a legal position *almost perfectly* (≥59/64). Every noisy video — chess3
included — is capped at ~50, because ~14 of 64 squares are always misplaced by the warp.

## Finding 3 — why threshold tuning cannot work (this is the key result)

The obvious "fix" is to raise the acceptance threshold (`move_threshold`, default 32) to
~55 so noise is rejected. The table shows why that fails: **the noisy videos top out at
~50, so a 55 cut deletes every move they produce — including any real ones.** On footage
this noisy, a *real* move scores in the same band as jitter, so no score threshold can
tell them apart. Raising the bar trades false positives for total recall loss.

Dead ends also ruled out with evidence:
- **Edge / adjacency filters:** game3's real h-pawn pushes, `a6`, and castling are also
  edge/adjacent (56% / 47%), so these filters would kill real moves.
- **Anti-reversal:** game3 contains a legitimate immediate reversal (knight `f6→e4→f6`,
  moves 6 & 8), so suppressing reversals breaks real games.
- **Temporal debounce:** contradicts `test_majority_vote_ignores_noisy_frame`, which
  requires a move to be found despite a noisy frame within 3 updates.

**Conclusion:** the noise is a *detection-quality* problem (pieces on wrong squares),
not a state-machine problem. The only real fix is to raise detection quality on this
camera → retrain on its own frames. This is what the score wall proves quantitatively.

---

## What was shipped now (safe, no move-logic change)

A **reliability gate** (`src/pipeline.py`, `RELIABILITY_WARN = 55`). The state machine
now tracks the match score of each accepted move and exposes the median as
`BoardStateMachine.reliability`. After processing, the pipeline warns when the median is
below 55:

```
WARNING: Low board-match reliability (42/64; clean footage scores >=55). Detections are
landing on the wrong squares, so many of the N moves are likely noise. This camera angle
needs a model retrained on its own frames ...
```

This makes the system **honest about its own output** — it flags chess3/game1/4/5 as
unreliable instead of silently emitting a confident-but-wrong PGN, while staying silent
on game3 (median 64). It changes **no** move decisions: game3 still produces 36 moves and
all 25 unit tests stay green.

## The actual fix (next step — needs manual labeling)

1. `python scripts/extract_cctv_frames.py --video videos/chess3.mp4 --count 120`  ✅ done
   → 120 warped 640×640 frames in `training_frames/cctv/` (the exact images YOLO sees).
2. Label them in Roboflow (12-class schema). *Manual; cannot be auto-labeled because
   chess3 has no known-correct PGN — unlike game3, which was auto-labeled from its PGN.*
3. Export YOLOv8, merge into `models/combined_dataset/`, run `scripts/train_combined.py`.
4. Re-run chess3; the reliability score should climb past 55 as pieces land on the right
   squares, at which point the existing state machine will track real moves cleanly.

The reliability gate is the measurable success metric for that retrain: **watch the
median move-score cross from ~42 toward ~60.**
