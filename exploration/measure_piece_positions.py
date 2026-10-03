"""Find where pieces actually appear in the calibrated warp.

Runs on the first 5 seconds (before any moves) of a ChessWorld video.
Uses the current model to detect pieces and records their pixel centroids.
Even with degraded model recall, the pieces it DOES detect are in correct positions.
Also uses color analysis (dark blobs = black pieces, bright blobs ≠ bg = white pieces)
to find pieces the model misses.

This gives ground-truth pixel positions for what should be grid rows/cols.
"""
import sys, json
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, 'D:/chessworldai_assignment/chessvision-pgn')
from src.board_detector import BoardDetector
from src.piece_detector import PieceDetector, PIECE_CODES

def rotate(img, r):
    if r==90: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if r==180: return cv2.rotate(img, cv2.ROTATE_180)
    if r==270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def get_opening_warps(video, cfg, n=8):
    cap = cv2.VideoCapture(video)
    total = 0
    while cap.grab(): total += 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    end = min(int(fps * 5), total)  # first 5 seconds only
    H = BoardDetector().get_homography(np.array(cfg["corners"], np.float32))
    maps = BoardDetector.undistort_maps(
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        cfg["k1"], cfg.get("k2",0)) if cfg["k1"] else None
    warps = []
    for i in range(0, end, max(1, end//n)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        if not ret: continue
        src = cv2.remap(f, maps[0], maps[1], cv2.INTER_LINEAR) if maps else f
        warps.append(rotate(cv2.warpPerspective(src, H, (640,640)), cfg["rotation"]))
    cap.release()
    return warps

for stem in ("chessworld",):
    cfg = json.loads((Path("calibration")/f"{stem}.json").read_text())
    pd = PieceDetector("models/piece_detector.pt")
    warps = get_opening_warps(f"videos/{stem}.mp4", cfg)
    print(f"\n{stem}: {len(warps)} opening frames")

    all_cy = []
    for warped in warps:
        res = pd.model(warped, conf=0.10, verbose=False)[0]
        for box in res.boxes:
            if not PIECE_CODES.get(res.names[int(box.cls[0])]): continue
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            cy = (y1+y2)/2
            all_cy.append(cy)

    all_cy.sort()
    print(f"  YOLO cy distribution (all detections conf>0.10):")
    # bin into groups
    hist = np.histogram(all_cy, bins=16, range=(0,640))
    for count, edge in zip(hist[0], hist[1]):
        bar = "█"*int(count/2)
        print(f"  y={edge:5.0f}-{edge+40:5.0f}: {bar} ({count})")

    # Also: dark blob analysis — find dense dark regions (black pieces)
    print(f"\n  Dark blob row profile (where black pieces likely are):")
    dark_counts = []
    for warped in warps:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        dark = (gray < 80).astype(float)
        dark_counts.append(dark.mean(axis=1))
    profile = np.mean(dark_counts, axis=0)
    peaks = []
    for y in range(1, 639):
        if profile[y] > profile[y-1] and profile[y] > profile[y+1] and profile[y] > 0.05:
            peaks.append((y, profile[y]))
    peaks.sort(key=lambda x: -x[1])
    print(f"  Top dark-pixel rows (black piece locations): {peaks[:8]}")
