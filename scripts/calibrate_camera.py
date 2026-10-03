"""Fixed-camera calibration for ChessWorld AI streams.

Computes, once per static camera: radial undistortion k1 (flattens IP-cam lens
curvature), the locked 4 board corners (correct, stable warp), rotation, and
border margin. Writes calibration/<stem>.json consumed by the pipeline.

Method: grid-search k1; for each, undistort -> detect green-board corners ->
warp to 640 -> score grid-line alignment (well-flattened boards put edge energy
on the 80px grid). Visual debug warps are saved for confirmation.

    python scripts/calibrate_camera.py --video videos/chess3.mp4
    python scripts/calibrate_camera.py --video videos/chess3.mp4 --k1-sweep " -0.6,-0.3,0,0.3"
"""
import sys, json, argparse
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, 'D:/chessworldai_assignment/chessvision-pgn')
from src.board_detector import BoardDetector
from src.piece_detector import PieceDetector, PIECE_CODES

BASE = Path('D:/chessworldai_assignment/chessvision-pgn')
DBG = BASE / 'exploration' / '_calib'
DBG.mkdir(parents=True, exist_ok=True)


def undistort(frame, k1, k2=0.0):
    h, w = frame.shape[:2]
    f = float(max(w, h))
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
    return cv2.undistort(frame, K, dist)


def green_corners(frame):
    """4 board corners (TL,TR,BR,BL) from the green checkerboard, or None.

    Closes the green squares into a solid board blob (downscaled for speed),
    takes the largest contour's convex hull, approximates to 4 corners.
    """
    h, w = frame.shape[:2]
    scale = 768.0 / w
    small = cv2.resize(frame, (768, int(h * scale)))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36, 35, 35), (90, 255, 255))
    # merge the green squares (gaps ~1 square wide at this scale) into a board blob
    k = np.ones((45, 45), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.05 * small.shape[0] * small.shape[1]:
        return None
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.02, 0.03, 0.05, 0.08, 0.12):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
    quad /= scale  # back to full-res coords
    return BoardDetector()._sort_corners(quad)


def grid_score(warped):
    """Edge energy concentrated on the 80px grid (higher = flatter/aligned)."""
    g = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)).sum(axis=0)
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1)).sum(axis=1)
    on = 0.0
    for i in range(9):
        x = min(i * 80, 639)
        on += gx[max(0, x - 2):x + 3].sum() + gy[max(0, x - 2):x + 3].sum()
    return float(on / (gx.sum() + gy.sum() + 1e-6))


def pick_rotation(warped, model):
    """Rotation (0/90/180/270) that puts white pieces at the bottom (flipped=False)."""
    best_rot, best = 0, -1e9
    for rot in (0, 90, 180, 270):
        w = warped
        if rot == 90:
            w = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180:
            w = cv2.rotate(warped, cv2.ROTATE_180)
        elif rot == 270:
            w = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        res = model(w, conf=0.25, verbose=False)[0]
        top = bot = 0
        for box in res.boxes:
            code = PIECE_CODES.get(res.names[int(box.cls[0])])
            if not code:
                continue
            _, y1, _, y2 = box.xyxy[0].tolist()
            cy = (y1 + y2) / 2
            if code.isupper():  # white
                bot += 1 if cy > 320 else 0
                top += 1 if cy <= 320 else 0
        s = bot - top
        if s > best:
            best, best_rot = s, rot
    return best_rot


def warp_from(frame, corners, size=640):
    bd = BoardDetector()
    H = bd.get_homography(corners.astype(np.float32))
    return cv2.warpPerspective(frame, H, (size, size))


def sample_frames(vp, n=6):
    cap = cv2.VideoCapture(vp)
    total = 0
    while cap.grab():
        total += 1
    frames = []
    for frac in np.linspace(0.05, 0.6, n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
        ret, f = cap.read()
        if ret:
            frames.append(f)
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--k1-sweep", default="-0.6,-0.45,-0.3,-0.15,0.0,0.15,0.3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stem = Path(args.video).stem
    k1s = [float(x) for x in args.k1_sweep.split(",")]
    frames = sample_frames(args.video)
    if not frames:
        print("no frames")
        return
    model = PieceDetector("models/piece_detector.pt").model

    best = None  # (score, k1, corners, warped)
    for k1 in k1s:
        scores, last = [], None
        for fr in frames:
            ud = undistort(fr, k1)
            corners = green_corners(ud)
            if corners is None:
                continue
            warped = warp_from(ud, corners)
            scores.append(grid_score(warped))
            last = (corners, warped, ud)
        if not scores:
            print(f"  k1={k1:+.2f}: no board")
            continue
        avg = float(np.mean(scores))
        print(f"  k1={k1:+.2f}: grid_score={avg:.4f} ({len(scores)} frames)")
        cv2.imwrite(str(DBG / f"{stem}_k1{k1:+.2f}.jpg"), last[1])
        if best is None or avg > best[0]:
            best = (avg, k1, last[0], last[1], last[2])

    if best is None:
        print("calibration FAILED — no board detected at any k1")
        return
    avg, k1, corners, warped, ud = best
    rotation = pick_rotation(warped, model)
    print(f"\nBEST: k1={k1:+.2f} grid_score={avg:.4f} rotation={rotation}")

    out = Path(args.out) if args.out else BASE / "calibration" / f"{stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "video": stem, "k1": k1, "k2": 0.0,
        "corners": corners.tolist(), "rotation": rotation, "border_margin": 0.0,
    }
    out.write_text(json.dumps(cfg, indent=2))
    print(f"wrote {out}")
    # final debug: rotated warp
    w = warped
    if rotation == 90:
        w = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        w = cv2.rotate(warped, cv2.ROTATE_180)
    elif rotation == 270:
        w = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(str(DBG / f"{stem}_FINAL.jpg"), w)


if __name__ == "__main__":
    main()
