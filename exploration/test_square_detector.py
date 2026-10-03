"""Verify SquareOccupancyDetector quality vs YOLO on chessworld/chess3."""
import sys, json
from pathlib import Path
import cv2, numpy as np, chess

sys.path.insert(0, 'D:/chessworldai_assignment/chessvision-pgn')
from src.board_detector import BoardDetector
from src.square_detector import SquareOccupancyDetector
from src.piece_detector import PieceDetector

def rotate(img, r):
    if r == 90: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if r == 180: return cv2.rotate(img, cv2.ROTATE_180)
    if r == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def get_warped(video, cfg, frac):
    cap = cv2.VideoCapture(video)
    n = 0
    while cap.grab(): n += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac))
    ret, frame = cap.read(); cap.release()
    if not ret: return None
    src = BoardDetector.undistort(frame, cfg["k1"], cfg.get("k2", 0.0))
    return rotate(cv2.warpPerspective(src,
        BoardDetector().get_homography(np.array(cfg["corners"], np.float32)),
        (640, 640)), cfg["rotation"])

def start_occ():
    b = chess.Board()
    occ = np.zeros((8, 8), bool)
    for sq in chess.SQUARES:
        if b.piece_at(sq):
            occ[7 - chess.square_rank(sq)][chess.square_file(sq)] = True
    return occ

for stem in ("chessworld", "chess3"):
    cfg = json.loads((Path("calibration") / f"{stem}.json").read_text())
    pd = PieceDetector("models/piece_detector.pt")
    sd = SquareOccupancyDetector()

    # collect start frames + compute board boundaries
    start_ws = [get_warped(f"videos/{stem}.mp4", cfg, f) for f in (0.01, 0.02, 0.03)]
    start_ws = [f for f in start_ws if f is not None]

    # get board boundaries from first start frame
    top, left, sq_h, sq_w = BoardDetector.detect_inner_board(start_ws[0])
    use_inner = top > 30 or left > 30 or sq_h < 65 or sq_w < 65
    if not use_inner:
        top, left, sq_h, sq_w = 0.0, 0.0, 80.0, 80.0

    ok = sd.calibrate(start_ws, top=top, left=left, sq_h=sq_h, sq_w=sq_w)
    socc = start_occ()

    print(f"\n{'='*66}")
    print(f"{stem}  calibrated={ok}  boundaries: top={top:.0f} left={left:.0f} sq_h={sq_h:.1f} sq_w={sq_w:.1f}")
    print(f"{'':22s} {'YOLO':>14}  {'SquareDet (fixed)':>18}")

    for tag, frac in (("start(32pcs)", 0.01), ("mid", 0.5), ("end", 0.97)):
        w = get_warped(f"videos/{stem}.mp4", cfg, frac)
        if w is None: continue

        yolo = pd.detect(w, top_offset=top if use_inner else 0,
                         left_offset=left if use_inner else 0,
                         sq_h=sq_h if use_inner else 0,
                         sq_w=sq_w if use_inner else 0)
        sq = sd.detect(w, top=top, left=left, sq_h=sq_h, sq_w=sq_w)

        yn = sum(1 for r in range(8) for c in range(8) if yolo[r][c])
        sn = sum(1 for r in range(8) for c in range(8) if sq[r][c])
        ym = sum(1 for r in range(8) for c in range(8) if (yolo[r][c] is not None)==socc[r][c])
        sm = sum(1 for r in range(8) for c in range(8) if (sq[r][c] is not None)==socc[r][c])
        print(f"  {tag:20s}  n={yn:2d} occ={ym:2d}    n={sn:2d} occ={sm:2d}")
