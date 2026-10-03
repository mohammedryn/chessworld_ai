"""Inspect what the CALIBRATED pipeline sees: warp + YOLO + grid + match-to-start."""
import sys, json
from pathlib import Path
import cv2, numpy as np, chess

sys.path.insert(0, 'D:/chessworldai_assignment/chessvision-pgn')
from src.board_detector import BoardDetector
from src.piece_detector import PieceDetector, PIECE_CODES

DBG = Path('D:/chessworldai_assignment/chessvision-pgn/exploration/_calchk')
DBG.mkdir(parents=True, exist_ok=True)


def rotate(img, r):
    if r == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if r == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if r == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def start_occupancy():
    b = chess.Board()
    occ = np.zeros((8, 8), bool)
    for sq in chess.SQUARES:
        if b.piece_at(sq):
            r, c = 7 - chess.square_rank(sq), chess.square_file(sq)
            occ[r][c] = True
    return occ


def main():
    video = sys.argv[1]
    stem = Path(video).stem
    cfg = json.loads((Path('calibration') / f"{stem}.json").read_text())
    corners = np.array(cfg["corners"], np.float32)
    H = BoardDetector().get_homography(corners)
    pd = PieceDetector("models/piece_detector.pt")
    start_occ = start_occupancy()

    cap = cv2.VideoCapture(video)
    n = 0
    while cap.grab():
        n += 1
    for tag, frac in (("start", 0.01), ("mid", 0.5), ("end", 0.97)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac))
        ret, frame = cap.read()
        if not ret:
            continue
        src = BoardDetector.undistort(frame, cfg["k1"], cfg.get("k2", 0.0))
        warped = rotate(cv2.warpPerspective(src, H, (640, 640)), cfg["rotation"])
        board = pd.detect(warped)
        npieces = sum(1 for r in range(8) for c in range(8) if board[r][c] is not None)
        match = sum(1 for r in range(8) for c in range(8)
                    if (board[r][c] is not None) == start_occ[r][c])
        # overlay grid + detections
        vis = warped.copy()
        for i in range(9):
            cv2.line(vis, (i * 80, 0), (i * 80, 640), (0, 0, 255), 1)
            cv2.line(vis, (0, i * 80), (640, i * 80), (0, 0, 255), 1)
        res = pd.model(warped, conf=0.25, verbose=False)[0]
        for box in res.boxes:
            code = PIECE_CODES.get(res.names[int(box.cls[0])])
            if not code:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            cv2.putText(vis, code, (int((x1 + x2) / 2) - 6, int((y1 + y2) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.imwrite(str(DBG / f"{stem}_{tag}.jpg"), vis)
        print(f"  {tag}: pieces={npieces}, occupancy_match_vs_START={match}/64")
    cap.release()


if __name__ == "__main__":
    main()
