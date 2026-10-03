"""Does lowering YOLO confidence recover chess3 recall, or is it a model-capability gap?"""
import sys, json
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, 'D:/chessworldai_assignment/chessvision-pgn')
from src.board_detector import BoardDetector
from src.piece_detector import PieceDetector, PIECE_CODES

def rotate(img, r):
    return {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(r) and cv2.rotate(img, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[r]) or img

video = sys.argv[1]; stem = Path(video).stem
cfg = json.loads((Path('calibration') / f"{stem}.json").read_text())
H = BoardDetector().get_homography(np.array(cfg["corners"], np.float32))
model = PieceDetector("models/piece_detector.pt").model
cap = cv2.VideoCapture(video); n = 0
while cap.grab(): n += 1
for tag, frac in (("start", 0.01), ("mid", 0.5), ("end", 0.97)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac)); ret, frame = cap.read()
    if not ret: continue
    src = BoardDetector.undistort(frame, cfg["k1"], cfg.get("k2", 0.0))
    warped = rotate(cv2.warpPerspective(src, H, (640, 640)), cfg["rotation"])
    counts = []
    for conf in (0.25, 0.15, 0.10, 0.05):
        res = model(warped, conf=conf, verbose=False)[0]
        c = sum(1 for b in res.boxes if PIECE_CODES.get(res.names[int(b.cls[0])]))
        counts.append(f"conf{conf}={c}")
    print(f"  {tag}: " + "  ".join(counts))
cap.release()
