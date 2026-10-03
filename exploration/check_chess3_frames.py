import sys
import json
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path('D:/chessworldai_assignment/chessvision-pgn')))

from src.board_detector import BoardDetector
from src.piece_detector import PieceDetector

def main():
    video_path = "D:/chessworldai_assignment/chessvision-pgn/videos/chess3.mp4"
    cal_path = "D:/chessworldai_assignment/chessvision-pgn/calibration/chess3.json"
    model_path = "D:/chessworldai_assignment/chessvision-pgn/models/piece_detector.pt"

    with open(cal_path) as f:
        cfg = json.load(f)

    corners = np.array(cfg["corners"], dtype=np.float32)
    k1 = cfg["k1"]
    bd = BoardDetector()
    pd = PieceDetector(model_path, confidence=0.15)

    cap = cv2.VideoCapture(video_path)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cal_maps = BoardDetector.undistort_maps(h0, w0, k1, 0.0)
    locked_H = bd.get_homography(corners)

    print("Scanning first 600 frames for a clean board...")
    for fi in range(0, 600, 30):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break
        src = cv2.remap(frame, cal_maps[0], cal_maps[1], cv2.INTER_LINEAR)
        warped = cv2.warpPerspective(src, locked_H, (640, 640))
        results = pd.model(warped, conf=0.15, verbose=False)[0]
        detected = len(results.boxes)
        print(f"Frame {fi:03d} | Detected Pieces: {detected}")
    cap.release()

if __name__ == '__main__':
    main()
