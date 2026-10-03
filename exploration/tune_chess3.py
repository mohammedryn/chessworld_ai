import sys
import json
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path('D:/chessworldai_assignment/chessvision-pgn')))

from src.board_detector import BoardDetector
from src.piece_detector import PieceDetector

def rotate(img, r):
    if r == 90: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if r == 180: return cv2.rotate(img, cv2.ROTATE_180)
    if r == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def main():
    video_path = "D:/chessworldai_assignment/chessvision-pgn/videos/chess3.mp4"
    cal_path = "D:/chessworldai_assignment/chessvision-pgn/calibration/chess3.json"
    model_path = "D:/chessworldai_assignment/chessvision-pgn/models/piece_detector.pt"

    with open(cal_path) as f:
        cfg = json.load(f)

    corners = np.array(cfg["corners"], dtype=np.float32)
    k1 = cfg["k1"]
    bd = BoardDetector()

    cap = cv2.VideoCapture(video_path)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cal_maps = BoardDetector.undistort_maps(h0, w0, k1, 0.0)
    locked_H = bd.get_homography(corners)

    # Grab frame at 964
    cap.set(cv2.CAP_PROP_POS_FRAMES, 964)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Could not read frame 964")
        return

    src = cv2.remap(frame, cal_maps[0], cal_maps[1], cv2.INTER_LINEAR)
    warped_base = cv2.warpPerspective(src, locked_H, (640, 640))

    print("Evaluating parameters at frame 964...")
    for conf in [0.10, 0.15, 0.20, 0.25]:
        pd = PieceDetector(model_path, confidence=conf)
        for rot in [0, 90, 180, 270]:
            warped = rotate(warped_base, rot)
            results = pd.model(warped, conf=conf, verbose=False)[0]
            detected_pieces = len(results.boxes)
            print(f"Conf: {conf:.2f} | Rotation: {rot:3d}° | Detected Pieces: {detected_pieces:2d}")

if __name__ == '__main__':
    main()
