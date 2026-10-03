import sys
import json
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path('D:/chessworldai_assignment/chessvision-pgn')))

from src.board_detector import BoardDetector

def main():
    video_path = "D:/chessworldai_assignment/chessvision-pgn/videos/chess3.mp4"
    cal_path = "D:/chessworldai_assignment/chessvision-pgn/calibration/chess3.json"

    with open(cal_path) as f:
        cfg = json.load(f)

    corners = np.array(cfg["corners"], dtype=np.float32)
    k1 = cfg["k1"]
    rot = cfg["rotation"]
    print(f"Loaded config rotation: {rot}, k1: {k1}")

    cap = cv2.VideoCapture(video_path)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video specs: {w0}x{h0} @ {fps:.2f} fps, Total frames: {total_frames}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 90)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Could not read frame 90")
        return

    bd = BoardDetector()
    cal_maps = BoardDetector.undistort_maps(h0, w0, k1, 0.0)
    src = cv2.remap(frame, cal_maps[0], cal_maps[1], cv2.INTER_LINEAR)
    
    # Check if corners are within frame bounds
    print("Corners:")
    for i, pt in enumerate(corners):
        in_bounds = 0 <= pt[0] < w0 and 0 <= pt[1] < h0
        print(f"  Corner {i}: {pt} | In bounds: {in_bounds}")

    # Draw corners on undistorted src frame and save
    vis_src = src.copy()
    for pt in corners:
        cv2.circle(vis_src, (int(pt[0]), int(pt[1])), 15, (0, 0, 255), -1)
    cv2.imwrite("D:/chessworldai_assignment/chessvision-pgn/exploration/chess3_corners.jpg", vis_src)
    print("Saved chess3_corners.jpg showing corner points on src frame.")

    # Warp and save
    locked_H = bd.get_homography(corners)
    warped = cv2.warpPerspective(src, locked_H, (640, 640))
    cv2.imwrite("D:/chessworldai_assignment/chessvision-pgn/exploration/chess3_warped_0.jpg", warped)
    print("Saved chess3_warped_0.jpg.")

if __name__ == '__main__':
    main()
