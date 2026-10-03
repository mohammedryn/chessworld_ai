import sys
import json
import cv2
import numpy as np
import chess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path('D:/chessworldai_assignment/chessvision-pgn')))

from src.board_detector import BoardDetector
from src.piece_detector import PieceDetector
from src.state_machine import BoardStateMachine

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
    
    # Try rotation = 180 and conf = 0.15
    rot = 180
    conf = 0.15
    border_margin = cfg.get("border_margin", 0.0)

    bd = BoardDetector()
    pd = PieceDetector(model_path, confidence=conf)
    sm = BoardStateMachine(min_frame_gap=60, move_threshold=32)
    sm.set_orientation(flipped=False)

    cap = cv2.VideoCapture(video_path)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cal_maps = BoardDetector.undistort_maps(h0, w0, k1, 0.0)
    locked_H = bd.get_homography(corners)

    frame_idx = 0
    moves_detected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % 3 != 0:
            continue

        src = cv2.remap(frame, cal_maps[0], cal_maps[1], cv2.INTER_LINEAR)
        warped = cv2.warpPerspective(src, locked_H, (640, 640))
        warped = rotate(warped, rot)

        top_off, left_off, sq_h, sq_w = BoardDetector.detect_inner_board(warped)
        use_inner = top_off > 30 or left_off > 30 or sq_h < 65 or sq_w < 65
        
        board_state = pd.detect(
            warped,
            border_margin=border_margin,
            top_offset=top_off if use_inner else 0.0,
            left_offset=left_off if use_inner else 0.0,
            sq_h=sq_h if use_inner else 0.0,
            sq_w=sq_w if use_inner else 0.0,
        )

        move = sm.update(board_state, frame_idx=frame_idx)
        if move is not None:
            moves_detected += 1
            print(f"Move {moves_detected}: {move.uci()} at frame {frame_idx}")

    cap.release()
    print(f"Total moves detected: {moves_detected}")

if __name__ == '__main__':
    main()
