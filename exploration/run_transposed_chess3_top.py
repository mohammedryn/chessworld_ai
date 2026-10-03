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
    
    sm = BoardStateMachine(min_frame_gap=60, move_threshold=32)
    sm.set_orientation(flipped=False)

    cap = cv2.VideoCapture(video_path)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cal_maps = BoardDetector.undistort_maps(h0, w0, k1, 0.0)
    locked_H = bd.get_homography(corners)

    frame_idx = 0
    moves_detected = 0

    print("Running transposed pipeline (A-Top) on chess3.mp4...", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % 3 != 0:
            continue

        src = cv2.remap(frame, cal_maps[0], cal_maps[1], cv2.INTER_LINEAR)
        warped = cv2.warpPerspective(src, locked_H, (640, 640))
        
        top_off, left_off, sq_h, sq_w = BoardDetector.detect_inner_board(warped)
        use_inner = top_off > 30 or left_off > 30 or sq_h < 65 or sq_w < 65
        
        board_state = pd.detect(
            warped,
            border_margin=0.0,
            top_offset=top_off if use_inner else 0.0,
            left_offset=left_off if use_inner else 0.0,
            sq_h=sq_h if use_inner else 0.0,
            sq_w=sq_w if use_inner else 0.0,
        )

        transposed_board = np.full((8, 8), None, dtype=object)
        for r in range(8):
            for c in range(8):
                if board_state[r][c] is not None:
                    # White-Right_A-Top: rank_idx = 7 - c, file_idx = r
                    rank_idx = 7 - c
                    file_idx = r
                    transposed_board[7 - rank_idx][file_idx] = board_state[r][c]

        move = sm.update(transposed_board, frame_idx=frame_idx)
        if move is not None:
            moves_detected += 1
            print(f"Move {moves_detected}: {move.uci()} at frame {frame_idx}", flush=True)

    cap.release()
    print(f"Total moves detected: {moves_detected}", flush=True)

if __name__ == '__main__':
    main()
