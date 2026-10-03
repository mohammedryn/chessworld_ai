import sys
import json
import cv2
import numpy as np
import chess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path('D:/chessworldai_assignment/chessvision-pgn')))

from src.board_detector import BoardDetector
from src.piece_detector import PieceDetector, PIECE_CODES
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
    rot = cfg["rotation"]
    border_margin = cfg.get("border_margin", 0.0)

    bd = BoardDetector()
    pd = PieceDetector(model_path, confidence=0.25)
    sm = BoardStateMachine(min_frame_gap=60, move_threshold=32)
    sm.set_orientation(flipped=False)

    cap = cv2.VideoCapture(video_path)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cal_maps = BoardDetector.undistort_maps(h0, w0, k1, 0.0)
    locked_H = bd.get_homography(corners)

    frame_idx = 0
    starting_board = chess.Board()

    print(f"Starting diagnosis... rotation={rot}, k1={k1}")

    while frame_idx < 300:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % 3 != 0:
            continue

        # Undistort + Warp
        src = cv2.remap(frame, cal_maps[0], cal_maps[1], cv2.INTER_LINEAR)
        warped = cv2.warpPerspective(src, locked_H, (640, 640))
        warped = rotate(warped, rot)

        # Detect inner board boundaries if active
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

        num_pieces = sum(1 for r in range(8) for c in range(8) if board_state[r][c] is not None)
        
        # Vote and check match
        sm.update(board_state, frame_idx=frame_idx)
        committed = sm._committed
        
        voted = sm._vote() if len(sm._window) >= sm._window_size else None
        voted_score = sm._match_score(starting_board, voted) if voted is not None else 0
        voted_pieces = sum(1 for r in range(8) for c in range(8) if voted[r][c] is not None) if voted is not None else 0

        print(f"Frame {frame_idx:03d} | Raw Pieces: {num_pieces:2d} | Voted Pieces: {voted_pieces:2d} | Start Match Score: {voted_score:2d} | Committed: {committed is not None}")

    cap.release()

if __name__ == '__main__':
    main()
