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

def main():
    video_path = "D:/chessworldai_assignment/chessvision-pgn/videos/chess3.mp4"
    cal_path = "D:/chessworldai_assignment/chessvision-pgn/calibration/chess3.json"
    model_path = "D:/chessworldai_assignment/chessvision-pgn/models/piece_detector.pt"

    with open(cal_path) as f:
        cfg = json.load(f)

    corners = np.array(cfg["corners"], dtype=np.float32)
    k1 = cfg["k1"]
    bd = BoardDetector()
    
    # We want to scan frame 964 where players have played some moves
    # Let's see the board state after the first 3-4 moves
    # The actual moves in the game (looks like Queen's Gambit or similar)
    # Let's just score against the starting board first to see if we get a high score
    # (since only a few moves have been made, starting board match score should be around 55-60)
    
    pd = PieceDetector(model_path, confidence=0.10)

    cap = cv2.VideoCapture(video_path)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cal_maps = BoardDetector.undistort_maps(h0, w0, k1, 0.0)
    locked_H = bd.get_homography(corners)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 180)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Could not read frame 964")
        return

    src = cv2.remap(frame, cal_maps[0], cal_maps[1], cv2.INTER_LINEAR)
    warped = cv2.warpPerspective(src, locked_H, (640, 640))
    
    board_state = pd.detect(
        warped,
        border_margin=0.0,
    )

    num_pieces = sum(1 for r in range(8) for c in range(8) if board_state[r][c] is not None)
    print(f"Frame 964: Detected {num_pieces} pieces in rotation 0° warped board.")

    starting_board = chess.Board()

    mappings = {
        "White-Left_A-Bottom": lambda r, c: (c, 7 - r), 
        "White-Left_A-Top": lambda r, c: (c, r),
        "White-Right_A-Bottom": lambda r, c: (7 - c, 7 - r),
        "White-Right_A-Top": lambda r, c: (7 - c, r),
        "White-Top_A-Left": lambda r, c: (7 - r, c),
        "White-Top_A-Right": lambda r, c: (7 - r, 7 - c),
        "White-Bottom_A-Left": lambda r, c: (r, c),
        "White-Bottom_A-Right": lambda r, c: (r, 7 - c),
    }

    for name, map_func in mappings.items():
        mapped_board = np.full((8, 8), None, dtype=object)
        for r in range(8):
            for c in range(8):
                if board_state[r][c] is not None:
                    piece_code, conf = board_state[r][c]
                    rank_idx, file_idx = map_func(r, c)
                    if 0 <= rank_idx < 8 and 0 <= file_idx < 8:
                        mapped_board[7 - rank_idx][file_idx] = (piece_code, conf)
        
        score = 0
        piece_matches = 0
        for r in range(8):
            for c in range(8):
                sq = chess.square(c, 7 - r)
                piece = starting_board.piece_at(sq)
                expected = piece.symbol() if piece else None
                detected = mapped_board[r][c][0] if mapped_board[r][c] is not None else None
                if expected == detected:
                    score += 1
                    if expected is not None:
                        piece_matches += 1
                        
        print(f"Mapping: {name:<25s} | Occupancy Score: {score:2d}/64 | Piece Identity Matches: {piece_matches:2d}")

if __name__ == '__main__':
    main()
