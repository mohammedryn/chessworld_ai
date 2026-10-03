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
    
    # Use conf = 0.20 to catch more pieces but keep precision high
    pd = PieceDetector(model_path, confidence=0.20)

    cap = cv2.VideoCapture(video_path)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cal_maps = BoardDetector.undistort_maps(h0, w0, k1, 0.0)
    locked_H = bd.get_homography(corners)

    # Grab frame at 90 (starting position)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 90)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Could not read frame 90")
        return

    src = cv2.remap(frame, cal_maps[0], cal_maps[1], cv2.INTER_LINEAR)
    warped = cv2.warpPerspective(src, locked_H, (640, 640))
    # rotation = 0 to keep pieces standing upright
    
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

    num_pieces = sum(1 for r in range(8) for c in range(8) if board_state[r][c] is not None)
    print(f"Detected {num_pieces} pieces in rotation 0° warped board.")

    starting_board = chess.Board()

    # Try different transposition mappings to align with starting board
    # Mappings are named after how the ranks and files of the warped board relate to chess board
    # In warped board: row is y (0 at top, 7 at bottom), col is x (0 at left, 7 at right)
    
    mappings = {
        # 1. White is on the left (col 0=rank 1, col 7=rank 8), A-file is at the bottom (row 7=file A, row 0=file H)
        "White-Left_A-Bottom": lambda r, c: (c, 7 - r), 
        # 2. White is on the left (col 0=rank 1, col 7=rank 8), A-file is at the top (row 0=file A, row 7=file H)
        "White-Left_A-Top": lambda r, c: (c, r),
        # 3. White is on the right (col 7=rank 1, col 0=rank 8), A-file is at the bottom (row 7=file A, row 0=file H)
        "White-Right_A-Bottom": lambda r, c: (7 - c, 7 - r),
        # 4. White is on the right (col 7=rank 1, col 0=rank 8), A-file is at the top (row 0=file A, row 7=file H)
        "White-Right_A-Top": lambda r, c: (7 - c, r),
        # 5. White is at the top (row 0=rank 1, row 7=rank 8), A-file is at the left (col 0=file A, col 7=file H)
        "White-Top_A-Left": lambda r, c: (7 - r, c),
        # 6. White is at the top (row 0=rank 1, row 7=rank 8), A-file is at the right (col 7=file A, col 0=file H)
        "White-Top_A-Right": lambda r, c: (7 - r, 7 - c),
        # 7. White is at the bottom (row 7=rank 1, row 0=rank 8), A-file is at the left (col 0=file A, col 7=file H)
        "White-Bottom_A-Left": lambda r, c: (r, c),
        # 8. White is at the bottom (row 7=rank 1, row 0=rank 8), A-file is at the right (col 7=file A, col 0=file H)
        "White-Bottom_A-Right": lambda r, c: (r, 7 - c),
    }

    for name, map_func in mappings.items():
        mapped_board = np.full((8, 8), None, dtype=object)
        for r in range(8):
            for c in range(8):
                if board_state[r][c] is not None:
                    piece_code, conf = board_state[r][c]
                    # Get chess rank and file (0-7)
                    rank_idx, file_idx = map_func(r, c)
                    if 0 <= rank_idx < 8 and 0 <= file_idx < 8:
                        # Row in 8x8 array is 7 - rank
                        mapped_board[7 - rank_idx][file_idx] = (piece_code, conf)
        
        # Calculate match score against starting position
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
