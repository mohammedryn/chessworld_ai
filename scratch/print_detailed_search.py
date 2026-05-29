import cv2
import numpy as np
import sys
import chess
from pathlib import Path

sys.path.append(str(Path("D:/chessworldai_assignment/chessvision-pgn").resolve()))

from src.board_detector import BoardDetector
from src.piece_detector import PieceDetector, PIECE_CODES

def search_game1():
    video_path = "D:/chessworldai_assignment/chessvision-pgn/videos/game1.mp4"
    if not Path(video_path).exists():
        video_path = "D:/chessworldai_assignment/videos/game1.mp4"
        
    cap = cv2.VideoCapture(video_path)
    board_detector = BoardDetector()
    piece_detector = PieceDetector("D:/chessworldai_assignment/chessvision-pgn/models/piece_detector.pt")
    
    warped_frames = []
    frame_idx = 0
    while len(warped_frames) < 3 and frame_idx < 300:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 3 != 0:
            continue
        corners = board_detector.detect(frame)
        if corners is not None:
            H = board_detector.get_homography(corners)
            warped = board_detector.warp(frame, H)
            warped_frames.append(warped)
    cap.release()
    
    starting_board = chess.Board()
    
    # We will test all rotations and print the highest match scores
    rotations = [0, 90, 180, 270]
    margins = [0.0, 10.0, 20.0]
    alphas = [0.1, 0.3, 0.5]
    
    for rot in rotations:
        for margin in margins:
            for alpha in alphas:
                total_matches = 0
                for warped in warped_frames:
                    warped_rot = warped.copy()
                    if rot == 90:
                        warped_rot = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                    elif rot == 180:
                        warped_rot = cv2.rotate(warped, cv2.ROTATE_180)
                    elif rot == 270:
                        warped_rot = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        
                    results = piece_detector.model(warped_rot, conf=0.25, verbose=False)[0]
                    board_state = np.full((8, 8), None, dtype=object)
                    for box in results.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_name = results.names[int(box.cls[0])]
                        piece_code = PIECE_CODES.get(cls_name)
                        if piece_code is not None:
                            cx = (x1 + x2) / 2
                            cy_mapped = y1 * alpha + y2 * (1.0 - alpha)
                            row, col = PieceDetector.center_to_square(cx, cy_mapped, border_margin=margin)
                            if board_state[row][col] is None or conf > board_state[row][col][1]:
                                board_state[row][col] = (piece_code, conf)
                                
                    matches = 0
                    for r in range(8):
                        for c in range(8):
                            sq = chess.square(c, 7 - r)
                            piece = starting_board.piece_at(sq)
                            expected = piece.symbol() if piece else None
                            detected = board_state[r][c][0] if board_state[r][c] is not None else None
                            if expected == detected:
                                matches += 1
                    total_matches += matches
                avg_matches = total_matches / len(warped_frames)
                print(f"Rot {rot}°, margin {margin}px, alpha {alpha}: avg_matches = {avg_matches:.1f}/64")

if __name__ == "__main__":
    search_game1()
