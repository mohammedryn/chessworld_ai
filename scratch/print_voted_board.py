import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path("D:/chessworldai_assignment/chessvision-pgn").resolve()))

from src.board_detector import BoardDetector
from src.piece_detector import PieceDetector, PIECE_CODES

def print_board(game_idx=1, rotation=270, margin=0.0, alpha=0.4, conf=0.25):
    video_path = f"D:/chessworldai_assignment/chessvision-pgn/videos/game{game_idx}.mp4"
    if not Path(video_path).exists():
        video_path = f"D:/chessworldai_assignment/videos/game{game_idx}.mp4"
        
    print(f"\n=================== game{game_idx}.mp4 (rot={rotation}°, margin={margin}px, alpha={alpha}, conf={conf}) ===================")
    
    cap = cv2.VideoCapture(video_path)
    board_detector = BoardDetector()
    piece_detector = PieceDetector("D:/chessworldai_assignment/chessvision-pgn/models/piece_detector.pt")
    
    frame_idx = 0
    while frame_idx < 300:
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
            
            if rotation == 90:
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                warped = cv2.rotate(warped, cv2.ROTATE_180)
            elif rotation == 270:
                warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
            results = piece_detector.model(warped, conf=conf, verbose=False)[0]
            board_state = np.full((8, 8), ".", dtype=object)
            
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                b_conf = float(box.conf[0])
                cls_name = results.names[int(box.cls[0])]
                piece_code = PIECE_CODES.get(cls_name, "?")
                cx = (x1 + x2) / 2
                cy_mapped = y1 * alpha + y2 * (1.0 - alpha)
                row, col = PieceDetector.center_to_square(cx, cy_mapped, border_margin=margin)
                board_state[row][col] = piece_code
                
            # Print the 8x8 board
            for r in range(8):
                row_str = " ".join(board_state[r])
                print(f"Row {r} (Rank {8-r}): {row_str}")
            break
            
    cap.release()

if __name__ == "__main__":
    print_board(1, 270, 0.0, 0.4, 0.25)
    print_board(2, 270, 15.0, 0.5, 0.2)
    print_board(4, 270, 25.0, 0.5, 0.3)
    print_board(5, 270, 0.0, 0.2, 0.25)
