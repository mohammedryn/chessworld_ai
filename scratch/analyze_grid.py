import cv2
import numpy as np
import sys
import chess
from pathlib import Path

sys.path.append(str(Path("D:/chessworldai_assignment/chessvision-pgn").resolve()))

from src.board_detector import BoardDetector
from src.piece_detector import PieceDetector, PIECE_CODES

def analyze(game_idx=1):
    video_path = f"D:/chessworldai_assignment/chessvision-pgn/videos/game{game_idx}.mp4"
    if not Path(video_path).exists():
        video_path = f"D:/chessworldai_assignment/videos/game{game_idx}.mp4"
        
    print(f"Analyzing game{game_idx}.mp4...")
    
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
            
            # Save raw warped
            cv2.imwrite(f"output/debug_game{game_idx}_warped_raw.jpg", warped)
            print("Saved raw warped frame.")
            
            # Let's run YOLO on each of the 4 rotations and print how many pieces are detected
            # and what the max match score is for different rotations and margins.
            rotations = [0, 90, 180, 270]
            margins = [0.0, 10.0, 15.0, 20.0, 25.0, 30.0]
            
            starting_board = chess.Board()
            
            for rot in rotations:
                warped_rot = warped.copy()
                if rot == 90:
                    warped_rot = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                elif rot == 180:
                    warped_rot = cv2.rotate(warped, cv2.ROTATE_180)
                elif rot == 270:
                    warped_rot = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                # Run YOLO
                results = piece_detector.model(warped_rot, conf=0.4, verbose=False)[0]
                detections = []
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_name = results.names[int(box.cls[0])]
                    piece_code = PIECE_CODES.get(cls_name)
                    if piece_code is not None:
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        detections.append((piece_code, cx, cy, conf, x1, y1, x2, y2))
                        
                print(f"  Rotation {rot}°: YOLO detected {len(detections)} pieces.")
                
                # Check different margins
                for margin in margins:
                    board_state = np.full((8, 8), None, dtype=object)
                    for piece_code, cx, cy, conf, _, _, _, _ in detections:
                        row, col = PieceDetector.center_to_square(cx, cy, border_margin=margin)
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
                                
                    if len(detections) > 0:
                        print(f"    Margin {margin}px: score={matches}/64")
                        
            break
            
    cap.release()

if __name__ == "__main__":
    analyze(1)
