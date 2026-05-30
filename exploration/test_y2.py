import cv2
import numpy as np
import sys
import chess
from pathlib import Path

sys.path.append(str(Path("D:/chessworldai_assignment/chessvision-pgn").resolve()))

from src.board_detector import BoardDetector
from src.piece_detector import PieceDetector, PIECE_CODES

def test_y2():
    video_paths = {
        1: "D:/chessworldai_assignment/chessvision-pgn/videos/game1.mp4",
        2: "D:/chessworldai_assignment/chessvision-pgn/videos/game2.mp4",
        3: "D:/chessworldai_assignment/chessvision-pgn/videos/game3.mp4",
        4: "D:/chessworldai_assignment/chessvision-pgn/videos/game4.mp4",
        5: "D:/chessworldai_assignment/chessvision-pgn/videos/game5.mp4"
    }
    
    piece_detector = PieceDetector("D:/chessworldai_assignment/chessvision-pgn/models/piece_detector.pt")
    board_detector = BoardDetector()
    
    # We will test game1, game2, game4, game5
    games_to_test = [1, 2, 4, 5]
    
    for game_idx in games_to_test:
        video_path = video_paths[game_idx]
        if not Path(video_path).exists():
            video_path = f"D:/chessworldai_assignment/videos/game{game_idx}.mp4"
        if not Path(video_path).exists():
            continue
            
        print(f"\nScanning game{game_idx}.mp4...")
        
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        warped = None
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
                break
        cap.release()
        
        if warped is None:
            print("  Could not find board")
            continue
            
        # Determine best rotation and margin first using standard center
        # Let's run a joint grid search over:
        # - confidence (0.2, 0.25, 0.3, 0.4)
        # - rotation (0, 90, 180, 270)
        # - border_margin (0 to 30)
        # - alpha (0.0 to 0.5 where y_mapped = y1 * alpha + y2 * (1 - alpha))
        
        confs = [0.2, 0.25, 0.3, 0.4]
        rotations = [0, 90, 180, 270]
        margins = [0.0, 10.0, 15.0, 20.0, 25.0]
        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        starting_board = chess.Board()
        
        best_conf = 0.4
        best_rot = 0
        best_margin = 0.0
        best_alpha = 0.5
        best_score = -1
        best_pieces = 0
        
        for conf in confs:
            for rot in rotations:
                warped_rot = warped.copy()
                if rot == 90:
                    warped_rot = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                elif rot == 180:
                    warped_rot = cv2.rotate(warped, cv2.ROTATE_180)
                elif rot == 270:
                    warped_rot = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                results = piece_detector.model(warped_rot, conf=conf, verbose=False)[0]
                detections = []
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    b_conf = float(box.conf[0])
                    cls_name = results.names[int(box.cls[0])]
                    piece_code = PIECE_CODES.get(cls_name)
                    if piece_code is not None:
                        detections.append((piece_code, x1, y1, x2, y2, b_conf))
                        
                if len(detections) < 5:
                    continue
                    
                for margin in margins:
                    for alpha in alphas:
                        board_state = np.full((8, 8), None, dtype=object)
                        for piece_code, x1, y1, x2, y2, b_conf in detections:
                            cx = (x1 + x2) / 2
                            # Apply alpha blending for y
                            cy_mapped = y1 * alpha + y2 * (1.0 - alpha)
                            
                            row, col = PieceDetector.center_to_square(cx, cy_mapped, border_margin=margin)
                            if board_state[row][col] is None or b_conf > board_state[row][col][1]:
                                board_state[row][col] = (piece_code, b_conf)
                                
                        # Calculate matches
                        matches = 0
                        for r in range(8):
                            for c in range(8):
                                sq = chess.square(c, 7 - r)
                                piece = starting_board.piece_at(sq)
                                expected = piece.symbol() if piece else None
                                detected = board_state[r][c][0] if board_state[r][c] is not None else None
                                if expected == detected:
                                    matches += 1
                                    
                        if matches > best_score:
                            best_score = matches
                            best_conf = conf
                            best_rot = rot
                            best_margin = margin
                            best_alpha = alpha
                            best_pieces = len(detections)
                            
        print(f"  Best config: conf={best_conf}, rotation={best_rot}°, margin={best_margin}px, alpha={best_alpha} (score={best_score}/64, pieces={best_pieces})")
                        
        print(f"  Best config: rotation={best_rot}°, margin={best_margin}px, alpha={best_alpha} (score={best_score}/64, pieces={best_pieces})")

if __name__ == "__main__":
    test_y2()
