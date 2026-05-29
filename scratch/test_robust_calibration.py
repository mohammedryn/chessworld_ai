import cv2
import numpy as np
import sys
import chess
from pathlib import Path

sys.path.append(str(Path("D:/chessworldai_assignment/chessvision-pgn").resolve()))

from src.board_detector import BoardDetector
from src.piece_detector import PieceDetector, PIECE_CODES

def robust_calibrate(video_path: str, piece_detector: PieceDetector, board_detector: BoardDetector):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0.0
        
    # Accumulate up to 5 frames with detected board
    warped_frames = []
    frame_idx = 0
    while len(warped_frames) < 5 and frame_idx < 300:
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
    
    if not warped_frames:
        return 0, 0.0
        
    starting_board = chess.Board()
    
    best_rotation = 0
    best_margin = 0.0
    best_score = -1
    
    rotations = [0, 90, 180, 270]
    margins = [0.0, 10.0, 15.0, 20.0, 25.0]
    
    # We use y_mapped = y1 * 0.3 + y2 * 0.7 to correct for perspective height
    alpha = 0.3
    calibration_conf = 0.25
    
    for rot in rotations:
        # Collect detections for all frames under this rotation
        all_detections = []
        for warped in warped_frames:
            warped_rot = warped.copy()
            if rot == 90:
                warped_rot = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180:
                warped_rot = cv2.rotate(warped, cv2.ROTATE_180)
            elif rot == 270:
                warped_rot = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
            results = piece_detector.model(warped_rot, conf=calibration_conf, verbose=False)[0]
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_name = results.names[int(box.cls[0])]
                piece_code = PIECE_CODES.get(cls_name)
                if piece_code is not None:
                    detections.append((piece_code, x1, y1, x2, y2, conf))
            all_detections.append(detections)
            
        for margin in margins:
            total_matches = 0
            total_pieces = 0
            for detections in all_detections:
                board_state = np.full((8, 8), None, dtype=object)
                for piece_code, x1, y1, x2, y2, conf in detections:
                    cx = (x1 + x2) / 2
                    cy_mapped = y1 * alpha + y2 * (1.0 - alpha)
                    row, col = PieceDetector.center_to_square(cx, cy_mapped, border_margin=margin)
                    if board_state[row][col] is None or conf > board_state[row][col][1]:
                        board_state[row][col] = (piece_code, conf)
                        
                # Compute matches
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
                total_pieces += len(detections)
                
            avg_matches = total_matches / len(warped_frames)
            avg_pieces = total_pieces / len(warped_frames)
            
            # Penalize if too few pieces detected
            if avg_pieces < 8:
                score = avg_matches - 20
            else:
                score = avg_matches
                
            if score > best_score:
                best_score = score
                best_rotation = rot
                best_margin = margin
                
    return best_rotation, best_margin

def test_all():
    piece_detector = PieceDetector("D:/chessworldai_assignment/chessvision-pgn/models/piece_detector.pt")
    board_detector = BoardDetector()
    
    for game_idx in range(1, 6):
        video_path = f"D:/chessworldai_assignment/chessvision-pgn/videos/game{game_idx}.mp4"
        if not Path(video_path).exists():
            video_path = f"D:/chessworldai_assignment/videos/game{game_idx}.mp4"
        if not Path(video_path).exists():
            continue
            
        rot, margin = robust_calibrate(video_path, piece_detector, board_detector)
        print(f"Game {game_idx}: locked rotation={rot}°, margin={margin}px")

if __name__ == "__main__":
    test_all()
