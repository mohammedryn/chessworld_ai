import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path("D:/chessworldai_assignment/chessvision-pgn").resolve()))

from src.board_detector import BoardDetector
from src.piece_detector import PieceDetector, PIECE_CODES

def check_game_starts():
    piece_detector = PieceDetector("D:/chessworldai_assignment/chessvision-pgn/models/piece_detector.pt")
    board_detector = BoardDetector()
    
    # We will test all 5 games
    for game_idx in range(1, 6):
        video_path = f"D:/chessworldai_assignment/chessvision-pgn/videos/game{game_idx}.mp4"
        if not Path(video_path).exists():
            video_path = f"D:/chessworldai_assignment/videos/game{game_idx}.mp4"
        if not Path(video_path).exists():
            continue
            
        print(f"\n=================== game{game_idx}.mp4 ===================")
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
            print("  Could not detect board")
            continue
            
        # Try all 4 rotations, draw grid, and print piece count
        rotations = [0, 90, 180, 270]
        for rot in rotations:
            warped_rot = warped.copy()
            if rot == 90:
                warped_rot = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180:
                warped_rot = cv2.rotate(warped, cv2.ROTATE_180)
            elif rot == 270:
                warped_rot = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
            results = piece_detector.model(warped_rot, conf=0.25, verbose=False)[0]
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_name = results.names[int(box.cls[0])]
                piece_code = PIECE_CODES.get(cls_name)
                if piece_code is not None:
                    detections.append(piece_code)
                    
            print(f"  Rotation {rot}°: YOLO detected {len(detections)} pieces (conf=0.25). Detections: {detections}")

if __name__ == "__main__":
    check_game_starts()
