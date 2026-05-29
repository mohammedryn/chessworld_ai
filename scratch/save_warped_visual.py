import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path("D:/chessworldai_assignment/chessvision-pgn").resolve()))

from src.board_detector import BoardDetector

def save_visuals(game_idx=1):
    video_path = f"D:/chessworldai_assignment/chessvision-pgn/videos/game{game_idx}.mp4"
    if not Path(video_path).exists():
        video_path = f"D:/chessworldai_assignment/videos/game{game_idx}.mp4"
        
    print(f"Generating visuals for game{game_idx}.mp4...")
    
    cap = cv2.VideoCapture(video_path)
    board_detector = BoardDetector()
    
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
            
            # Save the raw warped frame
            cv2.imwrite(f"output/game{game_idx}_warped_raw.jpg", warped)
            
            # Draw a grid on it for each rotation
            for rot in [0, 90, 180, 270]:
                warped_rot = warped.copy()
                if rot == 90:
                    warped_rot = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                elif rot == 180:
                    warped_rot = cv2.rotate(warped, cv2.ROTATE_180)
                elif rot == 270:
                    warped_rot = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                # Draw 8x8 grid
                for i in range(1, 8):
                    coord = int(i * 80)
                    # Vertical line
                    cv2.line(warped_rot, (coord, 0), (coord, 640), (0, 255, 0), 2)
                    # Horizontal line
                    cv2.line(warped_rot, (0, coord), (640, coord), (0, 255, 0), 2)
                    
                # Put labels on squares
                for row in range(8):
                    for col in range(8):
                        file_char = chr(ord('a') + col)
                        rank_char = str(8 - row)
                        sq_name = f"{file_char}{rank_char}"
                        cv2.putText(warped_rot, sq_name, (col * 80 + 5, row * 80 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                        
                cv2.imwrite(f"output/game{game_idx}_grid_{rot}.jpg", warped_rot)
                print(f"  Saved output/game{game_idx}_grid_{rot}.jpg")
            break
            
    cap.release()

if __name__ == "__main__":
    save_visuals(1)
    save_visuals(2)
    save_visuals(5)
