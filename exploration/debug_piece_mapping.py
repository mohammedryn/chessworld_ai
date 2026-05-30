import cv2
import numpy as np
import sys
import chess
from pathlib import Path

sys.path.append(str(Path("D:/chessworldai_assignment/chessvision-pgn").resolve()))

from src.piece_detector import PieceDetector, PIECE_CODES

def debug_mapping():
    # Load the warped frame of game1
    warped = cv2.imread("output/game1_warped_raw.jpg")
    if warped is None:
        print("Error: output/game1_warped_raw.jpg not found. Run save_warped_visual.py first.")
        return
        
    piece_detector = PieceDetector("D:/chessworldai_assignment/chessvision-pgn/models/piece_detector.pt")
    
    # We will test all 4 rotations of the warped frame
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
        
        print(f"\n--- ROTATION {rot}° ---")
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_name = results.names[int(box.cls[0])]
            piece_code = PIECE_CODES.get(cls_name)
            if piece_code is not None:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                detections.append((piece_code, cx, cy, x1, y1, x2, y2, conf))
                
        # Sort detections by y coordinate to see them row by row
        detections.sort(key=lambda x: x[2])
        
        for p, cx, cy, x1, y1, x2, y2, conf in detections:
            # Print coordinates and what squares they map to under different alphas and margins
            row_center, col_center = PieceDetector.center_to_square(cx, cy, border_margin=0.0)
            
            # Bottom mapping (alpha=0.2)
            cy_bottom = y1 * 0.2 + y2 * 0.8
            row_bottom, col_bottom = PieceDetector.center_to_square(cx, cy_bottom, border_margin=0.0)
            
            print(f"Piece {p} (conf={conf:.2f}): bbox=({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
            print(f"  Center: cx={cx:.1f}, cy={cy:.1f} -> row={row_center}, col={col_center} (sq={chess.square_name(chess.square(col_center, 7-row_center))})")
            print(f"  Bottom-80%: cy={cy_bottom:.1f} -> row={row_bottom}, col={col_bottom} (sq={chess.square_name(chess.square(col_bottom, 7-row_bottom))})")

if __name__ == "__main__":
    debug_mapping()
