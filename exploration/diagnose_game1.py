import cv2
import numpy as np
import sys
import chess
from pathlib import Path

# Add src to path
sys.path.append(str(Path("D:/chessworldai_assignment/chessvision-pgn").resolve()))

from src.board_detector import BoardDetector
from src.piece_detector import PieceDetector
from src.state_machine import BoardStateMachine
from src.pipeline import ChessVisionPipeline

def diagnose():
    for game_idx in range(1, 6):
        video_path = f"D:/chessworldai_assignment/chessvision-pgn/videos/game{game_idx}.mp4"
        if not Path(video_path).exists():
            video_path = f"D:/chessworldai_assignment/videos/game{game_idx}.mp4"
        if not Path(video_path).exists():
            print(f"Skipping game{game_idx} (not found)")
            continue
            
        print(f"\n=================== Diagnosing game{game_idx}.mp4 ===================")
        pipeline = ChessVisionPipeline(
            piece_model_path="D:/chessworldai_assignment/chessvision-pgn/models/piece_detector.pt"
        )
        
        # Run calibration
        rotation, border_margin = pipeline._calibrate_board(video_path)
        print(f"Calibrated: rotation={rotation}, border_margin={border_margin}")
        
        cap = cv2.VideoCapture(video_path)
        board_detector = BoardDetector()
        piece_detector = PieceDetector("D:/chessworldai_assignment/chessvision-pgn/models/piece_detector.pt")
        state_machine = BoardStateMachine()
        state_machine.set_orientation(flipped=False)
        
        frame_idx = 0
        detected_frames = []
        
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
                    
                board_state = piece_detector.detect(warped, border_margin=border_margin)
                detected_frames.append(board_state)
                
                # Print how many pieces are detected
                detected_pieces = sum(1 for r in range(8) for c in range(8) if board_state[r][c] is not None)
                print(f"Frame {frame_idx}: Detected {detected_pieces} pieces.")
                
                if len(detected_frames) >= 3:
                    # Run sliding window vote
                    state_machine._window.clear()
                    for df in detected_frames[-3:]:
                        state_machine._window.append(df)
                    voted = state_machine._vote()
                    
                    # Check match score with starting position
                    score = state_machine._match_score(state_machine._chess_board, voted)
                    print(f"  Voted state match score with starting board: {score}/64")
                    
                    # Let's inspect mismatched squares
                    mismatches = []
                    for r in range(8):
                        for c in range(8):
                            sq = chess.square(c, 7 - r)
                            piece = state_machine._chess_board.piece_at(sq)
                            expected = piece.symbol() if piece else None
                            detected = voted[r][c][0] if voted[r][c] is not None else None
                            if expected != detected:
                                mismatches.append(f"row={r}, col={c} (sq={chess.square_name(sq)}): expected={expected}, detected={detected}")
                    if mismatches:
                        print(f"  Mismatches (showing first 5):")
                        for m in mismatches[:5]:
                            print(f"    {m}")
                    break
                    
        cap.release()

if __name__ == "__main__":
    diagnose()
