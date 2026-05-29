import cv2
import numpy as np
import sys
import chess
from pathlib import Path

sys.path.append(str(Path("D:/chessworldai_assignment/chessvision-pgn").resolve()))

from src.board_detector import BoardDetector
from src.piece_detector import PieceDetector, PIECE_CODES
from src.state_machine import BoardStateMachine

def diagnose():
    video_path = "D:/chessworldai_assignment/chessvision-pgn/videos/game1.mp4"
    if not Path(video_path).exists():
        video_path = "D:/chessworldai_assignment/videos/game1.mp4"
        
    print(f"Diagnosing game1.mp4 at key frames...")
    
    cap = cv2.VideoCapture(video_path)
    board_detector = BoardDetector()
    piece_detector = PieceDetector("D:/chessworldai_assignment/chessvision-pgn/models/piece_detector.pt")
    state_machine = BoardStateMachine()
    state_machine.set_orientation(flipped=False)
    
    # Use locked calibration for game1
    rotation = 270
    border_margin = 10.0
    
    frame_idx = 0
    while True:
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
                
            # Override _find_legal_move inside the diagnostic script to test relaxed check
            # Redefine match score to compare occupancy only
            def relaxed_match_score(chess_board, detected):
                matches = 0
                for row in range(8):
                    for col in range(8):
                        if state_machine._flipped:
                            sq = chess.square(7 - col, 7 - row)
                        else:
                            sq = chess.square(col, 7 - row)
                        expected_occupied = (chess_board.piece_at(sq) is not None)
                        detected_occupied = (detected[row][col] is not None)
                        if expected_occupied == detected_occupied:
                            matches += 1
                return matches
                
            state_machine._match_score = relaxed_match_score
            
            # Test standard matching logic but with dynamic _committed initialization and relaxed matching
            def standard_find_legal_move(candidate):
                MAX_GHOST_VACATIONS = 3
                unexpected_vacations = state_machine._count_unexpected_vacations(candidate)
                
                best_move = None
                best_score = -1
                for move in state_machine._chess_board.legal_moves:
                    test = state_machine._chess_board.copy()
                    test.push(move)
                    
                    from_row, from_col = state_machine._sq_to_rc(move.from_square)
                    to_row, to_col = state_machine._sq_to_rc(move.to_square)
                    
                    # Source must be empty in at least one frame
                    if not any(state[from_row][from_col] is None for state in state_machine._window):
                        continue
                        
                    # Relaxed check: Destination must be occupied if expected to have a piece
                    dest_piece = test.piece_at(move.to_square)
                    expected_occupied = (dest_piece is not None)
                    detected_occupied = (candidate[to_row][to_col] is not None)
                    if expected_occupied != detected_occupied:
                        continue
                        
                    # Total unexpected vacations (excluding source)
                    extra_vacations = unexpected_vacations - 1
                    if extra_vacations > MAX_GHOST_VACATIONS:
                        continue
                        
                    score = state_machine._match_score(test, candidate)
                    if score > best_score:
                        best_score = score
                        best_move = move
                return best_move if best_score >= 46 else None  # Tolerates a few mismatches, but keeps it safe
                
            state_machine._find_legal_move = standard_find_legal_move
            
            # Make sure _committed starts as None
            if frame_idx == 3:
                state_machine._committed = None
                
            board_state = piece_detector.detect(warped, border_margin=border_margin)
            
            # Custom update logic to test dynamic committed initialization
            state_machine._window.append(board_state)
            if len(state_machine._window) >= state_machine._window_size:
                voted = state_machine._vote()
                if state_machine._committed is None:
                    num_pieces = sum(1 for r in range(8) for c in range(8) if voted[r][c] is not None)
                    score = state_machine._match_score(state_machine._chess_board, voted)
                    if num_pieces >= 12 and score >= 42:
                        state_machine._committed = voted
                        print(f"Frame {frame_idx}: Committed initialized to voted board state! (pieces={num_pieces}, score={score}/64)")
                else:
                    if not state_machine._boards_equal(voted, state_machine._committed):
                        move = state_machine._find_legal_move(voted)
                        if move is not None:
                            state_machine._chess_board.push(move)
                            state_machine._committed = voted
                            print(f"Frame {frame_idx}: SUCCESS! Detected move {move.uci()}")
            
            # Let's inspect at frame 144, 240, 396, 462
            if frame_idx in [138, 141, 144, 234, 237, 240, 390, 393, 396, 453, 456, 459]:
                if state_machine._committed is not None:
                    voted = state_machine._vote()
                    score = state_machine._match_score(state_machine._chess_board, voted)
                    print(f"Frame {frame_idx}: voted pieces matches = {score}/64")
                
                # Print detected pieces vs expected
                detected_pieces = []
                for r in range(8):
                    for c in range(8):
                        if voted[r][c] is not None:
                            sq = chess.square(c, 7 - r)
                            detected_pieces.append(f"{chess.square_name(sq)}:{voted[r][c][0]}")
                print(f"  Detected: {', '.join(detected_pieces)}")
                
        if frame_idx > 500:
            break
            
    cap.release()

if __name__ == "__main__":
    diagnose()
