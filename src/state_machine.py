import chess
import numpy as np
from collections import deque
from typing import Optional

from .piece_detector import BoardState


class BoardStateMachine:
    def __init__(self, window_size: int = 3, min_frame_gap: int = 0,
                 move_threshold: int = 32):
        self._window: deque = deque(maxlen=window_size)
        self._window_size = window_size
        self._flipped = False
        self._chess_board = chess.Board()
        self._committed: Optional[BoardState] = None
        self._min_frame_gap = min_frame_gap
        self._last_move_frame: int = -min_frame_gap
        self._current_frame: int = 0
        self._move_threshold = move_threshold

    def _get_starting_board_state(self) -> BoardState:
        board = np.full((8, 8), None, dtype=object)
        starting_board = chess.Board()
        for sq in chess.SQUARES:
            piece = starting_board.piece_at(sq)
            if piece is not None:
                row, col = self._sq_to_rc(sq)
                board[row][col] = (piece.symbol(), 1.0)
        return board

    def set_orientation(self, flipped: bool):
        """Call once at game start. flipped=True when black is at bottom of warped image."""
        self._flipped = flipped
        self._committed = None

    def update(self, board_state: BoardState, frame_idx: int = 0) -> Optional[chess.Move]:
        self._current_frame = frame_idx
        self._window.append(board_state)
        if len(self._window) < self._window_size:
            return None

        voted = self._vote()

        if self._committed is None:
            # Commit the first voted state that has a reasonable number of pieces.
            # Lower threshold (28/64) ensures we commit at the START of the video
            # (starting position) rather than waiting for a mid-game position that
            # accidentally scores higher — committing mid-game desynchronises
            # _chess_board (always at starting position) from _committed.
            num_pieces = sum(1 for r in range(8) for c in range(8) if voted[r][c] is not None)
            score = self._match_score(self._chess_board, voted)
            if num_pieces >= 8 and score >= 28:
                self._committed = voted
            return None

        if self._boards_equal(voted, self._committed):
            return None

        move = self._find_legal_move(voted)
        if move is not None:
            if (self._current_frame - self._last_move_frame) < self._min_frame_gap:
                move = None  # too soon — discard before pushing to chess_board
            else:
                self._chess_board.push(move)
                self._committed = voted
                self._last_move_frame = self._current_frame
        return move

    def _vote(self) -> BoardState:
        result: BoardState = np.full((8, 8), None, dtype=object)
        for row in range(8):
            for col in range(8):
                votes: dict = {}
                for state in self._window:
                    key = state[row][col][0] if state[row][col] is not None else None
                    votes[key] = votes.get(key, 0) + 1
                winner = max(votes, key=votes.__getitem__)
                for state in self._window:
                    cell = state[row][col]
                    cell_key = cell[0] if cell is not None else None
                    if cell_key == winner:
                        result[row][col] = cell
                        break
        return result

    def _boards_equal(self, a: BoardState, b: BoardState) -> bool:
        for row in range(8):
            for col in range(8):
                code_a = a[row][col][0] if a[row][col] is not None else None
                code_b = b[row][col][0] if b[row][col] is not None else None
                if code_a != code_b:
                    return False
        return True

    def _find_legal_move(self, candidate: BoardState) -> Optional[chess.Move]:
        # Count pieces that disappeared vs committed (noise = missed detections)
        # A real single move vacates exactly one square (the source).
        # Allow up to MAX_GHOST_VACATIONS extra vacated squares due to YOLO misses.
        MAX_GHOST_VACATIONS = 10

        unexpected_vacations = self._count_unexpected_vacations(candidate)

        best_move = None
        best_score = -1
        for move in self._chess_board.legal_moves:
            test = self._chess_board.copy()
            test.push(move)

            from_row, from_col = self._sq_to_rc(move.from_square)
            to_row, to_col = self._sq_to_rc(move.to_square)

            # Source square must be empty in AT LEAST ONE window frame.
            # (voted/majority may lag behind; we need the source to have
            # been vacated in at least 1 clean frame to accept the move.)
            if not any(state[from_row][from_col] is None for state in self._window):
                continue

            # Relaxed check: Destination must be occupied if expected to have a piece
            dest_piece = test.piece_at(move.to_square)
            expected_occupied = (dest_piece is not None)
            detected_occupied = (candidate[to_row][to_col] is not None)
            if expected_occupied != detected_occupied:
                continue

            # Total unexpected vacations (excluding this move's source) must be tolerable
            # Each legal move vacates its own source; subtract 1 from total count
            extra_vacations = unexpected_vacations - 1
            if extra_vacations > MAX_GHOST_VACATIONS:
                continue

            score = self._match_score(test, candidate)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move if best_score >= self._move_threshold else None

    def _sq_to_rc(self, sq: int) -> tuple[int, int]:
        """Convert python-chess square to (row, col) in detected board coordinates."""
        if self._flipped:
            return chess.square_rank(sq), 7 - chess.square_file(sq)
        return 7 - chess.square_rank(sq), chess.square_file(sq)

    def _count_unexpected_vacations(self, candidate: BoardState) -> int:
        """Count squares where committed has a piece but candidate does not."""
        count = 0
        for row in range(8):
            for col in range(8):
                c = self._committed[row][col][0] if self._committed[row][col] else None
                d = candidate[row][col][0] if candidate[row][col] else None
                if c is not None and d is None:
                    count += 1
        return count

    def _match_score(self, chess_board: chess.Board, detected: BoardState) -> int:
        matches = 0
        for row in range(8):
            for col in range(8):
                if self._flipped:
                    sq = chess.square(7 - col, 7 - row)
                else:
                    sq = chess.square(col, 7 - row)
                piece = chess_board.piece_at(sq)
                expected_occupied = (piece is not None)
                detected_occupied = (detected[row][col] is not None)
                if expected_occupied == detected_occupied:
                    matches += 1
        return matches

    def _position_matches(self, chess_board: chess.Board, detected: BoardState) -> bool:
        return self._match_score(chess_board, detected) >= 42

    @property
    def chess_board(self) -> chess.Board:
        return self._chess_board
