import chess
import numpy as np
from collections import deque
from typing import Optional

from .piece_detector import BoardState


class BoardStateMachine:
    def __init__(self, window_size: int = 3):
        self._window: deque = deque(maxlen=window_size)
        self._window_size = window_size
        self._committed: Optional[BoardState] = None
        self._chess_board = chess.Board()
        self._flipped = False
        self._pending_move: Optional[chess.Move] = None
        self._candidate: Optional[BoardState] = None

    def set_orientation(self, flipped: bool):
        """Call once at game start. flipped=True when black is at bottom of warped image."""
        self._flipped = flipped

    def update(self, board_state: BoardState) -> Optional[chess.Move]:
        self._window.append(board_state)
        if len(self._window) < self._window_size:
            return None

        voted = self._vote()

        if self._committed is None:
            self._committed = voted
            return None

        if self._boards_equal(voted, self._committed):
            # State matches committed — clear any pending transition
            self._pending_move = None
            self._candidate = None
            return None

        # voted differs from committed; check if we need to detect the move
        if self._pending_move is None:
            # Fresh candidate: try to find a legal move
            move = self._find_legal_move(voted)
            if move is None:
                return None
            # Speculatively push to chess board so subsequent checks work
            self._chess_board.push(move)
            self._pending_move = move
            self._candidate = voted

        # Commit and return move when window is unanimous on the new state;
        # also return (without committing yet) when majority already detected it
        if self._window_unanimous():
            self._committed = self._candidate
            result = self._pending_move
            self._pending_move = None
            self._candidate = None
            return result

        # Not yet unanimous — return the pending move (majority-detected)
        return self._pending_move

    def _window_unanimous(self) -> bool:
        """Return True if all frames in the window have the same piece codes."""
        if len(self._window) < self._window_size:
            return False
        first = self._window[0]
        for state in self._window:
            if not self._boards_equal(first, state):
                return False
        return True

    def _vote(self) -> BoardState:
        result: BoardState = np.full((8, 8), None, dtype=object)
        threshold = self._window_size // 2 + 1
        for row in range(8):
            for col in range(8):
                votes: dict = {}
                for state in self._window:
                    key = state[row][col][0] if state[row][col] is not None else None
                    votes[key] = votes.get(key, 0) + 1
                winner = max(votes, key=votes.__getitem__)
                if votes[winner] >= threshold:
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
        for move in self._chess_board.legal_moves:
            test = self._chess_board.copy()
            test.push(move)
            if self._position_matches(test, candidate):
                return move
        return None

    def _position_matches(self, chess_board: chess.Board, detected: BoardState) -> bool:
        for row in range(8):
            for col in range(8):
                if self._flipped:
                    sq = chess.square(7 - col, row)
                else:
                    sq = chess.square(col, 7 - row)

                piece = chess_board.piece_at(sq)
                expected = piece.symbol() if piece else None
                detected_code = detected[row][col][0] if detected[row][col] is not None else None

                if expected != detected_code:
                    return False
        return True

    @property
    def chess_board(self) -> chess.Board:
        return self._chess_board
