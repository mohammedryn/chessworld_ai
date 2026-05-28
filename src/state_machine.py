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
            return None

        move = self._find_legal_move(voted)
        if move is not None:
            self._chess_board.push(move)
            self._committed = voted
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
                    sq = chess.square(7 - col, 7 - row)
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
