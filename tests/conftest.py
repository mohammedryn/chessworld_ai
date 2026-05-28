import pytest
import numpy as np
import cv2


@pytest.fixture
def synthetic_board_frame():
    """480x640 BGR frame with a white-bordered quad — simulates a board."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    pts = np.array([[80, 40], [560, 40], [560, 440], [80, 440]], dtype=np.int32)
    cv2.fillPoly(frame, [pts], (255, 255, 255))
    cv2.rectangle(frame, (95, 55), (545, 425), (40, 40, 40), -1)
    return frame, pts.astype(np.float32)


@pytest.fixture
def static_gray_frame():
    return np.full((480, 640), 128, dtype=np.uint8)


@pytest.fixture
def starting_board_state():
    """Standard chess starting position as 8x8 object array.

    Row 0 = rank 8 (black back rank), Row 7 = rank 1 (white back rank).
    Uppercase = white piece, lowercase = black piece.
    Each cell: (piece_code: str, confidence: float) or None.
    """
    board = np.full((8, 8), None, dtype=object)
    back_black = list('rnbqkbnr')
    back_white = list('RNBQKBNR')
    for col in range(8):
        board[0][col] = (back_black[col], 0.9)
        board[1][col] = ('p', 0.9)
        board[6][col] = ('P', 0.9)
        board[7][col] = (back_white[col], 0.9)
    return board


@pytest.fixture
def after_e4_board_state(starting_board_state):
    """Starting position after 1.e4 — white pawn moved from e2 (row6,col4) to e4 (row4,col4)."""
    board = starting_board_state.copy()
    board[6][4] = None
    board[4][4] = ('P', 0.9)
    return board
