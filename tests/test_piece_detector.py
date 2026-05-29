import pytest
from src.piece_detector import PieceDetector


def test_center_to_square_top_left():
    assert PieceDetector.center_to_square(40.0, 40.0) == (0, 0)


def test_center_to_square_bottom_right():
    assert PieceDetector.center_to_square(639.0, 639.0) == (7, 7)


def test_center_to_square_exact_boundary():
    assert PieceDetector.center_to_square(80.0, 0.0) == (0, 1)
    assert PieceDetector.center_to_square(0.0, 80.0) == (1, 0)


def test_center_to_square_clamps_to_grid():
    assert PieceDetector.center_to_square(640.0, 640.0) == (7, 7)


def test_center_to_square_with_border_margin():
    # margin = 20.0, active size = 600.0, sq_size = 75.0
    # cx = 40.0 -> adj_x = 20.0 -> col = 0
    assert PieceDetector.center_to_square(40.0, 40.0, border_margin=20.0) == (0, 0)
    
    # cx = 94.0 -> adj_x = 74.0 -> col = 0
    assert PieceDetector.center_to_square(94.0, 94.0, border_margin=20.0) == (0, 0)
    
    # cx = 96.0 -> adj_x = 76.0 -> col = 1
    assert PieceDetector.center_to_square(96.0, 96.0, border_margin=20.0) == (1, 1)
    
    # cx = 320.0 -> adj_x = 300.0 -> col = 4
    assert PieceDetector.center_to_square(320.0, 320.0, border_margin=20.0) == (4, 4)

