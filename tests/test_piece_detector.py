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
