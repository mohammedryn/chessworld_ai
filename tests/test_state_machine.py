import chess
import numpy as np
import pytest
from src.state_machine import BoardStateMachine


def _feed_state(sm, state, count=3):
    """Feed the same state count times to fill the sliding window."""
    move = None
    for _ in range(count):
        move = sm.update(state)
    return move


def test_no_move_when_window_not_full(starting_board_state):
    sm = BoardStateMachine()
    sm.set_orientation(flipped=False)
    result = sm.update(starting_board_state)
    assert result is None


def test_detects_pawn_push_e2e4(starting_board_state, after_e4_board_state):
    sm = BoardStateMachine()
    sm.set_orientation(flipped=False)
    _feed_state(sm, starting_board_state)
    move = _feed_state(sm, after_e4_board_state)
    assert move is not None
    assert move == chess.Move.from_uci("e2e4")


def test_no_move_when_state_unchanged(starting_board_state):
    sm = BoardStateMachine()
    sm.set_orientation(flipped=False)
    _feed_state(sm, starting_board_state)
    move = _feed_state(sm, starting_board_state)
    assert move is None


def test_illegal_diff_returns_none(starting_board_state):
    """Moving two pieces simultaneously is not a legal chess move."""
    sm = BoardStateMachine()
    sm.set_orientation(flipped=False)
    _feed_state(sm, starting_board_state)

    bad_state = starting_board_state.copy()
    bad_state[6][4] = None
    bad_state[4][4] = ('P', 0.9)
    bad_state[6][3] = None
    bad_state[4][3] = ('P', 0.9)

    move = _feed_state(sm, bad_state)
    assert move is None


def test_majority_vote_ignores_noisy_frame(starting_board_state, after_e4_board_state):
    """2 clean frames + 1 noisy frame -> move still detected."""
    sm = BoardStateMachine()
    sm.set_orientation(flipped=False)
    _feed_state(sm, starting_board_state)

    noisy = starting_board_state.copy()
    noisy[4][4] = ('P', 0.3)  # ghost detection on e4

    sm.update(after_e4_board_state)
    sm.update(noisy)
    move = sm.update(after_e4_board_state)
    assert move is not None
    assert move == chess.Move.from_uci("e2e4")
