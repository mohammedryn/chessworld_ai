import chess
import numpy as np
import pytest
from src.state_machine import BoardStateMachine


def _feed_state(sm, state, count=3):
    """Feed the same state count times. Returns the first move detected (if any).

    A move is emitted exactly once by the state machine; feeding the same state
    again after commit returns None. This helper preserves the first non-None result.
    """
    detected = None
    for _ in range(count):
        result = sm.update(state)
        if result is not None:
            detected = result
    return detected


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


def test_unresolvable_noise_returns_none(starting_board_state):
    """A heavily corrupted board state (5+ pieces missing) cannot be resolved to a legal move."""
    sm = BoardStateMachine()
    sm.set_orientation(flipped=False)
    _feed_state(sm, starting_board_state)

    # Wipe out 5 pieces from random squares — more missing pieces than MAX_GHOST_VACATIONS=3
    bad_state = starting_board_state.copy()
    bad_state[6][4] = None  # e2 pawn gone
    bad_state[6][3] = None  # d2 pawn gone
    bad_state[6][2] = None  # c2 pawn gone
    bad_state[6][1] = None  # b2 pawn gone
    bad_state[7][0] = None  # a1 rook gone

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
