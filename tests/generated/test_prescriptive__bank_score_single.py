"""Prescriptive spec tests for _bank_score_single.

Target: src/model_atlas/query.py::_bank_score_single
Problem class: pure | σ=9 | Regime A
The novel scoring function claimed in README and DESIGN.md.
Every branch, constant, and boundary is pinned.
"""

import pytest

from model_atlas.query import _bank_score_single

# --- Branch: direction == 0 (want zero state) ---


def test_direction_zero_at_origin():
    """direction=0, position=0 → 1/(1+0) = 1.0"""
    assert _bank_score_single(0, 0) == 1.0


def test_direction_zero_penalizes_positive():
    """direction=0, position=+2 → 1/(1+2) = 0.333"""
    assert _bank_score_single(2, 0) == pytest.approx(1 / 3)


def test_direction_zero_penalizes_negative():
    """direction=0, position=-3 → 1/(1+3) = 0.25"""
    assert _bank_score_single(-3, 0) == 0.25


# --- Branch: aligned (same sign as direction) ---


def test_positive_direction_positive_position():
    """direction=+1, position=+2 → aligned → 1.0"""
    assert _bank_score_single(2, 1) == 1.0


def test_negative_direction_negative_position():
    """direction=-1, position=-3 → aligned → 1.0"""
    assert _bank_score_single(-3, -1) == 1.0


# --- Branch: neutral (position == 0, direction != 0) ---


def test_positive_direction_at_zero():
    """direction=+1, position=0 → neutral → 0.5"""
    assert _bank_score_single(0, 1) == 0.5


def test_negative_direction_at_zero():
    """direction=-1, position=0 → neutral → 0.5"""
    assert _bank_score_single(0, -1) == 0.5


# --- Branch: opposed (wrong direction) ---


def test_positive_direction_negative_position():
    """direction=+1, position=-1 → opposed, alignment=-1 → 1/(1+1) = 0.5"""
    assert _bank_score_single(-1, 1) == 0.5


def test_positive_direction_deep_negative():
    """direction=+1, position=-3 → opposed, alignment=-3 → 1/(1+3) = 0.25"""
    assert _bank_score_single(-3, 1) == 0.25


def test_negative_direction_positive_position():
    """direction=-1, position=+2 → opposed, alignment=-2 → 1/(1+2) = 0.333"""
    assert _bank_score_single(2, -1) == pytest.approx(1 / 3)


# --- SWAP mutation killer: arg order matters ---


def test_arg_order_matters():
    """Kills SWAP_0: position and direction args are not interchangeable.
    pos=-3, dir=0 → want-zero → 1/(1+3) = 0.25
    pos=0, dir=-3 → opposed branch → 0.5 (at zero with nonzero direction)
    """
    assert _bank_score_single(-3, 0) != _bank_score_single(0, -3)


# --- Boundary: result always in [0, 1] ---


def test_result_bounded():
    """Invariant: result always in [0, 1]."""
    for pos in range(-5, 6):
        for direction in (-1, 0, 1):
            r = _bank_score_single(pos, direction)
            assert 0 <= r <= 1.0, (
                f"Out of bounds: pos={pos}, dir={direction}, result={r}"
            )


# --- Purity ---


def test_pure():
    """Invariant: must be pure."""
    assert _bank_score_single(2, 1) == _bank_score_single(2, 1)
    assert _bank_score_single(-1, -1) == _bank_score_single(-1, -1)
