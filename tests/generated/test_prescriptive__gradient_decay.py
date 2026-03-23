"""Prescriptive spec tests for _gradient_decay.

Target: src/model_atlas/query.py::_gradient_decay
Problem class: pure | σ=2 | Regime A
Generated from prescriptive spec, filled with exact oracle values.
"""

import pytest

from model_atlas.query import _gradient_decay


# --- VALUE mutation killers (σ=2: these 2 tests pin every constant) ---

def test_distance_zero_returns_one():
    """Kills VALUE_0: replace 1.0 with 0.0 in numerator."""
    assert _gradient_decay(0) == 1.0


def test_distance_one_returns_half():
    """Kills VALUE_1: replace 1.0 with 0.0 in denominator offset."""
    assert _gradient_decay(1) == 0.5


# --- Formula verification (kills any arithmetic mutation) ---

def test_formula_exact():
    """Invariant: returns 1/(1+|distance|) for all tested values."""
    for d in range(-10, 11):
        assert _gradient_decay(d) == pytest.approx(1.0 / (1.0 + abs(d)))


# --- Property: result always in (0, 1] ---

def test_result_bounded():
    """Invariant: result always in (0,1]."""
    for d in range(100):
        r = _gradient_decay(d)
        assert 0 < r <= 1.0


# --- Property: monotonically decreasing ---

def test_monotonically_decreasing():
    """Invariant: monotonically decreasing for positive distance."""
    prev = _gradient_decay(0)
    for d in range(1, 50):
        curr = _gradient_decay(d)
        assert curr < prev
        prev = curr


# --- Purity: same input → same output ---

def test_pure():
    """Invariant: must be pure — calling twice gives same result."""
    assert _gradient_decay(3) == _gradient_decay(3)
    assert _gradient_decay(0) == _gradient_decay(0)
