"""MC/DC boundary tests for _score_constraint.

Each test pins a single comparison operator boundary — the exact value where
swapping <= to < (or >= to >, etc.) would change the result. These tests
are the minimum needed for MC/DC (DO-178C Level A) verification.
"""

from __future__ import annotations

import pytest

from src.model_atlas.query import _score_constraint
from src.model_atlas.query_types import BankConstraint


def _bc(**kwargs) -> BankConstraint:
    """Shorthand for BankConstraint with defaults."""
    defaults = {"bank": "TEST", "direction": None, "min_signed": None, "max_signed": None}
    defaults.update(kwargs)
    return BankConstraint(**defaults)


# --- Line 157: lo <= signed <= hi (range check) ---

class TestRangeBoundary:
    """MC/DC for the range check: lo <= signed <= hi"""

    def test_at_lo_boundary_inside(self):
        """signed == lo -> 1.0 (lo <= signed is True)"""
        assert _score_constraint(2, _bc(min_signed=2, max_signed=5)) == 1.0

    def test_at_lo_boundary_outside(self):
        """signed == lo - 1 -> decay (lo <= signed is False)"""
        result = _score_constraint(1, _bc(min_signed=2, max_signed=5))
        assert result < 1.0  # decayed
        assert result == pytest.approx(1.0 / (1.0 + 1))  # distance = 1

    def test_at_hi_boundary_inside(self):
        """signed == hi -> 1.0 (signed <= hi is True)"""
        assert _score_constraint(5, _bc(min_signed=2, max_signed=5)) == 1.0

    def test_at_hi_boundary_outside(self):
        """signed == hi + 1 -> decay (signed <= hi is False)"""
        result = _score_constraint(6, _bc(min_signed=2, max_signed=5))
        assert result < 1.0
        assert result == pytest.approx(1.0 / (1.0 + 1))


# --- Line 162: signed >= lo (lo-only check) ---

class TestLoOnlyBoundary:
    """MC/DC for lo-only: return 1.0 if signed >= lo"""

    def test_at_lo_exact(self):
        """signed == lo -> 1.0 (>= is True)"""
        assert _score_constraint(3, _bc(min_signed=3)) == 1.0

    def test_below_lo(self):
        """signed == lo - 1 -> decay (>= is False)"""
        result = _score_constraint(2, _bc(min_signed=3))
        assert result < 1.0
        assert result == pytest.approx(1.0 / (1.0 + 1))


# --- Line 165: signed <= hi (hi-only check) ---

class TestHiOnlyBoundary:
    """MC/DC for hi-only: return 1.0 if signed <= hi"""

    def test_at_hi_exact(self):
        """signed == hi -> 1.0 (<= is True)"""
        assert _score_constraint(3, _bc(max_signed=3)) == 1.0

    def test_above_hi(self):
        """signed == hi + 1 -> decay (<= is False)"""
        result = _score_constraint(4, _bc(max_signed=3))
        assert result < 1.0
        assert result == pytest.approx(1.0 / (1.0 + 1))


# --- Line 168: signed == 0 (direction zero check) ---

class TestDirectionZeroBoundary:
    """MC/DC for direction: if signed == 0 return 0.5"""

    def test_signed_zero(self):
        """signed == 0 -> 0.5"""
        assert _score_constraint(0, _bc(direction=1)) == 0.5

    def test_signed_nonzero(self):
        """signed != 0 -> not 0.5 (takes the else branch)"""
        result = _score_constraint(1, _bc(direction=1))
        assert result != 0.5
        assert result == 1.0  # aligned


# --- Line 170: signed * c.direction > 0 (alignment check) ---

class TestAlignmentBoundary:
    """MC/DC for direction alignment: signed * direction > 0"""

    def test_aligned_positive(self):
        """signed=2, direction=1 -> product=2 > 0 -> 1.0"""
        assert _score_constraint(2, _bc(direction=1)) == 1.0

    def test_anti_aligned(self):
        """signed=-1, direction=1 -> product=-1, not > 0 -> decay"""
        result = _score_constraint(-1, _bc(direction=1))
        assert result < 1.0
        assert result == pytest.approx(1.0 / (1.0 + 1))

    def test_zero_product_boundary(self):
        """signed=0, direction=1 -> product=0, not > 0 -> 0.5 (caught by == 0 check first)"""
        # This actually hits the signed == 0 branch first
        assert _score_constraint(0, _bc(direction=1)) == 0.5
