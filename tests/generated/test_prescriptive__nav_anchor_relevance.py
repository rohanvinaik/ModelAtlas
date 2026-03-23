"""Prescriptive spec tests for _nav_anchor_relevance.

Target: src/model_atlas/query.py::_nav_anchor_relevance
Problem class: pure | σ=6 | Regime A
Tests the IDF-weighted prefer scoring and exponential avoid penalty.
"""

import pytest

from model_atlas.config import NAVIGATE_AVOID_DECAY
from model_atlas.query import _nav_anchor_relevance

# --- No constraints → 1.0 ---


def test_no_constraints_returns_one():
    """Invariant: no prefer set and no avoid set returns 1.0."""
    result = _nav_anchor_relevance(
        model_anchor_set={"a", "b"},
        prefer_set=set(),
        avoid_set=set(),
        idf={},
        prefer_idf_total=0.0,
        has_constraints=False,
    )
    assert result == 1.0


# --- Avoid penalty: 0.5^count ---


def test_one_avoided_anchor_halves_score():
    """Invariant: avoid_penalty is 0.5^1 = 0.5."""
    result = _nav_anchor_relevance(
        model_anchor_set={"a", "bad"},
        prefer_set=set(),
        avoid_set={"bad"},
        idf={},
        prefer_idf_total=0.0,
        has_constraints=True,
    )
    assert result == pytest.approx(NAVIGATE_AVOID_DECAY**1)


def test_two_avoided_anchors_quarter_score():
    """Invariant: avoid_penalty is 0.5^2 = 0.25."""
    result = _nav_anchor_relevance(
        model_anchor_set={"a", "bad1", "bad2"},
        prefer_set=set(),
        avoid_set={"bad1", "bad2"},
        idf={},
        prefer_idf_total=0.0,
        has_constraints=True,
    )
    assert result == pytest.approx(NAVIGATE_AVOID_DECAY**2)


def test_three_avoided_anchors_eighth_score():
    """Invariant: avoid_penalty is 0.5^3 = 0.125."""
    result = _nav_anchor_relevance(
        model_anchor_set={"bad1", "bad2", "bad3"},
        prefer_set=set(),
        avoid_set={"bad1", "bad2", "bad3"},
        idf={},
        prefer_idf_total=0.0,
        has_constraints=True,
    )
    assert result == pytest.approx(NAVIGATE_AVOID_DECAY**3)


def test_avoided_anchor_not_present_no_penalty():
    """No avoided anchors matched → penalty = 0.5^0 = 1.0."""
    result = _nav_anchor_relevance(
        model_anchor_set={"a", "b"},
        prefer_set=set(),
        avoid_set={"bad"},
        idf={},
        prefer_idf_total=0.0,
        has_constraints=True,
    )
    assert result == 1.0


# --- Prefer scoring: IDF-weighted fraction ---


def test_prefer_all_matched():
    """All preferred anchors matched → prefer_score = 1.0."""
    idf = {"x": 2.0, "y": 3.0}
    total = 2.0 + 3.0
    result = _nav_anchor_relevance(
        model_anchor_set={"x", "y", "z"},
        prefer_set={"x", "y"},
        avoid_set=set(),
        idf=idf,
        prefer_idf_total=total,
        has_constraints=True,
    )
    assert result == pytest.approx(1.0)


def test_prefer_none_matched():
    """No preferred anchors matched → prefer_score ≈ 0."""
    idf = {"x": 2.0, "y": 3.0}
    total = 2.0 + 3.0
    result = _nav_anchor_relevance(
        model_anchor_set={"a", "b"},
        prefer_set={"x", "y"},
        avoid_set=set(),
        idf=idf,
        prefer_idf_total=total,
        has_constraints=True,
    )
    assert result == pytest.approx(0.0)


def test_prefer_partial_idf_weighted():
    """Partial match: IDF weighting means rare anchor contributes more."""
    idf = {"rare": 5.0, "common": 1.0}
    total = 5.0 + 1.0
    # Model has rare but not common
    result = _nav_anchor_relevance(
        model_anchor_set={"rare", "other"},
        prefer_set={"rare", "common"},
        avoid_set=set(),
        idf=idf,
        prefer_idf_total=total,
        has_constraints=True,
    )
    assert result == pytest.approx(5.0 / 6.0)


# --- Combined: prefer × avoid ---


def test_prefer_and_avoid_multiply():
    """Invariant: result = prefer_score × avoid_penalty."""
    idf = {"x": 2.0, "y": 3.0}
    total = 5.0
    result = _nav_anchor_relevance(
        model_anchor_set={"x", "bad"},
        prefer_set={"x", "y"},
        avoid_set={"bad"},
        idf=idf,
        prefer_idf_total=total,
        has_constraints=True,
    )
    prefer_score = 2.0 / 5.0  # only x matched
    avoid_penalty = 0.5  # one avoided
    assert result == pytest.approx(prefer_score * avoid_penalty)


# --- BOUNDARY: prefer_idf_total > 0 vs >= 0 ---


def test_prefer_idf_total_zero_returns_one():
    """When prefer_idf_total=0 (no prefer set), prefer_score defaults to 1.0."""
    result = _nav_anchor_relevance(
        model_anchor_set={"a"},
        prefer_set=set(),
        avoid_set=set(),
        idf={},
        prefer_idf_total=0.0,
        has_constraints=True,
    )
    assert result == 1.0


# --- Result bounded ---


def test_result_in_zero_one():
    """Invariant: result always in [0, 1]."""
    idf = {"a": 1.0, "b": 2.0, "c": 3.0}
    for prefer in [set(), {"a"}, {"a", "b"}, {"a", "b", "c"}]:
        for avoid in [set(), {"a"}, {"b", "c"}]:
            total = sum(idf.get(a, 0.0) for a in prefer)
            r = _nav_anchor_relevance(
                model_anchor_set={"a", "b"},
                prefer_set=prefer,
                avoid_set=avoid,
                idf=idf,
                prefer_idf_total=total,
                has_constraints=bool(prefer or avoid),
            )
            assert 0 <= r <= 1.0
