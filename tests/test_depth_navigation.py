"""Tests for depth constraints — the "how far" half of bank navigation.

A bank position is `[SIGN][DEPTH]`. `navigate()` had only ever seen the sign,
so `efficiency=+1` admitted a 13B model and a 400B one equally and "give me
the largest" returned 31B, 120B, 14B, 12B, 32B, 26B. The depth machinery
already existed (`BankConstraint.min_signed`, `find_models_by_bank_range`) but
`StructuredQuery` — what the engine and the MCP tool actually use — could not
express it.

Depth is a FILTER, applied by set intersection during candidate selection. It
decides who is admissible; what ORDERS the admissible set is the separate
scalar signal. These tests pin the filter and deliberately assert nothing
about sort order — see docs/navigation.md.
"""

from __future__ import annotations

import pytest

from model_atlas import db
from model_atlas.query_navigate import _depth_constrained_ids, _nav_candidates, navigate
from model_atlas.query_types import StructuredQuery


@pytest.fixture
def banked(conn):
    """One model at each EFFICIENCY depth, positive and negative."""
    anchor_id = db.get_or_create_anchor(conn, "chat", "CAPABILITY")
    for name, sign, depth in [
        ("zero", 0, 0),
        ("pos1", 1, 1), ("pos2", 1, 2), ("pos3", 1, 3),
        ("neg1", -1, 1), ("neg2", -1, 2),
    ]:
        mid = f"t/{name}"
        db.insert_model(conn, mid, author="t", source="huggingface")
        db.set_position(conn, mid, "EFFICIENCY", sign, depth)
        db.link_anchor(conn, mid, anchor_id, confidence=1.0)
    conn.commit()
    return conn


def _ids(conn, **kw) -> set[str]:
    return {r.model_id for r in navigate(conn, StructuredQuery(**kw))}


# ── The filter itself ─────────────────────────────────────────────────


def test_depth_constrained_ids_selects_along_one_direction(banked):
    assert _depth_constrained_ids(banked, "EFFICIENCY", 1, 1) == {
        "t/pos1", "t/pos2", "t/pos3"
    }
    assert _depth_constrained_ids(banked, "EFFICIENCY", 1, 3) == {"t/pos3"}
    assert _depth_constrained_ids(banked, "EFFICIENCY", -1, 2) == {"t/neg2"}


def test_depth_never_crosses_the_zero_state(banked):
    """A positive constraint must not admit negative-side models, however
    deep. Sign is the direction; depth is distance along it."""
    got = _depth_constrained_ids(banked, "EFFICIENCY", 1, 1)
    assert not {"t/neg1", "t/neg2", "t/zero"} & got


def test_deeper_constraint_narrows_monotonically(banked):
    sizes = [
        len(_depth_constrained_ids(banked, "EFFICIENCY", 1, d)) for d in (1, 2, 3)
    ]
    assert sizes == sorted(sizes, reverse=True) == [3, 2, 1]


# ── Through the query surface ─────────────────────────────────────────


def test_direction_alone_admits_every_depth(banked):
    """The defect this exists to fix: +1 alone cannot distinguish one step
    from three."""
    assert _ids(banked, efficiency=1) >= {"t/pos1", "t/pos2", "t/pos3"}


def test_min_depth_excludes_the_shallow_end(banked):
    got = _ids(banked, efficiency=1, min_depth={"EFFICIENCY": 2})
    assert got == {"t/pos2", "t/pos3"}


def test_min_depth_is_symmetric_across_directions(banked):
    assert _ids(banked, efficiency=-1, min_depth={"EFFICIENCY": 2}) == {"t/neg2"}


def test_min_depth_composes_with_require_anchors(banked):
    got = _ids(banked, efficiency=1, require_anchors=["chat"],
               min_depth={"EFFICIENCY": 3})
    assert got == {"t/pos3"}


def test_unsatisfiable_depth_returns_nothing_not_everything(banked):
    """An over-deep constraint must empty the set. Falling back to the
    unfiltered corpus would answer a question nobody asked."""
    assert _ids(banked, efficiency=1, min_depth={"EFFICIENCY": 99}) == set()


# ── Where depth does not apply ────────────────────────────────────────


def test_depth_is_ignored_at_the_zero_state(banked):
    """The zero state IS depth 0, so "at least N steps from it" is
    meaningless. The constraint is skipped rather than emptying the set."""
    assert _ids(banked, efficiency=0, min_depth={"EFFICIENCY": 3})


def test_depth_is_ignored_without_a_direction(banked):
    """A depth with no sign has no direction to travel."""
    assert _ids(banked, min_depth={"EFFICIENCY": 3})


def test_depth_on_an_unqueried_bank_is_ignored(banked):
    assert _ids(banked, efficiency=1, min_depth={"DOMAIN": 2}) >= {"t/pos1"}


def test_no_min_depth_behaves_exactly_as_before(banked):
    assert _ids(banked, efficiency=1) == _ids(banked, efficiency=1, min_depth=None)
    assert _ids(banked, efficiency=1) == _ids(banked, efficiency=1, min_depth={})


# ── Candidate-set contract ────────────────────────────────────────────


def test_candidate_order_is_preserved_under_filtering(banked):
    """Deterministic candidate order — the same query must not reorder
    between calls because a set was iterated."""
    directions = {"EFFICIENCY": 1}
    a = _nav_candidates(banked, set(), directions, {"EFFICIENCY": 1})
    b = _nav_candidates(banked, set(), directions, {"EFFICIENCY": 1})
    assert a == b
    unfiltered = _nav_candidates(banked, set(), directions, None)
    assert a == [m for m in unfiltered if m in set(a)]


def test_candidates_are_none_when_depth_admits_nothing(banked):
    assert _nav_candidates(banked, set(), {"EFFICIENCY": 1}, {"EFFICIENCY": 99}) is None


# ── The ceiling ───────────────────────────────────────────────────────


def test_max_depth_excludes_the_deep_end(banked):
    assert _ids(banked, efficiency=1, max_depth={"EFFICIENCY": 2}) == {
        "t/pos1", "t/pos2"
    }


def test_max_depth_at_zero_direction_is_a_band_around_the_zero_state(banked):
    """Unlike the floor, the ceiling DOES apply at direction 0 — "near the
    middle" has no side, so it admits both signs within the bound."""
    assert _ids(banked, efficiency=0, max_depth={"EFFICIENCY": 1}) == {
        "t/zero", "t/pos1", "t/neg1"
    }


def test_zero_direction_band_tightens(banked):
    assert _ids(banked, efficiency=0, max_depth={"EFFICIENCY": 0}) == {"t/zero"}


def test_floor_and_ceiling_compose_into_a_shell(banked):
    got = _ids(banked, efficiency=1,
               min_depth={"EFFICIENCY": 2}, max_depth={"EFFICIENCY": 2})
    assert got == {"t/pos2"}


def test_contradictory_bounds_return_nothing(banked):
    assert _ids(banked, efficiency=1,
                min_depth={"EFFICIENCY": 3}, max_depth={"EFFICIENCY": 1}) == set()


def test_max_depth_needs_a_specified_bank(banked):
    """No direction on the bank means the query never asked about it."""
    assert _ids(banked, max_depth={"EFFICIENCY": 0}) >= {"t/pos3"}


def test_no_max_depth_behaves_exactly_as_before(banked):
    assert _ids(banked, efficiency=1) == _ids(banked, efficiency=1, max_depth=None)
    assert _ids(banked, efficiency=1) == _ids(banked, efficiency=1, max_depth={})


def test_a_model_at_the_zero_state_survives_every_ceiling(banked):
    """Why max_depth cannot rescue the code-review case: a model recorded at
    (0,0) has depth 0, so no ceiling excludes it. GLM-4.6 and DeepSeek-V3 sit
    there because their parameter count is UNKNOWN, not because they are ~7B.
    That is a corpus defect, not a navigation one."""
    for bound in (0, 1, 2, 99):
        assert "t/zero" in _ids(banked, efficiency=0, max_depth={"EFFICIENCY": bound})
