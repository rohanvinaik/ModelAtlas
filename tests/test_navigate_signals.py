"""Tests for the v0.4.1 scoring-layer helpers.

Each helper is tested in isolation so a future refactor that breaks one
surfaces immediately, not at the query-battery level.
"""

from __future__ import annotations

from model_atlas import db
from model_atlas.query_navigate import (
    _apply_bank_weights,
    _mechanical_fraction,
    _mode_weights,
    _mmr_rerank,
    _nav_absence_bonus,
    _nav_bank_alignment_weighted,
    _nav_rare_boost,
    _pmi_map,
    _standards_and_probs,
    _submodular_combine,
)
from model_atlas.query_types import NavigationResult, StructuredQuery


# ─── Pure helpers (no DB) ─────────────────────────────────────────


def test_submodular_combine_empty():
    """Empty signal list is neutral (returns 1.0)."""
    assert _submodular_combine([]) == 1.0


def test_submodular_combine_single_signal():
    """A single signal counts fully — sorted-descending × decay^0 = signal."""
    assert abs(_submodular_combine([0.3]) - 1.3) < 1e-9


def test_submodular_combine_multiple_signals_diminish():
    """Sorted descending × decay: [0.4, 0.2, 0.1] with decay=0.7 →
    0.4 + 0.2*0.7 + 0.1*0.49 = 0.4 + 0.14 + 0.049 = 0.589."""
    result = _submodular_combine([0.1, 0.4, 0.2], decay=0.7)
    assert abs(result - (1.0 + 0.4 + 0.14 + 0.049)) < 1e-6


def test_submodular_combine_zero_signals_neutral():
    """All-zero signals produce 1.0."""
    assert _submodular_combine([0.0, 0.0, 0.0]) == 1.0


def test_apply_bank_weights_none_returns_ones():
    """No override → every active bank gets weight 1.0 (default)."""
    dirs = {"CAPABILITY": 1, "EFFICIENCY": -1}
    out = _apply_bank_weights(dirs, None)
    assert out == {"CAPABILITY": 1.0, "EFFICIENCY": 1.0}


def test_apply_bank_weights_neutralize_renormalizes():
    """Zero on one bank → its factor becomes 0; the other renormalizes
    to preserve total mass (N_active / active_mass scaling)."""
    dirs = {"CAPABILITY": 1, "EFFICIENCY": -1, "QUALITY": 1}
    out = _apply_bank_weights(dirs, {"QUALITY": 0.0})
    assert out["QUALITY"] == 0.0
    # active = 2 banks each at raw 1.0, active_mass = 2, scale = 2/2 = 1
    assert out["CAPABILITY"] == 1.0
    assert out["EFFICIENCY"] == 1.0


def test_apply_bank_weights_amplify_renormalizes():
    """Weight of 2 on one bank; others stay 1. Renormalized so total = N."""
    dirs = {"CAPABILITY": 1, "EFFICIENCY": -1}
    out = _apply_bank_weights(dirs, {"CAPABILITY": 2.0})
    # raw = {C:2, E:1}, active_mass=3, active_count=2, scale=2/3
    assert abs(out["CAPABILITY"] - 2 * (2 / 3)) < 1e-9
    assert abs(out["EFFICIENCY"] - 1 * (2 / 3)) < 1e-9
    # Sum of weights equals active_count (mass preserved)
    assert abs(sum(out.values()) - 2.0) < 1e-9


def test_apply_bank_weights_all_zero_returns_ones():
    """If ALL banks are neutralized, fall back to defaults — otherwise
    the score collapses to 1.0 for every candidate."""
    dirs = {"CAPABILITY": 1, "EFFICIENCY": -1}
    out = _apply_bank_weights(dirs, {"CAPABILITY": 0.0, "EFFICIENCY": 0.0})
    assert out == {"CAPABILITY": 1.0, "EFFICIENCY": 1.0}


def test_mode_weights_canonical_amps_pr():
    """canonical mode has higher K_PR than niche."""
    can = _mode_weights("canonical", mech_frac=0.5)
    nic = _mode_weights("niche", mech_frac=0.5)
    assert can["K_PR"] > nic["K_PR"]
    assert nic["K_RARE"] > can["K_RARE"]


def test_mode_weights_auto_scales_with_mech_frac():
    """auto mode: pure-mechanical query gives higher K_PR than pure-semantic."""
    mech = _mode_weights("auto", mech_frac=1.0)
    sem = _mode_weights("auto", mech_frac=0.0)
    assert mech["K_PR"] > sem["K_PR"]
    assert sem["K_RARE"] > mech["K_RARE"]


def test_mode_weights_balanced_is_fixed():
    """balanced mode ignores mech_frac (fixed defaults regardless)."""
    a = _mode_weights("balanced", mech_frac=0.1)
    b = _mode_weights("balanced", mech_frac=0.9)
    assert a == b


def test_mmr_rerank_preserves_singletons():
    """0 or 1 results → no reranking possible."""
    assert _mmr_rerank([], {}, lam=0.7) == []
    r = NavigationResult(model_id="X", score=1.0)
    assert _mmr_rerank([r], {"X": {"a"}}, lam=0.7) == [r]


def test_mmr_rerank_diversifies_near_duplicates():
    """Two near-duplicates + one different → different one placed 2nd
    even though its raw score is lower, because it adds diversity."""
    r1 = NavigationResult(model_id="A", score=1.0)
    r2 = NavigationResult(model_id="B", score=0.9)  # near-dup of A
    r3 = NavigationResult(model_id="C", score=0.85)  # different
    anchors = {
        "A": {"x", "y", "z"},
        "B": {"x", "y", "z", "w"},  # Jaccard(A,B) = 3/4 = 0.75 (very similar)
        "C": {"p", "q"},              # Jaccard(A,C) = 0
    }
    # Low lambda favours diversity; C should beat B for slot #2
    out = _mmr_rerank([r1, r2, r3], anchors, lam=0.3)
    assert out[0].model_id == "A"
    assert out[1].model_id == "C"
    assert out[2].model_id == "B"


def test_mmr_rerank_high_lambda_favours_relevance():
    """With lambda near 1.0, MMR reduces to plain relevance sort."""
    r1 = NavigationResult(model_id="A", score=1.0)
    r2 = NavigationResult(model_id="B", score=0.9)
    r3 = NavigationResult(model_id="C", score=0.5)
    anchors = {"A": {"x"}, "B": {"x"}, "C": {"y"}}
    out = _mmr_rerank([r1, r2, r3], anchors, lam=0.99)
    assert [r.model_id for r in out] == ["A", "B", "C"]


# ─── DB-touching helpers (in-memory sqlite via `conn` fixture) ────


def _seed_two_models_with_anchors(conn) -> None:
    """A generalist model with many anchors + a specialist with few,
    all sharing one 'code' anchor. Used to test anchor-set behaviors."""
    db.insert_model(conn, "generalist", author="test")
    db.insert_model(conn, "specialist", author="test")
    aid_common = db.get_or_create_anchor(conn, "common-anchor", "CAPABILITY")
    aid_rare = db.get_or_create_anchor(conn, "rare-anchor", "DOMAIN")
    aid_other = db.get_or_create_anchor(conn, "other-anchor", "TRAINING")
    db.link_anchor(conn, "generalist", aid_common, confidence=0.9)
    db.link_anchor(conn, "generalist", aid_rare, confidence=0.9)
    db.link_anchor(conn, "generalist", aid_other, confidence=0.9)
    db.link_anchor(conn, "specialist", aid_rare, confidence=0.9)
    conn.commit()


def test_pmi_map_returns_positive_pmi_for_over_represented(conn):
    """Anchor that appears in every candidate but only 50% of the corpus →
    high positive PMI. Anchor at background rate → 0 (filtered)."""
    _seed_two_models_with_anchors(conn)
    # Add a third model to corpus that only has the common anchor
    db.insert_model(conn, "outsider", author="test")
    db.link_anchor(conn, "outsider", db.get_or_create_anchor(conn, "common-anchor", "CAPABILITY"), confidence=0.9)
    conn.commit()
    # Candidate set = only the two seeded models (they share rare-anchor)
    pmi = _pmi_map(conn, ["generalist", "specialist"], corpus_total=3)
    # rare-anchor: 100% in candidates (2/2), 66% in corpus (2/3) → PMI = log(1.0 / 0.66) > 0
    assert "rare-anchor" in pmi
    assert pmi["rare-anchor"] > 0


def test_standards_excludes_require_anchors(conn):
    """Anchors named in `exclude` (the require_anchors) are omitted from
    standards — their absence is impossible so they carry no information."""
    _seed_two_models_with_anchors(conn)
    # Both candidates carry common-anchor + rare-anchor (specialist added common for this test)
    aid_common = db.get_or_create_anchor(conn, "common-anchor", "CAPABILITY")
    db.link_anchor(conn, "specialist", aid_common, confidence=0.9)
    conn.commit()
    standards = _standards_and_probs(
        conn, ["generalist", "specialist"], exclude={"common-anchor"}, threshold=0.5
    )
    assert "common-anchor" not in standards


def test_nav_rare_boost_empty_prefers_neutral():
    """Empty prefer_set → 0 (neutral)."""
    assert _nav_rare_boost({"x"}, set(), {"x": 5.0}) == 0.0


def test_nav_rare_boost_matches_pmi_weighted():
    """Boost = matched-prefer PMI / total-prefer PMI. Half-match → 0.5."""
    prefers = {"a", "b"}
    pmi = {"a": 3.0, "b": 3.0}
    # Model matches only 'a' → 3.0 / 6.0 = 0.5
    assert abs(_nav_rare_boost({"a", "z"}, prefers, pmi) - 0.5) < 1e-9


def test_nav_absence_bonus_zero_when_no_standards_missing(conn):
    """A model carrying every standard → no absence bonus (0)."""
    standards = {"s1": 0.8, "s2": 0.7}
    assert _nav_absence_bonus({"s1", "s2", "other"}, standards) == 0.0


def test_nav_absence_bonus_accumulates_surprise_mass(conn):
    """Missing every standard → sum of -log(P) surprise mass."""
    import math
    standards = {"s1": 0.5, "s2": 0.5}
    expected = -math.log(0.5) + -math.log(0.5)  # = 2 * log(2) ≈ 1.386
    assert abs(_nav_absence_bonus(set(), standards) - expected) < 1e-6


def test_mechanical_fraction_pure_mechanical(conn):
    """Query using only mechanical banks → mechanical_fraction ≈ 1.0."""
    q = StructuredQuery(efficiency=-1, compatibility=1)
    mf = _mechanical_fraction(q, idf={}, conn=conn)
    assert mf == 1.0


def test_mechanical_fraction_pure_semantic(conn):
    """Query using only semantic banks → mechanical_fraction ≈ 0.0."""
    q = StructuredQuery(capability=1, domain=1, quality=1)
    mf = _mechanical_fraction(q, idf={}, conn=conn)
    assert mf == 0.0


def test_mechanical_fraction_architecture_is_dual(conn):
    """ARCHITECTURE alone → 0.5 (both mechanical and semantic weights)."""
    q = StructuredQuery(architecture=1)
    mf = _mechanical_fraction(q, idf={}, conn=conn)
    assert mf == 0.5


def test_nav_bank_alignment_weighted_zero_weight_neutralizes(conn):
    """Bank with weight 0 → its factor is 1.0 (no contribution)."""
    positions = {"CAPABILITY": (-1, 2)}  # opposed to query direction
    dirs = {"CAPABILITY": 1}
    # Full weight → Monty Hall penalty applies
    with_weight = _nav_bank_alignment_weighted(positions, dirs, {"CAPABILITY": 1.0})
    without_weight = _nav_bank_alignment_weighted(positions, dirs, {"CAPABILITY": 0.0})
    assert without_weight == 1.0
    assert with_weight < 1.0  # penalty applied
