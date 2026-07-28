"""Tests for the cross-source order parameter.

The measure is easy to get subtly wrong in ways that still produce a number,
so these pin the mechanism rather than just the output — particularly
cancellation, which is the property that distinguishes an order parameter
from an average.
"""

from __future__ import annotations

import math

import pytest

from model_atlas import db
from model_atlas.bank_coherence import (
    DISSENT_THRESHOLD,
    check_efficiency_coherence,
    order_parameter,
    params_from_geometry,
    position_to_phase,
    size_to_position,
)


# ── The order parameter ───────────────────────────────────────────────


def test_identical_phases_are_perfectly_coherent():
    r, psi = order_parameter([1.0, 1.0, 1.0])
    assert r == pytest.approx(1.0)
    assert psi == pytest.approx(1.0)


def test_antiphase_estimators_cancel_to_zero():
    """The defining property. A scalar mean of the same two values gives their
    midpoint and hides the disagreement entirely; the vector sum annihilates."""
    r, _ = order_parameter([0.0, math.pi])
    assert r == pytest.approx(0.0, abs=1e-9)


def test_a_uniform_scatter_is_incoherent():
    n = 12
    r, _ = order_parameter([2 * math.pi * i / n for i in range(n)])
    assert r == pytest.approx(0.0, abs=1e-9)


def test_r_falls_monotonically_as_estimators_separate():
    spreads = [0.0, math.pi / 8, math.pi / 4, math.pi / 2, math.pi]
    rs = [order_parameter([0.0, s])[0] for s in spreads]
    assert rs == sorted(rs, reverse=True)


def test_a_zero_weight_source_does_not_vote():
    """Coverage is uneven — an absent fact must contribute no phasor rather
    than a neutral one, or missing data would fake agreement."""
    both = order_parameter([0.0, math.pi], [1.0, 1.0])[0]
    muted = order_parameter([0.0, math.pi], [1.0, 0.0])[0]
    assert both == pytest.approx(0.0, abs=1e-9)
    assert muted == 0.0  # one contributor -> no signal, not perfect agreement


def test_one_estimator_cannot_corroborate_itself():
    """Reporting 1.0 for a single source would read as perfect agreement when
    nothing agreed with anything."""
    assert order_parameter([1.2]) == (0.0, 0.0)
    assert order_parameter([]) == (0.0, 0.0)


def test_weights_pull_the_consensus_phase():
    _, psi_even = order_parameter([0.0, math.pi / 2], [1.0, 1.0])
    _, psi_skewed = order_parameter([0.0, math.pi / 2], [9.0, 1.0])
    assert psi_skewed < psi_even


# ── The embedding ─────────────────────────────────────────────────────


def test_the_zero_state_sits_at_the_middle_of_the_half_circle():
    assert position_to_phase(0) == pytest.approx(math.pi / 2)


def test_the_two_directions_are_antiphase():
    """Sub-1B versus frontier must cancel, not average to 'mid-sized'."""
    r, _ = order_parameter([position_to_phase(-4), position_to_phase(4)])
    assert r == pytest.approx(0.0, abs=1e-9)


def test_the_embedding_is_a_half_circle_so_extremes_cannot_wrap():
    """On a full circle the most severe disagreement would look like perfect
    agreement, which is the one failure a coherence measure cannot afford."""
    assert 0.0 <= position_to_phase(-4) < position_to_phase(4) <= math.pi
    assert position_to_phase(4) - position_to_phase(-4) == pytest.approx(math.pi)


def test_outlying_positions_clamp_rather_than_wrap():
    assert position_to_phase(99) == position_to_phase(4)
    assert position_to_phase(-99) == position_to_phase(-4)


def test_neighbouring_positions_stay_highly_coherent():
    r, _ = order_parameter([position_to_phase(0), position_to_phase(1)])
    assert r > DISSENT_THRESHOLD


# ── Buckets and geometry ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "billions,position",
    [(0.13, -3), (0.6, -2), (3.0, -1), (7.0, 0), (13.0, 1), (30.0, 2),
     (70.0, 3), (400.0, 4)],
)
def test_size_buckets_match_the_extractor(billions, position):
    assert size_to_position(billions) == position


def test_a_nonsense_size_is_unplaceable():
    assert size_to_position(0.0) is None
    assert size_to_position(-5.0) is None


def test_geometry_recovers_a_known_model_to_the_right_bucket():
    """Llama-3-8B-ish geometry. The formula runs high (no tied embeddings, no
    GQA reduction), so this asserts the BUCKET, which is what is compared."""
    b = params_from_geometry(num_layers=32, hidden=4096, intermediate=14336, vocab=128256)
    assert size_to_position(b) == 0  # 7B-class


def test_geometry_separates_a_tiny_model_from_a_frontier_one():
    tiny = params_from_geometry(num_layers=12, hidden=768, intermediate=3072, vocab=32000)
    huge = params_from_geometry(num_layers=126, hidden=16384, intermediate=53248, vocab=128256)
    assert size_to_position(tiny) < 0 < size_to_position(huge)


# ── The check ─────────────────────────────────────────────────────────


@pytest.fixture
def corpus(conn):
    def add(mid, stored, layers, hidden, inter, vocab):
        db.insert_model(conn, mid, author="t", source="huggingface")
        for k, v in [("parameter_count_b", stored), ("num_layers", layers),
                     ("hidden_size", hidden), ("intermediate_size", inter),
                     ("vocab_size", vocab)]:
            db.set_metadata(conn, mid, k, str(v), "float")

    # agrees: geometry implies ~7B, stored says 7B
    add("ok/agrees", 7.0, 32, 4096, 14336, 128256)
    # dissents: stored claims frontier, geometry implies a 135M model
    add("bad/parse-error", 627.0, 12, 576, 1536, 32000)
    conn.commit()
    return conn


def test_agreeing_sources_produce_no_finding(corpus):
    assert [f.model_id for f in check_efficiency_coherence(corpus)] == ["bad/parse-error"]


def test_a_finding_carries_both_estimates_for_triage(corpus):
    finding = check_efficiency_coherence(corpus)[0]
    assert finding.bank == "EFFICIENCY"
    assert finding.evidence["parameter_count_b"] == 627.0
    assert finding.evidence["config_geometry_b"] < 1.0
    assert finding.positions == {"stored": 4, "computed": -3}
    assert finding.r < DISSENT_THRESHOLD


def test_models_missing_an_estimator_produce_no_finding(conn):
    """A single source cannot dissent, so an incomplete model must be silent
    rather than flagged."""
    db.insert_model(conn, "t/partial", author="t", source="huggingface")
    db.set_metadata(conn, "t/partial", "parameter_count_b", "7.0", "float")
    conn.commit()
    assert check_efficiency_coherence(conn) == []


def test_the_check_never_writes(corpus):
    before = corpus.execute("SELECT COUNT(*) FROM model_metadata").fetchone()[0]
    check_efficiency_coherence(corpus)
    assert corpus.execute("SELECT COUNT(*) FROM model_metadata").fetchone()[0] == before


def test_findings_are_ordered_worst_first(corpus):
    db.insert_model(corpus, "bad/milder", author="t", source="huggingface")
    for k, v in [("parameter_count_b", 30.0), ("num_layers", 32),
                 ("hidden_size", 4096), ("intermediate_size", 14336),
                 ("vocab_size", 128256)]:
        db.set_metadata(corpus, "bad/milder", k, str(v), "float")
    corpus.commit()
    rs = [f.r for f in check_efficiency_coherence(corpus)]
    assert rs == sorted(rs)


def test_threshold_governs_how_much_dissent_counts(corpus):
    assert check_efficiency_coherence(corpus, threshold=0.0) == []
    assert len(check_efficiency_coherence(corpus, threshold=1.01)) == 2


def test_non_numeric_metadata_is_skipped_not_fatal(conn):
    db.insert_model(conn, "t/junk", author="t", source="huggingface")
    for k, v in [("parameter_count_b", "not-a-number"), ("num_layers", "32"),
                 ("hidden_size", "4096"), ("intermediate_size", "14336"),
                 ("vocab_size", "128256")]:
        db.set_metadata(conn, "t/junk", k, v, "str")
    conn.commit()
    assert check_efficiency_coherence(conn) == []


def test_findings_serialize_for_reporting(corpus):
    d = check_efficiency_coherence(corpus)[0].to_dict()
    assert d["bank"] == "EFFICIENCY"
    assert set(d) == {"model_id", "bank", "r", "positions", "evidence"}


# ── Guards against the oscillator choices that failed ─────────────────


def test_orthogonal_bank_positions_are_not_valid_oscillators():
    """Measured on the real corpus: an order parameter over a model's 8 bank
    positions gives r in [0.91, 0.98] for known-good and known-bad alike,
    because the banks are orthogonal by construction — small AND code-focused
    is not dissent. This is a documented dead end, kept as a test so it is not
    rediscovered."""
    good = [position_to_phase(p) for p in (0, 1, -1, 0, 1, 0, 0, -1)]
    bad = [position_to_phase(p) for p in (1, 1, -3, 0, 1, 1, 0, 0)]
    r_good, _ = order_parameter(good)
    r_bad, _ = order_parameter(bad)
    assert abs(r_good - r_bad) < 0.15, (
        "orthogonal banks do not separate good from bad; if this ever fails, "
        "the bank semantics changed and the dead end should be revisited"
    )


def test_a_scalar_mean_is_not_an_order_parameter():
    """A mean of the estimates hides exactly the disagreement the order
    parameter exists to expose."""
    phases = [position_to_phase(-4), position_to_phase(4)]
    assert sum(phases) / 2 == pytest.approx(position_to_phase(0))  # looks mid-sized
    assert order_parameter(phases)[0] == pytest.approx(0.0, abs=1e-9)  # is total dissent
