"""Tests for PageRank rank-percentile normalisation.

`pr_frac = pagerank / max(pagerank)` was the wrong transform for power-law
data: on the v0.4.0 corpus the median equals the minimum and the maximum is
~647x the median, so dividing by the maximum pinned 96.9% of models below 0.01
— under 0.2% of final score. The one term in the product that could ORDER
candidates was flattened into silence. See `docs/scoring-dynamic-range.md`.

Detective reported this function at 25 behaviours / 0 pinned, and specifically
that the surrounding suite was crash-dominated: mutants died by raising, not by
a value assertion, so the tests proved the code RAN without pinning WHAT it
returned. These pin the return value.
"""

from __future__ import annotations

from model_atlas.query_navigate import _pagerank_fractions


def _fracs(values: dict[str, float]) -> dict[str, float]:
    return _pagerank_fractions(values, list(values))


# ── Percentile semantics ──────────────────────────────────────────────


def test_highest_scores_one_and_lowest_scores_zero():
    f = _fracs({"lo": 1.0, "mid": 2.0, "hi": 3.0})
    assert f["hi"] == 1.0
    assert f["lo"] == 0.0
    assert f["mid"] == 0.5


def test_spread_is_uniform_regardless_of_value_skew():
    """The whole point. Three values spanning six orders of magnitude must
    still spread evenly — under `pr/max` the bottom two would both be ~0."""
    f = _fracs({"a": 1e-6, "b": 1e-3, "c": 1.0})
    assert sorted(f.values()) == [0.0, 0.5, 1.0]


def test_a_single_outlier_no_longer_crushes_everything_below_it():
    """The regression case: one model 647x the median used to force every
    other candidate to ~0. Percentile is scale-free, so it cannot."""
    values = {f"m{i}": 1.0 + i for i in range(10)}
    values["giant"] = 10_000.0
    f = _pagerank_fractions(values, list(values))
    assert f["giant"] == 1.0
    ordinary = [f[k] for k in values if k != "giant"]
    # Under pr/max these were all < 0.002. Now they occupy the whole range.
    assert min(ordinary) == 0.0
    assert max(ordinary) > 0.85
    assert len(set(ordinary)) == 10  # every one still distinguishable


def test_ties_collapse_to_one_percentile():
    """A large mass sits at the baseline PageRank. Spreading it across the
    bottom half would invent an ordering the data does not support."""
    f = _fracs({"a": 5.0, "b": 5.0, "c": 5.0, "top": 9.0})
    assert f["a"] == f["b"] == f["c"] == 0.0
    assert f["top"] == 1.0


def test_baseline_mass_lands_at_zero_not_mid_range():
    values = {f"floor{i}": 1.0 for i in range(50)}
    values.update({"mid": 2.0, "high": 3.0})
    f = _pagerank_fractions(values, list(values))
    assert all(f[k] == 0.0 for k in values if k.startswith("floor"))
    assert f["mid"] == 0.5 and f["high"] == 1.0


# ── Absent and degenerate inputs ──────────────────────────────────────


def test_missing_pagerank_scores_zero():
    """Absent is not central. A model with no stored score must not inherit
    a middling rank by accident."""
    f = _pagerank_fractions({"has": 5.0, "also": 1.0}, ["has", "also", "missing"])
    assert f["missing"] == 0.0


def test_zero_pagerank_is_excluded_from_the_percentile_scale():
    """A zero must not become a rung on the ladder. If it counted as a
    distinct value, the two real scores would be squeezed into the top of the
    range (0.5/1.0 instead of 0.0/1.0) and every genuine score would shift."""
    f = _pagerank_fractions({"z": 0.0, "a": 1.0, "b": 2.0}, ["z", "a", "b"])
    assert f["z"] == 0.0
    assert f["a"] == 0.0  # lowest REAL score, not the second rung
    assert f["b"] == 1.0


def test_two_distinct_values_span_the_full_range():
    """The `< 2` guard must admit exactly two — widening it to `<= 2` would
    silently neutralise the term for any two-candidate window."""
    f = _pagerank_fractions({"lo": 1.0, "hi": 2.0}, ["lo", "hi"])
    assert f == {"lo": 0.0, "hi": 1.0}


def test_all_candidates_tied_yields_no_signal():
    """One distinct value cannot order anything; every candidate scores 0 so
    the term stays neutral rather than awarding an arbitrary 1.0."""
    out = _fracs({"a": 4.0, "b": 4.0, "c": 4.0})
    assert out == {"a": 0.0, "b": 0.0, "c": 0.0}  # a dict, not None, and complete


def test_single_candidate_yields_no_signal():
    assert _fracs({"only": 7.0}) == {"only": 0.0}


def test_no_pagerank_at_all_yields_no_signal():
    f = _pagerank_fractions({}, ["a", "b"])
    assert f == {"a": 0.0, "b": 0.0}


def test_empty_candidate_set():
    assert _pagerank_fractions({}, []) == {}


# ── Contract ──────────────────────────────────────────────────────────


def test_every_candidate_gets_a_fraction():
    ids = ["a", "b", "c", "d"]
    f = _pagerank_fractions({"a": 1.0, "c": 2.0}, ids)
    assert set(f) == set(ids)


def test_all_fractions_are_within_the_unit_interval():
    f = _fracs({f"m{i}": float(i) * 3.7 for i in range(20)})
    assert all(0.0 <= v <= 1.0 for v in f.values())


def test_ordering_follows_pagerank():
    values = {"a": 3.0, "b": 1.0, "c": 2.0}
    f = _fracs(values)
    by_frac = sorted(values, key=lambda k: f[k])
    by_pr = sorted(values, key=lambda k: values[k])
    assert by_frac == by_pr


def test_is_deterministic_and_order_independent():
    """A pure function of the value multiset — candidate order must not move
    a score, or the same query would rank differently between calls."""
    values = {"a": 1.0, "b": 2.0, "c": 3.0}
    forward = _pagerank_fractions(values, ["a", "b", "c"])
    reversed_ = _pagerank_fractions(values, ["c", "b", "a"])
    assert forward == reversed_
    assert forward == _pagerank_fractions(values, ["a", "b", "c"])


def test_percentile_is_relative_to_the_candidate_set_not_the_corpus():
    """"Among the models that actually qualify, how central is this one" —
    so the same model scores differently in a different candidate set, which
    is the intended reading."""
    corpus_wide = _pagerank_fractions({"x": 2.0, "y": 1.0, "z": 3.0}, ["x", "y", "z"])
    narrowed = _pagerank_fractions({"x": 2.0, "y": 1.0}, ["x", "y"])
    assert corpus_wide["x"] == 0.5
    assert narrowed["x"] == 1.0
