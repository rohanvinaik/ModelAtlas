"""Tests for the corpus-quality eval harness.

The harness is the instrument the corpus work is measured with, so it needs
to be trustworthy before any of that work starts — a metric that scores a bad
window as fine, or that can be flattered by adding easy cases, is worse than
no metric.

These run on fixtures, not the real corpus: the harness must be testable
without a 351 MB download. `tests/test_evaluation_corpus.py` covers the
against-the-real-thing path and skips when it is absent.
"""

from __future__ import annotations


import pytest

from model_atlas import db
from model_atlas.evaluation.cases import CASES, EvalCase, case_by_name
from model_atlas.evaluation.facts import (
    ModelFacts,
    has_anchor,
    lacks_all_anchors,
    load_facts,
    min_downloads,
    not_a_vision_model,
    size_at_most,
    size_known,
)
from model_atlas.evaluation.harness import diff_reports, run_case, run_eval


@pytest.fixture
def corpus(conn):
    """Two chat models: one small with a known size, one with size unknown."""
    for mid, params, dl in [("good/small-1b", "1.0", "50000"), ("bad/unknown", None, "3")]:
        db.insert_model(conn, mid, author="t", source="huggingface")
        db.set_position(conn, mid, "EFFICIENCY", 0, 0)
        aid = db.get_or_create_anchor(conn, "chat", "CAPABILITY")
        db.link_anchor(conn, mid, aid, confidence=1.0)
        if params:
            db.set_metadata(conn, mid, "parameter_count_b", params, "float")
        db.set_metadata(conn, mid, "downloads", dl, "int")
    conn.commit()
    return conn


# ── Predicates ────────────────────────────────────────────────────────


def test_size_predicates_fail_on_unknown_rather_than_passing():
    """The whole point: 46% of the corpus has no parameter count and is
    positioned at the '~7B' zero state anyway. A predicate that let unknown
    pass would score that as fine and hide the defect."""
    unknown = ModelFacts(model_id="x", param_count_b=None)
    assert size_known()(unknown) is False
    assert size_at_most(8.0)(unknown) is False
    known = ModelFacts(model_id="y", param_count_b=3.0)
    assert size_at_most(8.0)(known) is True
    assert size_at_most(1.0)(known) is False


def test_anchor_predicates():
    f = ModelFacts(model_id="x", anchors=frozenset({"chat", "image-understanding"}))
    assert has_anchor("chat")(f) is True
    assert has_anchor("embedding")(f) is False
    assert lacks_all_anchors("embedding", "NER")(f) is True
    assert not_a_vision_model()(f) is False


def test_min_downloads_fails_on_unknown():
    assert min_downloads(100)(ModelFacts(model_id="x", downloads=None)) is False
    assert min_downloads(100)(ModelFacts(model_id="x", downloads=99)) is False
    assert min_downloads(100)(ModelFacts(model_id="x", downloads=100)) is True


def test_predicates_are_named_for_the_failure_report():
    """A failure must say which assertion broke, not '<lambda>'."""
    assert size_at_most(8.0).name == "size<=8.0B"
    assert has_anchor("chat").name == "has:chat"


def test_load_facts_reads_what_the_corpus_records(corpus):
    f = load_facts(corpus, "good/small-1b")
    assert f.param_count_b == 1.0
    assert f.downloads == 50000
    assert "chat" in f.anchors
    assert f.positions["EFFICIENCY"] == (0, 0)


def test_load_facts_leaves_absent_metadata_as_none(corpus):
    assert load_facts(corpus, "bad/unknown").param_count_b is None


# ── Scoring ───────────────────────────────────────────────────────────


def test_case_scores_the_whole_window_not_just_the_top_hit(corpus):
    """Scoring #1 alone would call a window good when #2 and #3 are junk."""
    case = EvalCase(
        name="t", ask="", query={"require_anchors": ["chat"]}, top_n=2,
        require_all=(size_known(),),
    )
    r = run_case(corpus, case)
    assert len(r.returned) == 2
    assert r.total_checks == 2  # one predicate x two results
    assert r.passed_checks == 1  # only the model with a known size
    assert r.score == 0.5


def test_failures_name_the_model_and_the_predicate(corpus):
    case = EvalCase(
        name="t", ask="", query={"require_anchors": ["chat"]}, top_n=2,
        require_all=(size_known(),),
    )
    assert "bad/unknown:size_known" in run_case(corpus, case).failures


def test_empty_window_scores_zero_not_a_vacuous_one(corpus):
    """No results is a failure to answer, not a set of satisfied constraints.
    Averaging a vacuous 1.0 would let a broken query improve the score."""
    case = EvalCase(
        name="t", ask="", query={"require_anchors": ["no-such-anchor"]},
        require_all=(size_known(),),
    )
    r = run_case(corpus, case)
    assert r.empty is True
    assert r.score == 0.0


def test_expect_any_of_matches_on_substring(corpus):
    case = EvalCase(
        name="t", ask="", query={"require_anchors": ["chat"]}, top_n=2,
        expect_any_of=("small-1b",),
    )
    r = run_case(corpus, case)
    assert r.expected_hits == ["small-1b"] and r.expected_misses == []


def test_aggregate_weighs_by_check_not_by_case(corpus):
    """Otherwise adding a tiny easy case would flatter the average as much as
    fixing a hard one."""
    big = EvalCase(
        name="big", ask="", query={"require_anchors": ["chat"]}, top_n=2,
        require_all=(size_known(), has_anchor("chat")),
    )
    small = EvalCase(
        name="small", ask="", query={"require_anchors": ["chat"]}, top_n=1,
        require_all=(has_anchor("chat"),),
    )
    rep = run_eval(corpus, (big, small))
    assert rep.total_checks == 5  # 2x2 + 1x1
    assert rep.passed_checks == 4  # only bad/unknown's size_known fails
    assert rep.score == pytest.approx(4 / 5)


def test_report_serializes_failures_for_tracking(corpus):
    rep = run_eval(corpus, (EvalCase(
        name="t", ask="", query={"require_anchors": ["chat"]}, top_n=2,
        require_all=(size_known(),)),))
    d = rep.to_dict()
    assert d["cases"]["t"]["failures"] == ["bad/unknown:size_known"]
    assert d["corpus_models"] == 2


# ── Baseline diffing ──────────────────────────────────────────────────


def test_diff_reports_movement_in_both_directions():
    base = {"score": 0.5, "cases": {"a": {"score": 1.0}, "b": {"score": 0.0}}}
    cur = {"score": 0.75, "cases": {"a": {"score": 0.5}, "b": {"score": 1.0}}}
    lines = "\n".join(diff_reports(base, cur))
    assert "50.0% → 75.0%" in lines
    assert "▼ a" in lines and "▲ b" in lines


def test_diff_reports_added_and_removed_cases():
    lines = "\n".join(diff_reports(
        {"score": 1.0, "cases": {"gone": {"score": 1.0}}},
        {"score": 1.0, "cases": {"fresh": {"score": 0.5}}},
    ))
    assert "+ fresh" in lines and "- gone" in lines


# ── The case set itself ───────────────────────────────────────────────


def test_case_names_are_unique():
    names = [c.name for c in CASES]
    assert len(names) == len(set(names))


def test_every_case_asserts_something():
    """A case with no expectations always scores 1.0 and inflates the total."""
    for c in CASES:
        assert c.require_all or c.expect_any_of or c.window, f"{c.name} asserts nothing"


def test_every_case_documents_why_it_exists():
    for c in CASES:
        assert c.ask, f"{c.name} has no plain-language ask"


def test_case_queries_are_valid_navigate_arguments():
    """Catches a typo'd kwarg before it shows up as a mysterious zero score."""
    from model_atlas.query_types import StructuredQuery

    for c in CASES:
        StructuredQuery(**c.query, limit=c.top_n)


def test_case_by_name_raises_on_unknown():
    assert case_by_name("rag_embeddings").name == "rag_embeddings"
    with pytest.raises(KeyError):
        case_by_name("nope")


def test_harness_never_writes_to_the_corpus(corpus):
    """Read-only, like the coherence audit. A metric that mutates what it
    measures is not a metric."""
    before = corpus.execute("SELECT COUNT(*) FROM model_anchors").fetchone()[0]
    run_eval(corpus, CASES[:2])
    assert corpus.execute("SELECT COUNT(*) FROM model_anchors").fetchone()[0] == before
def _f(model_id: str, size: float | None = None, **kw):
    return ModelFacts(model_id=model_id, param_count_b=size, **kw)


def test_size_span_rejects_a_window_of_near_identical_models():
    from model_atlas.evaluation.facts import sizes_span_at_least

    p = sizes_span_at_least(2.0)
    assert p([_f("a", 7.0), _f("b", 7.0), _f("c", 8.0)]) is False
    assert p([_f("a", 70.0), _f("b", 7.0)]) is True


def test_no_duplicate_lineage_catches_one_model_wearing_two_hats():
    from model_atlas.evaluation.facts import no_duplicate_lineage

    p = no_duplicate_lineage()
    assert p([_f("mlabonne/Daredevil-8B"), _f("mlabonne/Daredevil-8B-abliterated")]) is False
    assert p([_f("org/Qwen3-8B-GGUF"), _f("org/Qwen3-8B-MLX")]) is False
    assert p([_f("a/Qwen3-8B"), _f("b/Llama-3.1-8B")]) is True


def test_window_predicates_are_named_for_the_report():
    from model_atlas.evaluation.facts import no_duplicate_lineage, sizes_span_at_least

    assert sizes_span_at_least(2.0).name == "size_span>=2.0x"
    assert no_duplicate_lineage().name == "no_duplicate_lineage"


def test_window_assertion_scores_once_not_once_per_result(corpus):
    """Otherwise one ordering property would outweigh the whole on-topic
    check on a wide window."""
    from model_atlas.evaluation.facts import no_duplicate_lineage

    case = EvalCase(
        name="t", ask="", query={"require_anchors": ["chat"]}, top_n=2,
        window=(no_duplicate_lineage(),),
    )
    r = run_case(corpus, case)
    assert r.total_checks == 1  # one window predicate, not one per result


def test_window_failures_are_labelled_distinctly(corpus):
    from model_atlas.evaluation.facts import sizes_span_at_least

    case = EvalCase(
        name="t", ask="", query={"require_anchors": ["chat"]}, top_n=2,
        window=(sizes_span_at_least(1000.0),),
    )
    r = run_case(corpus, case)
    assert r.failures == ["<window>:size_span>=1000.0x"]
