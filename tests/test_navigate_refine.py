"""Tests for the refinement-guidance layer.

The contract under test: the engine reports what the query left unsaid, and
every reported option is directly applicable by merging `apply` into the
arguments the caller already sent. Nothing here may guess.
"""

from __future__ import annotations

import pytest

from model_atlas.query_navigate import (
    QUESTION_TEMPLATES,
    _anchor_hints,
    _axis_hints,
    build_refinement_guidance,
    render_question,
)
from model_atlas.query_types import NavigationResult, StructuredQuery


def _r(model_id: str, positions: dict, anchors: list[str] | None = None):
    return NavigationResult(
        model_id=model_id,
        score=1.0,
        positions={b: {"sign": s, "depth": d} for b, (s, d) in positions.items()},
        anchor_labels=anchors or [],
    )


# ─── Question skeletons ───────────────────────────────────────────


def test_render_fills_every_gap():
    out = render_question(
        "unconstrained_axis",
        {"range_low": "-2", "range_high": "+3", "bank": "EFFICIENCY",
         "answer_low": "smaller", "answer_high": "larger"},
    )
    assert "<" not in out and ">" not in out
    assert "EFFICIENCY" in out and "smaller" in out


def test_render_raises_on_missing_slot():
    """A template gap must never reach an MCP payload as literal '<bank>'.

    Raises on the first unfilled slot and names it, so the error points at
    the fix rather than at the template as a whole.
    """
    with pytest.raises(KeyError, match="range_high"):
        render_question("unconstrained_axis", {"range_low": "-2"})


def test_every_template_is_renderable_and_gapless():
    """Guards against a template gaining a slot nobody fills."""
    slots = {
        "count": 8, "range_low": "-1", "range_high": "+2", "bank": "DOMAIN",
        "answer_low": "a", "answer_high": "b",
        "present_in": 3, "out_of": 8, "anchor": "x",
        "scope": "all 50,906 models",
    }
    for tid in QUESTION_TEMPLATES:
        assert "<" not in render_question(tid, slots)


# ─── Axis hints ───────────────────────────────────────────────────


def test_specified_axis_is_not_asked_about():
    results = [_r("a", {"EFFICIENCY": (-1, 1)}), _r("b", {"EFFICIENCY": (1, 2)})]
    hints = _axis_hints(results, specified={"EFFICIENCY"})
    assert [h.bank for h in hints] == []


def test_uniform_axis_is_dropped():
    """Every result identical on a bank → asking would not narrow anything."""
    results = [_r("a", {"DOMAIN": (1, 1)}), _r("b", {"DOMAIN": (1, 1)})]
    assert _axis_hints(results, specified=set()) == []


def test_axes_ranked_by_spread():
    results = [
        _r("a", {"EFFICIENCY": (-1, 2), "QUALITY": (1, 0)}),
        _r("b", {"EFFICIENCY": (1, 3), "QUALITY": (1, 1)}),
    ]
    hints = _axis_hints(results, specified=set())
    assert hints[0].bank == "EFFICIENCY"  # wider spread ranks first
    assert hints[0].spread > hints[1].spread


def test_axis_option_applies_to_the_right_query_field():
    results = [_r("a", {"EFFICIENCY": (-1, 1)}), _r("b", {"EFFICIENCY": (1, 2)})]
    h = _axis_hints(results, specified=set())[0]
    assert [o.apply for o in h.options] == [{"efficiency": -1}, {"efficiency": 1}]


def test_options_never_point_outside_the_observed_range():
    """A window at 0..+3 must not offer -1 — answering it returns nothing."""
    results = [_r("a", {"DOMAIN": (0, 0)}), _r("b", {"DOMAIN": (1, 2)})]
    h = _axis_hints(results, specified=set())[0]
    assert h.range_low == 0 and h.range_high == 3
    assert [o.apply["domain"] for o in h.options] == [0, 1]


def test_options_split_a_wholly_negative_window():
    results = [_r("a", {"QUALITY": (-1, 2)}), _r("b", {"QUALITY": (0, 0)})]
    h = _axis_hints(results, specified=set())[0]
    assert h.range_low == -3 and h.range_high == 0
    assert [o.apply["quality"] for o in h.options] == [-1, 0]


# ─── Anchor hints ─────────────────────────────────────────────────


def test_universal_anchor_offers_no_choice():
    """An anchor on every result cannot split the window."""
    results = [_r("a", {}, ["chat"]), _r("b", {}, ["chat"])]
    assert _anchor_hints(results, idf={"chat": 5.0}, already=set()) == []


def test_already_constrained_anchor_is_not_re_asked():
    results = [_r("a", {}, ["chat"]), _r("b", {}, [])]
    assert _anchor_hints(results, {"chat": 5.0}, already={"chat"}) == []


def test_anchor_options_require_or_avoid():
    results = [_r("a", {}, ["tool-calling"]), _r("b", {}, [])]
    h = _anchor_hints(results, {"tool-calling": 5.0}, already=set())[0]
    assert h.present_in == 1 and h.out_of == 2
    assert [o.apply for o in h.options] == [
        {"require_anchors": ["tool-calling"]},
        {"avoid_anchors": ["tool-calling"]},
    ]


# ─── Guidance assembly ────────────────────────────────────────────


def test_no_prefer_anchors_flags_degraded_ranking():
    """The silent-mediocrity case must announce itself."""
    results = [_r("a", {"EFFICIENCY": (-1, 1)}), _r("b", {"EFFICIENCY": (1, 2)})]
    q = StructuredQuery(require_anchors=["code-generation"])
    g = build_refinement_guidance(results, q, idf={})
    assert g.ranking_degraded is True
    assert g.question_id == "ranking_degraded"


def test_prefer_anchors_present_asks_about_widest_axis():
    results = [_r("a", {"EFFICIENCY": (-1, 1)}), _r("b", {"EFFICIENCY": (1, 2)})]
    q = StructuredQuery(require_anchors=["code-generation"], prefer_anchors=["chat"])
    g = build_refinement_guidance(results, q, idf={})
    assert g.ranking_degraded is False
    assert g.question_id == "unconstrained_axis"
    assert g.unspecified_axes[0].bank == "EFFICIENCY"
    assert g.options[0].apply == {"efficiency": -1}


def test_empty_results_yield_no_guidance():
    g = build_refinement_guidance([], StructuredQuery(), idf={})
    assert g.question == "" and g.unspecified_axes == []


def test_guidance_is_deterministic():
    """Same window + query → byte-identical guidance."""
    results = [
        _r("a", {"EFFICIENCY": (-1, 1), "DOMAIN": (1, 1)}, ["x"]),
        _r("b", {"EFFICIENCY": (1, 2), "DOMAIN": (0, 0)}, ["y"]),
    ]
    q = StructuredQuery(prefer_anchors=["chat"])
    idf = {"x": 2.0, "y": 2.0}
    a = build_refinement_guidance(results, q, idf)
    b = build_refinement_guidance(results, q, idf)
    assert a == b


# ─── Unfiltered scope ─────────────────────────────────────────────


def test_no_require_anchors_flags_unfiltered_scope():
    """`require_anchors` is the only narrowing parameter; its absence means
    the candidate set was the whole corpus, and that must announce itself."""
    results = [_r("a", {}, ["chat"]), _r("b", {}, [])]
    q = StructuredQuery(prefer_anchors=["chat"])
    g = build_refinement_guidance(results, q, idf={"chat": 5.0})
    assert g.scope_unfiltered is True
    assert g.question_id == "unconstrained_query"


def test_bank_directions_do_not_narrow_scope():
    """Bank directions score the set, they never shrink it — a query carrying
    only directions is still unfiltered."""
    results = [_r("a", {"EFFICIENCY": (-1, 1)}, ["chat"]), _r("b", {}, [])]
    q = StructuredQuery(efficiency=-1, prefer_anchors=["chat"])
    g = build_refinement_guidance(results, q, idf={"chat": 5.0})
    assert g.scope_unfiltered is True


def test_require_anchors_clears_unfiltered_scope():
    results = [_r("a", {"EFFICIENCY": (-1, 1)}, ["chat"]), _r("b", {}, [])]
    q = StructuredQuery(require_anchors=["code-generation"], prefer_anchors=["x"])
    g = build_refinement_guidance(results, q, idf={})
    assert g.scope_unfiltered is False
    assert g.question_id != "unconstrained_query"


def test_unfiltered_scope_is_asked_before_degraded_ranking():
    """Filtering precedes ordering: a query missing BOTH is told to narrow
    first, because prefer_anchors cannot order what require never narrowed."""
    results = [_r("a", {}, ["chat"]), _r("b", {}, [])]
    q = StructuredQuery()
    g = build_refinement_guidance(results, q, idf={"chat": 5.0})
    assert g.scope_unfiltered is True and g.ranking_degraded is True
    assert g.question_id == "unconstrained_query"


def test_unfiltered_options_name_anchors_from_the_window():
    """Invariant 1 applies here too — an option must never point outside the
    window, or answering it would return an empty set."""
    results = [_r("a", {}, ["tool-calling"]), _r("b", {}, [])]
    g = build_refinement_guidance(results, StructuredQuery(), idf={"tool-calling": 5.0})
    assert g.options[0].apply == {"require_anchors": ["tool-calling"]}
    window_anchors = {a for r in results for a in r.anchor_labels}
    for o in g.options:
        assert set(o.apply["require_anchors"]) <= window_anchors


def test_corpus_total_sharpens_prose_but_not_the_question():
    """`corpus_total` is prose-only — it must not change which question fires."""
    results = [_r("a", {}, ["chat"]), _r("b", {}, [])]
    q = StructuredQuery()
    bare = build_refinement_guidance(results, q, idf={"chat": 5.0})
    sized = build_refinement_guidance(results, q, idf={"chat": 5.0}, corpus_total=50906)
    assert bare.question_id == sized.question_id == "unconstrained_query"
    assert "50,906 models" in sized.question
    assert "the entire corpus" in bare.question


def test_every_question_template_is_reachable():
    """Guards against a template that no code path can ever render."""
    assert set(QUESTION_TEMPLATES) == {
        "unconstrained_query",
        "ranking_degraded",
        "unconstrained_axis",
        "splitting_anchor",
    }


def test_degraded_ranking_offers_prefer_anchors_to_answer_with():
    """A question the caller cannot answer is a dead end. `ranking_degraded`
    asks for prefer_anchors, so it must hand back prefer_anchors — otherwise a
    caller driving the loop mechanically has nothing to merge and stalls."""
    results = [_r("a", {}, ["tool-calling"]), _r("b", {}, [])]
    q = StructuredQuery(require_anchors=["code-generation"])
    g = build_refinement_guidance(results, q, idf={"tool-calling": 5.0})
    assert g.question_id == "ranking_degraded"
    assert g.options, "ranking_degraded must offer answerable options"
    assert g.options[0].apply == {"prefer_anchors": ["tool-calling"]}


def test_every_question_with_hints_available_is_answerable():
    """No question type may return options=[] when the window offers hints."""
    results = [_r("a", {"DOMAIN": (0, 0)}, ["tool-calling"]), _r("b", {"DOMAIN": (1, 2)}, [])]
    idf = {"tool-calling": 5.0}
    for q in (
        StructuredQuery(),                                    # unconstrained_query
        StructuredQuery(require_anchors=["x"]),               # ranking_degraded
        StructuredQuery(require_anchors=["x"], prefer_anchors=["y"]),  # axis/anchor
    ):
        g = build_refinement_guidance(results, q, idf=idf)
        assert g.question_id, f"expected a question for {q}"
        assert g.options, f"{g.question_id} returned no answerable options"
