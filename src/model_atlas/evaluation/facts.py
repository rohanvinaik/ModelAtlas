"""The read-only fact bundle a predicate may inspect, and the predicates.

Deliberately narrow, in the same spirit as the certifier's `HFFacts`: an
assertion may only reason about things the corpus actually records, so a
failing check always points at a specific missing or wrong fact rather than
at a vague sense that the answer looked bad.

Predicates are *named* objects rather than bare lambdas so a failure report
can say `size_at_most(14B)` instead of `<function <lambda>>`.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelFacts:
    """What the corpus knows about one returned model."""

    model_id: str
    anchors: frozenset[str] = frozenset()
    positions: dict[str, tuple[int, int]] = field(default_factory=dict)
    param_count_b: float | None = None
    pipeline_tag: str = ""
    downloads: int | None = None


def load_facts(conn: sqlite3.Connection, model_id: str) -> ModelFacts:
    """Fetch the fact bundle for one model. Read-only."""
    anchors = {
        r[0]
        for r in conn.execute(
            """SELECT a.label FROM model_anchors ma
                 JOIN anchors a ON a.anchor_id = ma.anchor_id
                WHERE ma.model_id = ?""",
            (model_id,),
        )
    }
    positions = {
        r[0]: (r[1], r[2])
        for r in conn.execute(
            "SELECT bank, path_sign, path_depth FROM model_positions WHERE model_id = ?",
            (model_id,),
        )
    }
    meta = {
        r[0]: r[1]
        for r in conn.execute(
            """SELECT key, value FROM model_metadata
                WHERE model_id = ?
                  AND key IN ('parameter_count_b', 'pipeline_tag', 'downloads')""",
            (model_id,),
        )
    }

    def _num(key: str) -> float | None:
        try:
            return float(meta[key])
        except (KeyError, TypeError, ValueError):
            return None

    dl = _num("downloads")
    return ModelFacts(
        model_id=model_id,
        anchors=frozenset(anchors),
        positions=positions,
        param_count_b=_num("parameter_count_b"),
        pipeline_tag=meta.get("pipeline_tag") or "",
        downloads=int(dl) if dl is not None else None,
    )


@dataclass(frozen=True)
class Predicate:
    """A named, explainable assertion about one model."""

    name: str
    test: Callable[[ModelFacts], bool]

    def __call__(self, facts: ModelFacts) -> bool:
        return self.test(facts)


# ── Anchor predicates ────────────────────────────────────────────────


def has_anchor(label: str) -> Predicate:
    return Predicate(f"has:{label}", lambda f: label in f.anchors)


def lacks_anchor(label: str) -> Predicate:
    return Predicate(f"lacks:{label}", lambda f: label not in f.anchors)


def has_any_anchor(*labels: str) -> Predicate:
    wanted = frozenset(labels)
    return Predicate(
        f"has_any:{'|'.join(labels)}", lambda f: bool(wanted & f.anchors)
    )


def lacks_all_anchors(*labels: str) -> Predicate:
    unwanted = frozenset(labels)
    return Predicate(
        f"lacks_all:{'|'.join(labels)}", lambda f: not (unwanted & f.anchors)
    )


# ── Size predicates ──────────────────────────────────────────────────
#
# These fail on an UNKNOWN size rather than passing it. That is the point:
# 46% of the corpus has no parameter_count_b, and the extractor writes the
# EFFICIENCY zero state ("~7B") anyway, so a frontier model satisfies a
# "small model" query. An eval that let unknown pass would score that as
# fine and hide the very defect we are trying to measure.


def size_known() -> Predicate:
    return Predicate("size_known", lambda f: f.param_count_b is not None)


def size_at_most(billions: float) -> Predicate:
    return Predicate(
        f"size<={billions}B",
        lambda f: f.param_count_b is not None and f.param_count_b <= billions,
    )


def size_at_least(billions: float) -> Predicate:
    return Predicate(
        f"size>={billions}B",
        lambda f: f.param_count_b is not None and f.param_count_b >= billions,
    )


# ── Shape predicates ─────────────────────────────────────────────────


def pipeline_in(*tags: str) -> Predicate:
    wanted = frozenset(tags)
    return Predicate(
        f"pipeline_in:{'|'.join(tags)}", lambda f: f.pipeline_tag in wanted
    )


def not_a_vision_model() -> Predicate:
    """Excludes image understanding/generation.

    A 256M vision-language model carrying `code-generation` and `tool-calling`
    is the canonical over-attachment case — the anchors came from model-card
    boilerplate, not from anything the model does.
    """
    return lacks_all_anchors("image-understanding", "image-generation")


@dataclass(frozen=True)
class WindowPredicate:
    """An assertion about the ORDERED window, not about one model.

    Per-result predicates ask "is everything here on-topic". They cannot ask
    "is it in a sensible order" — and ordering is exactly what a filter-heavy
    scoring layer gets wrong. A window can be entirely on-topic and still
    ranked arbitrarily.
    """

    name: str
    test: Callable[[list[ModelFacts]], bool]

    def __call__(self, window: list[ModelFacts]) -> bool:
        return self.test(window)


def _known_sizes(window: list[ModelFacts]) -> list[float]:
    return [f.param_count_b for f in window if f.param_count_b is not None]


def larger_models_rank_higher() -> WindowPredicate:
    """Asking for large should put the largest first.

    Fails today: `_bank_score_single` is a step function for directional
    queries, so +1, +2, +4 and +6 all score 1.000 against `efficiency=+1` and
    "large" cannot prefer 70B to 13B. See docs/scoring-dynamic-range.md.
    """

    def test(window: list[ModelFacts]) -> bool:
        sizes = _known_sizes(window)
        if len(sizes) < 2:
            return False
        return sizes[0] >= max(sizes)

    return WindowPredicate("largest_ranks_first", test)


def smaller_models_rank_higher() -> WindowPredicate:
    """The mirror: asking for small should put the smallest first."""

    def test(window: list[ModelFacts]) -> bool:
        sizes = _known_sizes(window)
        if len(sizes) < 2:
            return False
        return sizes[0] <= min(sizes)

    return WindowPredicate("smallest_ranks_first", test)


def sizes_span_at_least(factor: float) -> WindowPredicate:
    """The window should not be N near-identical models.

    A recommendation of five 7B models dressed as a ranking is less useful
    than a spread the caller can choose from — and a collapsed spread is a
    symptom of a scoring layer that stopped discriminating.
    """

    def test(window: list[ModelFacts]) -> bool:
        sizes = _known_sizes(window)
        return len(sizes) >= 2 and max(sizes) >= min(sizes) * factor

    return WindowPredicate(f"size_span>={factor}x", test)


def no_duplicate_lineage() -> WindowPredicate:
    """Distinct recommendations, not one model and its quantisations.

    `Daredevil-8B` and `Daredevil-8B-abliterated` occupying two of three slots
    is a window of one answer wearing two hats.
    """

    def test(window: list[ModelFacts]) -> bool:
        stems = set()
        for f in window:
            name = f.model_id.split("/")[-1].lower()
            for suffix in ("-gguf", "-mlx", "-awq", "-gptq", "-abliterated",
                           "-instruct", "-base", "-4bit", "-8bit", "-int8"):
                name = name.replace(suffix, "")
            stems.add(name)
        return len(stems) == len(window)

    return WindowPredicate("no_duplicate_lineage", test)


def min_downloads(n: int) -> Predicate:
    """A weak popularity floor. Not a quality signal on its own, but a model
    with single-digit downloads outranking an established one is a symptom of
    a ranking that has stopped discriminating."""
    return Predicate(
        f"downloads>={n}", lambda f: f.downloads is not None and f.downloads >= n
    )
