"""Canonical questions, and what a good answer to each looks like.

Declarative data, the same discipline as `certifier/rules.py` and
`QUESTION_TEMPLATES`: the cases are editable without touching the harness,
and every expectation is a named predicate over facts the corpus records.

**Why predicates and not expected model IDs.** A list like
`["openai/whisper-large-v3"]` rots — the corpus is a periodic snapshot of a
fast-moving hub, and a better transcription model shipping next month should
not fail the suite. What must stay true is the *shape* of a good answer: a
transcription recommendation carries `speech-domain`, an on-device Mac
recommendation is small AND Apple-native, a code-review recommendation is not
a 256M vision model. Predicates encode that and survive corpus churn.

A few cases do pin well-known IDs via `expect_any_of`, where a specific model
is so canonical that its absence really is a failure. Those are a minority and
they are advisory — see `EvalCase.expect_any_of`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .facts import (
    Predicate,
    WindowPredicate,
    has_any_anchor,
    has_anchor,
    lacks_all_anchors,
    larger_models_rank_higher,
    min_downloads,
    no_duplicate_lineage,
    not_a_vision_model,
    size_at_least,
    size_at_most,
    size_known,
    smaller_models_rank_higher,
)


@dataclass(frozen=True)
class EvalCase:
    """One canonical question and the shape of an acceptable answer."""

    name: str
    """Stable identifier. Used as the key when diffing runs, so do not rename
    a case to mean something new — add a new one."""
    ask: str
    """The user's question in plain language, for the report."""
    query: dict
    """`navigate_models` arguments — how a competent agent would decompose
    `ask`. Kept explicit rather than derived, so the eval measures the ENGINE
    rather than some particular agent's decomposition skill."""
    top_n: int = 3
    """How far down the window the expectations apply."""
    require_all: tuple[Predicate, ...] = ()
    """Every result within `top_n` must satisfy these. This is the main
    signal — it asks "is the whole window on-topic", not "is #1 lucky"."""
    window: tuple[WindowPredicate, ...] = ()
    """Assertions about the ORDERED window as a whole — ordering, spread,
    diversity. `require_all` cannot express these, and ordering is precisely
    what a filter-heavy scoring layer gets wrong: every result can be on-topic
    while the ranking among them is arbitrary."""
    expect_any_of: tuple[str, ...] = ()
    """Model-ID substrings, at least one of which should appear in `top_n`.
    Advisory: a miss is reported and scored, but these are the brittle part
    of the suite and are deliberately few."""
    notes: str = ""
    """Why this case exists — usually a defect it was written to catch."""


CASES: tuple[EvalCase, ...] = (
    EvalCase(
        name="audio_transcription",
        ask="I need to transcribe audio in a Python app.",
        query={"require_anchors": ["speech-domain"], "prefer_anchors": ["high-downloads"]},
        require_all=(
            has_anchor("speech-domain"),
            not_a_vision_model(),
        ),
        expect_any_of=("whisper",),
        notes="Baseline sanity. This one already passes; it guards against regression.",
    ),
    EvalCase(
        name="mac_on_device_summarization",
        ask="On-device summarization for a Mac app.",
        query={
            "require_anchors": ["summarization", "Apple-Silicon-native"],
            "prefer_anchors": ["edge-deployable"],
            "efficiency": -1,
        },
        require_all=(
            has_anchor("summarization"),
            has_any_anchor("Apple-Silicon-native", "MLX-compatible", "GGUF-available"),
            size_at_most(8.0),
        ),
        notes=(
            "Strongest current answer, but size_at_most will fail on any result "
            "whose parameter count is unknown — which is the zero-state defect."
        ),
    ),
    EvalCase(
        name="rag_embeddings",
        ask="Embeddings for a RAG pipeline.",
        query={
            "require_anchors": ["embedding"],
            "prefer_anchors": ["high-downloads", "community-favorite"],
        },
        require_all=(
            has_anchor("embedding"),
            not_a_vision_model(),
        ),
        notes="Already good. Guards the embedding path against anchor drift.",
    ),
    EvalCase(
        name="code_review_bot_consumer_gpu",
        ask="A code review bot. Needs tool calling, runs on one consumer GPU.",
        query={
            "require_anchors": ["tool-calling", "code-generation"],
            "prefer_anchors": ["consumer-GPU-viable", "instruction-following"],
            "efficiency": 0,
        },
        require_all=(
            has_anchor("tool-calling"),
            has_anchor("code-generation"),
            not_a_vision_model(),
            size_at_most(34.0),
            min_downloads(1000),
        ),
        notes=(
            "THE failing case. As of v0.4.2 this returns two 256M vision-language "
            "models plus GLM-4.6 (~355B) under an efficiency=0 constraint, and a "
            "1-download SFT experiment at #1. Three defects in one query: anchor "
            "over-attachment, unknown size collapsing to the ~7B zero state, and a "
            "ranking that has stopped discriminating."
        ),
    ),
    EvalCase(
        name="medical_text_classification",
        ask="Classify medical text.",
        query={
            "require_anchors": ["medical-domain", "classification"],
            "prefer_anchors": ["high-downloads"],
        },
        require_all=(
            has_anchor("medical-domain"),
            has_anchor("classification"),
            lacks_all_anchors("biology-domain"),
        ),
        notes=(
            "DNABERT-2 (genomics) currently surfaces for clinical text. The "
            "biology-domain exclusion is a proxy for that misattribution; it is "
            "the kind of judgment the local model should settle, not a rule."
        ),
    ),
    EvalCase(
        name="small_local_chat_model",
        ask="A small chat model I can run locally on a laptop.",
        query={
            "require_anchors": ["chat"],
            "prefer_anchors": ["edge-deployable", "consumer-GPU-viable"],
            "efficiency": -1,
        },
        require_all=(
            has_anchor("chat"),
            size_known(),
            size_at_most(8.0),
            not_a_vision_model(),
        ),
        notes=(
            "The most common question anyone will ask this tool, and the one the "
            "zero-state defect hits hardest: 46% of the corpus has no parameter "
            "count and is positioned as '~7B mainstream' regardless of true size."
        ),
    ),
    EvalCase(
        name="rust_systems_code",
        ask="Something that writes Rust well.",
        query={
            "require_anchors": ["Rust-code"],
            "prefer_anchors": ["code-generation", "high-downloads"],
        },
        require_all=(
            has_anchor("Rust-code"),
            has_any_anchor("code-generation", "code-completion"),
            not_a_vision_model(),
        ),
        notes="Language-specialist retrieval — the niche case the atlas should win.",
    ),
)


CASES = CASES + (
    # ── Bank-gradient cases ──────────────────────────────────────────
    # These exercise the layer fix (b) changes. Written BEFORE the fix so the
    # failure is on record and the improvement is measurable rather than
    # asserted. `larger_models_rank_higher` fails today by construction:
    # _bank_score_single is a step function for directional queries, so +1,
    # +2, +4 and +6 all score 1.000 against efficiency=+1.
    EvalCase(
        name="large_frontier_reasoning",
        ask="The most capable reasoning model I can get; compute is not a concern.",
        query={
            "require_anchors": ["reasoning"],
            "prefer_anchors": ["high-downloads"],
            "efficiency": 1,
        },
        top_n=6,
        require_all=(has_anchor("reasoning"), size_known(), size_at_least(13.0)),
        window=(larger_models_rank_higher(),),
        notes=(
            "TARGET for fix (b). Asking for large must prefer 70B to 13B. "
            "top_n=6 deliberately: over three results the assertion passed by "
            "luck, while the real window ordered sizes 31B, 120B, 14B, 12B, "
            "32B, 26B — no size ordering at all. A window too small to detect "
            "disorder is a guardrail that does not guard."
        ),
    ),
    EvalCase(
        name="tiny_edge_deployment",
        ask="Something tiny for an embedded device.",
        query={
            "require_anchors": ["edge-deployable"],
            "prefer_anchors": ["sub-1B"],
            "efficiency": -1,
        },
        top_n=6,
        require_all=(has_anchor("edge-deployable"), size_known(), size_at_most(4.0)),
        window=(smaller_models_rank_higher(),),
        notes=(
            "Mirror of the above at the other end of the bank. Currently "
            "ordered correctly, but by PageRank rather than by the bank — so "
            "it guards against fix (b) breaking what already works."
        ),
    ),
    # ── Signals the suite never exercised ────────────────────────────
    EvalCase(
        name="similar_to_a_known_model",
        ask="Something like Llama-3.1-8B-Instruct but I want alternatives.",
        query={
            "similar_to": "meta-llama/Llama-3.1-8B-Instruct",
            "require_anchors": ["instruction-following"],
            "prefer_anchors": ["chat"],
        },
        require_all=(has_anchor("instruction-following"), not_a_vision_model()),
        window=(no_duplicate_lineage(),),
        notes=(
            "seed_similarity was never exercised by any case, so it was free to "
            "regress silently. Also guards against a window of one model and its "
            "own quantisations."
        ),
    ),
    EvalCase(
        name="chat_but_not_a_vision_model",
        ask="A chat model, and I explicitly do not want a multimodal one.",
        query={
            "require_anchors": ["chat"],
            "avoid_anchors": ["image-understanding", "multimodal"],
            "prefer_anchors": ["instruction-following"],
        },
        require_all=(has_anchor("chat"), not_a_vision_model(),
                     lacks_all_anchors("multimodal")),
        notes="avoid_anchors was never exercised. It is the only downward lever.",
    ),
    EvalCase(
        name="go_specialist_over_generalist",
        ask="A model that specifically knows Go, not a general coder.",
        query={
            "require_anchors": ["Go-code"],
            "prefer_anchors": ["code-generation"],
            "mode": "niche",
        },
        require_all=(has_anchor("Go-code"),),
        notes=(
            "v0.4.1's showcase claim, pinned. `mode` was never exercised, and "
            "this is the case any popularity term would work against — surfacing "
            "the specialist over the popular generalist is the stated purpose."
        ),
    ),
    # ── Breadth: task families with no coverage ──────────────────────
    EvalCase(
        name="translation",
        ask="Translate between languages in a service.",
        query={"require_anchors": ["translation"], "prefer_anchors": ["multilingual"]},
        require_all=(has_anchor("translation"), not_a_vision_model()),
        notes="Breadth. An untested task family can rot unnoticed.",
    ),
    EvalCase(
        name="named_entity_recognition",
        ask="Pull names and organisations out of documents.",
        query={"require_anchors": ["NER"], "prefer_anchors": ["high-downloads"]},
        require_all=(has_anchor("NER"),),
        notes="Breadth — a classic encoder task, structurally unlike generation.",
    ),
    EvalCase(
        name="image_generation",
        ask="Generate images from text prompts.",
        query={"require_anchors": ["image-generation"], "prefer_anchors": ["high-downloads"]},
        require_all=(
            has_anchor("image-generation"),
            lacks_all_anchors("code-generation", "tool-calling"),
        ),
        notes=(
            "The inverse over-attachment check: a diffusion model must not "
            "claim code-generation. 399 text-to-image models currently do."
        ),
    ),
    EvalCase(
        name="structured_output_extraction",
        ask="Reliable JSON output for an extraction pipeline.",
        query={
            "require_anchors": ["structured-output"],
            "prefer_anchors": ["instruction-following", "schema-following"],
        },
        require_all=(
            has_any_anchor("structured-output", "constrained-generation"),
            not_a_vision_model(),
        ),
        notes="Breadth — the agentic-tooling family, distinct from plain chat.",
    ),
)


def case_by_name(name: str) -> EvalCase:
    for c in CASES:
        if c.name == name:
            return c
    raise KeyError(f"no eval case named {name!r}")
