"""Data types for the navigational query engine.

Separated from query.py for better cohesion — types/schemas in one module,
query logic in another. Other modules can import these without pulling in
the full query engine and its dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BankConstraint:
    """A constraint on a single semantic bank."""

    bank: str
    direction: int | None = None  # +1 or -1, None = any
    target_position: int | None = None  # specific signed position to be near
    min_signed: int | None = None
    max_signed: int | None = None
    weight: float = 1.0


@dataclass
class ParsedQuery:
    """Structured representation of a parsed natural language query."""

    bank_constraints: list[BankConstraint]
    anchor_targets: list[str]  # desired anchor labels
    seed_model_ids: list[str]  # "models like X" seeds
    direction_vectors: dict[str, int]  # bank -> desired direction (+1/-1)
    raw_tokens: list[str]  # for fuzzy fallback


@dataclass
class SearchResult:
    """A model returned from a navigational search."""

    model_id: str
    score: float
    bank_score: float = 0.0
    anchor_score: float = 0.0
    spread_score: float = 0.0
    fuzzy_score: float = 0.0
    positions: dict[str, dict] = field(default_factory=dict)
    anchor_labels: list[str] = field(default_factory=list)
    vibe_summary: str = ""
    author: str = ""


@dataclass
class ComparisonResult:
    """Result of comparing two or more models."""

    models: list[str]
    shared_anchors: list[str]
    per_model_unique: dict[str, list[str]]
    jaccard_similarity: float
    bank_deltas: dict[str, dict]


@dataclass
class StructuredQuery:
    """Structured input for navigate_models — the calling LLM fills this in."""

    # Per-bank direction: -1 (toward negative), 0 (near zero), +1 (toward positive)
    architecture: int | None = None
    capability: int | None = None
    efficiency: int | None = None
    compatibility: int | None = None
    lineage: int | None = None
    domain: int | None = None
    quality: int | None = None
    training: int | None = None

    # Anchor targeting
    require_anchors: list[str] = field(default_factory=list)
    prefer_anchors: list[str] = field(default_factory=list)
    avoid_anchors: list[str] = field(default_factory=list)

    # Seed model for similarity
    similar_to: str | None = None

    # Context carry-over — anchors the caller has already established as
    # relevant (e.g. from a prior turn or the user's stated situation).
    # Not a hard filter (that's `require_anchors`); a soft bias on
    # `anchor_relevance` so results the calling context already leans
    # toward rank higher without excluding others. Sparse-wiki's `context`
    # arm of ground(mention, context) — the SAME mention resolves
    # differently depending on the surrounding facets.
    context_anchors: list[str] = field(default_factory=list)

    # Vibe targeting (Osgood EPA — Evaluation / Potency / Activity). Each
    # is a target in [-1, +1] or None for "don't care". A candidate's
    # stored `vibe_e/p/a` is compared to the target; distance in the
    # specified axes becomes a soft multiplicative factor. Absent
    # candidate EPA (no vibe_summary or no lexicon hits) is neutral,
    # never a penalty — abstention is honest.
    vibe_e: float | None = None
    vibe_p: float | None = None
    vibe_a: float | None = None

    # Query mode — shifts the weight of soft signals (PageRank vs
    # rare-anchor specificity vs absence-bonus). `auto` derives from the
    # query's mechanical vs semantic bank mix; `canonical` favours popular
    # incumbents; `niche` favours specialists; `balanced` uses defaults.
    mode: str = "auto"

    # Optional per-bank weight overrides. Missing bank → weight 1.0
    # (default), 0 → neutralize, > 1 → amplify. Applied as exponents on
    # each bank's alignment factor so the score stays in the same
    # multiplicative shape; renormalized across active banks so total
    # attention mass is preserved.
    bank_weights: dict[str, float] | None = None

    # Result control
    limit: int = 20

    def bank_directions(self) -> dict[str, int]:
        """Return {bank_name: direction} for all specified banks."""
        mapping = {
            "ARCHITECTURE": self.architecture,
            "CAPABILITY": self.capability,
            "EFFICIENCY": self.efficiency,
            "COMPATIBILITY": self.compatibility,
            "LINEAGE": self.lineage,
            "DOMAIN": self.domain,
            "QUALITY": self.quality,
            "TRAINING": self.training,
        }
        return {k: v for k, v in mapping.items() if v is not None}


@dataclass
class NavigationResult:
    """A model returned from navigate_models."""

    model_id: str
    score: float
    bank_alignment: float = 0.0
    anchor_relevance: float = 0.0
    seed_similarity: float = 0.0
    coherence: float = 1.0
    """Multiplicative coherence factor. 1.0 = no known internal contradictions;
    0.5 = half of proposed anchors were rejected/demoted by the certifier.
    Populated per-model into `model_metadata.certification_score` by
    scripts/recertify_corpus.py; navigate() reads it as a soft tiebreaker
    so candidates that all match the constraint set are ranked by how
    well their internal evidence hangs together."""
    positions: dict[str, dict] = field(default_factory=dict)
    anchor_labels: list[str] = field(default_factory=list)
    vibe_summary: str = ""
    author: str = ""
    pagerank_boost: float = 1.0
    """Multiplicative boost applied from this model's PageRank score
    (normalized against the candidate set's max). 1.0 = no PR data or
    lowest-in-set; up to 1.4× when this model is the top-PR candidate
    AND the mode weights favour PR. Surfaced so callers can see how
    much popularity is doing vs constraint-fit."""
    soft_combined: float = 1.0
    """Multiplicative product of the submodular-combined soft signals
    (PMI-match, rare-anchor boost, absence-bonus, superadditive PR×rare).
    1.0 = no soft signal fired (basic query with no rare/absent anchors);
    up to ~1.5× when specialist signals concentrate. Callers seeing this
    close to 1.0 on an esoteric query know the scoring layer wasn't given
    enough anchor context to differentiate; close to 1.5 means the
    specialist signals cleanly identified this result as the specific fit."""
    tie_cluster_id: int | None = None
    """When a run of top results falls within `epsilon` (default 0.05) of
    each other on `score`, they form a tie-cluster: the ordering inside
    the cluster is not distinguishable from the constraints alone. Same
    integer means same cluster; `None` means the result is a singleton at
    its score band. Sparse-wiki's abstain-when-too-close discipline
    applied at query time: don't fake ordering the constraints haven't
    earned."""
    discriminating_axis: str | None = None
    """For cluster members, the position bank with the highest cross-cluster
    variance — i.e. the axis a caller could ask about to break the tie.
    `None` on singletons and on clusters whose members share every bank
    (nothing there differentiates them). Bank name, e.g. `COMPATIBILITY`
    when GGUF vs safetensors is what would split the cluster."""


@dataclass
class RefinementOption:
    """One answer to a refinement question, with the query delta it implies.

    The caller does NOT rebuild the query. It picks an option and merges
    `apply` into the arguments it already sent — see `RefinementGuidance.
    merge_rule` for the two-line merge semantics.
    """

    answer: str
    """Plain-language answer, e.g. "smaller" or "yes"."""
    apply: dict = field(default_factory=dict)
    """Query delta. Scalar keys (`efficiency`) replace; list keys
    (`require_anchors`) append to what the caller already sent."""


@dataclass
class AxisHint:
    """One unconstrained bank, and how much constraining it would help."""

    bank: str
    spread: float
    """Population variance of `sign * (1 + depth)` across the returned window.
    0.0 = every result sits at the same position on this bank, so naming a
    direction would not narrow anything."""
    range_low: int
    range_high: int
    """Min/max signed position observed in the window, e.g. -2..+3."""
    distinct: int
    """How many distinct signed positions appear. 1 = no choice to make."""
    options: list[RefinementOption] = field(default_factory=list)
    """Ready-to-merge answers, e.g. "smaller" → `{"efficiency": -1}`."""


@dataclass
class AnchorHint:
    """One anchor that splits the returned window rather than covering it."""

    anchor: str
    present_in: int
    out_of: int
    idf: float
    """Rarity weight. A rare anchor splitting the window is a sharper question
    than a common one, because committing to it excludes more of the corpus."""
    options: list[RefinementOption] = field(default_factory=list)
    """Ready-to-merge answers: "yes" → require it, "no" → avoid it."""


@dataclass
class RefinementGuidance:
    """What the caller did NOT specify, and what specifying it would buy.

    The engine is deterministic, so it knows exactly which dimensions its own
    answer is silent on. Rather than present a ranking whose tail is arbitrary,
    `navigate()` reports the axes and anchors that would actually narrow the
    result set — turning one query into an interview. Every field is derived
    from the returned window; nothing here is inferred or guessed.
    """

    unspecified_axes: list[AxisHint] = field(default_factory=list)
    """Banks with no direction in the query, ranked by `spread` descending —
    the first entry is the single most informative thing the caller could add.
    Banks where the window is uniform are omitted: asking about them is noise."""
    splitting_anchors: list[AnchorHint] = field(default_factory=list)
    """Anchors present on some but not all results, ranked by how evenly they
    split the window (weighted by IDF). A 50/50 split carries the most
    information; a 7-of-8 split barely narrows anything."""
    ranking_degraded: bool = False
    """True when the query supplied no `prefer_anchors`. Three of the five
    soft signals (PMI-match, rare-boost, superadditive) score identically for
    every candidate that clears the `require` filter, so the ranking collapses
    toward PageRank + absence. The results are still correctly FILTERED; they
    are just not meaningfully ORDERED. Callers should treat the window as a
    set, not a ranking, until they add prefer_anchors."""
    question_id: str = ""
    """Which skeleton in `QUESTION_TEMPLATES` produced `question` — e.g.
    `unconstrained_axis`. Switch on this rather than parsing the prose; the
    wording may be reworded, the id is the stable contract."""
    question: str = ""
    """One plain-language question naming the highest-value refinement,
    rendered from the `question_id` skeleton with slots filled from the
    window. Empty when nothing would narrow the results further."""
    options: list[RefinementOption] = field(default_factory=list)
    """Answers to `question`, each carrying the query delta it implies. Pick
    one, merge its `apply`, re-call. This is the whole refinement loop."""
    merge_rule: str = (
        "Merge `apply` into the arguments you already sent; do not rebuild the "
        "query. Scalar keys replace; list keys append."
    )
    """Stated in-band so a caller reading only the tool output knows how to
    apply an option without consulting the docs."""
