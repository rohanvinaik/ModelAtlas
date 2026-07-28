"""Cross-source coherence per bank — the Kuramoto order parameter.

Step 2 of `docs/architecture-upgrade.md`.

A bank position is `[SIGN][DEPTH]`, and several *independent* facts can each
imply one. When they agree the extraction is corroborated; when they dissent,
something is wrong and the system should say so rather than pick a winner.
That is measured with the order parameter from coupled-oscillator theory:

    r·e^(iψ) = (1/N) · Σ e^(iθ_j)

`r` falls out of **cancellation** — antiphase estimators subtract, so `r → 0`
means the sources point opposite ways, and `r = 1` means they agree exactly. A
scalar average of the estimates cannot express that; the phase geometry is the
whole mechanism.

**Choosing oscillators is the hard part, and most candidates are invalid.**
Three failed on this corpus before this one worked:

- The 8 bank positions of one model. They are *orthogonal by construction* —
  small AND code-focused is not dissent — so there is no consensus phase to
  deviate from. Measured `r ∈ [0.91, 0.98]` for known-good and known-bad alike.
- Mean pairwise anchor PMI. Not an order parameter at all: a scalar mean has
  no circle and no cancellation, so `+2` and `−2` average to the same `0` as
  two zeros.
- Anchor surprisal conditioned on `pipeline_tag`. Fails because the
  contamination is *systematic*: 2,117 image-text-to-text models carry
  generative capability anchors, so within that tag it is typical. You cannot
  find outliers when the error is the norm.

The rule those failures teach: **oscillators must be independently authored
and must not include the thing being judged.** The anchor set is the output
under audit; it cannot also be a voter.

EFFICIENCY satisfies this today with two genuinely independent estimators:

1. `parameter_count_b` — extracted at ingest from safetensors or the model name
2. the config geometry — `num_layers`, `hidden_size`, `intermediate_size`,
   `vocab_size`, from which parameter count is *computable*

Different provenance, different failure modes, neither derived from the
anchors. Their disagreement is diagnostic rather than merely noisy — see
`check_efficiency_coherence`.

Read-only. Nothing here writes, and nothing consumes `r` yet; wiring it into
the scoring product is step 3.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import Any

# Parameter-count buckets, mirroring `_PARAM_RANGES` in extraction. Kept as a
# signed position rather than a bucket name so both estimators land in the same
# coordinate space before being embedded as phases.
_SIZE_BUCKETS: tuple[tuple[float, float, int], ...] = (
    (0.0, 0.5, -3),
    (0.5, 1.5, -2),
    (1.5, 5.0, -1),
    (5.0, 10.0, 0),
    (10.0, 20.0, 1),
    (20.0, 50.0, 2),
    (50.0, 100.0, 3),
    (100.0, float("inf"), 4),
)

POSITION_LIMIT = 4
"""Signed positions are clamped here before embedding. The banks do not run
deeper, and an outlier must not wrap past the far end of the half-circle."""

DISSENT_THRESHOLD = 0.95
"""`r` below this counts as dissent. Chosen from the measured distribution —
on the v0.4.2 corpus the median is 1.000 and the 10th percentile 0.981, so
0.95 selects a genuine tail (6.9%) rather than ordinary rounding noise."""


def size_to_position(billions: float) -> int | None:
    """Signed EFFICIENCY position for a parameter count, or None if unplaceable."""
    if billions <= 0:
        return None
    for low, high, position in _SIZE_BUCKETS:
        if low <= billions < high:
            return position
    return None


def position_to_phase(position: int) -> float:
    """Embed a signed bank position as a phase in [0, π].

    A HALF circle, deliberately, with the zero state at π/2. The two
    directions then sit at opposite ends and genuinely cancel — "one source
    says sub-1B, another says frontier" is antiphase, which is the whole point.
    A full circle would wrap the extremes back onto each other and make the
    most severe disagreement look like agreement.
    """
    clamped = max(-POSITION_LIMIT, min(POSITION_LIMIT, position))
    return (clamped + POSITION_LIMIT) / (2 * POSITION_LIMIT) * math.pi


def order_parameter(
    phases: list[float], weights: list[float] | None = None
) -> tuple[float, float]:
    """Kuramoto order parameter `(r, ψ)` over confidence-weighted phasors.

    `r = |Σ w·e^(iθ) / Σ w|` in [0, 1], `ψ` the consensus phase. A source with
    zero weight contributes no phasor at all rather than a neutral one — an
    absent fact must not vote, which matters here because coverage is uneven
    (only 72% of the corpus has a `pipeline_tag`, for instance).

    Fewer than two contributing phasors returns `(0.0, 0.0)`: one estimator
    cannot agree or disagree with anything, and reporting 1.0 would read as
    perfect corroboration.
    """
    w = weights if weights is not None else [1.0] * len(phases)
    pairs = [(t, wi) for t, wi in zip(phases, w) if wi > 0]
    if len(pairs) < 2:
        return 0.0, 0.0
    total = sum(wi for _, wi in pairs)
    real = sum(math.cos(t) * wi for t, wi in pairs) / total
    imag = sum(math.sin(t) * wi for t, wi in pairs) / total
    return math.hypot(real, imag), math.atan2(imag, real)


def params_from_geometry(
    num_layers: float, hidden: float, intermediate: float, vocab: float
) -> float:
    """Parameter count in billions, computed from transformer geometry.

    ``L·(4h² + 3·h·i) + 2·V·h`` — attention projections, an MLP with a gate,
    and untied embeddings. Deliberately approximate: it ignores tied
    embeddings, GQA's reduced KV projections, biases and norms, and runs about
    13% high at the median as a result.

    That bias is acceptable because this is a *second opinion*, not a
    measurement. It is compared bucket-to-bucket, and a 13% error moves a model
    across a bucket boundary only when it already sits on one — whereas the
    disagreements this exists to find span three or more buckets.
    """
    return (num_layers * (4 * hidden * hidden + 3 * hidden * intermediate)
            + 2 * vocab * hidden) / 1e9


@dataclass(frozen=True)
class CoherenceFinding:
    """One model whose sources disagree about a bank."""

    model_id: str
    bank: str
    r: float
    positions: dict[str, int]
    evidence: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "bank": self.bank,
            "r": round(self.r, 4),
            "positions": self.positions,
            "evidence": {k: round(v, 4) for k, v in self.evidence.items()},
        }


def _numeric_metadata(conn: sqlite3.Connection, keys: tuple[str, ...]) -> dict:
    out: dict[str, dict[str, float]] = {}
    placeholders = ",".join("?" for _ in keys)
    rows = conn.execute(
        f"SELECT model_id, key, value FROM model_metadata WHERE key IN ({placeholders})",
        keys,
    ).fetchall()
    for model_id, key, value in rows:
        try:
            out.setdefault(model_id, {})[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out


_EFFICIENCY_KEYS = (
    "parameter_count_b",
    "num_layers",
    "hidden_size",
    "intermediate_size",
    "vocab_size",
)


def check_efficiency_coherence(
    conn: sqlite3.Connection,
    *,
    threshold: float = DISSENT_THRESHOLD,
    limit: int = 200,
) -> list[CoherenceFinding]:
    """Models whose two EFFICIENCY estimators dissent, worst first.

    Read-only. Skips any model lacking either estimator — an absent source
    does not vote, so it produces no finding rather than a false one.

    The dissent it surfaces is of two genuinely different kinds, and the check
    deliberately does not adjudicate between them:

    * **extraction bugs** — `amd/AMD-Llama-135m` stored as 627B against a
      config implying 0.13B; `abacaj/llama-161M-100B` stored as 100B because
      the name's "100B" is a count of *training tokens*, not parameters.
    * **quantised artifacts** — `unsloth/Mistral-Large-…-bnb-4bit` reads 147B
      from config and 4B from storage. Both answer a real question, about
      different objects.

    Telling those apart is judgment, and judgment is what a low `r` routes
    toward. Flagging the disagreement is this function's whole job.
    """
    facts = _numeric_metadata(conn, _EFFICIENCY_KEYS)
    findings: list[CoherenceFinding] = []

    for model_id, f in facts.items():
        stored = f.get("parameter_count_b")
        if stored is None:
            continue
        geometry = ("num_layers", "hidden_size", "intermediate_size", "vocab_size")
        if not all(k in f for k in geometry):
            continue
        computed = params_from_geometry(
            f["num_layers"], f["hidden_size"], f["intermediate_size"], f["vocab_size"]
        )
        stored_pos = size_to_position(stored)
        computed_pos = size_to_position(computed)
        if stored_pos is None or computed_pos is None:
            continue

        r, _ = order_parameter(
            [position_to_phase(stored_pos), position_to_phase(computed_pos)]
        )
        if r < threshold:
            findings.append(
                CoherenceFinding(
                    model_id=model_id,
                    bank="EFFICIENCY",
                    r=r,
                    positions={"stored": stored_pos, "computed": computed_pos},
                    evidence={"parameter_count_b": stored, "config_geometry_b": computed},
                )
            )

    findings.sort(key=lambda x: (x.r, x.model_id))
    return findings[:limit]
