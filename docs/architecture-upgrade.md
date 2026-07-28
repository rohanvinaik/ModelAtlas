# Upgrading the conceptual architecture

ModelAtlas and TriageGeist are the same architecture at two stages of
evolution. Both decompose an object into orthogonal **banks**, both refuse to
put a model in the judgment path, both keep an escalation tier for genuine
ambiguity. TriageGeist is the later draft, and the differences are not
incidental — most of what this project has been fighting is a consequence of
one of them.

This document maps the gap and sequences the upgrade. Nothing here is
implemented yet.

## The root difference: a bank emits a signal, not a coordinate

TriageGeist (`src/banks.py`):

```python
@dataclass(frozen=True)
class BankSignal:
    bank: Bank
    esi_estimate: float    # the estimate
    confidence: float      # 0.0-1.0
    esi_floor: int         # hard minimum, 0 = unconstrained
    esi_ceiling: int       # hard maximum, 6 = unconstrained
    evidence: str          # human-readable provenance
```

ModelAtlas (`model_positions`):

```
(model_id, bank, path_sign, path_depth, path_nodes, zero_state)
```

A position and nothing else. **No confidence, no bounds, no provenance.**
Nearly every defect found in this session is downstream of that:

| symptom | root |
|---|---|
| 23,352 models "at ~7B" because size was unknown | no confidence — a guess is written like a fact |
| a 256M VLM carrying `code-generation`, `tool-calling` | no provenance — an anchor scraped from card boilerplate is indistinguishable from one derived from `pipeline_tag` |
| `coherence` factor dead at 99.95% = 1.0 | nothing to be coherent *about*; one estimator per bank cannot dissent |
| certifier rules feel brittle and vibes-based | with no confidence or provenance, a rule is the only place left to put the knowledge |

`BankPosition.known` (just landed) is a 1-bit approximation of `confidence`.
It fixed the specific defect and proved the shape is right.

## What "Kuramoto coherence" needs, and what it is not

The order parameter is a **vector sum on the circle**:

$$r e^{i\psi} = \frac{1}{N}\sum_j e^{i\theta_j}$$

`r` falls out of *cancellation* — antiphase oscillators subtract, a uniform
scatter gives r ≈ 0, alignment gives r = 1. A scalar mean of magnitudes has
none of that structure and is not a coherence measure. (Recorded because this
document's author tried exactly that and misreported the result.)

It also needs the right oscillators: **several estimators of ONE quantity.**
TriageGeist has that — 11 banks each estimating ESI, phase-embedded on a
half-circle so the extremes cannot wrap onto each other.

A naive port fails here, and it is worth writing down why: ModelAtlas's 8
banks are **orthogonal by construction**. Small *and* code-focused *and*
novel-architecture is not dissent — there is no consensus phase to deviate
from. Measured on the shipped corpus, an order parameter over a model's 8 bank
positions gives r ∈ [0.91, 0.98] for known-good and known-bad models alike. It
measures nothing.

The oscillators are one level down: **per bank, the several independent
evidence sources that each imply a position on it.**

```
EFFICIENCY   <- parameter_count_b | size anchors (sub-1B, 7B-class, frontier-class)
                | quantization tags | safetensors_info
CAPABILITY   <- pipeline_tag | capability anchors | config architecture
DOMAIN       <- domain anchors | pipeline_tag | vibe summary
```

Those estimate a common quantity, so they can genuinely dissent.
`SmolVLM-256M`: `pipeline_tag=image-text-to-text` implies one CAPABILITY
position; anchors claiming `code-generation` / `tool-calling` / `reasoning`
imply a very different one. Two phasors far apart on one circle → low `r`.
`Wan2.1-T2V-1.3B` tagged `translation` is the same shape, and so is
`DNABERT-2` tagged `medical-domain`.

**Embedding.** Bank positions are signed, roughly [−4, +4], and the sign is
meaningful. So θ ∈ [0, π] with the zero state at π/2 and the two directions
genuinely opposed — "one source says sub-1B, another says frontier" becomes
antiphase, which is the cancellation the measure exists to produce. Half-circle
for the same reason TriageGeist uses one: a full circle would wrap the extremes
back onto each other.

**Note on the source.** TriageGeist implements this pattern twice.
`src/coherence.py` uses confidence-weighted variance with a stepped boost table
(convergence → nonlinear confidence gain, divergence → penalty). The true order
parameter — `z = Σ e^{iθ}w / Σw`, `r = |z|`, `ψ = arg z`, plus per-bank signed
deviation `θ_i − ψ` and subset order parameters — lives in
`src/feature_engine.py`. Their `CLAUDE.md` records that the per-bank deviation
*features* were benchmarked and **removed** from the ensemble (easy Δ=+0.0000,
p=1.00; hard Δ=−0.0025, d=−0.68) while the magnitudes still render into the LLM
audit context. So the order parameter earned its keep as a **routing and audit**
signal, not as a predictive feature. We should expect the same here and measure
rather than assume.

## What else evolved

**Tiered resolution with provenance.** TriageGeist's `TriageDecision` carries
`method: "rules" | "coherence" | "model" | "llm"` and an `evidence` trail; each
scale resolves what it can and passes residuals up. ModelAtlas has one
monolithic multiplicative product and a `NavigationResult` that reports score
components but never says *which mechanism* decided.

**Escalation gated on dissent.** The LLM fires only where the top-2 are within
0.20 *and* the banks are in dissent — 29 patients in 20,000. ModelAtlas already
owns the best version of this idea in the `refine` loop: when the engine cannot
justify an ordering it asks one question instead of faking a rank. Today that
fires on *structural* gaps (no `require_anchors`, no `prefer_anchors`). With a
coherence signal it could also fire on *evidential* ones — "the sources
disagree about what this model is."

**Bank-supplied floor/ceiling.** Signals carry hard bounds that clamp the
consensus. ModelAtlas's certifier has `requires`/`forbids` but they act at
write time, not as query-time constraints on a bank.

## Sequence

Each step is measurable against `python -m model_atlas.evaluation` and lands
separately.

1. **Give bank signals confidence and provenance.** Widen `model_positions`
   (or a sibling table) with `confidence` and `evidence`, generalising
   `BankPosition.known`. Extraction already knows which tier produced each
   value — `deterministic` / `pattern` / `vibes` / `expansion` is recorded on
   *anchors* today and thrown away for positions.
2. **Per-bank evidence sources → an order parameter.** Compute `r` and `ψ` per
   (model, bank) from the sources listed above. Read-only at first: a new check
   in `coherence.py` reporting the corpus's low-`r` population, verified
   against known-bad models before anything consumes it.
3. **Consume `r` where the dead factor is.** `coherence` in the scoring product
   is 1.0 for 99.95% of the corpus. Replace it with the order parameter, and
   measure — this is the step most likely to move the eval, and most likely to
   move it *down* first for the reasons in `navigation.md`.
4. **Route dissent instead of ruling on it.** Low `r` marks the residue for the
   local model (task #5) rather than a hand-written forbid list. This is the
   step that retires "brittle rules based on vibes".
5. **Decision provenance.** `NavigationResult` gains `method` and an evidence
   trail, so a result can say whether a filter, the scalar, or a coherence
   judgment put it there.

Steps 1–2 are prerequisites and carry no behaviour change. Step 3 is where the
architecture actually shifts.

## What this does not change

The separation the last upgrade established stands: **navigation filters,
the scalar ranks.** Coherence is a third thing — a statement about how much
the corpus's own evidence agrees, which belongs to confidence and routing, not
to the ternary coordinates or the ordering scalar. Folding it into either
would repeat the mistake `navigation.md` documents.
