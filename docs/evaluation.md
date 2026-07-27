# Corpus-quality evaluation

`python -m model_atlas.evaluation`

## Why this exists

Every other check in the repo tests *mechanism*: that chunking is correct, that a
refinement option never points outside its window, that a template has no unfilled slot.
None of them can tell you whether the answers are any good. 968 passing tests coexisted
with a query that returned two 256M vision-language models for "a code review bot."

This is the instrument for that question. It is the thing corpus work is measured
against — dropping audit contradictions, correcting the EFFICIENCY zero state, adding
certifier rules, running the local model over the residue. Without a number, "the corpus
got better" is a feeling.

It earned its keep immediately. Two hypotheses about why answers were bad — that unknown
sizes collapsing into the "~7B" zero state was broadly degrading results, and that the
failing query had too thin a candidate set — were both **wrong**, and the harness showed
it in minutes. See [What it found first](#what-it-found-first).

## Running it

```bash
python -m model_atlas.evaluation                             # report
python -m model_atlas.evaluation --json                      # machine-readable
python -m model_atlas.evaluation --verbose                   # show passing results too
python -m model_atlas.evaluation --case code_review_bot_consumer_gpu
python -m model_atlas.evaluation --save-baseline data/eval/before.json
python -m model_atlas.evaluation --baseline data/eval/before.json
python -m model_atlas.evaluation --min-score 0.9             # non-zero exit below floor
```

Read-only. It runs `navigate()` exactly as the MCP tool does and inspects the window;
it never writes.

## How a case is written

A case pairs a plain-language ask with the *shape* of an acceptable answer:

```python
EvalCase(
    name="mac_on_device_summarization",
    ask="On-device summarization for a Mac app.",
    query={"require_anchors": ["summarization", "Apple-Silicon-native"],
           "prefer_anchors": ["edge-deployable"], "efficiency": -1},
    require_all=(
        has_anchor("summarization"),
        has_any_anchor("Apple-Silicon-native", "MLX-compatible", "GGUF-available"),
        size_at_most(8.0),
    ),
)
```

**Predicates, not expected model IDs.** A list like `["openai/whisper-large-v3"]` rots —
the corpus is a periodic snapshot of a fast-moving hub, and a better model shipping next
month should not fail the suite. What must stay true is the shape: a transcription
recommendation carries `speech-domain`; a Mac on-device recommendation is small *and*
Apple-native; a code-review recommendation is not a 256M vision model. `expect_any_of`
exists for the few genuinely canonical IDs, and is deliberately rare.

**`query` is written out, not derived.** The eval measures the engine, not some
particular agent's decomposition skill.

**Size predicates fail on unknown.** `size_at_most(8.0)` is false when
`parameter_count_b` is absent. 46% of the corpus has no parameter count and is
positioned at the "~7B mainstream" zero state regardless of true size — an eval that let
unknown pass would score that as fine and hide the defect.

## Scoring

A case's score is the fraction of (result, predicate) pairs that hold across the top-N
window, plus one point per satisfied `expect_any_of`. The headline number aggregates
over **checks, not cases**, so a case with many expectations weighs more than one with
few and adding an easy case cannot flatter the average by dilution.

Scoring the whole window rather than the top hit is deliberate: scoring `#1` alone calls
a window good when `#2` and `#3` are junk, which is precisely the failure mode here.

An empty window scores **zero**, not a vacuous 1.0 — returning nothing is a failure to
answer, not a set of satisfied constraints.

Blunt on purpose. Weighting predicates by importance invites tuning the metric instead
of the corpus.

## What it found first

Baseline on the v0.4.2 corpus: **92.5%** (62/67 checks), with a single failing case.

Six of seven cases pass, and the passes are real — `Qwen/Qwen3-0.6B` for "a small chat
model for my laptop", `gemma-3-270m-it-MLX-4bit` for Mac summarization, `bge-m3` and
`jina-embeddings-v3` for RAG. The one failure, `code_review_bot_consumer_gpu`, then
falsified two plausible diagnoses:

- **"Unknown sizes collapsing to the zero state degrade size queries."** They do not, in
  general. Querying `chat` at `efficiency` −1/0/+1 returns 1.1–4.0B / 6–8B / 12–35B with
  almost no unknowns. Anchors like `1B-class` carry the size signal redundantly and
  genuinely-sized models outcompete the unknowns.
- **"The failing query has too thin a candidate set."** It has 1,749 candidates.

The actual cause, from `score_breakdown` on that query:

| model | score | bank | anchor | seed | coherence | soft | downloads |
|---|---|---|---|---|---|---|---|
| `Shekswess/trlm-stage-2-sft-final-2` | 1.741 | 1.000 | 1.000 | 1.00 | 1.00 | 1.741 | 1 |
| `zai-org/GLM-4.6` | 1.728 | 1.000 | 1.000 | 1.00 | 1.00 | 1.728 | 81,982 |
| `SmolVLM2-256M-Video-Instruct` | 1.726 | 1.000 | 1.000 | 1.00 | 1.00 | 1.726 | 130,665 |

**Every filter factor is exactly 1.000 for every candidate**, so `final_score ==
soft_combined` and six of the seven multiplicative factors contribute nothing. Within
`soft_combined` the PageRank term spans 1.0029–1.0606 across a 130,000× spread in
downloads. Total dynamic range ≈1.5%, which is noise — so ordering among qualified
candidates is effectively arbitrary, and a 1-download experiment outranks
`Phi-4-mini-instruct`. Every query mode reproduces it; `canonical`, which is meant to
amp PageRank, still places a 1-download model fifth with the same score as the top hit.

`bank_alignment = 1.000` for GLM-4.6 under `efficiency=0` is the zero-state defect in its
purest form: with an unknown parameter count the model sits at EFFICIENCY `(0,0)`, so it
*perfectly* matches a query for the zero state. Real, but narrower than first claimed —
it corrupts `efficiency=0` queries rather than size queries generally.

## Adding a case

Add to `CASES` in `evaluation/cases.py`. Write it against a defect you have actually
observed and say so in `notes` — a case that never fails teaches nothing, and a case
added to raise the score is worse than none. Then re-baseline.
