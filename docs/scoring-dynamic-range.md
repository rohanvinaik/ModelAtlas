# Why the ranking does not rank

A diagnosis, measured on the v0.4.2 corpus. It explains why
`code_review_bot_consumer_gpu` returns a 1-download experiment above
`Phi-4-mini-instruct`, and why the fix is not corpus cleanup.

## The observation

`navigate()` composes seven multiplicative factors. On a real query
(`require=[tool-calling, code-generation]`, `prefer=[consumer-GPU-viable,
instruction-following]`, `efficiency=0`):

| model | score | bank | anchor | seed | coherence | soft | downloads |
|---|---|---|---|---|---|---|---|
| `Shekswess/trlm-stage-2-sft-final-2` | 1.741 | 1.000 | 1.000 | 1.00 | 1.00 | 1.741 | 1 |
| `Jackrong/Qwen3.5-9B-…-Distil` | 1.725 | 1.000 | 1.000 | 1.00 | 1.00 | 1.725 | 18,679 |
| `zai-org/GLM-4.6` | 1.728 | 1.000 | 1.000 | 1.00 | 1.00 | 1.728 | 81,982 |
| `SmolVLM-256M-Instruct-GGUF` | 1.740 | 1.000 | 1.000 | 1.00 | 1.00 | 1.740 | 19,239 |
| `SmolVLM2-256M-Video-Instruct` | 1.726 | 1.000 | 1.000 | 1.00 | 1.00 | 1.726 | 130,665 |

Six of seven factors are exactly `1.000` for every candidate, so
`final_score == soft_combined`. Total spread ≈1.5% — noise. Ordering among
qualified candidates is therefore arbitrary. Every query mode reproduces it,
including `canonical`, which exists to make mainstream models rise.

## Why each factor is flat

Three different causes, needing three different fixes. Lumping them together
is how two earlier hypotheses died.

### Neutral by design — correct, not a defect

`seed_similarity`, `context_bias`, `epa_alignment` return `1.0` when the query
supplies no `similar_to`, no `context_anchors`, no vibe target. An unused
signal should be neutral. Nothing to fix.

### Degenerate data

`coherence` is the certifier's `certification_score`. Its distribution:

```
1.0000   50,883      <- 99.95%
0.8889       11
0.9444        2
...
```

A multiplicative factor that is `1.0` for 99.95% of the corpus carries no
information. This is not saturation — the data itself is constant, because the
certifier passes essentially everything. Either the rules must discriminate or
the factor is dead weight in the product.

### Saturating filters — the structural problem

`bank_alignment` and `anchor_relevance` are both bounded at `1.0`. They can
**penalize a mismatch but never distinguish two full matches.**

`_nav_anchor_relevance` returns `matched_prefer_idf / total_prefer_idf`, so any
model carrying every `prefer_anchor` scores exactly `1.0`. Beyond that it is
blind: matching two rare prefers and matching two common ones score the same.

`_bank_score_single` is worse — for a *directional* query it is a step
function:

```
query efficiency=+1 (want large):
   model at +1  -> 1.000
   model at +2  -> 1.000
   model at +4  -> 1.000
   model at +6  -> 1.000

query efficiency=0 (want ~7B):
   model at +0  -> 1.000
   model at +1  -> 0.500
   model at +2  -> 0.333
```

Only `direction == 0` has a gradient. This contradicts the stated design in
`docs/DESIGN.md` §2.1 — *"Signed positions give gradient scoring — a 3B model
is close to 7B, not a binary mismatch"* — which holds for the zero state and
nowhere else. Asking for "large" cannot prefer 70B over 13B.

And `_bank_score_single(0, 0) == 1.0` is where the EFFICIENCY zero-state defect
finally bites: 32,697 models sit at `(0,0)`, **71% of them only because their
parameter count is unknown**, and every one scores a perfect `1.0` against
`efficiency=0`.

### The one live term, crushed

`soft_combined` is the only factor that rewards rather than filters. Its
strongest component is PageRank, and `pr_frac = pagerank / max(pagerank)`
destroys it — PageRank on a scale-free graph is power-law, so dividing by the
maximum pins almost everything at zero:

```
corpus pr_frac:  p50=0.001547   p90=0.004176   p99=0.026052
96.9% of models have pr_frac < 0.01  ->  contribute <0.2% of final score

meta-llama/Llama-3.1-8B-Instruct   pr_frac=0.305   K_PR*frac=0.061
zai-org/GLM-4.6                    pr_frac=0.032   K_PR*frac=0.006
deepseek-ai/DeepSeek-V3            pr_frac=0.029   K_PR*frac=0.006
Shekswess/trlm-stage-2-sft-final-2 pr_frac=0.003   K_PR*frac=0.001
```

DeepSeek-V3, with 1.4M downloads, gets a 0.6% boost. The *data* has 647× of
range between the median and the maximum; the *transform* throws it away.

## The shape of it

**Six of seven factors are filters.** Filters remove candidates; they do not
order them. Once a candidate clears every filter at `1.0`, nothing separates it
from any other that also cleared. The architecture has one ranking signal, and
that signal is deliberately compressed (submodular, decay 0.7) to stop any
single term running away.

That compression is a reasonable instinct applied to the wrong layer: it damps
the only term that could discriminate, while the six that cannot are left
untouched.

## Fixes, in order of value

1. **Renormalize PageRank.** Log or rank-percentile instead of `/max`.
   Contained, no contract change, and the biggest single recovery of dynamic
   range. Everything else is downstream of this.
2. **Give `bank_alignment` a gradient for directional queries** so depth
   matters and "large" prefers 70B to 13B. Restores the documented intent.
3. **Decide what `coherence` is for.** Either the certifier discriminates or
   the factor leaves the product. Multiplying by a constant is not a signal.
4. **Leave `anchor_relevance` as a filter.** The reward it lacks already
   exists in the soft layer as `rare_boost`; adding a second one invites
   double-counting. Fix the soft side instead.

Not on this list: corpus cleanup. Dropping bad anchors makes the *candidate
set* cleaner, which is worth doing on its own merits — but with a ~1.5% score
spread you would still be ordering the cleaned set arbitrarily.

## Guardrail

The eval suite has 7 cases / 67 checks, which is too thin to catch a
regression in a change to the scoring contract v0.4.1 was built on. Widen it
before touching any of the above, including cases that currently pass, so a
fix cannot quietly break what already works.
