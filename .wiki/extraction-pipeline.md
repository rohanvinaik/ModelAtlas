---
generated: true
generated_from: 
  - docs/wiki/extraction-pipeline.md
source_hash: 9c0a4c02d5c96bad
spec_hash: ea1290146758166b
file_hash: 901925d694ca937f
materializer_version: "0.2.0"
theory_scope: false
audience: operator
page_id: extraction-pipeline
---

# Extraction Pipeline


# Extraction Pipeline

**The semantic network is built by a multi-tier pipeline that progresses from cheap deterministic rules to expensive LLM classification, with each tier auditing the one before it. The result: 19K models classified across 8 dimensions, with provenance tracking on every decision.**

---

## Problem Space

Extracting structural information from HuggingFace models is a spectrum of difficulty:

- Architecture type? Read `config.json`. Deterministic.
- Parameter count? Parse the config or the name. Pattern matching.
- "Supports tool-calling"? Read the model card, understand the context. Requires language understanding.
- "What's the vibe of this model?" Requires synthesis across all available signals.

A single extraction strategy can't span this range. Rules can't understand prose. LLMs are too expensive to run on 19K models for every field. The pipeline architecture solves this by matching extraction cost to extraction difficulty.

---

## Core Model

### The Four Phases

```
Phase A: Fetch         ─── API calls to HuggingFace, cache raw JSON
    ↓
Phase B: Extract       ─── Tier 1 (deterministic) + Tier 2 (pattern matching)
    ↓
Phase C: Classify      ─── Tier 3 (LLM-based structured extraction)
    ↓
Phase D: Audit & Heal  ─── Deterministic re-check, correction, training data
```

Each phase has explicit state tracking. Every model records which phases it has completed, when, and with what results. The pipeline is resumable — interrupt at any point, restart where you left off.

### Phase A: Fetch

Stream model metadata from HuggingFace Hub API. Enrich with `config.json` (architecture details, layer counts, attention heads) and model card text (capabilities, training data, limitations). Cache everything as raw JSON.

**Output:** {{model_count}} enriched model records.

### Phase B: Deterministic Extraction

Two tiers that require no LLM:

**Tier 1 — Deterministic.** Read `config.json` architectures field → map to ARCHITECTURE bank position. Read parameter count → map to EFFICIENCY position. Read `base_model` field → create LINEAGE links. Confidence: 1.0.

**Tier 2 — Pattern matching.** Regex and heuristic rules against model names, tags, and card text. "Instruct" in the name → `instruction-following` anchor. "GGUF" in files → `GGUF-available` anchor. ~50 patterns covering common conventions. Confidence: 0.8.

**Output:** Bank positions and anchors for all models where the information is deterministically available.

### Phase C: LLM Classification

Three sub-phases for LLM-based extraction:

**C1: Smol-Hub-tldr** — A 360M parameter model fine-tuned specifically for model card summarization. Generates one-sentence summaries from card text. Runs on CPU in ~1 hour for 6,760 models with cards.

**C2: Structured extraction** — qwen2.5:3b via Ollama generates structured JSON: a prose summary + 1-5 extra [anchor](Glossary#anchor) labels per model. Covers the full corpus. Distributed across machines via standalone workers with zero codebase imports.

**C3: Quality gate** — Blind review: the same LLM evaluates generated summaries *without access to source material*. Scores specificity, coherence, and artifact detection (each 0-3). Models below 0.5 quality score are flagged for healing.

**Design principle:** Workers are standalone single-file scripts. They read JSONL, call an Ollama API, write JSONL. They can be `scp`'d to any machine with Python and `pip install openai`. This decouples compute from the codebase.

### Phase D: Audit and Heal

A three-layer error correction pipeline:

**D1: Deterministic audit.** Re-run Tier 2 pattern matchers against C2 results. Find contradictions (C2 said X, patterns say Y), gaps (patterns found X, C2 missed it), and confidence conflicts. Produces an [audit score](Glossary#audit-score) per model.

**D2: Dictionary expansion.** Add missing anchor labels via a strict YAML DSL. Boundary-aware matchers with AND/OR semantics. Creates new anchors and auto-links high-confidence matches.

**D3: LLM healing.** For models that failed audit, generate corrected classifications from raw evidence. Corrections are stored with full provenance (original → healed, with rationale).

**D4: Training data export.** Every correction becomes a DPO training example (prompt/chosen/rejected). The pipeline improves its own future LLM by learning from its mistakes.

---

## Evidence: The Provenance Chain

Every classification decision in the network is auditable:

| Question | Where to look |
|----------|--------------|
| Where did this anchor come from? | `anchors.source` — bootstrap, deterministic, pattern, vibe, expansion |
| How confident is this link? | `model_anchors.confidence` — 1.0 for deterministic, 0.5 for LLM |
| Was this model corrected? | `correction_events` — original response, healed response, rationale |
| What phase produced this data? | `phase_d_runs` — UUID, phase, config, summary for every D-phase run |
| Did the quality gate pass? | `model_metadata` — quality_score, quality_flags |

The provenance layer makes the difference between "the system says X" and "the system says X, extracted deterministically from config.json with confidence 1.0." Trust is earned through traceability.

---

## What This Is Not

- **Not a one-shot pipeline.** It's iterative: expand corpus → extract → classify → audit → heal → export training data → repeat.
- **Not dependent on a specific LLM.** The architecture is model-agnostic. qwen2.5:3b is the current workhorse; any local model behind an OpenAI-compatible API works.
- **Not centralized.** Workers distribute across any hardware with Python and network access. No shared database connections during extraction.

---

## Related Concepts

- [Data Model](Data-Model) — what the pipeline produces
- [Audit score](Glossary#audit-score) — the D1 quality metric
- [Anchor](Glossary#anchor) — how anchor provenance tracks extraction tier
- [System Overview](System-Overview) — where the pipeline fits in the architecture

---

*[← Query Engine](Query-Engine) · [Data Model →](Data-Model)*
