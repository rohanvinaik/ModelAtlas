# Phase C Intelligence Extraction Pipeline

Technical reference for ModelAtlas's multi-tier intelligence extraction pipeline. Phase C follows the completed fetch (Phase A) and deterministic extraction (Phase B) stages, applying LLM-based and offline validation methods to produce structured summaries, anchor tags, and quality scores for every model in the semantic network.

---

## 1. Pipeline Architecture

Phase A streams model metadata from HuggingFace Hub and Ollama, caching raw JSON and enriching with `config.json` and model card text. Phase B runs Tier 1 (deterministic) and Tier 2 (pattern matching) extraction to populate bank positions, anchors, and explicit links. Phase C picks up where B leaves off, applying intelligence extraction to produce the prose summary and supplementary anchors that Tier 1+2 cannot capture.

### 1.1 Dependency Graph

```
Phase B (done: 37,828 models)
    |
    +----> C1: Smol-Hub-tldr (360M, transformers, macpro only)
    |        Input: 6,760 models with card_text
    |        Output: smol_summary per model (~1 hour)
    |
    +----> C2: qwen2.5:3b (Ollama, macpro + homebridge)
    |        Input: ALL 37,828 models (enriched prompts)
    |        Output: qwen_summary + extra_anchors (~3 days)
    |
    |   [C1 and C2 run in parallel -- C1 finishes first]
    |
    +----> Summary Selection (smol preferred for card_text models)
    |
    +----> C3: Quality Gate (qwen2.5:3b, blind review)
    |        Input: generated summary + anchors ONLY (no source)
    |        Output: quality_score + flags (~3 days)
    |
    +----> C4: Ground Truth Validation (fast, offline)
    |        Input: our outputs vs parsed-model-cards (5K) + hub-tldr-summaries (5K)
    |        Output: accuracy metrics, flagged disagreements
    |
    +----> C5: Healing Pass (future -- infer missing fields)
```

### 1.2 Design Principles

**Standalone workers.** Each sub-phase worker (`phase_c1_worker.py`, `phase_c_worker.py`, `phase_c3_worker.py`) is a self-contained single-file script with zero ModelAtlas imports. Workers can be `scp`'d to any machine with Python and a minimal `pip install`. This decouples compute from the main codebase and enables distribution across heterogeneous hardware.

**JSONL interchange.** All data exchange between the orchestrator (`ingest.py`) and workers uses newline-delimited JSON. Export writes JSONL, workers read JSONL and write JSONL, merge reads JSONL. No binary formats, no shared database connections across machines.

**Graceful shutdown.** Every worker and the orchestrator handle SIGTERM and SIGINT, finishing the current model before exiting. Workers flush after every result line. Resume support via `--resume` flag reads existing output to build a skip set.

**Summary provenance.** Both `smol_summary` (C1) and `qwen_summary` (C2) are stored permanently in the metadata table. The selected `vibe_summary` is a pointer to the preferred source, not a destructive replacement. Audit trails are preserved.

---

## 2. Sub-Phase Specifications

### 2.1 C1: Smol-Hub-tldr Summarization

Generates one-sentence summaries from model card text using a small, specialized summarization model.

| Property | Value |
|----------|-------|
| Model | `davanstrien/Smol-Hub-tldr` (360M params) |
| Training | SFT'd on Llama 3.3 70B distillation of HuggingFace model cards |
| Input | `card_text` from enriched models (truncated to 2,048 tokens) |
| Output | One-sentence summary (`smol_summary`) |
| Prompt format | `<MODEL_CARD>{text}</MODEL_CARD>` |
| Generation config | `max_length=2048`, `max_new_tokens=150`, `temperature=0.4`, `do_sample=True` |
| Runtime | ~0.5s per model on CPU; ~1 hour for 6,760 models |
| Worker | `phase_c1_worker.py` (standalone, zero MA imports) |
| Dependencies | `transformers`, `torch` |

#### C1 Sub-tiers

**C1a: Core enriched models.** The 6,760 models from Phase B that have non-empty `card_text`. These are exported by `--export-c1` and processed by the C1 worker.

**C1b: Extended corpus.** Models beyond the Phase B set, sourced from:
- `librarian-bots/model_cards_with_metadata` (975K models, daily-refreshed)
- HuggingFace API direct queries for models not in the ingest DB

C1b results are merged with `--merge-c1`, which creates stub model entries in the network DB for models not already tracked. Stub entries record their provenance (`hf-api-extended` for tier 1, `librarian-bots` for tier 2).

#### C1 Worker Details

The worker (`phase_c1_worker.py`) loads the model once at startup and processes input JSONL line-by-line:

```bash
python phase_c1_worker.py --input cards.jsonl --output results_c1.jsonl
python phase_c1_worker.py --input cards.jsonl --output results_c1.jsonl --resume
```

The `--resume` flag reads the output file to build a skip set of already-processed `model_id` values, then opens the output in append mode. This enables safe restart after interruption without reprocessing.

The model is loaded as `AutoModelForSeq2SeqLM` (not causal LM) and runs on CUDA if available, falling back to CPU. Generation uses `torch.no_grad()` for inference. Only newly generated tokens (past the input length) are decoded.

### 2.2 C2: Ollama Structured Extraction

Generates structured JSON summaries and anchor tags using a 3B-parameter model via Ollama's OpenAI-compatible API. Covers the entire Phase B corpus, not just models with card text.

| Property | Value |
|----------|-------|
| Model | `qwen2.5:3b` via Ollama |
| API | OpenAI-compatible (`/v1/chat/completions`) |
| Input | Enriched prompts with metadata, config summary, card excerpt, existing anchors |
| Output | `qwen_summary` (prose) + `extra_anchors` (1-5 hyphenated tags) |
| Response format | `{"type": "json_object"}` (Ollama JSON mode) |
| Temperature | 0.3 |
| Distribution | 2 machines (macpro + homebridge) |
| Worker | `phase_c_worker.py` (standalone, zero MA imports) |
| Dependencies | `openai` |
| Runtime | ~5 models/min/machine; ~63 hours for 37,828 models on 2 machines |

#### C2 Prompt Construction

The prompt is built by `build_vibe_prompt()` in `extraction/vibes.py` from pre-extracted Tier 1+2 data. The template includes:

- `model_id`, `author`, `pipeline_tag`
- Tags (up to 15)
- Parameter count (from metadata, as `"{N}B parameters"` or `"unknown"`)
- Family (from LINEAGE bank anchors)
- Known capabilities (from CAPABILITY bank anchors)
- Already-assigned anchors (all anchors linked to the model)
- Config summary (extracted from `config.json`: architecture name, layer count, hidden dim, attention head topology, vocab size)
- Card excerpt (cleaned model card text, stripped of HTML, badges, code blocks, URLs; up to 500 chars)

The explicit instruction to avoid duplicating existing anchors prevents the model from re-discovering what Tier 1+2 already extracted.

#### C2 Export and Sharding

`export_phase_c2(num_shards)` queries `phase_b_done=1 AND phase_c2_done=0 AND phase_c_attempts < 3 AND likes >= 50`, builds prompts, and round-robin distributes across shard files:

```bash
python -m model_atlas.ingest --export-c2 2
# produces: ~/.cache/model-atlas/phase_c_work/shard_0.jsonl
#           ~/.cache/model-atlas/phase_c_work/shard_1.jsonl
```

#### C2 Worker Details

```bash
# On macpro (local Ollama):
python phase_c_worker.py --input shard_0.jsonl --output results_0.jsonl

# On homebridge (remote Ollama):
python phase_c_worker.py --input shard_1.jsonl --output results_1.jsonl \
    --url http://192.168.50.17:11434/v1
```

The worker validates responses with `_parse_and_validate()`:
- Checks for JSON object with non-empty `summary` string
- Checks `extra_anchors` is a list with at least one valid tag
- Filters out known placeholder strings (`tag1`, `tag2`, `example-tag`, etc.)
- Enforces lowercase-hyphenated format (`^[a-z][a-z0-9-]+$`) and minimum length of 3
- Truncates anchors to 5 maximum
- On validation failure, writes an error record and continues

#### C2 Merge

`merge_phase_c2()` reads result JSONL files and for each successful result:
- Stores `qwen_summary` as metadata (key: `"qwen_summary"`)
- Stores up to 5 `extra_anchors` as CAPABILITY-bank anchors with `confidence=0.5` and `source="vibe"`
- Marks `phase_c2_done=1` in the ingest DB

Error records (containing `"error"` key) are counted and skipped.

### 2.3 Summary Selection

After C1 and C2 complete, `select_final_summaries()` picks the best summary per model:

| Condition | Selection |
|-----------|-----------|
| `smol_summary` exists (model had `card_text`) | Use `smol_summary` |
| Only `qwen_summary` exists | Use `qwen_summary` |
| Neither exists | No `vibe_summary` assigned |

The selected summary is stored as `vibe_summary` in the metadata table. Both `smol_summary` and `qwen_summary` remain in metadata for audit. The rationale for preferring `smol_summary`: it is generated from the actual model card text by a model specifically fine-tuned for that task, while `qwen_summary` is generated from a structured prompt that may not capture card-specific nuances.

```bash
python -m model_atlas.ingest --select-summaries
```

### 2.4 C3: Quality Gate

A blind review pass that evaluates generated summaries without access to the original source material. This tests whether the summary and anchors are self-consistent and informative in isolation.

| Property | Value |
|----------|-------|
| Model | `qwen2.5:3b` via Ollama (same as C2) |
| Input | `vibe_summary` + all anchors per model |
| NOT provided | Original model card, config.json, raw metadata |
| Output | Three axis scores + quality flags |
| Worker | `phase_c3_worker.py` (standalone, zero MA imports) |
| Dependencies | `openai` |

#### Scoring Axes

Each axis is rated 0-3:

**Specificity (0-3):** Does the summary mention concrete details?
- 0 = Completely generic ("a great model")
- 1 = Vaguely specific ("a language model for code")
- 2 = Moderately specific ("a 7B instruction-tuned Llama model for code generation")
- 3 = Highly specific ("a 7B Llama-3.1 model fine-tuned on StarCoder data with DPO, targeting Python code completion")

**Coherence (0-3):** Is the summary well-formed and internally consistent?
- 0 = Garbled or contradictory
- 1 = Understandable but awkward
- 2 = Clear and well-written
- 3 = Publication-quality

**Artifacts (0-3):** Signs of hallucination, repetition, or generation artifacts?
- 0 = Severe artifacts (repeated phrases, obvious hallucination, broken text)
- 1 = Minor artifacts (slight repetition, one questionable claim)
- 2 = Clean with minor issues
- 3 = No artifacts detected

#### Quality Score Computation

```
quality_score = (specificity + coherence + artifacts) / 9.0
```

Range: 0.0 to 1.0. Threshold: **0.5** (equivalent to 4.5/9.0 raw).

#### Gate Logic

| Score | `phase_c3_done` | `phase_c_done` | Interpretation |
|-------|-----------------|----------------|----------------|
| >= 0.5 | 1 | 1 | Passed: summary is usable |
| < 0.5 | 1 | 0 | Failed: summary reviewed but rejected |

Models that fail the quality gate retain their `vibe_summary` in metadata (for debugging) but are not marked as Phase C complete. They remain candidates for the C5 healing pass.

Quality metadata stored:
- `quality_score` (float, 0.0-1.0)
- `quality_flags` (JSON list of string concerns)

#### C3 Worker Details

```bash
python phase_c3_worker.py --input quality_gate.jsonl --output results_c3.jsonl
python phase_c3_worker.py --input quality_gate.jsonl --output results_c3.jsonl \
    --model qwen2.5:3b --url http://192.168.50.17:11434/v1 --resume
```

The `--resume` flag works identically to C1: reads existing output, builds skip set, appends new results.

### 2.5 C4: Ground Truth Validation

Offline comparison of ModelAtlas outputs against curated reference datasets. No LLM required.

| Property | Value |
|----------|-------|
| Implementation | `ground_truth.py` |
| Dependencies | `datasets` (for loading HF datasets), `difflib` |
| LLM required | No |

#### Reference Datasets

| Dataset | Size | Source Model | Content |
|---------|------|-------------|---------|
| `davanstrien/hub-tldr-model-summaries-llama` | ~5K models | Llama 3.3 70B | One-sentence summaries |
| `davanstrien/parsed-model-cards` | ~5K models | QwQ-32B | Structured field extractions |

#### Validation Checks

**Summary similarity.** For each model present in both our output and the reference summaries, compute `difflib.SequenceMatcher` ratio on lowercased strings. Models with similarity below 0.2 are flagged as `low_summary_similarity`.

**Anchor coverage.** For each model present in both our anchors and the parsed model cards, extract expected fields from the parsed card (`base_model` -> lineage, `training_data` -> training-data, `language` -> multilingual/language) and measure what fraction our anchor set covers. Reported as `anchor_coverage_mean`.

**Parameter count disagreements.** Compares our `parameter_count_b` metadata against the reference `parameters` field. Values differing by more than 50% (relative) are flagged as `param_count_mismatch`. The comparison normalizes raw counts (> 1M) to billions.

#### Output Metrics

```python
{
    "total_compared": int,          # Total comparisons made
    "summary_comparisons": int,     # Models with both our and reference summaries
    "similarity_mean": float,       # Mean SequenceMatcher ratio
    "similarity_median": float,     # Median SequenceMatcher ratio
    "parsed_comparisons": int,      # Models with anchor coverage comparisons
    "anchor_coverage_mean": float,  # Mean fraction of expected fields covered
    "flagged_disagreements": [      # List of flagged issues
        {
            "model_id": str,
            "type": "low_summary_similarity" | "param_count_mismatch",
            ...
        }
    ]
}
```

```bash
python -m model_atlas.ingest --validate-ground-truth
```

### 2.6 Phase D: Post-Bootstrap Audit & Healing Pipeline

A three-layer error correction pipeline where each layer is progressively more expensive and more capable.

```
D0: Schema & Provenance Layer
    │   phase_d_runs, audit_findings, correction_events tables
    ▼
D1: Deterministic Audit ──── O(1) per model, reuses patterns.py matchers
    │   Compare C2 anchors vs _CAPABILITY_PATTERNS / _DOMAIN_PATTERNS
    │   Stable mismatch taxonomy: contradiction, gap, confidence-conflict
    ▼
D2: Dictionary Expansion ─── strict DSL, auto-link high-confidence only
    │   Add missing domain labels (biology, music, chemistry, physics)
    │   Boundary-aware matchers, AND/OR semantics
    ▼
D3: LLM Healing Pass ──── full C2-style responses from raw evidence
    │   ├── D3a: Local (qwen2.5:3b) ── bulk corrections
    │   └── D3b: Claude Code CLI ──── 0.1% inference tax per session
    ▼
D4: Fine-Tuning Data Export ── DPO format JSONL
    │   From correction_events (prompt/chosen/rejected)
    ▼
Repeat: expand corpus → extract → C2 → D1 audit → D2 expand → D3 heal → D4 export
```

#### D0: Provenance Schema

Three new tables in `network.db`:
- `phase_d_runs` — tracks each D-phase run (UUID, phase, status, config JSON, summary JSON)
- `audit_findings` — stores per-model mismatch findings (contradiction, gap, confidence_conflict)
- `correction_events` — stores healing corrections with original + healed responses for DPO training

#### D1: Deterministic Audit

`phase_d_audit.py` re-runs `_CAPABILITY_PATTERNS` and `_DOMAIN_PATTERNS` from `extraction/patterns.py` against each model's searchable text, then compares against C2-assigned anchors (confidence=0.5):

| Mismatch Type | Description | Example |
|---------------|-------------|---------|
| `contradiction` | C2 assigned X, deterministic says Y in same bank | Spark-TTS gets `code-domain` but patterns find `speech` signals |
| `gap` | Deterministic found X but C2 missed it entirely | Model named "instruct-7b" missing `instruction-following` |
| `confidence_conflict` | Same anchor, confidence gap > 0.3 | Both agree on `chat` but at 0.5 vs 0.9 confidence |

Per-model `audit_score` (0.0-1.0) stored in metadata. Models below `AUDIT_MISMATCH_THRESHOLD` (0.7) become healing candidates.

#### D2: Dictionary Expansion

`phase_d_expand.py` reads YAML expansion specs with a strict DSL:

```yaml
expansions:
  - label: "biology-domain"
    bank: "DOMAIN"
    mode: "auto_link"       # create_only | auto_link | queue_for_heal
    match_rules:
      operator: "OR"        # AND | OR
      conditions:
        - type: "tag_exact"
          value: "biology"
        - type: "name_regex"
          value: "\\bbio(?:logy|med)\\b"
      min_matches: 1
    confidence: 0.7
```

Matcher types: `tag_exact`, `tag_regex`, `pipeline_tag_in`, `name_regex`, `metadata_equals`.

#### D3: LLM Healing

`phase_d_heal.py` follows the export/merge pattern from Phase C. Healing prompts include raw_json evidence, card excerpt, current anchors, audit findings, and the full valid anchor dictionary. Workers output complete C2-style responses (`{summary, selected_anchors}`), and merge computes diffs.

`phase_d_worker.py` is a standalone worker (zero ModelAtlas imports) following the same pattern as `phase_c_worker.py`.

#### D4: Training Data Export

`phase_d_training.py` exports DPO-format JSONL from `correction_events`:
```json
{"prompt": "...", "chosen": "{healed}", "rejected": "{original}", "model_id": "...", "tier": "local"}
```

---

## 3. Database Schema: Ingest State Tracking

Phase C state is tracked in the ingest state database (`~/.cache/model-atlas/ingest_state.db`), separate from the semantic network database.

### 3.1 Ingest Models Table

```sql
CREATE TABLE IF NOT EXISTS ingest_models (
    model_id         TEXT PRIMARY KEY,
    source           TEXT DEFAULT 'huggingface',
    likes            INTEGER DEFAULT 0,
    phase_a_done     INTEGER DEFAULT 0,
    phase_b_done     INTEGER DEFAULT 0,
    phase_c_done     INTEGER DEFAULT 0,    -- Final gate: 1 only after C3 pass
    phase_c_attempts INTEGER DEFAULT 0,    -- Retry counter (max 3)
    raw_json         TEXT,
    fetched_at       TEXT,
    extracted_at     TEXT,
    vibed_at         TEXT,
    -- Enrichment columns (Phase A)
    config_json      TEXT,
    card_text        TEXT,
    enriched         INTEGER DEFAULT 0,
    -- Sub-phase tracking
    phase_c1_done    INTEGER DEFAULT 0,    -- Smol-Hub-tldr complete
    phase_c2_done    INTEGER DEFAULT 0,    -- Ollama extraction complete
    phase_c3_done    INTEGER DEFAULT 0,    -- Quality gate reviewed
    phase_c1_at      TEXT,
    phase_c2_at      TEXT,
    phase_c3_at      TEXT
);
```

### 3.2 State Machine

```
Phase B done
    |
    +---> C1: card_text != '' --> phase_c1_done = 1
    |
    +---> C2: likes >= 50, phase_c_attempts < 3 --> phase_c2_done = 1
    |
    +---> Summary Selection (reads c1/c2 metadata, writes vibe_summary)
    |
    +---> C3: phase_c2_done = 1 --> phase_c3_done = 1
    |         |
    |         +---> quality_score >= 0.5 --> phase_c_done = 1 (PASS)
    |         +---> quality_score < 0.5  --> phase_c_done = 0 (FAIL)
```

The `phase_c_done` flag is the terminal gate. Only models that pass C3 have `phase_c_done=1`. The `phase_c_attempts` counter caps retries at 3 (configurable via `VIBE_MAX_RETRIES`).

---

## 4. JSONL Schemas

All inter-process data exchange uses newline-delimited JSON (JSONL). One JSON object per line, UTF-8 encoded.

### 4.1 C1 Export (`cards.jsonl`)

```json
{"model_id": "author/model", "card_text": "# Model Card\n..."}
```

### 4.2 C1 Results

```json
{"model_id": "author/model", "smol_summary": "A fine-tuned model for..."}
```

Error case:

```json
{"model_id": "author/model", "error": "CUDA out of memory"}
```

### 4.3 C2 Export (`shard_N.jsonl`)

```json
{"model_id": "author/model", "prompt": "You are a concise ML model analyst..."}
```

### 4.4 C2 Results

```json
{"model_id": "author/model", "summary": "...", "extra_anchors": ["code-generation", "instruction-following"]}
```

Error case:

```json
{"model_id": "author/model", "error": "Connection refused"}
```

### 4.5 C3 Export (`shard_N.jsonl`)

```json
{"model_id": "author/model", "prompt": "You are a quality reviewer..."}
```

### 4.6 C3 Results

```json
{"model_id": "author/model", "quality_score": 0.78, "specificity": 2, "coherence": 3, "artifacts": 2, "flags": []}
```

Error case:

```json
{"model_id": "author/model", "error": "'specificity' must be int or float, got NoneType"}
```

---

## 5. External Datasets

| Resource | Size | Use | Notes |
|----------|------|-----|-------|
| `davanstrien/Smol-Hub-tldr` | 360M model | C1: fast summary generation from card text | SFT on Llama 3.3 70B distillation |
| `davanstrien/hub-tldr-model-summaries-llama` | ~5K models | C4: ground truth summaries | Generated by Llama 3.3 70B |
| `davanstrien/parsed-model-cards` | ~5K models | C4: structured ground truth fields | Parsed by QwQ-32B |
| `librarian-bots/model_cards_with_metadata` | ~975K models | C1b: extended corpus for summarization | Daily-refreshed by Librarian Bot |

---

## 6. Hardware

| Machine | Hostname | Role | CPU | RAM | Ollama | Python |
|---------|----------|------|-----|-----|--------|--------|
| macpro | 192.168.50.73 | All Python + local Ollama | Xeon 6C 3.5GHz | 12GB ECC | qwen2.5:3b | 3.11 |
| homebridge | 192.168.50.17 | Remote Ollama only | i7 4C 2.6GHz | 16GB | qwen2.5:3b | None |

**macpro** runs the orchestrator (`ingest.py`), all export/merge commands, C1 workers (transformers + torch), and one instance of the C2/C3 workers against localhost Ollama. **homebridge** runs only Ollama and accepts remote API calls. Workers are distributed by `scp`'ing the standalone worker script and a shard file, then running remotely.

### 6.1 Worker Deployment

```bash
# Distribute C2 work to homebridge:
scp phase_c_worker.py shard_1.jsonl homebridge:~/phase_c/
ssh homebridge "cd ~/phase_c && python phase_c_worker.py \
    --input shard_1.jsonl --output results_1.jsonl"

# Collect results:
scp homebridge:~/phase_c/results_1.jsonl ~/.cache/model-atlas/phase_c_work/
```

---

## 7. Configuration

All Phase C constants live in `src/model_atlas/config.py`:

| Constant | Value | Purpose |
|----------|-------|---------|
| `VIBE_OLLAMA_MODEL` | `qwen2.5:3b` | Ollama model for C2 and C3 |
| `VIBE_OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Default Ollama endpoint |
| `VIBE_MAX_RETRIES` | `3` | Max C2 attempts per model |
| `INGEST_VIBE_MIN_LIKES` | `50` | Minimum likes for Phase C eligibility |
| `QUALITY_GATE_MIN_SCORE` | `0.5` | C3 pass threshold (4.5/9.0) |
| `SMOL_HUB_TLDR_MODEL` | `davanstrien/Smol-Hub-tldr` | C1 model name |
| `SMOL_HUB_TLDR_MAX_LENGTH` | `2048` | C1 max input token length |
| `SMOL_HUB_TLDR_MAX_NEW_TOKENS` | `150` | C1 max generated tokens |
| `SMOL_HUB_TLDR_TEMPERATURE` | `0.4` | C1 generation temperature |
| `MAX_CARD_TEXT_LENGTH` | `2000` | Card text truncation (chars) |
| `PHASE_C_WORK_DIR` | `~/.cache/model-atlas/phase_c_work` | C2 shard directory |
| `PHASE_C1_WORK_DIR` | `~/.cache/model-atlas/phase_c1_work` | C1 export directory |
| `PHASE_C3_WORK_DIR` | `~/.cache/model-atlas/phase_c3_work` | C3 shard directory |
| `INGEST_BATCH_SIZE` | `50` | DB commit batch size |

---

## 8. CLI Reference

All commands are invoked through the ingest module:

```
python -m model_atlas.ingest <command>
```

### 8.1 C1 Commands

| Flag | Arguments | Description |
|------|-----------|-------------|
| `--export-c1` | (none) | Export `card_text` models to `cards.jsonl` for Smol-Hub-tldr processing |
| `--merge-c1` | `FILE [FILE...]` | Merge C1 result JSONL files into network DB |

### 8.2 C2 Commands

| Flag | Alias | Arguments | Description |
|------|-------|-----------|-------------|
| `--export-c2` | `--export-phase-c` | `NUM_SHARDS` | Export C2 work items to `NUM_SHARDS` sharded JSONL files |
| `--c2-worker` | `--worker` | `INPUT_FILE` | Run C2 worker on a shard JSONL file |
| `--worker-output` | | `OUTPUT_FILE` | Output path for worker results (default: auto-named) |
| `--merge-c2` | `--merge-phase-c` | `FILE [FILE...]` | Merge C2 result JSONL files into network DB |

### 8.3 C3 Commands

| Flag | Arguments | Description |
|------|-----------|-------------|
| `--export-c3` | `NUM_SHARDS` | Export C3 quality gate items to `NUM_SHARDS` sharded JSONL files |
| `--merge-c3` | `FILE [FILE...]` | Merge C3 quality gate result JSONL files |

### 8.4 Summary Selection and Validation

| Flag | Arguments | Description |
|------|-----------|-------------|
| `--select-summaries` | (none) | Pick best summary (smol vs qwen) per model, store as `vibe_summary` |
| `--validate-ground-truth` | (none) | Run C4 ground truth comparison against reference datasets |
| `--status` | (none) | Show ingest progress with C1/C2/C3 breakdown |

### 8.5 Shared Options

| Flag | Default | Description |
|------|---------|-------------|
| `--vibe-min-likes` | `50` | Minimum likes for Phase C eligibility |
| `--vibe-backend` | `transformers` | Backend for inline vibe extraction: `transformers` or `ollama` |
| `--ollama-model` | (from config) | Ollama model name override |
| `--ollama-url` | (from config) | Ollama API base URL override |
| `-v`, `--verbose` | off | Enable DEBUG-level logging |

### 8.6 Phase D Commands

| Flag | Arguments | Description |
|------|-----------|-------------|
| `--audit-c2` | (none) | Run D1 deterministic audit on C2 anchors |
| `--expand-dictionary` | `SPEC_FILE` | D2: expand dictionary from YAML spec |
| `--dry-run` | (none) | Preview expansion counts (with `--expand-dictionary`) |
| `--export-d3` | `NUM_SHARDS` | D3: export healing prompts to sharded JSONL |
| `--heal-tier` | `local\|claude` | D3: healing tier (default: local) |
| `--heal-budget` | `N` | D3: max models to heal (default: 100) |
| `--heal-seed` | `INT` | D3: random seed for reproducible selection (default: 42) |
| `--merge-d3` | `FILE [FILE...]` | D3: merge healing result JSONL files |
| `--run-id` | `RUN_ID` | D3: run_id for merge (from export output) |
| `--export-training-data` | `OUTPUT_PATH` | D4: export DPO training data |
| `--training-tier` | `local\|claude\|all` | D4: filter by tier (default: all) |
| `--phase-d-status` | (none) | Show Phase D run history and stats |

### 8.7 Standalone Workers

Workers are separate scripts, not invoked through `python -m model_atlas.ingest`:

```bash
# C1 worker
python phase_c1_worker.py --input cards.jsonl --output results_c1.jsonl [--resume]

# C2 worker
python phase_c_worker.py --input shard.jsonl --output results.jsonl \
    [--model qwen2.5:3b] [--url http://localhost:11434/v1]

# C3 worker
python phase_c3_worker.py --input quality_gate.jsonl --output results_c3.jsonl \
    [--model qwen2.5:3b] [--url http://localhost:11434/v1] [--resume]
```

---

## 9. Operational Runbook

### Day 0: Setup

1. Ensure Ollama is running on both machines with `qwen2.5:3b` pulled:
   ```bash
   # On macpro and homebridge:
   ollama pull qwen2.5:3b
   ollama serve
   ```

2. Verify connectivity:
   ```bash
   curl http://<IP>/v1/models   # macpro
   curl http://<IP>/v1/models   # homebridge
   ```

3. Confirm Phase B is complete:
   ```bash
   python -m model_atlas.ingest --status
   # Expected: Phase B (extracted) >= 37,000
   ```

### Day 1: C1a + C2 Export

4. Export C1 work items (models with card_text):
   ```bash
   python -m model_atlas.ingest --export-c1
   # Output: ~/.cache/model-atlas/phase_c1_work/cards.jsonl
   ```

5. Export C2 work items (all eligible models, 2 shards):
   ```bash
   python -m model_atlas.ingest --export-c2 2
   # Output: ~/.cache/model-atlas/phase_c_work/shard_0.jsonl
   #         ~/.cache/model-atlas/phase_c_work/shard_1.jsonl
   ```

6. Distribute C2 shard_1 to homebridge:
   ```bash
   scp phase_c_worker.py ~/.cache/model-atlas/phase_c_work/shard_1.jsonl \
       homebridge:~/phase_c/
   ```

### Day 1-2: C1a + C2 Parallel Execution

7. Start C1 worker on macpro (runs ~1 hour):
   ```bash
   python phase_c1_worker.py \
       --input ~/.cache/model-atlas/phase_c1_work/cards.jsonl \
       --output ~/.cache/model-atlas/phase_c1_work/results_c1.jsonl
   ```

8. Start C2 workers in parallel:
   ```bash
   # On macpro (local Ollama):
   python phase_c_worker.py \
       --input ~/.cache/model-atlas/phase_c_work/shard_0.jsonl \
       --output ~/.cache/model-atlas/phase_c_work/results_0.jsonl

   # On homebridge (remote):
   ssh homebridge "cd ~/phase_c && python phase_c_worker.py \
       --input shard_1.jsonl --output results_1.jsonl"
   ```

9. C1 completes first (~1 hour). Merge immediately:
   ```bash
   python -m model_atlas.ingest --merge-c1 \
       ~/.cache/model-atlas/phase_c1_work/results_c1.jsonl
   ```

### Day 2-3 (optional): C1b Extended Corpus

10. While C2 is still running, process extended corpus through C1:
    ```bash
    # Prepare extended corpus cards.jsonl from librarian-bots or HF API
    python phase_c1_worker.py \
        --input extended_cards.jsonl \
        --output results_c1b.jsonl --resume

    # Merge extended results (creates stub models for unknown model_ids):
    python -m model_atlas.ingest --merge-c1 results_c1b.jsonl
    ```

### Day 3-4: Merge C2

11. Collect C2 results from homebridge:
    ```bash
    scp homebridge:~/phase_c/results_1.jsonl \
        ~/.cache/model-atlas/phase_c_work/
    ```

12. Merge all C2 results:
    ```bash
    python -m model_atlas.ingest --merge-c2 \
        ~/.cache/model-atlas/phase_c_work/results_0.jsonl \
        ~/.cache/model-atlas/phase_c_work/results_1.jsonl
    ```

### Day 4: Summary Selection

13. Run summary selection (smol preferred for card_text models):
    ```bash
    python -m model_atlas.ingest --select-summaries
    ```

14. Verify status:
    ```bash
    python -m model_atlas.ingest --status
    ```

### Day 4-5: C3 Quality Gate

15. Export C3 work items:
    ```bash
    python -m model_atlas.ingest --export-c3 2
    ```

16. Distribute and run C3 workers (same pattern as C2):
    ```bash
    # On macpro:
    python phase_c3_worker.py \
        --input ~/.cache/model-atlas/phase_c3_work/shard_0.jsonl \
        --output ~/.cache/model-atlas/phase_c3_work/results_c3_0.jsonl

    # On homebridge:
    scp phase_c3_worker.py ~/.cache/model-atlas/phase_c3_work/shard_1.jsonl \
        homebridge:~/phase_c3/
    ssh homebridge "cd ~/phase_c3 && python phase_c3_worker.py \
        --input shard_1.jsonl --output results_c3_1.jsonl"
    ```

### Day 7-8: Merge C3 + C4 Validation

17. Collect and merge C3 results:
    ```bash
    scp homebridge:~/phase_c3/results_c3_1.jsonl \
        ~/.cache/model-atlas/phase_c3_work/
    python -m model_atlas.ingest --merge-c3 \
        ~/.cache/model-atlas/phase_c3_work/results_c3_0.jsonl \
        ~/.cache/model-atlas/phase_c3_work/results_c3_1.jsonl
    ```

18. Run C4 ground truth validation:
    ```bash
    python -m model_atlas.ingest --validate-ground-truth
    ```

19. Final status check:
    ```bash
    python -m model_atlas.ingest --status
    # Expected output:
    # Phase C (complete): N (models that passed quality gate)
    #   C1 (smol-tldr):    ~6,760
    #   C2 (ollama):       ~37,828
    #   C3 (quality gate): ~37,828
    ```

### Day 8+: Ongoing

- C1b continues processing extended corpus models as new cards appear in `librarian-bots/model_cards_with_metadata`
- Periodic re-export of C2 picks up newly fetched models (Phase A incremental runs add new models above the likes threshold)

### Day 9+: Phase D Operations

20. Run D1 audit on C2 results:
    ```bash
    python -m model_atlas.ingest --audit-c2
    ```

21. Expand dictionary with missing domain anchors:
    ```bash
    # Preview first
    python -m model_atlas.ingest --expand-dictionary data/expansions/domain_specialization.yaml --dry-run
    # Apply
    python -m model_atlas.ingest --expand-dictionary data/expansions/domain_specialization.yaml
    ```

22. Export D3 healing work (local tier):
    ```bash
    python -m model_atlas.ingest --export-d3 2 --heal-tier local --heal-budget 100 --heal-seed 42
    # Note the run_id in output
    ```

23. Run D3 worker on each shard:
    ```bash
    python phase_d_worker.py --input d3_local_shard_0.jsonl --output d3_results_0.jsonl
    ```

24. Merge D3 results:
    ```bash
    python -m model_atlas.ingest --merge-d3 d3_results_0.jsonl d3_results_1.jsonl --run-id <RUN_ID>
    ```

25. Export DPO training data:
    ```bash
    python -m model_atlas.ingest --export-training-data training_data.jsonl
    ```

26. Check Phase D status:
    ```bash
    python -m model_atlas.ingest --phase-d-status
    ```

---

## 10. Module Map

```
src/model_atlas/
+-- ingest.py                  Pipeline orchestrator: export, merge, select, status
+-- db_ingest.py               Ingest state DB schema and migration
+-- ground_truth.py            C4: offline validation against reference datasets
+-- extraction/
|   +-- vibes.py               Prompt templates, extractors, quality gate prompt
|   |   +-- SmolHubTldrExtractor     C1 in-process extractor
|   |   +-- OllamaVibeExtractor      C2 in-process extractor
|   |   +-- VibeExtractor            Outlines-based extractor (legacy)
|   |   +-- build_vibe_prompt()       C2 prompt construction
|   |   +-- build_quality_gate_prompt()  C3 prompt construction
|   +-- deterministic.py       Tier 1: ModelInput dataclass
|   +-- patterns.py            Tier 2: regex/heuristics
|   +-- pipeline.py            extract_and_store() orchestrator
+-- config.py                  All Phase C constants
+-- phase_c_worker.py          Standalone C2 worker (zero MA imports)
+-- phase_c1_worker.py         Standalone C1 worker (zero MA imports)
+-- phase_c3_worker.py         Standalone C3 worker (zero MA imports)
+-- phase_d_audit.py           D1: deterministic audit of C2 anchors
+-- phase_d_expand.py          D2: dictionary expansion with strict DSL
+-- phase_d_heal.py            D3: healing export/merge orchestration
+-- phase_d_worker.py          Standalone D3 healing worker (zero MA imports)
+-- phase_d_training.py        D4: DPO training data export from corrections
```

---

## 11. Error Handling and Recovery

### 11.1 Worker Failures

All workers write results line-by-line with `fout.flush()` after each result. On crash, the output file contains all completed results. The `--resume` flag (C1 and C3 workers) or manual JSONL line counting enables restart without reprocessing.

For C2 workers without `--resume`, partial output files can still be merged; `merge_phase_c2` skips duplicate `model_id` entries via the `phase_c2_done` flag.

### 11.2 Ollama Failures

Connection errors, timeouts, and malformed responses are caught per-model. The error is recorded in the output JSONL and the worker continues to the next model. Error records are counted during merge and skipped.

### 11.3 Retry Budget

The `phase_c_attempts` counter in the ingest DB tracks how many times C2 export has included a model. Models with `phase_c_attempts >= VIBE_MAX_RETRIES` (default: 3) are excluded from future exports. This prevents infinite retry loops on models that consistently cause errors.

### 11.4 Signal Handling

All workers and the orchestrator register handlers for SIGTERM and SIGINT that set a `_shutdown` flag. The current model finishes processing before the worker exits cleanly. This enables graceful shutdown via `kill <pid>` or Ctrl+C.

---

## 12. Related Work

### 12.1 Documentation Coverage in ML

Liang et al. (arXiv 2402.05160) analyzed documentation completeness across HuggingFace model cards and found significant gaps: 74.3% include training information, 17.4% document limitations, and only 2% report environmental impact. Phase C is designed to extract maximum signal from the documentation that does exist, and the C5 healing pass is planned to infer missing fields from structural signals.

### 12.2 Machine-Readable Model Metadata (MRM3)

The MRM3 initiative proposes standardized machine-readable metadata for ML models. ModelAtlas's anchor dictionary and bank positions provide a complementary structured representation that can be aligned with MRM3 schemas.

### 12.3 Transformers TrainingSummary

The `transformers/modelcard.py` module defines a canonical `TrainingSummary` schema with fields for model name, language, license, tags, datasets, and evaluation metrics. The C5 healing pass plans to use this schema as a template for inferring missing documentation fields from code-level signals.
