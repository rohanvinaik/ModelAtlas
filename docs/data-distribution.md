# Data Distribution

How the ModelAtlas semantic network is packaged, versioned, and distributed to users.

---

## 1. Overview

The semantic network (`network.db`) is a ~80MB SQLite file containing the full model graph: 19K+ models, 166 anchors across 8 banks, 128K+ model-anchor links, and per-model metadata (summaries, quality scores, extraction provenance). It lives at `~/.cache/model-atlas/network.db` on the user's machine.

The database is distributed separately from the code because:
- It changes on a different cadence (code changes daily, data changes per correction-pipeline run)
- It's too large for productive git history (binary diffs on SQLite are useless)
- Users need the data once, not on every `git pull`

---

## 2. GitHub Releases (Current)

Each beta milestone produces a GitHub Release with `network.db` attached as a release asset.

### Release procedure

```bash
# 1. Tag the release
git tag -a v0.2.0-beta.1 -m "Beta 1: 19.5K models, 166 anchors, 3K validated"
git push origin v0.2.0-beta.1

# 2. Create the release with the database attached
gh release create v0.2.0-beta.1 \
  ~/.cache/model-atlas/network.db \
  --title "v0.2.0-beta.1" \
  --notes "$(cat <<'EOF'
## Semantic Network Beta 1

- **19,498 models** in the network
- **166 anchors** across 8 banks (ARCHITECTURE, CAPABILITY, EFFICIENCY, COMPATIBILITY, LINEAGE, DOMAIN, QUALITY, TRAINING)
- **128K+ model-anchor links**
- **~7,300 models** with LLM-generated enrichment (C2: qwen2.5:3b summaries + capability anchors)
- **~3,000 models** independently validated by Gemini against raw HuggingFace metadata
- All top-downloaded HuggingFace models present and enriched

### Download

```
curl -L -o ~/.cache/model-atlas/network.db \
  https://github.com/rohanvinaik/ModelAtlas/releases/download/v0.2.0-beta.1/network.db
```

Then run `uv run model-atlas` as usual.
EOF
)"
```

### Versioning policy

Releases follow the code version with a beta suffix: `v0.2.0-beta.1`, `v0.2.0-beta.2`, etc. A new release is cut when:
- The correction pipeline produces a meaningfully improved database (e.g., after a D3 healing pass merges)
- Anchor dictionary changes (new banks, renamed anchors) that affect query behavior
- Significant model count increases (corpus expansion beyond current likes threshold)

The release notes document: model count, anchor count, validation coverage, and what changed since the last release.

### User workflow

```bash
# Install
uv sync

# Download the pre-built network (one-time, or on new release)
mkdir -p ~/.cache/model-atlas
curl -L -o ~/.cache/model-atlas/network.db \
  https://github.com/rohanvinaik/ModelAtlas/releases/latest/download/network.db

# Run
uv run model-atlas
```

Without the pre-built network, ModelAtlas starts with an empty database. Users can build their own via `hf_build_index`, but the pre-built network includes multi-tier extraction (deterministic + pattern matching + LLM enrichment + quality gate + audit corrections) that takes days of compute to reproduce.

---

## 3. HuggingFace Dataset (Planned)

The network will also be published as a HuggingFace dataset for discoverability within the ML community.

### Why HuggingFace

- ML practitioners already know how to pull HF datasets
- Download stats provide usage signal
- Dataset cards are a natural place to document the anchor dictionary, extraction methodology, and validation metrics
- Versioning via dataset revisions aligns with the correction pipeline cadence

### Dataset structure

```
rohanvinaik/model-atlas-network/
  network.db              # The full SQLite database
  README.md               # Dataset card with methodology, coverage, validation stats
  anchor_dictionary.json  # Exported anchor labels by bank (for programmatic use)
  validation_summary.json # Gemini validation aggregate metrics
```

### Dataset card contents

The dataset card will document:
- What the database contains (schema, model count, anchor count)
- How it was built (extraction tiers, correction pipeline)
- Validation metrics (Gemini audit results, D1 audit coverage)
- Anchor dictionary reference (all labels by bank with descriptions)
- Known limitations (long-tail accuracy, anchor coverage gaps)
- License and attribution

### User workflow (planned)

```python
from huggingface_hub import hf_hub_download

db_path = hf_hub_download(
    repo_id="rohanvinaik/model-atlas-network",
    filename="network.db",
    repo_type="dataset",
    cache_dir="~/.cache/model-atlas"
)
```

Or via CLI:

```bash
huggingface-cli download rohanvinaik/model-atlas-network network.db \
  --repo-type dataset --local-dir ~/.cache/model-atlas
```

---

## 4. What's NOT Distributed

- `ingest_state.db` — pipeline state tracking (which models have been through each phase). Only needed for running the ingestion pipeline, not for querying.
- `phase_c_work/`, `phase_c1_work/`, `phase_c3_work/` — intermediate JSONL files from worker shards. Ephemeral build artifacts.
- `model_cards/` — cached raw model card text. Can be re-fetched from HuggingFace API.
- `gemini_validation.jsonl` — raw Gemini audit results. Summarized in release notes; full data available on request.

---

## 5. Building from Source

Users who want to build their own network from scratch:

```bash
# Fetch and extract models (Phase A+B)
python -m model_atlas.ingest --fetch --extract

# Or build incrementally via MCP tools
# (hf_build_index fetches, extracts, and stores in one call)
```

See [`pipeline.md`](pipeline.md) for the full multi-phase extraction pipeline, including C1/C2/C3 worker deployment for LLM enrichment.
