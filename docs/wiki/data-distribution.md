# Data Distribution

**The semantic network is distributed separately from the code — as a SQLite file attached to GitHub Releases. One download, no running services, no accounts.**

---

## Problem Space

The `network.db` file changes on a different cadence than the code. Code changes daily; the database changes per correction-pipeline run. Binary diffs on SQLite are useless in git history. And users need the data once, not on every `git pull`.

This creates a distribution problem: how do you deliver an 80MB binary artifact that's versioned independently from the source code?

---

## Core Model

### GitHub Releases (Current)

Each beta milestone produces a GitHub Release with `network.db` attached as a release asset.

**Download:**
```bash
mkdir -p ~/.cache/model-atlas
curl -L -o ~/.cache/model-atlas/network.db \
  https://github.com/rohanvinaik/ModelAtlas/releases/latest/download/network.db
```

**Versioning:** Releases follow the code version with a beta suffix: `v0.2.0-beta.1`, `v0.2.0-beta.2`. A new release is cut when:
- The correction pipeline produces a meaningfully improved database
- Anchor dictionary changes affect query behavior
- Significant model count increases (corpus expansion)

Release notes document: model count, anchor count, validation coverage, and what changed.

### HuggingFace Dataset (Planned)

The network will also be published as a HuggingFace dataset for discoverability within the ML community:

```
rohanvinaik/model-atlas-network/
  network.db              # The full SQLite database
  anchor_dictionary.json  # Exported anchor labels by bank
  validation_summary.json # Gemini audit aggregate metrics
  README.md               # Dataset card with methodology
```

ML practitioners already know how to pull HF datasets. Download stats provide usage signal. Dataset cards document the [extraction pipeline](Extraction-Pipeline) methodology.

---

## Evidence: What's Not Distributed

| Artifact | Why excluded |
|----------|-------------|
| `ingest_state.db` | Pipeline state tracking — only needed for running extraction, not querying |
| `phase_c_work/` | Intermediate JSONL worker shards — ephemeral build artifacts |
| `model_cards/` | Cached raw model card text — re-fetchable from HuggingFace API |
| `gemini_validation.jsonl` | Raw audit results — summarized in release notes |

Everything needed to *use* ModelAtlas is in the release. Everything needed to *build* ModelAtlas is in the source repo.

### Building From Source

Users who want to build their own network:
```bash
python -m model_atlas.ingest --fetch --extract
```

Or incrementally through MCP tools (`hf_build_index`). But the pre-built network includes multi-tier extraction (deterministic + pattern + LLM + audit) that takes days of distributed compute to reproduce.

---

## What This Is Not

- **Not a streaming service.** The database is a static file. Updates come as new releases, not real-time sync.
- **Not locked to GitHub.** The HuggingFace dataset provides an alternative distribution channel. The SQLite file is the canonical format regardless of host.
- **Not large.** ~{{db_size_mb}} is smaller than most npm dependency trees. It fits comfortably in memory on any modern machine.

---

## Related Concepts

- [Getting Started](Getting-Started) — download and setup instructions
- [Extraction Pipeline](Extraction-Pipeline) — how the database is built
- [System Overview](System-Overview) — where the database fits in the architecture

---

*[← Data Model](Data-Model) · [The Gap →](The-Gap)*
