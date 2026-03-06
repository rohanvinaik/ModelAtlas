# Wiki Materializer — Design & Implementation Plan

A deterministic documentation materialization system. Canonical repo docs are the source of truth. Generated wiki pages are derived views with full provenance. GitHub Wiki is an optional publish adapter, not the authoring surface.

Adapted from [LintGate #272](https://github.com/rohanvinaik/LintGate/issues/272). ModelAtlas serves as the beta implementation; the core engine is designed to be project-agnostic for LintGate adoption.

---

## 1. Core Contract

Six invariants that define the system:

### 1.1 Canonical Source Map

One config file (`wiki.yaml`) declares the full materialization spec:

```yaml
materializer_version: "0.1.0"

defaults:
  theory_scope: false
  audience: user

pages:
  - id: theory-overview
    title: "Theory: Signed Hierarchies & Semantic Navigation"
    audience: theory
    sources:
      - path: docs/DESIGN.md
        sections:
          - "Core Thesis"
          - "Data Model"
          - "Anchor Dictionary"
          - "Query Engine"
      - path: README.md
        sections:
          - "The gap"
          - "The idea"
          - "What this is not"

  - id: architecture-server
    title: "Architecture: MCP Server & Query Engine"
    audience: operator
    sources:
      - path: docs/DESIGN.md
        sections:
          - "Query Engine"
          - "MCP Server"
      - path: src/model_atlas/server.py
        extract: docstrings

  - id: architecture-pipeline
    title: "Architecture: Extraction Pipeline"
    audience: operator
    sources:
      - path: docs/pipeline.md
        sections: all

  - id: architecture-data
    title: "Architecture: Data Distribution & Versioning"
    audience: operator
    sources:
      - path: docs/data-distribution.md
        sections: all

  - id: user-guide
    title: "User Guide: Installation & Usage"
    audience: user
    sources:
      - path: README.md
        sections:
          - "Quick start"
          - "Usage"
          - "Beta status"

  - id: home
    title: "Home"
    audience: user
    sources: []  # Generated index — no canonical source
    auto_index: true
```

Every materialized page traces to specific source files and sections. If a page has no `sources`, it must declare `auto_index: true` (generated navigation page).

### 1.2 Deterministic Materializer

Same inputs → byte-identical outputs. This applies to all generated artifacts: page files, manifest, and frontmatter. This means:

- No timestamps anywhere in generated output (provenance uses content hashes, not dates)
- No non-deterministic ordering (sections rendered in config-declared order, manifest pages sorted by `id`)
- Source content is read, hashed, then rendered through templates
- Hash algorithm: SHA-256, hex-encoded, truncated to 16 chars for readability
- Materializer version is pinned in config and stamped in output

Determinism is testable: run materializer twice on same inputs, diff all outputs (pages + manifest), expect zero differences.

### 1.3 Provenance Frontmatter

Every generated page starts with YAML frontmatter:

```yaml
---
generated: true
generated_from:
  - docs/DESIGN.md
  - README.md
source_hash: a1b2c3d4e5f60718    # SHA-256 of concatenated source content
spec_hash: 9f8e7d6c5b4a3021      # SHA-256 of this page's config entry + template identity
file_hash: 3c4d5e6f7a8b9012      # SHA-256 of entire file (frontmatter + body)
materializer_version: "0.1.0"
theory_scope: false
audience: theory
page_id: theory-overview
---
```

Fields:
- `generated`: always `true` — identifies machine-generated pages
- `generated_from`: list of source file paths (relative to repo root)
- `source_hash`: hash of all source content that produced this page — changes when sources change
- `spec_hash`: hash of this page's normalized config entry from `wiki.yaml` (title, sections, audience, sources, template identity) — changes when the page spec changes, even without a version bump
- `file_hash`: hash of the entire generated file including frontmatter — the single authoritative hash for drift detection
- `materializer_version`: from `wiki.yaml` — changes when the materializer itself changes
- `theory_scope`: `false` by default — only `true` for explicitly promoted pages
- `audience`: `user`, `operator`, or `theory`
- `page_id`: stable identifier matching `wiki.yaml`

### 1.4 Manifest as Source of Truth

One manifest file (`.wiki/manifest.json`) captures the full state:

```json
{
  "materializer_version": "0.1.0",
  "aggregate_hash": "f1e2d3c4b5a60918",
  "pages": [
    {
      "id": "theory-overview",
      "title": "Theory: Signed Hierarchies & Semantic Navigation",
      "path": ".wiki/theory-overview.md",
      "source_hash": "a1b2c3d4e5f60718",
      "spec_hash": "9f8e7d6c5b4a3021",
      "file_hash": "3c4d5e6f7a8b9012",
      "sources": ["docs/DESIGN.md", "README.md"],
      "audience": "theory",
      "theory_scope": false
    }
  ]
}
```

The `aggregate_hash` is the hash of all `file_hash` values sorted by `id`. This allows a single comparison to detect any drift across the entire wiki.

The manifest contains no timestamps. All state is expressed through content hashes, keeping the entire output deterministic.

### 1.5 Drift Detection

Two modes:

**Local (human-readable):**
```
$ python -m model_atlas.wiki drift

Wiki drift report:
  theory-overview    STALE   source changed (docs/DESIGN.md modified)
  architecture-data  OK
  user-guide         OK
  home               STALE   page missing

2 stale, 2 current, 0 orphaned
Run `python -m model_atlas.wiki materialize` to regenerate.
```

**CI mode (nonzero exit on drift):**
```
$ python -m model_atlas.wiki drift --check
ERROR: 2 pages are stale. Run materializer to update.
Exit code: 1
```

Drift is detected by:
1. Read manifest and current `wiki.yaml`
2. If materializer version in config differs from manifest → ALL STALE (rendering logic changed)
3. For each page in config:
   a. Hash current source files, compare to `source_hash` in manifest → STALE if different (source content changed)
   b. Hash current page config entry, compare to `spec_hash` in manifest → STALE if different (title, sections, audience, or other config changed)
   c. Hash page file on disk, compare to `file_hash` in manifest → STALE if different (file was manually edited or corrupted)
   d. If page file missing → STALE (page deleted or never generated)
4. If page file exists in `.wiki/` but not in manifest → ORPHANED (page was manually created or config removed it)

### 1.6 Explicit Promotion Path

Generated pages default to `theory_scope: false`. A separate promotion registry (`wiki.yaml` or a dedicated `promotions.yaml`) lists pages that have been manually reviewed and promoted for theory ingestion:

```yaml
# In wiki.yaml
promotions:
  - page_id: theory-overview
    promoted_by: rohanvinaik
    reason: "Core thesis and data model are stable theory claims"
```

Promotion is a deliberate human action, not an automatic side effect of generation. The materializer sets `theory_scope: true` in frontmatter only for promoted pages. Theory extractors (LintGate's or any future ModelAtlas equivalent) check `theory_scope` before ingesting.

---

## 2. Architecture

```
wiki.yaml                    Source docs (README.md, docs/*.md)
    │                                │
    ▼                                ▼
┌──────────────────────────────────────┐
│         Source Map Parser            │
│  (reads config, resolves sections)   │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│        Section Extractor             │
│  (heading-based chunking,            │
│   docstring extraction)              │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│        Page Renderer                 │
│  (assembles sections in config       │
│   order, generates frontmatter,      │
│   builds index pages)                │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│        Manifest Writer               │
│  (hashes, metadata, aggregate hash)  │
└──────────────┬───────────────────────┘
               │
               ▼
    .wiki/                           .wiki/manifest.json
    ├── home.md
    ├── theory-overview.md
    ├── architecture-server.md
    ├── architecture-pipeline.md
    ├── architecture-data.md
    └── user-guide.md

               │
               ▼ (optional, separate adapter)
┌──────────────────────────────────────┐
│        Publish Adapter               │
│  (GitHub Wiki, or other backends)    │
└──────────────────────────────────────┘
```

The publish adapter is a separate module that reads `.wiki/` and pushes to a remote. It is not part of the core materialization logic. Local usefulness is independent from GitHub auth/network.

---

## 3. Module Layout

```
src/model_atlas/
  wiki/
    __init__.py           # Package init, public API
    __main__.py           # CLI entry point for `python -m model_atlas.wiki`
    config.py             # Source map parser, wiki.yaml schema
    extractor.py          # Section extraction from markdown/Python files
    renderer.py           # Page assembly, frontmatter generation
    manifest.py           # Manifest read/write, hash computation
    drift.py              # Drift detection, reporting, CI mode
    publish.py            # Publish adapter interface (future)

wiki.yaml                 # Canonical source map (repo root)
.wiki/                    # Generated output (gitignored or committed — project choice)
  manifest.json
  home.md
  theory-overview.md
  ...
```

---

## 4. Implementation Phases

### Phase 1: Core Engine (this PR)

Scope: `materialize` + `drift` + `manifest` + tests.

**Deliverables:**
- `wiki.yaml` for ModelAtlas with 5-6 pages covering the natural doc taxonomy
- Section extractor that handles heading-based markdown chunking
- Deterministic page renderer with provenance frontmatter
- Manifest writer with aggregate hashing
- Drift detector with human and CI modes
- CLI: `python -m model_atlas.wiki materialize`, `python -m model_atlas.wiki drift [--check]`
- Tests:
  - Determinism: materialize twice, assert byte-identical output
  - Drift detection: modify source, assert drift reported
  - Provenance: assert all frontmatter fields present and correct
  - Round-trip: materialize → read manifest → verify hashes match files on disk

**Not in scope:** publish adapter, theory promotion enforcement, cross-links.

### Phase 2: CI Gating

Scope: add `drift --check` to CI once output format is stable.

**Deliverables:**
- GitHub Actions step: `python -m model_atlas.wiki drift --check`
- Fails the build if materialized wiki is stale relative to source docs
- Forces docs to stay in sync with code changes

**Prerequisite:** Phase 1 output format is stable (no more frontmatter schema changes).

### Phase 3: Publish Adapter

Scope: push `.wiki/` to GitHub Wiki (or other backends).

**Deliverables:**
- Publish adapter interface (abstract base with `push(pages, manifest)`)
- GitHub Wiki adapter (depends on LintGate #282 capability layer)
- Strip frontmatter on publish (GitHub Wiki doesn't render YAML frontmatter)
- Idempotent: re-publish unchanged pages is a no-op

**Prerequisite:** LintGate #282 lands, GitHub Wiki enabled on repo.

### Phase 4: LintGate Adoption

Scope: port the core engine to LintGate, write a `wiki.yaml` for 51 source docs.

**Deliverables:**
- Same engine, different config
- Theory promotion enforcement (check `theory_scope` before theory extraction ingests wiki pages)
- Cross-link generation between Theory and Architecture pages
- 18 wiki pages (6 Theory + 12 Architecture from issue #272)

---

## 5. Design Decisions

### Output directory: `.wiki/` committed or gitignored?

**Committed** (recommended for ModelAtlas beta). Rationale:
- Small number of pages, small diffs
- CI drift check verifies committed state matches source
- Users can read wiki pages without running the materializer
- GitHub renders markdown in `.wiki/` directory if someone browses the repo

For LintGate (18 pages from 18,500 lines of source): may want gitignored + CI-generated to avoid noisy diffs.

### Section extraction: heading-based vs AST

Heading-based for Phase 1. Match sections by heading text (case-insensitive, prefix-match). This is simple, deterministic, and covers ModelAtlas's doc style. AST-based parsing can be added later if needed for complex documents.

### `sections: all` shorthand

When a page source declares `sections: all`, the entire file is included (minus any YAML frontmatter). This avoids enumerating every heading for files that map 1:1 to a wiki page.

### Cross-links

Not in Phase 1. When added, cross-links should be declared in `wiki.yaml` (not auto-discovered) to maintain determinism and explicit provenance:

```yaml
cross_links:
  - from: theory-overview
    to: architecture-server
    context: "Query engine implementation"
```

---

## 6. Audience Taxonomy

Three audiences for ModelAtlas:

| Audience | Who | What they read |
|----------|-----|----------------|
| `user` | Someone installing the MCP server | Quick start, tool reference, beta status |
| `operator` | Someone running the extraction pipeline | Pipeline phases, worker deployment, data distribution |
| `theory` | Someone understanding the architecture | Signed hierarchies, anchor semantics, why not embeddings |

LintGate adds `developer` (someone contributing to LintGate itself). The audience field is metadata for navigation and filtering, not access control.

---

## 7. Relationship to LintGate

The core engine (`config.py`, `extractor.py`, `renderer.py`, `manifest.py`, `drift.py`) is designed to be project-agnostic. The only project-specific artifact is `wiki.yaml`.

When LintGate adopts this:
1. Copy/adapt the core engine modules (or extract to a shared package)
2. Write a LintGate-specific `wiki.yaml` with 18 pages
3. Add promotion enforcement to theory extraction
4. Add cross-link generation
5. Add publish adapter using #282 capability layer

The ModelAtlas beta validates the contract. LintGate inherits it.
