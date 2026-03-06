---
generated: true
generated_from: 
  - README.md
source_hash: fe625fdc9c58f780
spec_hash: b442a02451288ea0
file_hash: ab5663de7b47fff8
materializer_version: "0.1.0"
theory_scope: false
audience: user
page_id: user-guide
---

# User Guide: Installation & Usage


## Beta status

ModelAtlas is in active beta. The semantic network contains **19,498 models** with **166 anchors** across all 8 banks and **128K+ model-anchor links**. The top-downloaded models on HuggingFace are all present and enriched.

**What works today:**
- Navigation queries return meaningfully different results than keyword search. "Small code models with tool-calling" surfaces LiquidAI LFM-1.2B, Qwen3-4B tool-calling variants, and Llama-3.2-1B function-calling adapters — real answers to a query HuggingFace can't express.
- The 8-bank coordinate system captures structural relationships that tags and filters miss. Signed directions, anchor set intersections, and IDF-weighted similarity all function as described.
- ~7,300 models have LLM-generated enrichment (summaries + capability anchors) beyond deterministic extraction. The remaining ~12K have full Tier 1+2 structural data.

**Validation:**
- ~3,000 models independently validated by Gemini against raw HuggingFace metadata.
- A multi-phase correction pipeline (deterministic audit, dictionary expansion, LLM healing) is actively improving anchor accuracy, with a target of 90-95%+ in the final network.

**What this means for users:**
The network is directionally correct and dense enough for the core use case: giving an LLM a structural sense of model space it doesn't have in its weights. Popular models are well-covered. The long tail is still being refined. Expect anchor accuracy to improve steadily as the correction pipeline converges.


## Quick start

**1. Install:**

```bash
uv sync
```

**2. Download the pre-built network:**

The semantic network is distributed as a SQLite file attached to [GitHub Releases](https://github.com/rohanvinaik/ModelAtlas/releases). Download the latest `network.db` and place it in the cache directory:

```bash

## Usage

The primary tool is `navigate_models`. The calling LLM fills in structured parameters; ModelAtlas does deterministic scoring.

```python
navigate_models(
    efficiency=-1,           # small
    capability=1,            # capable
    require_anchors=["code-generation"],
    prefer_anchors=["instruction-following", "tool-calling"],
    avoid_anchors=["embedding"]
)