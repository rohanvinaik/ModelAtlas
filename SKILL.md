---
name: hf-model-search
description: Navigate a semantic network of ML models — find models by architecture, capability, efficiency, lineage, and domain specialization
---

# HuggingFace Model Search

## When to Use

Use this tool when you need to:
- Find ML models matching specific technical characteristics ("small transformer with tool-calling that runs on Apple Silicon")
- Explore model lineage and family trees ("what fine-tunes exist for Llama 3.1 8B?")
- Navigate model space relationally ("models like X but smaller and more code-focused")
- Compare models structurally ("what anchors do these two share? what distinguishes them?")
- Find fine-tuning candidates ("most general base model with capability Y")
- Discover what exists in a capability space ("what tool-calling models exist under 3B params?")

## How It Works

Models are positioned in a **7-bank semantic space**. Each bank is an orthogonal dimension with a meaningful zero state. Queries navigate this space rather than filtering flat columns.

### The Banks

| Bank | Zero State | What It Captures |
|------|-----------|-----------------|
| **ARCHITECTURE** | Standard transformer decoder | Structural type. ~95% of filtering starts here. |
| **CAPABILITY** | General language model | What it can do: code, reasoning, tool-calling, NER, orchestration |
| **EFFICIENCY** | ~7B mainstream | Size/resource profile. Tiny ↔ massive. |
| **COMPATIBILITY** | Standard transformers + PyTorch | Formats, frameworks, hardware targets |
| **LINEAGE** | Base model of a family | Family tree. Predecessors (negative) ↔ derivatives (positive) |
| **DOMAIN** | General knowledge | Specialization depth. General ↔ ultra-narrow |
| **QUALITY** | Established, mainstream | Community signals. Legacy ↔ trending |

### Anchors

Models link to shared characteristics from an **anchor dictionary** ("instruction-following", "tool-calling", "Apple-Silicon-native", "RLHF-tuned", etc.). Models sharing anchors are semantically related. Anchor sets support set operations:
- **Intersection** = shared capabilities
- **Symmetric difference** = distinguishing features
- **Jaccard similarity** = overall semantic overlap

## Available MCP Tools

1. **hf_search_models** — Primary navigational search. Combines bank-position constraints with anchor similarity. Always start here.
2. **hf_get_model_detail** — Full network position for one model: all 7 bank positions, anchor set, lineage links, overflow metadata.
3. **hf_compare_models** — Structural comparison via anchor set operations and per-bank position deltas.
4. **hf_build_index** — Fetch models from source, run through extraction pipeline, add to network. Additive — multiple calls enrich the same network.
5. **hf_index_status** — Network stats: total models, bank breakdowns, anchor dictionary size.

## Tool Reference

### hf_search_models

The primary tool. Translates natural language queries into bank-position constraints + anchor matching.

**Key parameters:**
- `query` (required): Describe what you're looking for in natural language
- `task`: Pipeline tag — constrains ARCHITECTURE and CAPABILITY banks
- `author`: Filter by org
- `library`: Constrains COMPATIBILITY bank (transformers, gguf, mlx, diffusers)
- `min_likes`: QUALITY bank minimum threshold
- `limit`: How many results (default 20)

**Output includes:** Per-result bank positions, anchor overlap with query, lineage info, and overflow metadata.

### hf_build_index

Fetches models from a source API, runs them through the extraction pipeline, and adds them to the semantic network. This is **additive** — each call enriches the same network, never fragments it.

```
hf_build_index(task="text-generation", limit=2000, min_likes=10)  # Broad text models
hf_build_index(task="feature-extraction", limit=1000, min_likes=5)  # Embedding models
hf_build_index(author="meta-llama", limit=500)  # Everything from Meta
# All three calls ADD to the same network
```

**When to build:**
- The network is empty or sparse for the region you're searching
- `hf_index_status` shows low coverage for the relevant banks
- You need models from a source/task that hasn't been indexed yet

**When NOT to build:**
- The network already has good coverage for your query
- You're doing a simple name-resolution query (fuzzy matching handles that without the network)

### hf_get_model_detail

Returns the full semantic profile of one model:
- Position in all 7 banks (signed distance from each zero state)
- Complete anchor set (what characteristics it has)
- Lineage links (what it's fine-tuned from, what's fine-tuned from it)
- Overflow metadata (benchmark scores, exact dates, file sizes, license)

Use after search to understand WHY a model was returned and how it relates to others.

### hf_compare_models

Structural comparison of 2+ models using anchor set operations:
- **Shared anchors**: what they have in common
- **Distinguishing anchors**: what makes each one unique
- **Per-bank position delta**: how they differ along each dimension
- **Jaccard similarity**: overall semantic overlap score

Use when the user is choosing between specific models.

## Best Practices

**DO:**
- Think about queries as navigation, not filtering. "Models like X but more Y" is a direction in semantic space.
- Use `hf_get_model_detail` to understand a model's full semantic profile before recommending it
- Use `hf_compare_models` to show the user exactly HOW two models differ
- Build the index additively — multiple focused builds are better than one huge one
- Use the ARCHITECTURE bank first for filtering — it's the cleanest signal

**DON'T:**
- Don't think of this as a keyword search engine. The value is relational/hierarchical.
- Don't rebuild the entire index when you just need to add models from a new source/task
- Don't ignore lineage — the user often wants "the best base model to fine-tune", not just "the best model"
- Don't skip bank position info in results — it tells the user WHERE in model space this sits

## Workflow Examples

### Finding a fine-tuning candidate
```
# "I need a small base model I can fine-tune for code with LoRA on Apple Silicon"
hf_search_models(query="small base model for code fine-tuning, MLX compatible", library="mlx")
# Results show models positioned: EFFICIENCY(-1 to -2), CAPABILITY(+code), LINEAGE(0 = base), COMPATIBILITY(+MLX)
# Pick the best candidate, then check its full profile:
hf_get_model_detail(model_id="Qwen/Qwen2.5-Coder-1.5B")
```

### Exploring a model family
```
# "What exists in the Llama 3 family?"
hf_search_models(query="Llama 3", author="meta-llama")
# Results show the family tree via LINEAGE bank positions:
#   Llama-3-8B (LINEAGE: 0, base) → Llama-3-8B-Instruct (LINEAGE: +1, fine-tune)
#   → community/Llama-3-8B-Instruct-GGUF-Q4 (LINEAGE: +3, derivative)
```

### Comparing models structurally
```
# "Should I use Qwen2.5-Coder-7B or CodeLlama-7B?"
hf_compare_models(model_ids=["Qwen/Qwen2.5-Coder-7B", "codellama/CodeLlama-7b-hf"])
# Output shows:
#   Shared anchors: code-generation, instruction-following, 7B-class
#   Qwen-only: tool-calling, structured-output, multilingual
#   CodeLlama-only: infilling, Llama-family
#   Bank deltas: CAPABILITY(Qwen +1 richer), LINEAGE(different families), DOMAIN(both code-specialized)
```

### Navigational query
```
# "I have Mistral-7B-Instruct but need something smaller with tool-calling"
hf_search_models(query="like Mistral-7B-Instruct but smaller with tool-calling")
# The search navigates: same CAPABILITY region, decrease EFFICIENCY, add "tool-calling" anchor
# Returns models that share Mistral's anchor profile but are positioned smaller in EFFICIENCY bank
```
