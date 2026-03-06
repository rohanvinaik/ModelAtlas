# Getting Started

**In five minutes you will have a navigable semantic network of 19K+ ML models running locally, queryable through any MCP-compatible client.**

---

## Problem Space

HuggingFace Hub is a file host with search. You can filter by task, sort by downloads, and grep model cards. But you cannot ask "what's similar to this model but smaller and more code-focused?" — that requires structural understanding that HuggingFace doesn't encode.

ModelAtlas adds that structure. The pre-built network contains positions across 8 semantic dimensions, 170 anchor labels, and explicit model-to-model links — all queryable through a local MCP server.

---

## Install

```bash
git clone https://github.com/rohanvinaik/ModelAtlas.git
cd ModelAtlas
uv sync
```

## Download the Pre-Built Network

The semantic network is a ~80MB SQLite file distributed via [GitHub Releases](https://github.com/rohanvinaik/ModelAtlas/releases):

```bash
mkdir -p ~/.cache/model-atlas
curl -L -o ~/.cache/model-atlas/network.db \
  https://github.com/rohanvinaik/ModelAtlas/releases/latest/download/network.db
```

Without this file, ModelAtlas starts with an empty database. You *can* build your own via `hf_build_index`, but the pre-built network includes multi-tier extraction that took days of distributed compute to produce.

## Configure Your MCP Client

Add ModelAtlas to your client's MCP configuration:

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "model-atlas": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/ModelAtlas", "model-atlas"]
    }
  }
}
```

**Claude Code** (`.mcp.json` in your project root):
```json
{
  "mcpServers": {
    "model-atlas": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/ModelAtlas", "model-atlas"]
    }
  }
}
```

Replace `/path/to/ModelAtlas` with the actual path to your clone.

## Your First Query

Once configured, ask your LLM client naturally:

> "Find me a small code model with instruction-following that runs on consumer hardware"

The LLM will call `navigate_models` with decomposed parameters. ModelAtlas returns scored results with explanations of *why* each model matched — which [anchors](Glossary#anchor) contributed, how each [bank](Glossary#bank) aligned.

See [Query Examples](Query-Examples) for more of what you can ask.

---

## What This Is Not

- **Not a HuggingFace replacement.** HF is a data source. ModelAtlas adds the structural layer HF doesn't expose.
- **Not an API you call directly.** The MCP server is the interface; the LLM is the user-facing layer.
- **Not a cloud service.** Everything runs locally. The database is a file on your disk.

---

## Related Concepts

- [Query Examples](Query-Examples) — cookbook of real queries
- [The Gap](The-Gap) — why this tool exists
- [Data Distribution](Data-Distribution) — versioning and update policy

---

*[← Home](Home) · [Query Examples →](Query-Examples)*
