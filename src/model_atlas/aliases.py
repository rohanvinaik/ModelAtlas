"""First-class alias resolver.

Sparse-wiki's silent-coverage lesson: exact-label lookup was the single
biggest hole — 851K Wikipedia redirects filled it. Mirror the shape here:

  * `anchor_aliases` table — alias → canonical anchor_id (case-insensitive)
  * `model_aliases` table  — alias → canonical model_id (case-insensitive)

Called at the top of any label / model resolution path. Cheap: indexed
lookup, no fuzzy scoring. Fuzzy match still exists downstream for the
"never-heard-of-it" case; aliases catch the "I said `gguf` and meant
`GGUF-available`" case that fuzzy should not have to reach for.
"""
from __future__ import annotations

import sqlite3


_ALIAS_SCHEMA = """
CREATE TABLE IF NOT EXISTS anchor_aliases (
    alias TEXT NOT NULL,
    anchor_id INTEGER NOT NULL,
    source TEXT DEFAULT '',
    PRIMARY KEY (alias, anchor_id),
    FOREIGN KEY (anchor_id) REFERENCES anchors(anchor_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_anchor_aliases_alias ON anchor_aliases(alias);

CREATE TABLE IF NOT EXISTS model_aliases (
    alias TEXT NOT NULL,
    model_id TEXT NOT NULL,
    source TEXT DEFAULT '',
    PRIMARY KEY (alias, model_id),
    FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_model_aliases_alias ON model_aliases(alias);
"""


def ensure_alias_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_ALIAS_SCHEMA)
    conn.commit()


def _norm(s: str) -> str:
    """Alias-key normalization: lowercase, strip, collapse whitespace,
    strip common separators. `Qwen 2.5-Coder` and `qwen2.5 coder` should
    map to the same key."""
    s = s.strip().lower()
    for ch in " -_.":
        s = s.replace(ch, "")
    return s


def resolve_anchor(conn: sqlite3.Connection, mention: str) -> int | None:
    """Return anchor_id for `mention` if any alias or canonical label matches."""
    # Canonical label direct match
    row = conn.execute(
        "SELECT anchor_id FROM anchors WHERE lower(label) = ?",
        (mention.lower(),),
    ).fetchone()
    if row:
        return int(row[0])
    # Alias table (normalized)
    key = _norm(mention)
    row = conn.execute(
        "SELECT anchor_id FROM anchor_aliases WHERE alias = ? LIMIT 1",
        (key,),
    ).fetchone()
    if row:
        return int(row[0])
    return None


def resolve_model(conn: sqlite3.Connection, mention: str) -> str | None:
    """Return canonical model_id for `mention` if any alias or exact-id matches."""
    row = conn.execute(
        "SELECT model_id FROM models WHERE lower(model_id) = ?",
        (mention.lower(),),
    ).fetchone()
    if row:
        return str(row[0])
    key = _norm(mention)
    row = conn.execute(
        "SELECT model_id FROM model_aliases WHERE alias = ? LIMIT 1",
        (key,),
    ).fetchone()
    if row:
        return str(row[0])
    return None


def add_anchor_alias(
    conn: sqlite3.Connection, alias: str, anchor_id: int, *, source: str = ""
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO anchor_aliases (alias, anchor_id, source) VALUES (?, ?, ?)",
        (_norm(alias), anchor_id, source),
    )


def add_model_alias(
    conn: sqlite3.Connection, alias: str, model_id: str, *, source: str = ""
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO model_aliases (alias, model_id, source) VALUES (?, ?, ?)",
        (_norm(alias), model_id, source),
    )


# ---------------------------------------------------------------------------
# Seed rules — quick-and-dirty coverage of the obvious anchor aliases
# ---------------------------------------------------------------------------

# (alias_source, canonical_label) — seeded at bootstrap and re-seedable.
# Only static, unambiguous mappings; anything model-specific goes through
# the model_aliases table populated separately.
_ANCHOR_ALIAS_SEEDS: tuple[tuple[str, str], ...] = (
    # Format families
    ("gguf", "GGUF-available"),
    ("gptq", "GPTQ-available"),
    ("awq", "AWQ-available"),
    ("exl2", "EXL2-available"),
    ("onnx", "ONNX-available"),
    ("st", "safetensors"),
    ("safe tensors", "safetensors"),
    # Common model type shorthands
    ("qwen2", "Qwen-family"),
    ("qwen 2", "Qwen-family"),
    ("qwen2.5", "Qwen-family"),
    ("qwen3", "Qwen-family"),
    ("llama2", "Llama-family"),
    ("llama3", "Llama-family"),
    ("llama-3", "Llama-family"),
    ("mistral7b", "Mistral-family"),
    ("mixtral", "Mistral-family"),
    ("phi3", "Phi-family"),
    ("gemma2", "Gemma-family"),
    ("gemma3", "Gemma-family"),
    ("deepseek r1", "DeepSeek-family"),
    ("deepseek v2", "DeepSeek-family"),
    ("deepseek v3", "DeepSeek-family"),
    # Capability shorthands
    ("code", "code-generation"),
    ("chat model", "chat"),
    ("tool use", "tool-calling"),
    ("function call", "function-calling"),
    ("embed", "embedding"),
    ("multimodal model", "multimodal"),
    # Efficiency shorthands
    ("small", "sub-1B"),
    ("tiny", "sub-1B"),
    ("edge", "edge-deployable"),
    ("consumer gpu", "consumer-GPU-viable"),
    ("quant", "quantized"),
    ("quantised", "quantized"),
    # Context tiers
    ("32k", "long-context-32k"),
    ("32k context", "long-context-32k"),
    ("128k", "long-context-128k"),
    ("128k context", "long-context-128k"),
    ("1m context", "long-context-1m"),
    # Compatibility
    ("mlx", "Apple-Silicon-native"),
    ("apple silicon", "Apple-Silicon-native"),
    ("m1", "Apple-Silicon-native"),
    ("m2", "Apple-Silicon-native"),
    ("cpu", "CPU-inference"),
    # Architecture
    ("moe", "mixture-of-experts"),
    ("mixture of experts", "mixture-of-experts"),
    ("gqa", "grouped-query-attention"),
    ("vit", "vision-transformer"),
    ("diffusion model", "diffusion"),
    ("ssm model", "ssm"),
)


def seed_anchor_aliases(conn: sqlite3.Connection) -> int:
    """Idempotently seed the anchor_aliases table from _ANCHOR_ALIAS_SEEDS.

    Returns the number of aliases inserted this call. Existing (alias,
    anchor_id) pairs are skipped by the PRIMARY KEY constraint."""
    ensure_alias_schema(conn)
    n_added = 0
    for alias, canonical in _ANCHOR_ALIAS_SEEDS:
        row = conn.execute(
            "SELECT anchor_id FROM anchors WHERE label = ?", (canonical,)
        ).fetchone()
        if not row:
            continue
        cur = conn.execute(
            "INSERT OR IGNORE INTO anchor_aliases (alias, anchor_id, source) VALUES (?, ?, ?)",
            (_norm(alias), int(row[0]), "seed"),
        )
        n_added += cur.rowcount
    conn.commit()
    return n_added
