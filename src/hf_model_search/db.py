"""SQLite database for the semantic network.

Stores models positioned across 7 orthogonal semantic banks, connected
through a shared anchor dictionary. This is the primary storage — the
model_metadata table is overflow for data that doesn't decompose into
the network.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Iterator

from .config import NETWORK_DB_PATH

# The seven semantic banks
BANKS = (
    "ARCHITECTURE",
    "CAPABILITY",
    "EFFICIENCY",
    "COMPATIBILITY",
    "LINEAGE",
    "DOMAIN",
    "QUALITY",
)

# Zero states for each bank (the semantic origin)
ZERO_STATES = {
    "ARCHITECTURE": "standard transformer decoder",
    "CAPABILITY": "general language model",
    "EFFICIENCY": "~7B mainstream",
    "COMPATIBILITY": "standard transformers + PyTorch",
    "LINEAGE": "base/foundational model",
    "DOMAIN": "general knowledge",
    "QUALITY": "established, mainstream adoption",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    model_id    TEXT PRIMARY KEY,
    author      TEXT,
    source      TEXT DEFAULT 'huggingface',
    display_name TEXT
);

CREATE TABLE IF NOT EXISTS model_positions (
    model_id    TEXT REFERENCES models(model_id),
    bank        TEXT,
    path_sign   INTEGER,
    path_depth  INTEGER,
    path_nodes  TEXT,
    zero_state  TEXT,
    PRIMARY KEY (model_id, bank)
);

CREATE TABLE IF NOT EXISTS model_links (
    source_id   TEXT REFERENCES models(model_id),
    target_id   TEXT REFERENCES models(model_id),
    relation    TEXT,
    weight      REAL DEFAULT 1.0,
    PRIMARY KEY (source_id, target_id, relation)
);

CREATE TABLE IF NOT EXISTS anchors (
    anchor_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT UNIQUE,
    bank        TEXT,
    category    TEXT
);

CREATE TABLE IF NOT EXISTS model_anchors (
    model_id    TEXT REFERENCES models(model_id),
    anchor_id   INTEGER REFERENCES anchors(anchor_id),
    weight      REAL DEFAULT 1.0,
    PRIMARY KEY (model_id, anchor_id)
);

CREATE TABLE IF NOT EXISTS model_metadata (
    model_id    TEXT REFERENCES models(model_id),
    key         TEXT,
    value       TEXT,
    value_type  TEXT,
    PRIMARY KEY (model_id, key)
);

CREATE INDEX IF NOT EXISTS idx_positions_bank ON model_positions(bank);
CREATE INDEX IF NOT EXISTS idx_links_source ON model_links(source_id);
CREATE INDEX IF NOT EXISTS idx_links_target ON model_links(target_id);
CREATE INDEX IF NOT EXISTS idx_model_anchors_model ON model_anchors(model_id);
CREATE INDEX IF NOT EXISTS idx_model_anchors_anchor ON model_anchors(anchor_id);
CREATE INDEX IF NOT EXISTS idx_metadata_key ON model_metadata(key);
"""

# Bootstrap anchor dictionary — initial vocabulary across all 7 banks
_BOOTSTRAP_ANCHORS = [
    # ARCHITECTURE
    ("transformer", "ARCHITECTURE", "type"),
    ("encoder-only", "ARCHITECTURE", "type"),
    ("encoder-decoder", "ARCHITECTURE", "type"),
    ("decoder-only", "ARCHITECTURE", "type"),
    ("mixture-of-experts", "ARCHITECTURE", "type"),
    ("mamba", "ARCHITECTURE", "type"),
    ("rwkv", "ARCHITECTURE", "type"),
    ("ssm", "ARCHITECTURE", "type"),
    ("hybrid", "ARCHITECTURE", "type"),
    ("diffusion", "ARCHITECTURE", "type"),
    ("vision-transformer", "ARCHITECTURE", "type"),
    # CAPABILITY
    ("instruction-following", "CAPABILITY", "training"),
    ("chat", "CAPABILITY", "training"),
    ("RLHF-tuned", "CAPABILITY", "training"),
    ("DPO-tuned", "CAPABILITY", "training"),
    ("tool-calling", "CAPABILITY", "skill"),
    ("function-calling", "CAPABILITY", "skill"),
    ("code-generation", "CAPABILITY", "skill"),
    ("code-completion", "CAPABILITY", "skill"),
    ("creative-writing", "CAPABILITY", "skill"),
    ("reasoning", "CAPABILITY", "skill"),
    ("math", "CAPABILITY", "skill"),
    ("NER", "CAPABILITY", "skill"),
    ("orchestration", "CAPABILITY", "skill"),
    ("structured-output", "CAPABILITY", "skill"),
    ("embedding", "CAPABILITY", "skill"),
    ("classification", "CAPABILITY", "skill"),
    ("translation", "CAPABILITY", "skill"),
    ("summarization", "CAPABILITY", "skill"),
    ("question-answering", "CAPABILITY", "skill"),
    ("image-generation", "CAPABILITY", "skill"),
    ("image-understanding", "CAPABILITY", "skill"),
    ("multimodal", "CAPABILITY", "skill"),
    ("time-series", "CAPABILITY", "skill"),
    ("long-context", "CAPABILITY", "feature"),
    # EFFICIENCY
    ("sub-1B", "EFFICIENCY", "size"),
    ("1B-class", "EFFICIENCY", "size"),
    ("3B-class", "EFFICIENCY", "size"),
    ("7B-class", "EFFICIENCY", "size"),
    ("13B-class", "EFFICIENCY", "size"),
    ("30B-class", "EFFICIENCY", "size"),
    ("70B-class", "EFFICIENCY", "size"),
    ("frontier-class", "EFFICIENCY", "size"),
    ("consumer-GPU-viable", "EFFICIENCY", "hardware"),
    ("edge-deployable", "EFFICIENCY", "hardware"),
    ("quantized", "EFFICIENCY", "optimization"),
    # COMPATIBILITY
    ("GGUF-available", "COMPATIBILITY", "format"),
    ("GPTQ-available", "COMPATIBILITY", "format"),
    ("AWQ-available", "COMPATIBILITY", "format"),
    ("EXL2-available", "COMPATIBILITY", "format"),
    ("safetensors", "COMPATIBILITY", "format"),
    ("ONNX-available", "COMPATIBILITY", "format"),
    ("Apple-Silicon-native", "COMPATIBILITY", "hardware"),
    ("MLX-compatible", "COMPATIBILITY", "framework"),
    ("llama-cpp-compatible", "COMPATIBILITY", "framework"),
    ("vLLM-compatible", "COMPATIBILITY", "framework"),
    ("TensorRT-compatible", "COMPATIBILITY", "framework"),
    ("transformers-compatible", "COMPATIBILITY", "framework"),
    ("diffusers-compatible", "COMPATIBILITY", "framework"),
    # LINEAGE
    ("Llama-family", "LINEAGE", "family"),
    ("Mistral-family", "LINEAGE", "family"),
    ("Qwen-family", "LINEAGE", "family"),
    ("Phi-family", "LINEAGE", "family"),
    ("Gemma-family", "LINEAGE", "family"),
    ("GPT-family", "LINEAGE", "family"),
    ("Falcon-family", "LINEAGE", "family"),
    ("StableLM-family", "LINEAGE", "family"),
    ("DeepSeek-family", "LINEAGE", "family"),
    ("Yi-family", "LINEAGE", "family"),
    ("Command-family", "LINEAGE", "family"),
    ("base-model", "LINEAGE", "role"),
    ("fine-tune", "LINEAGE", "role"),
    ("merge", "LINEAGE", "role"),
    ("distillation", "LINEAGE", "role"),
    # DOMAIN
    ("code-domain", "DOMAIN", "broad"),
    ("science-domain", "DOMAIN", "broad"),
    ("medical-domain", "DOMAIN", "broad"),
    ("legal-domain", "DOMAIN", "broad"),
    ("finance-domain", "DOMAIN", "broad"),
    ("multilingual", "DOMAIN", "broad"),
    ("creative-domain", "DOMAIN", "broad"),
    ("math-domain", "DOMAIN", "broad"),
    # QUALITY
    ("trending", "QUALITY", "signal"),
    ("high-downloads", "QUALITY", "signal"),
    ("community-favorite", "QUALITY", "signal"),
    ("official-release", "QUALITY", "signal"),
    ("experimental", "QUALITY", "signal"),
    ("deprecated", "QUALITY", "signal"),
]


def _ensure_db_dir() -> None:
    NETWORK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Get a connection to the network database."""
    _ensure_db_dir()
    conn = sqlite3.connect(str(NETWORK_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection | None = None) -> Iterator[sqlite3.Connection]:
    """Context manager for database transactions."""
    own_conn = conn is None
    c = conn if conn is not None else get_connection()
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        if own_conn:
            c.close()


def init_db(conn: sqlite3.Connection | None = None) -> None:
    """Create tables and bootstrap anchor dictionary."""
    with transaction(conn) as c:
        c.executescript(_SCHEMA)
        # Bootstrap anchors (skip duplicates)
        for label, bank, category in _BOOTSTRAP_ANCHORS:
            c.execute(
                "INSERT OR IGNORE INTO anchors (label, bank, category) VALUES (?, ?, ?)",
                (label, bank, category),
            )


# --- CRUD Operations ---


def insert_model(
    conn: sqlite3.Connection,
    model_id: str,
    author: str = "",
    source: str = "huggingface",
    display_name: str = "",
) -> None:
    """Insert or update a model entity."""
    conn.execute(
        """INSERT INTO models (model_id, author, source, display_name)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(model_id) DO UPDATE SET
               author=excluded.author,
               source=excluded.source,
               display_name=excluded.display_name""",
        (model_id, author, source, display_name or model_id.split("/")[-1]),
    )


def set_position(
    conn: sqlite3.Connection,
    model_id: str,
    bank: str,
    path_sign: int,
    path_depth: int,
    path_nodes: list[str] | None = None,
) -> None:
    """Set a model's position in a semantic bank."""
    conn.execute(
        """INSERT INTO model_positions (model_id, bank, path_sign, path_depth, path_nodes, zero_state)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(model_id, bank) DO UPDATE SET
               path_sign=excluded.path_sign,
               path_depth=excluded.path_depth,
               path_nodes=excluded.path_nodes""",
        (
            model_id,
            bank,
            path_sign,
            path_depth,
            json.dumps(path_nodes) if path_nodes else None,
            ZERO_STATES.get(bank, ""),
        ),
    )


def add_link(
    conn: sqlite3.Connection,
    source_id: str,
    target_id: str,
    relation: str,
    weight: float = 1.0,
) -> None:
    """Add an explicit relationship between two models."""
    conn.execute(
        """INSERT OR REPLACE INTO model_links (source_id, target_id, relation, weight)
           VALUES (?, ?, ?, ?)""",
        (source_id, target_id, relation, weight),
    )


def get_or_create_anchor(
    conn: sqlite3.Connection, label: str, bank: str, category: str = ""
) -> int:
    """Get an anchor's ID, creating it if it doesn't exist."""
    row = conn.execute(
        "SELECT anchor_id FROM anchors WHERE label = ?", (label,)
    ).fetchone()
    if row:
        return row["anchor_id"]
    cursor = conn.execute(
        "INSERT INTO anchors (label, bank, category) VALUES (?, ?, ?)",
        (label, bank, category),
    )
    return int(cursor.lastrowid)  # always set after INSERT


def link_anchor(
    conn: sqlite3.Connection,
    model_id: str,
    anchor_id: int,
    weight: float = 1.0,
) -> None:
    """Link a model to an anchor."""
    conn.execute(
        """INSERT OR REPLACE INTO model_anchors (model_id, anchor_id, weight)
           VALUES (?, ?, ?)""",
        (model_id, anchor_id, weight),
    )


def set_metadata(
    conn: sqlite3.Connection,
    model_id: str,
    key: str,
    value: str,
    value_type: str = "str",
) -> None:
    """Set overflow metadata for a model."""
    conn.execute(
        """INSERT OR REPLACE INTO model_metadata (model_id, key, value, value_type)
           VALUES (?, ?, ?, ?)""",
        (model_id, key, value, value_type),
    )


# --- Query Helpers ---


def get_model(conn: sqlite3.Connection, model_id: str) -> dict | None:
    """Get a model with all its positions, anchors, and metadata."""
    row = conn.execute(
        "SELECT * FROM models WHERE model_id = ?", (model_id,)
    ).fetchone()
    if not row:
        return None
    model = dict(row)

    # Positions
    positions = conn.execute(
        "SELECT bank, path_sign, path_depth, path_nodes, zero_state FROM model_positions WHERE model_id = ?",
        (model_id,),
    ).fetchall()
    model["positions"] = {
        p["bank"]: {
            "sign": p["path_sign"],
            "depth": p["path_depth"],
            "nodes": json.loads(p["path_nodes"]) if p["path_nodes"] else [],
            "zero_state": p["zero_state"],
        }
        for p in positions
    }

    # Anchors
    anchors = conn.execute(
        """SELECT a.label, a.bank, a.category, ma.weight
           FROM model_anchors ma JOIN anchors a ON ma.anchor_id = a.anchor_id
           WHERE ma.model_id = ?""",
        (model_id,),
    ).fetchall()
    model["anchors"] = [
        {
            "label": a["label"],
            "bank": a["bank"],
            "category": a["category"],
            "weight": a["weight"],
        }
        for a in anchors
    ]

    # Links
    outgoing = conn.execute(
        "SELECT target_id, relation, weight FROM model_links WHERE source_id = ?",
        (model_id,),
    ).fetchall()
    incoming = conn.execute(
        "SELECT source_id, relation, weight FROM model_links WHERE target_id = ?",
        (model_id,),
    ).fetchall()
    model["links"] = {
        "outgoing": [dict(link) for link in outgoing],
        "incoming": [dict(link) for link in incoming],
    }

    # Metadata
    metadata = conn.execute(
        "SELECT key, value, value_type FROM model_metadata WHERE model_id = ?",
        (model_id,),
    ).fetchall()
    model["metadata"] = {
        m["key"]: {"value": m["value"], "type": m["value_type"]} for m in metadata
    }

    return model


def get_anchor_set(conn: sqlite3.Connection, model_id: str) -> set[str]:
    """Get the set of anchor labels for a model."""
    rows = conn.execute(
        """SELECT a.label FROM model_anchors ma
           JOIN anchors a ON ma.anchor_id = a.anchor_id
           WHERE ma.model_id = ?""",
        (model_id,),
    ).fetchall()
    return {r["label"] for r in rows}


def find_models_by_anchor(conn: sqlite3.Connection, anchor_label: str) -> list[str]:
    """Find all model_ids that have a given anchor."""
    rows = conn.execute(
        """SELECT ma.model_id FROM model_anchors ma
           JOIN anchors a ON ma.anchor_id = a.anchor_id
           WHERE a.label = ?""",
        (anchor_label,),
    ).fetchall()
    return [r["model_id"] for r in rows]


def find_models_by_bank_range(
    conn: sqlite3.Connection,
    bank: str,
    min_signed: int | None = None,
    max_signed: int | None = None,
) -> list[dict]:
    """Find models within a signed position range in a bank.

    Signed position = path_sign * path_depth.
    """
    query = """
        SELECT model_id, path_sign, path_depth, (path_sign * path_depth) as signed_pos
        FROM model_positions WHERE bank = ?
    """
    params: list = [bank]
    if min_signed is not None:
        query += " AND (path_sign * path_depth) >= ?"
        params.append(min_signed)
    if max_signed is not None:
        query += " AND (path_sign * path_depth) <= ?"
        params.append(max_signed)
    query += " ORDER BY signed_pos"
    return [dict(r) for r in conn.execute(query, params).fetchall()]


def network_stats(conn: sqlite3.Connection) -> dict:
    """Get summary statistics about the network."""
    model_count = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    anchor_count = conn.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
    link_count = conn.execute("SELECT COUNT(*) FROM model_links").fetchone()[0]
    position_count = conn.execute("SELECT COUNT(*) FROM model_positions").fetchone()[0]

    # Per-bank breakdown
    bank_counts = {}
    for row in conn.execute(
        "SELECT bank, COUNT(*) as cnt FROM model_positions GROUP BY bank"
    ).fetchall():
        bank_counts[row["bank"]] = row["cnt"]

    # Source breakdown
    source_counts = {}
    for row in conn.execute(
        "SELECT source, COUNT(*) as cnt FROM models GROUP BY source"
    ).fetchall():
        source_counts[row["source"]] = row["cnt"]

    return {
        "total_models": model_count,
        "total_anchors": anchor_count,
        "total_links": link_count,
        "total_positions": position_count,
        "models_per_bank": bank_counts,
        "models_per_source": source_counts,
    }
