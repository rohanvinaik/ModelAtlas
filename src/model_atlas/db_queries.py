"""Read-only query helpers for the semantic network database.

Separated from db.py (schema + CRUD) for cohesion — these functions
only read from the database, never write.
"""

from __future__ import annotations

import json
import sqlite3
from math import log


def get_model(conn: sqlite3.Connection, model_id: str) -> dict | None:
    """Get a model with all its positions, anchors, and metadata."""
    row = conn.execute(
        "SELECT * FROM models WHERE model_id = ?", (model_id,)
    ).fetchone()
    if not row:
        return None
    model = dict(row)

    model["positions"] = _fetch_positions(conn, model_id)
    model["anchors"] = _fetch_anchors(conn, model_id)
    model["links"] = _fetch_links(conn, model_id)
    model["metadata"] = _fetch_metadata(conn, model_id)

    return model


def _fetch_positions(conn: sqlite3.Connection, model_id: str) -> dict:
    rows = conn.execute(
        "SELECT bank, path_sign, path_depth, path_nodes, zero_state "
        "FROM model_positions WHERE model_id = ?",
        (model_id,),
    ).fetchall()
    return {
        p["bank"]: {
            "sign": p["path_sign"],
            "depth": p["path_depth"],
            "nodes": json.loads(p["path_nodes"]) if p["path_nodes"] else [],
            "zero_state": p["zero_state"],
        }
        for p in rows
    }


def _fetch_anchors(conn: sqlite3.Connection, model_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT a.label, a.bank, a.category, ma.weight, ma.confidence "
        "FROM model_anchors ma JOIN anchors a ON ma.anchor_id = a.anchor_id "
        "WHERE ma.model_id = ?",
        (model_id,),
    ).fetchall()
    return [
        {
            "label": a["label"],
            "bank": a["bank"],
            "category": a["category"],
            "weight": a["weight"],
            "confidence": a["confidence"],
        }
        for a in rows
    ]


def _fetch_links(conn: sqlite3.Connection, model_id: str) -> dict:
    outgoing = conn.execute(
        "SELECT target_id, relation, weight FROM model_links WHERE source_id = ?",
        (model_id,),
    ).fetchall()
    incoming = conn.execute(
        "SELECT source_id, relation, weight FROM model_links WHERE target_id = ?",
        (model_id,),
    ).fetchall()
    return {
        "outgoing": [dict(link) for link in outgoing],
        "incoming": [dict(link) for link in incoming],
    }


def _fetch_metadata(conn: sqlite3.Connection, model_id: str) -> dict:
    rows = conn.execute(
        "SELECT key, value, value_type FROM model_metadata WHERE model_id = ?",
        (model_id,),
    ).fetchall()
    return {m["key"]: {"value": m["value"], "type": m["value_type"]} for m in rows}


def get_anchor_set(conn: sqlite3.Connection, model_id: str) -> set[str]:
    """Get the set of anchor labels for a model."""
    rows = conn.execute(
        "SELECT a.label FROM model_anchors ma "
        "JOIN anchors a ON ma.anchor_id = a.anchor_id "
        "WHERE ma.model_id = ?",
        (model_id,),
    ).fetchall()
    return {r["label"] for r in rows}


def find_models_by_anchor(conn: sqlite3.Connection, anchor_label: str) -> list[str]:
    """Find all model_ids that have a given anchor."""
    rows = conn.execute(
        "SELECT ma.model_id FROM model_anchors ma "
        "JOIN anchors a ON ma.anchor_id = a.anchor_id "
        "WHERE a.label = ?",
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
    query = (
        "SELECT model_id, path_sign, path_depth, (path_sign * path_depth) as signed_pos "
        "FROM model_positions WHERE bank = ?"
    )
    params: list = [bank]
    if min_signed is not None:
        query += " AND (path_sign * path_depth) >= ?"
        params.append(min_signed)
    if max_signed is not None:
        query += " AND (path_sign * path_depth) <= ?"
        params.append(max_signed)
    query += " ORDER BY signed_pos"
    return [dict(r) for r in conn.execute(query, params).fetchall()]


def compute_anchor_idf(conn: sqlite3.Connection) -> dict[str, float]:
    """Compute IDF for all anchors: log(N / count_models_with_anchor).

    Returns {anchor_label: idf_value}. Rare anchors get high IDF,
    ubiquitous anchors get low IDF.
    """
    total = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    if total == 0:
        return {}
    rows = conn.execute(
        "SELECT a.label, COUNT(ma.model_id) as cnt "
        "FROM anchors a LEFT JOIN model_anchors ma ON a.anchor_id = ma.anchor_id "
        "GROUP BY a.anchor_id, a.label"
    ).fetchall()
    return {r["label"]: log(total / max(r["cnt"], 1)) for r in rows}


def _in_clause(n: int) -> str:
    """Build a parameterized IN clause with *n* placeholders."""
    return ",".join("?" for _ in range(n))


def batch_get_positions(
    conn: sqlite3.Connection, model_ids: list[str]
) -> dict[str, dict[str, tuple[int, int]]]:
    """Batch-fetch bank positions for a set of models.

    Returns {model_id: {bank: (sign, depth)}}.
    """
    if not model_ids:
        return {}
    sql = (
        "SELECT model_id, bank, path_sign, path_depth "
        "FROM model_positions WHERE model_id IN (%s)" % _in_clause(len(model_ids))
    )
    rows = conn.execute(sql, model_ids).fetchall()
    result: dict[str, dict[str, tuple[int, int]]] = {}
    for r in rows:
        result.setdefault(r["model_id"], {})[r["bank"]] = (
            r["path_sign"],
            r["path_depth"],
        )
    return result


def batch_get_anchor_sets(
    conn: sqlite3.Connection, model_ids: list[str]
) -> dict[str, set[str]]:
    """Batch-fetch anchor label sets for a set of models.

    Returns {model_id: {anchor_label, ...}}.
    """
    if not model_ids:
        return {}
    sql = (
        "SELECT ma.model_id, a.label "
        "FROM model_anchors ma JOIN anchors a ON ma.anchor_id = a.anchor_id "
        "WHERE ma.model_id IN (%s)" % _in_clause(len(model_ids))
    )
    rows = conn.execute(sql, model_ids).fetchall()
    result: dict[str, set[str]] = {}
    for r in rows:
        result.setdefault(r["model_id"], set()).add(r["label"])
    return result


def network_stats(conn: sqlite3.Connection) -> dict:
    """Get summary statistics about the network."""
    model_count = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    anchor_count = conn.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
    link_count = conn.execute("SELECT COUNT(*) FROM model_links").fetchone()[0]
    position_count = conn.execute("SELECT COUNT(*) FROM model_positions").fetchone()[0]

    bank_counts = {
        row["bank"]: row["cnt"]
        for row in conn.execute(
            "SELECT bank, COUNT(*) as cnt FROM model_positions GROUP BY bank"
        ).fetchall()
    }
    source_counts = {
        row["source"]: row["cnt"]
        for row in conn.execute(
            "SELECT source, COUNT(*) as cnt FROM models GROUP BY source"
        ).fetchall()
    }

    return {
        "total_models": model_count,
        "total_anchors": anchor_count,
        "total_links": link_count,
        "total_positions": position_count,
        "models_per_bank": bank_counts,
        "models_per_source": source_counts,
    }
