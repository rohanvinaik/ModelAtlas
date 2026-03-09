"""D1: Deterministic audit of C2 anchor assignments.

Compares C2-assigned anchors (confidence=0.5) against deterministic signals
from extraction/patterns.py. Mismatch types: contradiction, gap,
confidence_conflict, unsupported.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field

from . import db
from .extraction.patterns import (
    _CAPABILITY_PATTERNS,
    _DOMAIN_PATTERNS,
)

logger = logging.getLogger(__name__)


@dataclass
class AuditResult:
    """Summary of a D1 audit run."""

    run_id: str
    total_audited: int = 0
    total_mismatches: int = 0
    per_bank_rates: dict[str, float] = field(default_factory=dict)
    per_type_counts: dict[str, int] = field(default_factory=dict)


def _run_capability_patterns(searchable: str) -> list[str]:
    """Run capability patterns against searchable text, return matched labels."""
    found: list[str] = []
    for pattern, anchor in _CAPABILITY_PATTERNS:
        if re.search(pattern, searchable, re.IGNORECASE):
            found.append(anchor)
    return found


def _run_domain_patterns(searchable: str) -> list[tuple[str, int]]:
    """Run domain patterns against searchable text, return (label, depth) pairs."""
    found: list[tuple[str, int]] = []
    seen: set[str] = set()
    for pattern, anchor, depth in _DOMAIN_PATTERNS:
        if re.search(pattern, searchable, re.IGNORECASE) and anchor not in seen:
            found.append((anchor, depth))
            seen.add(anchor)
    return found


def _get_c2_anchors(
    conn: sqlite3.Connection, model_id: str
) -> dict[str, list[tuple[str, float]]]:
    """Get C2-assigned anchors grouped by bank. Returns {bank: [(label, confidence)]}."""
    rows = conn.execute(
        """SELECT a.label, a.bank, ma.confidence
           FROM model_anchors ma
           JOIN anchors a ON ma.anchor_id = a.anchor_id
           WHERE ma.model_id = ? AND ma.confidence = 0.5""",
        (model_id,),
    ).fetchall()
    by_bank: dict[str, list[tuple[str, float]]] = {}
    for r in rows:
        by_bank.setdefault(r["bank"], []).append((r["label"], r["confidence"]))
    return by_bank


def _get_det_anchors(
    conn: sqlite3.Connection, model_id: str
) -> dict[str, list[tuple[str, float]]]:
    """Get deterministic/pattern anchors (confidence >= 0.8) grouped by bank."""
    rows = conn.execute(
        """SELECT a.label, a.bank, ma.confidence
           FROM model_anchors ma
           JOIN anchors a ON ma.anchor_id = a.anchor_id
           WHERE ma.model_id = ? AND ma.confidence >= 0.8""",
        (model_id,),
    ).fetchall()
    by_bank: dict[str, list[tuple[str, float]]] = {}
    for r in rows:
        by_bank.setdefault(r["bank"], []).append((r["label"], r["confidence"]))
    return by_bank


def _build_searchable(model_id: str, raw: dict, pipeline_tag: str) -> str:
    """Build searchable text from raw_json and metadata."""
    parts = [model_id, raw.get("author", ""), pipeline_tag, *raw.get("tags", [])]
    return " ".join(parts).lower()


def _get_model_context(
    conn: sqlite3.Connection,
    ingest_conn: sqlite3.Connection | None,
    model_id: str,
) -> tuple[str, dict]:
    """Retrieve pipeline_tag and raw_json for a model."""
    row = conn.execute(
        "SELECT value FROM model_metadata WHERE model_id = ? AND key = 'pipeline_tag'",
        (model_id,),
    ).fetchone()
    pipeline_tag = row[0] if row else ""

    raw: dict = {}
    if ingest_conn is not None:
        row = ingest_conn.execute(
            "SELECT raw_json FROM ingest_models WHERE model_id = ?",
            (model_id,),
        ).fetchone()
        if row and row[0]:
            try:
                raw = json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                pass

    return pipeline_tag, raw


def _check_contradictions(
    conn: sqlite3.Connection,
    run_id: str,
    model_id: str,
    pipeline_tag: str,
    bank_pairs: list[tuple[str, set[str], set[str]]],
) -> int:
    """Flag C2 anchors not supported by deterministic signals."""
    findings = 0
    for bank, c2_set, det_set in bank_pairs:
        if not (det_set and c2_set):
            continue
        sorted_det = sorted(det_set)
        for c2_label in c2_set - det_set:
            db.insert_audit_finding(
                conn,
                run_id=run_id,
                model_id=model_id,
                mismatch_type="contradiction",
                bank=bank,
                c2_anchor=c2_label,
                det_anchor=None,
                severity=0.7,
                detail={
                    "pipeline_tag": pipeline_tag,
                    "det_found": sorted_det,
                },
            )
            findings += 1
    return findings


def _check_gaps(
    conn: sqlite3.Connection,
    run_id: str,
    model_id: str,
    pipeline_tag: str,
    bank_pairs: list[tuple[str, set[str], set[str]]],
) -> int:
    """Flag deterministic anchors that C2 missed."""
    findings = 0
    for bank, c2_set, det_set in bank_pairs:
        for det_label in det_set - c2_set:
            existing = conn.execute(
                """SELECT ma.confidence FROM model_anchors ma
                   JOIN anchors a ON ma.anchor_id = a.anchor_id
                   WHERE ma.model_id = ? AND a.label = ?""",
                (model_id, det_label),
            ).fetchone()
            if existing and existing[0] >= 0.5:
                continue
            db.insert_audit_finding(
                conn,
                run_id=run_id,
                model_id=model_id,
                mismatch_type="gap",
                bank=bank,
                c2_anchor=None,
                det_anchor=det_label,
                severity=0.5,
                detail={"pipeline_tag": pipeline_tag},
            )
            findings += 1
    return findings


def _check_confidence_conflicts(
    conn: sqlite3.Connection,
    run_id: str,
    model_id: str,
    c2_anchors: dict[str, list[tuple[str, float]]],
    det_anchors: dict[str, list[tuple[str, float]]],
) -> int:
    """Flag same anchor with large confidence gap between C2 and deterministic."""
    findings = 0
    for bank in ("CAPABILITY", "DOMAIN"):
        c2_map = {label: conf for label, conf in c2_anchors.get(bank, [])}
        det_map = {label: conf for label, conf in det_anchors.get(bank, [])}
        for label in set(c2_map) & set(det_map):
            gap = abs(c2_map[label] - det_map[label])
            if gap > 0.3:
                db.insert_audit_finding(
                    conn,
                    run_id=run_id,
                    model_id=model_id,
                    mismatch_type="confidence_conflict",
                    bank=bank,
                    c2_anchor=label,
                    det_anchor=label,
                    severity=gap,
                    detail={
                        "c2_confidence": c2_map[label],
                        "det_confidence": det_map[label],
                    },
                )
                findings += 1
    return findings


def _check_unsupported(
    conn: sqlite3.Connection,
    run_id: str,
    model_id: str,
    pipeline_tag: str,
    bank_tuples: list[tuple[str, set[str], set[str], set[str]]],
) -> int:
    """Flag C2 anchors in a bank where det found nothing but found signals elsewhere."""
    findings = 0
    for bank, c2_set, det_set, other_det in bank_tuples:
        if not (c2_set and not det_set and other_det):
            continue
        sorted_other = sorted(other_det)
        for c2_label in c2_set:
            db.insert_audit_finding(
                conn,
                run_id=run_id,
                model_id=model_id,
                mismatch_type="unsupported",
                bank=bank,
                c2_anchor=c2_label,
                det_anchor=None,
                severity=0.6,
                detail={
                    "pipeline_tag": pipeline_tag,
                    "other_bank_det": sorted_other,
                },
            )
            findings += 1
    return findings


def _audit_single_model(
    conn: sqlite3.Connection,
    ingest_conn: sqlite3.Connection | None,
    run_id: str,
    model_id: str,
) -> int:
    """Audit one model. Returns number of findings."""
    pipeline_tag, raw = _get_model_context(conn, ingest_conn, model_id)
    searchable = _build_searchable(model_id, raw, pipeline_tag)

    c2_anchors = _get_c2_anchors(conn, model_id)
    det_anchors = _get_det_anchors(conn, model_id)

    det_cap = set(_run_capability_patterns(searchable))
    det_dom = {label for label, _ in _run_domain_patterns(searchable)}
    c2_cap = {label for label, _ in c2_anchors.get("CAPABILITY", [])}
    c2_dom = {label for label, _ in c2_anchors.get("DOMAIN", [])}

    bank_pairs = [
        ("CAPABILITY", c2_cap, det_cap),
        ("DOMAIN", c2_dom, det_dom),
    ]

    findings = 0
    findings += _check_contradictions(conn, run_id, model_id, pipeline_tag, bank_pairs)
    findings += _check_gaps(conn, run_id, model_id, pipeline_tag, bank_pairs)
    findings += _check_confidence_conflicts(
        conn, run_id, model_id, c2_anchors, det_anchors
    )
    findings += _check_unsupported(
        conn,
        run_id,
        model_id,
        pipeline_tag,
        [
            ("CAPABILITY", c2_cap, det_cap, det_dom),
            ("DOMAIN", c2_dom, det_dom, det_cap),
        ],
    )
    return findings


def _finalize_audit(
    conn: sqlite3.Connection,
    run_id: str,
    model_ids: list[str],
    num_models: int,
    total_mismatches: int,
) -> AuditResult:
    """Compute summary stats, score models, and finalize the audit run."""
    type_counts: dict[str, int] = {}
    for row in conn.execute(
        "SELECT mismatch_type, COUNT(*) FROM audit_findings WHERE run_id = ? GROUP BY mismatch_type",
        (run_id,),
    ).fetchall():
        type_counts[row[0]] = row[1]

    bank_mismatches: dict[str, int] = {}
    for row in conn.execute(
        "SELECT bank, COUNT(*) FROM audit_findings WHERE run_id = ? GROUP BY bank",
        (run_id,),
    ).fetchall():
        bank_mismatches[row[0]] = row[1]

    for model_id in model_ids:
        model_findings = conn.execute(
            "SELECT COUNT(*) FROM audit_findings WHERE run_id = ? AND model_id = ?",
            (run_id, model_id),
        ).fetchone()[0]
        audit_score = max(0.0, 1.0 - (model_findings * 0.2))
        db.set_metadata(
            conn, model_id, "audit_score", str(round(audit_score, 2)), "float"
        )

    conn.commit()

    per_bank_rates: dict[str, float] = {}
    if model_ids:
        for bank_name, count in bank_mismatches.items():
            per_bank_rates[bank_name] = round(count / num_models, 4)

    summary = {
        "total_audited": num_models,
        "total_mismatches": total_mismatches,
        "per_type_counts": type_counts,
        "per_bank_rates": per_bank_rates,
    }
    db.finish_phase_d_run(conn, run_id, "completed", summary)
    conn.commit()

    logger.info(
        "D1 audit complete: %d models, %d mismatches, types=%s",
        num_models,
        total_mismatches,
        type_counts,
    )

    return AuditResult(
        run_id=run_id,
        total_audited=num_models,
        total_mismatches=total_mismatches,
        per_bank_rates=per_bank_rates,
        per_type_counts=type_counts,
    )


def audit_c2(
    conn: sqlite3.Connection,
    ingest_conn: sqlite3.Connection | None = None,
) -> AuditResult:
    """Run D1 deterministic audit on all models with C2 anchors.

    Args:
        conn: Network database connection.
        ingest_conn: Optional ingest database connection (for raw_json).

    Returns:
        AuditResult with run summary.
    """
    run_id = db.create_phase_d_run(conn, "d1", config={"phase": "audit"})

    # Find all models that have C2-assigned anchors (confidence=0.5)
    rows = conn.execute(
        """SELECT DISTINCT ma.model_id
           FROM model_anchors ma
           WHERE ma.confidence = 0.5"""
    ).fetchall()
    model_ids = [r[0] for r in rows]
    num_models = len(model_ids)

    logger.info("D1 audit: %d models with C2 anchors", num_models)

    total_mismatches = 0

    for idx, model_id in enumerate(model_ids):
        findings = _audit_single_model(conn, ingest_conn, run_id, model_id)
        total_mismatches += findings

        if (idx + 1) % 500 == 0:
            conn.commit()
            logger.info("D1 audit: %d/%d models audited...", idx + 1, num_models)

    conn.commit()

    result = _finalize_audit(conn, run_id, model_ids, num_models, total_mismatches)
    return result
