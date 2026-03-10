"""D3: Healing orchestration — candidate selection, prompt build, and export.

Merge logic lives in phase_d_merge.py. Two tiers: D3a (local Ollama) and
D3b (Claude CLI, 0.1% budget).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass

from . import db
from .config import (
    AUDIT_MISMATCH_THRESHOLD,
    HEAL_CLAUDE_BUDGET_FRACTION,
    HEAL_DEFAULT_SEED,
    PHASE_D_WORK_DIR,
)
from .phase_d_merge import merge_d3 as merge_d3  # noqa: F401, E402

logger = logging.getLogger(__name__)


_HEALING_PROMPT_TEMPLATE = """\
You are correcting a previous ML model classification. The original classification had errors detected by automated audit.
## Model Information
- Model ID: {model_id}  Author: {author}  Pipeline: {pipeline_tag}
- Tags: {tags}  Params: {param_count}  Card excerpt: {card_excerpt}
## Raw metadata
{raw_fields}
## Current anchors (from previous classification)
{current_anchors}
## Audit findings (what was wrong)
{audit_findings}
## Valid anchor dictionary — select ONLY from these:
CAPABILITY: {capability_anchors}
DOMAIN: {domain_anchors}
## Instructions
Provide a corrected classification as valid JSON:
- "summary": One-sentence description of this model's purpose and distinguishing features
- "selected_anchors": Array of 1-5 anchors from the valid dictionary
Focus on correcting the audit findings. Use raw metadata (tags, pipeline_tag, author) as ground truth."""


@dataclass
class HealExportResult:
    """Summary of a D3 export."""

    run_id: str
    total_exported: int = 0
    num_shards: int = 0
    tier: str = ""


def _get_anchor_labels_by_bank(conn: sqlite3.Connection, bank: str) -> list[str]:
    """All anchor labels in a bank."""
    rows = conn.execute(
        "SELECT label FROM anchors WHERE bank = ? ORDER BY label",
        (bank,),
    ).fetchall()
    return [r[0] for r in rows]


def _get_model_anchors_with_detail(
    conn: sqlite3.Connection, model_id: str
) -> list[dict]:
    """Get all anchors for a model with bank and confidence."""
    rows = conn.execute(
        """SELECT a.label, a.bank, ma.confidence
           FROM model_anchors ma
           JOIN anchors a ON ma.anchor_id = a.anchor_id
           WHERE ma.model_id = ?
           ORDER BY a.bank, a.label""",
        (model_id,),
    ).fetchall()
    return [{"label": r[0], "bank": r[1], "confidence": r[2]} for r in rows]


def _get_audit_findings_for_model(
    conn: sqlite3.Connection, model_id: str
) -> list[dict]:
    """Get audit findings for a model."""
    rows = conn.execute(
        """SELECT mismatch_type, bank, c2_anchor, det_anchor, severity, detail
           FROM audit_findings
           WHERE model_id = ?
           ORDER BY severity DESC""",
        (model_id,),
    ).fetchall()
    results = []
    for r in rows:
        finding = {
            "type": r[0],
            "bank": r[1],
            "c2_anchor": r[2],
            "det_anchor": r[3],
            "severity": r[4],
        }
        if r[5]:
            try:
                finding["detail"] = json.loads(r[5])
            except (json.JSONDecodeError, TypeError):
                pass
        results.append(finding)
    return results


def _deterministic_sample(items: list[str], n: int, seed: int) -> list[str]:
    """Deterministic sample using hash-based scoring (no PRNG)."""
    scored = []
    for item in items:
        h = hashlib.sha256(f"{seed}:{item}".encode()).hexdigest()
        scored.append((h, item))
    scored.sort()
    return [item for _, item in scored[:n]]


def select_healing_candidates(
    conn: sqlite3.Connection,
    ingest_conn: sqlite3.Connection | None,
    tier: str,
    budget: int,
    seed: int = HEAL_DEFAULT_SEED,
) -> tuple[list[str], dict]:
    """Select models for healing. Returns (model_ids, selection_metadata).

    Local tier: all models with audit_score < threshold.
    Claude tier: top N by downloads + random injection of high-error models.
    """
    if tier == "local":
        rows = conn.execute(
            """SELECT model_id FROM model_metadata
               WHERE key = 'audit_score'
                 AND CAST(value AS REAL) < ?
               ORDER BY CAST(value AS REAL) ASC""",
            (AUDIT_MISMATCH_THRESHOLD,),
        ).fetchall()
        candidates = [r[0] for r in rows]
        selected = candidates[:budget]
        metadata = {
            "tier": tier,
            "seed": seed,
            "total_candidates": len(candidates),
            "selected": len(selected),
            "strategy": "lowest_audit_score",
        }

    elif tier == "claude":
        # Top by downloads
        total_models = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
        claude_budget = min(
            budget, max(1, int(total_models * HEAL_CLAUDE_BUDGET_FRACTION))
        )

        # Get models with low audit scores, sorted by downloads
        rows = conn.execute(
            """SELECT mm_score.model_id, CAST(COALESCE(mm_dl.value, '0') AS INTEGER) as downloads
               FROM model_metadata mm_score
               LEFT JOIN model_metadata mm_dl
                 ON mm_score.model_id = mm_dl.model_id AND mm_dl.key = 'downloads'
               WHERE mm_score.key = 'audit_score'
                 AND CAST(mm_score.value AS REAL) < ?
               ORDER BY downloads DESC""",
            (AUDIT_MISMATCH_THRESHOLD,),
        ).fetchall()

        candidates = [r[0] for r in rows]

        if len(candidates) <= claude_budget:
            selected = candidates
        else:
            # Top half by downloads, bottom half deterministic sample from high-error
            top_n = claude_budget // 2
            selected = candidates[:top_n]

            remaining = candidates[top_n:]
            random_n = claude_budget - top_n
            selected.extend(_deterministic_sample(remaining, random_n, seed))

        metadata = {
            "tier": tier,
            "seed": seed,
            "total_candidates": len(candidates),
            "selected": len(selected),
            "claude_budget": claude_budget,
            "strategy": "top_downloads_plus_deterministic_high_error",
        }

    else:
        raise ValueError(f"Unknown tier: {tier!r}. Must be 'local' or 'claude'.")

    return selected, metadata


def build_healing_prompt(
    model_id: str,
    raw_json: dict,
    card_excerpt: str,
    current_anchors: list[dict],
    audit_findings: list[dict],
    capability_anchors: list[str],
    domain_anchors: list[str],
) -> str:
    """Build a healing prompt from raw evidence and audit context."""
    # Format raw fields
    raw_parts = []
    for key in ("tags", "pipeline_tag", "author", "library_name"):
        val = raw_json.get(key, "")
        if val:
            raw_parts.append(f"- {key}: {val}")

    # Format current anchors
    anchor_strs = []
    for a in current_anchors:
        anchor_strs.append(
            f"  {a['label']} ({a['bank']}, confidence={a['confidence']})"
        )

    # Format audit findings
    finding_strs = []
    for f in audit_findings:
        parts = [f"- {f['type']}"]
        if f.get("bank"):
            parts.append(f"bank={f['bank']}")
        if f.get("c2_anchor"):
            parts.append(f"c2_assigned={f['c2_anchor']}")
        if f.get("det_anchor"):
            parts.append(f"det_found={f['det_anchor']}")
        finding_strs.append(" ".join(parts))

    tags = raw_json.get("tags", [])

    return _HEALING_PROMPT_TEMPLATE.format(
        model_id=model_id,
        author=raw_json.get("author", "unknown"),
        pipeline_tag=raw_json.get("pipeline_tag", "unknown"),
        tags=", ".join(tags[:15]) if tags else "none",
        param_count=raw_json.get("param_count", "unknown"),
        card_excerpt=card_excerpt or "none",
        raw_fields="\n".join(raw_parts) or "none",
        current_anchors="\n".join(anchor_strs) or "none",
        audit_findings="\n".join(finding_strs) or "none",
        capability_anchors=", ".join(capability_anchors),
        domain_anchors=", ".join(domain_anchors),
    )


def _build_export_item(
    conn: sqlite3.Connection,
    ingest_conn: sqlite3.Connection | None,
    model_id: str,
    cap_labels: list[str],
    dom_labels: list[str],
    all_valid: list[str],
    run_id: str,
) -> dict:
    """Build a single D3 export item for one model."""
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

    card_row = conn.execute(
        "SELECT value FROM model_metadata WHERE model_id = ? AND key = 'smol_summary'",
        (model_id,),
    ).fetchone()
    card_excerpt = card_row[0] if card_row else ""

    current_anchors = _get_model_anchors_with_detail(conn, model_id)
    findings = _get_audit_findings_for_model(conn, model_id)

    prompt = build_healing_prompt(
        model_id=model_id,
        raw_json=raw,
        card_excerpt=card_excerpt,
        current_anchors=current_anchors,
        audit_findings=findings,
        capability_anchors=cap_labels,
        domain_anchors=dom_labels,
    )

    qwen_row = conn.execute(
        "SELECT value FROM model_metadata WHERE model_id = ? AND key = 'qwen_summary'",
        (model_id,),
    ).fetchone()
    c2_anchors = [
        a["label"] for a in current_anchors if abs(a["confidence"] - 0.5) < 1e-9
    ]
    original_response = json.dumps(
        {
            "summary": qwen_row[0] if qwen_row else "",
            "selected_anchors": c2_anchors,
        }
    )

    return {
        "model_id": model_id,
        "healing_prompt": prompt,
        "valid_anchors": all_valid,
        "run_id": run_id,
        "original_prompt": prompt,
        "original_response": original_response,
    }


def export_d3(
    conn: sqlite3.Connection,
    ingest_conn: sqlite3.Connection | None,
    tier: str,
    budget: int,
    num_shards: int = 1,
    seed: int = HEAL_DEFAULT_SEED,
) -> HealExportResult:
    """Export D3 healing prompts to sharded JSONL files.

    Returns HealExportResult with run_id and export stats.
    """
    selected, sel_metadata = select_healing_candidates(
        conn, ingest_conn, tier, budget, seed
    )

    if not selected:
        logger.info("export_d3: no healing candidates found")
        run_id = db.create_phase_d_run(
            conn,
            f"d3{'a' if tier == 'local' else 'b'}",
            config={"tier": tier, "budget": budget, "seed": seed, **sel_metadata},
        )
        db.finish_phase_d_run(conn, run_id, "completed", {"exported": 0})
        conn.commit()
        return HealExportResult(run_id=run_id, tier=tier)

    run_id = db.create_phase_d_run(
        conn,
        f"d3{'a' if tier == 'local' else 'b'}",
        config={"tier": tier, "budget": budget, "seed": seed, **sel_metadata},
    )
    conn.commit()

    PHASE_D_WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Cache dictionary labels
    cap_labels = _get_anchor_labels_by_bank(conn, "CAPABILITY")
    dom_labels = _get_anchor_labels_by_bank(conn, "DOMAIN")
    all_valid = sorted(set(cap_labels + dom_labels))

    # Open shard files using ExitStack for safe resource management
    from contextlib import ExitStack

    with ExitStack() as stack:
        shard_files = [
            stack.enter_context(
                open(PHASE_D_WORK_DIR / f"d3_{tier}_shard_{i}.jsonl", "w")
            )
            for i in range(num_shards)
        ]

        for idx, model_id in enumerate(selected):
            item = _build_export_item(
                conn, ingest_conn, model_id, cap_labels, dom_labels, all_valid, run_id
            )
            shard_idx = idx % num_shards
            shard_files[shard_idx].write(json.dumps(item) + "\n")

    db.finish_phase_d_run(
        conn,
        run_id,
        "exported",
        {"exported": len(selected), "num_shards": num_shards},
    )
    conn.commit()

    logger.info(
        "export_d3: wrote %d prompts across %d shards (%s tier)",
        len(selected),
        num_shards,
        tier,
    )

    return HealExportResult(
        run_id=run_id,
        total_exported=len(selected),
        num_shards=num_shards,
        tier=tier,
    )
