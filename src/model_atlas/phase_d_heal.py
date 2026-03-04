"""D3: Healing orchestration — export, prompt build, and merge.

Follows the export/merge pattern from ingest_phase_c.py. Workers produce
complete C2-style responses (summary + selected_anchors), and the merge step
computes diffs and stores correction_events for DPO training data.

Two tiers:
  - D3a (local): qwen2.5:3b via Ollama, bulk corrections
  - D3b (claude): Claude Code CLI, 0.1% budget, high-value models
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
from dataclasses import dataclass

from . import db
from .config import (
    AUDIT_MISMATCH_THRESHOLD,
    HEAL_CLAUDE_BUDGET_FRACTION,
    HEAL_DEFAULT_SEED,
    PHASE_D_WORK_DIR,
)

logger = logging.getLogger(__name__)


_HEALING_PROMPT_TEMPLATE = """You are correcting a previous ML model classification. The original classification had errors detected by automated audit.

## Model Information
- Model ID: {model_id}
- Author: {author}
- Pipeline tag: {pipeline_tag}
- Tags: {tags}
- Parameter count: {param_count}
- Card excerpt: {card_excerpt}

## Raw metadata
{raw_fields}

## Current anchors (from previous classification)
{current_anchors}

## Audit findings (what was wrong)
{audit_findings}

## Valid anchor dictionary
Select ONLY from these anchors:
CAPABILITY: {capability_anchors}
DOMAIN: {domain_anchors}

## Instructions
Based on the raw evidence above, provide a corrected classification. Output valid JSON with:
- "summary": A one-sentence description of this model's purpose and distinguishing features
- "selected_anchors": Array of 1-5 anchors from the valid dictionary that best describe this model's capabilities and domain

Focus on correcting the audit findings. Use the raw metadata (tags, pipeline_tag, author) as ground truth."""


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
    rng = random.Random(seed)  # NOSONAR — reproducible sampling, not security

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
        claude_budget = min(budget, max(1, int(total_models * HEAL_CLAUDE_BUDGET_FRACTION)))

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
            # Top half by downloads, bottom half random from high-error
            top_n = claude_budget // 2
            selected = candidates[:top_n]

            remaining = candidates[top_n:]
            random_n = claude_budget - top_n
            selected.extend(rng.sample(remaining, min(random_n, len(remaining))))

        metadata = {
            "tier": tier,
            "seed": seed,
            "total_candidates": len(candidates),
            "selected": len(selected),
            "claude_budget": claude_budget,
            "strategy": "top_downloads_plus_random_high_error",
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
        anchor_strs.append(f"  {a['label']} ({a['bank']}, confidence={a['confidence']})")

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
            conn, f"d3{'a' if tier == 'local' else 'b'}",
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
            # Get raw_json from ingest DB
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

            # Get card excerpt
            card_row = conn.execute(
                "SELECT value FROM model_metadata WHERE model_id = ? AND key = 'smol_summary'",
                (model_id,),
            ).fetchone()
            card_excerpt = card_row[0] if card_row else ""

            # Get current anchors and audit findings
            current_anchors = _get_model_anchors_with_detail(conn, model_id)
            findings = _get_audit_findings_for_model(conn, model_id)

            # Build healing prompt
            prompt = build_healing_prompt(
                model_id=model_id,
                raw_json=raw,
                card_excerpt=card_excerpt,
                current_anchors=current_anchors,
                audit_findings=findings,
                capability_anchors=cap_labels,
                domain_anchors=dom_labels,
            )

            # Build original C2 response (for DPO training data)
            qwen_row = conn.execute(
                "SELECT value FROM model_metadata WHERE model_id = ? AND key = 'qwen_summary'",
                (model_id,),
            ).fetchone()
            c2_anchors = [
                a["label"]
                for a in current_anchors
                if abs(a["confidence"] - 0.5) < 1e-9
            ]
            original_response = json.dumps({
                "summary": qwen_row[0] if qwen_row else "",
                "selected_anchors": c2_anchors,
            })

            item = {
                "model_id": model_id,
                "healing_prompt": prompt,
                "valid_anchors": all_valid,
                "run_id": run_id,
                "original_prompt": prompt,
                "original_response": original_response,
            }

            shard_idx = idx % num_shards
            shard_files[shard_idx].write(json.dumps(item) + "\n")

    db.finish_phase_d_run(
        conn, run_id, "exported",
        {"exported": len(selected), "num_shards": num_shards},
    )
    conn.commit()

    logger.info(
        "export_d3: wrote %d prompts across %d shards (%s tier)",
        len(selected), num_shards, tier,
    )

    return HealExportResult(
        run_id=run_id,
        total_exported=len(selected),
        num_shards=num_shards,
        tier=tier,
    )


def merge_d3(
    conn: sqlite3.Connection,
    files: list[str],
    run_id: str,
) -> dict[str, int]:
    """Merge D3 healing results into network DB.

    For each healed response:
      - Compute diff (anchors_added, anchors_removed)
      - Apply anchor changes
      - Store correction_event

    Returns {"merged": N, "skipped": N, "errors": N, "anchors_added": N, "anchors_removed": N}.
    """
    merged = 0
    skipped = 0
    errors = 0
    total_added = 0
    total_removed = 0

    for fpath in files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    errors += 1
                    continue

                model_id = item.get("model_id", "")
                if not model_id:
                    errors += 1
                    continue

                if "error" in item:
                    skipped += 1
                    continue

                summary = item.get("summary", "")
                selected_anchors = item.get("selected_anchors", [])
                rationale = item.get("rationale", "")

                if not summary:
                    skipped += 1
                    continue

                # Get original C2 anchors for diff
                original_response_str = item.get("original_response", "{}")
                try:
                    original = json.loads(original_response_str)
                except (json.JSONDecodeError, TypeError):
                    original = {}
                original_anchors = set(original.get("selected_anchors", []))
                healed_anchors = {a.strip().lower() for a in selected_anchors if isinstance(a, str) and a.strip()}

                anchors_added = sorted(healed_anchors - original_anchors)
                anchors_removed = sorted(original_anchors - healed_anchors)

                # Apply anchor changes
                for label in anchors_added:
                    row = conn.execute(
                        "SELECT anchor_id FROM anchors WHERE label = ?",
                        (label,),
                    ).fetchone()
                    if row:
                        db.link_anchor(conn, model_id, row[0], confidence=0.6)
                        total_added += 1

                for label in anchors_removed:
                    conn.execute(
                        """DELETE FROM model_anchors
                           WHERE model_id = ?
                             AND anchor_id = (SELECT anchor_id FROM anchors WHERE label = ?)
                             AND confidence = 0.5""",
                        (model_id, label),
                    )
                    total_removed += 1

                # Update summary if changed
                healed_response = json.dumps({
                    "summary": summary,
                    "selected_anchors": sorted(healed_anchors),
                })

                if summary != original.get("summary", ""):
                    db.set_metadata(conn, model_id, "qwen_summary", summary, "str")

                # Store correction event
                db.insert_correction_event(
                    conn,
                    run_id=run_id,
                    model_id=model_id,
                    tier=item.get("tier", "local"),
                    original_prompt=item.get("original_prompt"),
                    original_response=original_response_str,
                    healed_response=healed_response,
                    anchors_added=anchors_added,
                    anchors_removed=anchors_removed,
                    rationale=rationale,
                )

                merged += 1

    conn.commit()

    result = {
        "merged": merged,
        "skipped": skipped,
        "errors": errors,
        "anchors_added": total_added,
        "anchors_removed": total_removed,
    }

    db.finish_phase_d_run(conn, run_id, "completed", result)
    conn.commit()

    logger.info(
        "merge_d3: merged=%d skipped=%d errors=%d added=%d removed=%d",
        merged, skipped, errors, total_added, total_removed,
    )

    return result
