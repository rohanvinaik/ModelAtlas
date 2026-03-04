"""D4: Training data export from correction_events.

Exports DPO-format JSONL where each correction becomes a training example:
  - prompt: the original C2 prompt
  - chosen: the healed (corrected) response
  - rejected: the original C2 response

This enables fine-tuning future C2 workers to avoid the same mistakes.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .config import PHASE_D_TRAINING_DIR

logger = logging.getLogger(__name__)


@dataclass
class TrainingStats:
    """Summary of a D4 training data export."""

    total_examples: int = 0
    by_tier: dict[str, int] | None = None
    output_path: str = ""


def export_training_data(
    conn: sqlite3.Connection,
    output_path: str | Path | None = None,
    tier: str = "all",
) -> TrainingStats:
    """Export DPO-format JSONL from correction_events.

    Output format per line:
    {
        "prompt": original_c2_prompt,
        "chosen": healed_response,
        "rejected": original_response,
        "model_id": ...,
        "tier": "local" | "claude",
        "run_id": ...
    }

    Args:
        conn: Network database connection.
        output_path: Output file path. Defaults to PHASE_D_TRAINING_DIR/dpo_training.jsonl.
        tier: Filter by tier: "local", "claude", or "all".

    Returns:
        TrainingStats with export summary.
    """
    if output_path is None:
        PHASE_D_TRAINING_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PHASE_D_TRAINING_DIR / "dpo_training.jsonl"

    query = """SELECT model_id, tier, run_id,
                      original_prompt, original_response, healed_response
               FROM correction_events
               WHERE original_prompt IS NOT NULL
                 AND original_response IS NOT NULL
                 AND healed_response IS NOT NULL"""
    params: tuple = ()

    if tier != "all":
        query += " AND tier = ?"
        params = (tier,)

    query += " ORDER BY event_id"

    rows = conn.execute(query, params).fetchall()

    total = 0
    by_tier: dict[str, int] = {}

    with open(output_path, "w") as f:
        for row in rows:
            model_id = row[0]
            event_tier = row[1]
            run_id = row[2]
            original_prompt = row[3]
            original_response = row[4]
            healed_response = row[5]

            # Skip if original and healed are identical (no actual correction)
            if original_response == healed_response:
                continue

            example = {
                "prompt": original_prompt,
                "chosen": healed_response,
                "rejected": original_response,
                "model_id": model_id,
                "tier": event_tier,
                "run_id": run_id,
            }

            f.write(json.dumps(example) + "\n")
            total += 1
            by_tier[event_tier] = by_tier.get(event_tier, 0) + 1

    logger.info(
        "D4 export: %d training examples written to %s (by_tier=%s)",
        total,
        output_path,
        by_tier,
    )

    return TrainingStats(
        total_examples=total,
        by_tier=by_tier,
        output_path=str(output_path),
    )


def get_training_data_stats(conn: sqlite3.Connection) -> dict:
    """Get training data statistics from correction_events.

    Returns counts by tier and summary of available data.
    """
    total = conn.execute(
        """SELECT COUNT(*) FROM correction_events
           WHERE original_prompt IS NOT NULL
             AND original_response IS NOT NULL
             AND healed_response IS NOT NULL"""
    ).fetchone()[0]

    by_tier: dict[str, int] = {}
    for row in conn.execute(
        """SELECT tier, COUNT(*) FROM correction_events
           WHERE original_prompt IS NOT NULL
             AND original_response IS NOT NULL
             AND healed_response IS NOT NULL
           GROUP BY tier"""
    ).fetchall():
        by_tier[row[0]] = row[1]

    # Count distinct models corrected
    distinct_models = conn.execute(
        "SELECT COUNT(DISTINCT model_id) FROM correction_events"
    ).fetchone()[0]

    # Count by mismatch type from associated audit findings
    by_mismatch: dict[str, int] = {}
    for row in conn.execute(
        """SELECT af.mismatch_type, COUNT(DISTINCT ce.model_id)
           FROM correction_events ce
           JOIN audit_findings af ON ce.model_id = af.model_id
           GROUP BY af.mismatch_type"""
    ).fetchall():
        by_mismatch[row[0]] = row[1]

    return {
        "total_corrections": total,
        "distinct_models": distinct_models,
        "by_tier": by_tier,
        "by_mismatch_type": by_mismatch,
    }
