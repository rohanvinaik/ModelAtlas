"""Bridge helpers used by the legacy Phase C / Phase E merges.

The merges historically iterate over LLM-emitted `selected_anchors` per bank
and blindly link them at fixed confidence (0.4 for Phase E, 0.5 for Phase C).
Phase 7 of the audit-pipeline overhaul routes those selections through the
certifier before writing.

`filter_certified_labels()` is the drop-in call: pass the labels the LLM
proposed, get back the labels the certifier accepted (CERTIFIED, DEMOTED
kept, AUTO_ADDED skipped — auto-adds should come from `apply_auto_adds()`
called once per model, not per bank).
"""
from __future__ import annotations

import sqlite3

from ..contract import (
    AnchorEmission,
    Bank,
    CertificationOutcome,
    EvidenceType,
    Provenance,
    VOCABULARY,
)
from .certifier import certify
from .rules import HFFacts


def _load_hf_facts_lite(conn: sqlite3.Connection, model_id: str) -> HFFacts:
    """Fast HFFacts loader — reads only the fields rules currently trigger on."""
    keys = (
        "pipeline_tag", "model_type", "library_name", "license",
        "parameter_count_b", "context_length", "quantization_level",
        "has_safetensors", "safetensors_total_size",
    )
    ph = ",".join("?" for _ in keys)
    rows = conn.execute(
        f"""SELECT key, value FROM model_metadata
            WHERE model_id = ? AND key IN ({ph})""",
        (model_id, *keys),
    ).fetchall()
    md = {k: v for k, v in rows}
    try:
        param_b: float | None = float(md["parameter_count_b"]) if "parameter_count_b" in md else None
    except (TypeError, ValueError):
        param_b = None
    try:
        ctx: int | None = int(md["context_length"]) if "context_length" in md else None
    except (TypeError, ValueError):
        ctx = None
    return HFFacts(
        model_id=model_id,
        pipeline_tag=str(md.get("pipeline_tag", "") or ""),
        model_type=str(md.get("model_type", "") or ""),
        library_name=str(md.get("library_name", "") or ""),
        license=str(md.get("license", "") or ""),
        param_count_b=param_b,
        context_length=ctx,
        safetensors_present=(md.get("has_safetensors", "") in ("true", "1"))
                            or bool(md.get("safetensors_total_size")),
        quantization_level=str(md.get("quantization_level", "") or ""),
    )


def filter_certified_labels(
    conn: sqlite3.Connection,
    model_id: str,
    bank: Bank,
    proposed_labels: list[str],
    *,
    evidence_source: EvidenceType,
    extractor: str,
    confidence: float,
) -> tuple[list[str], list[str]]:
    """Route LLM-proposed labels for one bank through the certifier.

    Returns (kept, rejected).

      * `kept` — labels the merge SHOULD write (they were CERTIFIED, DEMOTED,
        or WARNING). Order preserved from `proposed_labels`.
      * `rejected` — labels the certifier vetoed. Caller logs them for the
        merge's stats so they still surface in the run summary.

    AUTO_ADDED labels are NOT returned here — they need model-level, not
    bank-level, application. Call `apply_auto_adds()` once per model at
    end-of-merge if you want to write them.
    """
    facts = _load_hf_facts_lite(conn, model_id)
    proposed: list[AnchorEmission] = []
    for label in proposed_labels:
        static = VOCABULARY.get(label)
        # If label is not in static vocab, try the live vocab via the DB.
        # For unknown-bank labels we skip — the merge's existing invalid
        # counter will catch them via `_validate_anchor`.
        if static is None:
            row = conn.execute(
                "SELECT bank FROM anchors WHERE lower(label) = ?", (label.lower(),)
            ).fetchone()
            if not row:
                continue
            try:
                bank_of = Bank(row[0])
            except ValueError:
                continue
        else:
            bank_of = static[0]
        # Skip labels whose bank doesn't match the calling bank — protects
        # against LLMs slotting an ARCHITECTURE anchor into a CAPABILITY
        # bank's selection.
        if bank_of is not bank:
            continue
        try:
            proposed.append(AnchorEmission(
                model_id=model_id, label=label, bank=bank_of, confidence=confidence,
                evidence=Provenance(evidence_source, "", extractor),
            ))
        except ValueError:
            continue

    result = certify(facts, proposed)
    kept: list[str] = []
    rejected: list[str] = []
    for v in result.verdicts:
        if v.outcome is CertificationOutcome.REJECTED:
            rejected.append(v.emission.label)
        elif v.outcome is CertificationOutcome.AUTO_ADDED:
            continue  # handled elsewhere
        else:
            kept.append(v.emission.label)
    return kept, rejected
