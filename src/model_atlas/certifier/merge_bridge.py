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

    Returns (kept, rejected). Every proposed label lands in exactly one of
    the two — nothing is dropped silently, or the merge's stats undercount.

      * `kept` — labels the merge SHOULD evaluate: those CERTIFIED, DEMOTED,
        or WARNING, plus labels absent from every vocabulary. The latter are
        passed through rather than dropped so `_validate_anchor` counts them
        as invalid; the certifier has no opinion on a label it can't resolve.
        Order preserved from `proposed_labels`.
      * `rejected` — labels the certifier vetoed: those REJECTED on the facts,
        and those resolving to a bank other than the calling one. A vetoed
        label must never pass through — it resolves in `anchors`, so
        `_validate_anchor` would link it and undo the veto. Caller logs them
        for the merge's stats so they still surface in the run summary.

    AUTO_ADDED labels are NOT returned here — they need model-level, not
    bank-level, application. Call `apply_auto_adds()` once per model at
    end-of-merge if you want to write them.
    """
    facts = _load_hf_facts_lite(conn, model_id)
    proposed: list[AnchorEmission] = []
    # Labels the certifier has no opinion on because they aren't in any
    # vocabulary. They pass through so `_validate_anchor` counts them as
    # invalid — dropping them here would swallow them uncounted.
    unknown: list[str] = []
    vetoed: list[str] = []
    for label in proposed_labels:
        static = VOCABULARY.get(label)
        # If label is not in static vocab, try the live vocab via the DB.
        if static is None:
            row = conn.execute(
                "SELECT bank FROM anchors WHERE lower(label) = ?", (label.lower(),)
            ).fetchone()
            if not row:
                unknown.append(label)
                continue
            try:
                bank_of = Bank(row[0])
            except ValueError:
                # Label exists but carries a bank this build doesn't know.
                # It must not pass through: `_validate_anchor` would resolve
                # and link it on the strength of the anchors row alone.
                vetoed.append(label)
                continue
        else:
            bank_of = static[0]
        # Veto labels whose bank doesn't match the calling bank — protects
        # against LLMs slotting an ARCHITECTURE anchor into a CAPABILITY
        # bank's selection. These resolve in `anchors`, so they cannot pass
        # through to `_validate_anchor` either.
        if bank_of is not bank:
            vetoed.append(label)
            continue
        try:
            proposed.append(AnchorEmission(
                model_id=model_id, label=label, bank=bank_of, confidence=confidence,
                evidence=Provenance(evidence_source, "", extractor),
            ))
        except ValueError:
            vetoed.append(label)
            continue

    result = certify(facts, proposed)
    certified: set[str] = set()
    rejected: list[str] = list(vetoed)
    for v in result.verdicts:
        if v.outcome is CertificationOutcome.REJECTED:
            rejected.append(v.emission.label)
        elif v.outcome is CertificationOutcome.AUTO_ADDED:
            continue  # handled elsewhere
        else:
            certified.add(v.emission.label)

    passthrough = set(unknown)
    kept = [
        label for label in proposed_labels
        if label in certified or label in passthrough
    ]
    return kept, rejected
