"""LLM-invocation gating — decide when Phase C / Phase E should call the LLM.

Ported from triagegeist's `UNCERTAINTY_THRESHOLD` pattern: the LLM residual
only fires when the deterministic layer left genuine ambiguity. Anything
already covered by structural HF facts should skip the LLM entirely.

For ModelAtlas, "sufficiently covered" means: every bank has at least one
anchor whose provenance is a trusted evidence source (pipeline_tag,
model_type, config_json, safetensors, library_name, license). If any bank
is empty OR only carries inferred-tier anchors, the LLM has real work to
do and gets invoked.

This module is pure — it reads the DB but never writes. Callers (Phase
C/E export scripts, phase_c_worker if we ever gate at the worker layer)
consult `should_invoke_llm()` before enqueueing a model.
"""
from __future__ import annotations

import sqlite3

from .contract import Bank

# Bank labels as they appear in the anchors.bank column
_BANK_LABELS = tuple(b.value for b in Bank)


def bank_coverage_summary(
    conn: sqlite3.Connection, model_id: str
) -> dict[str, tuple[int, float]]:
    """Return {bank: (n_anchors, max_confidence)} for one model.

    Callers can then decide their own coverage predicate. Uses live DB — no
    static caching, so this always reflects the current state.
    """
    rows = conn.execute(
        """
        SELECT a.bank, COUNT(*), COALESCE(MAX(ma.confidence), 0.0)
        FROM model_anchors ma
        JOIN anchors a ON ma.anchor_id = a.anchor_id
        WHERE ma.model_id = ?
        GROUP BY a.bank
        """,
        (model_id,),
    ).fetchall()
    coverage: dict[str, tuple[int, float]] = {b: (0, 0.0) for b in _BANK_LABELS}
    for bank, n, max_conf in rows:
        coverage[bank] = (int(n), float(max_conf))
    return coverage


def should_invoke_llm(
    conn: sqlite3.Connection,
    model_id: str,
    *,
    min_covered_banks: int = 6,
    ambiguous_bank_floor: float = 0.3,
) -> bool:
    """Return True when the LLM has real work to do on this model.

    A model is "already covered" (LLM skipped) when:
      * at least `min_covered_banks` of the 8 banks have any anchor at all,
        AND
      * no bank present carries only very-weak (<`ambiguous_bank_floor`)
        anchors — a bank present only at conf 0.1-0.2 is a genuinely
        ambiguous signal that the LLM could disambiguate.

    This is looser than the trusted-evidence gate: it accepts pattern-match
    anchors at 0.4-0.8 as "sufficient" because the certifier (Phase 3) has
    already verified those don't contradict HF facts. The LLM's marginal
    value is only in NEW information for uncovered banks, not confirming
    what patterns already found.

    Cheap by design — one indexed GROUP BY per model.
    """
    coverage = bank_coverage_summary(conn, model_id)
    covered_banks = sum(1 for (n, _) in coverage.values() if n > 0)
    if covered_banks < min_covered_banks:
        return True
    # Present banks whose best anchor is ambiguously weak → LLM can refine
    for (n, max_conf) in coverage.values():
        if 0 < n and max_conf < ambiguous_bank_floor:
            return True
    return False


def filter_needs_llm(
    conn: sqlite3.Connection,
    model_ids: list[str],
    *,
    min_covered_banks: int = 6,
    ambiguous_bank_floor: float = 0.3,
) -> tuple[list[str], list[str]]:
    """Split a candidate list into (needs_llm, already_covered).

    O(N) with one GROUP BY per model. Callers with very large lists that
    want a single-query alternative can grep on `_BANK_LABELS` and reuse
    the coverage counts directly.
    """
    needs, covered = [], []
    for mid in model_ids:
        if should_invoke_llm(
            conn, mid,
            min_covered_banks=min_covered_banks,
            ambiguous_bank_floor=ambiguous_bank_floor,
        ):
            needs.append(mid)
        else:
            covered.append(mid)
    return needs, covered
