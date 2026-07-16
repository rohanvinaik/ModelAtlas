#!/usr/bin/env python3
"""Retroactively re-certify every model against the deterministic rule library.

Reads current HF-facts (pipeline_tag, model_type, library_name, tags,
safetensors, quantization, config, param count) and current anchor
assignments per model. Runs the certifier. Emits:

  ADDS      — anchors the rules require but the model lacks (AUTO_ADDED)
  DROPS     — anchors that violate a Tier-1 structural rule (REJECTED)
  DEMOTES   — anchors that violate a Tier-2 rule (confidence lowered)
  WARNINGS  — anchors that draw a soft flag (kept, but noted)

Dry-run by default. `--apply` writes the diff via db.link_anchor for adds
and direct DELETE for drops (legacy write path — Phase 7 migrates all
writes through the reconciler). `--limit N` for spot-checks; `--sample`
picks a random N models for early smoke tests.

Snapshot the DB before --apply. This script deletes rows.
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter

from model_atlas import db
from model_atlas.certifier import HFFacts, certify
from model_atlas.contract import (
    AnchorEmission,
    Bank,
    CertificationOutcome,
    EvidenceType,
    Provenance,
)


def _load_hf_facts(conn: sqlite3.Connection, model_id: str) -> HFFacts:
    """Populate HFFacts from the model + model_metadata tables."""
    md_rows = conn.execute(
        "SELECT key, value FROM model_metadata WHERE model_id = ?",
        (model_id,),
    ).fetchall()
    md = {k: v for k, v in md_rows}

    def _f(k: str) -> str:
        return str(md.get(k, "") or "")

    def _fnum(k: str) -> float | None:
        try:
            return float(md[k]) if k in md and md[k] not in (None, "") else None
        except (TypeError, ValueError):
            return None

    def _iint(k: str) -> int | None:
        try:
            return int(md[k]) if k in md and md[k] not in (None, "") else None
        except (TypeError, ValueError):
            return None

    tags_raw = md.get("tags") or "[]"
    try:
        tags = tuple(json.loads(tags_raw)) if tags_raw.startswith("[") else tuple(tags_raw.split(","))
    except json.JSONDecodeError:
        tags = ()

    return HFFacts(
        model_id=model_id,
        pipeline_tag=_f("pipeline_tag"),
        model_type=_f("model_type"),
        library_name=_f("library_name"),
        license=_f("license"),
        tags=tuple(str(t).strip() for t in tags if t),
        param_count_b=_fnum("parameter_count_b"),
        context_length=_iint("context_length"),
        safetensors_present=(md.get("has_safetensors", "false") in ("true", "1")
                             or bool(_iint("safetensors_total_size"))),
        quantization_level=_f("quantization_level"),
        config={},
    )


def _load_current_emissions(
    conn: sqlite3.Connection, model_id: str, live_vocab: dict[str, Bank]
) -> list[AnchorEmission]:
    """Reconstruct AnchorEmission objects from current model_anchors rows."""
    rows = conn.execute(
        """
        SELECT a.label, a.bank, ma.confidence
        FROM model_anchors ma
        JOIN anchors a ON ma.anchor_id = a.anchor_id
        WHERE ma.model_id = ?
        """,
        (model_id,),
    ).fetchall()
    out: list[AnchorEmission] = []
    for label, bank_str, conf in rows:
        # Skip anchors whose bank isn't in the vocabulary — shouldn't happen
        # but defensive against DB drift.
        try:
            bank = live_vocab.get(label) or Bank(bank_str)
        except ValueError:
            continue
        # We don't know the original evidence, so tag as LLM_INFERENCE
        # (worst-case tier). The certifier only uses evidence tier to
        # arbitrate Tier-1 forbid collisions, which is exactly the case
        # where "we don't know if this was structural" should default to
        # "assume inferred, let structural rules win."
        try:
            out.append(AnchorEmission(
                model_id=model_id, label=label, bank=bank, confidence=float(conf),
                evidence=Provenance(EvidenceType.LLM_INFERENCE, "recertify_reconstructed",
                                    "recertify_corpus"),
            ))
        except ValueError:
            # bank mismatch vs static vocab — skip, log later
            continue
    return out


def _load_live_vocab(conn: sqlite3.Connection) -> dict[str, Bank]:
    """Load {label: Bank} from the live anchors table."""
    vocab: dict[str, Bank] = {}
    for label, bank_str in conn.execute("SELECT label, bank FROM anchors"):
        try:
            vocab[label] = Bank(bank_str)
        except ValueError:
            continue
    return vocab


def _apply_diff(
    conn: sqlite3.Connection,
    model_id: str,
    adds: list[AnchorEmission],
    drops: list[AnchorEmission],
    live_vocab: dict[str, Bank],
) -> tuple[int, int]:
    """Apply ADDS via db.link_anchor + DROPS via direct DELETE. Returns (n_added, n_dropped)."""
    n_add = 0
    n_drop = 0
    for e in adds:
        row = conn.execute(
            "SELECT anchor_id FROM anchors WHERE lower(label) = ?", (e.label.lower(),)
        ).fetchone()
        if not row:
            continue
        db.link_anchor(conn, model_id, row[0], confidence=e.confidence)
        n_add += 1
    for e in drops:
        row = conn.execute(
            "SELECT anchor_id FROM anchors WHERE lower(label) = ?", (e.label.lower(),)
        ).fetchone()
        if not row:
            continue
        conn.execute(
            "DELETE FROM model_anchors WHERE model_id = ? AND anchor_id = ?",
            (model_id, row[0]),
        )
        n_drop += 1
    return n_add, n_drop


def _iter_target_models(
    conn: sqlite3.Connection, limit: int | None, sample: bool
) -> list[str]:
    """Return the model_id list to walk. Optionally samples."""
    rows = conn.execute("SELECT model_id FROM models").fetchall()
    ids = [r[0] for r in rows]
    if sample and limit:
        random.seed(0)  # deterministic sample
        return random.sample(ids, min(limit, len(ids)))
    if limit:
        return ids[:limit]
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Write the diff. Default: dry-run.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N models (for smoke test).")
    parser.add_argument("--sample", action="store_true",
                        help="With --limit, pick a random sample instead of the first N.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-model diff (noisy).")
    args = parser.parse_args()

    conn = db.get_connection()
    db.init_db(conn)

    live_vocab = _load_live_vocab(conn)
    target_ids = _iter_target_models(conn, args.limit, args.sample)

    print(f"Loaded live vocabulary: {len(live_vocab)} anchor labels")
    print(f"Walking {len(target_ids)} models "
          f"({'apply' if args.apply else 'DRY RUN'})")
    print()

    total_adds = 0
    total_drops = 0
    total_demoted = 0
    total_warnings = 0
    rule_hits: Counter[str] = Counter()
    models_with_diff = 0

    for i, mid in enumerate(target_ids, 1):
        if i % 500 == 0:
            print(f"  progress: {i}/{len(target_ids)}  adds={total_adds} drops={total_drops}")
        try:
            facts = _load_hf_facts(conn, mid)
            current = _load_current_emissions(conn, mid, live_vocab)
        except sqlite3.Error as exc:
            print(f"  ERR loading {mid}: {exc}", file=sys.stderr)
            continue

        result = certify(facts, current, live_vocab=live_vocab)

        adds: list[AnchorEmission] = []
        drops: list[AnchorEmission] = []
        n_demoted_local = 0
        for v in result.verdicts:
            if v.outcome is CertificationOutcome.AUTO_ADDED:
                adds.append(v.emission)
                rule_hits[v.rule_name] += 1
                total_adds += 1
            elif v.outcome is CertificationOutcome.REJECTED:
                drops.append(v.emission)
                rule_hits[v.rule_name] += 1
                total_drops += 1
            elif v.outcome is CertificationOutcome.DEMOTED:
                total_demoted += 1
                n_demoted_local += 1
                rule_hits[v.rule_name] += 1
            elif v.outcome is CertificationOutcome.WARNING:
                total_warnings += 1
                rule_hits[v.rule_name] += 1

        if adds or drops:
            models_with_diff += 1

        # Phase 6: coherence score = fraction of proposed anchors that
        # survived unrejected/undemoted. Written as certification_score
        # metadata for navigate() to consume as a soft tiebreaker.
        n_current = max(len(current), 1)
        coherence_score = max(
            0.0,
            1.0 - (len(drops) + 0.5 * n_demoted_local) / n_current,
        )

        if args.verbose and (adds or drops):
            print(f"  {mid}  coherence={coherence_score:.2f}")
            for a in adds:
                print(f"    +ADD {a.label}")
            for d in drops:
                print(f"    -DROP {d.label}")

        if args.apply:
            if adds or drops:
                _apply_diff(conn, mid, adds, drops, live_vocab)
            db.set_metadata(
                conn, mid, "certification_score",
                f"{coherence_score:.4f}", "float",
            )

    if args.apply:
        conn.commit()

    print()
    print("=" * 60)
    print(f"Models walked:     {len(target_ids)}")
    print(f"Models with diff:  {models_with_diff}")
    print(f"  Total ADDs:      {total_adds}")
    print(f"  Total DROPs:     {total_drops}")
    print(f"  Total DEMOTES:   {total_demoted}")
    print(f"  Total WARNINGS:  {total_warnings}")
    print()
    print("Top rules fired:")
    for rule, n in rule_hits.most_common(15):
        print(f"  {n:>7d}  {rule}")

    if not args.apply:
        print()
        print("(DRY RUN — pass --apply to write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
