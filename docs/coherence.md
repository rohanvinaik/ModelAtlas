# Coherence audit — periodic health report

Read-only checks that surface drift in the canonical network over time.
The audit is *not* a CI gate — it doesn't block any operation. It's a
health report for the maintainer to act on at human cadence.

See `PERSISTENT_KNOWLEDGE_GROUNDED_DATABASES.md` §39-§44 for the doctrine.

## What it checks

| Check | What it surfaces | Recommended response |
|-------|------------------|----------------------|
| **Bank orthogonality** | Pearson r between every pair of bank positions. Pairs with \|r\| ≥ 0.7 are flagged. | High correlations not in the original design suggest derivation rules are coupled, or the corpus shifted in a way that collapsed dimensions. Review `extraction/deterministic.py` for shared inputs. |
| **NULL coverage per bank** | For each bank, how many models have / don't have a position. | A bank with growing NULL fraction is a coverage gap — targeted backfill via re-extraction or LLM-assisted authoring. |
| **Anchor orphans** | Anchors with zero `model_anchors` assignments. | Either remove from vocabulary or expand pattern-matching to use them. Orphans are dead weight. |
| **Anchor oversaturation** | Anchors assigned to >50% of models. | No longer discriminative. Split into more specific sub-anchors (e.g., split `decoder-only` into `decoder-only-causal` and `decoder-only-mixture`). |
| **Uncited canonical writes** | Audit log entries whose `reason` is empty, a single short word (`fix`, `update`), or otherwise too thin to source. | Manually review and add proper citations, or roll back. |

## CLI usage

```bash
# Human-readable Markdown report
python -m model_atlas.coherence

# JSON for piping into other tools
python -m model_atlas.coherence --json

# Tighter orthogonality threshold (default 0.7)
python -m model_atlas.coherence --correlation-threshold 0.5

# Stricter oversaturation threshold (default 50%)
python -m model_atlas.coherence --oversaturation-pct 40
```

## Programmatic usage

```python
from model_atlas import coherence, db

conn = db.get_connection()
report = coherence.run_audit(conn)

# Act on findings programmatically
if report.bank_correlations_suspicious:
    print("Orthogonality at risk:", report.bank_correlations_suspicious)
if len(report.anchor_orphans) > 20:
    print(f"{len(report.anchor_orphans)} orphan anchors — vocabulary needs pruning")
```

## What it does NOT check (yet)

- **Zero-state drift** (§21, §63): would compare current percentile
  thresholds against the original derivation rules' implicit zero state.
  Not yet implemented — needs the recalibration loop first.
- **Bi-modal split violations** (§40): would detect canonical-table
  writes whose audit-log `reason` looks observation-shaped. Cannot run
  reliably until legacy writes have audit-log entries.
- **Reconciler silent-sync failure** (§64): hub-side check that each
  worker host has contributed within the last N hours. Requires the
  reconciler to record per-host run timestamps, which it does in
  `reconciler_processed.processed_at` but no aggregator exists yet.

These are follow-ons. The current audit covers the high-value drift
modes that are detectable from the current schema.

## Cadence

Weekly is fine. The doc suggests Monday morning, alongside the
sync-and-reconcile routine. The audit is fast (a few queries; seconds
on a 100K-model corpus).

## See also

- `PERSISTENT_KNOWLEDGE_GROUNDED_DATABASES.md` §39-§44, §62-§68.
- `src/model_atlas/coherence.py` — the implementation.
- `tests/test_coherence.py` — contract pinned by tests.
