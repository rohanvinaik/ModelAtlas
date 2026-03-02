# Architecture of Inquiry — Protocol Reference

Five opt-in features that close the loop between behavioral detection and theory extraction. All disabled by default. Enable in `.claude/lintgate.yaml` under `controlplane.inquiry.*`.

## Features

### theory_grounded_signals

When a behavioral signal fires, relevant theory claims are pulled from the project's theory profile and appended as a coda to the finding message.

- **Implementation**: `lintgate/channels/behavior_channel.py` — `SIGNAL_THEORY_MAP`, `_ground_finding_in_theory()`
- **Coda cap**: 150 characters max, 1-2 claims per finding
- **Dedup**: Consecutive identical codas for the same signal are suppressed
- **Requires**: Theory profile cache (extracted once per ControlPlane run)

### prediction_tracking

Before Bash commands, the agent registers a falsifiable prediction via `constraint_check` with structured expected outcomes.

- **Implementation**: `lintgate/controlplane/behavior_compass.py` — `Prediction`, `PredictionExpectation`, `_check_predictions()`
- **Prediction types**: `exit_code` (int), `error_signature` (substring), `stdout_contains` (substring)
- **Matching**: Exact full command-signature match. Empty/unknown sigs rejected.
- **Accuracy modulation**: After 5+ checked predictions. >70% softens by -0.15. <30% amplifies by +0.15.
- **Expiry**: Unchecked predictions expire after 20 events.

### theory_coherence_check

When the constraint proposer generates a new rule, it checks the rule against the project's theory profile.

- **Implementation**: `lintgate/controlplane/constraint_proposer.py` — `TheoryCoherenceResult`, `check_theory_coherence()`
- **Output**: `aligned`, `supporting_claims`, `contradicting_claims`, `coherence_score` (-1.0 to +1.0)
- **Metadata-only**: Confidence is NOT auto-adjusted (conservative by design).
- **Config gated**: Only runs when `inquiry.theory_coherence_check` is True.

### living_context

CLAUDE.md becomes a living document. Behavioral discoveries flow back as managed-section patches.

- **Implementation**: `lintgate/context_bootstrap.py` — `ContextPatch`, `generate_context_patch()`, `apply_context_patch()`
- **Managed sections**: `<!-- LINTGATE:BEGIN section_id vN -->` / `<!-- LINTGATE:END section_id -->`. Section IDs: `machine_rules`, `do_dont`, `theory_alignment`, `context_map`
- **Cumulative rebasing**: `context_patch_apply` re-reads on-disk state. Multiple patches to the same section compose correctly.
- **Apply is always explicit**: Use `context_patch_review` to inspect, `context_patch_apply` to write.

### session_gate

Advisory warning on file modification when the context bootstrap hasn't passed minimum validity.

- **Implementation**: `lintgate/context_auditor.py` — `SessionReadiness`, `check_session_readiness()`
- **On failure**: Advisory warning + short-circuits expensive channels.
- **On pass**: Marks session ready; subsequent events skip the check.

## Enabling

```yaml
controlplane:
  enabled: true
  inquiry:
    theory_grounded_signals: true
    prediction_tracking: true
    theory_coherence_check: true
    living_context: true
    session_gate: true
```

All five features require `controlplane.enabled: true`. Each degrades gracefully when its dependencies are unavailable.
