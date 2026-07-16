"""The AnswerCertifier — apply the rule library to a proposed anchor set.

Called by every write path in the audit pipeline (Phase 7 migrates the
legacy paths through here). Given the HF facts for a model and a set of
proposed AnchorEmissions from any tier of extractor, returns a
CertificationResult with per-emission verdicts:

  CERTIFIED   — passed all rules
  DEMOTED     — passed but tier collision lowered confidence
  REJECTED    — hard-rule (Tier-1) veto
  AUTO_ADDED  — rule required a label the emitter didn't propose
  WARNING     — soft flag, emission survives

The certifier is pure — no DB access, no I/O. Callers pass in HFFacts and
receive verdicts; the audit-log write is a separate concern owned by the
reconciler layer.
"""
from __future__ import annotations

from collections import defaultdict

from ..contract import (
    AnchorEmission,
    Bank,
    CertificationOutcome,
    CertificationResult,
    CertifiedEmission,
    EvidenceType,
    Provenance,
    TRUSTED_EVIDENCE,
    VOCABULARY,
)
from .rules import ALL_RULES, HFFacts, Rule, RuleTier


def _label_bank(label: str, live_vocab: dict[str, Bank] | None = None) -> Bank | None:
    """Look up the bank for a label. Consults live vocab first (for runtime-
    added anchors), then falls back to the static BOOTSTRAP_ANCHORS."""
    if live_vocab and label in live_vocab:
        return live_vocab[label]
    static = VOCABULARY.get(label)
    return static[0] if static else None


def certify(
    facts: HFFacts,
    proposed: list[AnchorEmission],
    *,
    live_vocab: dict[str, Bank] | None = None,
    rules: tuple[Rule, ...] = ALL_RULES,
) -> CertificationResult:
    """Apply rules to `proposed`. Return per-emission verdicts.

    Parameters
    ----------
    facts:
        HF-fact bundle for this model. See HFFacts.
    proposed:
        Every AnchorEmission the upstream extractors want to write. Order
        preserved in the output for CERTIFIED/DEMOTED/WARNING outcomes;
        AUTO_ADDED emissions appended at the end grouped by rule.
    live_vocab:
        Optional {label: bank} map of anchors registered in the live DB
        beyond the static BOOTSTRAP_ANCHORS. Passed by callers with an
        active connection; None means static-only.
    rules:
        The rule set to apply. Defaults to ALL_RULES; tests can pass a
        smaller subset.

    Returns
    -------
    CertificationResult with one CertifiedEmission per (input emission +
    any auto-added emission). Callers use `.certified()` for what to write
    and `.rejected()` for what to log as diagnostic signal.
    """
    verdicts: list[CertifiedEmission] = []
    proposed_labels: set[str] = {e.label for e in proposed}

    # Fire all rules once and collect their (requires, forbids, tier, name, reason) actions.
    fired: list[Rule] = [r for r in rules if _safe_trigger(r, facts)]

    # ---- Handle REQUIRES (auto-add missing labels) ----
    auto_add_reasons: dict[str, list[Rule]] = defaultdict(list)
    for rule in fired:
        for label in rule.requires:
            if label not in proposed_labels:
                auto_add_reasons[label].append(rule)

    # ---- Handle FORBIDS (per-emission decision) ----
    # Map label -> highest-tier rule that forbids it (lower tier = stricter)
    forbid_by_label: dict[str, Rule] = {}
    for rule in fired:
        for label in rule.forbids:
            existing = forbid_by_label.get(label)
            if existing is None or rule.tier.value < existing.tier.value:
                forbid_by_label[label] = rule

    # ---- Emit verdicts for proposed emissions ----
    for emission in proposed:
        forbid_rule = forbid_by_label.get(emission.label)
        if forbid_rule is None:
            verdicts.append(CertifiedEmission(
                outcome=CertificationOutcome.CERTIFIED,
                emission=emission,
                rule_name="",
                reason="",
            ))
            continue

        # Structural evidence outranks structural forbid: if the emission
        # itself is Tier-1 structural (e.g. safetensors detected), and the
        # forbid is also Tier-1, we don't reject — we log and let both
        # through. This is a defensive escape hatch; in practice Tier-1 vs
        # Tier-1 collisions indicate a genuine rule bug and should be
        # caught in _sanity_check_rules(). We route to WARNING here.
        emission_is_structural = emission.evidence.source_type in TRUSTED_EVIDENCE
        if forbid_rule.tier is RuleTier.STRUCTURAL and emission_is_structural:
            verdicts.append(CertifiedEmission(
                outcome=CertificationOutcome.WARNING,
                emission=emission,
                rule_name=forbid_rule.name,
                reason=f"tier-1 vs tier-1 collision: {_render_reason(forbid_rule, facts)}",
            ))
            continue

        if forbid_rule.tier is RuleTier.STRUCTURAL:
            verdicts.append(CertifiedEmission(
                outcome=CertificationOutcome.REJECTED,
                emission=emission,
                rule_name=forbid_rule.name,
                reason=_render_reason(forbid_rule, facts),
            ))
        elif forbid_rule.tier is RuleTier.SEMI_STRUCTURAL:
            demoted = _demote(emission, factor=0.5)
            verdicts.append(CertifiedEmission(
                outcome=CertificationOutcome.DEMOTED,
                emission=demoted,
                original=emission,
                rule_name=forbid_rule.name,
                reason=_render_reason(forbid_rule, facts),
            ))
        else:  # INFERRED
            verdicts.append(CertifiedEmission(
                outcome=CertificationOutcome.WARNING,
                emission=emission,
                rule_name=forbid_rule.name,
                reason=_render_reason(forbid_rule, facts),
            ))

    # ---- Emit verdicts for auto-added labels ----
    for label, causing_rules in auto_add_reasons.items():
        bank = _label_bank(label, live_vocab)
        if bank is None:
            # Rule references a label not in vocab. Skip auto-add; log
            # the drift as a WARNING synthesized against the first rule.
            first_rule = causing_rules[0]
            verdicts.append(CertifiedEmission(
                outcome=CertificationOutcome.WARNING,
                emission=AnchorEmission(
                    model_id=facts.model_id,
                    label="<unresolved>",
                    bank=Bank.QUALITY,  # placeholder — never written
                    confidence=0.0,
                    evidence=Provenance(
                        source_type=EvidenceType.DERIVED,
                        source_ref=label,
                        extractor="certifier",
                    ),
                ),
                rule_name=first_rule.name,
                reason=f"required label {label!r} not in vocabulary",
            ))
            continue

        first_rule = causing_rules[0]
        auto = AnchorEmission(
            model_id=facts.model_id,
            label=label,
            bank=bank,
            confidence=1.0,
            evidence=Provenance(
                source_type=EvidenceType.DERIVED,
                source_ref=f"rule={first_rule.name}",
                extractor="certifier",
            ),
        )
        verdicts.append(CertifiedEmission(
            outcome=CertificationOutcome.AUTO_ADDED,
            emission=auto,
            rule_name=first_rule.name,
            reason=f"required by rule: {_render_reason(first_rule, facts)}",
        ))

    return CertificationResult(model_id=facts.model_id, verdicts=tuple(verdicts))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_trigger(rule: Rule, facts: HFFacts) -> bool:
    """Fire a rule's trigger; swallow AttributeError for robustness against
    malformed facts (rare, but the certifier must never crash the pipeline)."""
    try:
        return bool(rule.trigger(facts))
    except (AttributeError, TypeError, KeyError):
        return False


def _render_reason(rule: Rule, facts: HFFacts) -> str:
    if not rule.reason_template:
        return rule.name
    try:
        return f"{rule.name}: {rule.reason_template.format(facts=facts)}"
    except (KeyError, IndexError, ValueError):
        return f"{rule.name}: {rule.reason_template}"


def _demote(emission: AnchorEmission, *, factor: float) -> AnchorEmission:
    """Return a copy of the emission with confidence * factor."""
    return AnchorEmission(
        model_id=emission.model_id,
        label=emission.label,
        bank=emission.bank,
        confidence=max(0.0, min(1.0, emission.confidence * factor)),
        evidence=emission.evidence,
        weight=emission.weight,
    )
