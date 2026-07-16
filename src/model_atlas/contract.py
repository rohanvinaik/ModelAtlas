"""Typed contract for anchor emissions — the audit-pipeline foundation.

Every anchor that lands on a model — whether from deterministic extraction,
pattern matching, LLM inference, or web enrichment — passes through this
contract. The types enforce:

  1. Closed vocabulary — labels must be in the seeded anchor table
     (BOOTSTRAP_ANCHORS). Unknown labels are structurally rejected
     at construction time, not silently dropped downstream.

  2. Bank consistency — an emission's `bank` must match the vocabulary's
     bank for that label. Mismatch is structural, not a warning.

  3. Provenance — every emission carries where the evidence came from
     (source_type + source_ref), so the certifier and the audit log can
     verify that the claim traces back to concrete HF facts, not a
     free-text LLM guess with a made-up confidence.

  4. Evidence typing — `EvidenceType` distinguishes "structural HF facts"
     (pipeline_tag, config.json, safetensors metadata) from "inferred
     signal" (LLM, pattern match). The certifier treats these tiers
     differently; the LLM tier is subject to hard-rule veto by the
     structural tier.

Design ported in spirit from triagegeist/src/triage_contract.py, adapted
to the model-metadata domain. This module has ZERO side effects and MUST
NOT import anything that writes to the DB. It is a pure type layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .db_bootstrap import BOOTSTRAP_ANCHORS


class Bank(str, Enum):
    """The eight orthogonal semantic banks. Values match column labels in the
    canonical `anchors` table so serialization is a no-op."""
    ARCHITECTURE = "ARCHITECTURE"
    CAPABILITY = "CAPABILITY"
    EFFICIENCY = "EFFICIENCY"
    COMPATIBILITY = "COMPATIBILITY"
    LINEAGE = "LINEAGE"
    DOMAIN = "DOMAIN"
    QUALITY = "QUALITY"
    TRAINING = "TRAINING"


class EvidenceType(str, Enum):
    """Where an anchor's supporting evidence lives.

    Ordered by trust tier — structural facts (top) veto inferred signal
    (bottom) via the certifier's hard-floor rules.
    """
    # Tier 1 — structural HF facts (near-inviolable)
    PIPELINE_TAG = "pipeline_tag"           # HF API `pipeline_tag` field
    MODEL_TYPE = "model_type"               # config.json `model_type`
    CONFIG_JSON = "config_json"             # any other config.json field
    SAFETENSORS_METADATA = "safetensors"    # safetensors index metadata
    LIBRARY_NAME = "library_name"           # HF `library_name`
    LICENSE = "license"                     # HF `license`
    # Tier 2 — semi-structural (tag conventions + patterns)
    TAG_STRING = "tag_string"               # HF `tags` list (e.g. "quantization:gguf")
    NAME_PATTERN = "name_pattern"           # matches model_id / author regex
    # Tier 3 — inferred (rebuttable)
    MODEL_CARD = "model_card"               # extracted from README
    WEB_SOURCE = "web_source"               # Phase E web scrape
    LLM_INFERENCE = "llm_inference"         # LLM-derived, no other evidence
    BENCHMARK = "benchmark"                 # numeric benchmark threshold
    DERIVED = "derived"                     # cross-anchor derivation


TRUSTED_EVIDENCE = frozenset({
    EvidenceType.PIPELINE_TAG,
    EvidenceType.MODEL_TYPE,
    EvidenceType.CONFIG_JSON,
    EvidenceType.SAFETENSORS_METADATA,
    EvidenceType.LIBRARY_NAME,
    EvidenceType.LICENSE,
})
"""Structural evidence types. Anchors backed by these can veto Tier-3 emissions."""


@dataclass(frozen=True)
class Provenance:
    """Concrete pointer to the evidence source. Never free-text prose."""
    source_type: EvidenceType
    source_ref: str = ""
    """Machine-readable ref: for PIPELINE_TAG this is the tag value;
    for CONFIG_JSON this is the config key path; for WEB_SOURCE this is
    the URL. Empty when the evidence type itself is self-referential
    (e.g., NAME_PATTERN where source_ref would just repeat the pattern).
    """
    extractor: str = ""
    """The function/tool that produced the emission (e.g., "extract_deterministic",
    "phase_e_worker", "recertify_corpus"). For audit-trail routing."""


# ---------------------------------------------------------------------------
# Vocabulary loading — the single source of truth
# ---------------------------------------------------------------------------


def _load_vocabulary() -> dict[str, tuple[Bank, str]]:
    """Return {anchor_label: (bank, category)} from BOOTSTRAP_ANCHORS.

    Called once at import time and cached in VOCABULARY. Extended anchors
    added at runtime via admin.ensure_anchor() are NOT in this table — those
    are dynamically-materialized labels (e.g. author-specific `Qwen-family`
    variants). Runtime validation therefore accepts anchors present in the
    live `anchors` table too; this loader is only the static seed floor.
    """
    vocab: dict[str, tuple[Bank, str]] = {}
    for label, bank_str, category in BOOTSTRAP_ANCHORS:
        try:
            bank = Bank(bank_str)
        except ValueError as exc:
            raise ValueError(
                f"BOOTSTRAP_ANCHORS entry {label!r} has unknown bank {bank_str!r} "
                f"(not one of {[b.value for b in Bank]})"
            ) from exc
        vocab[label] = (bank, category)
    return vocab


VOCABULARY: dict[str, tuple[Bank, str]] = _load_vocabulary()
"""Static seed vocabulary loaded from BOOTSTRAP_ANCHORS.

Runtime-added anchors (via ensure_anchor()) are not in this dict; the
certifier consults the live DB for those. This mapping is the contract
floor — any label the pipeline emits MUST resolve to a bank via either
this mapping or a live `anchors` row.
"""


# ---------------------------------------------------------------------------
# Anchor emission — the atomic unit that flows through certifier -> reconciler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorEmission:
    """A single proposed anchor assignment for a model.

    Constructed by extractors (deterministic / pattern / LLM / web). Certified
    (accepted / rejected / auto-adjusted) by AnswerCertifier. Written to the
    DB via the reconciler in Phase 7 (currently written directly via
    db.link_anchor pending migration).

    Immutable so downstream layers can safely deduplicate and reorder.
    """
    model_id: str
    label: str
    bank: Bank
    confidence: float
    evidence: Provenance
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"AnchorEmission({self.model_id!r}, {self.label!r}): confidence "
                f"{self.confidence!r} outside [0, 1]"
            )
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError(
                f"AnchorEmission({self.model_id!r}, {self.label!r}): weight "
                f"{self.weight!r} outside [0, 1]"
            )
        # Static vocab check — runtime-added anchors will fall through to a
        # live-DB check in the certifier; here we catch obvious typos early.
        static = VOCABULARY.get(self.label)
        if static is not None and static[0] is not self.bank:
            raise ValueError(
                f"AnchorEmission({self.model_id!r}, {self.label!r}): declared "
                f"bank {self.bank.value} disagrees with vocabulary "
                f"{static[0].value}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "label": self.label,
            "bank": self.bank.value,
            "confidence": self.confidence,
            "weight": self.weight,
            "evidence": {
                "source_type": self.evidence.source_type.value,
                "source_ref": self.evidence.source_ref,
                "extractor": self.evidence.extractor,
            },
        }


# ---------------------------------------------------------------------------
# Certification outcome — the certifier's per-emission verdict
# ---------------------------------------------------------------------------


class CertificationOutcome(str, Enum):
    """Per-emission verdict from the certifier."""
    CERTIFIED = "certified"
    """Emission passes all rules — write."""
    REJECTED = "rejected"
    """Emission contradicts a hard rule — drop, log reason."""
    DEMOTED = "demoted"
    """Emission accepted but confidence lowered — write with adjusted confidence."""
    AUTO_ADDED = "auto_added"
    """Emission not in proposed set but required by structural evidence — added."""
    WARNING = "warning"
    """Emission passes but the certifier noticed something worth logging."""


@dataclass(frozen=True)
class CertifiedEmission:
    """The certifier's verdict on one emission (or auto-added one).

    Terminal record that goes to both the write layer and the audit log.
    `original` is None when this was auto-added by a rule (no user emission
    to reference).
    """
    outcome: CertificationOutcome
    emission: AnchorEmission
    """The (possibly-adjusted) emission to write. For REJECTED this is the
    original emission that got rejected; downstream must check outcome."""
    original: AnchorEmission | None = None
    """The user's original emission, if this was DEMOTED or AUTO_ADDED. Lets
    the audit trail show the delta."""
    reason: str = ""
    """Human-readable rule name + short explanation (e.g. "pipeline_tag_image_text_to_text: pipeline_tag=image-text-to-text forbids image-generation")."""
    rule_name: str = ""
    """Machine-readable ID of the rule that fired (or "" if none)."""


@dataclass(frozen=True)
class CertificationResult:
    """Full certifier output for one model's proposed anchor set."""
    model_id: str
    verdicts: tuple[CertifiedEmission, ...]

    def certified(self) -> tuple[CertifiedEmission, ...]:
        """Emissions to actually write (CERTIFIED, DEMOTED, AUTO_ADDED, WARNING)."""
        return tuple(
            v for v in self.verdicts
            if v.outcome is not CertificationOutcome.REJECTED
        )

    def rejected(self) -> tuple[CertifiedEmission, ...]:
        return tuple(
            v for v in self.verdicts
            if v.outcome is CertificationOutcome.REJECTED
        )

    def by_outcome(self) -> dict[CertificationOutcome, int]:
        counts: dict[CertificationOutcome, int] = {}
        for v in self.verdicts:
            counts[v.outcome] = counts.get(v.outcome, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# JSON schema for Ollama grammar-constrained output (Phase 8 consumes this)
# ---------------------------------------------------------------------------


def bank_output_schema(bank: Bank, allowed_labels: list[str]) -> dict[str, Any]:
    """Emit the JSON schema an LLM must match when selecting anchors for one bank.

    Used in Phase 8 to grammar-constrain Ollama output. The `allowed_labels`
    list is bank-scoped — the LLM literally cannot emit an off-vocab or
    wrong-bank label.
    """
    return {
        "type": "object",
        "properties": {
            "bank": {"const": bank.value},
            "selected_anchors": {
                "type": "array",
                "items": {"enum": sorted(allowed_labels)},
                "uniqueItems": True,
            },
            "evidence_span": {
                "type": "string",
                "description": (
                    "Exact quote from the source text that supports the selection. "
                    "Empty if no textual evidence (structural fact only)."
                ),
            },
        },
        "required": ["bank", "selected_anchors"],
        "additionalProperties": False,
    }
