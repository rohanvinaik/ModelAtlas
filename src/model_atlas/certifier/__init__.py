"""Answer-certifier for the model-atlas audit pipeline.

Central export: `certify(...)` — takes a model's HF-fact bundle plus a set of
proposed anchor emissions from upstream extractors, applies the deterministic
rule library, and returns a per-emission verdict (CERTIFIED / DEMOTED /
REJECTED / AUTO_ADDED / WARNING) with reason + rule name attached.

Design mirrors triagegeist/src/answer_certifier.py — the LLM's decision is
not accepted unless it passes deterministic checks that verify internal
consistency between the chosen anchors and the underlying HF facts.
Certification failures are SIGNAL, not just rejection: they get logged and
feed back into rule-refinement.
"""
from __future__ import annotations

from .rules import ALL_RULES, HFFacts, RuleTier
from .certifier import certify

__all__ = ["certify", "ALL_RULES", "HFFacts", "RuleTier"]
