"""Corpus-quality evaluation.

Canonical questions with expected answer *shapes*, scored against the real
corpus. The instrument that tells you whether a corpus fix helped.

    python -m model_atlas.evaluation

See `harness` for the scoring rules and `cases` for the questions.
"""

from __future__ import annotations

from .cases import CASES, EvalCase, case_by_name
from .facts import ModelFacts, Predicate, load_facts
from .harness import CaseResult, EvalReport, format_report, run_case, run_eval

__all__ = [
    "CASES",
    "CaseResult",
    "EvalCase",
    "EvalReport",
    "ModelFacts",
    "Predicate",
    "case_by_name",
    "format_report",
    "load_facts",
    "run_case",
    "run_eval",
]
