"""Run the canonical questions and score the answers.

The instrument for corpus-quality work. Every fix to the corpus — dropping
audit contradictions, correcting the EFFICIENCY zero state, adding certifier
rules, running the local model over the residue — is supposed to make these
numbers go up. Without a number, "the corpus got better" is a feeling.

Read-only. Runs `navigate()` exactly as the MCP tool does and inspects the
window it returns; it never writes.

Scoring is deliberately blunt. A case's score is the fraction of
(result, predicate) pairs that hold across the top-N window, plus one point
per satisfied `expect_any_of`. Blunt because the alternative — weighting
predicates by importance — invites tuning the metric instead of the corpus.

CLI usage::

    python -m model_atlas.evaluation                    # report to stdout
    python -m model_atlas.evaluation --json             # machine-readable
    python -m model_atlas.evaluation --save-baseline b.json
    python -m model_atlas.evaluation --baseline b.json  # diff against it
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass, field

from ..query_navigate import navigate
from ..query_types import StructuredQuery
from .cases import CASES, EvalCase
from .facts import ModelFacts, load_facts


@dataclass
class ResultCheck:
    """How one returned model fared against one case's predicates."""

    model_id: str
    rank: int
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    name: str
    ask: str
    returned: list[str]
    checks: list[ResultCheck]
    expected_hits: list[str]
    expected_misses: list[str]
    score: float
    total_checks: int
    passed_checks: int
    empty: bool = False

    @property
    def failures(self) -> list[str]:
        """Flat `model_id:predicate` list of everything that failed."""
        return [
            f"{c.model_id}:{p}" for c in self.checks for p in c.failed
        ]


def run_case(conn: sqlite3.Connection, case: EvalCase) -> CaseResult:
    """Execute one case and score the window it returns."""
    query = StructuredQuery(**case.query, limit=case.top_n)
    results = navigate(conn, query)[: case.top_n]
    returned = [r.model_id for r in results]

    checks: list[ResultCheck] = []
    passed = 0
    total = 0

    for rank, model_id in enumerate(returned, 1):
        facts: ModelFacts = load_facts(conn, model_id)
        rc = ResultCheck(model_id=model_id, rank=rank)
        for pred in case.require_all:
            total += 1
            if pred(facts):
                passed += 1
                rc.passed.append(pred.name)
            else:
                rc.failed.append(pred.name)
        checks.append(rc)

    hits: list[str] = []
    misses: list[str] = []
    for needle in case.expect_any_of:
        total += 1
        if any(needle.lower() in m.lower() for m in returned):
            passed += 1
            hits.append(needle)
        else:
            misses.append(needle)

    # An empty window scores zero rather than a vacuous 1.0 — returning
    # nothing is a failure to answer, not a set of satisfied constraints.
    empty = not returned
    if empty:
        score = 0.0
        total = max(total, 1)
    else:
        score = passed / total if total else 1.0

    return CaseResult(
        name=case.name,
        ask=case.ask,
        returned=returned,
        checks=checks,
        expected_hits=hits,
        expected_misses=misses,
        score=score,
        total_checks=total,
        passed_checks=passed,
        empty=empty,
    )


@dataclass
class EvalReport:
    cases: list[CaseResult]
    score: float
    passed_checks: int
    total_checks: int
    corpus_models: int

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "passed_checks": self.passed_checks,
            "total_checks": self.total_checks,
            "corpus_models": self.corpus_models,
            "cases": {
                c.name: {
                    "score": round(c.score, 4),
                    "returned": c.returned,
                    "failures": c.failures,
                    "expected_misses": c.expected_misses,
                }
                for c in self.cases
            },
        }


def run_eval(
    conn: sqlite3.Connection, cases: tuple[EvalCase, ...] = CASES
) -> EvalReport:
    """Run every case and aggregate.

    The headline `score` is over CHECKS, not cases — so a case with many
    expectations weighs more than a case with few, and adding a hard case
    cannot flatter the average by dilution.
    """
    results = [run_case(conn, c) for c in cases]
    passed = sum(r.passed_checks for r in results)
    total = sum(r.total_checks for r in results)
    corpus = int(conn.execute("SELECT COUNT(*) FROM models").fetchone()[0])
    return EvalReport(
        cases=results,
        score=(passed / total) if total else 0.0,
        passed_checks=passed,
        total_checks=total,
        corpus_models=corpus,
    )


def diff_reports(baseline: dict, current: dict) -> list[str]:
    """Human-readable per-case movement between two runs."""
    lines: list[str] = []
    b_score, c_score = baseline.get("score", 0.0), current.get("score", 0.0)
    delta = c_score - b_score
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
    lines.append(f"overall {b_score:.1%} → {c_score:.1%}  {arrow}{abs(delta):.1%}")
    b_cases = baseline.get("cases", {})
    c_cases = current.get("cases", {})
    for name in sorted(set(b_cases) | set(c_cases)):
        b = b_cases.get(name, {}).get("score")
        c = c_cases.get(name, {}).get("score")
        if b is None:
            lines.append(f"  + {name}: new, {c:.0%}")
        elif c is None:
            lines.append(f"  - {name}: removed (was {b:.0%})")
        elif abs(c - b) > 1e-9:
            mark = "▲" if c > b else "▼"
            lines.append(f"  {mark} {name}: {b:.0%} → {c:.0%}")
    return lines


def format_report(report: EvalReport, verbose: bool = False) -> str:
    """Plain-text report. The failure list is the actionable part."""
    out: list[str] = []
    out.append(f"Corpus quality: {report.score:.1%} "
               f"({report.passed_checks}/{report.total_checks} checks) "
               f"over {report.corpus_models:,} models")
    out.append("")
    for c in sorted(report.cases, key=lambda r: r.score):
        bar = "PASS" if c.score == 1.0 else f"{c.score:.0%}"
        out.append(f"[{bar:>4}] {c.name}")
        out.append(f"        {c.ask}")
        if c.empty:
            out.append("        (no results)")
        for rc in c.checks:
            status = "ok" if not rc.failed else "  ".join(rc.failed)
            if rc.failed or verbose:
                out.append(f"        #{rc.rank} {rc.model_id}")
                out.append(f"            {status}")
        for miss in c.expected_misses:
            out.append(f"        expected something matching {miss!r}, absent")
        out.append("")
    return "\n".join(out)


__all__ = [
    "CaseResult",
    "EvalReport",
    "ResultCheck",
    "asdict",
    "diff_reports",
    "format_report",
    "run_case",
    "run_eval",
]
