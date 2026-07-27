"""Eval-harness checks that need the real corpus.

Skipped when `network.db` is absent or empty, so CI stays green without a
351 MB download while a developer who has the corpus gets the real signal.

The floor here is deliberately a FLOOR, not the current score. It exists to
catch a collapse — a corpus regeneration that drops an anchor bank, a scoring
change that empties a window — not to freeze today's number. Ratcheting it up
after a genuine improvement is the intended workflow; tuning it down to make a
red run green is not.
"""

from __future__ import annotations

import pytest

from model_atlas import config, db
from model_atlas.evaluation.cases import CASES
from model_atlas.evaluation.harness import run_eval

# Measured 92.5% on the v0.4.2 corpus. Set below that so normal churn in a
# fast-moving hub does not fail the suite spuriously.
SCORE_FLOOR = 0.75


def _corpus_available() -> bool:
    if not config.NETWORK_DB_PATH.exists():
        return False
    try:
        conn = db.get_connection()
    except Exception:
        return False
    try:
        return int(conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]) > 1000
    finally:
        conn.close()


requires_corpus = pytest.mark.skipif(
    not _corpus_available(),
    reason="real corpus not present (see releases for network.db)",
)


@pytest.fixture(scope="module")
def report():
    conn = db.get_connection()
    try:
        yield run_eval(conn, CASES)
    finally:
        conn.close()


@requires_corpus
def test_corpus_quality_is_above_the_floor(report):
    assert report.score >= SCORE_FLOOR, (
        f"corpus quality {report.score:.1%} fell below the {SCORE_FLOOR:.0%} floor\n"
        + "\n".join(f"  {c.name}: {c.score:.0%}  {c.failures}" for c in report.cases)
    )


@requires_corpus
def test_no_case_returns_an_empty_window(report):
    """A canonical question with no answer means a required anchor vanished
    from the vocabulary — a corpus-regeneration bug, not a ranking nuance."""
    empty = [c.name for c in report.cases if c.empty]
    assert not empty, f"cases returned nothing: {empty}"


@requires_corpus
def test_every_case_returns_a_full_window(report):
    """Fewer results than asked for means the require filter is too narrow to
    be a useful recommendation, even when what came back is correct."""
    thin = [
        f"{c.name} ({len(c.returned)})"
        for c in report.cases
        if len(c.returned) < 2
    ]
    assert not thin, f"cases returned an unusably thin window: {thin}"
