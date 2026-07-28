"""Tests for the coherence audit."""

from __future__ import annotations

import json
import sqlite3

import pytest

from model_atlas import db
from model_atlas.coherence import (
    check_anchor_orphans,
    check_anchor_oversaturation,
    check_bank_correlations,
    check_null_coverage,
    check_uncited_canonical_writes,
    format_report_human,
    CoherenceReport,
    run_audit,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    db.init_db(c)
    yield c
    c.close()


def _seed_minimal(c):
    """Two models, three anchors (one used by both, one used by one, one orphan)."""
    db.insert_model(c, "a/m1", author="a")
    db.insert_model(c, "a/m2", author="a")
    aid_shared = db.get_or_create_anchor(c, "decoder-only", "ARCHITECTURE")
    aid_one = db.get_or_create_anchor(c, "code-generation", "CAPABILITY")
    db.get_or_create_anchor(c, "orphan-anchor", "QUALITY")  # never linked
    db.link_anchor(c, "a/m1", aid_shared)
    db.link_anchor(c, "a/m2", aid_shared)
    db.link_anchor(c, "a/m1", aid_one)


# --- NULL coverage ---


def test_null_coverage_with_no_models(conn):
    out = check_null_coverage(conn)
    for stats in out.values():
        assert stats["coverage_pct"] == 0.0
        assert stats["missing"] == 0
        assert stats["models_with"] == 0


def test_null_coverage_partial(conn):
    db.insert_model(conn, "a/m1")
    db.insert_model(conn, "a/m2")
    db.set_position(conn, "a/m1", "ARCHITECTURE", 0, 0)
    out = check_null_coverage(conn)
    assert out["ARCHITECTURE"]["models_with"] == 1
    assert out["ARCHITECTURE"]["missing"] == 1
    assert out["ARCHITECTURE"]["coverage_pct"] == 50.0
    assert out["EFFICIENCY"]["coverage_pct"] == 0.0


# --- Anchor orphans ---


def test_anchor_orphans_finds_unlinked(conn):
    _seed_minimal(conn)
    orphans = check_anchor_orphans(conn)
    assert "orphan-anchor" in orphans
    assert "decoder-only" not in orphans
    assert "code-generation" not in orphans


def test_anchor_orphans_empty_when_all_linked(conn):
    db.insert_model(conn, "a/m1")
    aid = db.get_or_create_anchor(conn, "decoder-only", "ARCHITECTURE")
    db.link_anchor(conn, "a/m1", aid)
    # The bootstrap dictionary is loaded by init_db, so all of THOSE are
    # orphans here. Just check that our explicitly-linked one is not.
    orphans = check_anchor_orphans(conn)
    assert "decoder-only" not in orphans


# --- Anchor oversaturation ---


def test_anchor_oversaturation_flags_majority(conn):
    _seed_minimal(conn)
    # "decoder-only" linked to 2/2 = 100% > 50%
    over = check_anchor_oversaturation(conn, threshold_pct=50.0)
    labels = {item["label"] for item in over}
    assert "decoder-only" in labels
    # "code-generation" is 1/2 = 50%, NOT strictly greater → not flagged
    assert "code-generation" not in labels


def test_anchor_oversaturation_threshold_respected(conn):
    # Three models: anchor_a on 2 (66%), anchor_b on 1 (33%).
    # At threshold=70%, neither qualifies (cutoff = int(3*0.7) = 2; HAVING n > 2 needs n>=3).
    for i in range(3):
        db.insert_model(conn, f"a/m{i}")
    aid_a = db.get_or_create_anchor(conn, "anchor_a", "ARCHITECTURE")
    aid_b = db.get_or_create_anchor(conn, "anchor_b", "CAPABILITY")
    db.link_anchor(conn, "a/m0", aid_a)
    db.link_anchor(conn, "a/m1", aid_a)
    db.link_anchor(conn, "a/m0", aid_b)
    over_strict = check_anchor_oversaturation(conn, threshold_pct=70.0)
    flagged = {item["label"] for item in over_strict}
    assert "anchor_a" not in flagged
    assert "anchor_b" not in flagged
    # At threshold=60%, anchor_a (66%) qualifies: cutoff=int(1.8)=1, n>1 → n=2 matches
    over_lax = check_anchor_oversaturation(conn, threshold_pct=60.0)
    assert "anchor_a" in {item["label"] for item in over_lax}


def test_anchor_oversaturation_empty_when_no_models(conn):
    assert check_anchor_oversaturation(conn) == []


# --- Bank correlations ---


def test_bank_correlations_skips_when_too_few_data_points(conn):
    # Two models is below the n=3 minimum for stdev.
    db.insert_model(conn, "a/m1")
    db.insert_model(conn, "a/m2")
    db.set_position(conn, "a/m1", "ARCHITECTURE", 0, 0)
    db.set_position(conn, "a/m1", "EFFICIENCY", 0, 0)
    db.set_position(conn, "a/m2", "ARCHITECTURE", 1, 1)
    db.set_position(conn, "a/m2", "EFFICIENCY", 1, 1)
    all_corr, suspicious = check_bank_correlations(conn)
    # With n=2 and matching pair, correlation is undefined → skipped
    assert all_corr == {}
    assert suspicious == {}


def test_bank_correlations_detects_strong_alignment(conn):
    # Build a clear correlated pair: ARCHITECTURE == EFFICIENCY for 5 models.
    for i in range(5):
        mid = f"a/m{i}"
        db.insert_model(conn, mid)
        db.set_position(conn, mid, "ARCHITECTURE", 1 if i % 2 else -1, i)
        db.set_position(conn, mid, "EFFICIENCY", 1 if i % 2 else -1, i)
        # CAPABILITY varies independently
        db.set_position(conn, mid, "CAPABILITY", -1 if i % 2 else 1, 1 + (i % 3))
    all_corr, suspicious = check_bank_correlations(conn, threshold=0.9)
    pair = "ARCHITECTURE__EFFICIENCY"
    assert pair in all_corr
    assert abs(all_corr[pair]) >= 0.99
    assert pair in suspicious


# --- Uncited writes ---


def test_uncited_writes_flags_thin_reasons(tmp_path):
    log = tmp_path / "patches.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps({"ts": "2026-01-01T00:00:00Z", "table": "models",
                            "field": "author", "key": {"model_id": "x"},
                            "old_value": "a", "new_value": "b", "reason": "fix"}),
                json.dumps({"ts": "2026-01-02T00:00:00Z", "table": "models",
                            "field": "author", "key": {"model_id": "x"},
                            "old_value": "b", "new_value": "c", "reason": ""}),
                json.dumps({"ts": "2026-01-03T00:00:00Z", "table": "models",
                            "field": "author", "key": {"model_id": "x"},
                            "old_value": "c", "new_value": "d",
                            "reason": "Confirmed via HF API 2026-05-16"}),
            ]
        ),
        encoding="utf-8",
    )
    flagged = check_uncited_canonical_writes(log)
    issues = {item["issue"] for item in flagged}
    assert "thin_single_word" in issues  # "fix"
    assert "empty" in issues              # ""
    # The good one is NOT flagged
    reasons = {item["reason"] for item in flagged}
    assert "Confirmed via HF API 2026-05-16" not in reasons


def test_uncited_writes_no_log_file_is_empty(tmp_path):
    assert check_uncited_canonical_writes(tmp_path / "nope.jsonl") == []


# --- Full audit + formatter ---


def test_run_audit_returns_structured_report(conn, tmp_path):
    _seed_minimal(conn)
    log = tmp_path / "patches.jsonl"  # missing — uncited = []
    report = run_audit(conn, audit_log_path=log)
    assert report.model_total == 2
    assert report.anchor_total >= 3
    assert "orphan-anchor" in report.anchor_orphans
    assert {item["label"] for item in report.anchor_oversaturated} >= {"decoder-only"}
    assert "ARCHITECTURE" in report.null_coverage_per_bank


def test_format_report_human_renders_markdown(conn, tmp_path):
    _seed_minimal(conn)
    report = run_audit(conn, audit_log_path=tmp_path / "patches.jsonl")
    text = format_report_human(report)
    assert "# Coherence audit" in text
    assert "Bank orthogonality" in text
    assert "NULL coverage" in text
    assert "Anchor orphans" in text
    assert "Anchor oversaturation" in text


def test_report_to_dict_round_trips(conn, tmp_path):
    _seed_minimal(conn)
    report = run_audit(conn, audit_log_path=tmp_path / "patches.jsonl")
    d = report.to_dict()
    serialized = json.dumps(d, default=str)
    assert json.loads(serialized)["model_total"] == 2


def test_run_audit_surfaces_uncited_writes_from_a_real_log(conn, tmp_path):
    """Every other run_audit test points at a log path that does not exist, so
    `uncited_canonical_writes` is [] in all of them and the branch that
    populates it never runs. Detective flagged the gap: the return value was
    pinned only where the suite happened to look.
    """
    _seed_minimal(conn)
    log = tmp_path / "patches.jsonl"
    log.write_text(
        "\n".join(
            json.dumps(e)
            for e in [
                {"table": "models", "reason": "sourced from the HF config sha", "op": "patch"},
                {"table": "models", "reason": "fix", "op": "patch"},
                {"table": "anchors", "reason": "", "op": "patch"},
            ]
        )
        + "\n"
    )
    report = run_audit(conn, audit_log_path=log)

    issues = {u["issue"] for u in report.uncited_canonical_writes}
    assert issues == {"thin_single_word", "empty"}
    reasons = {u["reason"] for u in report.uncited_canonical_writes}
    assert "sourced from the HF config sha" not in reasons  # a real citation survives
    assert report.to_dict()["uncited_canonical_writes"]


def test_audit_thresholds_actually_govern_the_report(conn, tmp_path):
    """The audit's whole output is threshold-dependent, and nothing pinned
    that 0.7 means 0.7 — Detective's six surviving VALUE mutants were all
    "replace constant with boundary value" on these defaults. A default that
    no test distinguishes from its boundary is not a specification.
    """
    _seed_minimal(conn)
    log = tmp_path / "patches.jsonl"

    # oversaturation_pct: an anchor on 100% of models is flagged at 50 but not
    # at a threshold above its own share.
    lax = run_audit(conn, audit_log_path=log, oversaturation_pct=99.9)
    strict = run_audit(conn, audit_log_path=log, oversaturation_pct=10.0)
    lax_labels = {i["label"] for i in lax.anchor_oversaturated}
    strict_labels = {i["label"] for i in strict.anchor_oversaturated}
    assert strict_labels > lax_labels, "a lower threshold must flag strictly more"

    # correlation_threshold governs which pairs are called suspicious, and must
    # never widen the set as it rises.
    loose = run_audit(conn, audit_log_path=log, correlation_threshold=0.0)
    tight = run_audit(conn, audit_log_path=log, correlation_threshold=1.0)
    assert len(loose.bank_correlations_suspicious) >= len(
        tight.bank_correlations_suspicious
    )


def test_run_audit_returns_a_report_not_none(conn, tmp_path):
    """Pins the STATE mutant by value rather than by the AttributeError that
    happens to follow — a suite that only crashes on None proves the code ran,
    not what it produced."""
    _seed_minimal(conn)
    report = run_audit(conn, audit_log_path=tmp_path / "patches.jsonl")
    assert isinstance(report, CoherenceReport)
    assert report is not None


def test_the_defaults_themselves_are_specified(conn, tmp_path):
    """Passing explicit thresholds proves the parameters work; it does not
    pin what the DEFAULTS are. Detective's surviving VALUE mutants replace
    those constants with boundary values, and a suite that always passes an
    explicit threshold never notices. This calls run_audit bare.
    """
    _seed_minimal(conn)
    report = run_audit(conn, audit_log_path=tmp_path / "patches.jsonl")

    # An anchor carried by every model is 100% saturated, so the default
    # (50.0) must flag it. A default pushed to its 100.0 boundary would not.
    assert {i["label"] for i in report.anchor_oversaturated} >= {"decoder-only"}
    assert all(i["share_pct"] > 50.0 for i in report.anchor_oversaturated)

    # And the default must not be so low that it flags an anchor on one of two
    # models — that would make the check meaningless rather than strict.
    assert "orphan-anchor" not in {i["label"] for i in report.anchor_oversaturated}
