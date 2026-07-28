"""A bank position is a SIGNAL, not a bare coordinate.

Step 1 of docs/architecture-upgrade.md. TriageGeist's banks emit
`BankSignal(estimate, confidence, floor, ceiling, evidence)`; ModelAtlas's
emitted `(sign, depth)` and nothing else, and most of the defects found while
auditing this corpus turned out to be downstream of that:

  - a guess written like a fact (23,352 models "at ~7B" because size was
    unknown) — no confidence
  - an anchor scraped from card boilerplate indistinguishable from one derived
    from `pipeline_tag` — no provenance
  - `coherence` dead at 99.95% = 1.0 — one estimator per bank cannot dissent

These pin the carrier. Nothing consumes `confidence` or `evidence` yet; that
is step 2 onward, and this step is deliberately behaviour-neutral.
"""

from __future__ import annotations

import sqlite3

import pytest

from model_atlas import db
from model_atlas.extraction.deterministic import BankPosition, _extract_efficiency


@pytest.fixture
def legacy(tmp_path):
    """A DB carrying the pre-signal schema, to exercise the migration."""
    conn = sqlite3.connect(tmp_path / "legacy.db")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE models (model_id TEXT PRIMARY KEY, author TEXT,
                             source TEXT, display_name TEXT);
        CREATE TABLE model_positions (
            model_id TEXT, bank TEXT, path_sign INTEGER, path_depth INTEGER,
            path_nodes TEXT, zero_state TEXT, PRIMARY KEY (model_id, bank));
        CREATE TABLE anchors (anchor_id INTEGER PRIMARY KEY AUTOINCREMENT,
                              label TEXT UNIQUE, bank TEXT, category TEXT);
        CREATE TABLE model_anchors (model_id TEXT, anchor_id INTEGER,
                                    weight REAL, PRIMARY KEY (model_id, anchor_id));
        CREATE TABLE model_metadata (model_id TEXT, key TEXT, value TEXT,
                                     value_type TEXT, PRIMARY KEY (model_id, key));
        INSERT INTO models VALUES ('old/model', 'o', 'huggingface', '');
        INSERT INTO model_positions VALUES ('old/model', 'EFFICIENCY', 0, 0, NULL, '~7B');
        """
    )
    conn.commit()
    return conn


# ── The carrier ───────────────────────────────────────────────────────


def test_positions_carry_confidence_and_evidence(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(model_positions)")}
    assert {"confidence", "evidence"} <= cols


def test_set_position_stores_the_signal(conn):
    db.insert_model(conn, "t/m", author="t", source="huggingface")
    db.set_position(conn, "t/m", "EFFICIENCY", -1, 2,
                    confidence=0.6, evidence="name_pattern=1B")
    row = conn.execute(
        "SELECT confidence, evidence FROM model_positions WHERE model_id='t/m'"
    ).fetchone()
    assert row["confidence"] == 0.6
    assert row["evidence"] == "name_pattern=1B"


def test_set_position_updates_the_signal_on_conflict(conn):
    """A re-extraction with better evidence must replace the weaker claim,
    not leave a stale provenance string attached to a new position."""
    db.insert_model(conn, "t/m", author="t", source="huggingface")
    db.set_position(conn, "t/m", "EFFICIENCY", -1, 2,
                    confidence=0.6, evidence="name_pattern=1B")
    db.set_position(conn, "t/m", "EFFICIENCY", 0, 0,
                    confidence=1.0, evidence="parameter_count_b=7")
    row = conn.execute(
        "SELECT path_sign, confidence, evidence FROM model_positions WHERE model_id='t/m'"
    ).fetchone()
    assert (row["path_sign"], row["confidence"]) == (0, 1.0)
    assert row["evidence"] == "parameter_count_b=7"


def test_defaults_preserve_pre_signal_behaviour(conn):
    """Callers not yet taught to supply a signal must keep working, and must
    not be silently marked low-confidence."""
    db.insert_model(conn, "t/m", author="t", source="huggingface")
    db.set_position(conn, "t/m", "QUALITY", 1, 1)
    row = conn.execute(
        "SELECT confidence, evidence FROM model_positions WHERE model_id='t/m'"
    ).fetchone()
    assert row["confidence"] == 1.0
    assert row["evidence"] == ""


# ── Migration ─────────────────────────────────────────────────────────


def test_migration_adds_the_columns_to_a_legacy_db(legacy):
    before = {r[1] for r in legacy.execute("PRAGMA table_info(model_positions)")}
    assert "confidence" not in before
    db.init_db(legacy)
    after = {r[1] for r in legacy.execute("PRAGMA table_info(model_positions)")}
    assert {"confidence", "evidence"} <= after


def test_migration_defaults_existing_rows_to_full_confidence(legacy):
    """Deliberately wrong for the corpus's 23,352 unknown-size rows, and the
    only default that leaves current behaviour unchanged. Correcting them is a
    migration with its own audit trail, not a schema default."""
    db.init_db(legacy)
    row = legacy.execute(
        "SELECT confidence, evidence FROM model_positions WHERE model_id='old/model'"
    ).fetchone()
    assert row["confidence"] == 1.0
    assert row["evidence"] == ""


def test_migration_is_idempotent(legacy):
    db.init_db(legacy)
    db.init_db(legacy)  # must not raise on the second ALTER
    assert legacy.execute("SELECT COUNT(*) FROM model_positions").fetchone()[0] == 1


# ── The extractor supplies it ─────────────────────────────────────────


def test_a_measured_size_records_the_fact_that_produced_it(conn):
    pos, _ = _extract_efficiency(7.0)
    assert pos.confidence == 1.0
    assert pos.evidence == "parameter_count_b=7"


def test_an_absent_size_records_why_it_is_unknown(conn):
    pos, _ = _extract_efficiency(None)
    assert pos.confidence == 0.0
    assert pos.evidence == "parameter_count_b=None"


def test_an_unbucketable_size_is_unknown_but_says_so_differently(conn):
    """Knowing the number and failing to place it is a different failure from
    never having the number, and the provenance should say which."""
    pos, _ = _extract_efficiency(-1.0)
    assert pos.confidence == 0.0
    assert "unbucketed" in pos.evidence


def test_known_is_derived_from_confidence(conn):
    """`known` was the 1-bit prototype of `confidence`; it stays as a derived
    property so the write path and its tests keep reading naturally."""
    assert BankPosition(confidence=1.0).known is True
    assert BankPosition(confidence=0.25).known is True
    assert BankPosition(confidence=0.0).known is False


# ── Bucket edges ──────────────────────────────────────────────────────
#
# Detective left 13 BOUNDARY/VALUE behaviours unpinned on `_extract_efficiency`,
# and they are the `_PARAM_RANGES` edges — the numbers that decide which size
# class a model lands in. An off-by-one there silently misfiles models, and
# every depth constraint built on the bank inherits the error.


@pytest.mark.parametrize(
    "params,sign,depth,anchor",
    [
        (0.0, -1, 3, "sub-1B"),      # lower edge of the first bucket
        (0.49, -1, 3, "sub-1B"),
        (0.5, -1, 2, "1B-class"),    # boundaries are inclusive-below
        (1.49, -1, 2, "1B-class"),
        (1.5, -1, 1, "3B-class"),
        (4.99, -1, 1, "3B-class"),
        (5.0, 0, 0, "7B-class"),     # the zero state begins exactly here
        (9.99, 0, 0, "7B-class"),
        (10.0, 1, 1, "13B-class"),
        (19.99, 1, 1, "13B-class"),
        (20.0, 1, 2, "30B-class"),
        (49.99, 1, 2, "30B-class"),
        (50.0, 1, 3, "70B-class"),
        (99.99, 1, 3, "70B-class"),
        (100.0, 1, 4, "frontier-class"),
        (2000.0, 1, 4, "frontier-class"),
    ],
)
def test_size_bucket_edges(params, sign, depth, anchor):
    pos, anchors = _extract_efficiency(params)
    assert (pos.sign, pos.depth) == (sign, depth)
    assert anchor in anchors
    assert pos.known is True


def test_consumer_gpu_and_edge_anchors_have_their_own_thresholds():
    """These are separate cutoffs from the bucket edges and drift independently."""
    assert "consumer-GPU-viable" in _extract_efficiency(3.99)[1]
    assert "consumer-GPU-viable" not in _extract_efficiency(4.0)[1]
    assert "edge-deployable" in _extract_efficiency(0.99)[1]
    assert "edge-deployable" not in _extract_efficiency(1.0)[1]


# ── Step 1.5: raw facts survive extraction ────────────────────────────


def test_raw_structural_facts_are_retained():
    """Extraction READ these and discarded them, leaving the corpus unable to
    answer whether its own sources agreed. A coherence layer needs several
    INDEPENDENT estimators per bank, and independence means "derived from a
    different raw fact" — so the raw facts have to survive."""
    from model_atlas.extraction.deterministic import ModelInput, extract

    meta = extract(ModelInput(
        model_id="org/m",
        tags=["license:apache-2.0", "gguf"],
        safetensors_info={"total": 7_000_000_000, "parameters": {"BF16": 7_000_000_000}},
        config={"architectures": ["LlamaForCausalLM"], "hidden_size": 4096,
                "quantization_config": {"bits": 4}},
    )).metadata
    assert {"tags", "architectures", "safetensors_total",
            "safetensors_dtypes", "quantization_config"} <= set(meta)


def test_license_is_read_from_the_tag_where_hf_actually_puts_it():
    """`ModelInfo` has no `.license` attribute, so every ingest path's
    `getattr(info, "license", "")` returned "" — which is why the shipped
    corpus holds 0 license rows despite the collector always asking for one."""
    from model_atlas.extraction.deterministic import ModelInput, extract

    meta = extract(ModelInput(model_id="org/m", tags=["license:apache-2.0"])).metadata
    assert meta["license"][0] == "apache-2.0"


def test_an_explicit_license_still_wins_over_the_tag():
    from model_atlas.extraction.deterministic import ModelInput, extract

    meta = extract(ModelInput(model_id="org/m", license_str="mit",
                              tags=["license:apache-2.0"])).metadata
    assert meta["license"][0] == "mit"


def test_absent_raw_facts_add_no_rows():
    """A model with nothing to record must not gain empty placeholder rows."""
    from model_atlas.extraction.deterministic import ModelInput, extract

    meta = extract(ModelInput(model_id="org/bare")).metadata
    assert not {"tags", "architectures", "safetensors_total",
                "quantization_config"} & set(meta)
