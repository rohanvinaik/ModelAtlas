"""Tests for D2 dictionary expansion."""

from __future__ import annotations

import json
import sqlite3

import pytest
import yaml

from model_atlas import db
from model_atlas.phase_d_expand import expand_dictionary


@pytest.fixture
def network_conn():
    """In-memory network database."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    db.init_db(conn)
    return conn


def _add_model(conn, model_id, author="test", pipeline_tag="text-generation", tags=None):
    db.insert_model(conn, model_id, author=author)
    if pipeline_tag:
        db.set_metadata(conn, model_id, "pipeline_tag", pipeline_tag, "str")
    if tags:
        db.set_metadata(conn, model_id, "tags", json.dumps(tags), "json")


def _write_spec(tmp_path, expansions):
    spec_path = tmp_path / "test_spec.yaml"
    spec_path.write_text(yaml.dump({"expansions": expansions}))
    return spec_path


class TestExpandDictionary:
    def test_create_only_mode(self, network_conn, tmp_path):
        """create_only creates anchor but links nothing."""
        spec = _write_spec(tmp_path, [{
            "label": "test-domain",
            "bank": "DOMAIN",
            "category": "test",
            "mode": "create_only",
            "match_rules": {"operator": "OR", "conditions": [{"type": "tag_exact", "value": "test"}], "min_matches": 1},
        }])
        _add_model(network_conn, "test/model-a")

        result = expand_dictionary(network_conn, spec)
        assert result.anchors_created == 1
        assert result.models_linked == 0

        row = network_conn.execute(
            "SELECT label, bank FROM anchors WHERE label = 'test-domain'"
        ).fetchone()
        assert row is not None
        assert row[1] == "DOMAIN"

    def test_auto_link_tag_exact(self, network_conn, tmp_path):
        """auto_link with tag_exact links matching models."""
        spec = _write_spec(tmp_path, [{
            "label": "biology-domain",
            "bank": "DOMAIN",
            "category": "specialization",
            "mode": "auto_link",
            "confidence": 0.7,
            "match_rules": {
                "operator": "OR",
                "conditions": [{"type": "tag_exact", "value": "biology"}],
                "min_matches": 1,
            },
        }])
        # Model with biology tag (via anchor)
        _add_model(network_conn, "test/bio-model")
        anchor_id = db.get_or_create_anchor(network_conn, "biology", "DOMAIN", source="test")
        db.link_anchor(network_conn, "test/bio-model", anchor_id)

        # Model without biology
        _add_model(network_conn, "test/other-model")

        result = expand_dictionary(network_conn, spec)
        assert result.anchors_created == 1
        assert result.models_linked == 1
        assert result.per_label["biology-domain"]["matched"] == 1

    def test_auto_link_name_regex(self, network_conn, tmp_path):
        """auto_link with name_regex matches model_id."""
        spec = _write_spec(tmp_path, [{
            "label": "physics-domain",
            "bank": "DOMAIN",
            "category": "specialization",
            "mode": "auto_link",
            "confidence": 0.7,
            "match_rules": {
                "operator": "OR",
                "conditions": [{"type": "name_regex", "value": r"\bphysics\b"}],
                "min_matches": 1,
            },
        }])
        _add_model(network_conn, "org/physics-gpt")
        _add_model(network_conn, "org/other-model")

        result = expand_dictionary(network_conn, spec)
        assert result.models_linked == 1

    def test_auto_link_pipeline_tag_in(self, network_conn, tmp_path):
        """auto_link with pipeline_tag_in matches pipeline_tag."""
        spec = _write_spec(tmp_path, [{
            "label": "music-domain",
            "bank": "DOMAIN",
            "category": "specialization",
            "mode": "auto_link",
            "confidence": 0.7,
            "match_rules": {
                "operator": "OR",
                "conditions": [{"type": "pipeline_tag_in", "value": ["text-to-audio"]}],
                "min_matches": 1,
            },
        }])
        _add_model(network_conn, "org/music-gen", pipeline_tag="text-to-audio")
        _add_model(network_conn, "org/text-model", pipeline_tag="text-generation")

        result = expand_dictionary(network_conn, spec)
        assert result.models_linked == 1

    def test_and_operator(self, network_conn, tmp_path):
        """AND operator requires all conditions to match."""
        spec = _write_spec(tmp_path, [{
            "label": "bio-gen",
            "bank": "DOMAIN",
            "category": "specialization",
            "mode": "auto_link",
            "confidence": 0.7,
            "match_rules": {
                "operator": "AND",
                "conditions": [
                    {"type": "name_regex", "value": r"\bbio\b"},
                    {"type": "pipeline_tag_in", "value": ["text-generation"]},
                ],
            },
        }])
        # Matches both conditions
        _add_model(network_conn, "org/bio-llm", pipeline_tag="text-generation")
        # Only matches name
        _add_model(network_conn, "org/bio-classify", pipeline_tag="text-classification")
        # Only matches pipeline_tag
        _add_model(network_conn, "org/regular-llm", pipeline_tag="text-generation")

        result = expand_dictionary(network_conn, spec)
        assert result.models_linked == 1

    def test_queue_for_heal_mode(self, network_conn, tmp_path):
        """queue_for_heal creates audit findings."""
        spec = _write_spec(tmp_path, [{
            "label": "chem-domain",
            "bank": "DOMAIN",
            "category": "specialization",
            "mode": "queue_for_heal",
            "match_rules": {
                "operator": "OR",
                "conditions": [{"type": "name_regex", "value": r"\bchem\b"}],
                "min_matches": 1,
            },
        }])
        _add_model(network_conn, "org/chem-model")

        result = expand_dictionary(network_conn, spec)
        assert result.models_queued == 1

        finding = network_conn.execute(
            "SELECT mismatch_type FROM audit_findings WHERE model_id = 'org/chem-model'"
        ).fetchone()
        assert finding[0] == "expansion_candidate"

    def test_dry_run_no_writes(self, network_conn, tmp_path):
        """dry_run previews counts without writing."""
        spec = _write_spec(tmp_path, [{
            "label": "test-anchor",
            "bank": "DOMAIN",
            "mode": "auto_link",
            "confidence": 0.7,
            "match_rules": {
                "operator": "OR",
                "conditions": [{"type": "name_regex", "value": r"\btest\b"}],
                "min_matches": 1,
            },
        }])
        _add_model(network_conn, "org/test-model")

        result = expand_dictionary(network_conn, spec, dry_run=True)
        assert result.anchors_created == 0  # dry run doesn't create

        row = network_conn.execute(
            "SELECT 1 FROM anchors WHERE label = 'test-anchor'"
        ).fetchone()
        assert row is None

    def test_existing_anchor_not_recreated(self, network_conn, tmp_path):
        """Pre-existing anchors are not duplicated."""
        db.get_or_create_anchor(network_conn, "biology-domain", "DOMAIN", source="bootstrap")

        spec = _write_spec(tmp_path, [{
            "label": "biology-domain",
            "bank": "DOMAIN",
            "mode": "create_only",
            "match_rules": {"operator": "OR", "conditions": [{"type": "tag_exact", "value": "x"}], "min_matches": 1},
        }])

        result = expand_dictionary(network_conn, spec)
        assert result.anchors_created == 0

    def test_run_record_created(self, network_conn, tmp_path):
        """Expansion creates a phase_d_runs record."""
        spec = _write_spec(tmp_path, [{
            "label": "x-domain",
            "bank": "DOMAIN",
            "mode": "create_only",
            "match_rules": {"operator": "OR", "conditions": [{"type": "tag_exact", "value": "x"}], "min_matches": 1},
        }])

        result = expand_dictionary(network_conn, spec)

        row = network_conn.execute(
            "SELECT phase, status FROM phase_d_runs WHERE run_id = ?",
            (result.run_id,),
        ).fetchone()
        assert row[0] == "d2"
        assert row[1] == "completed"
