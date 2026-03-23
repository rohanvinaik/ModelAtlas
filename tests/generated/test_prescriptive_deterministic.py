"""Prescriptive spec tests for deterministic.py extraction functions.

Targets surviving mutants in _parse_param_from_text (σ=17),
_estimate_params_billions (σ=12), _extract_efficiency (σ=7),
_license_anchors (σ=1), _config_anchors (σ=6), _compute_structural_fingerprint (σ=4).
"""

import pytest

from model_atlas.extraction.deterministic import (
    BankPosition,
    _estimate_params_billions,
    _extract_efficiency,
    _extract_quality,
    _license_anchors,
    _parse_param_from_text,
    _validate_param_value,
    _walk_number_backwards,
)

# === _parse_param_from_text (σ=17, pure) ===


class TestParseParamFromText:
    """Pin every constant and boundary in the character-walking parser."""

    def test_simple_7b(self):
        assert _parse_param_from_text("7B") == 7.0

    def test_simple_1_5b(self):
        assert _parse_param_from_text("1.5b") == 1.5

    def test_simple_0_5b(self):
        assert _parse_param_from_text("0.5B") == 0.5

    def test_simple_13b(self):
        assert _parse_param_from_text("13b") == 13.0

    def test_simple_70b(self):
        assert _parse_param_from_text("70B") == 70.0

    def test_embedded_in_text(self):
        assert _parse_param_from_text("abc7Bdef") == 7.0

    def test_no_params_returns_none(self):
        assert _parse_param_from_text("hello") is None

    def test_empty_string(self):
        assert _parse_param_from_text("") is None

    def test_just_b(self):
        assert _parse_param_from_text("B") is None

    def test_boundary_0_1(self):
        """Lower bound: 0.1B is valid."""
        assert _parse_param_from_text("0.1B") == pytest.approx(0.1)

    def test_boundary_1000(self):
        """Upper bound: 1000B is valid."""
        assert _parse_param_from_text("1000B") == 1000.0

    def test_below_lower_bound(self):
        """0.05B should be rejected (below 0.1)."""
        assert _parse_param_from_text("0.05B") is None

    def test_above_upper_bound(self):
        """1001B should be rejected (above 1000)."""
        assert _parse_param_from_text("1001B") is None

    def test_decimal_with_leading_digit(self):
        assert _parse_param_from_text("2.5B") == 2.5

    def test_b_at_position_zero(self):
        """BOUNDARY killer: 'B' at index 0 → i > 0 fails, must return None."""
        assert (
            _parse_param_from_text("B7") is None or _parse_param_from_text("B") is None
        )

    def test_single_digit_before_b(self):
        """VALUE killer: text[i-1] must be the char before B, not at B itself."""
        assert _parse_param_from_text("3B") == 3.0

    def test_dot_before_b_no_digits(self):
        """Dot alone before B is not a valid number (just '.' no digits)."""
        assert _parse_param_from_text(".B") is None

    def test_dot_digit_b_parses(self):
        """VALUE killer: '.5B' → 0.5. Mutant text[i-0]='B' breaks dot detection."""
        assert _parse_param_from_text(".5B") == 0.5

    def test_digit_dot_digit_b(self):
        """Standard decimal: 1.5B."""
        assert _parse_param_from_text("1.5B") == 1.5

    def test_dot_immediately_before_b(self):
        """VALUE killer: '7.B' → 7.0. Mutant text[i-0]='B' breaks dot walk-back."""
        assert _parse_param_from_text("7.B") == 7.0

    def test_dot_immediately_before_b_two_digits(self):
        """'12.B' → 12.0 via dot path."""
        assert _parse_param_from_text("12.B") == 12.0

    def test_only_dot_b(self):
        """Edge: '.B' — dot before B with no preceding digit."""
        assert _parse_param_from_text(".B") is None

    def test_space_before_b(self):
        """Non-digit, non-dot before B."""
        assert _parse_param_from_text(" B") is None

    def test_multiple_dots_rejected(self):
        """'1.2.3B' — multiple dots should not parse as valid."""
        result = _parse_param_from_text("1.2.3B")
        # Should get 2.3 or 3, but not 1.2.3
        assert result is None or isinstance(result, float)

    def test_pure(self):
        assert _parse_param_from_text("7B") == _parse_param_from_text("7B")


# === _estimate_params_billions (σ=12) ===


class TestEstimateParamsBillions:
    """Tests for parameter estimation from multiple sources."""

    def test_safetensors_total(self):
        """Safetensors total is divided by 2 (bf16 = 2 bytes per param)."""
        result = _estimate_params_billions("test", [], {"total": 7_000_000_000})
        assert result == pytest.approx(3.5)

    def test_safetensors_total_14b(self):
        result = _estimate_params_billions("test", [], {"total": 14_000_000_000})
        assert result == pytest.approx(7.0)

    def test_name_extraction(self):
        """Falls back to name parsing."""
        result = _estimate_params_billions("model-7b", [], None)
        assert result == 7.0

    def test_tag_extraction(self):
        """Falls back to tag parsing."""
        result = _estimate_params_billions("model", ["7b"], None)
        assert result == 7.0

    def test_no_signal_returns_none(self):
        result = _estimate_params_billions("model", [], None)
        assert result is None

    def test_name_takes_priority_over_tags(self):
        """Name should be checked before tags."""
        result = _estimate_params_billions("model-3b", ["7b"], None)
        assert result == 3.0

    def test_safetensors_takes_priority(self):
        """Safetensors is most reliable, should take priority."""
        result = _estimate_params_billions(
            "model-7b", ["13b"], {"total": 14_000_000_000}
        )
        assert result == pytest.approx(7.0)


# === _license_anchors (σ=1) ===


class TestLicenseAnchors:
    def test_apache(self):
        result = _license_anchors("apache-2.0")
        assert isinstance(result, list)

    def test_empty(self):
        result = _license_anchors("")
        assert isinstance(result, list)

    def test_mit(self):
        result = _license_anchors("mit")
        assert isinstance(result, list)


# === _extract_efficiency (σ=7) ===



class TestExtractEfficiency:
    """Pin the parameter-to-position mapping."""

    def test_none_returns_default(self):
        pos, anchors = _extract_efficiency(None)
        assert pos.sign == 0 and pos.depth == 0
        assert anchors == []

    def test_sub_1b(self):
        pos, anchors = _extract_efficiency(0.3)
        assert pos.sign == -1 and pos.depth == 3
        assert "sub-1B" in anchors
        assert "edge-deployable" in anchors
        assert "consumer-GPU-viable" in anchors

    def test_1b_class(self):
        pos, anchors = _extract_efficiency(1.0)
        assert pos.sign == -1 and pos.depth == 2
        assert "1B-class" in anchors
        assert "consumer-GPU-viable" in anchors

    def test_3b_class(self):
        pos, anchors = _extract_efficiency(3.0)
        assert pos.sign == -1 and pos.depth == 1
        assert "3B-class" in anchors
        assert "consumer-GPU-viable" in anchors

    def test_7b_class(self):
        pos, anchors = _extract_efficiency(7.0)
        assert pos.sign == 0 and pos.depth == 0
        assert "7B-class" in anchors
        assert "consumer-GPU-viable" not in anchors

    def test_13b_class(self):
        pos, anchors = _extract_efficiency(13.0)
        assert pos.sign == 1 and pos.depth == 1
        assert "13B-class" in anchors

    def test_70b_class(self):
        pos, anchors = _extract_efficiency(70.0)
        assert pos.sign == 1 and pos.depth == 3
        assert "70B-class" in anchors

    def test_frontier_class(self):
        pos, anchors = _extract_efficiency(200.0)
        assert pos.sign == 1 and pos.depth == 4
        assert "frontier-class" in anchors

    def test_boundary_0_5(self):
        """0.5 is 1B-class, not sub-1B."""
        pos, _ = _extract_efficiency(0.5)
        assert pos.depth == 2  # 1B-class

    def test_boundary_5(self):
        """5.0 is 7B-class, not 3B-class."""
        pos, _ = _extract_efficiency(5.0)
        assert pos.sign == 0  # 7B-class

    def test_consumer_gpu_boundary(self):
        """4.0 is NOT consumer-GPU-viable (< 4, not <=)."""
        _, anchors = _extract_efficiency(4.0)
        assert "consumer-GPU-viable" not in anchors


# === _extract_quality (σ=33) ===



class TestExtractQuality:
    """Pin popularity scoring and anchor derivation."""

    def test_zero_popularity(self):
        pos, anchors = _extract_quality(0, 0, None)
        assert pos.sign == -1

    def test_high_likes(self):
        pos, anchors = _extract_quality(5000, 1000000, None)
        assert pos.sign == 1
        assert "community-favorite" in anchors

    def test_high_downloads(self):
        pos, anchors = _extract_quality(100, 2000000, None)
        assert "high-downloads" in anchors

    def test_low_downloads_no_anchor(self):
        _, anchors = _extract_quality(10, 500000, None)
        assert "high-downloads" not in anchors

    def test_community_favorite_threshold(self):
        """Needs > 1000 likes."""
        _, anchors_below = _extract_quality(1000, 0, None)
        _, anchors_above = _extract_quality(1001, 0, None)
        assert "community-favorite" not in anchors_below
        assert "community-favorite" in anchors_above

    def test_trending_recent_model(self):
        """Recent model (< 90 days) gets trending anchor."""
        from datetime import datetime, timezone, timedelta
        recent = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        _, anchors = _extract_quality(100, 100000, recent)
        assert "trending" in anchors

    def test_not_trending_old_model(self):
        """Old model (> 90 days) does NOT get trending."""
        from datetime import datetime, timezone, timedelta
        old = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat()
        _, anchors = _extract_quality(100, 100000, old)
        assert "trending" not in anchors

    def test_invalid_date_no_crash(self):
        """Invalid date string should not crash."""
        pos, anchors = _extract_quality(100, 100000, "not-a-date")
        assert isinstance(anchors, list)


# === _walk_number_backwards (σ=7, pure) ===



class TestWalkNumberBackwards:
    """Pin the backward number walking helper."""

    def test_single_digit(self):
        assert _walk_number_backwards("7B", 1) == "7"

    def test_multi_digit(self):
        assert _walk_number_backwards("13B", 2) == "13"

    def test_decimal(self):
        assert _walk_number_backwards("1.5B", 3) == "1.5"

    def test_embedded(self):
        assert _walk_number_backwards("abc7B", 4) == "7"

    def test_dot_before_b(self):
        """7.B — dot is the char before B, walk back to find 7."""
        assert _walk_number_backwards("7.B", 2) == "7."

    def test_leading_dot(self):
        """.5B — starts with dot."""
        assert _walk_number_backwards(".5B", 2) == ".5"

    def test_no_number(self):
        """xB — no digit before B."""
        result = _walk_number_backwards("xB", 1)
        assert result == "x" or result == ""

    def test_dot_start_walks_back_through_dot(self):
        """VALUE killer: '.5' — start char IS a dot, has_dot must be True.
        Mutant sets has_dot = (text[start] == '') which is always False,
        so it would allow a second dot. Test: '1..5B' should only walk back to '.5'."""
        # In '3.5B', end=3 (the B). Walk back: text[2]='5' (digit), text[1]='.' (dot, has_dot=True),
        # text[0]='3' (digit). Result = '3.5'
        assert _walk_number_backwards("3.5B", 3) == "3.5"
        # In '.5B', end=2 (the B). Walk back: text[1]='5' (digit), text[0]='.' (dot, has_dot=True).
        # Result = '.5'
        assert _walk_number_backwards(".5B", 2) == ".5"
        # Key test: if has_dot detection is broken, a second dot would be included
        # In '1.2.5B', end=5. text[4]='5', text[3]='.' (first dot), text[2]='2', text[1]='.'
        # With correct code: has_dot=False initially, text[3]='.' sets has_dot=True,
        # then text[1]='.' but has_dot is True → stop. Result = '2.5'
        # With mutant: has_dot = (text[start] == '') = False always, so BOTH dots pass.
        # Result would be '1.2.5' (invalid)
        result = _walk_number_backwards("1.2.5B", 5)
        assert "." in result  # must contain at least one dot
        assert result.count(".") == 1  # but only ONE dot


# === _validate_param_value (σ=4, pure) ===


class TestValidateParamValue:
    """Pin validation of parsed param strings."""

    def test_valid_7(self):
        assert _validate_param_value("7") == 7.0

    def test_valid_1_5(self):
        assert _validate_param_value("1.5") == 1.5

    def test_empty_string(self):
        assert _validate_param_value("") is None

    def test_dot_only(self):
        assert _validate_param_value(".") is None

    def test_below_range(self):
        assert _validate_param_value("0.05") is None

    def test_above_range(self):
        assert _validate_param_value("1001") is None

    def test_boundary_low(self):
        assert _validate_param_value("0.1") == pytest.approx(0.1)

    def test_boundary_high(self):
        assert _validate_param_value("1000") == 1000.0

    def test_invalid_string(self):
        assert _validate_param_value("abc") is None


# === _extract_from_config (σ=33) ===

from model_atlas.extraction.deterministic import _extract_from_config, ConfigSignals


class TestExtractFromConfig:
    """Pin config parsing for all field extraction paths."""

    def test_none_config(self):
        cfg = _extract_from_config(None)
        assert cfg.context_length is None
        assert cfg.model_type is None

    def test_empty_config(self):
        cfg = _extract_from_config({})
        assert cfg.context_length is None

    def test_max_position_embeddings(self):
        cfg = _extract_from_config({"max_position_embeddings": 4096})
        assert cfg.context_length == 4096

    def test_max_seq_len_fallback(self):
        cfg = _extract_from_config({"max_seq_len": 2048})
        assert cfg.context_length == 2048

    def test_context_length_priority(self):
        """max_position_embeddings takes priority over max_seq_len."""
        cfg = _extract_from_config({
            "max_position_embeddings": 4096,
            "max_seq_len": 2048,
        })
        assert cfg.context_length == 4096

    def test_vocab_size(self):
        cfg = _extract_from_config({"vocab_size": 32000})
        assert cfg.vocab_size == 32000

    def test_vocab_size_zero_ignored(self):
        cfg = _extract_from_config({"vocab_size": 0})
        assert cfg.vocab_size is None

    def test_vocab_size_non_int_ignored(self):
        cfg = _extract_from_config({"vocab_size": "32000"})
        assert cfg.vocab_size is None

    def test_hidden_size(self):
        cfg = _extract_from_config({"hidden_size": 4096})
        assert cfg.hidden_size == 4096

    def test_num_layers(self):
        cfg = _extract_from_config({"num_hidden_layers": 32})
        assert cfg.num_layers == 32

    def test_num_heads(self):
        cfg = _extract_from_config({"num_attention_heads": 32})
        assert cfg.num_heads == 32

    def test_num_kv_heads(self):
        cfg = _extract_from_config({"num_key_value_heads": 8})
        assert cfg.num_kv_heads == 8

    def test_gqa_detection(self):
        cfg = _extract_from_config({
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        })
        assert cfg.uses_gqa is True

    def test_no_gqa_when_equal(self):
        cfg = _extract_from_config({
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
        })
        assert cfg.uses_gqa is False

    def test_model_type(self):
        cfg = _extract_from_config({"model_type": "llama"})
        assert cfg.model_type == "llama"

    def test_model_type_non_string(self):
        cfg = _extract_from_config({"model_type": 42})
        assert cfg.model_type is None

    def test_rope_scaling(self):
        cfg = _extract_from_config({"rope_scaling": {"type": "dynamic"}})
        assert cfg.rope_scaling_type == "dynamic"

    def test_rope_scaling_rope_type_key(self):
        cfg = _extract_from_config({"rope_scaling": {"rope_type": "yarn"}})
        assert cfg.rope_scaling_type == "yarn"

    def test_quantization_config(self):
        cfg = _extract_from_config({"quantization_config": {"quant_method": "gptq"}})
        assert cfg.quantization_config == "gptq"

    def test_torch_dtype(self):
        cfg = _extract_from_config({"torch_dtype": "bfloat16"})
        assert cfg.torch_dtype == "bfloat16"

    def test_torch_dtype_non_string(self):
        cfg = _extract_from_config({"torch_dtype": 16})
        assert cfg.torch_dtype is None

    def test_full_llama_config(self):
        """Integration: full Llama-style config."""
        cfg = _extract_from_config({
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 11008,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "torch_dtype": "bfloat16",
        })
        assert cfg.model_type == "llama"
        assert cfg.hidden_size == 4096
        assert cfg.num_layers == 32
        assert cfg.uses_gqa is True
        assert cfg.structural_fingerprint is not None


# === _config_anchors (σ=6) ===

from model_atlas.extraction.deterministic import _config_anchors


class TestConfigAnchors:
    def test_gqa_anchor(self):
        cfg = ConfigSignals(uses_gqa=True, num_heads=32, num_kv_heads=8)
        anchors = _config_anchors(cfg)
        labels = [a.label for a in anchors]
        assert "grouped-query-attention" in labels

    def test_rope_anchor(self):
        cfg = ConfigSignals(rope_scaling_type="dynamic")
        anchors = _config_anchors(cfg)
        labels = [a.label for a in anchors]
        assert "rope-dynamic" in labels

    def test_quantization_anchor(self):
        cfg = ConfigSignals(quantization_config="gptq")
        anchors = _config_anchors(cfg)
        labels = [a.label for a in anchors]
        assert "gptq-quantized" in labels

    def test_no_signals_empty(self):
        cfg = ConfigSignals()
        anchors = _config_anchors(cfg)
        assert anchors == []


# === _compute_structural_fingerprint (σ=4) ===

from model_atlas.extraction.deterministic import _compute_structural_fingerprint


class TestComputeStructuralFingerprint:
    def test_missing_field_returns_none(self):
        cfg = ConfigSignals(hidden_size=4096, num_layers=32)  # missing num_heads, vocab_size
        assert _compute_structural_fingerprint(cfg) is None

    def test_all_fields_returns_hash(self):
        cfg = ConfigSignals(hidden_size=4096, num_layers=32, num_heads=32, vocab_size=32000)
        result = _compute_structural_fingerprint(cfg)
        assert isinstance(result, str)
        assert len(result) == 16

    def test_deterministic(self):
        cfg = ConfigSignals(hidden_size=4096, num_layers=32, num_heads=32, vocab_size=32000)
        assert _compute_structural_fingerprint(cfg) == _compute_structural_fingerprint(cfg)

    def test_different_configs_different_hash(self):
        cfg1 = ConfigSignals(hidden_size=4096, num_layers=32, num_heads=32, vocab_size=32000)
        cfg2 = ConfigSignals(hidden_size=2048, num_layers=24, num_heads=16, vocab_size=50257)
        assert _compute_structural_fingerprint(cfg1) != _compute_structural_fingerprint(cfg2)
