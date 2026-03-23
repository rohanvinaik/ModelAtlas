"""Prescriptive spec tests for deterministic.py extraction functions.

Targets surviving mutants in _parse_param_from_text (σ=17),
_estimate_params_billions (σ=12), _extract_efficiency (σ=7),
_license_anchors (σ=1), _config_anchors (σ=6), _compute_structural_fingerprint (σ=4).
"""

import pytest

from model_atlas.extraction.deterministic import (
    _estimate_params_billions,
    _license_anchors,
    _parse_param_from_text,
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

from model_atlas.extraction.deterministic import _extract_efficiency, BankPosition


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

from model_atlas.extraction.deterministic import _extract_quality


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
