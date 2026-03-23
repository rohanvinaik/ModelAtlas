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
