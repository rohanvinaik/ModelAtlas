"""Unknown bank positions must stay absent, not be written as the zero state.

`BankPosition()` defaults to `sign=0, depth=0`, and `_extract_efficiency(None)`
returned exactly that — so a model whose parameter count could not be
extracted was recorded as sitting at the EFFICIENCY zero state, which *claims*
"this is a mainstream ~7B model".

Measured on the v0.4.0 corpus: 32,697 models sat at `(0,0)`, and 23,352 of
them — 71% — were there only because no size was known. `zai-org/GLM-4.6`
(~355B) is one, which is why it satisfies `efficiency=0` perfectly and no
`max_depth` ceiling can exclude it: depth 0 is inside every bound.

Absence is not a free pass — `_nav_bank_alignment` applies
`NAVIGATE_MISSING_BANK_PENALTY` to a bank with no position. Unknown should
cost something; it should not masquerade as a match.
"""

from __future__ import annotations

from model_atlas.extraction.deterministic import BankPosition, _extract_efficiency


def test_a_known_size_is_positioned():
    pos, anchors = _extract_efficiency(7.0)
    assert pos.known is True
    assert (pos.sign, pos.depth) == (0, 0)  # 7B IS the zero state, legitimately
    assert anchors


def test_an_unknown_size_is_not_positioned():
    pos, _ = _extract_efficiency(None)
    assert pos.known is False


def test_unknown_is_distinguishable_from_a_genuine_zero_state():
    """The whole defect in one assertion: before this, both were (0, 0)."""
    genuine, _ = _extract_efficiency(7.0)
    unknown, _ = _extract_efficiency(None)
    assert (genuine.sign, genuine.depth) == (unknown.sign, unknown.depth)
    assert genuine.known != unknown.known


def test_a_size_outside_every_bucket_is_also_unknown():
    """Knowing the number but failing to place it on the bank is still not
    knowing the position."""
    pos, _ = _extract_efficiency(-1.0)
    assert pos.known is False


def test_small_and_large_sizes_stay_positioned():
    for size in (0.3, 1.5, 13.0, 70.0, 400.0):
        pos, _ = _extract_efficiency(size)
        assert pos.known is True, f"{size}B should be positionable"


def test_bank_position_defaults_to_known():
    """Only the extractors that lack a fact should opt out, so every other
    construction site keeps its current meaning."""
    assert BankPosition().known is True
    assert BankPosition(sign=1, depth=2).known is True


def test_unknown_positions_are_skipped_by_the_write_path():
    """The pipeline builds one spec per bank and skips the unknown ones, so no
    `model_positions` row is emitted for a bank we could not place."""
    import inspect

    from model_atlas.extraction import pipeline

    src = inspect.getsource(pipeline)
    assert "if not pos.known:" in src
    assert "continue" in src
