"""Tests for `vibe_axes` — EPA (Evaluation/Potency/Activity) from vibe_summary text."""

from __future__ import annotations

from model_atlas import db, vibe_axes


def test_empty_text_abstains():
    """Empty / whitespace-only summaries yield None, not (0, 0, 0). An
    honest absence is the whole point — a false neutral would let a
    caller treat "no data" and "genuinely neutral" the same way."""
    assert vibe_axes.derive_epa("") is None
    assert vibe_axes.derive_epa("   \n\t") is None


def test_no_lexicon_hits_abstains():
    """A summary that avoids every lexicon word abstains too — the
    module never invents an axis reading from text it can't decompose."""
    assert vibe_axes.derive_epa("xyz qwerty asdf zxcv") is None


def test_positive_evaluation():
    epa = vibe_axes.derive_epa("A capable, reliable, polished model.")
    assert epa is not None
    e, _, _ = epa
    assert e > 0.5


def test_negative_evaluation():
    epa = vibe_axes.derive_epa("A brittle, buggy, weak checkpoint.")
    assert epa is not None
    e, _, _ = epa
    assert e < -0.5


def test_potency_size_direction():
    """`massive` / `huge` push P positive; `tiny` / `mini` push P negative."""
    heavy = vibe_axes.derive_epa("A massive, heavy, dense model.")
    light = vibe_axes.derive_epa("A tiny, mini, compact model.")
    assert heavy is not None and light is not None
    assert heavy[1] > 0.3
    assert light[1] < -0.3


def test_activity_speed_direction():
    fast = vibe_axes.derive_epa("A fast, quick, instant, snappy runner.")
    slow = vibe_axes.derive_epa("A slow, sluggish, batch model.")
    assert fast is not None and slow is not None
    assert fast[2] > 0.5
    assert slow[2] < -0.3


def test_clipping_to_unit_range():
    """Extreme summaries still saturate at ±1 exactly, never overshoot."""
    epa = vibe_axes.derive_epa(
        "massive huge big powerful heavyweight heavy dense strong impressive"
    )
    assert epa is not None
    assert -1.0 <= epa[0] <= 1.0
    assert -1.0 <= epa[1] <= 1.0
    assert -1.0 <= epa[2] <= 1.0


def test_tokenizer_preserves_hyphens():
    """`long-context` and `state-of-the-art` are single lexicon entries.
    Splitting on `-` would silently zero their contribution."""
    epa = vibe_axes.derive_epa("state-of-the-art long-context")
    assert epa is not None
    # sota has E+0.7, long-context has P+0.5 → both should be > 0
    assert epa[0] > 0
    assert epa[1] > 0


def test_store_epa_round_trip(conn):
    db.insert_model(conn, "test/M", author="test")
    vibe_axes.store_epa(conn, "test/M", (0.5, -0.2, 0.7))
    conn.commit()
    loaded = vibe_axes.load_epa(conn, "test/M")
    assert loaded is not None
    e, p, a = loaded
    assert abs(e - 0.5) < 1e-3
    assert abs(p - -0.2) < 1e-3
    assert abs(a - 0.7) < 1e-3


def test_load_epa_partial_write_returns_none(conn):
    """All-or-nothing: two axes present, third missing → None. A caller
    that would silently accept `(e, p, None)` is running against a
    broken invariant, so `load_epa` refuses to hand it back."""
    db.insert_model(conn, "test/Partial", author="test")
    db.set_metadata(conn, "test/Partial", "vibe_e", "0.5", "float")
    db.set_metadata(conn, "test/Partial", "vibe_p", "0.3", "float")
    # vibe_a deliberately omitted
    conn.commit()
    assert vibe_axes.load_epa(conn, "test/Partial") is None


def test_derive_and_store_reads_vibe_summary(conn):
    """The composed read → derive → write chain — the entrypoint that
    a batch script calls once per model."""
    db.insert_model(conn, "test/Vibed", author="test")
    db.set_metadata(
        conn,
        "test/Vibed",
        "vibe_summary",
        "A fast, capable, lightweight model for realtime chat.",
        "str",
    )
    conn.commit()
    epa = vibe_axes.derive_and_store(conn, "test/Vibed")
    assert epa is not None
    conn.commit()
    loaded = vibe_axes.load_epa(conn, "test/Vibed")
    assert loaded == epa


def test_derive_and_store_returns_none_on_no_summary(conn):
    db.insert_model(conn, "test/Silent", author="test")
    conn.commit()
    assert vibe_axes.derive_and_store(conn, "test/Silent") is None
