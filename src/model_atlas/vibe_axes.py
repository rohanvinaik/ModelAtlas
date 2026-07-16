"""Vibe axes — Osgood EPA (Evaluation / Potency / Activity) derived from vibe_summary.

Sparse-wiki lesson: EPA is a low-dimensional projection of *what the entity
feels like*, orthogonal to the structural anchors. For ML models the same
question — "is this thing capable? heavy? fast?" — reads as three
independent axes that anchor banks don't cleanly express.

The primitives path sparse-wiki uses (NSM → EPA) needs a semantic parser
we don't have. Direct lexicon → EPA is the honest lift: a hand-curated
map of common ML-vibe vocabulary to EPA contributions, summed over
tokens in `vibe_summary`, clipped to [-1, +1].

Stored as three `model_metadata` rows per model: `vibe_e`, `vibe_p`,
`vibe_a`. Absent when `vibe_summary` is empty or contains no lexicon
hits — abstention, not zero.
"""

from __future__ import annotations

import re
import sqlite3

from . import db


# Hand-curated lexicon. Each entry: word (lowercased) → (E, P, A) contribution.
# Weights are additive per occurrence; the final vector is clipped to [-1, +1].
# Values around ±0.4 read as "one strong signal moves the axis a good chunk";
# a summary with 2-3 hits saturates the axis, which matches human intuition.
_LEXICON: dict[str, tuple[float, float, float]] = {
    # ── Evaluation (good/bad, capable/limited) ──
    "capable": (0.5, 0.0, 0.0),
    "reliable": (0.5, 0.0, 0.0),
    "solid": (0.4, 0.2, 0.0),
    "robust": (0.5, 0.3, 0.0),
    "polished": (0.4, 0.0, 0.0),
    "excellent": (0.7, 0.0, 0.0),
    "strong": (0.5, 0.4, 0.0),
    "impressive": (0.6, 0.2, 0.0),
    "state-of-the-art": (0.7, 0.3, 0.0),
    "sota": (0.7, 0.3, 0.0),
    "leading": (0.5, 0.3, 0.0),
    "frontier": (0.6, 0.5, 0.0),
    "flagship": (0.5, 0.4, 0.0),
    "premium": (0.4, 0.2, 0.0),
    "weak": (-0.4, -0.3, 0.0),
    "poor": (-0.5, 0.0, 0.0),
    "brittle": (-0.4, -0.2, 0.0),
    "buggy": (-0.5, 0.0, 0.0),
    "limited": (-0.3, -0.2, 0.0),
    "experimental": (-0.2, 0.0, 0.2),
    "unstable": (-0.4, 0.0, 0.1),
    "outdated": (-0.4, -0.2, -0.2),
    "deprecated": (-0.5, -0.2, -0.2),
    # ── Potency (large/small, heavy/light) ──
    "large": (0.0, 0.6, 0.0),
    "huge": (0.0, 0.8, 0.0),
    "massive": (0.0, 0.9, 0.0),
    "big": (0.0, 0.5, 0.0),
    "small": (0.0, -0.5, 0.0),
    "tiny": (0.0, -0.7, 0.0),
    "mini": (0.0, -0.6, 0.0),
    "compact": (0.1, -0.4, 0.2),
    "lightweight": (0.1, -0.5, 0.3),
    "efficient": (0.3, -0.2, 0.2),
    "heavy": (0.0, 0.6, -0.2),
    "heavyweight": (0.0, 0.7, -0.2),
    "powerful": (0.4, 0.6, 0.0),
    "dense": (0.0, 0.5, -0.1),
    "sparse": (0.0, -0.3, 0.1),
    "moe": (0.2, 0.4, 0.0),
    # ── Activity (fast/slow, dynamic/static) ──
    "fast": (0.2, 0.0, 0.6),
    "quick": (0.2, 0.0, 0.5),
    "instant": (0.2, 0.0, 0.7),
    "responsive": (0.3, 0.0, 0.5),
    "snappy": (0.2, 0.0, 0.6),
    "slow": (-0.2, 0.0, -0.6),
    "sluggish": (-0.3, 0.0, -0.6),
    "streaming": (0.1, 0.0, 0.4),
    "interactive": (0.2, 0.0, 0.5),
    "realtime": (0.2, 0.0, 0.7),
    "batch": (0.0, 0.1, -0.3),
    "offline": (0.0, 0.0, -0.4),
    # ── Capability shorthands (cross-axis) ──
    "reasoning": (0.4, 0.4, 0.0),
    "instruct": (0.3, 0.0, 0.2),
    "chat": (0.2, 0.0, 0.3),
    "code": (0.3, 0.0, 0.0),
    "multimodal": (0.3, 0.3, 0.0),
    "vision": (0.2, 0.2, 0.0),
    "long-context": (0.3, 0.5, 0.0),
    "quantized": (0.0, -0.3, 0.2),
    "distilled": (0.1, -0.4, 0.3),
    "finetuned": (0.2, 0.0, 0.0),
    "specialized": (0.3, 0.0, 0.0),
    "general": (0.2, 0.2, 0.0),
    "generalist": (0.2, 0.3, 0.0),
}


# Tokenizer: lowercase + split on non-alphanumeric (keeps hyphens because
# `long-context`, `state-of-the-art` are lexicon entries).
_TOKEN_RE = re.compile(r"[a-z0-9\-]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase + tokenize `text` on non-alnum boundaries, keeping hyphens.

    Preserving hyphens matters: `long-context`, `state-of-the-art` are single
    lexicon entries, not three tokens. Anything else (`.`, `,`, whitespace)
    is a boundary.
    """
    return _TOKEN_RE.findall(text.lower())


def derive_epa(text: str) -> tuple[float, float, float] | None:
    """Derive (E, P, A) from `text` by summing lexicon contributions and clipping.

    Returns None when `text` is empty OR contains zero lexicon hits — absent is
    honestly `None`, not a false-neutral `(0, 0, 0)` that a caller might treat as
    a real reading. A neutral reading with `hits > 0` returns `(0.0, 0.0, 0.0)`.

    Contributions from the same word firing multiple times accumulate (a summary
    that says "fast fast fast" IS more of an activity claim than one that says
    "fast" once); clip caps the axis so no runaway.
    """
    if not text or not text.strip():
        return None
    e = p = a = 0.0
    hits = 0
    for tok in _tokenize(text):
        if tok in _LEXICON:
            de, dp, da = _LEXICON[tok]
            e += de
            p += dp
            a += da
            hits += 1
    if hits == 0:
        return None
    return (
        max(-1.0, min(1.0, e)),
        max(-1.0, min(1.0, p)),
        max(-1.0, min(1.0, a)),
    )


def store_epa(
    conn: sqlite3.Connection, model_id: str, epa: tuple[float, float, float]
) -> None:
    """Write the three EPA rows to `model_metadata`. Idempotent (INSERT OR REPLACE)."""
    e, p, a = epa
    db.set_metadata(conn, model_id, "vibe_e", f"{e:.4f}", "float")
    db.set_metadata(conn, model_id, "vibe_p", f"{p:.4f}", "float")
    db.set_metadata(conn, model_id, "vibe_a", f"{a:.4f}", "float")


def derive_and_store(conn: sqlite3.Connection, model_id: str) -> tuple[float, float, float] | None:
    """Read `vibe_summary` for `model_id`, derive EPA, write it back. Returns the
    written vector, or None when there was nothing to derive (no vibe_summary or
    zero lexicon hits).

    Composes the read/derive/write chain that Phase 3 promises — one call per
    model that has a `vibe_summary`. Absent summary → absent EPA rows: never
    fabricate the axes.
    """
    row = conn.execute(
        "SELECT value FROM model_metadata WHERE model_id = ? AND key = 'vibe_summary'",
        (model_id,),
    ).fetchone()
    if not row or not row[0]:
        return None
    epa = derive_epa(row[0])
    if epa is None:
        return None
    store_epa(conn, model_id, epa)
    return epa


def load_epa(
    conn: sqlite3.Connection, model_id: str
) -> tuple[float, float, float] | None:
    """Read stored (E, P, A) for `model_id`, or None when any axis is absent.

    All-or-nothing: a partial write (two of three axes) reads as absent, not
    as `(e, p, None)` — a caller that would silently accept two out of three
    axes would be running against a broken invariant. `derive_and_store`
    writes all three atomically, so this only fires on corruption.
    """
    rows = conn.execute(
        """SELECT key, value FROM model_metadata
           WHERE model_id = ? AND key IN ('vibe_e', 'vibe_p', 'vibe_a')""",
        (model_id,),
    ).fetchall()
    d = {r[0]: r[1] for r in rows}
    if not all(k in d for k in ("vibe_e", "vibe_p", "vibe_a")):
        return None
    try:
        return float(d["vibe_e"]), float(d["vibe_p"]), float(d["vibe_a"])
    except (TypeError, ValueError):
        return None
