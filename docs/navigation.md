# Navigation: sign, depth, and the separate scalar

The shape ModelAtlas inherits from sparse-wiki, and where the engine had
drifted from it.

## The contract

A bank position is **`[SIGN][DEPTH]`**:

```
+4:SPATIAL/Earth/Europe/France/IdF/Paris     4 steps MORE SPECIFIC than Earth
-3:SPATIAL/Earth/Sol/Local_Bubble/Milky_Way  3 steps MORE ABSTRACT than Earth
 0:TEMPORAL/Present                          the zero state itself
```

**Sign** is balanced ternary — `+1` / `0` / `-1` — and says which way from the
zero state. **Depth** says how far. They are different questions and the
architecture answers them separately.

A query is a **set intersection over those constraints**:

```
query = {
    SPATIAL:  {sign: "+", contains: "Europe"},
    TEMPORAL: {sign: "-"},                     # past
    SCALE:    {sign: "+", min_depth: 1},       # national and above
}
→ entities_spatial ∩ entities_temporal ∩ entities_scale
```

Then, and only then, the admissible set is **ranked by a separate scalar** —
in sparse-wiki, `entity.pagerank` when no context narrows further.

That separation is the whole design. Navigation decides **who is admissible**.
The scalar decides **what comes first**. Scale is a magnitude, and magnitudes
multiply; direction is a ternary coordinate, and coordinates filter. Folding
one into the other produces a number that means neither.

## Where the engine had drifted

The depth half existed in the codebase and had simply never been wired to the
primary engine:

| | expresses depth? |
|---|---|
| `BankConstraint` (`min_signed`, `max_signed`, `target_position`) | yes |
| `db_queries.find_models_by_bank_range` | yes |
| `query.py`, the natural-language path | yes |
| **`StructuredQuery` → `navigate()` → `navigate_models`** | **no — sign only** |

So the primary recommendation engine kept the sign and dropped the depth.
`efficiency=+1` admitted a 13B model and a 400B one on identical terms, and
"the most capable reasoning model, compute is no concern" returned:

```
31B, 120B, 14B, 12B, 32B, 26B
```

Not disordered — *unfiltered*. The query had no way to say "at least three
steps positive."

### The wrong fix, and why it was tempting

The obvious repair is to make `_bank_score_single` reward depth: score `+4`
above `+1` so larger models sort higher. It would have moved the number, and
it is wrong. Depth would become a scoring term, scale would be folded back
into the ternary alignment factor, and the two questions the architecture
separates would be answered by one number again. The step function is not a
bug in the score — alignment is *binary because it is a filter*.

An eval assertion had already encoded the same mistake (`largest_ranks_first`,
"asking for large must sort by size descending"). It is deleted. Sort order is
not part of the contract; the shape of the admissible window is.

## Using it

```python
StructuredQuery(
    require_anchors=["reasoning"],
    efficiency=1,                      # which way
    min_depth={"EFFICIENCY": 2},       # how far
)
```

Measured on the v0.4.2 corpus, `require=["reasoning"], efficiency=+1`:

| `min_depth` | candidates | sizes returned |
|---|---|---|
| none | 7,608 | 31B, 120B, 14B, **12B**, 32B, 26B |
| `{"EFFICIENCY": 1}` | 3,310 | **12B**, 120B, 32B, 31B, 14B, 26B |
| `{"EFFICIENCY": 2}` | 2,443 | 26B, 120B, 32B, 30B, 32B, 31B |
| `{"EFFICIENCY": 3}` | 691 | 120B, 235B, 162B, 80B, 60B, 120B |

Symmetric downward — `efficiency=-1, min_depth={"EFFICIENCY": 2}` moves
0.6/2.0/0.3/**2.6**/0.5 to 0.6/0.3/0.5/0.5/0.5.

Note the ordering inside each row is *not* size-descending, and that is
correct: depth selected the shape, the scalar ordered it.

Rough EFFICIENCY guide: depth 1 ≈ one class out (3B / 13B), 2 ≈ two
(1B / 30B), 3+ ≈ the far end (sub-1B / 70B+).

### Where it does not apply

- **Direction `0`** — the zero state *is* depth 0, so "at least N steps from
  it" is meaningless. The constraint is skipped, not treated as unsatisfiable.
- **No direction set** — a depth with no sign has no direction to travel.
  Skipped.
- **An unsatisfiable depth returns nothing**, deliberately. Falling back to the
  unfiltered corpus would answer a question nobody asked.

## The scalar half, still open

Ranking is `soft_combined`, whose strongest term is PageRank — now
rank-percentile rather than max-normalised (see
[`scoring-dynamic-range.md`](scoring-dynamic-range.md)). Two things remain
unresolved there and are *scalar* questions, not navigation ones:

- **PageRank is not popularity.** Log-log *r* = 0.259 over the 24,434 models
  carrying both. It measures ancestry — the top of it is FLUX.1-dev, SDXL,
  Qwen2.5-7B, the bases people fine-tune *from*. `all-MiniLM-L6-v2` has 249M
  downloads and ranks 333rd.
- **SCALE is its own dimension in sparse-wiki** (zero `Regional`, `+` Global,
  `−` Local), derived from PageRank plus significance. ModelAtlas has no
  SCALE bank; EFFICIENCY carries both "how big is the artifact" and "how
  significant is it", which are not the same axis.
