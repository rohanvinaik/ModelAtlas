---
paths:
  - "**/*.py"
---

# Theory Rules

This file stores extracted theory signals for `model-atlas`.

## Facet Summaries
- Core Theory: No strong signal extracted for this facet yet.
- Problem-Solving: A structured semantic network of ML models, exposed as an MCP tool, so the LLM you're already talking to can see the model landscape instead of guessing at it.
- Alignment: Why signed hierarchies instead of flat categories: A categorical "size" field with values {small, medium, large} can't express proximity.
- Architecture: Four indexed queries instead of 18K individual get_model() calls.
- Anti-Patterns: No strong signal extracted for this facet yet.
- Key Abstractions: No strong signal extracted for this facet yet.

## High-Signal Anti-Patterns
- Do not try a 4th approach without first enumerating all known constraints and verifying which ones the new approach actually addresses.
- Do not discover constraints one-at-a-time through failure — enumerate the full constraint space upfront by reading before acting.
- Do not re-attempt an approach that already failed unless the conditions that caused the failure have changed.
- Do not use O(n²) algorithms when O(n) alternatives exist — quadratic membership checks on lists, re.compile inside loops, and sorted()[0] instead of min() are structural mistakes, not style issues.
- Do not treat N instances of the same root cause as N separate problems — cluster issues by shared fix before diving into individual repairs.

## Enforceable Rules
- No enforceable rules extracted yet.

## Extraction Quality
- Validity status: weak
- Docs scanned: 7
- Total claims: 18
- Missing required facets: core_theory
- Warning: Missing required facets: core_theory
- Warning: No enforceable rules found (existing or proposed).
