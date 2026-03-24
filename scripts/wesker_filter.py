"""Monty Hall filtering — exclude irrelevant mutation categories (§6.1).

Layer 1 (exclusionary): If a function has no comparisons, boundary mutants
cannot survive (there's nothing to mutate), so generating them wastes budget.
The filter reveals which "doors" have no prize before opening them.

Layer 2 (predictive priors): When cached mutation data exists, use historical
per-category survival rates to prioritize categories most likely to have
surviving mutants, directing budget where it matters most.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from wesker_engine import MutationCategory


@dataclass
class CategoryPrior:
    """A mutation category with its expected survival probability."""

    category: MutationCategory
    prior: float  # 0.0 = never survives, 1.0 = always survives


@dataclass
class _FunctionSignals:
    """Structural signals extracted from a function AST for mutation filtering."""

    param_count: int = 0
    has_comparisons: bool = False
    has_self_assigns: bool = False
    has_global_nonlocal: bool = False
    has_isinstance: bool = False


def _collect_signals(func_node: ast.FunctionDef) -> _FunctionSignals:
    """Walk the function AST to collect structural signals."""
    signals = _FunctionSignals(param_count=len(func_node.args.args))
    for node in ast.walk(func_node):
        if isinstance(node, ast.Compare):
            signals.has_comparisons = True
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                signals.has_self_assigns = True
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            signals.has_global_nonlocal = True
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
        ):
            signals.has_isinstance = True
    return signals


def filter_categories(
    func_node: ast.FunctionDef,
    is_pure: bool = False,
) -> set[MutationCategory]:
    """Layer 1: Exclusionary filtering (§6.1).

    Returns the set of categories relevant to this function.
    Categories where the function has no structural support are excluded.
    """
    sig = _collect_signals(func_node)
    relevant: set[MutationCategory] = {MutationCategory.VALUE}

    if sig.param_count >= 2:
        relevant.add(MutationCategory.SWAP)
    if sig.has_comparisons:
        relevant.add(MutationCategory.BOUNDARY)
    if not is_pure and (sig.has_self_assigns or sig.has_global_nonlocal):
        relevant.add(MutationCategory.STATE)
    if sig.has_isinstance:
        relevant.add(MutationCategory.TYPE)

    return relevant


# ── Layer 2: Predictive priors (§6.2) ────────────────────────────────


_DEFAULT_PRIOR = 0.5  # uniform when no history


def prioritize_categories(
    relevant: set[MutationCategory],
    cached_state: dict | None = None,
) -> list[CategoryPrior]:
    """Layer 2: Predictive priors from cached mutation data.

    Takes the Layer 1 exclusionary output and annotates each category
    with a survival prior derived from previous profiling runs. Returns
    categories ordered by descending prior (highest-survival first),
    so budget-limited runs test the most informative categories first.

    When no cached data exists, all priors are uniform (0.5).

    Note: ``per_category`` in cached mutation state is a *list* of dicts
    (``[{"category": "VALUE", "total": 10, "survived": 3}, ...]``),
    not a dict keyed by category name.
    """
    # Build lookup from the list format used by mutation engine output
    cat_lookup: dict[str, dict] = {}
    if cached_state:
        raw = cached_state.get("per_category", [])
        if isinstance(raw, list):
            for entry in raw:
                cat_name = entry.get("category", "")
                if cat_name:
                    cat_lookup[cat_name] = entry
        elif isinstance(raw, dict):
            cat_lookup = raw  # defensive: handle dict format too

    priors: list[CategoryPrior] = []
    for cat in relevant:
        cat_data = cat_lookup.get(cat.value, {})
        if cat_data:
            total = cat_data.get("total", 0)
            survived = cat_data.get("survived", 0)
            prior = survived / total if total > 0 else _DEFAULT_PRIOR
        else:
            prior = _DEFAULT_PRIOR
        priors.append(CategoryPrior(category=cat, prior=round(prior, 3)))

    # Sort by prior descending — highest survival first for budget efficiency
    priors.sort(key=lambda p: p.prior, reverse=True)
    return priors
