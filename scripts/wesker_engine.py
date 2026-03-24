"""AST mutation engine — in-process mutant generation and evaluation.

Implements §6.4 dispatch table: category→AST-transform mapping.
Generates mutants by AST rewriting (no subprocess spawning), evaluates
them by running targeted tests in the same process against a sandboxed
namespace. Respects per-function time budgets.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import time
import types
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class MutationCategory(str, Enum):
    """Semantic mutation category (§6.4 dispatch table)."""

    VALUE = "VALUE"
    SWAP = "SWAP"
    STATE = "STATE"
    BOUNDARY = "BOUNDARY"
    TYPE = "TYPE"


@dataclass
class Mutant:
    """A single AST-level mutation."""

    category: MutationCategory
    original_node: ast.AST
    mutated_node: ast.AST
    description: str
    location: int = 0
    mutant_id: str = ""


@dataclass
class MutantResult:
    """Result of evaluating a single mutant against tests."""

    mutant: Mutant
    killed: bool = False
    killed_by: str | None = None  # "assertion" | "crash" | "timeout"
    test_name: str | None = None
    elapsed_ms: float = 0.0


@dataclass
class CategoryResult:
    """Aggregated results for one mutation category."""

    category: MutationCategory
    total: int = 0
    killed: int = 0
    survived: int = 0
    killed_by_assertion: int = 0
    killed_by_crash: int = 0
    timed_out: int = 0

    @property
    def survival_rate(self) -> float:
        return self.survived / self.total if self.total > 0 else 0.0


@dataclass
class SamplingResult:
    """Result of inline mutation sampling for a function."""

    function_key: str = ""
    categories_tested: int = 0
    total_mutants: int = 0
    total_killed: int = 0
    total_survived: int = 0
    survival_rate: float = 0.0
    coverage_depth: str = "sampled"
    per_category: list[CategoryResult] = field(default_factory=list)
    budget_exhausted: bool = False
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "function_key": self.function_key,
            "categories_tested": self.categories_tested,
            "total_mutants": self.total_mutants,
            "total_killed": self.total_killed,
            "total_survived": self.total_survived,
            "survival_rate": round(self.survival_rate, 3),
            "coverage_depth": self.coverage_depth,
            "budget_exhausted": self.budget_exhausted,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "per_category": [
                {
                    "category": cr.category.value,
                    "total": cr.total,
                    "killed": cr.killed,
                    "survived": cr.survived,
                    "survival_rate": round(cr.survival_rate, 3),
                }
                for cr in self.per_category
            ],
        }


@dataclass
class ProfilingResult:
    """Result of exhaustive mutation profiling for a function."""

    function_key: str = ""
    categories_tested: int = 0
    total_mutants: int = 0
    total_killed: int = 0
    total_survived: int = 0
    survival_rate: float = 0.0
    coverage_depth: str = "profiled"
    is_gateable: bool = True
    per_category: list[CategoryResult] = field(default_factory=list)
    kill_matrix: dict[str, list[str]] = field(default_factory=dict)
    survivor_records: list[dict] = field(default_factory=list)
    killed_records: list[dict] = field(default_factory=list)
    budget_exhausted: bool = False
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "function_key": self.function_key,
            "categories_tested": self.categories_tested,
            "total_mutants": self.total_mutants,
            "total_killed": self.total_killed,
            "total_survived": self.total_survived,
            "survival_rate": round(self.survival_rate, 3),
            "coverage_depth": self.coverage_depth,
            "is_gateable": self.is_gateable,
            "budget_exhausted": self.budget_exhausted,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "per_category": [
                {
                    "category": cr.category.value,
                    "total": cr.total,
                    "killed": cr.killed,
                    "survived": cr.survived,
                    "killed_by_assertion": cr.killed_by_assertion,
                    "killed_by_crash": cr.killed_by_crash,
                    "survival_rate": round(cr.survival_rate, 3),
                }
                for cr in self.per_category
            ],
        }
        if self.kill_matrix:
            d["kill_matrix"] = self.kill_matrix
        if self.survivor_records:
            d["survivor_records"] = self.survivor_records
        if self.killed_records:
            d["killed_records"] = self.killed_records
        return d


# ── §6.4 Dispatch Table: Category → AST Transform ────────────────


class _BaseMutator(ast.NodeTransformer):
    """Base class for all category mutators — tracks ``applied`` state."""

    def __init__(self, target_index: int = 0):
        self.current = 0
        self.target = target_index
        self.applied = False


class _ValueMutator(_BaseMutator):
    """Replace constants with boundary values."""

    # Types we can actually mutate — others (None, bytes, complex, Ellipsis)
    # are left unchanged by _mutate_constant, so we must not count them as
    # targets or mark ``applied`` when we encounter them.
    _MUTABLE_TYPES = (bool, int, float, str)

    def __init__(
        self,
        target_index: int = 0,
        docstring_positions: set[tuple[int, int]] | None = None,
    ):
        super().__init__(target_index)
        self._ds_pos = docstring_positions or set()

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if self.applied:
            return node
        if not isinstance(node.value, self._MUTABLE_TYPES):
            return node
        # Skip docstring constants — they produce equivalent mutants.
        if (
            self._ds_pos
            and isinstance(node.value, str)
            and (node.lineno, node.col_offset) in self._ds_pos
        ):
            return node
        if self.current == self.target:
            mutated = self._mutate_constant(node)
            if mutated is not node:
                self.applied = True
                return mutated
            # Defensive: if _mutate_constant somehow returned the original,
            # do not mark applied — skip this target.
            return node
        self.current += 1
        return node

    @staticmethod
    def _mutate_constant(node: ast.Constant) -> ast.Constant:
        v = node.value
        if isinstance(v, bool):
            return ast.Constant(value=not v)
        if isinstance(v, int):
            return ast.Constant(value=0 if v != 0 else 1)
        if isinstance(v, float):
            return ast.Constant(value=0.0 if v else 1.0)
        if isinstance(v, str):
            return ast.Constant(value="" if v else "mutated")
        return node


class _BoundaryMutator(_BaseMutator):
    """Off-by-one on comparisons: < → <=, >= → >, etc."""

    _SWAP = {
        ast.Lt: ast.LtE,
        ast.LtE: ast.Lt,
        ast.Gt: ast.GtE,
        ast.GtE: ast.Gt,
    }

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        if self.applied:
            return self.generic_visit(node)
        new_ops = []
        for op in node.ops:
            if not self.applied and self.current == self.target:
                swapped = self._SWAP.get(type(op))
                if swapped:
                    new_ops.append(swapped())
                    self.applied = True
                else:
                    new_ops.append(op)
            else:
                new_ops.append(op)
            self.current += 1
        node.ops = new_ops
        return self.generic_visit(node)


class _SwapMutator(_BaseMutator):
    """Transpose two parameters in a function call."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if self.applied or len(node.args) < 2:
            return self.generic_visit(node)
        if self.current == self.target:
            self.applied = True
            node.args = list(node.args)
            node.args[0], node.args[1] = node.args[1], node.args[0]
        self.current += 1
        return self.generic_visit(node)


class _StateMutator(_BaseMutator):
    """Remove self.x = ... assignments or replace return with return None."""

    def __init__(self, target_index: int = 0, mode: str = "remove_assign"):
        super().__init__(target_index)
        self.mode = mode

    def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
        if self.applied or self.mode != "remove_assign":
            return node
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                if self.current == self.target:
                    self.applied = True
                    return ast.Pass()
                self.current += 1
        return node

    def visit_Return(self, node: ast.Return) -> ast.AST:
        if self.applied or self.mode != "return_none":
            return node
        if node.value is not None:
            if self.current == self.target:
                self.applied = True
                return ast.Return(value=ast.Constant(value=None))
            self.current += 1
        return node


class _TypeMutator(_BaseMutator):
    """Replace isinstance(x, T) with True."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if self.applied:
            return self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "isinstance":
            if self.current == self.target:
                self.applied = True
                return ast.Constant(value=True)
            self.current += 1
        return self.generic_visit(node)


# ── Mutant Generation ─────────────────────────────────────────────


def _docstring_positions(func_node: ast.FunctionDef) -> set[tuple[int, int]]:
    """Return (lineno, col_offset) of docstring Constant nodes in a function.

    A docstring is the first statement if it's ``Expr(Constant(str))``.
    We collect positions so that both counting and mutation can skip them
    using position-based identity (survives ``copy.deepcopy``).
    """
    positions: set[tuple[int, int]] = set()
    if (
        func_node.body
        and isinstance(func_node.body[0], ast.Expr)
        and isinstance(func_node.body[0].value, ast.Constant)
        and isinstance(func_node.body[0].value.value, str)
    ):
        ds = func_node.body[0].value
        positions.add((ds.lineno, ds.col_offset))
    return positions


def _count_targets(func_node: ast.FunctionDef, category: MutationCategory) -> int:
    """Count how many mutation targets exist for a category in a function."""
    counter = _TARGET_COUNTERS.get(category)
    if counter is None:
        return 0
    # VALUE needs docstring exclusion — pass positions through.
    if category == MutationCategory.VALUE:
        ds_pos = _docstring_positions(func_node)
        return sum(_count_value_target(node, ds_pos) for node in ast.walk(func_node))
    return sum(counter(node) for node in ast.walk(func_node))


def _count_value_target(
    node: ast.AST,
    docstring_positions: set[tuple[int, int]] | None = None,
) -> int:
    # Only count constants whose types _ValueMutator can actually mutate.
    # None, bytes, complex, and Ellipsis are left unchanged by _mutate_constant,
    # so counting them produces phantom mutants that always survive.
    # Skip docstring constants — they produce equivalent mutants that waste budget.
    if isinstance(node, ast.Constant) and isinstance(node.value, _ValueMutator._MUTABLE_TYPES):
        if (
            docstring_positions
            and isinstance(node.value, str)
            and (node.lineno, node.col_offset) in docstring_positions
        ):
            return 0
        return 1
    return 0


def _count_boundary_target(node: ast.AST) -> int:
    if not isinstance(node, ast.Compare):
        return 0
    return sum(1 for op in node.ops if type(op) in _BoundaryMutator._SWAP)


def _count_swap_target(node: ast.AST) -> int:
    return 1 if isinstance(node, ast.Call) and len(node.args) >= 2 else 0


def _is_self_assign(target: ast.AST) -> bool:
    return (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    )


def _count_state_assign_target(node: ast.AST) -> int:
    """Count self.x = ... assignments (remove_assign mode)."""
    if isinstance(node, ast.Assign):
        return sum(1 for t in node.targets if _is_self_assign(t))
    return 0


def _count_state_return_target(node: ast.AST) -> int:
    """Count return-with-value nodes (return_none mode)."""
    if isinstance(node, ast.Return) and node.value is not None:
        return 1
    return 0


def _count_state_target(node: ast.AST) -> int:
    return _count_state_assign_target(node) + _count_state_return_target(node)


def _count_type_target(node: ast.AST) -> int:
    return (
        1
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "isinstance"
        else 0
    )


_TARGET_COUNTERS: dict[MutationCategory, Callable[[ast.AST], int]] = {
    MutationCategory.VALUE: _count_value_target,
    MutationCategory.BOUNDARY: _count_boundary_target,
    MutationCategory.SWAP: _count_swap_target,
    MutationCategory.STATE: _count_state_target,
    MutationCategory.TYPE: _count_type_target,
}


def _generate_state_mutants(
    func_node: ast.FunctionDef,
    max_per_category: int,
) -> list[Mutant]:
    """Generate STATE mutants across both sub-modes (assign + return).

    STATE has two independent sub-modes with separate target indices:
    - remove_assign: replaces ``self.x = expr`` with ``pass``
    - return_none: replaces ``return expr`` with ``return None``

    Each sub-mode gets its own target count and transformer pass so that
    target indices align correctly with what the transformer visits.
    """
    mutants: list[Mutant] = []
    cat = MutationCategory.STATE

    sub_modes = [
        ("remove_assign", "remove state assignment", _count_state_assign_target),
        ("return_none", "replace return with None", _count_state_return_target),
    ]

    for mode, desc, counter in sub_modes:
        target_count = sum(counter(node) for node in ast.walk(func_node))
        limit = min(target_count, max_per_category) if max_per_category > 0 else target_count

        for i in range(limit):
            mutated_tree = copy.deepcopy(func_node)
            transformer = _StateMutator(i, mode)
            mutated_node = transformer.visit(mutated_tree)
            ast.fix_missing_locations(mutated_node)

            if transformer.applied:
                mid = f"{cat.value}_{mode}_{i}"
                mutants.append(
                    Mutant(
                        category=cat,
                        original_node=func_node,
                        mutated_node=mutated_node,
                        description=f"{mid}: {desc}",
                        location=getattr(func_node, "lineno", 0),
                        mutant_id=mid,
                    )
                )

    return mutants


def generate_mutants(
    func_node: ast.FunctionDef,
    categories: set[MutationCategory],
    max_per_category: int = 0,
    seed: int | None = None,
) -> list[Mutant]:
    """Generate mutants for a function across specified categories.

    Args:
        func_node: The function AST node to mutate.
        categories: Set of mutation categories to generate.
        max_per_category: Max mutants per category (0 = unlimited).
        seed: Deterministic seed for shuffling target indices. When set and
              ``max_per_category > 0``, the target indices are shuffled before
              truncation so different seeds sample different mutants from the
              same function. ``None`` (default) preserves AST-walk order.
    """
    mutants: list[Mutant] = []
    ds_pos = _docstring_positions(func_node)

    for cat in sorted(categories, key=lambda c: c.value):
        # STATE needs special handling: two independent sub-modes with
        # separate target counts so indices align with the transformer.
        if cat == MutationCategory.STATE:
            mutants.extend(_generate_state_mutants(func_node, max_per_category))
            continue

        target_count = _count_targets(func_node, cat)
        indices = list(range(target_count))

        # Stable shuffle: deterministic per seed, varies across iterations.
        if seed is not None and max_per_category > 0 and target_count > max_per_category:
            indices = _stable_target_order(indices, seed=seed, category=cat.value)

        limit = min(target_count, max_per_category) if max_per_category > 0 else target_count
        selected = indices[:limit]

        for i in selected:
            mutated_tree = copy.deepcopy(func_node)
            transformer, desc = _make_transformer(cat, i, ds_pos)
            mutated_node = transformer.visit(mutated_tree)
            ast.fix_missing_locations(mutated_node)

            if transformer.applied:
                mid = f"{cat.value}_{i}"
                mutants.append(
                    Mutant(
                        category=cat,
                        original_node=func_node,
                        mutated_node=mutated_node,
                        description=f"{mid}: {desc}",
                        location=getattr(func_node, "lineno", 0),
                        mutant_id=mid,
                    )
                )

    return mutants


def _stable_target_order(indices: list[int], *, seed: int, category: str) -> list[int]:
    """Return a deterministic pseudo-shuffled order for target indices."""
    return sorted(indices, key=lambda idx: _stable_target_key(seed, category, idx))


def _stable_target_key(seed: int, category: str, idx: int) -> bytes:
    """Build a stable hash key for deterministic mutant sampling order."""
    payload = f"{seed}:{category}:{idx}".encode()
    return hashlib.sha256(payload).digest()


def _make_transformer(
    category: MutationCategory,
    index: int,
    docstring_positions: set[tuple[int, int]] | None = None,
) -> tuple[_BaseMutator, str]:
    """Create the appropriate transformer for a category + target index."""
    if category == MutationCategory.VALUE:
        return _ValueMutator(index, docstring_positions), "replace constant with boundary value"
    if category == MutationCategory.BOUNDARY:
        return _BoundaryMutator(index), "off-by-one comparison"
    if category == MutationCategory.SWAP:
        return _SwapMutator(index), "transpose call arguments"
    if category == MutationCategory.STATE:
        return _StateMutator(index, "remove_assign"), "remove state assignment"
    if category == MutationCategory.TYPE:
        return _TypeMutator(index), "replace isinstance with True"
    msg = f"Unknown category: {category}"
    raise ValueError(msg)


@dataclass
class BoundaryInput:
    """A synthesized boundary test input from a Compare mutation."""

    parameter: str
    boundary_value: int | float
    inputs: list[tuple[str, int | float]]  # [(param, value), ...]


def extract_boundary_inputs(mutant: Mutant) -> list[BoundaryInput]:
    """Extract boundary test inputs from a BOUNDARY mutant.

    Walks the original Compare node to find the parameter name and constant
    involved, then synthesizes inputs at boundary, boundary-1, boundary+1.
    Only works for Compare nodes comparing a Name to a numeric Constant.
    """
    if mutant.category != MutationCategory.BOUNDARY:
        return []

    results: list[BoundaryInput] = []
    orig_compares = [n for n in ast.walk(mutant.original_node) if isinstance(n, ast.Compare)]
    mut_compares = [n for n in ast.walk(mutant.mutated_node) if isinstance(n, ast.Compare)]

    for orig_cmp, mut_cmp in zip(orig_compares, mut_compares, strict=False):
        # Find the op that changed
        for orig_op, mut_op in zip(orig_cmp.ops, mut_cmp.ops, strict=False):
            if type(orig_op) is type(mut_op):
                continue
            # Found the mutated comparison — extract param + constant
            param, const = _extract_compare_parts(orig_cmp)
            if param and const is not None and isinstance(const, (int, float)):
                offsets = [0, -1, 1]
                inputs = [(param, const + off) for off in offsets]
                results.append(
                    BoundaryInput(
                        parameter=param,
                        boundary_value=const,
                        inputs=inputs,
                    )
                )
    return results


def _extract_compare_parts(
    cmp_node: ast.Compare,
) -> tuple[str | None, int | float | None]:
    """Extract (parameter_name, constant_value) from a Compare node.

    Handles both ``x < 10`` and ``10 < x`` orientations.
    """
    left = cmp_node.left
    comparators = cmp_node.comparators

    if isinstance(left, ast.Name) and len(comparators) == 1:
        comp = comparators[0]
        if isinstance(comp, ast.Constant) and isinstance(comp.value, (int, float)):
            return left.id, comp.value
    if (
        isinstance(left, ast.Constant)
        and isinstance(left.value, (int, float))
        and len(comparators) == 1
        and isinstance(comparators[0], ast.Name)
    ):
        return comparators[0].id, left.value
    return None, None


# ── Mutant Evaluation ─────────────────────────────────────────────


def _patch_mutant_into_test(
    test_fn: Callable[..., None],
    qualname: str | None,
    mutated_obj: Any,
) -> tuple[bool, Any, Any]:
    """Patch mutated function into the test's namespace.

    Tries __globals__ first (works for dynamically imported modules),
    then falls back to inspect.getmodule.

    Returns (patched, saved_original, patch_target) where patch_target
    is either a dict (__globals__) or a module object.
    """
    if not qualname:
        return False, None, None

    func_name = qualname.split(".")[-1]

    # Primary: use __globals__ — the test function's defining module globals.
    # Works for bound methods, regular functions, and dynamically imported modules.
    test_globals = getattr(test_fn, "__globals__", None)
    # For bound methods, __globals__ is on the underlying function
    if test_globals is None:
        underlying = getattr(test_fn, "__func__", None)
        if underlying is not None:
            test_globals = getattr(underlying, "__globals__", None)

    closure_bindings = _get_closure_bindings(test_fn)

    import inspect

    test_module = inspect.getmodule(test_fn)

    owner = _resolve_qualified_owner(test_globals, closure_bindings, test_module, qualname)
    if owner is not None and hasattr(owner, func_name):
        saved = _get_raw_attr(owner, func_name)
        setattr(owner, func_name, _preserve_descriptor_shape(saved, mutated_obj))
        return True, saved, owner

    closure_cell = _find_closure_cell(closure_bindings, func_name)
    if closure_cell is not None:
        saved = closure_cell.cell_contents
        closure_cell.cell_contents = _preserve_closure_binding_shape(saved, mutated_obj)
        return True, saved, ("closure_cell", closure_cell)

    if test_globals is not None and func_name in test_globals:
        saved = test_globals[func_name]
        test_globals[func_name] = _preserve_closure_binding_shape(saved, mutated_obj)
        return True, saved, test_globals

    # Fallback: inspect.getmodule (works for regular module-level functions)
    if test_module is not None and hasattr(test_module, func_name):
        saved = getattr(test_module, func_name)
        setattr(test_module, func_name, _preserve_closure_binding_shape(saved, mutated_obj))
        return True, saved, test_module

    return False, None, None


def _resolve_qualified_owner(
    test_globals: dict[str, Any] | None,
    closure_bindings: list[tuple[str, Any, Any]],
    test_module: Any,
    qualname: str,
) -> Any:
    """Resolve the owning object for a qualified symbol like ``Class.method``."""
    if "." not in qualname:
        return None

    import inspect

    owner_parts = qualname.split(".")[:-1]
    root_name = owner_parts[0]
    candidates: list[Any] = []
    seen: set[int] = set()

    def _add_candidate(obj: Any) -> None:
        if obj is None:
            return
        marker = id(obj)
        if marker in seen:
            return
        seen.add(marker)
        candidates.append(obj)

    def _add_from_value(value: Any) -> None:
        if value is None:
            return
        if inspect.ismodule(value) and hasattr(value, root_name):
            _add_candidate(getattr(value, root_name))
            return
        if isinstance(value, type):
            if value.__name__ == root_name:
                _add_candidate(value)
            return
        bound_self = getattr(value, "__self__", None)
        if bound_self is not None:
            owner = bound_self if isinstance(bound_self, type) else type(bound_self)
            if getattr(owner, "__name__", "") == root_name:
                _add_candidate(owner)
            return
        owner_type = type(value)
        if getattr(owner_type, "__name__", "") == root_name:
            _add_candidate(owner_type)

    for _, value, _ in closure_bindings:
        _add_from_value(value)

    if test_globals is not None:
        _add_candidate(test_globals.get(root_name))
        for value in test_globals.values():
            _add_from_value(value)

    if test_module is not None and hasattr(test_module, root_name):
        _add_candidate(getattr(test_module, root_name))

    for candidate in candidates:
        current = candidate
        for part in owner_parts[1:]:
            if not hasattr(current, part):
                current = None
                break
            current = getattr(current, part)
        if current is not None:
            return current
    return None


def _get_closure_bindings(test_fn: Callable[..., None]) -> list[tuple[str, Any, Any]]:
    """Extract ``(freevar_name, value, cell)`` bindings from a test closure."""
    underlying = getattr(test_fn, "__func__", test_fn)
    cells = getattr(underlying, "__closure__", None) or ()
    code = getattr(underlying, "__code__", None)
    freevars = getattr(code, "co_freevars", ())

    bindings: list[tuple[str, Any, Any]] = []
    for name, cell in zip(freevars, cells, strict=False):
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        bindings.append((name, value, cell))
    return bindings


def _find_closure_cell(
    closure_bindings: list[tuple[str, Any, Any]],
    func_name: str,
) -> Any:
    """Find the closure cell that directly binds a symbol name."""
    for name, _, cell in closure_bindings:
        if name == func_name:
            return cell
    return None


def _get_raw_attr(owner: Any, attr_name: str) -> Any:
    """Get the raw stored attribute to preserve descriptor identity."""
    from collections.abc import Mapping

    namespace = getattr(owner, "__dict__", None)
    if isinstance(namespace, Mapping) and attr_name in namespace:
        return namespace[attr_name]
    return getattr(owner, attr_name)


def _unwrap_descriptor(obj: Any) -> Any:
    """Extract the underlying callable from classmethod/staticmethod wrappers."""
    if isinstance(obj, (classmethod, staticmethod)):
        return obj.__func__
    return obj


def _preserve_descriptor_shape(original: Any, mutated_obj: Any) -> Any:
    """Wrap the mutant to match the original descriptor semantics."""
    if isinstance(original, classmethod):
        if isinstance(mutated_obj, classmethod):
            return mutated_obj
        return classmethod(_unwrap_descriptor(mutated_obj))
    if isinstance(original, staticmethod):
        if isinstance(mutated_obj, staticmethod):
            return mutated_obj
        return staticmethod(_unwrap_descriptor(mutated_obj))
    return _unwrap_descriptor(mutated_obj)


def _preserve_closure_binding_shape(original: Any, mutated_obj: Any) -> Any:
    """Wrap the mutant to match common closure-bound callable shapes."""
    if isinstance(original, types.MethodType):
        return types.MethodType(_unwrap_descriptor(mutated_obj), original.__self__)
    return _unwrap_descriptor(mutated_obj)


def _unpatch_mutant(
    patched: bool,
    saved: Any,
    patch_target: Any,
    func_name: str | None,
) -> None:
    """Restore the original function after mutation evaluation."""
    if not patched or saved is None or func_name is None:
        return
    if isinstance(patch_target, dict):
        patch_target[func_name] = saved
    elif (
        isinstance(patch_target, tuple)
        and len(patch_target) == 2
        and patch_target[0] == "closure_cell"
    ):
        patch_target[1].cell_contents = saved
    else:
        setattr(patch_target, func_name, saved)


def evaluate_mutant(
    mutant: Mutant,
    test_functions: list[Callable[..., None]],
    original_func: Callable[..., Any],  # noqa: ARG001 — kept for API compat
    timeout_ms: float = 5000,
    qualname: str | None = None,
) -> MutantResult:
    """Evaluate a mutant against test functions.

    Compiles the mutated function, then monkey-patches it into each test's
    module namespace before invoking the test with zero args (standard pytest
    contract). The original function is restored after each test.
    """
    start = time.monotonic()

    # Compile mutated function
    try:
        module_ast = ast.Module(body=[mutant.mutated_node], type_ignores=[])  # type: ignore[list-item]
        ast.fix_missing_locations(module_ast)
        code = compile(module_ast, "<mutant>", "exec")
        namespace: dict[str, Any] = {}
        exec(code, namespace)  # noqa: S102  # nosec B102 — intentional: compiling AST mutants
        func_name = getattr(mutant.mutated_node, "name", None)
        mutated_obj = namespace.get(func_name) if func_name else None
        if mutated_obj is None:
            return MutantResult(
                mutant=mutant,
                killed=True,
                killed_by="crash",
                elapsed_ms=_elapsed(start),
            )
    except Exception:
        return MutantResult(
            mutant=mutant,
            killed=True,
            killed_by="crash",
            elapsed_ms=_elapsed(start),
        )

    # Run tests against mutated function
    for test_fn in test_functions:
        remaining_ms = timeout_ms - _elapsed(start)
        if remaining_ms <= 0:
            return MutantResult(
                mutant=mutant, killed=True, killed_by="timeout", elapsed_ms=_elapsed(start)
            )
        # Strategy: monkey-patch the mutated function into the test's namespace
        # so the test calls the mutant instead of the original. Uses __globals__
        # (the test function's defining module globals) which works reliably for
        # both regular imports and dynamically loaded test modules. Falls back to
        # inspect.getmodule for inline test callables without __globals__.
        patch_name = qualname or func_name
        patched, saved, patch_target = _patch_mutant_into_test(test_fn, patch_name, mutated_obj)
        try:
            result = _run_test_with_timeout(
                test_fn,
                _unwrap_descriptor(mutated_obj),
                patched,
                remaining_ms,
            )
            if result is not None:
                return MutantResult(
                    mutant=mutant,
                    killed=True,
                    killed_by=result,
                    test_name=getattr(test_fn, "__name__", "unknown"),
                    elapsed_ms=_elapsed(start),
                )
        finally:
            _unpatch_mutant(patched, saved, patch_target, func_name)

    return MutantResult(mutant=mutant, killed=False, elapsed_ms=_elapsed(start))


def _run_test_with_timeout(
    test_fn: Callable[..., None],
    mutated_func: Any,
    patched: bool,
    timeout_ms: float,
) -> str | None:
    """Run a single test function with a hard thread-based timeout.

    Returns the kill reason ("assertion", "crash", "timeout") if killed,
    or None if the test passed (mutant survived this test).
    """
    import threading

    result_box: list[str | None] = [None]  # None = survived

    def _target() -> None:
        try:
            if patched:
                test_fn()
            else:
                try:
                    test_fn(mutated_func)
                except TypeError:
                    test_fn()
        except AssertionError:
            result_box[0] = "assertion"
        except Exception:
            result_box[0] = "crash"
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:  # pragma: no cover — pytest.outcomes.Failed etc.
            result_box[0] = "crash"

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_ms / 1000.0)

    if thread.is_alive():
        # Thread is stuck — treat as timeout kill.
        # Daemon thread will be cleaned up on process exit.
        return "timeout"

    return result_box[0]


def _elapsed(start: float) -> float:
    return (time.monotonic() - start) * 1000


# ── Sampling & Profiling ──────────────────────────────────────────


def run_function_sampling(
    func_node: ast.FunctionDef,
    func_key: str,
    categories: set[MutationCategory],
    test_functions: list[Callable[..., None]],
    original_func: Callable[..., Any],
    budget_ms: float = 500,
    max_per_category: int = 3,
    per_mutant_timeout_ms: float = 500,
    seed: int | None = None,
) -> SamplingResult:
    """Inline sampling mode — generate ≤max_per_category mutants per category.

    Evaluates within time budget. This is the "active hypothesis testing"
    from §6.2: each sampled mutant tests whether the test suite distinguishes
    a specific behavioral dimension.

    Args:
        budget_ms: Total wall-clock budget for the entire sampling run.
        per_mutant_timeout_ms: Timeout for evaluating a single mutant.
            Separate from budget_ms to prevent one slow mutant from
            consuming the entire budget.
        seed: Deterministic shuffle seed for target selection. Different seeds
            sample different mutants from the same function, reducing sampling
            bias across convergence iterations.
    """
    start = time.monotonic()
    mutants = generate_mutants(
        func_node,
        categories,
        max_per_category=max_per_category,
        seed=seed,
    )

    results_by_cat: dict[MutationCategory, CategoryResult] = {}
    budget_exhausted = False
    all_results: list[MutantResult] = []
    qualname = func_key.split("::", 1)[1] if "::" in func_key else getattr(func_node, "name", None)

    for mutant in mutants:
        if _elapsed(start) > budget_ms:
            budget_exhausted = True
            break

        result = evaluate_mutant(
            mutant,
            test_functions,
            original_func,
            timeout_ms=per_mutant_timeout_ms,
            qualname=qualname,
        )
        all_results.append(result)

        cr = results_by_cat.setdefault(mutant.category, CategoryResult(category=mutant.category))
        cr.total += 1
        if result.killed:
            cr.killed += 1
            if result.killed_by == "assertion":
                cr.killed_by_assertion += 1
            elif result.killed_by == "crash":
                cr.killed_by_crash += 1
        else:
            cr.survived += 1

    per_cat = list(results_by_cat.values())
    total = sum(cr.total for cr in per_cat)
    killed = sum(cr.killed for cr in per_cat)
    survived = total - killed

    return SamplingResult(
        function_key=func_key,
        categories_tested=len(per_cat),
        total_mutants=total,
        total_killed=killed,
        total_survived=survived,
        survival_rate=survived / total if total > 0 else 0.0,
        per_category=per_cat,
        budget_exhausted=budget_exhausted,
        elapsed_ms=_elapsed(start),
    )


def run_function_profiling(
    func_node: ast.FunctionDef,
    func_key: str,
    categories: set[MutationCategory],
    test_functions: list[Callable[..., None]],
    original_func: Callable[..., Any],
    per_mutant_timeout_ms: float = 5000,
    budget_ms: float | None = None,
) -> ProfilingResult:
    """Exhaustive profiling mode — generate all mutants, evaluate with optional budget.

    Returns full survival profile with kill matrix for convergence analysis.
    Result has coverage_depth="profiled" and is_gateable=True.

    Args:
        per_mutant_timeout_ms: Timeout for evaluating a single mutant.
        budget_ms: Optional total wall-clock budget. None means unlimited.
            When exceeded, returns partial results with budget_exhausted=True.
    """
    start = time.monotonic()
    mutants = generate_mutants(func_node, categories)

    results_by_cat: dict[MutationCategory, CategoryResult] = {}
    kill_matrix: dict[str, list[str]] = {}
    survivor_records: list[dict] = []
    killed_records: list[dict] = []
    budget_exhausted = False
    qualname = func_key.split("::", 1)[1] if "::" in func_key else getattr(func_node, "name", None)

    for mutant in mutants:
        if budget_ms is not None and _elapsed(start) > budget_ms:
            budget_exhausted = True
            break

        result = evaluate_mutant(
            mutant,
            test_functions,
            original_func,
            timeout_ms=per_mutant_timeout_ms,
            qualname=qualname,
        )

        cr = results_by_cat.setdefault(mutant.category, CategoryResult(category=mutant.category))
        cr.total += 1
        if result.killed:
            cr.killed += 1
            if result.killed_by == "assertion":
                cr.killed_by_assertion += 1
            elif result.killed_by == "crash":
                cr.killed_by_crash += 1
            elif result.killed_by == "timeout":
                cr.timed_out += 1
            if result.test_name:
                kill_matrix.setdefault(mutant.description, []).append(result.test_name)
            killed_records.append({"mutant": mutant.description, "killed_by": result.killed_by, "test": result.test_name})
        else:
            cr.survived += 1
            survivor_records.append({"mutant": mutant.description})

    per_cat = list(results_by_cat.values())
    total = sum(cr.total for cr in per_cat)
    killed = sum(cr.killed for cr in per_cat)
    survived = total - killed

    return ProfilingResult(
        function_key=func_key,
        categories_tested=len(per_cat),
        total_mutants=total,
        total_killed=killed,
        total_survived=survived,
        survival_rate=survived / total if total > 0 else 0.0,
        per_category=per_cat,
        kill_matrix=kill_matrix,
        survivor_records=survivor_records,
        killed_records=killed_records,
        budget_exhausted=budget_exhausted,
        elapsed_ms=_elapsed(start),
    )
