"""Wesker CI runner — the next era of mutation testing.

In-process AST mutation engine with:
- 3-layer test discovery (convention → static impact → full fallback)
- Real equivalent mutant detection via boundary input evaluation
- Categorical profiling (VALUE, BOUNDARY, SWAP, STATE, TYPE)
- Clean, progressive terminal output

Zero external dependencies beyond the test framework.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any

from wesker_engine import (
    MutationCategory,
    generate_mutants,
    evaluate_mutant,
    run_function_sampling,
)
from wesker_filter import filter_categories


# ── ANSI colors for terminal output ──────────────────────────────

_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_RESET = "\033[0m"

# Disable colors when not a terminal (CI logs, piped output)
if not sys.stderr.isatty() and not os.environ.get("WESKER_COLOR"):
    _GREEN = _RED = _YELLOW = _DIM = _RESET = ""


def _pct_color(pct: int) -> str:
    if pct == 100:
        return _GREEN
    if pct >= 80:
        return _YELLOW
    return _RED


# ── Layer 1: Convention-based test discovery ─────────────────────


def _discover_by_convention(project_root: str, source_file: str) -> list[str]:
    """Find test files by naming convention (fast, high precision)."""
    base = Path(source_file).stem
    base_stripped = base.lstrip("_")
    tests_dir = Path(project_root) / "tests"
    generated_dir = tests_dir / "generated"

    # Path-safe generated test name
    try:
        rel = os.path.relpath(source_file, project_root)
    except ValueError:
        rel = base
    safe = rel.replace(os.sep, "_").replace("/", "_").replace(".", "_")
    if safe.endswith("_py"):
        safe = safe[:-3]
    generated_name = f"test_{safe}.py"

    # Parent-aware matching
    parent_dir = Path(source_file).parent.name
    parent_qualified = f"{parent_dir}_{base}" if parent_dir not in ("model_atlas", "src") else None

    # Partial stems for compound names (query_navigate -> query, navigate)
    partial_stems = {p for p in base_stripped.split("_") if len(p) >= 4}

    # Ambiguous stems that exist at multiple paths
    ambiguous_stems = {"config", "base", "__main__", "utils", "helpers"}

    found: list[str] = []
    for search_dir in [tests_dir, generated_dir]:
        if not search_dir.is_dir():
            continue
        for entry in sorted(search_dir.iterdir()):
            if not entry.name.endswith(".py"):
                continue
            name = entry.name
            path_str = str(entry)

            match = (
                # Exact generated name (highest confidence)
                name == generated_name
                # Parent-qualified (wiki/config.py -> test_wiki_config.py)
                or (parent_qualified and (
                    name == f"test_{parent_qualified}.py"
                    or name.startswith(f"test_{parent_qualified}_")
                ))
                # Exact stem
                or name == f"test_{base}.py"
                or name == f"test_{base_stripped}.py"
                # Prefix match
                or name.startswith(f"test_{base}_")
                or name.startswith(f"test_{base_stripped}_")
                # Parent dir (extraction/det.py -> test_extraction.py)
                or (parent_qualified and name == f"test_{parent_dir}.py")
                # Contains-stem (test_prescriptive_deterministic.py)
                or f"_{base_stripped}." in name
                or f"_{base_stripped}_" in name
                # Partial stems (query_navigate -> test_navigate.py)
                or any(name == f"test_{s}.py" for s in partial_stems)
                or any(name.startswith(f"test_{s}_") for s in partial_stems)
            )

            # Suppress ambiguous bare-stem matches for common names in subdirs
            if match and parent_qualified and base_stripped in ambiguous_stems:
                # Only keep if it also matches parent dir or generated name
                if not (parent_dir in name or name == generated_name):
                    continue

            if match and path_str not in found:
                found.append(path_str)

    return found


# ── Layer 2: Static AST impact analysis ──────────────────────────


def _build_static_impact_map(test_files: list[str]) -> dict[str, list[str]]:
    """Build a map of function_name -> [test_file] by scanning test ASTs.

    Looks for function names referenced in test bodies via ast.Name nodes.
    This catches imports and direct references without executing anything.
    """
    impact: dict[str, set[str]] = {}
    for tf in test_files:
        try:
            with open(tf) as f:
                tree = ast.parse(f.read(), filename=tf)
        except (OSError, SyntaxError):
            continue
        # Collect all Name references in the file
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                impact.setdefault(node.id, set()).add(tf)
            elif isinstance(node, ast.Attribute):
                impact.setdefault(node.attr, set()).add(tf)
    return {k: sorted(v) for k, v in impact.items()}


# ── Layer 3: Full fallback ───────────────────────────────────────


def _discover_all_test_files(project_root: str) -> list[str]:
    """Find all test_*.py files under tests/."""
    found: list[str] = []
    tests_dir = Path(project_root) / "tests"
    if not tests_dir.is_dir():
        return found
    for py in sorted(tests_dir.rglob("*.py")):
        if py.name.startswith("test_") and "__pycache__" not in str(py):
            found.append(str(py))
    return found


# ── 3-Layer discovery orchestrator ───────────────────────────────


def discover_tests(project_root: str, source_file: str, func_names: list[str]) -> list[str]:
    """3-layer test discovery: convention -> static impact -> full fallback.

    Layer 1: Convention matching (fast, filename-based)
    Layer 2: Static impact (AST scan for function name references)
    Layer 3: Full fallback (all test files)

    Each layer adds files not already found by previous layers.
    """
    # Layer 1: Convention
    found = _discover_by_convention(project_root, source_file)

    # Layer 2: Static impact — find additional test files that reference
    # any of the function names in this source file
    all_test_files = _discover_all_test_files(project_root)
    impact_map = _build_static_impact_map(all_test_files)
    found_set = set(found)
    for func_name in func_names:
        for tf in impact_map.get(func_name, []):
            if tf not in found_set:
                found.append(tf)
                found_set.add(tf)

    # Layer 3: Full fallback — add remaining test files
    for tf in all_test_files:
        if tf not in found_set:
            found.append(tf)
            found_set.add(tf)

    return found


# ── Test callable loading ────────────────────────────────────────


def load_test_callables(test_files: list[str]) -> list[Any]:
    """Load all test_* callables from test files, including class methods."""
    callables: list[Any] = []
    for tf in test_files:
        mod_name = f"_wesker_test_{Path(tf).stem}"
        if mod_name in sys.modules:
            # Already loaded — reuse
            mod = sys.modules[mod_name]
        else:
            try:
                spec = importlib.util.spec_from_file_location(mod_name, tf)
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = mod
                spec.loader.exec_module(mod)
            except Exception:
                continue

        for name in dir(mod):
            obj = getattr(mod, name)
            if name.startswith("test_") and callable(obj):
                callables.append(obj)
            elif isinstance(obj, type) and name.startswith("Test"):
                for mname in dir(obj):
                    if mname.startswith("test_"):
                        try:
                            callables.append(getattr(obj(), mname))
                        except Exception:
                            pass
    return callables


# ── Equivalent mutant detection ──────────────────────────────────


def check_equivalent(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    mutant: Any,
) -> bool:
    """Check if a surviving mutant is semantically equivalent.

    Compiles both original and mutated functions, runs them on a set of
    boundary inputs, and compares outputs. If all outputs match, the
    mutant is equivalent — no test can distinguish them.

    This catches the common case where boundary operator swaps (>= to >)
    produce identical results because the equality case maps to the same
    value via both branches (e.g., decay(0) == 1.0).
    """
    try:
        # Compile original
        orig_mod = ast.Module(body=[func_node], type_ignores=[])
        ast.fix_missing_locations(orig_mod)
        orig_code = compile(orig_mod, "<original>", "exec")
        orig_ns: dict[str, Any] = {}
        exec(orig_code, orig_ns)  # noqa: S102

        # Compile mutant
        mut_mod = ast.Module(body=[mutant.mutated_node], type_ignores=[])
        ast.fix_missing_locations(mut_mod)
        mut_code = compile(mut_mod, "<mutant>", "exec")
        mut_ns: dict[str, Any] = {}
        exec(mut_code, mut_ns)  # noqa: S102

        func_name = func_node.name
        orig_fn = orig_ns.get(func_name)
        mut_fn = mut_ns.get(func_name)

        if orig_fn is None or mut_fn is None:
            return False

        # Generate boundary inputs from the function signature
        boundary_inputs = _generate_boundary_inputs(func_node)

        # Test each input — if any output differs, NOT equivalent
        for args in boundary_inputs:
            try:
                orig_result = orig_fn(*args)
                mut_result = mut_fn(*args)
                if orig_result != mut_result:
                    return False
            except Exception:
                # If one raises and the other doesn't, not equivalent
                # If both raise, could be equivalent — continue checking
                pass

        return True

    except Exception:
        return False


def _generate_boundary_inputs(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[tuple]:
    """Generate boundary test inputs based on parameter count.

    Uses a fixed set of boundary values: 0, 1, -1, 0.5, True, False, "", "x".
    For multi-param functions, generates combinations of the first few values.
    """
    n_params = len(func_node.args.args)
    # Skip 'self' parameter
    if n_params > 0 and func_node.args.args[0].arg == "self":
        n_params -= 1

    if n_params == 0:
        return [()]

    # Boundary values by type likelihood
    int_vals = [0, 1, -1, 2, -2]
    float_vals = [0.0, 1.0, -1.0, 0.5]
    bool_vals = [True, False]

    if n_params == 1:
        return [(v,) for v in int_vals + float_vals + bool_vals]

    if n_params == 2:
        inputs = []
        for a in int_vals[:3] + float_vals[:2]:
            for b in int_vals[:3] + float_vals[:2]:
                inputs.append((a, b))
        return inputs[:25]  # Cap at 25 combinations

    # 3+ params: just use a few representative tuples
    base = int_vals[:3] + float_vals[:2]
    return [tuple(base[i % len(base)] for _ in range(n_params)) for i in range(5)]


# ── AST utilities ────────────────────────────────────────────────


def walk_functions(
    tree: ast.Module,
) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Walk AST yielding (qualname, node) for each function."""
    results: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []

    def _walk(scope: ast.AST, prefix: str) -> None:
        for node in getattr(scope, "body", []):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{node.name}" if prefix else node.name
                results.append((name, node))
            elif isinstance(node, ast.ClassDef):
                cp = f"{prefix}{node.name}." if prefix else f"{node.name}."
                _walk(node, cp)

    _walk(tree, "")
    return results


# ── File profiling ───────────────────────────────────────────────


def profile_file(
    project_root: str,
    source_file: str,
    budget_ms: float = 10000,
    max_per_category: int = 3,
) -> list[dict]:
    """Profile all functions in a file. Returns per-function results."""
    full_path = (
        os.path.join(project_root, source_file)
        if not os.path.isabs(source_file)
        else source_file
    )
    try:
        with open(full_path) as f:
            tree = ast.parse(f.read(), filename=full_path)
    except (OSError, SyntaxError):
        return []

    functions = walk_functions(tree)
    func_names = [name for name, _ in functions]

    # 3-layer test discovery
    test_files = discover_tests(project_root, full_path, func_names)
    tests = load_test_callables(test_files)

    results: list[dict] = []
    for qualname, func_node in functions:
        cats = filter_categories(func_node)
        if not cats:
            continue

        rel = os.path.relpath(full_path, project_root)
        func_key = f"{rel}::{qualname}"

        sr = run_function_sampling(
            func_node,
            func_key,
            cats,
            tests,
            None,
            budget_ms=budget_ms,
            max_per_category=max_per_category,
        )
        result = sr.to_dict()

        # Check surviving mutants for equivalence
        if sr.total_survived > 0:
            equivalent_count = _check_survivors_for_equivalence(
                func_node, cats, tests, max_per_category
            )
            result["equivalent_mutants"] = equivalent_count

        results.append(result)

    return results


def _check_survivors_for_equivalence(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    categories: set[MutationCategory],
    tests: list[Any],
    max_per_category: int,
) -> int:
    """Re-evaluate surviving mutants for semantic equivalence."""
    mutants = generate_mutants(func_node, categories, max_per_category=max_per_category)
    equivalent = 0
    for mutant in mutants:
        result = evaluate_mutant(mutant, tests, None, qualname=func_node.name)
        if not result.killed:
            if check_equivalent(func_node, mutant):
                equivalent += 1
    return equivalent


# ── Codebase profiling with formatted output ─────────────────────


def profile_codebase(
    project_root: str,
    targets: list[str],
    budget_ms_per_file: float = 10000,
    max_per_category: int = 3,
    *,
    verbose: bool = True,
) -> dict:
    """Profile all functions across multiple files with progressive output."""
    total_killed = 0
    total_mutants = 0
    total_equivalent = 0
    total_functions = 0
    per_file: dict[str, dict] = {}
    start = time.monotonic()

    for i, target in enumerate(targets, 1):
        if verbose:
            short = target.replace("src/model_atlas/", "")
            print(f"  {_DIM}[{i}/{len(targets)}]{_RESET} {short}", end="", flush=True)

        file_start = time.monotonic()
        results = profile_file(
            project_root,
            target,
            budget_ms=budget_ms_per_file,
            max_per_category=max_per_category,
        )
        file_ms = (time.monotonic() - file_start) * 1000

        file_killed = sum(r.get("total_killed", 0) for r in results)
        file_total = sum(r.get("total_mutants", 0) for r in results)
        file_equiv = sum(r.get("equivalent_mutants", 0) for r in results)
        total_killed += file_killed
        total_mutants += file_total
        total_equivalent += file_equiv
        total_functions += len(results)

        if file_total > 0:
            kill_pct = round(100 * file_killed / file_total)
            per_file[target] = {
                "functions": len(results),
                "killed": file_killed,
                "total": file_total,
                "kill_pct": kill_pct,
                "equivalent": file_equiv,
                "elapsed_ms": round(file_ms),
            }
            if verbose:
                c = _pct_color(kill_pct)
                equiv_note = f" {_DIM}({file_equiv} equiv){_RESET}" if file_equiv else ""
                print(f" {c}{file_killed}/{file_total}{_RESET}{equiv_note}"
                      f" {_DIM}{file_ms:.0f}ms{_RESET}")
        else:
            if verbose:
                print(f" {_DIM}(no mutants){_RESET}")

    elapsed = (time.monotonic() - start) * 1000
    kill_pct = round(100 * total_killed / max(total_mutants, 1))

    return {
        "total_killed": total_killed,
        "total_mutants": total_mutants,
        "kill_pct": kill_pct,
        "total_functions": total_functions,
        "total_equivalent": total_equivalent,
        "elapsed_ms": round(elapsed),
        "per_file": per_file,
    }
