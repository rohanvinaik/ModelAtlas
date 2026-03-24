"""Wesker CI runner — standalone mutation testing for badge generation.

Uses the same in-process AST mutation engine as LintGate, adapted for
CI without any LintGate dependencies. Loads test callables directly,
evaluates mutants via monkey-patching, reports per-function kill rates.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from wesker_engine import (
    MutationCategory,
    evaluate_mutant,
    generate_mutants,
    run_function_sampling,
)
from wesker_filter import filter_categories


def discover_test_files(project_root: str, source_file: str) -> list[str]:
    """Find test files relevant to a source file by naming convention."""
    base = Path(source_file).stem
    base_stripped = base.lstrip("_")
    tests_dir = Path(project_root) / "tests"
    generated_dir = tests_dir / "generated"

    # Compute path-safe generated test name
    try:
        rel = os.path.relpath(source_file, project_root)
    except ValueError:
        rel = base
    safe = rel.replace(os.sep, "_").replace("/", "_").replace(".", "_")
    if safe.endswith("_py"):
        safe = safe[:-3]
    generated_name = f"test_{safe}.py"

    # Also try partial stems — e.g., query_navigate -> query, navigate
    partial_stems = set()
    for part in base_stripped.split("_"):
        if len(part) >= 4:  # Skip very short parts
            partial_stems.add(part)

    found: list[str] = []
    for search_dir in [tests_dir, generated_dir]:
        if not search_dir.is_dir():
            continue
        for entry in sorted(search_dir.iterdir()):
            if not entry.name.endswith(".py"):
                continue
            name = entry.name
            match = (
                name == f"test_{base}.py"
                or name == f"test_{base_stripped}.py"
                or name.startswith(f"test_{base}_")
                or name.startswith(f"test_{base_stripped}_")
                or name == generated_name
                # Suffix match for generated tests
                or name.endswith(f"_{base}.py")
                or name.endswith(f"_{base_stripped}.py")
                # Partial stem match — catches test_navigate.py for query_navigate.py
                or any(name == f"test_{stem}.py" for stem in partial_stems)
                or any(name.startswith(f"test_{stem}_") for stem in partial_stems)
            )
            if match and str(entry) not in found:
                found.append(str(entry))
    return found


def load_test_callables(test_files: list[str]) -> list[Any]:
    """Load all test_* callables from test files, including class methods."""
    callables: list[Any] = []
    for tf in test_files:
        mod_name = f"_wesker_test_{Path(tf).stem}"
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

    test_files = discover_test_files(project_root, full_path)
    tests = load_test_callables(test_files)

    results: list[dict] = []
    for qualname, func_node in walk_functions(tree):
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
        results.append(sr.to_dict())

    return results


def profile_codebase(
    project_root: str,
    targets: list[str],
    budget_ms_per_file: float = 10000,
    max_per_category: int = 3,
) -> dict:
    """Profile all functions across multiple files. Returns aggregate metrics."""
    total_killed = 0
    total_mutants = 0
    total_functions = 0
    per_file: dict[str, dict] = {}

    for target in targets:
        results = profile_file(
            project_root, target,
            budget_ms=budget_ms_per_file,
            max_per_category=max_per_category,
        )
        file_killed = sum(r.get("total_killed", 0) for r in results)
        file_total = sum(r.get("total_mutants", 0) for r in results)
        total_killed += file_killed
        total_mutants += file_total
        total_functions += len(results)

        if file_total > 0:
            per_file[target] = {
                "functions": len(results),
                "killed": file_killed,
                "total": file_total,
                "kill_pct": round(100 * file_killed / file_total),
            }

    kill_pct = round(100 * total_killed / max(total_mutants, 1))
    return {
        "total_killed": total_killed,
        "total_mutants": total_mutants,
        "kill_pct": kill_pct,
        "total_functions": total_functions,
        "per_file": per_file,
    }
