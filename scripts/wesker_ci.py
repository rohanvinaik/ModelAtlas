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

    # Parent-aware matching for files in subdirs (wiki/config.py -> test_wiki_config)
    parent_dir = Path(source_file).parent.name
    parent_qualified = f"{parent_dir}_{base}" if parent_dir not in ("model_atlas", "src") else None

    # Partial stems — e.g., query_navigate -> query, navigate
    partial_stems = set()
    for part in base_stripped.split("_"):
        if len(part) >= 4:
            partial_stems.add(part)

    # Two-pass: high-confidence matches first, then broad matches.
    # If high-confidence matches find enough tests, skip broad matches
    # to avoid pulling in tests for a different module with the same stem.
    high_confidence: list[str] = []
    broad: list[str] = []

    for search_dir in [tests_dir, generated_dir]:
        if not search_dir.is_dir():
            continue
        for entry in sorted(search_dir.iterdir()):
            if not entry.name.endswith(".py"):
                continue
            name = entry.name
            path_str = str(entry)

            # High confidence: generated name or parent-qualified
            if name == generated_name:
                high_confidence.append(path_str)
            elif parent_qualified and (
                name == f"test_{parent_qualified}.py"
                or name.startswith(f"test_{parent_qualified}_")
            ):
                high_confidence.append(path_str)
            elif (
                (name.endswith(f"_{base}.py") or name.endswith(f"_{base_stripped}.py"))
                # Suffix match is only high-confidence if the name also contains
                # the parent dir (avoids test_config.py matching wiki/config.py)
                and (not parent_qualified or parent_dir in name)
            ):
                high_confidence.append(path_str)
            # Broad: bare stem match, partial stem match, parent dir match,
            # and contains-stem match (catches test_prescriptive_deterministic.py)
            elif (
                name == f"test_{base}.py"
                or name == f"test_{base_stripped}.py"
                or name.startswith(f"test_{base}_")
                or name.startswith(f"test_{base_stripped}_")
                or any(name == f"test_{stem}.py" for stem in partial_stems)
                or any(name.startswith(f"test_{stem}_") for stem in partial_stems)
                # Parent dir match (extraction/deterministic.py -> test_extraction.py)
                or (parent_qualified and name == f"test_{parent_dir}.py")
                # Contains-stem (test_prescriptive_deterministic.py contains "deterministic")
                or (f"_{base_stripped}." in name or f"_{base_stripped}_" in name)
            ):
                broad.append(path_str)

    # For files with very common names in subdirectories (wiki/config.py),
    # broad stem matches pull in wrong-module tests (test_config.py tests
    # model_atlas/config.py, not wiki/config.py). Only suppress broad matches
    # when the stem is ambiguous (exists at multiple paths).
    ambiguous_stems = {"config", "base", "__main__"}
    suppress_broad = parent_qualified is not None and base_stripped in ambiguous_stems
    if suppress_broad and high_confidence:
        return list(dict.fromkeys(high_confidence))
    return list(dict.fromkeys(high_confidence + broad))


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

    # Always load all test files as fallback. Convention-matched files are loaded
    # first (and their callables appear first for monkey-patching priority), but
    # cross-cutting test files (test_mutation_gaps.py, test_integration.py) that
    # exercise functions from multiple modules are also included.
    all_test_files = _discover_all_test_files(project_root)
    extra = [f for f in all_test_files if f not in set(test_files)]
    if extra:
        fallback_tests = load_test_callables(extra)
        tests.extend(fallback_tests)

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
            project_root,
            target,
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
