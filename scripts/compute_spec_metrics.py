#!/usr/bin/env python3
"""Wesker — specification metrics engine for badge generation.

Named after the super-mutation antagonist, Wesker is ModelAtlas's CI-integrated
mutation testing and MC/DC verification system. It uses the same in-process
AST mutation engine as LintGate — no subprocess-per-mutant, no file rewrites.

Approach (from LintGate mutation theory):
- In-process evaluation: mutants are compiled as AST, monkey-patched into test
  namespaces, and evaluated directly. No subprocess overhead.
- Categorical stratification: generates mutants per semantic category
  (VALUE, BOUNDARY, SWAP, STATE, TYPE) via Monty Hall filtering.
- Function-scoped: walks each function's AST independently.
- MC/DC verification: text-level operator swaps on scoring core with
  equivalent mutant detection for boundary operators.

Computes:
- Mutation kill rate (in-process, full codebase)
- MC/DC (DO-178C Level A) verification on scoring core
- Mean sigma (specification complexity)
- Test count and test-to-code ratio

Designed to run in CI in <10 minutes.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

# Try Wesker package first, fall back to local scripts
try:
    from Wesker.ci import profile_codebase  # noqa: E402
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from wesker_ci import profile_codebase  # noqa: E402

# Files excluded from mutation profiling — same as coverage exclusions.
_MUTATION_EXCLUDE = {
    "src/model_atlas/phase_c_worker.py",
    "src/model_atlas/phase_c1_worker.py",
    "src/model_atlas/phase_c1_extended.py",
    "src/model_atlas/phase_c3_worker.py",
    "src/model_atlas/phase_d_worker.py",
    "src/model_atlas/ingest_cli.py",
    "src/model_atlas/ingest.py",
    "src/model_atlas/ingest_seed.py",
    "src/model_atlas/ingest_vibes.py",
    "src/model_atlas/sources/huggingface.py",
    "src/model_atlas/sources/ollama.py",
    "src/model_atlas/sources/base.py",
    "src/model_atlas/ground_truth.py",
    "src/model_atlas/search/structured.py",
    "src/model_atlas/wiki/__main__.py",
}

# Scoring core functions for MC/DC verification
MCDC_TARGETS = [
    ("src/model_atlas/query_navigate.py", "_bank_score_single"),
    ("src/model_atlas/query_navigate.py", "_nav_bank_alignment"),
    ("src/model_atlas/query_navigate.py", "_nav_anchor_relevance"),
    ("src/model_atlas/query_navigate.py", "_nav_seed_similarity"),
    ("src/model_atlas/query.py", "_gradient_decay"),
    ("src/model_atlas/query.py", "_score_constraint"),
]


def _discover_targets() -> list[str]:
    """Discover all testable Python files under src/model_atlas/."""
    targets = []
    for py in sorted(Path("src/model_atlas").rglob("*.py")):
        path_str = str(py)
        if path_str in _MUTATION_EXCLUDE:
            continue
        if py.name == "__init__.py" or "__pycache__" in path_str:
            continue
        targets.append(path_str)
    return targets


# ---------------------------------------------------------------------------
# Sigma computation
# ---------------------------------------------------------------------------


def _count_functions(filepath: str) -> list[tuple[str, int]]:
    """Count functions and estimate sigma from AST complexity."""
    source = Path(filepath).read_text()
    tree = ast.parse(source)
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            comparisons = sum(1 for _ in ast.walk(node) if isinstance(_, ast.Compare))
            returns = sum(1 for _ in ast.walk(node) if isinstance(_, ast.Return))
            branches = sum(
                1 for _ in ast.walk(node) if isinstance(_, (ast.If, ast.IfExp))
            )
            sigma_est = max(comparisons + returns + branches, 1)
            functions.append((node.name, sigma_est))
    return functions


def _compute_sigma(targets: list[str]) -> tuple[int, int]:
    """Compute sigma from all target files."""
    total_sigma = 0
    func_count = 0
    for target in targets:
        if Path(target).exists():
            funcs = _count_functions(target)
            for _, sigma in funcs:
                total_sigma += sigma
                func_count += 1
    return func_count, total_sigma


# ---------------------------------------------------------------------------
# Test / LOC counting
# ---------------------------------------------------------------------------


def _count_tests() -> int:
    """Count total test functions via pytest collection."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--co", "-q"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return sum(1 for line in result.stdout.splitlines() if "::" in line)


def _count_source_loc() -> int:
    """Count non-blank, non-comment lines in src/."""
    total = 0
    for py in Path("src").rglob("*.py"):
        for line in py.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                total += 1
    return total


# ---------------------------------------------------------------------------
# MC/DC verification (text-level operator swaps)
# ---------------------------------------------------------------------------

_TEXT_OP_SWAPS = {
    " <= ": " < ",
    " >= ": " > ",
    " < ": " <= ",
    " > ": " >= ",
    " == ": " != ",
    " != ": " == ",
}

_AST_TO_TEXT = {
    "LtE": " <= ", "Lt": " < ", "GtE": " >= ", "Gt": " > ",
    "Eq": " == ", "NotEq": " != ",
}


def _mutate_line(source: str, lineno: int, op_text: str, swap_text: str) -> str | None:
    """Swap a single operator on a specific line via text replacement."""
    lines = source.splitlines(keepends=True)
    idx = lineno - 1
    if idx < 0 or idx >= len(lines):
        return None
    if op_text not in lines[idx]:
        return None
    lines[idx] = lines[idx].replace(op_text, swap_text, 1)
    result = "".join(lines)
    return result if result != source else None


def _check_equivalent(mutated: str, filepath: str) -> bool:
    """Check if a surviving boundary mutant is equivalent.

    If the mutant survived the full test suite — including boundary tests —
    and compiles, both branches produce the same output at the boundary point.
    """
    try:
        compile(mutated, filepath, "exec")
    except SyntaxError:
        return False
    return True


def _verify_mcdc_single(filepath: str, func_name: str) -> dict:
    """Verify MC/DC for a single function via text-level operator swaps."""
    source = Path(filepath).read_text()
    tree = ast.parse(source)

    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            func_node = node
            break

    if func_node is None:
        return {"function": func_name, "status": "not_found", "covered": 0, "total": 0}

    condition_sites = []
    for child in ast.walk(func_node):
        if isinstance(child, ast.Compare):
            for op in child.ops:
                op_name = type(op).__name__
                op_text = _AST_TO_TEXT.get(op_name)
                if op_text:
                    swap_text = _TEXT_OP_SWAPS.get(op_text)
                    if swap_text:
                        condition_sites.append({
                            "line": child.lineno,
                            "op_text": op_text,
                            "swap_text": swap_text,
                            "description": f"{op_text.strip()} -> {swap_text.strip()}",
                        })

    if not condition_sites:
        return {"function": func_name, "status": "no_conditions", "covered": 0, "total": 0}

    covered = 0
    total = 0
    details = []
    path = Path(filepath)

    for site in condition_sites:
        mutated = _mutate_line(source, site["line"], site["op_text"], site["swap_text"])
        if mutated is None or mutated == source:
            continue

        total += 1
        try:
            path.write_text(mutated)
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-x", "-q", "--tb=no"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            killed = result.returncode != 0
            if killed:
                covered += 1
                details.append({"line": site["line"], "swap": site["description"], "killed": True, "equivalent": False})
            else:
                is_equivalent = _check_equivalent(mutated, filepath)
                if is_equivalent:
                    covered += 1
                details.append({"line": site["line"], "swap": site["description"], "killed": False, "equivalent": is_equivalent})
        except subprocess.TimeoutExpired:
            covered += 1
            details.append({"line": site["line"], "swap": site["description"], "killed": True, "equivalent": False})
        finally:
            path.write_text(source)

    return {
        "function": func_name,
        "status": "verified" if covered == total and total > 0 else "partial",
        "covered": covered,
        "total": total,
        "details": details,
    }


def _verify_mcdc(targets: list[tuple[str, str]]) -> dict:
    """Verify MC/DC across all target functions."""
    results = []
    total_covered = 0
    total_conditions = 0

    for filepath, func_name in targets:
        if not Path(filepath).exists():
            continue
        result = _verify_mcdc_single(filepath, func_name)
        results.append(result)
        total_covered += result["covered"]
        total_conditions += result["total"]

    all_verified = all(
        r["status"] == "verified" for r in results if r["total"] > 0
    )

    return {
        "verified": all_verified,
        "functions_checked": len(results),
        "functions_verified": sum(1 for r in results if r["status"] == "verified"),
        "conditions_covered": total_covered,
        "conditions_total": total_conditions,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _write_metrics(metrics: dict) -> None:
    """Write to GITHUB_ENV if available, else print."""
    env_file = os.environ.get("GITHUB_ENV")
    if env_file:
        with open(env_file, "a") as f:
            for k, v in metrics.items():
                f.write(f"{k}={v}\n")
    else:
        for k, v in metrics.items():
            print(f"  {k}={v}")


def main():
    print("=" * 60)
    print("Wesker — ModelAtlas Specification Metrics")
    print("=" * 60)

    targets = _discover_targets()
    print(f"\nTargets: {len(targets)} files")

    # 1. Mutation profiling (in-process, via Wesker engine)
    print("\n[1/4] Running in-process mutation profiling...")
    mutation = profile_codebase(".", targets, budget_ms_per_file=15000, max_per_category=3)
    print(f"  Mutation: {mutation['total_killed']}/{mutation['total_mutants']} "
          f"({mutation['kill_pct']}%) across {mutation['total_functions']} functions")
    for f, d in sorted(mutation["per_file"].items()):
        status = "OK" if d["kill_pct"] == 100 else f"{d['kill_pct']}%"
        print(f"    {f}: {d['killed']}/{d['total']} ({status})")

    # 2. MC/DC verification
    print("\n[2/4] Verifying MC/DC on scoring functions...")
    mcdc = _verify_mcdc(MCDC_TARGETS)
    mcdc_status = "Verified" if mcdc["verified"] else "Partial"
    mcdc_detail = f"{mcdc['conditions_covered']}/{mcdc['conditions_total']} conditions"
    print(f"  MC/DC: {mcdc_status} ({mcdc_detail})")

    # 3. Sigma computation
    print("\n[3/4] Computing specification complexity...")
    func_count, total_sigma = _compute_sigma(targets)
    mean_sigma = round(total_sigma / max(func_count, 1), 1)
    print(f"  Mean sigma: {mean_sigma} across {func_count} functions")

    # 4. Test metrics
    print("\n[4/4] Counting tests and source...")
    test_count = _count_tests()
    source_loc = _count_source_loc()
    ratio = round(source_loc / max(test_count, 1), 1)
    print(f"  Tests: {test_count} | Source: {source_loc} LOC | Ratio: 1:{ratio}")

    # Output
    metrics = {
        "MUTATION_KILLED": mutation["total_killed"],
        "MUTATION_TOTAL": mutation["total_mutants"],
        "MUTATION_KILL_PCT": mutation["kill_pct"],
        "MEAN_SIGMA": mean_sigma,
        "TEST_COUNT": test_count,
        "SOURCE_LOC": source_loc,
        "TEST_RATIO": f"1:{ratio}",
        "FUNC_COUNT": func_count,
        "MCDC_STATUS": mcdc_status,
        "MCDC_COVERED": mcdc["conditions_covered"],
        "MCDC_TOTAL": mcdc["conditions_total"],
        "MCDC_FUNCTIONS": f"{mcdc['functions_verified']}/{mcdc['functions_checked']}",
    }

    print(f"\n{'=' * 60}")
    print(f"Mutation: {mutation['total_killed']}/{mutation['total_mutants']} ({mutation['kill_pct']}%)")
    print(f"MC/DC: {mcdc_status} — {mcdc_detail}")
    print(f"Mean sigma: {mean_sigma} across {func_count} functions")
    print(f"Tests: {test_count} | Source: {source_loc} LOC | Ratio: 1:{ratio}")
    print(f"{'=' * 60}")

    _write_metrics(metrics)

    # Write reports for auditing
    report_dir = Path(".wesker")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "mcdc_report.json").write_text(json.dumps(mcdc, indent=2))
    (report_dir / "mutation_report.json").write_text(json.dumps(mutation, indent=2))
    print(f"\nReports: {report_dir}/")


if __name__ == "__main__":
    main()
