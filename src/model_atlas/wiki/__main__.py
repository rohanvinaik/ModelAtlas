"""CLI entry point for `python -m model_atlas.wiki`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .drift import check_drift
from .renderer import materialize


def _find_repo_root() -> Path:
    """Walk up from CWD to find the repo root (has wiki.yaml)."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "wiki.yaml").exists():
            return parent
    return cwd


def cmd_materialize(args: argparse.Namespace) -> int:
    repo_root = Path(args.root)
    config = load_config(repo_root / "wiki.yaml")
    output_dir = repo_root / ".wiki"

    manifest = materialize(config, repo_root, output_dir)
    print(f"Materialized {len(manifest.pages)} pages to {output_dir}/")
    print(f"Aggregate hash: {manifest.aggregate_hash}")
    return 0


def cmd_drift(args: argparse.Namespace) -> int:
    repo_root = Path(args.root)
    config = load_config(repo_root / "wiki.yaml")
    output_dir = repo_root / ".wiki"

    report = check_drift(config, repo_root, output_dir)

    if args.check:
        if not report.is_clean:
            print(
                f"ERROR: {report.stale_count} stale, "
                f"{report.orphaned_count} orphaned. "
                "Run materializer to update.",
                file=sys.stderr,
            )
            return 1
        return 0

    print(report.format_human())
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    repo_root = Path(args.root)
    config = load_config(repo_root / "wiki.yaml")
    output_dir = repo_root / ".wiki"

    from .manifest import load_manifest

    manifest = load_manifest(output_dir / "manifest.json")

    print(f"Config: {len(config.pages)} pages defined in wiki.yaml")
    print(f"Materializer version: {config.materializer_version}")

    if manifest is None:
        print("Manifest: not found (run materialize first)")
    else:
        print(f"Manifest: {len(manifest.pages)} pages materialized")
        print(f"Aggregate hash: {manifest.aggregate_hash}")

        report = check_drift(config, repo_root, output_dir)
        if report.is_clean:
            print("Drift: clean")
        else:
            print(
                f"Drift: {report.stale_count} stale, {report.orphaned_count} orphaned"
            )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m model_atlas.wiki",
        description="Wiki materializer — deterministic doc generation",
    )
    parser.add_argument(
        "--root",
        default=str(_find_repo_root()),
        help="Repository root (default: auto-detect from CWD)",
    )

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("materialize", help="Generate wiki pages from source docs")

    drift_parser = sub.add_parser("drift", help="Check for drift")
    drift_parser.add_argument(
        "--check",
        action="store_true",
        help="CI mode: nonzero exit on drift",
    )

    sub.add_parser("status", help="Show wiki status")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    commands = {
        "materialize": cmd_materialize,
        "drift": cmd_drift,
        "status": cmd_status,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
