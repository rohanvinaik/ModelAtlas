"""CLI interface for the ModelAtlas ingestion pipeline.

Dispatches to pipeline functions (phase A/B/C) and Phase C/D management
commands based on command-line arguments.
"""

from __future__ import annotations

import argparse
import json
import logging

from . import db, db_ingest
from .config import INGEST_MIN_LIKES, INGEST_VIBE_MIN_LIKES
from .ingest import print_status, run


def _cmd_status(args: argparse.Namespace) -> None:
    """Handle --status command."""
    ingest_conn = db_ingest.get_connection()
    db_ingest.init_db(ingest_conn)
    print_status(ingest_conn)
    ingest_conn.close()

    network_conn = db.get_connection()
    db.init_db(network_conn)
    from .ingest_phase_c import print_phase_c_status

    print_phase_c_status(network_conn)
    network_conn.close()


def _cmd_phase_c(args: argparse.Namespace) -> bool:
    """Handle Phase C export/merge commands. Returns True if handled."""
    from . import ingest_phase_c as pc

    simple_cmds = [
        ("export_c1", lambda c, a: f"Exported {pc.export_c1(c)} model_ids for C1"),
        ("merge_c1", lambda c, a: f"C1 merge: {pc.merge_c1(c, a.merge_c1)}"),
        (
            "export_c2",
            lambda c, a: (
                f"Exported {pc.export_c2(c, num_shards=a.export_c2, min_likes=a.min_likes)}"
                f" C2 prompts across {a.export_c2} shards"
            ),
        ),
        ("merge_c2", lambda c, a: f"C2 merge: {pc.merge_c2(c, a.merge_c2)}"),
        (
            "export_c3",
            lambda c, a: (
                f"Exported {pc.export_c3(c, num_shards=a.export_c3)}"
                f" C3 quality gate prompts across {a.export_c3} shards"
            ),
        ),
        ("merge_c3", lambda c, a: f"C3 merge: {pc.merge_c3(c, a.merge_c3)}"),
        (
            "select_summaries",
            lambda c, a: f"Summary selection: {pc.select_summaries(c)}",
        ),
    ]
    for attr, handler in simple_cmds:
        val = getattr(args, attr, None)
        if val is not None and val is not False:
            network_conn = db.get_connection()
            db.init_db(network_conn)
            print(handler(network_conn, args))
            network_conn.close()
            return True

    if args.validate_ground_truth:
        print("C4 offline validation not yet implemented")
        return True

    return False


def _cmd_audit_c2() -> None:
    """D1: audit C2 anchors against ground truth."""
    from .phase_d_audit import audit_c2

    network_conn = db.get_connection()
    db.init_db(network_conn)
    ingest_conn = db_ingest.get_connection()
    db_ingest.init_db(ingest_conn)
    result = audit_c2(network_conn, ingest_conn)
    network_conn.close()
    ingest_conn.close()
    print(
        f"D1 audit: {result.total_audited} models, "
        f"{result.total_mismatches} mismatches, "
        f"types={result.per_type_counts}"
    )


def _cmd_expand_dict(args: argparse.Namespace) -> None:
    """D2: expand anchor dictionary from a spec file."""
    from .phase_d_expand import expand_dictionary

    network_conn = db.get_connection()
    db.init_db(network_conn)
    result = expand_dictionary(
        network_conn, args.expand_dictionary, dry_run=args.dry_run
    )
    network_conn.close()
    dry = "(dry run)" if args.dry_run else ""
    print(
        f"D2 expansion{dry}: "
        f"{result.anchors_created} anchors created, "
        f"{result.models_linked} models linked, "
        f"{result.models_queued} queued"
    )
    for label, stats in result.per_label.items():
        print(
            f"  {label}: matched={stats['matched']} "
            f"linked={stats.get('linked', 0)} queued={stats.get('queued', 0)}"
        )


def _cmd_phase_d(args: argparse.Namespace) -> bool:
    """Handle Phase D commands. Returns True if handled."""
    if args.audit_c2:
        _cmd_audit_c2()
        return True

    if args.expand_dictionary:
        _cmd_expand_dict(args)
        return True

    if args.export_d3 is not None:
        from .phase_d_heal import export_d3

        network_conn = db.get_connection()
        db.init_db(network_conn)
        ingest_conn = db_ingest.get_connection()
        db_ingest.init_db(ingest_conn)
        result = export_d3(
            network_conn,
            ingest_conn,
            tier=args.heal_tier,
            budget=args.heal_budget,
            num_shards=args.export_d3,
            seed=args.heal_seed,
        )
        network_conn.close()
        ingest_conn.close()
        print(
            f"D3 export: {result.total_exported} models across "
            f"{result.num_shards} shards (tier={result.tier}, run_id={result.run_id})"
        )
        return True

    if args.merge_d3:
        from .phase_d_heal import merge_d3

        if not args.run_id:
            print("Error: --merge-d3 requires --run-id")
            return True
        network_conn = db.get_connection()
        db.init_db(network_conn)
        result = merge_d3(network_conn, args.merge_d3, args.run_id)
        network_conn.close()
        print(f"D3 merge: {result}")
        return True

    if args.export_training_data:
        from .phase_d_training import export_training_data

        network_conn = db.get_connection()
        db.init_db(network_conn)
        stats = export_training_data(
            network_conn, args.export_training_data, tier=args.training_tier
        )
        network_conn.close()
        print(
            f"D4 export: {stats.total_examples} examples to {stats.output_path} "
            f"(by_tier={stats.by_tier})"
        )
        return True

    if args.phase_d_status:
        _cmd_phase_d_status()
        return True

    return False


def _cmd_phase_d_status() -> None:
    """Show Phase D run history and training data stats."""
    from .phase_d_training import get_training_data_stats

    network_conn = db.get_connection()
    db.init_db(network_conn)

    print("Phase D Run History")
    print("=" * 60)
    rows = network_conn.execute(
        "SELECT run_id, phase, status, started_at, summary "
        "FROM phase_d_runs ORDER BY started_at DESC LIMIT 20"
    ).fetchall()
    for r in rows:
        print(f"  {r[0][:8]}... phase={r[1]} status={r[2]} started={r[3]}")
        if r[4]:
            try:
                s = json.loads(r[4])
                print(f"    summary: {s}")
            except (ValueError, TypeError):
                pass

    stats = get_training_data_stats(network_conn)
    print(
        f"\nTraining Data: {stats['total_corrections']} corrections, "
        f"{stats['distinct_models']} models"
    )
    if stats["by_tier"]:
        print(f"  by tier: {stats['by_tier']}")
    if stats["by_mismatch_type"]:
        print(f"  by mismatch: {stats['by_mismatch_type']}")

    network_conn.close()


def _cmd_phase_e(args: argparse.Namespace) -> bool:
    """Handle Phase E commands. Returns True if handled."""
    if args.export_e is not None:
        from .ingest_phase_e_export import export_phase_e

        network_conn = db.get_connection()
        db.init_db(network_conn)
        bank_list = args.export_e_banks.upper().split(",") if args.export_e_banks else None
        n = export_phase_e(
            network_conn,
            num_shards=args.export_e,
            banks=bank_list,
            min_downloads=args.export_e_min_downloads,
            full_corpus=args.export_e_full_corpus,
        )
        network_conn.close()
        print(f"Phase E export: {n} models across {args.export_e} shards")
        return True

    if args.merge_e:
        from .ingest_phase_e import merge_phase_e

        network_conn = db.get_connection()
        db.init_db(network_conn)
        result = merge_phase_e(
            network_conn, args.merge_e, dry_run=args.merge_e_dry_run
        )
        network_conn.close()
        dry = " (DRY RUN)" if args.merge_e_dry_run else ""
        print(f"Phase E merge{dry}: {result}")
        return True

    if args.phase_e_status:
        from .ingest_phase_e import phase_e_status

        network_conn = db.get_connection()
        db.init_db(network_conn)
        status = phase_e_status(network_conn)
        network_conn.close()
        print("Phase E Web Enrichment Status")
        print("=" * 50)
        print(f"  Models enriched: {status['web_enriched_models']}/{status['total_models']} "
              f"({status['coverage_pct']}%)")
        print(f"  Web summaries: {status['web_summaries']}")
        print(f"  Web anchor links: {status['web_anchor_links']}")
        print(f"  Benchmark scores: {status['benchmark_scores']} "
              f"({status['benchmark_models']} models)")
        if status["recent_runs"]:
            print("  Recent runs:")
            for r in status["recent_runs"]:
                print(f"    {r['run_id']}... status={r['status']} started={r['started']}")
        return True

    return False


def _cmd_seed(args: argparse.Namespace) -> None:
    """Handle --seed command."""
    from .ingest_seed import seed

    network_conn = db.get_connection()
    db.init_db(network_conn)
    pass_names = args.seed if args.seed else None
    results = seed(network_conn, passes=pass_names)
    stats = db.network_stats(network_conn)
    network_conn.close()
    print("\nSeed complete:")
    for name, count in results.items():
        print(f"  {name}: {count} models indexed")
    print(
        f"\nNetwork total: {stats['total_models']} models, "
        f"{stats['total_anchors']} anchors"
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="ModelAtlas background ingestion daemon",
    )
    parser.add_argument(
        "--phase",
        default="ab",
        help="Phases to run: a, b, c, ab, abc (default: ab)",
    )
    parser.add_argument(
        "--min-likes",
        type=int,
        default=INGEST_MIN_LIKES,
        help=f"Minimum likes for Phase A (default: {INGEST_MIN_LIKES})",
    )
    parser.add_argument(
        "--vibe-min-likes",
        type=int,
        default=INGEST_VIBE_MIN_LIKES,
        help=f"Minimum likes for Phase C vibes (default: {INGEST_VIBE_MIN_LIKES})",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run in daemon mode (loop with 24h sleep)",
    )
    parser.add_argument(
        "--daemon-sleep",
        type=int,
        default=86400,
        help="Sleep seconds between daemon cycles (default: 86400)",
    )
    parser.add_argument(
        "--seed",
        nargs="*",
        metavar="PASS",
        help="Seed the network via multi-pass HF streaming",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show ingest progress and exit",
    )

    # Phase C
    parser.add_argument("--export-c1", action="store_true", help="Export C1 model_ids")
    parser.add_argument(
        "--merge-c1", nargs="+", metavar="FILE", help="Merge C1 results"
    )
    parser.add_argument(
        "--export-c2", type=int, metavar="NUM_SHARDS", help="Export C2 prompts"
    )
    parser.add_argument(
        "--merge-c2", nargs="+", metavar="FILE", help="Merge C2 results"
    )
    parser.add_argument(
        "--export-c3", type=int, metavar="NUM_SHARDS", help="Export C3 prompts"
    )
    parser.add_argument(
        "--merge-c3", nargs="+", metavar="FILE", help="Merge C3 results"
    )
    parser.add_argument(
        "--select-summaries", action="store_true", help="Pick best summary per model"
    )
    parser.add_argument(
        "--validate-ground-truth", action="store_true", help="Run C4 validation"
    )

    # Phase D
    parser.add_argument("--audit-c2", action="store_true", help="D1 audit C2 anchors")
    parser.add_argument(
        "--expand-dictionary", metavar="SPEC_FILE", help="D2 expand dictionary"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview expansion")
    parser.add_argument(
        "--export-d3", type=int, metavar="NUM_SHARDS", help="D3 export healing"
    )
    parser.add_argument(
        "--heal-tier", choices=["local", "claude"], default="local", help="D3 tier"
    )
    parser.add_argument("--heal-budget", type=int, default=100, help="D3 budget")
    parser.add_argument("--heal-seed", type=int, default=42, help="D3 seed")
    parser.add_argument(
        "--merge-d3", nargs="+", metavar="FILE", help="D3 merge results"
    )
    parser.add_argument("--run-id", help="D3 run_id for merge")
    parser.add_argument(
        "--export-training-data", metavar="OUTPUT_PATH", help="D4 export"
    )
    parser.add_argument(
        "--training-tier",
        choices=["local", "claude", "all"],
        default="all",
        help="D4 tier",
    )
    parser.add_argument("--phase-d-status", action="store_true", help="Phase D status")

    # Phase E (web enrichment)
    parser.add_argument(
        "--export-e", type=int, metavar="NUM_SHARDS", help="Export Phase E web enrichment input"
    )
    parser.add_argument(
        "--export-e-banks", default=None,
        help="Comma-separated banks for Phase E export (default: all)",
    )
    parser.add_argument(
        "--export-e-min-downloads", type=int, default=100,
        help="Phase E priority download threshold (default: 100)",
    )
    parser.add_argument(
        "--export-e-full-corpus", action="store_true",
        help="Include models below download threshold",
    )
    parser.add_argument(
        "--merge-e", nargs="+", metavar="FILE", help="Merge Phase E results"
    )
    parser.add_argument(
        "--merge-e-dry-run", action="store_true", help="Phase E merge dry run"
    )
    parser.add_argument("--phase-e-status", action="store_true", help="Phase E status")

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return parser


def main() -> None:
    """CLI entry point — dispatches to subcommand handlers."""
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.status:
        _cmd_status(args)
        return

    if _cmd_phase_c(args):
        return

    if _cmd_phase_d(args):
        return

    if _cmd_phase_e(args):
        return

    if args.seed is not None:
        _cmd_seed(args)
        return

    run(
        phases=args.phase,
        min_likes=args.min_likes,
        vibe_min_likes=args.vibe_min_likes,
        daemon=args.daemon,
        daemon_sleep=args.daemon_sleep,
    )


if __name__ == "__main__":
    main()
