"""CLI for the corpus-quality eval. See `harness` for what it measures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .. import db
from .cases import CASES
from .harness import diff_reports, format_report, run_eval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m model_atlas.evaluation",
        description="Score the corpus against canonical questions.",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--verbose", action="store_true", help="show passing results too")
    parser.add_argument("--case", action="append", help="run only these cases (repeatable)")
    parser.add_argument("--save-baseline", metavar="PATH", help="write results as a baseline")
    parser.add_argument("--baseline", metavar="PATH", help="diff against a saved baseline")
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        metavar="F",
        help="exit non-zero if the overall score falls below F (e.g. 0.8)",
    )
    args = parser.parse_args(argv)

    cases = CASES
    if args.case:
        wanted = set(args.case)
        cases = tuple(c for c in CASES if c.name in wanted)
        missing = wanted - {c.name for c in cases}
        if missing:
            print(f"unknown case(s): {', '.join(sorted(missing))}", file=sys.stderr)
            return 2

    conn = db.get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
        if total == 0:
            print(
                "Corpus is empty — the eval needs the real network.db.\n"
                "Download it from a release that carries the asset:\n"
                "  https://github.com/rohanvinaik/ModelAtlas/releases",
                file=sys.stderr,
            )
            return 2
        report = run_eval(conn, cases)
    finally:
        conn.close()

    payload = report.to_dict()

    if args.save_baseline:
        Path(args.save_baseline).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"baseline written to {args.save_baseline}", file=sys.stderr)

    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        for line in diff_reports(baseline, payload):
            print(line)
        print()

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_report(report, verbose=args.verbose))

    if args.min_score is not None and report.score < args.min_score:
        print(
            f"FAIL: score {report.score:.1%} is below the floor {args.min_score:.1%}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
