#!/usr/bin/env python
"""Apply a pass/warn/fail quality gate to a pipeline run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.evaluation.quality_gate import check_run_quality  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory to gate.")
    parser.add_argument(
        "--profile",
        choices=["smoke", "real-run", "portfolio"],
        default="smoke",
        help="Gate strictness profile.",
    )
    parser.add_argument("--pack", help="Optional exported portfolio pack to validate.")
    parser.add_argument("--output", help="JSON output path. Defaults to RUN_DIR/quality_gate.json.")
    parser.add_argument(
        "--markdown-output",
        help="Markdown output path. Defaults to RUN_DIR/quality_gate.md.",
    )
    parser.add_argument("--min-query-reports", type=int, help="Override profile query-report threshold.")
    parser.add_argument(
        "--min-evaluated-queries",
        type=int,
        help="Override profile evaluated-query threshold.",
    )
    parser.add_argument("--min-evidence-ratio", type=float, help="Override profile evidence ratio.")
    parser.add_argument(
        "--allow-dry-run",
        action="store_true",
        help="Allow dry-run outputs even when the selected profile normally requires a real run.",
    )
    parser.add_argument(
        "--disallow-dry-run",
        action="store_true",
        help="Disallow dry-run outputs even when the selected profile normally permits them.",
    )
    parser.add_argument(
        "--require-pack",
        action="store_true",
        help="Require a validated portfolio pack for this gate.",
    )
    parser.add_argument(
        "--no-require-pack",
        action="store_true",
        help="Do not require a portfolio pack even when the selected profile normally does.",
    )
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Exit non-zero for warning-level criteria as well as failures.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.allow_dry_run and args.disallow_dry_run:
        print("Choose only one of --allow-dry-run or --disallow-dry-run.", file=sys.stderr)
        return 2
    if args.require_pack and args.no_require_pack:
        print("Choose only one of --require-pack or --no-require-pack.", file=sys.stderr)
        return 2

    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else run_dir / "quality_gate.json"
    markdown_output = Path(args.markdown_output) if args.markdown_output else run_dir / "quality_gate.md"
    allow_dry_run = _tri_state(args.allow_dry_run, args.disallow_dry_run)
    require_pack = _tri_state(args.require_pack, args.no_require_pack)

    try:
        report = check_run_quality(
            run_dir,
            profile=args.profile,
            pack_dir=args.pack,
            min_query_reports=args.min_query_reports,
            min_evaluated_queries=args.min_evaluated_queries,
            min_evidence_ratio=args.min_evidence_ratio,
            allow_dry_run=allow_dry_run,
            require_pack=require_pack,
        )
        report.to_json(output)
        report.to_markdown(markdown_output)
    except Exception as exc:
        print(f"check_run_quality failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report.to_dict(), indent=2))
    print(f"\nWrote {output}")
    print(f"Wrote {markdown_output}")
    if not report.passed:
        return 1
    if args.strict_warnings and report.status != "pass":
        return 1
    return 0


def _tri_state(enabled: bool, disabled: bool) -> bool | None:
    if enabled:
        return True
    if disabled:
        return False
    return None


if __name__ == "__main__":
    raise SystemExit(main())
