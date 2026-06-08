#!/usr/bin/env python
"""Audit external-facing project/run text for overclaiming risk."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.evaluation.claim_audit import write_claim_audit  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT), help="Project root to scan.")
    parser.add_argument("--run-dir", help="Optional pipeline run directory to include.")
    parser.add_argument("--pack", help="Optional exported portfolio pack directory to include.")
    parser.add_argument("--output", help="JSON output path. Defaults to RUN_DIR/claim_audit.json if --run-dir is set.")
    parser.add_argument(
        "--markdown-output",
        help="Markdown output path. Defaults to RUN_DIR/claim_audit.md if --run-dir is set.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on warnings as well as failures.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = write_claim_audit(
            root=args.root,
            run_dir=args.run_dir,
            pack_dir=args.pack,
            output=args.output,
            markdown_output=args.markdown_output,
        )
    except Exception as exc:
        print(f"audit_claims failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report.to_dict(), indent=2))
    if args.output:
        print(f"\nWrote {args.output}")
    elif args.run_dir:
        print(f"\nWrote {Path(args.run_dir) / 'claim_audit.json'}")
    if args.markdown_output:
        print(f"Wrote {args.markdown_output}")
    elif args.run_dir:
        print(f"Wrote {Path(args.run_dir) / 'claim_audit.md'}")
    if report.status == "fail":
        return 1
    if args.strict and report.status == "warn":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
