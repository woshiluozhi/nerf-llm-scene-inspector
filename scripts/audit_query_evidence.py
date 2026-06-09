#!/usr/bin/env python
"""Audit query reports for visual and localization evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.evaluation.query_evidence_audit import (  # noqa: E402
    write_query_evidence_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory.")
    parser.add_argument(
        "--queries-dir",
        help="Query results directory. Defaults to RUN_DIR/queries.",
    )
    parser.add_argument(
        "--output",
        help="JSON output path. Defaults to RUN_DIR/query_evidence_audit.json.",
    )
    parser.add_argument(
        "--markdown-output",
        help="Markdown output path. Defaults to RUN_DIR/query_evidence_audit.md.",
    )
    parser.add_argument("--json", action="store_true", help="Print only JSON payload.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    audit = write_query_evidence_audit(
        run_dir,
        queries_dir=args.queries_dir,
        output=args.output,
        markdown_output=args.markdown_output,
    )
    payload = audit.to_dict()
    print(json.dumps(payload, indent=2))
    if not args.json:
        print(f"\nWrote {Path(args.output) if args.output else run_dir / 'query_evidence_audit.json'}")
        print(f"Wrote {Path(args.markdown_output) if args.markdown_output else run_dir / 'query_evidence_audit.md'}")
    return 0 if audit.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
