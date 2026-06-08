#!/usr/bin/env python
"""Create a CV/professor-outreach submission packet from run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.evaluation.submission_packet import write_submission_packet  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory.")
    parser.add_argument(
        "--output",
        help="Output directory. Defaults to RUN_DIR/submission_packet.",
    )
    parser.add_argument("--pack", help="Optional validated portfolio pack directory.")
    parser.add_argument(
        "--pack-validation",
        help="Optional existing portfolio_pack_validation.json. Overrides live pack validation.",
    )
    parser.add_argument("--repo-url", default="", help="Repository URL to include in the packet.")
    parser.add_argument("--ci-url", default="", help="Successful CI run URL to include in the packet.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        packet = write_submission_packet(
            args.run_dir,
            output_dir=args.output,
            pack_dir=args.pack,
            pack_validation_path=args.pack_validation,
            repo_url=args.repo_url,
            ci_url=args.ci_url,
        )
    except Exception as exc:
        print(f"create_submission_packet failed: {exc}", file=sys.stderr)
        return 1

    output = Path(args.output) if args.output else Path(args.run_dir) / "submission_packet"
    print(json.dumps(packet.to_dict(), indent=2))
    print(f"\nWrote {output / 'submission_packet.json'}")
    print(f"Wrote {output / 'submission_checklist.md'}")
    print(f"Wrote {output / 'cv_project_entry.md'}")
    print(f"Wrote {output / 'professor_email_brief.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
